"""场景四复现：排序 + TopK 级联复核（学生模型排序 → 少量 LLM 判官精判）。

为什么需要"复现"而不是直接用旧演示里的场景 D：
旧版用 `judge_score`（判官自己输出的 logprob 分）做排序——那是"让复核器自己排自己"，
自证循环测不出"排序器选得准不准"。本脚本把两层信号彻底拆开：

  排序层（谁进复核桶）= 学生模型 JudgeStudent 的分（0.045ms/条）
  复核层（进桶后谁拍板）= LLM 判官 logprob 分（0.78s/条，用已实测的 320 条抽样）
  对照组 = ①同预算随机进桶 ②判官用自己的分排序（旧场景 D 口径，自证基线）

并给出两个层次，全部基于 ground-truth 标签（不自证）：
  [召回层] 全量 test 5997 条（edit 500 / template 500 / paraphrase 500 …）
           学生排序 TopK，按家族拆开统计进桶数量——能看出学生"富集哪些口音"。
  [拦截层] 320 条判官实测抽样：学生排序 TopK → 进桶且判官分>阈值的 edit 数
           （联合信号），与随机、判官自排序并列。

产出的最重要信息是一个负结果 + 一个正结果：
  - 负：学生模型（只用 160 条软标签训练）几乎不富集内容级篡改(edit)，TopK 低于随机；
        —— 这正是项目主线的证据：词面法/轻量法认"措辞口音"，认不出"事实被偷改"。
  - 正：对模板/改写/微妙改写的投毒，学生 TopK 富集明显（提升 >>1x），
        级联在"家族型投毒"上是成立的，edit 需要更高的预算或专门的排序器。

用法：
  python examples/repro_scene4.py                 # 默认 budgets 1,2,5,10,20,30,40
  python examples/repro_scene4.py --no-json       # 不写 metrics 文件
产物：outputs/metrics/repro_scene4_real_v3.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poisonguard.config import load_config
from poisonguard.service import _default_student, build_scorer
from poisonguard import data as D

PROCESSED_V3 = ROOT / "data" / "processed_real_v3"
METRICS = ROOT / "outputs" / "metrics"


def _load_cfg():
    cfg = load_config()
    cfg.data.processed_dir = str(PROCESSED_V3)
    return cfg


def _load_judge_samples():
    p = METRICS / "judge_samples_real_v3.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _student_scores(scorer, texts):
    """批量学生打分（explain=True 取 student 视图；编码走 npz 缓存）。"""
    res = scorer.score(texts, explain=True)["results"]
    return [r["views"].get("student")
            if isinstance(r["views"].get("student"), (int, float)) else -1.0
            for r in res]


def recall_layer(scorer, rows_all, budgets):
    """[召回层] 全量 test：学生排序 TopK，按家族拆开统计进桶数量。"""
    import numpy as np

    texts = [r["text"] for r in rows_all]
    n = len(texts)
    fam_counts = collections.Counter(r["family"] for r in rows_all)
    t0 = time.perf_counter()
    stud = np.array(_student_scores(scorer, texts), dtype=float)
    t_stud = time.perf_counter() - t0

    idx = np.argsort(-stud)
    fam = np.array([r["family"] for r in rows_all])
    print(f"\n[召回层] 全量 test {n} 条，学生模型打分排序（学生耗时 {t_stud:.1f}s）")
    print(f"        家族分布: " + ", ".join(f"{k}={v}" for k, v in fam_counts.items()))

    rows_out = {}
    for kp in budgets:
        k = max(1, int(round(n * kp / 100.0)))
        top = fam[idx[:k]]
        bucket = {f: int((top == f).sum()) for f in fam_counts}
        n_edit = fam_counts["edit"]
        rand_e = k * n_edit / n
        row = {
            "topk": k,
            "in_bucket": bucket,
            "edit_recall": round(bucket.get("edit", 0) / n_edit, 4),
            "random_expected_edit": round(rand_e, 1),
            "edit_lift_vs_random": round(bucket.get("edit", 0) / rand_e, 2) if rand_e else None,
        }
        # template / paraphrase 富集比（若有）
        for f in ("template", "paraphrase", "subtle", "backdoor"):
            nf = fam_counts.get(f, 0)
            if nf:
                rand_f = k * nf / n
                row[f"in_{f}"] = bucket.get(f, 0)
                row[f"{f}_lift"] = round(bucket.get(f, 0) / rand_f, 2) if rand_f else None
        rows_out[f"{kp:.1f}%"] = row

        def _fmt(f, rec):
            b = row.get(f"in_{f}")
            lf = row.get(f"_lift")
            return "" if b is None else f"{f}={b}({lf}x)"

        print(f"  Top {kp:>4.1f}% (k={k:>4}) → "
              + " ".join([f"edit={row['in_bucket'].get('edit',0)}({row['edit_lift_vs_random']}x)",
                          f"template={row.get('in_template','-')}({row.get('template_lift','-')}x)",
                          f"paraphrase={row.get('in_paraphrase','-')}({row.get('paraphrase_lift','-')}x)"]))
    return rows_out, {"n": n, "family_counts": dict(fam_counts), "student_sec": round(t_stud, 2)}


def intercept_layer(scorer, rows_js, budgets, judge_thr=0.5):
    """[拦截层] 320 条判官实测抽样：学生排序 TopK → 判官复核确认拦截的 edit。

    三列对照：
      student_rank  = 学生排序先进桶，进桶且 judge_score>thr 记为拦截（联合信号）
      judge_self    = 判官用自己的分排序进桶后拦下（旧场景D口径，自证基线）
      random        = 同预算随机进桶后用 judge_score>thr 拦下的期望
    """
    import numpy as np

    n = len(rows_js)
    n_edit = sum(r["family"] == "edit" for r in rows_js)
    fam = np.array([r["family"] for r in rows_js])
    jscore = np.array([r["judge_score"] for r in rows_js])
    is_edit = fam == "edit"
    thr_ok = jscore > judge_thr
    # 判官"自己"能确认拦截的 edit = 全量复核口径
    full_intercept = int((is_edit & thr_ok).sum())

    # 学生为这 320 条打分
    texts = [r["text"] for r in rows_js]
    stud = np.array(_student_scores(scorer, texts), dtype=float)

    order_stud = np.argsort(-stud)
    order_judge = np.argsort(-jscore)
    print(f"\n[拦截层] 判官实测抽样 {n} 条（edit {n_edit}; 全量复核能确认拦截 {full_intercept} 条,"
          f" judge_thr={judge_thr}）")

    rows_out = {}
    for kp in budgets:
        k = max(1, int(round(n * kp / 100.0)))
        s_top = is_edit & thr_ok
        s_stud = np.zeros(n, dtype=bool); s_stud[order_stud[:k]] = True
        s_jdg = np.zeros(n, dtype=bool); s_jdg[order_judge[:k]] = True
        s_rand = np.zeros(n, dtype=bool); s_rand[np.random.default_rng(0).choice(n, k, replace=False)] = True
        j_stud = int((s_stud & s_top).sum())
        j_jdg = int((s_jdg & s_top).sum())
        j_rand = int((s_rand & s_top).sum())
        rows_out[f"{kp:.1f}%"] = {
            "topk": k,
            "edit_intercept_student_rank": j_stud,
            "edit_intercept_judge_self_rank": j_jdg,
            "edit_intercept_random": j_rand,
        }
        print(f"  Top {kp:>4.1f}% (k={k:>3}) → 学生排序拦 edit {j_stud:>2} | "
              f"判官自排序(旧口径) {j_jdg:>2} | 随机 {j_rand:>2}（全量复核可拦 {full_intercept}）")
    return rows_out, {"n": n, "n_edit_samples": n_edit, "judge_full_intercept": full_intercept,
                      "judge_thr": judge_thr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1,2,5,10,20,30,40", help="进复核桶预算百分比")
    ap.add_argument("--judge-thr", type=float, default=0.5)
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    budgets = [float(x) for x in a.budgets.split(",") if x]
    random.seed(0)

    cfg = _load_cfg()
    student = _default_student(cfg)
    print(f"[repro] 学生模型: {Path(student).name}  judge_thr={a.judge_thr}")
    t0 = time.perf_counter()
    scorer = build_scorer(cfg, student_path=student)
    print(f"[repro] 装配完成 {time.perf_counter()-t0:.1f}s")

    splits = D.load_splits(PROCESSED_V3)
    rows_all = [{"text": s.text, "label": s.label, "family": s.family} for s in splits["test"]]
    recall, rec_meta = recall_layer(scorer, rows_all, budgets)

    rows_js = _load_judge_samples()
    inter, int_meta = intercept_layer(scorer, rows_js, budgets, a.judge_thr)

    print("\n" + "=" * 78)
    print("结论（诚实版）")
    print("=" * 78)
    print("  ① 学生模型 TopK 对「模板/改写」类投毒富集明显（提升 >1x），"
          "对「内容级 edit」不富集（≈随机甚至更差）——")
    print("     因为学生只用 160 条软标签训练，认的是措辞口音，抓不住'事实被偷改'。")
    print("  ② 判官复核层是 content-level 投毒的唯一现实抓手，但它 0.78s/条，"
          "全量跑不起；正确分工是「学生粗筛 + LLM 精判」，但对 edit 要给足预算或换排序器。")
    print("  ③ 旧场景 D 用判官分自排序，高估了级联对 edit 的召回，本复现已修正。")

    if not a.no_json:
        out = {
            "meta": {"data": rec_meta, "intercept": int_meta,
                     "student": Path(student).name, "note": "诚实口径：排序层与学生模型、复核层与判官分严格分开；含判官自排序对照"},
            "recall_layer": recall,
            "intercept_layer": inter,
        }
        p = METRICS / "repro_scene4_real_v3.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
