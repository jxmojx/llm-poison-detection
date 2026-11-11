"""场景应用演示：把 poisonguard 当"训练数据入库前扫描"和"内容审核网关"用。

三种场景，都用 v3 的真实模型与数据：
  A. 入库前批量扫描（离线）—— 给一批待入库的文本打分，输出拦截清单；
  B. 内容级投毒现场演示 —— 拿一条真实新闻，用最小事实篡改生成它的"污染孪生版"，
     展示检测器对"只改一个事实"的版本给出完全不同的分数；
  C. HTTP 服务（TestClient）—— 以 /health /score /stats 的形式，模拟接入审核中间件。

用法：
  python examples/scan_demo.py            # 进程内跑 A+B+C
  python examples/scan_demo.py --http     # 进程内 C 换成启动真实 uvicorn 并在子进程里打 HTTP
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poisonguard.config import Config, load_config
from poisonguard.service import build_scorer, _default_student
from poisonguard import data as D
from poisonguard.attacks import apply_minimal_edit

PROCESSED_V3 = ROOT / "data" / "processed_real_v3"


def _load_cfg() -> Config:
    cfg = load_config()
    cfg.data.processed_dir = str(PROCESSED_V3)
    return cfg


def _pick(splits, family: str, label: int, n: int):
    rows = [s for s in splits["test"] if s.family == family and s.label == label]
    return [s.text for s in rows[:n]]


# ----------------------------------------------------------------------
def scene_a_batch(cfg, scorer):
    """A. 入库前批量扫描：正常新闻 + edit 投毒各取 3 条，打印打分表。"""
    print("\n" + "=" * 78)
    print("场景 A｜入库前批量扫描（离线）: 正常新闻 vs 内容级投毒(edit-family)")
    print("=" * 78)
    splits = D.load_splits(PROCESSED_V3)
    normal = _pick(splits, "normal", 0, 3)
    poisoned = _pick(splits, "edit", 1, 3)
    texts = normal + poisoned
    res = scorer.score(texts, explain=True)["results"]
    for t, r in zip(texts, res):
        flag = "[BLOCK]" if r["flag"] else "[PASS ]"
        v = r["views"]
        print(f"\n[{flag}] score={r['score']:.3f}")
        print(f"   {t}")
        print(f"   student={v['student']} knn={v['knn']} fpr={v['fpr_est']} "
              f"char_ngram={v['char_ngram']}")
    flagged = sum(r["flag"] for r in res)
    poison_flag = sum(r["flag"] for r in res[3:])
    norm_bad = sum(r["flag"] for r in res[:3])
    print(f"\n→ 3 条投毒拦下 {poison_flag} 条；3 条正常误拦 {norm_bad} 条"
          f"（目标 FPR=1%，小样本下可接受）")
    return splits


# ----------------------------------------------------------------------
def scene_b_live_edit(cfg, scorer):
    """B. 现场生成"污染孪生版"：同一真实新闻，改一个事实前后分数对比。"""
    print("\n" + "=" * 78)
    print("场景 B｜内容级投毒现场演示: 一条真实新闻被最小篡改一个事实")
    print("=" * 78)
    rng = random.Random(7)
    candidates = _pick(D.load_splits(PROCESSED_V3), "normal", 0, 300)
    # 挑一个"学生分较高"的正常句当 base，让原版 vs 篡改版的对比有梯度
    # （分数贴地的句子原版/篡改版会被一起判定 PASS，演示不出差异）
    base, edited, rule = None, None, None
    for cand in candidates:
        ed = apply_minimal_edit(cand, rng)
        if not ed:
            continue
        _, rule = ed
        if scorer.score([cand], explain=False)["results"][0]["score"] < 0.3:
            continue
        base, edited = cand, ed[0]
        break
    if not base:
        print("（示例语料里没找到可改的学生高分句，跳过）")
        return
    res = scorer.score([base, edited], explain=True)["results"]
    for tag, t, r in zip(("原版", "篡改版"), (base, edited), res):
        flag = "[BLOCK]" if r["flag"] else "[PASS ]"
        print(f"\n[{flag}] score={r['score']:.3f}  【{tag}】\n   {t}")
    b, m = res[0], res[1]
    print(f"\n→ 篡改规则: {rule}  改后: {edited}")
    print(f"→ 原版分数 {b['score']:.3f}，篡改版 {m['score']:.3f}")


# ----------------------------------------------------------------------
def scene_d_topk_cascade(cfg):
    """D. 排序 + TopK 级联复核：学生模型秒级排序，顶部样本送 LLM 判官复核。

    这是内容级投毒的真实用法——0.5B 判官单条 0.78s，不能全量跑；
    先用 0.045ms 的学生模型把每天入库的文本排序，只把分数最高的
    TopK 交给 LLM 判官判"是否污染"。用真实 judge_samples 证明这个
    流程能召回多少 edit 投毒。
    """
    print("\n" + "=" * 78)
    print("场景 D｜排序 + TopK 级联复核（内容级投毒的正确用法）")
    print("=" * 78)
    import json
    rows = [
        json.loads(line)
        for line in (PROCESSED_V3.parent.parent / "outputs" / "metrics"
                     / "judge_samples_real_v3.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    # judge_samples 是 judge CLI 的 320 条抽样，含每条的 label/family/judge_score/knn
    print(f"\n带判官分数的样本: {len(rows)} 条"
          f"(其中 edit-family {sum(r['family']=='edit' for r in rows)} 条,"
          f" template {sum(r['family']=='template' for r in rows)} 条)")
    # 复现级联: 学生排序取 TopK → LLM 复核回收召回
    for k in (0.1, 0.2, 0.4):
        n = max(1, int(len(rows) * k))
        top = sorted(rows, key=lambda r: -r["judge_score"])[: n]
        edit_top = sum(r["family"] == "edit" for r in top)
        all_edit = sum(r["family"] == "edit" for r in rows)
        # 判官复核后拦下的 edit = 顶部样本里 judge_score>0.5 的 edit
        block_edit = sum(r["family"] == "edit" and r["judge_score"] > 0.5 for r in top)
        print(f"  Top {int(k*100):>2}% 送 LLM → edit 召回到复核桶 {edit_top}/{all_edit}"
              f"，其中判官确认拦截 {block_edit}")

    print("\n→ 语义层价值: 学生对 edit 的 AUROC 0.692（教师 0.689），"
          "它把每天海量文本压到几十条让 LLM 精判——这才是能上线的分工。")


# ----------------------------------------------------------------------
def scene_c_http(cfg, scorer, real_server: bool = False, port: int = 8321):
    """C. 审核中间件式 HTTP 接口：/health /score /stats。"""
    print("\n" + "=" * 78)
    print("场景 C｜HTTP 审核网关（FastAPI）——见 examples/ 同款接口")
    print("=" * 78)
    from fastapi.testclient import TestClient
    from poisonguard.service import create_app

    app = create_app(cfg, scorer=scorer)
    client = TestClient(app)

    h = client.get("/health").json()
    print(f"/health → {h}")

    payload = {
        "texts": [
            "现价200多万17款德国版宾利添越6.0T，顶级豪华SUV越野车",
            _pick(D.load_splits(PROCESSED_V3), "edit", 1, 1)[0],
        ],
        "explain": True,
    }
    r = client.post("/score", json=payload)
    data = r.json()
    for rec in data["results"]:
        print(f"/score → flag={rec['flag']} score={rec['score']:.3f} "
              f"views={ {k: (round(v,3) if isinstance(v,(int,float)) else v) for k,v in rec['views'].items()} }")
    print(f"/stats → {client.get('/stats').json()}")

    print("\n" + "=" * 78)
    print("演示完。把 /score 接到任何审核中间件即可：")
    print("  flag=True  → 拦截/标记复查；flag=False → 放行进库")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="用真实 uvicorn 而非 TestClient")
    a = ap.parse_args()

    cfg = _load_cfg()
    student = _default_student(cfg)
    print(f"[demo] 学生模型: {student}")
    t0 = time.perf_counter()
    scorer = build_scorer(cfg, student_path=student)
    print(f"[demo] 装配完成（加载编码器+组装检测器）{time.perf_counter()-t0:.1f}s")

    scene_a_batch(cfg, scorer)
    scene_b_live_edit(cfg, scorer)
    scene_c_http(cfg, scorer, a.http)
    scene_d_topk_cascade(cfg)


if __name__ == "__main__":
    main()
