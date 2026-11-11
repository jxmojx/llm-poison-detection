# -*- coding: utf-8 -*-
"""换领域落地演示（iflytek：软件 App 描述域）.

先决条件（在仓库根目录依次执行）:
  python -m poisonguard.cli download --corpus iflytek
  python -m poisonguard.cli data  --clean-source tnews --real-file real_iflytek.txt --edit --hardneg 400
  python -m poisonguard.cli detect  --clean-source tnews --real-file real_iflytek.txt --edit --hardneg 400
  python -m poisonguard.cli judge  --clean-source tnews --real-file real_iflytek.txt --edit --hardneg 400 --limit 40 --dev-limit 20
  python -m poisonguard.cli distill --clean-source tnews --real-file real_iflytek.txt --edit --hardneg 400

本脚本演示：
  1) 学生模型(0.152ms/条)对 test 全量(4861 条,含 500 edit)排序；
  2) 分家族看 Top-k 富集度（诚实口径：edit 不富集，模板/改写富集 ~2-3×）；
  3) 判官全量复核可确认的 edit 数量（语义层才是 edit 的抓手）。
输出落盘 outputs/metrics/repro_iflytek_scan.json。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from poisonguard.config import load_config
from poisonguard.service import build_scorer

OUT = Path(__file__).resolve().parent.parent / "outputs" / "metrics" / "repro_iflytek_scan.json"


def main() -> None:
    cfg = load_config("configs/default.yaml")
    cfg.data.processed_dir = "data/processed_iflytek"

    stud = None
    for p in sorted(Path("outputs/metrics").glob("student_judge_iflytek.pkl")):
        stud = str(p)
        break
    if not stud:
        print("[err] 未找到 outputs/metrics/student_judge_iflytek.pkl，先跑 distill 步骤")
        sys.exit(1)

    scorer = build_scorer(cfg, student_path=stud)

    with open("data/processed_iflytek/test.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    texts = [r["text"] for r in rows]
    print(f"[scan] 测试集 {len(rows)} 条 (含 edit 500 / hardneg 200)")

    t0 = time.time()
    scores: list[float] = []
    for i in range(0, len(texts), 512):
        r = scorer.score(texts[i:i + 512], explain=False)
        scores.extend(x["score"] for x in r["results"])
    dt = time.time() - t0
    print(f"[scan] 打分 {len(scores)} 条 耗时 {dt:.2f}s => {dt / len(scores) * 1000:.3f} ms/条")

    s = np.array(scores, dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)
    fam = np.array([r["family"] for r in rows])
    ord_idx = np.argsort(-s)

    summary: dict = {"n": len(scores), "ms_per_item": round(dt / len(scores) * 1000, 3),
                     "topk": {}, "judge_edit_confirm": None}
    print("\n[recall] 学生 Top-k（按分降序，检出正类/该族总数）")
    for k, label in [(243, "5%"), (486, "10%"), (972, "20%")]:
        top = set(ord_idx[:k].tolist())
        row = {}
        for fname in ["edit", "template", "paraphrase", "subtle", "backdoor", "hardneg"]:
            idx = np.where(fam == fname)[0]
            hit = int(sum(1 for i in idx if i in top))
            row[fname] = [hit, int(len(idx))]
            print(f"  Top{k}({label}) {fname}: {hit}/{len(idx)}")
        summary["topk"][label] = row

    # 判官全量复核可确认的 edit
    try:
        with open("outputs/metrics/judge_samples_iflytek.jsonl", encoding="utf-8") as f:
            jr = [json.loads(l) for l in f]
        got = {r["text"]: r.get("judge_score") or 0.0
               for r in jr if r.get("family") == "edit" and r.get("label") == 1}
        conf = int(sum(1 for v in got.values() if v > 0.5))
        summary["judge_edit_confirm"] = [conf, len(got)]
        print(f"\n[judge] 判官全量复核可确认 edit: {conf}/{len(got)}")
    except FileNotFoundError:
        print("\n[judge] judge_samples_iflytek.jsonl 缺失，跳过")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
