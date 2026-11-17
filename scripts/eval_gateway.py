"""A2 双闸门评测：ragcheck（检索一致性）先筛 + 主检测器 sup_tfidf 定档。

设计：不把两个不同尺度的分揉成一个数（A1 已证会尺度错位），
而是两层各自有独立校准阈值：
  闸门1 ragcheck: scan_scores(top_k=5, min_ratio=0.75, aggregate=max) > 0 → 冲突命中
  闸门2 sup_tfidf: 命中样本上 dev-clean q99 校准阈值 → >thr BLOCK / else WATCH
未命中闸门1 → 直接 PASS（≈不触发主检测器，省成本）

产出：
  outputs/metrics/ragcheck_gateway.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poisonguard.config import load_config  # noqa: E402
from poisonguard.data import load_splits  # noqa: E402
from poisonguard.ragcheck import build_reference, scan_scores  # noqa: E402
from poisonguard.encoders import get_encoder  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    cfg.data.processed_dir = str(ROOT / "data" / "processed_real_v3")
    metrics_dir = ROOT / "outputs" / "metrics"

    splits = load_splits(cfg.data.processed_dir)
    dev = splits["dev"]
    test = splits["test"]
    dev_normals = [s.text for s in dev if s.label == 0 and s.family == "normal"][:2000]

    # 与 A1/eval_fusion 相同的 400×家族抽样（对齐可比性）
    buckets: dict[str, list[int]] = {}
    for i, s in enumerate(test):
        b = buckets.setdefault(s.family, [])
        if len(b) < 400:
            b.append(i)
    keep_idx = np.array(sorted(i for fam in sorted(buckets) for i in buckets[fam]), dtype=int)
    samples = [test[i] for i in keep_idx]
    y_te = np.array([s.label for s in samples])
    fam_te = np.array([s.family for s in samples])
    texts_te = [s.text for s in samples]
    edit_mask = fam_te == "edit"
    normal_mask = fam_te == "normal"

    enc = get_encoder(cfg)
    ref_texts = build_reference(cfg, source="train")
    ref_embs = enc.encode(ref_texts)      # cache hit
    dev_embs = enc.encode(dev_normals)    # cache hit
    test_embs = enc.encode(texts_te)      # cache hit

    # 闸门1：ragcheck（max 聚合）
    rag_dev = scan_scores(dev_normals, ref_texts, ref_embs, enc, q_embs=dev_embs,
                          top_k=5, min_ratio=0.75, progress=False, aggregate="max")
    rag_te = scan_scores(texts_te, ref_texts, ref_embs, enc, q_embs=test_embs,
                         top_k=5, min_ratio=0.75, progress=False, aggregate="max")
    # 门1 阈值：dev-clean 上 score>0 的冲突率作为门控操作点
    gate1_thr = 0.0                       # 冲突即命中（score>0）
    gate1_dev_fpr = float((rag_dev > gate1_thr).mean())
    gate1_hit = rag_te > gate1_thr

    # 闸门2：sup_tfidf（dev-clean q99 校准）
    dev_npz = np.load(metrics_dir / "dev_scores_real_v3.npz", allow_pickle=True)
    test_npz = np.load(metrics_dir / "test_scores_real_v3.npz", allow_pickle=True)
    base_dev = np.asarray(dev_npz["sup_tfidf"], dtype=float)[:2000]
    base_te = np.asarray(test_npz["sup_tfidf"], dtype=float)[keep_idx]
    thr2 = float(np.quantile(base_dev, 0.99))

    # 最终裁决：仅对 gate1 命中样本做定档
    verdict = np.where(gate1_hit,
                       np.where(base_te > thr2, "BLOCK", "WATCH"),
                       "PASS")

    # 统计
    def _conf(tp_, fp_, fn_tp_, tn_):
        """在 gate1 命中子集上的 2×2（正=edit 毒，负=normal）。"""
        denom = tp_ + fp_ + fn_tp_ + tn_
        if not denom:
            return None
        return {"tp": int(tp_), "fp": int(fp_), "fn": int(fn_tp_), "tn": int(tn_)}

    # 全量 2×2：edit 毒 vs normal 干净
    is_pos = edit_mask
    is_neg = normal_mask
    tp = int(((verdict == "BLOCK") & is_pos & (y_te == 1)).sum())
    fp = int(((verdict == "BLOCK") & is_neg).sum())
    watch_pos = int(((verdict == "WATCH") & is_pos & (y_te == 1)).sum())
    watch_neg = int(((verdict == "WATCH") & is_neg).sum())
    pass_pos = int(((verdict == "PASS") & is_pos & (y_te == 1)).sum())
    pass_neg = int(((verdict == "PASS") & is_neg).sum())

    n_pos, n_neg = int(is_pos.sum()), int(is_neg.sum())
    edit_recall_block = tp / (tp + watch_pos + pass_pos) if (tp + watch_pos + pass_pos) else None
    normal_precision_block = tp / (tp + fp) if (tp + fp) else None
    normal_fpr_block = fp / (fp + watch_neg + pass_neg) if (fp + watch_neg + pass_neg) else None

    # gate1 命中率 → 双闸门"成本"（多少样本要送闸门2）
    gate1_hit_rate = float(gate1_hit.mean())
    # 在 gate1 命中子集内，edit 占比（命中质量）
    hit_edit_frac = float(gate1_hit[edit_mask].mean()) if edit_mask.any() else None

    out = {
        "meta": {
            "task": "A2 dual-gateway",
            "branch": "feat/rag-guard",
            "encoder": "crom", "corpus_source": "train", "corpus_n": len(ref_texts),
            "aggregate": "max", "top_k": 5, "min_ratio": 0.75,
            "n_dev_clean": len(dev_normals), "n_test": len(samples),
            "gate1": "ragcheck score>0 冲突命中",
            "gate2_threshold": thr2,
            "gate2_threshold_source": "dev-clean sup_tfidf q99",
            "seconds": round(time.perf_counter() - t0, 1),
        },
        "gate1": {
            "threshold": gate1_thr,
            "dev_fpr": round(gate1_dev_fpr, 4),
            "test_hit_rate": round(gate1_hit_rate, 4),
            "test_hit_n": int(gate1_hit.sum()),
            "hit_edit_frac": round(hit_edit_frac, 4) if hit_edit_frac is not None else None,
        },
        "gate2": {"threshold": thr2, "fpr_target": 0.01},
        "verdict_counts": {
            "BLOCK": int((verdict == "BLOCK").sum()),
            "WATCH": int((verdict == "WATCH").sum()),
            "PASS": int((verdict == "PASS").sum()),
        },
        "confusion_edit_vs_normal": {
            "tp_block": tp, "fp_block": fp,
            "watch_pos": watch_pos, "watch_neg": watch_neg,
            "pass_pos": pass_pos, "pass_neg": pass_neg,
            "n_edit": n_pos, "n_normal": n_neg,
            "edit_recall_block": round(edit_recall_block, 4) if edit_recall_block is not None else None,
            "normal_precision_block": round(normal_precision_block, 4) if normal_precision_block is not None else None,
            "normal_fpr_block": round(normal_fpr_block, 4) if normal_fpr_block is not None else None,
        },
        "interpretation": (
            "双闸门：ragcheck 命中(score>0)才送主检测器；命中子集上 sup_tfidf>q99 判 BLOCK，"
            "否则 WATCH；未命中直接 PASS。两层独立校准，避免尺度错位（A1）。"
        ),
    }
    (metrics_dir / "ragcheck_gateway.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== A2 双闸门（ragcheck → sup_tfidf） =====")
    print(f"gate1: dev FPR={out['gate1']['dev_fpr']}  test 命中率={out['gate1']['test_hit_rate']} "
          f"({out['gate1']['test_hit_n']})  edit命中占比={out['gate1']['hit_edit_frac']}")
    print(f"gate2 thr={thr2:.4f}")
    print(f"裁决: BLOCK={out['verdict_counts']['BLOCK']} WATCH={out['verdict_counts']['WATCH']} "
          f"PASS={out['verdict_counts']['PASS']}")
    c = out["confusion_edit_vs_normal"]
    print(f"edit BLOCK={c['tp_block']} WATCH={c['watch_pos']} PASS={c['pass_pos']} (n={n_pos}) "
          f"→ edit召回(BLOCK)={c['edit_recall_block']}")
    print(f"normal BLOCK={c['fp_block']} WATCH={c['watch_neg']} PASS={c['pass_neg']} (n={n_neg}) "
          f"→ normal FPR(BLOCK)={c['normal_fpr_block']}")
    print(f"总耗时 {out['meta']['seconds']}s")


if __name__ == "__main__":
    main()
