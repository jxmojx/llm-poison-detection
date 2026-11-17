"""A1 融合尺度校准：dev-fit → test-apply 的融合评测。

问题：eval_fusion.py 用 _normalize 对 dev/test 各自做 z-score(自身 μ/σ)，
     dev 池与 test 池(类分布不同)的融合分尺度不齐 → 拿 dev q99 卡 test 全崩(TPR@1%≈0)。
修复：在 dev clean pool 上拟合每个组件的 μ/σ，同一套统计量套到 test，
     q99 阈值也从 dev 融合分取 → 校准工作点成立。

产出：
  outputs/metrics/ragcheck_fusion_calibrated.json
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
from poisonguard.evaluate import auroc  # noqa: E402
from poisonguard.ragcheck import build_reference, scan_scores  # noqa: E402
from poisonguard.encoders import get_encoder  # noqa: E402


def _auc(y, s):
    if len(set(y)) < 2 or s is None or len(s) != len(y):
        return None
    try:
        return round(float(auroc(y, s)), 4)
    except Exception:
        return None


def _dev_fit_fused(base_dev, rag_dev, base_te, rag_te, w):
    """dev 上拟合 μ/σ，套到 test；返回 dev/test 融合分（同一全局尺度）。"""
    mu_b, sd_b = float(base_dev.mean()), float(base_dev.std() or 1.0)
    mu_r, sd_r = float(rag_dev.mean()), float(rag_dev.std() or 1.0)
    zb_dev = (base_dev - mu_b) / sd_b
    zr_dev = (rag_dev - mu_r) / sd_r
    fused_dev = w * zb_dev + (1.0 - w) * zr_dev
    zb_te = (base_te - mu_b) / sd_b
    zr_te = (rag_te - mu_r) / sd_r
    fused_te = w * zb_te + (1.0 - w) * zr_te
    stats = {"mu_base": mu_b, "sd_base": sd_b, "mu_rag": mu_r, "sd_rag": sd_r}
    return fused_dev, fused_te, stats


def _apply_fit(fused_dev, fused_te, y_te, edit_mask, normal_mask):
    thr = float(np.quantile(fused_dev, 0.99))
    tpr1 = float((fused_te[y_te == 1] > thr).mean())
    edit_rec = float((fused_te[edit_mask] > thr).mean())
    norm_fpr = float((fused_te[normal_mask] > thr).mean())
    return {
        "threshold_dev_q99": thr,
        "tpr_at_fpr1_dev": round(tpr1, 4),
        "edit_recall_dev": round(edit_rec, 4),
        "normal_fpr_dev": round(norm_fpr, 4),
    }


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    cfg.data.processed_dir = str(ROOT / "data" / "processed_real_v3")
    metrics_dir = ROOT / "outputs" / "metrics"

    splits = load_splits(cfg.data.processed_dir)
    dev = splits["dev"]
    test = splits["test"]
    dev_normals = [s.text for s in dev if s.label == 0 and s.family == "normal"][:2000]

    # 与 eval_fusion 相同的 400×家族抽样（保持一致的可比性）
    buckets: dict[str, list[int]] = {}
    for i, s in enumerate(test):
        b = buckets.setdefault(s.family, [])
        if len(b) < 400:
            b.append(i)
    keep_idx = sorted(i for fam in sorted(buckets) for i in buckets[fam])
    samples = [test[i] for i in keep_idx]
    y_te = np.array([s.label for s in samples])
    fam_te = np.array([s.family for s in samples])
    texts_te = [s.text for s in samples]
    edit_mask = fam_te == "edit"
    normal_mask = fam_te == "normal"
    en_mask = edit_mask | normal_mask

    enc = get_encoder(cfg)
    ref_texts = build_reference(cfg, source="train")
    ref_embs = enc.encode(ref_texts)      # cache hit
    dev_embs = enc.encode(dev_normals)    # cache hit
    test_embs = enc.encode(texts_te)      # cache hit

    # 原始组件分（不经任何自归一化）
    dev_npz = np.load(metrics_dir / "dev_scores_real_v3.npz", allow_pickle=True)
    test_npz = np.load(metrics_dir / "test_scores_real_v3.npz", allow_pickle=True)
    base_dev = np.asarray(dev_npz["sup_tfidf"], dtype=float)[:2000]
    base_te = np.asarray(test_npz["sup_tfidf"], dtype=float)[keep_idx]

    rag_dev = scan_scores(dev_normals, ref_texts, ref_embs, enc, q_embs=dev_embs,
                          top_k=5, min_ratio=0.75, progress=False, aggregate="max")
    rag_te = scan_scores(texts_te, ref_texts, ref_embs, enc, q_embs=test_embs,
                         top_k=5, min_ratio=0.75, progress=False, aggregate="max")
    print(f"[calib] scan done dev={len(rag_dev)} test={len(rag_te)}")

    methods = {
        "fused_eq_w0.5": (0.5, "等权 z：dev-fit μ/σ 套 test"),
        "fused_z_w0.7base": (0.7, "加权 z(0.7 base + 0.3 rag)：dev-fit μ/σ 套 test"),
    }
    res = {}
    for name, (w, note) in methods.items():
        fdev, fte, stats = _dev_fit_fused(base_dev, rag_dev, base_te, rag_te, w)
        cal = _apply_fit(fdev, fte, y_te, edit_mask, normal_mask)
        res[name] = {
            "w_base": w,
            "note": note,
            "stats": stats,
            "auroc_all": _auc(y_te, fte),
            "edit_norm_auroc": _auc(y_te[en_mask], fte[en_mask]),
            **cal,
            "mean_fused_dev": round(float(fdev.mean()), 4),
            "mean_fused_test": round(float(fte.mean()), 4),
        }

    # 对照：eval_fusion 自归一化版本的旧口径（读产物复核一致性）
    old = json.loads((metrics_dir / "ragcheck_fusion.json").read_text(encoding="utf-8"))
    ref_old = {
        "fused_eq_old": old["fusion_by_method"]["fused_eq"]["tpr_at_fpr1_dev"],
        "fused_z_old": old["fusion_by_method"]["fused_z"]["tpr_at_fpr1_dev"],
        "ragcheck_alone_edit": old["fusion_by_method"]["ragcheck"]["edit_recall_dev"],
    }

    out = {
        "meta": {
            "task": "A1 fusion calibration fix",
            "branch": "feat/rag-guard",
            "encoder": "crom", "corpus_source": "train", "corpus_n": len(ref_texts),
            "aggregate": "max", "top_k": 5, "min_ratio": 0.75,
            "n_dev_clean": len(dev_normals), "n_test": len(samples),
            "seconds": round(time.perf_counter() - t0, 1),
            "method": "dev-fit mu/sigma applied to test; q99 threshold from fused dev",
        },
        "fusion_by_method": res,
        "reference_old_eval": ref_old,
    }
    (metrics_dir / "ragcheck_fusion_calibrated.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== A1 校准后融合（dev-fit → test-apply） =====")
    for name, r in res.items():
        print(f"{name:24s} AUC(all)={r['auroc_all']} edit_norm_AUC={r['edit_norm_auroc']} "
              f"TPR@1%(dev)={r['tpr_at_fpr1_dev']} edit召回={r['edit_recall_dev']} "
              f"normFPR={r['normal_fpr_dev']} thr={r['threshold_dev_q99']:.4f} "
              f"fused_dev mean {r['mean_fused_dev']} / test {r['mean_fused_test']}")
    print("旧口径(自归一化)对照:", ref_old)
    print(f"总耗时 {out['meta']['seconds']}s")


if __name__ == "__main__":
    main()
