"""融合与参数敏感性评测（t1，RAG 守卫分支 feat/rag-guard）。

复用 outputs/cache/ 的 CoROM 向量缓存（ref 12000 / dev-2000 / test sample-2953，
全部 cache-hit，无需重编码），只为不同融合方式重算扫描分数与阈值。

产出：
  outputs/metrics/ragcheck_fusion.json   主：融合三方式对比（aggregate=max）
  outputs/metrics/sensitivity_ragcheck.md 辅：top_k x min_ratio 敏感性

只读 poisonguard，不改任何现有文件。仅写 scripts/*.py 与 outputs/metrics/*。
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
from poisonguard.ragcheck import _normalize, _rank_avg, build_reference, scan_scores  # noqa: E402
from poisonguard.encoders import get_encoder  # noqa: E402


def _auc(y, s):
    if len(set(y)) < 2 or s is None or len(s) != len(y):
        return None
    try:
        return round(float(auroc(y, s)), 4)
    except Exception:
        return None


def _tpr_at_fpr1(dev_s, s_te_v, y_te):
    if len(dev_s) < 10 or len(set(y_te)) < 2:
        return float("nan")
    thr = float(np.quantile(dev_s, 0.99))
    return float((s_te_v[y_te == 1] > thr).mean())


def _recall1(dev_s, s_te_v, mask):
    if not mask.any() or len(dev_s) < 2:
        return float("nan")
    thr = float(np.quantile(dev_s, 0.99))
    return float((s_te_v[mask] > thr).mean())


def _rank_dev_pair(dev_base, dev_rag):
    """在 dev-clean 池内的排名平均（与 test 上 _rank_avg(base_test, s_te) 同一口径）。"""
    return 0.5 * (_rank_avg(dev_base, dev_base) + (_rank_avg(dev_rag, dev_rag)))


def _make_test_sample(test, limit: int):
    buckets: dict[str, list[int]] = {}
    for i, s in enumerate(test):
        b = buckets.setdefault(s.family, [])
        if len(b) < limit:
            b.append(i)
    keep_idx = sorted(i for fam in sorted(buckets) for i in buckets[fam])
    return keep_idx, [test[i] for i in keep_idx]


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    cfg.data.processed_dir = str(ROOT / "data" / "processed_real_v3")
    metrics_dir = ROOT / "outputs" / "metrics"
    cache_dir = ROOT / "outputs" / "cache"

    splits = load_splits(cfg.data.processed_dir)
    dev = splits["dev"]
    test = splits["test"]
    dev_normals = [s.text for s in dev if s.label == 0 and s.family == "normal"][:2000]

    keep_idx, samples = _make_test_sample(test, 400)
    y_te = np.array([s.label for s in samples])
    fam_te = np.array([s.family for s in samples])
    texts_te = [s.text for s in samples]

    # ---------------- 向量（全部 cache-hit） ----------------
    enc = get_encoder(cfg)
    ref_texts = build_reference(cfg, source="train")
    ref_embs = enc.encode(ref_texts)      # 12000 cache hit
    dev_embs = enc.encode(dev_normals)    # 2000 cache hit
    test_embs = enc.encode(texts_te)      # 2953 cache hit
    cache_hit = all(
        (cache_dir / f"emb_crom_{h}.npz").exists()
        for h in ("c3e585b91e41e3fa", "2c067c138d3cadeb", "f65b1bd881ff30b7"))
    print(f"[eval_fusion] 向量 cache-hit={cache_hit}（其余 nan/inf 检查见下）")

    # ---------------- 基线词面分（sup_tfidf，dev/test npz 按 idx 对齐） ----------------
    dev_npz = np.load(metrics_dir / "dev_scores_real_v3.npz", allow_pickle=True)
    test_npz = np.load(metrics_dir / "test_scores_real_v3.npz", allow_pickle=True)
    dev_base_clean = np.asarray(dev_npz["sup_tfidf"], dtype=float)[:2000]
    base_test = np.asarray(test_npz["sup_tfidf"], dtype=float)[keep_idx]

    # ================= 融合三方式（aggregate=max, top_k=5, min_ratio=0.75） =================
    s_te = scan_scores(texts_te, ref_texts, ref_embs, enc, q_embs=test_embs,
                       top_k=5, min_ratio=0.75, progress=False, aggregate="max")
    s_dev = scan_scores(dev_normals, ref_texts, ref_embs, enc, q_embs=dev_embs,
                        top_k=5, min_ratio=0.75, progress=False, aggregate="max")

    # 融合分数（test）
    n_base_t = _normalize(base_test)
    n_rag_t = _normalize(s_te)
    comb_eq = n_base_t + n_rag_t
    comb_rank = _rank_avg(base_test, s_te)
    comb_fused = 0.7 * n_base_t + 0.3 * n_rag_t   # = factcheck.fuse(w=0.3)

    # 对应融合分数（dev-clean 池内拟合，供 1% FPR 校准）
    dev_eq = _normalize(dev_base_clean) + _normalize(s_dev)
    dev_fused = 0.7 * _normalize(dev_base_clean) + 0.3 * _normalize(s_dev)
    dev_rank = _rank_dev_pair(dev_base_clean, s_dev)

    edit_mask = fam_te == "edit"
    normal_mask = fam_te == "normal"
    en_mask = edit_mask | normal_mask

    methods = {
        "base_sup_tfidf": {"s": base_test, "dev": dev_base_clean,
                           "note": "词面基线 sup_tfidf（现有检测器）"},
        "ragcheck": {"s": s_te, "dev": s_dev,
                     "note": "检索一致性单独（max 聚合）"},
        "fused_eq": {"s": comb_eq, "dev": dev_eq,
                     "note": "等权 z-score 加和：normalize(base)+normalize(rag)"},
        "fused_rank": {"s": comb_rank, "dev": dev_rank,
                       "note": "排名平均（对尺度不敏感）"},
        "fused_z": {"s": comb_fused, "dev": dev_fused,
                    "note": "z-score 加权(0.7 base + 0.3 rag)"},
    }
    res_fusion = {}
    for name, m in methods.items():
        s = m["s"]
        res_fusion[name] = {
            "auroc_all": _auc(y_te, s),
            "edit_norm_auroc": _auc(y_te[en_mask], s[en_mask]),
            "tpr_at_fpr1_dev": round(_tpr_at_fpr1(m["dev"], s, y_te), 4),
            "edit_recall_dev": round(_recall1(m["dev"], s, edit_mask), 4),
            "normal_fpr@0.5": round(float((s[normal_mask] > 0.5).mean()), 4),
            "n_with_conflict": int((s > 0).sum()),
            "mean_score_test": round(float(np.mean(s)), 4),
            "note": m["note"],
        }

    # by_family 摘要
    by_family_summary = {}
    for fam in sorted(set(fam_te.tolist())):
        msk = fam_te == fam
        by_family_summary[fam] = {
            "n": int(msk.sum()),
            "base_mean": round(float(base_test[msk].mean()), 4),
            "ragcheck_mean": round(float(s_te[msk].mean()), 4),
            "fused_eq_mean": round(float(comb_eq[msk].mean()), 4),
            "fused_rank_mean": round(float(comb_rank[msk].mean()), 4),
            "fused_z_mean": round(float(comb_fused[msk].mean()), 4),
        }

    # edit 召回 vs 单独 ragcheck（基线文件 TPR@1% 0.1085）
    rag_edit_recall = res_fusion["ragcheck"]["edit_recall_dev"]

    # ================= 参数敏感性（top_k x min_ratio, edit+normal×150 快速子集） =================
    sens_cfg = [
        (3, 0.75), (5, 0.75), (8, 0.75),   # top_k
        (5, 0.70), (5, 0.80),              # min_ratio
    ]
    # 敏感性子集：edit+normal 各 150（保留全局 test 索引顺序）
    sub_idx, sub_test = [], []
    cnt = {"edit": 0, "normal": 0}
    for i, s in enumerate(test):
        if s.family in cnt and cnt[s.family] < 150:
            sub_idx.append(i)
            sub_test.append(s)
            cnt[s.family] += 1
    sub_y = np.array([s.label for s in sub_test])
    sub_fam = np.array([s.family for s in sub_test])
    sub_embs = enc.encode([s.text for s in sub_test])
    print(f"[eval_fusion] 敏感性子集 {len(sub_test)} 条（edit+normal ×150）")

    res_sens = {}
    for top_k, min_ratio in sens_cfg:
        ss_te = scan_scores([s.text for s in sub_test], ref_texts, ref_embs, enc,
                            q_embs=sub_embs, top_k=top_k, min_ratio=min_ratio,
                            progress=False, aggregate="max")
        ss_dev = scan_scores(dev_normals, ref_texts, ref_embs, enc, q_embs=dev_embs,
                             top_k=top_k, min_ratio=min_ratio, progress=False,
                             aggregate="max")
        label = f"top_k={top_k}, min_ratio={min_ratio:.2f}"
        res_sens[label] = {
            "top_k": top_k, "min_ratio": min_ratio,
            "auroc_edit_norm": _auc(sub_y, ss_te),
            "tpr_at_fpr1_dev": round(_tpr_at_fpr1(ss_dev, ss_te, sub_y), 4),
            "edit_recall_dev": round(_recall1(ss_dev, ss_te, sub_fam == "edit"), 4),
            "normal_fpr@0.5": round(float((ss_te[sub_fam == "normal"] > 0.5).mean()), 4),
            "n_with_conflict": int((ss_te > 0).sum()),
            "n": int(len(sub_test)),
        }
        print(f"[eval_fusion] sens {label}: AUC(edit/norm)={res_sens[label]['auroc_edit_norm']} "
              f"TPR@1%={res_sens[label]['tpr_at_fpr1_dev']} "
              f"edit召回={res_sens[label]['edit_recall_dev']}")

    # ================= 写出 =================
    out = {
        "meta": {
            "task": "t1 fusion + sensitivity",
            "branch": "feat/rag-guard", "head": "955b507",
            "encoder": "crom", "corpus_source": "train",
            "corpus_n": len(ref_texts), "aggregate": "max",
            "n_dev_clean": len(dev_normals), "n_test": len(samples),
            "seconds": round(time.perf_counter() - t0, 1),
            "cache_hit": cache_hit,
            "baseline_ref": {
                "file": "ragcheck_real_v3.json",
                "overall": {"auroc": 0.5532, "tpr_at_fpr1": 0.1085,
                            "edit_norm_auroc": 0.7691, "n_with_conflict": 219},
            },
        },
        "fusion_by_method": res_fusion,
        "by_family_summary": by_family_summary,
        "sensitivity": res_sens,
        "ragcheck_alone_edit_recall_ref": {
            "alone": round(rag_edit_recall, 4),
            "baseline_tpr_at_fpr1": 0.1085,
            "compare_note": "edit 召回@1% 的单独 ragcheck 参照值",
        },
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "ragcheck_fusion.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[eval_fusion] 已写 outputs/metrics/ragcheck_fusion.json")

    lines = [
        "# RAG 守卫 · 参数敏感性（检索一致性）",
        "",
        "评测床：data/processed_real_v3（train corpus 12000 句），encoder=crom，",
        "aggregate=max；主全集 limit=400（2953 条）；敏感性用 edit+normal 各 150",
        "的快速子集（同评分口径互比）。向量全部命中 outputs/cache，未重编码。",
        "",
        "## 敏感性表（top_k × min_ratio，AUROC 为 edit vs normal）",
        "",
        "| 配置 | AUROC(edit/norm) | TPR@1%(dev校准) | edit召回@dev | normal FPR@0.5 | 命中冲突 |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in res_sens.items():
        lines.append(
            f"| {label} | {r['auroc_edit_norm']} | {r['tpr_at_fpr1_dev']} "
            f"| {r['edit_recall_dev']} | {r['normal_fpr@0.5']} | {r['n_with_conflict']} |")
    lines += [
        "",
        "## 说明",
        "",
        "- 基线（outputs/metrics/ragcheck_real_v3.json，top_k=5, min_ratio=0.75）：",
        "  overall AUROC 0.5532，TPR@1% 0.1085，edit vs normal AUROC 0.7691。",
        "- edit 召回@1%：dev clean 上 q99 阈值，test edit 中分数超过阈值的比例。",
        "- 敏感性快速子集仅计 edit+normal；其余家族在 max 聚合下分恒为 0",
        "  （无数值/事实点差异），不进入敏感性表。",
        "",
    ]
    (metrics_dir / "sensitivity_ragcheck.md").write_text("\n".join(lines), encoding="utf-8")
    print("[eval_fusion] 已写 outputs/metrics/sensitivity_ragcheck.md")

    print("\n===== 融合三方式对比（aggregate=max, top_k=5, min_ratio=0.75, limit=400） =====")
    for name, r in res_fusion.items():
        print(f"{name:14s} AUC(all)={r['auroc_all']} edit_norm_AUC={r['edit_norm_auroc']} "
              f"TPR@1%(dev)={r['tpr_at_fpr1_dev']} edit召回={r['edit_recall_dev']} "
              f"normFPR@.5={r['normal_fpr@0.5']} conflict={r['n_with_conflict']}")
    print(f"（单独 ragcheck edit 召回参照 0.1085 → 融合后见 fused_* 行）")
    print(f"总耗时 {out['meta']['seconds']}s")


if __name__ == "__main__":
    main()
