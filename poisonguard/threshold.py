"""阈值口径（P0-1）：让「上线用的阈值」只有一个来源。

起因是一次真实的自查：`ingest` 用学生模型自带的 `threshold=0.5`，而同一个模型在
dev 干净集上的 1% 工作点其实是 `0.9981`——同一批数据，误报率 60% 与 1% 的差别
完全来自这个口径不一致。

规则：
1. 打包（`deploy`）时就在 dev 干净样本上**标定**一次，把阈值写进 `manifest`；
2. 加载（`run`/`ingest --bundle`）时以包内阈值为准；
3. 本地拟合态（`ingest` 不带 `--bundle`）也在启动时标定；
4. `--threshold` 永远可以显式覆盖，但会打印它是"人工覆盖"，避免悄悄生效。
"""

from pathlib import Path

import numpy as np

CLEAN_FAMILIES = ("normal", "hardneg")


def clean_dev_texts(cfg) -> list[str]:
    """dev 里"干净且在分布内"的样本文本：标定阈值只用这些。"""
    from .data import load_splits

    dev = load_splits(cfg.data.processed_dir).get("dev", [])
    return [s.text for s in dev if s.label == 0 and s.family in CLEAN_FAMILIES]


def scorer_scores(scorer, texts: list[str]) -> np.ndarray:
    """统一取分：PoisonScorer 与 DeployedScorer 的输出形状相同（results[].score）。"""
    out = scorer.score(list(texts))
    return np.asarray([float(r["score"]) for r in out["results"]], dtype=np.float64)


def calibrate_threshold(scorer, clean_texts: list[str],
                        target_fpr: float = 0.01) -> dict:
    """在干净样本上取 1% 分位作为上线阈值（与 `Calibration.fit` 同一口径）。"""
    texts = [t for t in clean_texts if isinstance(t, str) and t.strip()]
    if not texts:
        raise ValueError("标定阈值需要非空的 dev 干净样本")
    s = np.sort(scorer_scores(scorer, texts))
    idx = int(round((1.0 - target_fpr) * (len(s) - 1)))
    thr = float(s[idx])
    achieved = float((s > thr).mean())
    return {
        "threshold": thr,
        "achieved_fpr": achieved,
        "n_clean": int(len(s)),
        "target_fpr": float(target_fpr),
        "method": "empirical_quantile",
        "score_mean_clean": float(s.mean()),
        "score_max_clean": float(s.max()),
    }


def resolve_threshold(scorer, cfg=None, *, override: float | None = None,
                      calibrate: bool = True, target_fpr: float | None = None) -> dict:
    """决定这次用哪个阈值，并说明来源（cli / calibrated / manifest / default）。"""
    if override is not None:
        return {"threshold": float(override), "source": "cli_override",
                "detail": "命令行 --threshold 显式覆盖（不推荐长期这么用）"}
    if calibrate and cfg is not None:
        texts = clean_dev_texts(cfg)
        if texts:
            cal = calibrate_threshold(
                scorer, texts,
                target_fpr if target_fpr is not None else cfg.detector.target_fpr)
            return {"threshold": cal["threshold"], "source": "calibrated",
                    "detail": f"在 {cal['n_clean']} 条 dev 干净样本上标定"
                              f"（目标 FPR {cal['target_fpr']:.2%}，"
                              f"实测 {cal['achieved_fpr']:.2%}）",
                    "calibration": cal}
    return {"threshold": float(getattr(scorer, "threshold", 0.5)),
            "source": "scorer_default",
            "detail": "用打分包/打分器自带的阈值（与标定口径可能不一致）"}


def describe(res: dict) -> str:
    return f"threshold={res['threshold']:.4f}（来源 {res['source']}：{res['detail']}）"
