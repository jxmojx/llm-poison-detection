"""评估工具：指标、置信区间、显著性检验、论文级绘图。

指标选择(这是"AI 岗简历项目"必须站得住的部分)：
- 只有准确率是不够的：测试集正类比例 ~60%，但真实投毒比例只有 1% 量级。
- 因此主指标用 **AUROC / AUPRC**(与正类比例无关的排序能力) +
  **TPR@FPR=1%**(真实运维场景:一天几十万条数据，误报率必须压到 1% 以下)，
  再辅以校准阈值下的 Precision / Recall / F1。
- 用 Bootstrap 给 95% 置信区间，用 McNemar 检验判断两个检测器的差异是否显著。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
def _check(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if len(y) != len(s):
        raise ValueError(f"长度不一致: y={len(y)} score={len(s)}")
    return y, s


def auroc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score

    y, s = _check(y_true, scores)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(y_true, scores) -> float:
    from sklearn.metrics import average_precision_score

    y, s = _check(y_true, scores)
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, s))


def tpr_at_fpr(y_true, scores, fpr_target: float = 0.01) -> float:
    from sklearn.metrics import roc_curve

    y, s = _check(y_true, scores)
    if y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    idx = int(np.searchsorted(fpr, fpr_target, side="right")) - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def tpr_at_fpr_calibrated(y_cal, s_cal, y_eval, s_eval, fpr_target: float = 0.01) -> float:
    """在**校准集**上定阈值，在**评估集**上量 TPR。

    和 `tpr_at_fpr` 的区别是关键的：`tpr_at_fpr` 用评估集自己的 ROC 找点，
    等于"看了答案再划阈值"，是乐观上界；这里阈值只由校准集的干净样本决定，
    才是上线后能拿到的数字。差值越大，说明方法的泛化越依赖测试集分布。
    """
    yc, sc = _check(y_cal, s_cal)
    ye, se = _check(y_eval, s_eval)
    neg = sc[yc == 0]
    if neg.size == 0 or ye.min() == ye.max():
        return float("nan")
    thr = float(np.quantile(neg, 1.0 - fpr_target))
    pos = se[ye == 1]
    if pos.size == 0:
        return float("nan")
    return float((pos > thr).mean())


def ece(y_true, scores, bins: int = 15) -> float:
    """Expected Calibration Error：|平均置信度 − 实际正例率| 的分箱加权和。

    判官输出的是概率，概率"能不能当概率用"必须单独检验——logprob 打分常见
    的问题是过度自信（阈值被推到 0.96 就是征兆），ECE 把这件事量化出来。
    """
    y, s = _check(y_true, scores)
    if y.size < bins or y.min() == y.max():
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += abs(float(y[m].mean()) - float(s[m].mean())) * float(m.mean())
    return float(total)


def binary_metrics(y_true, scores, threshold: float) -> dict:
    y, s = _check(y_true, scores)
    pred = (s > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "accuracy": round((tp + tn) / max(len(y), 1), 4),
        "auroc": round(auroc(y, s), 4),
        "auprc": round(auprc(y, s), 4),
        "tpr_at_fpr1": round(tpr_at_fpr(y, s, 0.01), 4),
        "tpr_at_fpr5": round(tpr_at_fpr(y, s, 0.05), 4),
        "n": int(len(y)),
    }


# ----------------------------------------------------------------------
def bootstrap_ci(y_true, scores, metric=auroc, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    y, s = _check(y_true, scores)
    rng = np.random.default_rng(seed)
    vals = []
    idx_all = np.arange(len(y))
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(y), replace=True)
        try:
            v = metric(y[idx], s[idx])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    vals = np.array(vals)
    return (
        float(metric(y, s)),
        float(np.quantile(vals, alpha / 2)),
        float(np.quantile(vals, 1 - alpha / 2)),
    )


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """成对比较两个检测器(同一测试集)，判断差异是否显著。"""
    from scipy.stats import binomtest, chi2

    y = np.asarray(y_true).astype(int)
    a = np.asarray(pred_a).astype(int) == y
    b = np.asarray(pred_b).astype(int) == y
    only_a = int((a & ~b).sum())  # a 对 b 错
    only_b = int((~a & b).sum())  # b 对 a 错
    n = only_a + only_b
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "only_a": 0, "only_b": 0, "test": "none"}
    if n < 25:
        p = binomtest(only_a, n, 0.5).pvalue
        return {"statistic": float(min(only_a, only_b)), "p_value": float(p),
                "only_a": only_a, "only_b": only_b, "test": "exact_binomial"}
    stat = (abs(only_a - only_b) - 1) ** 2 / n
    p = float(chi2.sf(stat, df=1))
    return {"statistic": float(stat), "p_value": p, "only_a": only_a,
            "only_b": only_b, "test": "chi2_corrected"}


def recall_by_family(samples, preds) -> dict:
    """按攻击家族统计召回率(只统计正类样本)。"""
    fams: dict[str, list[int]] = {}
    for s, p in zip(samples, preds):
        if s.label != 1:
            continue
        fams.setdefault(s.family, []).append(int(p))
    return {f: {"n": len(v), "recall": round(float(np.mean(v)), 4)} for f, v in fams.items()}


def fpr_by_group(samples, preds) -> dict:
    """按干净样本来源(normal / ood)统计误报率。"""
    groups: dict[str, list[int]] = {}
    for s, p in zip(samples, preds):
        if s.label != 0:
            continue
        groups.setdefault(s.family, []).append(int(p))
    return {g: {"n": len(v), "fpr": round(float(np.mean(v)), 4)} for g, v in groups.items()}


# ----------------------------------------------------------------------
def cascade_curve(
    first_scores,
    judge_scores,
    y_true,
    budgets=(0.05, 0.10, 0.20, 0.40, 0.60, 1.0),
    judge_threshold: float = 0.5,
) -> list[dict]:
    """级联检测的**召回-成本曲线**。

    运维真正要回答的问题不是"级联省了多少次调用"，而是
    "如果我只付得起 X% 的 LLM 调用，能抓到多少投毒"。
    做法：按第一级分数从高到低排序，只把前 `budget` 比例的样本送 LLM 判定，
    其余直接当干净；命中判官阈值的才算检出。

    与 v2 的区别：v2 用固定分位(0.80~0.995)给出一个单点("调用 19.4%、召回 16.7%")，
    读者无法判断这是门控选错了还是级联本身不成立。曲线把整条权衡摊开，
    并且用 `random_recall` 给出"同样预算下随机猜"的对照，直接看增益。
    """
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(first_scores, dtype=np.float64).ravel()
    js = np.asarray(judge_scores, dtype=np.float64).ravel()
    n = len(y)
    if n == 0:
        return []
    n_pos = int((y == 1).sum())
    order = np.argsort(-s)
    rows: list[dict] = []
    for b in budgets:
        k = max(1, min(n, int(round(b * n))))
        sel = order[:k]
        pred = np.zeros(n, dtype=int)
        pred[sel] = (js[sel] > judge_threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int((n_pos) - tp)
        rows.append({
            "budget": b,
            "call_ratio": round(k / n, 4),
            "n_calls": k,
            "recall": round(tp / max(tp + fn, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            # 同样预算下"随机挑 k 条送 LLM"的期望召回(超几何均值)
            "random_recall": round(b * n_pos / n, 4),
        })
    return rows


# ----------------------------------------------------------------------
def setup_matplotlib_font() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def plot_roc(curves: dict[str, tuple], path: str | Path, title: str = "ROC 曲线") -> Path:
    from sklearn.metrics import roc_curve

    setup_matplotlib_font()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.4, 5.2))
    for name, (y, s) in curves.items():
        y = np.asarray(y).astype(int)
        if y.min() == y.max():
            continue
        fpr, tpr, _ = roc_curve(y, np.asarray(s, dtype=float))
        auc = auroc(y, s)
        plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="随机基线")
    plt.axvline(0.01, color="tab:red", ls=":", lw=1, label="FPR=1%")
    plt.xlabel("误报率 FPR")
    plt.ylabel("召回率 TPR")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[plot] {path}")
    return path


def plot_family_heatmap(matrix: dict[str, dict[str, float]], path: str | Path,
                        title: str = "各攻击家族召回率 (%)") -> Path:
    setup_matplotlib_font()
    import matplotlib.pyplot as plt

    detectors = list(matrix.keys())
    families = sorted({f for m in matrix.values() for f in m})
    data = np.array([[matrix[d].get(f, np.nan) * 100 for f in families] for d in detectors])

    plt.figure(figsize=(1.6 + 1.5 * len(families), 1.2 + 0.8 * len(detectors)))
    im = plt.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.xticks(range(len(families)), families, rotation=15)
    plt.yticks(range(len(detectors)), detectors)
    for i in range(len(detectors)):
        for j in range(len(families)):
            v = data[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                         color="black" if v > 45 else "white")
    plt.title(title)
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[plot] {path}")
    return path


def plot_bars(series: dict[str, float], path: str | Path, ylabel: str = "",
              title: str = "", ref_line: float | None = None) -> Path:
    setup_matplotlib_font()
    import matplotlib.pyplot as plt

    names = list(series.keys())
    vals = [series[k] for k in names]
    plt.figure(figsize=(max(6, 1.1 * len(names)), 4.2))
    bars = plt.bar(names, vals, color="#4C78A8")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                 va="bottom", fontsize=9)
    if ref_line is not None:
        plt.axhline(ref_line, color="tab:red", ls="--", lw=1)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[plot] {path}")
    return path


def plot_curve(x, ys: dict[str, list[float]], path: str | Path, xlabel: str = "",
               ylabel: str = "", title: str = "") -> Path:
    setup_matplotlib_font()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.2, 4.4))
    for name, y in ys.items():
        plt.plot(x, y, marker="o", lw=1.8, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[plot] {path}")
    return path


def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default),
                    encoding="utf-8")
    print(f"[eval] 指标已保存 {path}")
    return path
