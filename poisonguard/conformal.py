"""Conformal Prediction（A2）：把「1% FPR」从经验估计升级成带有限样本保证的声明。

背景
----
现有的 `Calibration.fit` 用 dev 干净样本的经验分位数定阈值，然后声称
「上线误报率 ≈ 1%」。这个声称是**无保证的经验值**：校准集只有 N 条，
经验阈值只是样本分位数，在 test 上的真实误报率围绕 α 波动，没有任何
「≤ α 以高概率成立」的定理撑腰。面试官问「你怎么知道是 1%？」时，
诚实的回答是「这是经验估计，样本越多越准」——而 A2 把它升级成：

    P(新干净样本被误报为投毒) ≤ α ，且该声称带有限样本保证。

Split Conformal（Papadopoulos et al. 2002；Lei, G'Sell, Rinaldo,
Tibshirani & Wasserman 2018）给出精确形式：

    校准集 (Z_1..Z_n) 与测试样本 X 交换（i.i.d. 即交换）
    → 非正式分数 s(X_i) 与 s(X) 全部交换
    → 令 q̂ = 校准分数第 ⌈(1−α)(n+1)⌉ 小的顺序统计量：
        P( s(X) > q̂ ) ≤ (n+1−⌈(1−α)(n+1)⌉)/(n+1) ≤ α

把「s(X) 超过阈值」翻译成本项目的检测语言：
   · 分数 = 检测器原始分（越高越像投毒），干净样本应拿低分
   · 干净样本被 s(X)>q̂ 误判 = 一次误报（FPR 错）
   · 于是上式给出：**下一个干净样本被误报的概率 ≤ α，不需要渐近假设、
     小校准集也严格成立**（只要求「校准集与线上干净数据同分布」这一条交换性）。

本项目还加一个「诚实两层」习惯：
   - 构造侧：阈值只由 dev 干净样本定，test 永远不参与。
   - 验证侧：把 test 干净样本当 hold-out，实测误报率 r̂；
     两个数都写进报告——「保证 ≤ α」与「实测 r̂」的关系是面试亮点：
     α 是上界，r̂ 是它实际落在哪，两者都要讲。

本模块实现
    split_threshold(cal_scores, alpha)      : 阈值 + 可证上界
    fpr_bound_tight(cal_scores, te_clean, alpha): 双层（构造+验证）封装
    run_conformal(cfg, out_path, alpha)     : cli 'conformal' 入口，纯吃缓存

唯一假设：dev 与 test 的干净样本同分布（本项目同一 real 语料、不同随机种子）。
CPU 上秒级完成（不重编码，直接读 score npz 缓存）。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .detectors import Calibration
from .evaluate import recall_by_family, save_json

ALPHA_DEFAULT = 0.01  # 目标 FPR（顺带 = 项目里 cfg.detector.target_fpr）


# ----------------------------------------------------------------------
def split_threshold(cal_scores: np.ndarray, alpha: float = ALPHA_DEFAULT):
    """Split-conformal 阈值：校准分数第 ⌈(1−α)(n+1)⌉ 小的那个。

    返回 (qhat, bound, ceil_term, n)：
      qhat     ：判定「异常/投毒」的分数阈值（分数 > qhat 判为投毒）。
      bound    ：k/(n+1) = (n+1−⌈(1−α)(n+1)⌉)/(n+1)，
                 「下一个干净样本被误报」的**可证**概率上界（≤ α）。
      ceil_term：⌈(1−α)(n+1)⌉，日志展示用。
      n        ：校准样本数。
    """
    s = np.asarray(cal_scores, dtype=np.float64).ravel()
    n = int(s.size)
    if n == 0:
        raise ValueError("校准集为空")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha 必须在 (0,1)，收到 {alpha}")
    k = int(np.ceil((1.0 - alpha) * (n + 1)))  # 允许的「非误报」顺序位数量
    if k <= n:
        qhat = float(np.sort(s)[k - 1])
    else:  # n 极小且 alpha 极小导致 k>n：保守取最大值（0 误报）
        qhat = float(np.max(s))
    bound = (n + 1 - k) / (n + 1)
    return qhat, bound, k, n


def fpr_bound_tight(cal_scores: np.ndarray,
                    test_scores: np.ndarray,
                    alpha: float = ALPHA_DEFAULT) -> dict:
    """两层封装：构造侧出阈值，验证侧在 hold-out 上量实测 + 有限样本宽。

    构造侧（有保证）：第 ⌈(1−α)(n+1)⌉ 分位数 → P(单次误报) ≤ bound ≤ α。
    验证侧（诚实）：  m 条 test 干净样本实测误报率 r̂，
                     有限样本上界 r̂ + sqrt(log(2/δ)/(2m))（Hoeffding 单侧，δ=0.05）。
    返回 dict。test 只验证、绝不参与构造。
    """
    qhat, bound, k, n = split_threshold(cal_scores, alpha)
    t = np.asarray(test_scores, dtype=np.float64).ravel()
    m = int(t.size)
    r_hat = float((t > qhat).mean()) if m else float("nan")
    delta = 0.05
    width = float(np.sqrt(np.log(2.0 / delta) / (2.0 * max(m, 1)))) if m else float("nan")
    return {
        "conformal_threshold": round(qhat, 6),
        "fpr_bound_conformal": round(bound, 6),
        "alpha": alpha,
        "n_cal": n,
        "k_ceil": int(k),
        "test_n": m,
        "test_fpr_achieved": round(float(r_hat), 6),
        "test_fpr_upper_hoeffding": round(float(r_hat + width), 6) if m else None,
        "bound_kind": "split_conformal_exchangeability",
        "note": (
            f"阈值=第⌈(1−α)(n+1)⌉={k}小的校准分；"
            f"可证 P(下一个干净样本误报) ≤ {bound:.4f}（交换性，不依赖 test）；"
            f"test 仅验证(m={m})，实测误报 {r_hat:.4f}。"
        ),
    }


# ----------------------------------------------------------------------
def suffix_of(cfg) -> str:
    """与 detect/redteam 同一套后缀推导（`_real_v3`/`_afqmc`/``）。"""
    name = Path(cfg.data.processed_dir).name
    if name.startswith("processed"):
        return name[len("processed"):]
    return ""


def _metrics_path(cfg, name: str) -> Path:
    stem, dot, ext = name.partition(".")
    return Path(cfg.metrics_dir) / (f"{stem}{suffix_of(cfg)}{dot}{ext}" if dot else name)


def _load_npz(cfg, name: str) -> dict:
    p = Path(cfg.metrics_dir) / f"{name}{suffix_of(cfg)}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"缺少分数缓存 {p}。请先运行 detect 生成 dev_scores/test_scores。")
    return np.load(p, allow_pickle=True)


def _clean_mask(families: np.ndarray) -> np.ndarray:
    """分布内干净样本 mask（与 cli._indist_clean_mask 同一约定：normal+hardneg）。"""
    fam = np.asarray(families)
    return (fam == "normal") | (fam == "hardneg")


# ----------------------------------------------------------------------
def run_conformal(cfg, out_path: Path | None = None,
                  alpha: float = ALPHA_DEFAULT) -> dict:
    """主入口：对全部检测器算 split-conformal 阈值 + 可证 FPR 上界 + TPR。

    纯吃 dev/test 分数缓存，不重编码，CPU 秒级完成。
    """
    t0 = time.time()
    dev = _load_npz(cfg, "dev_scores")
    test = _load_npz(cfg, "test_scores")

    test_y = np.asarray(test["y"], dtype=int)

    # 家族名从 processed jsonl 取（npz 只存了 test 的 families；dev 用 jsonl 对齐）
    from .data import load_splits
    splits = load_splits(Path(cfg.data.processed_dir))
    dev_fams = np.array([s.family for s in splits["dev"]])
    test_fams = np.array([s.family for s in splits["test"]])
    if dev_fams.size != len(dev["y"]) or test_fams.size != len(test_y):
        raise ValueError("npz 与 processed jsonl 样本数不一致，请重新 detect")

    dev_clean = _clean_mask(dev_fams)
    test_clean = _clean_mask(test_fams)

    det_names = [k for k in dev.files if k not in ("y", "families")]

    out: dict = {
        "meta": {
            "method": "split_conformal_prediction",
            "alpha": alpha,
            "n_cal_clean": int(dev_clean.sum()),
            "n_test_clean": int(test_clean.sum()),
            "n_test_pos": int((test_y == 1).sum()),
            "assumption": "校准集与测试干净侧同分布（同一 real 语料、不同随机种子）",
            "lemma": (
                "交换性引理（Lei et al. 2018 Lemma 2）：P(下一个干净样本被误报) "
                "≤ (n+1−⌈(1−α)(n+1)⌉)/(n+1) — 无渐近假设，小校准集也严格成立"
            ),
        },
        "by_detector": {},
    }

    # 伪 Sample 只用来复用 recall_by_family（它只看 label/family 两个字段）
    from types import SimpleNamespace
    samples = [
        SimpleNamespace(text="", label=int(l), family=str(f))
        for l, f in zip(test_y.tolist(), test_fams.tolist())
    ]

    for name in det_names:
        sd = np.asarray(dev[name], dtype=np.float64)
        st = np.asarray(test[name], dtype=np.float64)
        cal_scores = sd[dev_clean]
        te_clean_scores = st[test_clean]
        te_pos_scores = st[test_y == 1]

        # 对照物：现有经验分位数阈值（无保证）
        cal = Calibration.fit(cal_scores, alpha)
        # 有保证：split-conformal 阈值 + 两层界
        cf = fpr_bound_tight(cal_scores, te_clean_scores, alpha)
        thr = cf["conformal_threshold"]

        te_fp = float((te_clean_scores > thr).mean())
        tpr = float((te_pos_scores > thr).mean()) if te_pos_scores.size else float("nan")
        pred = (st > thr).astype(int)
        fam_rec = recall_by_family(samples, pred)

        out["by_detector"][name] = {
            "empirical_threshold": round(cal.raw_threshold, 6),
            "empirical_dev_fpr": round(cal.achieved_fpr, 6),
            **cf,
            "test_tpr_at_conformal": round(float(tpr), 6),
            "family_recall": fam_rec,
            "n_pos": int(te_pos_scores.size),
        }
        print(f"[conformal] {name:<12} 经验thr={cal.raw_threshold:.4g} → "
              f"CF thr={thr:.4g} (可证FPR上界 {cf['fpr_bound_conformal']:.4f}, "
              f"实测test FPR {te_fp:.4f}, TPR {tpr:.4f})")

    out["seconds"] = round(time.time() - t0, 2)
    if out_path is None:
        out_path = _metrics_path(cfg, "conformal.json")
    save_json(out, out_path)
    print(f"[conformal] 已保存 {out_path}（用时 {out['seconds']}s）")
    return out


def main() -> None:
    raise SystemExit("请通过 `python -m poisonguard.cli conformal ...` 调用")
