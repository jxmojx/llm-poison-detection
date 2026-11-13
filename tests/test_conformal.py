"""Conformal Prediction（A2）的纯逻辑测试：不下载模型、不读真实语料。

覆盖的是「数学是否正确」而不是「分数好不好」：
  · split_threshold 的顺序统计量公式与 n+1 修正
  · 可证上界 bound ≤ α（交换性引理留给它的唯一承诺）
  · 阈值对 α 的单调性、空校准集/非法 α 的拒绝
  · fpr_bound_tight 的验证侧口径（test 只验证、不参与构造）
  · 分布内干净样本 mask 与 suffix/路径推导
  · run_conformal 端到端：在极小伪数据上跑通并落盘
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from poisonguard.conformal import (_clean_mask, _metrics_path, fpr_bound_tight,
                                   run_conformal, split_threshold, suffix_of)


def _cfg(tmp_path: Path, processed_name: str = "processed_tmp"):
    metrics = tmp_path / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        metrics_dir=str(metrics),
        data=SimpleNamespace(processed_dir=str(tmp_path / processed_name)),
    )


# ----------------------------------------------------------------------
def test_split_threshold_is_order_statistic_with_plus_one():
    """qhat = 第 ⌈(1−α)(n+1)⌉ 小的校准分；bound = (n+1−k)/(n+1)。"""
    rng = np.random.RandomState(0)
    s = rng.normal(size=99)
    qhat, bound, k, n = split_threshold(s, alpha=0.10)
    k_expect = int(np.ceil(0.90 * (n + 1)))          # 90 of 100
    assert k == k_expect
    assert n == 99
    assert qhat == pytest.approx(float(np.sort(s)[k - 1]))
    assert bound == pytest.approx((n + 1 - k) / (n + 1))
    assert bound <= 0.10 + 1e-12


def test_bound_never_exceeds_alpha_for_many_sizes():
    """有限样本下界恒 ≤ α：小校准集只会更保守，绝不会更宽松。"""
    rng = np.random.RandomState(1)
    for n in (5, 17, 100, 999, 5000):
        for alpha in (0.01, 0.05, 0.10):
            _, bound, k, _ = split_threshold(rng.normal(size=n), alpha)
            assert 0.0 <= bound <= alpha + 1e-12
            assert 0 <= k <= n + 1


def test_threshold_monotone_in_alpha():
    """alpha 越小（要的误报越少）→ 阈值越高（越难判阳性）。"""
    rng = np.random.RandomState(2)
    s = rng.normal(size=500)
    q10, _, _, _ = split_threshold(s, alpha=0.10)
    q05, _, _, _ = split_threshold(s, alpha=0.05)
    q01, _, _, _ = split_threshold(s, alpha=0.01)
    assert q01 >= q05 >= q10


def test_empty_calibration_and_bad_alpha_rejected():
    with pytest.raises(ValueError, match="校准集为空"):
        split_threshold(np.array([]), 0.01)
    with pytest.raises(ValueError, match="alpha"):
        split_threshold(np.zeros(10), 0.0)
    with pytest.raises(ValueError, match="alpha"):
        split_threshold(np.zeros(10), 1.5)


def test_extreme_alpha_falls_back_to_max():
    """n 极小 + alpha 极小时 k>n：保守取最大校准分（0 误报）。"""
    s = np.array([0.1, 0.2, 0.3])
    qhat, bound, k, n = split_threshold(s, alpha=1e-6)
    assert k > n
    assert qhat == pytest.approx(0.3)
    assert 0.0 <= bound <= 1e-6


# ----------------------------------------------------------------------
def test_fpr_bound_tight_validation_layer_never_touches_construction():
    """构造侧只用 cal_scores；test 只用于量 r̂ 与有限样本宽。"""
    rng = np.random.RandomState(3)
    cal = rng.normal(size=400)
    te = rng.normal(size=800)
    r = fpr_bound_tight(cal, te, alpha=0.01)
    qhat, bound, _, n = split_threshold(cal, 0.01)
    assert r["conformal_threshold"] == pytest.approx(round(qhat, 6))
    assert r["fpr_bound_conformal"] == pytest.approx(round(bound, 6))
    assert r["n_cal"] == n
    assert r["test_n"] == 800
    assert r["test_fpr_achieved"] == pytest.approx(round(float((te > qhat).mean()), 6))
    # Hoeffding 单侧宽：sqrt(log(2/0.05)/(2m))
    width = np.sqrt(np.log(2.0 / 0.05) / (2.0 * 800))
    assert r["test_fpr_upper_hoeffding"] == pytest.approx(round(float((te > qhat).mean()) + width, 6))


def test_holdout_fpr_near_alpha_on_iid_clean():
    """干净分数 i.i.d. 时，hold-out 误报率应贴近 α（不是 0，也不该远超）。"""
    rng = np.random.RandomState(4)
    fprs = []
    for _ in range(20):
        cal = rng.normal(size=2000)
        te = rng.normal(size=2000)
        r = fpr_bound_tight(cal, te, alpha=0.01)
        fprs.append(r["test_fpr_achieved"])
    m = float(np.mean(fprs))
    assert m <= 0.02, f"平均误报 {m:.4f} 明显偏离 1%"


def test_deterministic_no_random_state():
    """同一输入两次调用结果一致（阈值/上界是纯函数）。"""
    s = np.arange(100, dtype=float)
    assert split_threshold(s, 0.01) == split_threshold(s, 0.01)


# ----------------------------------------------------------------------
def test_clean_mask_normal_and_hardneg_only():
    fams = np.array(["normal", "hardneg", "ood", "edit", "template"])
    assert _clean_mask(fams).tolist() == [True, True, False, False, False]


def test_suffix_of_matches_detect_convention():
    assert suffix_of(_cfg(Path("/tmp/x"), "processed_real_v3")) == "_real_v3"
    assert suffix_of(_cfg(Path("/tmp/x"), "processed_afqmc")) == "_afqmc"
    assert suffix_of(_cfg(Path("/tmp/x"), "processed")) == ""
    assert suffix_of(_cfg(Path("/tmp/x"), "other_dir")) == ""


def test_metrics_path_inserts_suffix():
    cfg = _cfg(Path("/tmp/x"), "processed_real_v3")
    assert _metrics_path(cfg, "conformal.json").name == "conformal_real_v3.json"


# ----------------------------------------------------------------------
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def test_run_conformal_end_to_end_on_fake_scores(tmp_path: Path):
    """极小伪数据端到端：读 npz → 逐检测器出阈值/上界/TPR → 落盘 JSON。"""
    cfg = _cfg(tmp_path, "processed_tmp")
    proc = Path(cfg.data.processed_dir)
    proc.mkdir(parents=True, exist_ok=True)

    dev_rows = ([{"text": f"d{i}", "label": 0, "family": "normal", "split": "dev"} for i in range(10)]
                + [{"text": f"h{i}", "label": 0, "family": "hardneg", "split": "dev"} for i in range(5)]
                + [{"text": f"a{i}", "label": 1, "family": "edit", "split": "dev"} for i in range(5)])
    test_rows = ([{"text": f"n{i}", "label": 0, "family": "normal", "split": "test"} for i in range(10)]
                 + [{"text": f"o{i}", "label": 0, "family": "ood", "split": "test"} for i in range(5)]
                 + [{"text": f"p{i}", "label": 1, "family": "backdoor", "split": "test"} for i in range(5)])
    _write_jsonl(proc / "train.jsonl", dev_rows[:3])
    _write_jsonl(proc / "dev.jsonl", dev_rows)
    _write_jsonl(proc / "test.jsonl", test_rows)

    rng = np.random.RandomState(7)
    dev_y = np.array([r["label"] for r in dev_rows])
    test_y = np.array([r["label"] for r in test_rows])
    dev_s = np.where(dev_y == 1, rng.normal(10, 0.5, dev_y.size), rng.normal(0, 0.1, dev_y.size))
    te_s = np.where(test_y == 1, rng.normal(10, 0.5, test_y.size), rng.normal(0, 0.1, test_y.size))
    np.savez_compressed(Path(cfg.metrics_dir) / "dev_scores_tmp.npz", y=dev_y, det=dev_s)
    np.savez_compressed(Path(cfg.metrics_dir) / "test_scores_tmp.npz", y=test_y,
                        families=np.array([r["family"] for r in test_rows]), det=te_s)

    out_path = Path(cfg.metrics_dir) / "conformal_tmp.json"
    res = run_conformal(cfg, out_path=out_path, alpha=0.10)

    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["meta"]["n_cal_clean"] == 15
    assert saved["meta"]["n_test_clean"] == 10
    assert saved["meta"]["n_test_pos"] == 5
    d = saved["by_detector"]["det"]
    assert d["fpr_bound_conformal"] <= 0.10 + 1e-12
    assert d["conformal_threshold"] >= d["empirical_threshold"]
    assert d["test_tpr_at_conformal"] == 1.0          # 攻击分 10，干净分 0，必分开
    assert d["test_fpr_achieved"] == 0.0
    assert res["by_detector"]["det"]["n_pos"] == 5


def test_run_conformal_missing_npz_explains_how_to_fix(tmp_path: Path):
    cfg = _cfg(tmp_path, "processed_tmp")
    (Path(cfg.data.processed_dir)).mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="请先运行 detect"):
        run_conformal(cfg, out_path=tmp_path / "x.json")
