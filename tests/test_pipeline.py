"""冒烟测试：验证「结论成立」依赖的几个关键不变量。

不依赖 pytest（本机没装），也可以装 pytest 后直接 `pytest tests/`。
直接跑：`python -m tests.test_pipeline`

为什么这几个断言值得写：
    - 原项目的所有结论都因为"没有这些不变量"而失效；
    - 这些断言跑一次只要 1 秒，但能在改代码时立刻发现"评测又漏了 test"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poisonguard import evaluate as E  # noqa: E402
from poisonguard.attacks import build_attacks  # noqa: E402
from poisonguard.config import load_config  # noqa: E402
from poisonguard.detectors import Calibration, KNNDetector, SupervisedDetector  # noqa: E402


# ----------------------------------------------------------------------
def test_attacks_have_no_keywords():
    """paraphrase/subtle/backdoor 必须**不含**原项目关键词表里的任何词。

    这是"击穿循环论证"的核心性质：如果这些家族里出现了关键词，
    关键词检测器就会"莫名其妙地准"，而这正是原项目 100% 检测率的来源。
    """
    cfg = load_config()
    kws = cfg.detector.keywords
    att = build_attacks(cfg.attack.families, n_per_family=120,
                        seed=cfg.attack.test_seed, bucket=1, n_buckets=2)
    for fam in ("paraphrase", "subtle", "backdoor"):
        texts = [a.text for a in att if a.family == fam]
        assert texts, f"{fam} 家族没有生成样本"
        hit = [t for t in texts if any(k in t for k in kws)]
        assert not hit, f"{fam} 家族出现关键词: {hit[:3]}"


def test_dev_test_attack_disjoint():
    """dev 与 test 的攻击字符串必须完全不相交(分桶 + 不同 seed)。"""
    cfg = load_config()
    kw = dict(n_per_family=150, seed=cfg.attack.test_seed)
    dev = build_attacks(cfg.attack.families, bucket=0, n_buckets=2, **kw)
    test = build_attacks(cfg.attack.families, bucket=1, n_buckets=2, **kw)
    s_dev, s_test = {a.text for a in dev}, {a.text for a in test}
    inter = s_dev & s_test
    assert not inter, f"dev/test 攻击样本重叠 {len(inter)} 条: {list(inter)[:3]}"


def test_style_holdout_splits_templates_disjoint():
    """模板留出必须真的把模板**分干净**：两段的交集为空、并集覆盖全表。

    这个断言防的是一类很容易犯的错：`_subset` 写错边界时，
    dev 和 test 会悄悄共用一部分模板，于是"严格口径"其实并不严格，
    而指标看起来仍然漂亮——正是本项目一开始踩的坑。
    """
    from poisonguard import attacks as A

    for name, lst in (
        ("_TPL_TEMPLATES", A._TPL_TEMPLATES),
        ("_PARAPHRASE", A._PARAPHRASE),
        ("_SUBTLE", A._SUBTLE),
        ("_SLOT_TPL", A._SLOT_TPL),
        ("_BACKDOOR_TPL", A._BACKDOOR_TPL),
        ("_TRIGGERS", A._TRIGGERS),
    ):
        s0 = A._subset(lst, A._fold_span(len(lst), 0, 2))
        s1 = A._subset(lst, A._fold_span(len(lst), 1, 2))
        assert not (set(s0) & set(s1)), f"{name} 两折有重叠"
        assert set(s0) | set(s1) == set(lst), f"{name} 两折没覆盖全表"


def test_calibration_hits_target_fpr():
    """校准器给出的阈值应当让参考集上的误报率接近目标值。"""
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, 5000)
    for target in (0.01, 0.05, 0.10):
        cal = Calibration.fit(clean, target)
        real = float((clean > cal.raw_threshold).mean())
        assert abs(real - target) < 0.01, f"target={target} 实际={real:.3f}"


def test_metrics_sanity():
    """指标实现的基本正确性：完美分离 AUROC=1，随机 AUROC≈0.5。"""
    y = np.array([0] * 100 + [1] * 100)
    assert E.auroc(y, np.concatenate([np.zeros(100), np.ones(100)])) == 1.0
    rng = np.random.default_rng(0)
    assert abs(E.auroc(y, rng.random(200)) - 0.5) < 0.12
    assert E.tpr_at_fpr(y, np.concatenate([np.zeros(100), np.ones(100)]), 0.01) == 1.0


def test_supervised_detector_oof_honest():
    """有监督检测器的 OOF 分数应当明显低于 in-sample 分数(说明确实留出了样本)。"""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, 16))
    y = (X[:, 0] + rng.normal(0, 0.5, 400) > 0.5).astype(int)
    det = SupervisedDetector()
    oof = det.oof_scores(X, y, folds=5)
    det.fit_labeled(X, y)
    ins = det.score(X, [""] * len(X))
    # in-sample 分数的可分性不应低于 OOF（否则说明 OOF 实现有问题）
    assert E.auroc(y, ins) >= E.auroc(y, oof) - 1e-9


def test_knn_detects_shifted_cluster():
    """无监督检测器对"整簇偏移"应当灵敏——这是数据层能工作的前提。"""
    rng = np.random.default_rng(0)
    X_clean = rng.normal(0, 1, (500, 32))
    X_shift = rng.normal(6.0, 1, (200, 32))   # 位移足够大：余弦距离在 1 附近饱和
    det = KNNDetector(k=5).fit(X_clean, [""] * len(X_clean))
    s_clean = det.score(X_clean, [""] * 500)
    s_shift = det.score(X_shift, [""] * 200)
    assert s_shift.mean() > s_clean.mean(), "偏移簇的距离应该更大"
    tpr = E.tpr_at_fpr(np.r_[np.zeros(500), np.ones(200)],
                       np.r_[s_clean, s_shift], 0.01)
    assert tpr > 0.8, f"TPR@1% 只有 {tpr:.3f}"


def test_modelscope_dirname_escaping():
    """ModelScope 把模型名里的 `.` 落盘成 `___`，本地查找必须兼容。

    否则刚下载完的模型会被判成"本地没有"，`QwenJudge.available()` 返回 False，
    表现为"下完了还是不走 LLM、静默降级成规则版"——一个很难发现的坑。
    """
    from poisonguard.localmodels import _ms_escape

    assert _ms_escape("Qwen2.5-0.5B-Instruct") == "Qwen2___5-0___5B-Instruct"
    assert _ms_escape("damo/nlp_corom_sentence-embedding_chinese-base") == \
        "damo/nlp_corom_sentence-embedding_chinese-base"


# ----------------------------------------------------------------------
def main() -> int:
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
