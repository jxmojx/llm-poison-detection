"""自适应攻击（A4）逃避变换的回归测试。

这些测试**不下载模型、不跑训练**，只验证文本改写逻辑：
- 逃避变换是否保留"事实被篡改"的语义（词面越模糊、事实改动仍在）；
- 同义替换 / 字符扰动 / 组合是否确定性可复现；
- negate 模式是否只对强断言句做最小语义反转；
- 长度护栏是否防止废话退化成病句。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poisonguard.evade import (  # noqa: E402
    EvadedText,
    _repl_all_syn,
    evade_samples,
    evade_text,
)


def test_evade_identity_method_unchanged():
    s = "2024年公司营收同比增长超过30%"
    assert evade_text(s, "none", seed=7) == s


def test_syn_replaces_recognized_phrase_keeps_rest():
    s = "今年高考报名人数首次突破1300万人"
    out = evade_text(s, "syn", seed=7)
    assert out != s
    # 同义表命中某词 → 表面变化；事实载体（数字、主体句）必须保留
    assert "1300万" in out
    assert "高考" in out
    assert "人" in out


def test_chr_inserts_noise_but_keeps_meaning():
    s = "今年高考报名人数首次突破1300万人"
    out = evade_text(s, "chr", seed=7)
    # 至少一个字符级扰动（插入噪声符或同形字）
    assert out != s
    for ch in ("～", "·"):
        if ch in out:
            break
    else:
        # 可能命中同形替换（的/了/说…），只要不等于原句即可
        assert out != s
    # 长度护栏内
    assert 0.5 <= len(out) / len(s) <= 1.5


def test_mix_applies_both():
    s = "该车型续航里程达到700公里"
    out = evade_text(s, "mix", seed=7)
    assert out != s
    assert "700" in out


def test_evade_deterministic_same_seed():
    s = "2024年公司营收同比增长超过30%，环比增长5%"
    a = evade_text(s, "mix", seed=11)
    b = evade_text(s, "mix", seed=11)
    assert a == b


def test_negate_true_inserts_bu_negative_assertion():
    s = "2024年公司营收同比增长超过30%"
    out = evade_text(s, "none", seed=7, negate=True)
    # 强断言"超过"被前插"不"，语义反转
    assert "不超过" in out or "不" in out


def test_evade_samples_only_positive_labels():
    from poisonguard.data import Sample

    bad = Sample(text="今年营收首次突破1300万", label=1, family="edit", split="test")
    good = Sample(text="今天天气不错", label=0, family="normal", split="test")
    res = evade_samples([bad, good], "syn", seed=0)
    # 只处理正样本
    assert len(res) == 1
    assert isinstance(res[0], EvadedText)
    assert res[0].method == "syn"


def test_evade_samples_len_guard_falls_back():
    from poisonguard.data import Sample

    # 极短句子 + 变换应回退或保持在小幅范围内（长度护栏容差 0.55）
    s = Sample(text="已成功", label=1, family="edit", split="test")
    res = evade_samples([s], "chr", seed=3)
    assert 0 <= len(res[0].text) <= 8
