"""强攻击（A3）的纯逻辑测试：不下载模型、不碰数据。

覆盖的是**不变量**，因为它们才是会悄悄退化的东西：
- 零宽/全角/同形变换必须「可见文本不变 / 长度关系正确」，否则测出来的
  杀伤不是攻击造成的，而是我们改坏了文本；
- 变换必须确定性可复现（同一 seed 同一输出）；
- 零宽攻击必须真的破坏字符 n-gram 的相似度（否则等于没攻击）。
"""

from __future__ import annotations

import re

import pytest

from poisonguard.strongattack import (ZERO_WIDTH, _bt_cache_path, _insert_zero_width,
                                      obfuscate, obfuscate_samples, strong_transform)

SENT = "某地警方通报：今年高考报名人数首次突破1300万人"


def _strip_zw(s: str) -> str:
    return "".join(ch for ch in s if ch not in ZERO_WIDTH)


# ---------------------------------------------------------------- 零宽
def test_zero_width_keeps_visible_text_and_adds_invisibles():
    out = obfuscate(SENT, "zero_width", seed=1)
    assert out != SENT
    assert _strip_zw(out) == SENT, "零宽只能插入不可见字符，不能改动可见文本"
    assert sum(ch in ZERO_WIDTH for ch in out) >= 4


def test_zero_width_deterministic_for_same_seed():
    a = obfuscate(SENT, "zero_width", seed=7)
    b = obfuscate(SENT, "zero_width", seed=7)
    assert a == b


def test_zero_width_breaks_char_ngram_similarity():
    """零宽攻击的机制：char n-gram 余弦显著下降（否则这个攻击是无效的）。"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    atk = obfuscate(SENT, "zero_width", seed=2)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    m = vec.fit_transform([SENT, atk])
    sim = float((m[0] @ m[1].T).toarray()[0, 0])
    assert sim < 0.75, f"零宽攻击后与原文的字符 n-gram 余弦应显著下降，实测 {sim:.3f}"


def test_insert_zero_width_every_param():
    out = _insert_zero_width("abcdef", __import__("random").Random(0), every=2)
    assert _strip_zw(out) == "abcdef"
    assert sum(ch in ZERO_WIDTH for ch in out) == 3  # 6/2


# ---------------------------------------------------------------- 同形 / 全角
def test_homoglyph_keeps_length_and_changes_chars():
    out = obfuscate("国家统计局2024年发布数据abc", "homoglyph", seed=3, p=1.0)
    assert len(out) == len("国家统计局2024年发布数据abc")
    assert out != "国家统计局2024年发布数据abc"


def test_fullwidth_ascii_expands_to_fullwidth_forms():
    out = obfuscate("2024 GDP +3%", "fullwidth", seed=0)
    assert "２" in out and "０" in out, "ASCII 数字应被换成全角"
    assert len(out) == len("2024 GDP +3%"), "全角是 1:1 替换，长度不变"


def test_mix_combines_zero_width_and_other():
    out = obfuscate(SENT, "mix", seed=4)
    assert any(ch in ZERO_WIDTH for ch in out)


# ---------------------------------------------------------------- 契约
def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="未知"):
        obfuscate(SENT, "nope", seed=0)
    with pytest.raises(ValueError):
        strong_transform([SENT], "nope")


def test_obfuscate_samples_pairs_are_aligned():
    class S:
        def __init__(self, text, label=1, family="edit"):
            self.text, self.label, self.family = text, label, family

    samples = [S("句子一"), S("句子二", label=0), S("句子三")]
    pairs = obfuscate_samples(samples, "zero_width", seed=0)
    # 只处理 label==1 → 两条
    assert len(pairs) == 2
    origs = [p[0] for p in pairs]
    assert origs == ["句子一", "句子三"]
    assert all(_strip_zw(a) == o for o, a in pairs)


def test_bt_cache_path_uses_cache_dir():
    class Cfg:
        outputs_dir = "/tmp/x_outputs"

    p = _bt_cache_path(Cfg(), "v3")
    assert p.name == "backtrans_v3.jsonl"
    assert "x_outputs" in str(p)


def test_strong_transform_routes_obfuscation():
    out, meta = strong_transform([SENT], "zero_width", seed=1)
    assert _strip_zw(out[0]) == SENT
    assert meta["kind"] == "char_obfuscation"


def test_zero_width_regex_detects_any_invisible():
    """审计视角：任何 U+200B..U+200D / FEFF 都应被一条正则抓到。"""
    pat = re.compile("[\u200b\u200c\u200d\ufeff]")
    assert pat.search(obfuscate(SENT, "zero_width", seed=9))
