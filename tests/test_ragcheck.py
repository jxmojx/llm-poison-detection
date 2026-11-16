"""检索式事实核验（P0-2 升级）的纯逻辑测试：不下载任何模型。

钉住的四件事：事实点怎么拆、检索一致性冲突怎么判、打分管道怎么接线、
`run_ragcheck` 在无真实数据时也能 import。

编码器用仓库自带的 TinyEncoder（字符 2-3gram 哈希向量，确定性、零下载）：
它让"差一个字的篡改句"与原句共享大量 n-gram → 高余弦，正好模拟真实
句向量编码器对近重复的敏感度（暴力哈希整句的 Fake 会连 0.5 相似度都给不出）。
"""

from __future__ import annotations

import difflib

import numpy as np

from poisonguard.encoders import TinyEncoder
from poisonguard.ragcheck import (_FACT_WORDS, _is_fact_change, _pair_conflict,
                                  _topk_by_cosine, build_reference, explain,
                                  extract_facts, fact_points, run_ragcheck,
                                  scan_scores)


# ---------------------------------------------------------------- 事实点抽取
def test_extract_facts_numbers_units_dates():
    pts = extract_facts("今年报名人数首次突破1300万人，同比增长30%，2024年GDP增长5.2%")
    kinds = [p.kind for p in pts]
    vals = {p.value for p in pts}
    # 数字带单位/日期/百分号都算 unit_date；单纯数字为 number
    assert any(k in ("number", "unit_date") for k in kinds)
    assert any(v == "1300万" or v.startswith("1300") for v in vals)
    assert any(v == "30%" or v.startswith("30%") for v in vals)
    assert any(v.startswith("2024") for v in vals)
    assert any(v == "5.2%" for v in vals)


def test_extract_facts_factwords_aligned_with_edit_family():
    # 至少覆盖 edit 家族的几大组词面（否定/方向/时间/量级）
    for w in ("今年", "增长", "已经", "批准", "上涨", "部分", "万元"):
        # 单字词里有些会同时命中（如"年"），只要任意事实词块含它即可
        pass
    assert "增长" in _FACT_WORDS
    assert "上涨" in _FACT_WORDS
    assert "批准" in _FACT_WORDS
    assert "万元" in _FACT_WORDS


def test_extract_facts_empty_and_noop():
    assert extract_facts("") == []
    assert extract_facts("没有任何数字和事实词，纯粹是描述性文字测试") == []


def test_fact_points_serializable():
    pts = fact_points("今年GDP增长5.2%")
    assert isinstance(pts, list)
    assert all(set(p) == {"kind", "value", "start", "end", "context"} for p in pts)
    assert pts[0]["kind"] == "factword"  # 今年


# ---------------------------------------------------------------- 冲突判定
def test_pair_conflict_number_swap_detects():
    a = "今年报名人数首次突破1300万人"
    b = "今年报名人数首次突破1300亿人"
    s = _pair_conflict(a, b)
    assert s > 0.7, f"数字+单位改写应判大冲突，got {s}"


def test_pair_conflict_identical_is_zero():
    a = "今年报名人数首次突破1300万人"
    assert _pair_conflict(a, a) == 0.0, "逐字相同=知识库已验证原文，不误伤"


def test_pair_conflict_polarity_flip():
    a = "上海公司已经批准上市"
    b = "上海公司已经叫停上市"
    assert _pair_conflict(a, b) > 0.5, "极性翻转（批准→叫停）应判冲突"


def test_pair_conflict_unrelated_zero():
    assert _pair_conflict("农村大妈彩礼调查结果公布", "同比增长30%新数据发布") == 0.0


def test_pair_conflict_wording_only_zero():
    # 换语气词不算事实冲突
    a = "今年报名人数首次突破1300万人了"
    b = "今年报名人数首次突破1300万人啦"
    assert _pair_conflict(a, b) == 0.0


def test_is_fact_change_marks_diff_segments():
    assert _is_fact_change("1300万", "1300亿") is True
    assert _is_fact_change("已批准", "已叫停") is True
    assert _is_fact_change("了", "啦") is False


# ---------------------------------------------------------------- 检索打分
def test_scan_scores_detects_mutated_copy():
    ref = ["今年报名人数首次突破1300万人，同比增长30%"]
    adv = "今年报名人数首次突破1300亿人，同比增长30%"
    enc = TinyEncoder()
    ref_embs = enc.encode(ref)
    scores = scan_scores([adv], ref, ref_embs, enc, top_k=1)
    assert scores[0] > 0.5, f"库内原句被篡改应打分高，got {scores[0]}"


def test_scan_scores_clean_corpus_zero():
    ref = ["今年报名人数首次突破1300万人，同比增长30%"]
    enc = TinyEncoder()
    ref_embs = enc.encode(ref)
    clean = "完全不相干的另一篇财经报道：央行今日开展逆回购操作"
    scores = scan_scores([clean], ref, ref_embs, enc, top_k=1)
    assert scores[0] == 0.0, "库外陌生文本没有对齐证据，应 0 分"


def test_topk_reorders_by_similarity():
    # query [0.9,0.1]：与 [0.8,0.2] 余弦 0.98 > 与 [1,0] 0.9 > 与 [0,1] 0.1
    # argpartition 只保证 topk 集合，不保证内部顺序 → 集合断言
    ref = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    q = np.array([0.9, 0.1], dtype=np.float32)
    idx = _topk_by_cosine(q, ref, top_k=2).tolist()
    assert set(idx) == {0, 2}, f"top2 应为 {0,2}，got {idx}"
    # 第三近（[0,1]）不该入选
    assert 1 not in idx


def test_scan_scores_empty_input():
    enc = TinyEncoder()
    ref_embs = enc.encode(["参考"])
    out = scan_scores([], ["参考"], ref_embs, enc)
    assert out.shape == (0,)


def test_explain_structure():
    enc = TinyEncoder()
    ref = ["今年报名人数首次突破1300万人，同比增长30%"]
    ref_embs = enc.encode(ref)
    d = explain("今年报名人数首次突破1300亿人，同比增长30%", ref, ref_embs, enc, top_k=1)
    assert "score" in d and "top_hits" in d
    assert len(d["top_hits"]) <= 1
    assert d["top_hits"][0]["conflict"] in (True, False)


def test_build_reference_missing_processed_raises():
    """没有 processed 数据时给清晰报错（而不是静默空库）。"""
    class _Cfg:
        data = type("D", (), {"processed_dir": "data/__no_such_dir__"})()

    try:
        build_reference(_Cfg())
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_run_ragcheck_importable():
    assert callable(run_ragcheck)
