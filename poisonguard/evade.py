"""自适应攻击：攻击者知道检测器用的是"字符 n-gram + 信息密度"特征后做的逃避变换。

背景
----
主实验里 edit 家族（最小事实篡改）在 tnews 上被 sup_tfidf（char 2-3gram + 信息密度）抓 4%、
iflytek 抓 1.4%。一个立刻会被问的问题是："这个 4% 是检测器的能力上限，还是攻击太弱？
攻击者知道你的特征以后，把词面痕迹抹掉，还能抓到几成？"

本模块回答这个问题，做法是**逃避变换**：对已经生成的 edit 正样本，攻击者再配 3 层"擦掉词面痕迹"
的改写（保留事实被篡改的语义，只抹掉让 n-gram 分类器兴奋的字符痕迹）：

- syn 同义替换：常见量词/程度词/连接词换说法（"同比增长"→"较上年同期上升"）
- chr 字符扰动：局部插入空格 / 同形替换（"加强"→"加强"）/"了" 增删
- mix 组合：先同义再字符扰动（真实攻击者会叠加）

关键设计
--------
1) **negate=False 只擦词面、不改变语义**，用来回答"词面痕迹值多少钱"；
   收到 `negate=True` 时允许做最小语义反转（加减一个"不"字），单独报告，用于回答
   "攻击者发真实假新闻时，连语义都可以改"。两者不混。
2) 每次变换用**独立随机种子**，同一句不重复产出同一结果；全程确定性可复现。
3) 本模块**不知道也不导入**任何检测器实现，只做文本改写——"自适应"体现在
   "知道要抹 n-gram 痕迹"这一条假设上，不泄漏其它信息。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------
# 同义/近义替换表（按语义近似度分组，只能组内替换）
# ----------------------------------------------------------------------
_SYN_GROUPS: list[tuple[str, ...]] = [
    ("同比增长", "比去年同期增长", "较上年同期上升", "较上期增长", "同比上升"),
    ("环比", "与上期相比", "较前期", "较上个月"),
    ("左右", "上下", "前后", "附近"),
    ("超过", "突破", "高于", "超出", "胜于"),
    ("不到", "不足", "低于", "未及", "不满"),
    ("大约", "约莫", "大概", "约有", "估计"),
    ("其中", "其中包含", "这里面", "在这之中"),
    ("目前", "眼下", "现阶段", "现在", "当前"),
    ("已经", "已然", "早己", "业已"),
    ("首次", "头一回", "第一次", "首度"),
    ("再次", "又一次", "再度", "重新"),
    ("称为", "被称作", "被叫做", "叫作", "名为"),
    ("持续", "不断", "一直", "接连"),
    ("迅速", "快速", "飞快", "很快"),
]

_SYN_FLAT: list[tuple[str, str]] = []
for _g in _SYN_GROUPS:
    for _i, _w in enumerate(_g):
        for _j, _w2 in enumerate(_g):
            if _i != _j:
                _SYN_FLAT.append((_w, _w2))

# 同形/同音干扰字符（对中文 n-gram 的本质破坏：局部加空格、增删"的/了"）
_VARIANT_PAIRS = [
    ("的", "の"),
    ("了", "了"),
    ("我", "莪"),
    ("他", "怹"),
    ("说", "説"),
]
_CHAR_NOISE = ("～", "·", " ", "")


# ----------------------------------------------------------------------
@dataclass
class EvadedText:
    text: str
    method: str   # none / syn / chr / mix


def _repl_all_syn(text: str, rng: random.Random, p: float = 0.6) -> str:
    """命中同义表则随机换一个（有概率替换，模拟攻击者不一定每处都换）。"""
    out = text
    for w, w2 in _SYN_FLAT:
        if w in out and rng.random() < p:
            out = out.replace(w, w2)
    return out


def _perturb_chars(text: str, rng: random.Random, p: float = 0.25) -> str:
    """字符级扰动：插一个可打印噪声符 / 增删最常用的"的/了" / 同形替换。"""
    chars = list(text)
    step = max(1, int(p * len(chars)))
    for i in range(step, len(chars), step):
        c = chars[i]
        if rng.random() < 0.5:
            # 词内同形替换
            for a, b in _VARIANT_PAIRS:
                if c == a:
                    chars[i] = b
                    break
        elif rng.random() < 0.5:
            # 词前插一个噪声符（对 char-wb 恰好打断该字符)
            chars.insert(i, rng.choice(_CHAR_NOISE))
    return "".join(chars)


def evade_text(text: str, method: str, seed: int = 0, negate: bool = False) -> str:
    """对一句文本做逃避变换（独立 seed，确定性可复现）。

    method: none=原样 / syn=同义替换 / chr=字符扰动 / mix=syn+chr
    negate: False=只擦词面痕迹(保留语义)；True=允许对内容做最小语义反转(加/减"不")，
            单列到 results, 与词面方法分开。
    """
    rng = random.Random(seed)
    out = text

    if method == "syn":
        out = _repl_all_syn(out, rng)
    elif method == "chr":
        out = _perturb_chars(out, rng)
    elif method == "mix":
        out = _repl_all_syn(out, rng)
        out = _perturb_chars(out, rng)
    elif method != "none":
        raise ValueError(f"未知逃避方法: {method}")

    if negate:
        # 最小语义反转：在主干前/后加"不"，或删掉一个"不"。
        # 只对"看起来像断言/结论"的句子做，避免把普通陈述改成病句。
        runits = re.findall(r"最|更|超|首次|唯一|一定|必须|该|本|目前|将|会|已", out)
        if runits:
            # 取句中第一个强断言词，前后插入"不"
            m = re.search(r"(最|更|超|首次|唯一|一定|必须|该|本|目前|将|会|已)", out)
            if m:
                pos = m.start()
                out = out[:pos] + "不" + out[pos:]
    return out


def evade_samples(samples, method: str, seed: int = 42,
                  negate: bool = False) -> list[EvadedText]:
    """对一组正样本文本做逃避，返回 (改写文本, 方法) 列表。

    只处理 label==1；对每句用不同子种子（确定性），便于复现。
    """
    rng = random.Random(seed)
    out: list[EvadedText] = []
    for i, s in enumerate(samples):
        text = s.text if hasattr(s, "text") else s
        if hasattr(s, "label") and s.label != 1:
            continue  # 只测正样本
        t = evade_text(text, method, seed=rng.randint(0, 2**31 - 1), negate=negate)
        left = 1.0 - len(t) / max(1, len(text))
        if abs(left) > 0.55:
            t = text  # 变换太离谱（长度变化过大），回退原文
        out.append(EvadedText(t, method))
    return out
