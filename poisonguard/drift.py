"""域漂移体检（P1 换域支撑）：开箱即用地判断「这份输入离训练域有多远」。

把一个域守卫生效的条件拆成三根可测的轴：
  1. **长度漂移**  —— 输入平均长度 vs 训练侧 dev 干净集（scan.py 已有 hint，这里独立成报告）。
  2. **向量漂移**  —— 输入样本的 CoROM 编码质心，到 dev 干净集质心/最近邻的距离。
     域换得越狠（如 tnews→iflytek 长文本），向量距离拉得越开。
  3. **字符分布漂移** —— 输入与 dev 干净集的 unigram KL 散度 + 词表覆盖率
     （OOV 占比越高，越说明"这不是训练域的语言"）。

产出：一页「漂移体检表」，给出 Green/Amber/Red 三档 + 可解释的证据数字，
让操作者决定"这份输入值不值得用已训守卫打分"。纯逻辑部分（切分/KL/质心/
分档）完全不依赖编码器，测试用默认字符统计路径即可；需要向量轴时才注入编码器。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# 纯逻辑核心（CI 可测，不碰模型）
# ----------------------------------------------------------------------
def char_unigram(text: str) -> dict[str, int]:
    """字符 unigram 计数（中文字符+标点+数字字母，按字符）。"""
    out: dict[str, int] = {}
    for ch in (text or ""):
        out[ch] = out.get(ch, 0) + 1
    return out


def _freq(counter: dict[str, int]) -> dict[str, float]:
    tot = sum(counter.values()) or 1
    return {k: v / tot for k, v in counter.items()}


def kl_divergence(p: dict[str, float], q: dict[str, float],
                  eps: float = 1e-9) -> float:
    """KL(p||q)：对 p 中出现而 q 中没有的字符给 log((p+eps)/eps) 惩罚。"""
    total = 0.0
    for k, pk in p.items():
        qk = q.get(k, 0.0)
        total += pk * math.log((pk + eps) / (qk + eps))
    return float(total)


def vocab_coverage(p: dict[str, float], vocab: set[str]) -> float:
    """输入分布 p 里落在参考词表 vocab 内的质量占比（0..1，越高越像训练域）。"""
    covered = 0.0
    for k, pk in p.items():
        if k in vocab:
            covered += pk
    return float(covered)


def mean_std(arr: list[float]) -> tuple[float, float]:
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def length_stats(texts: list[str]) -> dict:
    lens = [len(t) for t in texts]
    m, sd = mean_std(lens)
    return {"n": len(lens), "mean_len": m, "std_len": sd,
            "min_len": float(min(lens)) if lens else None,
            "max_len": float(max(lens)) if lens else None}


def char_profile(texts: list[str], limit: int | None = 2000) -> dict:
    """合并前 limit 段文本的字符分布。"""
    texts = (texts or [])[:limit] if limit else (texts or [])
    counter: dict[str, int] = {}
    for t in texts:
        for ch, n in char_unigram(t).items():
            counter[ch] = counter.get(ch, 0) + n
    return {"counter": counter, "freq": _freq(counter),
            "vocab": set(counter.keys())}


def drift_report(texts: list[str], ref_texts: list[str],
                 ref_lens: list[float] | None = None,
                 embs_in: np.ndarray | None = None,
                 embs_ref: np.ndarray | None = None,
                 len_thr: float = 40.0, kl_thr: float = 0.45,
                 cov_thr: float = 0.85, vec_thr: float = 0.35) -> dict:
    """三轴漂移体检（输入 vs 参考 dev 干净集）。

    返回 dict：axes(normalized: {length, vector, vocab} 在 [0,1] 的差距分，越大越漂，
    gap_ok/gap_warn/gap_bad），summary（Green/Amber/Red + 一句话 + 各轴原始值）。

    阈值校准（real_v3 实测）：同域 tnews 头 500 条 KL≈0.32/cov 0.984；
    跨域 afqmc 短文本 KL≈2.68/cov 0.954；默认 kl_thr=0.45 使同域 Green、
    跨域 Amber/Red。长度轴 len_thr=40 与 scan.py 的漂移告警保持一致。
    """
    if not texts:
        raise ValueError("输入文本为空")
    if not ref_texts:
        raise ValueError("参考语料为空（需先跑 data 生成训练域 dev 干净集）")

    # 轴 1：长度
    in_l = [len(t) for t in texts]
    in_m = float(np.mean(in_l))
    ref_m = float(np.mean(ref_lens)) if ref_lens is not None else float(
        np.mean([len(t) for t in ref_texts]))
    len_gap = abs(in_m - ref_m)
    len_norm = min(1.0, len_gap / len_thr)

    # 轴 2：向量（可选注入编码后矩阵）
    vec_gap = 0.0
    vec_norm = 0.0
    vec_detail: dict = {}
    if embs_in is not None and embs_ref is not None and len(embs_ref):
        c_in = embs_in.mean(axis=0)
        c_ref = embs_ref.mean(axis=0)
        dist = float(np.linalg.norm(c_in - c_ref))
        # 参考自身离散度做归一（同域样本间距离）
        if len(embs_ref) >= 2:
            ref_spread = float(
                np.median(np.linalg.norm(embs_ref - embs_ref.mean(axis=0), axis=1)))
        else:
            ref_spread = 1.0
        vec_gap = float(dist)
        vec_norm = min(1.0, dist / max(ref_spread * 3.0, 1e-6))
        vec_detail = {"centroid_dist": round(dist, 4),
                      "ref_spread": round(ref_spread, 4),
                      "norm": round(vec_norm, 4)}

    # 轴 3：字符分布（KL + 覆盖率）
    p_in = char_profile(texts)
    p_ref = char_profile(ref_texts)
    kl = kl_divergence(p_in["freq"], p_ref["freq"])
    cov = vocab_coverage(p_in["freq"], p_ref["vocab"])
    kl_norm = min(1.0, kl / kl_thr)
    cov_norm = min(1.0, max(0.0, (cov_thr - cov) / cov_thr))

    # 汇总分档
    axes = {
        "length": {"gap": round(len_gap, 2), "ref_mean_len": round(ref_m, 2),
                   "in_mean_len": round(in_m, 2), "norm": round(len_norm, 4),
                   "ok": len_gap <= len_thr},
        "vector": {"gap": round(vec_gap, 4), "norm": round(vec_norm, 4),
                   "ok": vec_gap <= vec_thr, **vec_detail},
        "vocab": {"kl": round(kl, 4), "coverage": round(cov, 4),
                  "norm": round(kl_norm * 0.5 + cov_norm * 0.5, 4),
                  "ok": kl <= kl_thr and cov >= cov_thr},
    }
    # 综合：三个轴都 ok → Green；任一 warn（norm>0.5）→ Amber；任一 bad（norm>0.8）→ Red
    norm = [
        axes["length"]["norm"], axes["vector"]["norm"],
        axes["vocab"]["norm"],
    ]
    worst = max(norm)
    if worst >= 0.8:
        status, tone = "Red", "漂移大：输入与训练域差异显著，守卫分数可信度低，建议换域重训"
    elif worst >= 0.5:
        status, tone = "Amber", "漂移中等：输入部分偏离训练域，分数仅供参考"
    else:
        status, tone = "Green", "漂移小：输入与训练域接近，守卫分数可信"

    dead_axes = [k for k in ("length", "vector", "vocab")
                 if not axes[k]["ok"]]
    summary = {
        "status": status, "tone": tone,
        "worst_norm": round(worst, 4),
        "len_drift": round(len_gap, 2),
        "vec_drift": round(vec_gap, 4),
        "kl_drift": round(kl, 4),
        "vocab_cov": round(cov, 4),
        "dead_axes": dead_axes,
    }
    return {"axes": axes, "summary": summary}


def format_report(report: dict) -> str:
    s = report["summary"]
    ax = report["axes"]
    L = [
        f"域漂移体检：{s['status']}",
        f"  说明：{s['tone']}",
        f"  长度轴  输入均长 {ax['length']['in_mean_len']:.1f} 字 vs 训练域 {ax['length']['ref_mean_len']:.1f} 字"
        f"  → 差距 {ax['length']['gap']:.1f} 字（{'OK' if ax['length']['ok'] else '漂移'}）",
        f"  字符轴  KL={ax['vocab']['kl']:.4f}  词表覆盖率={ax['vocab']['coverage']:.3f}"
        f"  （{'OK' if ax['vocab']['ok'] else '漂移'}）",
    ]
    if ax["vector"]["gap"]:
        L.append(f"  向量轴  质心距={ax['vector']['gap']:.4f}"
                 f"  （{'OK' if ax['vector']['ok'] else '漂移'}）")
    if s["dead_axes"]:
        L.append(f"  ⚠️ 告警轴：{', '.join(s['dead_axes'])}")
    return "\n".join(L)


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------
def run_drift_check(cfg, input_path: str | Path, out_path: str | Path | None = None,
                    column: str = "text", fmt: str = "auto", limit: int = 500,
                    with_encoder: bool = True, seed: int = 42) -> dict:
    """对输入文件 vs 训练域 dev 干净集做三轴漂移体检。"""
    from .data import load_splits
    from .scan import load_rows

    t0 = time.perf_counter()
    rows = load_rows(Path(input_path), fmt=fmt, column=column, limit=limit)
    texts = [r["text"] for r in rows]
    dev = load_splits(cfg.data.processed_dir)["dev"]
    ref = [s.text for s in dev if s.label == 0 and s.family == "normal"]

    embs_in = embs_ref = None
    if with_encoder:
        try:
            from .encoders import get_encoder

            enc = get_encoder(cfg)
            n_in = min(len(texts), 500)
            embs_in = enc.encode(texts[:n_in])
            embs_ref = enc.encode(ref[:1000])
        except Exception as e:
            print(f"[drift] 编码器不可用，跳过向量轴（{type(e).__name__}: {e}）")
            with_encoder = False

    report = drift_report(texts, ref, embs_in=embs_in, embs_ref=embs_ref)
    report["meta"] = {
        "input": str(input_path), "n_input": len(texts),
        "n_ref": len(ref), "with_encoder": bool(with_encoder),
        "seconds": round(time.perf_counter() - t0, 2),
        "domain": Path(cfg.data.processed_dir).name,
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"[drift] 已保存 {out_path}")
    print(format_report(report))
    return report
