"""检索式事实核验（P1/P0 升级版：数字级 P0-2 已被证否，AUROC 0.353 < 随机）。

为什么替换数字级方案：
    `factcheck.py` 用 Qwen 的**无条件先验**问「这个数字在句子里合不合理」——
    它不知道 1300 万是不是真人数，只知道「首次突破 1300 亿」不像正常新闻。
    实测（`outputs/metrics/factcheck_real_v3.json`）normal 均值分(5.34)反而比
    edit(4.81)更高、整体 AUROC 0.353 还不如随机——因为正常新闻里的具体数字
    对语言模型本来就是低概率的（冷门数字见得多才有高概率），先验把"罕见"当"可疑"。

本模块改成**对照语料/文档库检索核验**：不靠模型先验，靠"知识库里有没有
同事实的佐证/反证"。做法：
  1. 把被检文本拆成**事实点**（数字+单位 / 日期 / 事实性词面：方向·时间·极性·量级）
      ——与 `attacks.py` 的 edit 家族改法一一对应（那一族改的就是这些事实点）；
  2. 用现有 CoROM 编码器把被检文本与**可信参考语料**（训练干净集/文档库）编码余弦检索
      top-k 最近邻；
  3. 对每个近邻做**最小编辑对齐**（difflib.SequenceMatcher）：
      - 若两段是近重复（ratio≥min_ratio）且差异块里含事实点
        （数字/单位/事实词）→ 检索一致性冲突，可疑度 = ratio；
      - 若逐字相同（ratio==1.0，本身就是知识库已验证原文）→ 0 分（不误伤重复入库）；
      - 若差异块不含事实点（只是措辞不同）→ 0 分。

这是"有知识库 vs 无知识库"的对照：数字级在真空里猜，检索级在文档库上对账。
注意诚实口径：本评测的 edit 样本**就是从参考语料源头逐字篡改来的**，所以只要原句
还在知识库里，检索级就应无悬念抓出——这正是落地场景②（RAG 知识库语料审核，
向量库写入前用已在库的可信语料对账）的教科书用例；而对"库外改写/换措辞"的投毒，
检索级天然不覆盖（那就是向量库语义层 + 判官的事），本模块也要在报告里直说。
"""

from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# 事实点定义（与 attacks.py 的 edit 家族改法对齐）
# ----------------------------------------------------------------------
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# 数字后面紧跟的这些单位/量词，数字才有事实含义（型号/批次等被 attacks 排除）
_UNIT_CHARS = set(
    "%％万亿千百元美元岁年月日天小时分钟秒人名次倍公里千米米吨公斤度"
    "所家项起例届层楼辆台件份条篇部种"
)
# edit 家族的词面事实词表（否定/量级/方向/时间/单位五组，取改写的两端）
_FACT_WORD_LIST = (
    # negation
    "并未", "已经", "否认", "承认", "不会", "将会", "不再", "仍然", "尚未", "未能",
    # quantifier
    "部分", "全部", "多数", "所有", "一些", "大量", "大约", "超过", "左右", "以上",
    "可能", "必然", "疑似", "确系", "或将", "预计", "证实",
    # direction / polarity
    "上涨", "下跌", "增长", "下滑", "提高", "降低", "批准", "叫停", "盈利", "亏损",
    "达到", "不足", "不到", "同意", "反对", "允许", "禁止", "高于", "低于",
    # time
    "今年", "去年", "昨天", "明天", "本月", "上个月", "上午", "深夜",
    # unit
    "万元", "亿元", "公里", "米", "吨", "公斤", "周岁", "个月",
)
_FACT_WORDS = frozenset(_FACT_WORD_LIST)
# 简单中文机构/实体尾缀启发式（说明用；不作为主判据）
_ENTITY_RE = re.compile(
    r"[\u4e00-\u9fff]{2,8}(?:公司|集团|股份|银行|大学|医院|政府|部门|委员会|中心|总局|研究院|股份)"
)

CANNOT = "在参考语料中找不到可对齐的佐证/反证（指给措辞级语义层）"


@dataclass
class FactPoint:
    kind: str            # number / unit_date / factword / entity
    value: str
    start: int
    end: int
    context: str = ""    # 事实点前后 window 字窗口（人类可读 / 后续 masked-probe 用）

    def as_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value,
                "start": self.start, "end": self.end, "context": self.context}


# ----------------------------------------------------------------------
def _is_unit_char(ch: str) -> bool:
    return ch in _UNIT_CHARS


def extract_facts(text: str, window: int = 8) -> list[FactPoint]:
    """把被检文本拆成事实点：数字±单位、日期、事实词、实体启发式。"""
    text = text or ""
    pts: list[FactPoint] = []
    for m in _NUM_RE.finditer(text):
        value = m.group(0)
        start, end = m.start(), m.end()
        kind = "number"
        nxt = text[end:end + 1]
        if _is_unit_char(nxt):
            kind = "unit_date"
            end += 1
        # 年份+年月日是最常见的事实数字
        if end < len(text) and text[end:end + 1] in ("年", "月", "日", "号"):
            kind = "unit_date"
            end += 1
        ctx = text[max(0, start - window): min(len(text), end + window)]
        pts.append(FactPoint(kind, text[start:end], start, end, ctx))
    # 事实性词面：与 edit 家族改写的词重合 → 值得核验
    for w in _FACT_WORDS:
        pos = text.find(w)
        if pos >= 0:
            ctx = text[max(0, pos - window): min(len(text), pos + len(w) + window)]
            pts.append(FactPoint("factword", w, pos, pos + len(w), ctx))
    # 实体启发式（只做展示，score 不依赖它）
    for m in _ENTITY_RE.finditer(text):
        pts.append(FactPoint("entity", m.group(0), m.start(), m.end(),
                             text[max(0, m.start() - window): min(len(text), m.end() + window)]))
    # 相邻/重叠事实点合并（以最长的匹配词优先，避免单位断在中间）
    pts = sorted(pts, key=lambda x: (x.start, -x.end))
    merged: list[FactPoint] = []
    for p in pts:
        if merged and p.start < merged[-1].end:
            prev = merged[-1]
            if p.end <= prev.end:
                continue  # 完全被覆盖
            ctx = prev.context if len(prev.context) >= len(p.context) else p.context
            merged[-1] = FactPoint(prev.kind, prev.value, prev.start, p.end, ctx)
        else:
            merged.append(p)
    pts = merged
    return pts


def fact_points(text: str, window: int = 8) -> list[dict]:
    """便于 JSON 序列化的提取入口。"""
    return [p.as_dict() for p in extract_facts(text, window=window)]


# ----------------------------------------------------------------------
def _is_fact_change(qseg: str, rseg: str) -> bool:
    """一段差异（query 侧 qseg vs 参考侧 rseg）是否是"事实性篡改"。

    判定：任一侧含数字、或以单位/事实词收口。这正是 edit 家族的改法；
    纯措辞差异（换几个无关词）不算事实冲突。
    """
    if not qseg and not rseg:
        return False
    for s in (qseg, rseg):
        if _NUM_RE.search(s):
            return True
        if any(_is_unit_char(ch) for ch in s):
            return True
        for w in _FACT_WORDS:
            if w in s:
                return True
    return False


def _pair_conflict(query: str, ref: str, min_ratio: float = 0.75,
                   detail: bool = False) -> float | dict:
    """检索一致性：近重复且差异块含事实点 → 冲突分；否则 0。

    - detail=False（默认）：返回 0..1 的连续分（=近重复度 ratio），0 表示无证据。
    - detail=True：返回 {ratio, n_fact, conflict_facts, weight, score}，
      score 与 detail=False 一致（近重复度 × 事实点加权 1 + 0.25×(n_fact-1)，
      让多事实点冲突比单点冲突更突出）。

    ratio==1.0 时视为"知识库已验证原文"（重复入库）→ 0 分，不误伤。
    """
    if not query or not ref:
        return {"ratio": 0.0, "n_fact": 0, "conflict_facts": [],
                "weight": 1.0, "score": 0.0} if detail else 0.0
    if min(len(query), len(ref)) < 6:
        return {"ratio": 0.0, "n_fact": 0, "conflict_facts": [],
                "weight": 1.0, "score": 0.0} if detail else 0.0
    sm = difflib.SequenceMatcher(None, query, ref, autojunk=False)
    ratio = sm.ratio()
    if ratio < min_ratio or ratio >= 1.0:
        if detail:
            return {"ratio": ratio, "n_fact": 0, "conflict_facts": [],
                    "weight": 1.0, "score": 0.0}
        return 0.0
    conflict: list[str] = []
    n_fact = 0
    for tag, q0, q1, r0, r1 in sm.get_opcodes():
        if tag == "equal":
            continue
        qs, rs = query[q0:q1], ref[r0:r1]
        if _is_fact_change(qs, rs):
            n_fact += 1
            conflict.append(qs or rs)
    if n_fact == 0:
        if detail:
            return {"ratio": ratio, "n_fact": 0, "conflict_facts": [],
                    "weight": 1.0, "score": 0.0}
        return 0.0
    weight = 1.0 + 0.25 * (n_fact - 1)   # 多事实点冲突更突出（封顶，避免无上限）
    score = float(ratio) * weight
    if detail:
        return {"ratio": float(ratio), "n_fact": n_fact,
                "conflict_facts": conflict, "weight": float(weight),
                "score": score}
    return score


def _topk_by_cosine(query: np.ndarray, ref_embs: np.ndarray, top_k: int) -> np.ndarray:
    """query [d] × ref_embs [M,d] → 最多 top_k 个最近邻下标（降序）。"""
    sims = query @ ref_embs.T
    k = min(top_k, len(sims))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx


def scan_scores(
    texts: list[str],
    ref_texts: list[str],
    ref_embs: np.ndarray,
    enc,
    q_embs: np.ndarray | None = None,
    top_k: int = 5,
    batch_size: int = 64,
    min_ratio: float = 0.75,
    progress: bool = True,
    aggregate: str = "max",
) -> np.ndarray:
    """对每段文本做检索一致性打分。

    - ref_texts/ref_embs：可信参考语料（已编码，L2 归一化）。
    - enc：.encode(texts)-> [N,d] 的编码器（测试可注入 FakeEncoder）。
    - q_embs：可选，避免重复编码（批量调用时一次编码）。
    - aggregate：如何把 top-k 近邻的冲突合并成一个分数：
        "max"   = 最像的被篡改近邻（默认，保守定位）
        "mean"  = 各近邻冲突分数均值（对多个近邻同时被篡改更敏感）
        "count" = 冲突近邻个数/top_k（对"知识库里几篇都在写同一事实"更鲁棒）
    返回与 texts 等长的 float64 分数（越大越可疑）。
    """
    if aggregate not in ("max", "mean", "count"):
        raise ValueError(f"aggregate 必须是 max/mean/count，got {aggregate!r}")
    if q_embs is None:
        q_embs = enc.encode(list(texts))
    scores = np.zeros(len(texts), dtype=np.float64)
    for i, (t, q) in enumerate(zip(texts, q_embs)):
        if not t:
            continue
        idx = _topk_by_cosine(q, ref_embs, top_k)
        if len(idx) == 0:
            continue
        vals: list[float] = []
        n_conflict = 0
        for j in idx.tolist():
            c = _pair_conflict(t, ref_texts[j], min_ratio=min_ratio)
            sim = float(q @ ref_embs[j])
            v = sim * c
            vals.append(v)
            if c > 0:
                n_conflict += 1
        if aggregate == "max":
            scores[i] = max(vals) if vals else 0.0
        elif aggregate == "mean":
            scores[i] = float(np.mean(vals)) if vals else 0.0
        else:  # "count"
            scores[i] = n_conflict / len(idx) if idx.size else 0.0
    return scores


def explain(text: str, ref_texts: list[str], ref_embs: np.ndarray, enc,
            top_k: int = 3, min_ratio: float = 0.75) -> dict:
    """给一条文本的可读解释：命中的近邻、相似度、冲突事实。"""
    q = enc.encode([text])[0]
    idx = _topk_by_cosine(q, ref_embs, top_k)
    hits = []
    for j in idx.tolist():
        ref = ref_texts[j]
        sm = difflib.SequenceMatcher(None, text, ref, autojunk=False)
        ratio = sm.ratio()
        conflict = _pair_conflict(text, ref, min_ratio=min_ratio) > 0
        facts = [p.as_dict() for p in extract_facts(text)]
        hits.append({
            "ref_idx": int(j), "cosine": round(float(q @ ref_embs[j]), 4),
            "ratio": round(float(ratio), 4), "conflict": conflict,
            "facts": facts, "ref_text": ref,
        })
    return {"score": float(scan_scores([text], ref_texts, ref_embs, enc, top_k=top_k,
                                       min_ratio=min_ratio)[0]),
            "top_hits": hits}


# ----------------------------------------------------------------------
def build_reference(cfg, source: str = "train", max_n: int = 12000) -> list[str]:
    """从已切分的干净数据里取"可信参考语料"（默认 train 的 normal 句）。

    source: train / dev / test / "train+dev" —— 只取 family==normal 的干净句当作
    已入库的可信文档库。这正是落地场景②：向量库/文档库里"已入库且可信"的文档。
    """
    from .data import load_splits

    splits = load_splits(cfg.data.processed_dir)
    parts = source.split("+") if source else []
    texts: list[str] = []
    for part in parts:
        pool = splits.get(part, [])
        texts += [s.text for s in pool
                  if s.label == 0 and s.family == "normal" and s.text.strip()]
    if not texts:
        raise ValueError(f"参考语料为空（source={source!r}；先跑 data 生成切分）")
    if max_n and len(texts) > max_n:
        texts = texts[:max_n]
    return texts


def evaluation_rows(scores: np.ndarray, labels: np.ndarray,
                    families: list[str]) -> dict:
    """按家族报告：n / 覆盖（有无事实点）/ 平均分。"""
    rows: dict[str, dict] = {}
    for fam in sorted(set(families)):
        m = np.array([f == fam for f in families])
        s = scores[m]
        rows[fam] = {"n": int(m.sum()),
                     "mean_score": round(float(np.mean(s)), 4) if len(s) else None,
                     "max_score": round(float(np.max(s)), 4) if len(s) else None}
    return rows


def run_ragcheck(
    cfg,
    out_path: str | Path | None = None,
    encoder_name: str | None = None,
    top_k: int = 5,
    min_ratio: float = 0.75,
    corpus_source: str = "train",
    limit: int = 400,
    seed: int = 42,
    extra_note: str = "",
    aggregate: str = "max",
) -> dict:
    """主入口：构建参考语料 → 编码 → 检索引擎 → 与数字级基线/现有检测器对比。

    报告口径与 run_factcheck 对齐（family 分层抽样 + 整体 AUROC + TPR@1%），
    并额外给出 edit-vs-normal 的内容级对照，保证两组数字可直接比。
    """
    from . import evaluate as E
    from .data import load_splits
    from .encoders import get_encoder

    t0 = time.perf_counter()
    splits = load_splits(cfg.data.processed_dir)
    dev = splits.get("dev", [])
    test = splits.get("test", [])
    dev_normals = [s.text for s in dev if s.label == 0 and s.family == "normal"]
    ref_texts = build_reference(cfg, source=corpus_source)

    clean_dev = [s.text for s in dev if s.label == 0 and s.family == "normal"]
    # 抽样：分层每族取前 limit 条，但保留它们在 test 里的原始下标（用于与
    # test_scores{SUFFIX}.npz 的分数按索引对齐融合，避免"长度相等即对齐"）
    keep_idx: np.ndarray | None = None
    if limit:
        buckets: dict[str, list] = {}
        for i, s in enumerate(test):                       # 只在 test 全量里起步
            b = buckets.setdefault(s.family, [])
            if len(b) < limit:
                b.append(i)
        keep_idx = np.array(sorted(i for fam in sorted(buckets)
                                   for i in buckets[fam]), dtype=int)
        samples = [test[i] for i in keep_idx]
    else:
        samples = list(test)
    test = samples

    if encoder_name:
        # CLI 显式覆盖编码器（crom/bge/tiny）
        old = cfg.encoder.name
        cfg.encoder.name = encoder_name
        try:
            enc = get_encoder(cfg)
        finally:
            cfg.encoder.name = old
    else:
        enc = get_encoder(cfg)

    if len(dev_normals) > 2000:
        dev_normals = dev_normals[:2000]

    print(f"[ragcheck] 参考语料 {len(ref_texts)} 句（{corpus_source}/normal）")
    tref = time.perf_counter()
    ref_embs = enc.encode(ref_texts)
    print(f"[ragcheck] 参考编码 {len(ref_texts)} 条 {time.perf_counter()-tref:.1f}s")

    td = time.perf_counter()
    Xq_dev = enc.encode(dev_normals)
    s_dev = scan_scores(dev_normals, ref_texts, ref_embs, enc, q_embs=Xq_dev,
                        top_k=top_k, min_ratio=min_ratio, progress=False,
                        aggregate=aggregate)
    print(f"[ragcheck] dev 干净集打分 {len(dev_normals)} 条 {time.perf_counter()-td:.1f}s")

    tt = time.perf_counter()
    Xq_te = enc.encode([s.text for s in test])
    s_te = scan_scores([s.text for s in test], ref_texts, ref_embs, enc, q_embs=Xq_te,
                       top_k=top_k, min_ratio=min_ratio, progress=False,
                       aggregate=aggregate)
    print(f"[ragcheck] test 打分 {len(test)} 条 {time.perf_counter()-tt:.1f}s")

    y_te = np.array([s.label for s in test])
    fam_te = np.array([s.family for s in test])
    texts_te = [s.text for s in test]

    # ---------------- 整体 + edit 对照 ----------------
    def _auc(y, s):
        if len(set(y)) < 2:
            return None
        try:
            return float(E.auroc(y, s))
        except Exception:
            return None

    overall = {"auroc": _auc(y_te, s_te),
               "tpr_at_fpr1": _tpr_cal(s_dev, y_te, s_te),
               "n_evaluated": int(len(test)),
               "n_with_conflict": int((s_te > 0).sum()),
               "edit_norm_auroc": _auc(
                   y_te[(fam_te == "edit") | (fam_te == "normal")],
                   s_te[(fam_te == "edit") | (fam_te == "normal")])}
    rows = evaluation_rows(s_te, y_te, fam_te.tolist())

    # ---------------- 与数字级基线（同 one 评测床上跑的 factcheck_real_v3.json）对照 ----------------
    numeric = None
    fpath = Path(cfg.metrics_dir) / "factcheck.json"
    if not fpath.exists():
        fpath = Path(cfg.metrics_dir) / "factcheck_real_v3.json"
    if fpath.exists():
        try:
            d = json.loads(fpath.read_text(encoding="utf-8"))
            numeric = {
                "auroc": d.get("overall", {}).get("auroc"),
                "tpr_at_fpr1": d.get("overall", {}).get("tpr_at_fpr1"),
                "by_family": d.get("by_family", {}),
            }
        except Exception:
            numeric = None

    # ---------------- 与现有检测器融合 / 对比（读 detect 缓存分） ----------------
    fusion = None
    try:
        from .conformal import suffix_of

        npz = np.load(Path(cfg.metrics_dir) / f"test_scores{suffix_of(cfg)}.npz",
                      allow_pickle=True)
        n_te = len(test)
        if "sup_tfidf" in npz.files:
            base_full = np.asarray(npz["sup_tfidf"], dtype=float)
            if len(base_full) == n_te:
                base = base_full                       # 全量对齐（limit=0）
            elif keep_idx is not None and len(base_full) > max(keep_idx, default=0):
                base = base_full[keep_idx]             # 抽样：按原 test 下标取子集
            else:
                base = None
        else:
            base = None
        if base is not None:
            from .factcheck import fuse as _fuse

            n_base = _normalize(base)
            n_rag = _normalize(s_te)
            comb_eq = n_base + n_rag                      # 等权加和
            comb_rank = _rank_avg(base, s_te)             # 排名平均
            comb_fused = _fuse(base, s_te, w=0.3)         # 保留原逻辑做对照
            edit_mask = (fam_te == "edit")
            normal_mask = (fam_te == "normal")
            fusion = {
                "base_auroc_all": _auc(y_te, base),
                "ragcheck_auroc_all": _auc(y_te, s_te),
                "fused_auroc_all": _auc(y_te, comb_fused),
                "fused_eq_auroc_all": _auc(y_te, comb_eq),
                "fused_rank_auroc_all": _auc(y_te, comb_rank),
                "edit_recall@1_base": round(float(_recall1(s_dev, y_te, base, edit_mask)), 4),
                "edit_recall@1_ragcheck": round(float(_recall1(s_dev, y_te, s_te, edit_mask)), 4),
                "edit_recall@1_fused": round(float(_recall1(s_dev, y_te, comb_fused, edit_mask)), 4),
                "edit_recall@1_fused_eq": round(float(_recall1(s_dev, y_te, comb_eq, edit_mask)), 4),
                "edit_recall@1_fused_rank": round(float(_recall1(s_dev, y_te, comb_rank, edit_mask)), 4),
                "normal_fpr@0.5_thr": round(float((s_te[normal_mask] > 0.5).mean()), 4),
                "aligned_by": "index" if keep_idx is not None else "length",
            }
    except Exception as e:  # 没有超过缓存的干净集时跳过
        fusion = None
        print(f"[ragcheck] 跳过融合（{type(e).__name__}: {e}）")

    res = {
        "meta": {
            "method": "retrieval-consistency",
            "encoder": getattr(enc, "name", encoder_name or cfg.encoder.name),
            "corpus_source": corpus_source, "corpus_n": len(ref_texts),
            "top_k": top_k, "min_ratio": min_ratio, "limit": limit, "seed": seed,
            "n_dev_clean": len(dev_normals), "n_test": len(test),
            "seconds": round(time.perf_counter() - t0, 1),
            "note": "分数=检索一致性冲突(0..1)：近重复且差异块含事实点。"
                    "与 factcheck.py 数字级（无条件先验, AUROC 0.353）形成"
                    "「有知识库 vs 无知识库」对照。" + (extra_note or ""),
        },
        "overall": overall,
        "by_family": rows,
        "baseline_numeric_level": numeric,
        "fusion_with_sup_tfidf": fusion,
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"[ragcheck] 已保存 {out_path}")
    print(f"[ragcheck] AUROC(全家族)={overall['auroc']} "
          f"AUROC(edit vs normal)={overall['edit_norm_auroc']} "
          f"TPR@1%={overall['tpr_at_fpr1']} 命中冲突={overall['n_with_conflict']}")
    for fam in ("edit", "normal", "template", "paraphrase", "subtle", "backdoor", "hardneg", "ood"):
        if fam in rows:
            r = rows[fam]
            print(f"[ragcheck] {fam:10s} n={r['n']:4d} 均值={r['mean_score']} 峰值={r['max_score']}")
    if fusion:
        print(f"[ragcheck] 融合 vs 词面: 融合全 AUROC={fusion['fused_auroc_all']} "
              f"edit召回@1% base={fusion['edit_recall@1_base']} → "
              f"ragcheck={fusion['edit_recall@1_ragcheck']} → fused={fusion['edit_recall@1_fused']}")
    return res


# ----------------------------------------------------------------------
def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std() or 1.0
    return (x - x.mean()) / sd


def _rank_avg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """把两个不同尺度的分数都转成 [0,1] 排名占比后取均值。

    对尺度不敏感（词面分大多贴 0/1，检索分偏小），适合异构分数融合。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"_rank_avg 要求等长，got {len(a)} vs {len(b)}")
    ra = np.argsort(np.argsort(-a)).astype(float) / (len(a) or 1)
    rb = np.argsort(np.argsort(-b)).astype(float) / (len(b) or 1)
    return 0.5 * (ra + rb)


def _tpr_cal(s_dev: np.ndarray, y_te: np.ndarray, s_te: np.ndarray) -> float:
    """在 dev 干净集上定 1% FPR 阈值，在 test 上量 TPR。"""
    if len(s_dev) < 10 or len(set(y_te)) < 2:
        return float("nan")
    thr = float(np.quantile(s_dev, 0.99))
    return float((s_te[y_te == 1] > thr).mean()) if (y_te == 1).any() else float("nan")


def _recall1(s_dev: np.ndarray, y_te: np.ndarray, s_te: np.ndarray,
             mask: np.ndarray) -> float:
    thr = float(np.quantile(s_dev, 0.99))
    if not mask.any():
        return float("nan")
    return float((s_te[mask] > thr).mean())
