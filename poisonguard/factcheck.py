"""数字级事实校验（P0-2）：问「这个数字在这句话里合不合理」。

为什么专门盯数字：`edit` 家族是**内容级最小篡改**——把「1300 万」改成「1300 亿」、
把「30%」改成「300%」，措辞、模板、句长全不变，所以字符 n-gram（0.040）和
词面审计都抓不到。而这类篡改有一个共同点：**数字与上下文的搭配变得不合理**。

做法（不依赖外部知识库，只用语言模型本身的先验）：
1. 用正则找出文本里的「数字+单位」；
2. 用 Qwen 做主干的 teacher-forcing，只取这些数字 token 的 `log p(token | 前缀)`；
3. 数字越"突兀"，条件概率越低 → 取负均值作为可疑度。

这是**用生成模型的先验替代知识库**：它不知道 1300 万是不是真的人数，但它知道
「报名人数首次突破 1300 万」比「突破 1300 亿」更像中文新闻。
"""

from __future__ import annotations

import re
import time

import numpy as np

# 数字 + 常见量词/单位（宁可多抓，漏抓会让分数缺失）
NUM_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|％|万|亿|千|百|元|美元|岁|年|个月|月|天|小时|分钟|秒|人|名|次|倍|"
    r"公里|千米|米|吨|公斤|度|所|家|项|起|例|宗|届|层|楼|辆|台|件|份|条|篇|部|种)"
)


def number_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in NUM_RE.finditer(text or "")]


def has_number(text: str) -> bool:
    return bool(number_spans(text))


_UNIT_TOKENS = ("%", "％", "万", "亿", "千", "百", "元", "岁", "年", "月", "天",
                "人", "名", "次", "倍", "米", "吨", "度", "家", "项", "起", "例")


def _number_token_mask(tokens: list[str]) -> list[int]:
    """数字位 + 紧跟的量词（"1300 亿" 与 "1300 万" 的差别全在量词上）。"""
    idx = []
    for i, tok in enumerate(tokens):
        t = tok.strip()
        if not t:
            continue
        if any(ch.isdigit() for ch in t):
            idx.append(i)
    unit = {i + 1 for i in idx
            if i + 1 < len(tokens) and tokens[i + 1].strip() in _UNIT_TOKENS}
    return sorted(set(idx) | unit)


class FactScorer:
    """数字位置的平均条件对数概率（越负越可疑）。"""

    name = "fact_num_logprob"

    def __init__(self, model, tok, max_length: int = 64, device: str = "cpu"):
        self.model, self.tok = model, tok
        self.max_length = int(max_length)
        self.device = device

    @classmethod
    def load(cls, cfg, model_id: str | None = None, **kw):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .localmodels import ensure_model

        mid = model_id or cfg.qwen.model_id
        path = ensure_model(mid, cfg.qwen.model_cache)
        tok = AutoTokenizer.from_pretrained(str(path))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.float32).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, tok, **kw)

    def score(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        import torch

        out = np.full(len(texts), np.nan, dtype=np.float64)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tok([t or "" for t in chunk], return_tensors="pt",
                           truncation=True, max_length=self.max_length,
                           padding=True)
            with torch.no_grad():
                logits = self.model(**enc).logits
            logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target = enc["input_ids"][:, 1:]
            picked = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            for bi, t in enumerate(chunk):
                toks = self.tok.convert_ids_to_tokens(
                    enc["input_ids"][bi].tolist())
                keep = _number_token_mask(toks)
                # 去掉第一个位置（没有前缀可预测）
                keep = [k for k in keep if k >= 1]
                if not keep:
                    continue
                vals = picked[bi, [k - 1 for k in keep]]
                out[i + bi] = float(vals.mean().item())
        return out


def suspiciousness(scores: np.ndarray) -> np.ndarray:
    """把「平均对数概率」翻成「越大越可疑」；缺失保持 NaN。"""
    s = np.asarray(scores, dtype=float).copy()
    out = -s
    out[np.isnan(s)] = np.nan
    return out


def fill_missing(scores: np.ndarray, value: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    return np.where(np.isnan(s), float(value), s)


def fuse(z_base: np.ndarray, z_fact: np.ndarray, w: float = 0.3) -> np.ndarray:
    """两路分数做 z-score 后加权；NaN 已由调用方填。"""
    def _z(x):
        x = np.asarray(x, dtype=float)
        sd = x.std() or 1.0
        return (x - x.mean()) / sd

    return (1.0 - w) * _z(z_base) + w * _z(z_fact)


def evaluation_rows(scores: np.ndarray, labels: np.ndarray,
                    families: list[str]) -> dict:
    """按家族报告：有数字样本占比 + 平均可疑度。"""
    rows: dict[str, dict] = {}
    for fam in sorted(set(families)):
        m = np.array([f == fam for f in families])
        s = scores[m]
        ok = ~np.isnan(s)
        rows[fam] = {"n": int(m.sum()), "n_with_number": int(ok.sum()),
                     "coverage": float(ok.mean()) if len(s) else 0.0,
                     "mean_susp": float(np.nanmean(s)) if ok.any() else None}
    return rows


def run_factcheck(cfg, out_path=None, model_id: str | None = None,
                  limit: int = 0, seed: int = 42) -> dict:
    """在 dev 干净集 + test 上跑数字校验，报 AUROC / TPR@1% / 分家族，并与现有分数融合。"""
    from pathlib import Path

    import json

    from . import evaluate as E
    from .data import load_splits

    t0 = time.perf_counter()
    splits = load_splits(cfg.data.processed_dir)
    dev = splits.get("dev", [])
    test = splits.get("test", [])
    clean_dev = [s for s in dev if s.label == 0 and s.family in ("normal", "hardneg")]
    if limit:
        # 分层抽样：每个家族最多 limit 条（否则 test 前 N 条全是 normal，评不出东西）
        clean_dev = clean_dev[:limit]
        buckets: dict[str, list] = {}
        for s in test:
            b = buckets.setdefault(s.family, [])
            if len(b) < limit:
                b.append(s)
        test = [s for fam in sorted(buckets) for s in buckets[fam]]

    texts_dev = [s.text for s in clean_dev]
    texts_test = [s.text for s in test]
    print(f"[factcheck] 装载模型（{model_id or cfg.qwen.model_id}）…")
    sc = FactScorer.load(cfg, model_id=model_id)

    td = time.perf_counter()
    s_dev = suspiciousness(sc.score(texts_dev))
    print(f"[factcheck] dev 打分 {len(texts_dev)} 条 {time.perf_counter()-td:.1f}s")

    tt = time.perf_counter()
    s_te = suspiciousness(sc.score(texts_test))
    print(f"[factcheck] test 打分 {len(texts_test)} 条 {time.perf_counter()-tt:.1f}s")

    y_te = np.array([s.label for s in test])
    fam_te = [s.family for s in test]
    rows = evaluation_rows(s_te, y_te, fam_te)

    # 只在有数字的样本上评（缺失值会稀释 AUROC）
    have = ~np.isnan(s_te)
    auroc = E.auroc(y_te[have], s_te[have]) if have.sum() > 10 and len(set(y_te[have])) > 1 else None
    dev_have = ~np.isnan(s_dev)
    tpr1 = devcal = None
    if dev_have.sum() > 10 and have.sum() > 10 and len(set(y_te[have])) > 1:
        thr = float(np.quantile(s_dev[dev_have], 0.99))
        tpr1 = float((s_te[have][y_te[have] == 1] > thr).mean())
        devcal = tpr1

    # 与现有检测器融合（读 detect 的缓存分数）
    fusion = None
    try:
        from .conformal import suffix_of

        npz = np.load(Path(cfg.metrics_dir) / f"test_scores{suffix_of(cfg)}.npz",
                      allow_pickle=True)
        if "sup_tfidf" in npz.files and len(npz["sup_tfidf"]) == len(test):
            base = np.asarray(npz["sup_tfidf"], dtype=float)
            fill = float(np.nanmean(s_te)) if np.isfinite(np.nanmean(s_te)) else -1.0
            comb = fuse(base, fill_missing(s_te, fill), w=0.3)
            fusion = {"base_auroc": E.auroc(y_te, base),
                      "fused_auroc": E.auroc(y_te, comb),
                      "base_edit_recall@1%": float(np.mean(
                          base[(y_te == 1) & (np.array(fam_te) == "edit")]
                          > float(np.quantile(base[y_te == 0], 0.99)))),
                      "fused_edit_recall@1%": float(np.mean(
                          comb[(y_te == 1) & (np.array(fam_te) == "edit")]
                          > float(np.quantile(comb[y_te == 0], 0.99))))}
    except FileNotFoundError:
        fusion = None

    res = {
        "meta": {"model": model_id or cfg.qwen.model_id, "limit": limit,
                 "n_dev_clean": len(texts_dev), "n_test": len(texts_test),
                 "seed": seed, "seconds": round(time.perf_counter() - t0, 1),
                 "note": "分数= -数字位置平均对数概率（越大越可疑）；只在含数字样本上评"},
        "overall": {"auroc": auroc, "tpr_at_fpr1": tpr1,
                    "n_evaluated": int(have.sum()),
                    "coverage": float(have.mean())},
        "by_family": rows,
        "fusion_with_sup_tfidf": fusion,
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"[factcheck] 已保存 {out_path}")
    print(f"[factcheck] AUROC={auroc} TPR@1%={tpr1} 覆盖={have.mean():.1%}")
    for fam, r in rows.items():
        if fam in ("edit", "normal", "template", "paraphrase", "subtle", "backdoor"):
            print(f"[factcheck] {fam:11s} n={r['n']:4d} 含数字={r['coverage']:.0%} "
                  f"均值={r['mean_susp']}")
    return res
