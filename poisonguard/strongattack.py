"""A3：更强的投毒攻击 —— 检验「词面 + 语义」双模在真实攻击者能力下是否还成立。

背景
----
A4（`evade.py`）已经证明：**词面逃避**（同义替换 / 字符扰动 / 组合）几乎压不动
edit 召回，说明那 4% 是检测器的能力上限、不是攻击太弱。但那只是"攻击者换词汇"。
一个更尖锐的问题：如果攻击者用**更强的武器**——LLM 回译改写、字符级混淆、
针对编码器的梯度对抗后缀（GCG）——检测器还站得住吗？

本模块给出 5 类强攻击，全部**围绕同一批真实投毒正样本**做（内容事实保持不变，
只改"表面形态"或"表征位置"），并复用 detect 的拟合/校准/打分链路量配对跌幅：

    zero_width  零宽字符插入（U+200B/200C/FEFF）——屏幕上完全不可见，char n-gram 全碎
    homoglyph   同形字符替换（数字→全角、拉丁→西里尔、常见汉字异体）——人眼几乎无差
    fullwidth   全角化（ASCII → 全角）——词表变了、字形几乎没变
    backtrans   LLM 回译（中→英→中，Qwen2.5-0.5B 两跳）——换整句措辞但保留事实
    gcg         白盒梯度对抗后缀：对 CoROM 编码器做 greedy-coordinate-gradient，
                让**编码器表征**远离投毒流形、贴近干净质心（攻击 embedding 类检测器）

诚实边界（写进报告）
--------------------
1. 回译与 GCG 是**抽样**（CPU 成本），obfuscation 三类是全量；两者分开报，不混。
2. GCG 优化的是编码器余弦（白盒假设：攻击者知道编码器），它**不保证**对 char n-gram
   检测器有效——这正是"词面 vs 语义两种检测器谁更抗打"的对照点。
3. 所有方法只改形态、不改事实，因此"召回跌幅"直接等于"该方法对检测器的破坏力"。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# 1) 字符级混淆：零宽 / 同形 / 全角
# ----------------------------------------------------------------------
ZERO_WIDTH = ("\u200b", "\u200c", "\u200d", "\ufeff")

# 同形替换表：视觉几乎相同，但 UTF-8 码位不同 → char n-gram 全部换掉
_HOMOGLYPHS: dict[str, str] = {
    # 数字 → 全角数字
    **{str(d): chr(0xFF10 + d) for d in range(10)},
    # 拉丁字母 → 全角
    **{chr(ord("a") + i): chr(0xFF41 + i) for i in range(26)},
    **{chr(ord("A") + i): chr(0xFF21 + i) for i in range(26)},
    # 拉丁 → 西里尔同形
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
    # 常见汉字异体/近似字（这些在 n-gram 上是完全不同的 token）
    "国": "囯", "门": "鬥", "已": "己", "未": "末", "人": "入", "日": "曰",
    "土": "士", "干": "千", "子": "孑", "大": "太", "王": "壬",
}

_FULLWIDTH = {chr(i): chr(i + 0xFEE0) for i in range(0x21, 0x7F)}
_FULLWIDTH[" "] = "\u3000"


def obfuscate(text: str, method: str, seed: int = 0, p: float = 0.35) -> str:
    """对文本做字符级混淆（确定性：同一 seed 结果一致）。

    method: zero_width / homoglyph / fullwidth / mix
    """
    rng = random.Random(seed)
    if method == "zero_width":
        return _insert_zero_width(text, rng)
    if method == "homoglyph":
        return _replace_homoglyph(text, rng, p)
    if method == "fullwidth":
        return "".join(_FULLWIDTH.get(c, c) for c in text)
    if method == "mix":
        out = _replace_homoglyph(text, rng, p)
        out = "".join(_FULLWIDTH.get(c, c) for c in out)
        return _insert_zero_width(out, rng)
    raise ValueError(f"未知混淆方法: {method}")


def _insert_zero_width(text: str, rng: random.Random, every: int = 2) -> str:
    """每隔 every 个字符插一个零宽字符：视觉不变、字符 n-gram 全断。"""
    out: list[str] = []
    for i, c in enumerate(text):
        out.append(c)
        if i % every == 0 and i < len(text) - 1:
            out.append(rng.choice(ZERO_WIDTH))
    return "".join(out)


def _replace_homoglyph(text: str, rng: random.Random, p: float) -> str:
    return "".join(_HOMOGLYPHS.get(c, c) if rng.random() < p else c for c in text)


def obfuscate_samples(samples, method: str, seed: int = 42):
    """对正样本做混淆；返回 (原文本, 混淆文本) 列表（配对，便于逐条比较）。"""
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    for s in samples:
        if getattr(s, "label", 1) != 1:
            continue
        t = obfuscate(s.text, method, seed=rng.randint(0, 2**31 - 1))
        pairs.append((s.text, t))
    return pairs


# ----------------------------------------------------------------------
# 2) LLM 回译（中→英→中），带 jsonl 缓存
# ----------------------------------------------------------------------
_BT_PROMPT_EN = "Translate the following Chinese sentence into English. Output only the English translation:\n{text}"
_BT_PROMPT_ZH = "Translate the following English sentence into Chinese. Output only the Chinese translation:\n{text}"


def _bt_cache_path(cfg, tag: str) -> Path:
    d = Path(cfg.outputs_dir) / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"backtrans_{tag}.jsonl"


def _load_bt_cache(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["src"]] = row["dst"]
    return out


def backtranslate(texts: list[str], cfg, tag: str = "v3",
                  model=None, tok=None) -> tuple[list[str], dict]:
    """中→英→中 两跳回译（Qwen2.5-0.5B Instruct，greedy）。

    带 jsonl 缓存：同一句只翻一次，重跑不重复付 CPU 成本。
    返回 (回译文本, meta{model, n_cached, n_generated, seconds})。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .localmodels import ensure_model

    t0 = time.time()
    cache_path = _bt_cache_path(cfg, tag)
    cache = _load_bt_cache(cache_path)

    model_id = cfg.qwen.model_id
    todo = [t for t in texts if t not in cache]
    if model is None and todo:
        path = ensure_model(model_id, cfg.qwen.model_cache)
        tok = tok or AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.float32).eval()

    def _gen(prompts: list[str], max_new_tokens: int = 64) -> list[str]:
        outs: list[str] = []
        with torch.inference_mode():
            for p in prompts:
                msgs = [{"role": "user", "content": p}]
                prompt = tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
                enc = tok(prompt, return_tensors="pt", truncation=True, max_length=384)
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
                txt = tok.decode(out[0][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
                outs.append(txt)
        return outs

    n_gen = 0
    if todo:
        print(f"[strongattack] 回译 {len(todo)} 句（中→英→中）…")
        en = _gen([_BT_PROMPT_EN.format(text=t) for t in todo])
        zh = _gen([_BT_PROMPT_ZH.format(text=e) for e in en])
        with cache_path.open("a", encoding="utf-8") as f:
            for src, dst in zip(todo, zh):
                if not dst:
                    continue
                cache[src] = dst
                f.write(json.dumps({"src": src, "dst": dst}, ensure_ascii=False) + "\n")
                n_gen += 1

    out = [cache.get(t, t) for t in texts]
    meta = {"model": model_id, "n_cached": len(texts) - n_gen,
            "n_generated": n_gen, "seconds": round(time.time() - t0, 1)}
    return out, meta


# ----------------------------------------------------------------------
# 3) GCG 对抗后缀（白盒：攻击者知道编码器）
# ----------------------------------------------------------------------
# 候选字表：常用中文虚词/标点（作为后缀拼进去不改变事实语义）
_GCG_VOCAB = list("的了是在和有也就都很还没被把对从为以而且或将能够这那个们时地得着过又更最再只")
_GCG_L = 3  # 后缀长度（字符数）


def _load_bone(cfg):
    """加载**检测器自己用的编码器**（CoROM），白盒 GCG 的目标就是它。

    注意不要用判官主干：那会变成"攻击一个检测器并不使用的模型"，
    GCG 的杀伤就没有意义了。
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    md = str(cfg.encoder.model_dir)
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModel.from_pretrained(md, torch_dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def gcg_suffix(texts: list[str], cfg, clean_texts: list[str],
               steps: int = 4, topk: int = 6, tag: str = "v3") -> tuple[list[str], dict]:
    """对编码器做 GCG：给每句加一个 3 字后缀，使表征更贴近**干净质心**。

    白盒假设（攻击者知道编码器）；目标是最小化与干净质心的余弦距离，
    即尽量让 encoder 认为它"像干净文本"。结果缓存在 jsonl 里。
    """
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    cache_path = _bt_cache_path(cfg, f"gcg_{tag}")
    cache = _load_bt_cache(cache_path)
    todo = [t for t in texts if t not in cache]

    n_gen = 0
    if todo:
        model, tok = _load_bone(cfg)
        emb_layer = model.get_input_embeddings()
        # 单字候选：只保留 tokenizer 里恰好是 1 个 token 的字（未知字会退化成多 token/None）
        keep_chars, keep_ids = [], []
        for ch in _GCG_VOCAB:
            ids = tok.encode(ch, add_special_tokens=False)
            if len(ids) == 1:
                keep_chars.append(ch)
                keep_ids.append(int(ids[0]))
        if len(keep_ids) < 8:
            raise RuntimeError(f"GCG 候选字表可用项过少({len(keep_ids)})，请检查 tokenizer")
        vocab_t = torch.tensor(keep_ids, dtype=torch.long)
        vocab_emb = emb_layer.weight[vocab_t].detach()          # [V,768]

        # 干净质心：用 dev 干净文本的子集（不含 test，避免泄漏）；256 条足够稳
        cen_texts = clean_texts[:256]
        with torch.no_grad():
            cent_sum = torch.zeros(model.config.hidden_size)
            n_used = 0
            for i in range(0, len(cen_texts), 32):
                batch = tok(cen_texts[i:i + 32], return_tensors="pt",
                            truncation=True, max_length=64, padding=True)
                h = model(**batch).last_hidden_state
                m = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
                pooled = F.normalize(pooled, dim=-1)
                cent_sum += pooled.sum(0)
                n_used += pooled.shape[0]
            centroid = F.normalize(cent_sum / max(n_used, 1), dim=-1).clone()

        def _cos(prefix_ids: torch.Tensor, suffix_ids: torch.Tensor) -> torch.Tensor:
            """给定 prefix token ids + suffix 的 vocab 下标，返回与质心的余弦（可批量）。"""
            pe = emb_layer(prefix_ids)                       # [1,P,768]
            # 批量：suffix_ids [B,L] 是 vocab 下标 → 直接查表
            se = vocab_emb[suffix_ids]                       # [B,L,768]
            B = se.shape[0]
            x = torch.cat([pe.expand(B, -1, -1), se], dim=1)
            h = model(inputs_embeds=x).last_hidden_state
            pooled = h.mean(1)
            return F.normalize(pooled, dim=-1) @ centroid

        # 注意：不能整段 inference_mode —— GCG 要靠 autograd 拿梯度
        for k, text in enumerate(todo, 1):
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=60)["input_ids"]
            with torch.no_grad():
                suffix = torch.randint(0, len(vocab_t), (_GCG_L,))
                best = float(_cos(ids, suffix.unsqueeze(0))[0])
            for _step in range(steps):
                # 梯度近似：对 one-hot 求导，取每位置 topk
                oh = F.one_hot(suffix, len(vocab_t)).float().unsqueeze(0)  # [1,L,V]
                oh.requires_grad_(True)
                se = oh @ vocab_emb                                     # [1,L,768]
                pe = emb_layer(ids).expand(1, -1, -1)
                h = model(inputs_embeds=torch.cat([pe, se], 1)).last_hidden_state
                cosv = (F.normalize(h.mean(1), dim=-1) @ centroid).sum()
                g = torch.autograd.grad(cosv, oh)[0][0]                  # [L,V]
                cand = g.topk(topk, dim=-1).indices                       # [L,k]
                # 批量评估：每个位置 × 每个候选（各一行）
                rows = []
                for pos in range(_GCG_L):
                    for ci in range(topk):
                        s = suffix.clone()
                        s[pos] = cand[pos, ci]
                        rows.append(s)
                cands = torch.stack(rows)                                # [L*k, L]
                with torch.no_grad():
                    scores = _cos(ids, cands)
                j = int(scores.argmax())
                if float(scores[j]) > best + 1e-6:
                    best = float(scores[j])
                    suffix = cands[j].clone()
            suffix_text = "".join(keep_chars[int(i)] for i in suffix.tolist())
            dst = text + suffix_text
            cache[text] = dst
            with cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"src": text, "dst": dst}, ensure_ascii=False) + "\n")
            n_gen += 1
            print(f"[strongattack] GCG {k}/{len(todo)} (cos={best:.4f})")

    out = [cache.get(t, t) for t in texts]
    meta = {"model": cfg.qwen.model_id, "suffix_len": _GCG_L, "steps": steps,
            "topk": topk, "n_cached": len(texts) - n_gen, "n_generated": n_gen,
            "seconds": round(time.time() - t0, 1)}
    return out, meta


# ----------------------------------------------------------------------
def strong_transform(texts: list[str], method: str, cfg=None, seed: int = 42,
                     clean_texts: list[str] | None = None,
                     btag: str = "v3") -> tuple[list[str], dict]:
    """统一入口：按 method 返回变换后的文本 + meta。"""
    if method in ("zero_width", "homoglyph", "fullwidth", "mix"):
        rng = random.Random(seed)
        return ([obfuscate(t, method, seed=rng.randint(0, 2**31 - 1)) for t in texts],
                {"method": method, "kind": "char_obfuscation", "n": len(texts)})
    if method == "backtrans":
        return backtranslate(texts, cfg, tag=btag)
    if method == "gcg":
        if not clean_texts:
            raise ValueError("gcg 需要 clean_texts（dev 干净文本）作为质心")
        return gcg_suffix(texts, cfg, clean_texts, tag=btag)
    raise ValueError(f"未知强攻击方法: {method}")


# ----------------------------------------------------------------------
def run_attack_study(cfg, method: str = "zero_width", sample: int = 0,
                     with_judge: bool = False, seed: int = 42,
                     out_path: Path | None = None) -> dict:
    """对真实投毒正样本做强攻击，量「原样召回 vs 强攻击后召回」配对跌幅。

    复用 detect 的拟合/校准/打分链路（`cli._fit_and_eval`），只把受攻击行
    的向量替换成变换后的向量——保证两列数字在**同一批样本、同一阈值**下可比。
    """
    from .cli import (_fit_and_eval, _labels, _load_or_build_splits,
                      _main_eval_mask, _metrics_path, _texts)
    from .encoders import get_encoder
    from .evaluate import save_json

    t0 = time.time()
    splits = _load_or_build_splits(cfg)
    tr, dev, test = splits["train"], splits["dev"], splits["test"]

    enc = get_encoder(cfg)
    Xtr = enc.encode(_texts(tr))
    Xdev = enc.encode(_texts(dev))
    Xte = enc.encode(_texts(test))

    pos_idx = [i for i, s in enumerate(test) if s.label == 1]
    rng = random.Random(seed)
    if sample and sample < len(pos_idx):
        pos_idx = sorted(rng.sample(pos_idx, sample))
    texts_pos = [test[i].text for i in pos_idx]

    clean_texts = [s.text for s in dev if s.label == 0 and s.family in ("normal", "hardneg")]
    attacked, meta = strong_transform(texts_pos, method, cfg, seed=seed,
                                      clean_texts=clean_texts)

    attacked_full = [s.text for s in test]
    for i, t in zip(pos_idx, attacked):
        attacked_full[i] = t
    new_embs = enc.encode([attacked_full[i] for i in pos_idx])
    Xatk = Xte.copy()
    Xatk[pos_idx] = new_embs

    _dev_scores, te_scores, ev_scores, cals, metrics, _w = _fit_and_eval(
        cfg, Xtr, _texts(tr), dev, Xdev, test, Xte,
        eval_text=attacked_full, eval_X=Xatk)

    idx = np.array(pos_idx)
    by_det: dict[str, dict] = {}
    by_fam: dict[str, dict] = {}
    # fusion 在 _fit_and_eval 内部是"后拼"的，evaded 侧可能没带出来 → 用权重重拼
    # fusion 在 _fit_and_eval 内部用「各检测器映射到 FPR 后的分数」加权融合，
    # 且 evaded 侧没有带出 fusion → 用同一套映射重拼
    if "fusion" in te_scores and "fusion" not in ev_scores and _w:
        if all(k in ev_scores for k in _w):
            ev_scores = dict(ev_scores)
            ev_scores["fusion"] = sum(w * cals[k].to_fpr(ev_scores[k])
                                      for k, w in _w.items())
    for name in te_scores:
        if name not in ev_scores:
            continue
        base_s = te_scores[name]
        atk_s = ev_scores[name]
        pb = cals[name].predict(base_s)
        pa = cals[name].predict(atk_s)
        by_det[name] = {
            "recall_original": round(float(pb[idx].mean()), 4),
            "recall_attacked": round(float(pa[idx].mean()), 4),
            "mean_score_delta": round(float((base_s[idx] - atk_s[idx]).mean()), 4),
            "pct_scores_dropped": round(float((base_s[idx] > atk_s[idx]).mean()), 4),
        }
        fam: dict[str, dict] = {}
        for j, i in enumerate(pos_idx):
            f = test[i].family
            d = fam.setdefault(f, {"n": 0, "_b": [], "_a": []})
            d["n"] += 1
            d["_b"].append(int(pb[i]))
            d["_a"].append(int(pa[i]))
        by_fam[name] = {
            f: {"n": v["n"], "recall_original": round(float(np.mean(v["_b"])), 4),
                "recall_attacked": round(float(np.mean(v["_a"])), 4)}
            for f, v in fam.items()
        }

    judge = None
    if with_judge:
        from .judge import build_judge
        j = build_judge(cfg, "qwen_logprob",
                        cache_path=_metrics_path(cfg, "judge_cache.json"),
                        scoring="logprob")
        s0 = j.score([test[i].text for i in pos_idx])
        s1 = j.score(attacked)
        judge = {
            "name": "qwen_logprob",
            "mean_original": round(float(s0.mean()), 4),
            "mean_attacked": round(float(s1.mean()), 4),
            "mean_delta": round(float((s0 - s1).mean()), 4),
            "pct_dropped": round(float((s0 > s1).mean()), 4),
        }

    res = {
        "meta": {
            "method": method,
            "kind": meta.get("kind", "llm_or_gradient"),
            "n_pos_transformed": len(pos_idx),
            "n_test": len(test),
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "attack_meta": meta,
            "examples": [{"original": a, "attacked": b}
                         for a, b in list(zip(texts_pos, attacked))[:2]],
        },
        "by_detector": by_det,
        "by_family": by_fam,
        "judge": judge,
        "seconds": round(time.time() - t0, 1),
    }
    if out_path is None:
        out_path = _metrics_path(cfg, f"strongattack_{method}.json")
    save_json(res, out_path)
    for name, v in by_det.items():
        print(f"[strongattack] {name:<12} 召回 {v['recall_original']:.3f} → "
              f"{v['recall_attacked']:.3f}（Δ分 {v['mean_score_delta']:+.4f}）")
    print(f"[strongattack] 已保存 {out_path}（{res['seconds']}s）")
    return res
