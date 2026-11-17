"""改写鲁棒性边界探测：syn / chr / mix 变换下 ragcheck 检出率的塌缩。

动机（诚实边界实测）：
    检索式事实核验（poisonguard/ragcheck.py，只读）靠 CoROM 余弦 top-k 近邻 +
    "近重复(ratio>=0.75) + 事实改动"判冲突，设计边界是"in-KB 篡改"：库内近重复
    存在才给分。本项目对"改写型投毒"的诚实交代是"OOS：改写后与库内句子不再近
    重复，0 分不断言"。本脚本把这件事**实测**出来：对 edit 家族（改过库内原文的
    最小篡改）做三类改写（同义 syn / 字符噪声 chr / 混合 mix），看 ragcheck 检出率
    相对基线（不改写）塌多少。

方法：
    1. 从 data/processed_real_v3/test.jsonl 各取 edit(正) / normal(负) ~100 条（seed 固定）；
    2. 对每族、每种变换 (none/syn/chr/mix) 用 poisonguard.evade.evade_text 生成变体
       （每样本独立确定性子种子）——变换对正负两侧都做，阈值同口径；
    3. 先 TinyEncoder 快速 sanity（纯逻辑接线），再真实 CoROM（768d，参考库复用
       outputs/cache 既有 12000-row 缓存，靠绝对 model_dir 命中）小批量打分；
    4. scan_scores(top_k=5, min_ratio=0.75, aggregate=max)，阈值在 normal 分布 99 分位
       标定 FPR=1%；
    5. 额外诊断：每变换实际改动比例 changed_frac、基线命中样本的检出存留率
       flag_retention、edit/normal 分离度（均值差），让"检出率没塌"不是"变换没生效"；
    6. 写入 outputs/metrics/robust_ragcheck.json：每变换检出率、与基线差、中文结论
       （塌陷 → "改写型 OOS 边界确实存在"；不塌陷 → 说明轻改写跨不过检索层闸门，
        边界在"整句形态脱离库内"时才硬）。

用法：python scripts/robust_ragcheck.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------- 常量 ----------------
PROCESSED_DIR = "data/processed_real_v3"
METHODS = ["none", "syn", "chr", "mix"]
N_SAMPLE = 100
SEED = 42
TOP_K = 5
MIN_RATIO = 0.75
CORPUS_N = 12000
OUT_JSON = ROOT / "outputs" / "metrics" / "robust_ragcheck.json"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sample_rows(rows: list[dict], family: str, label: int, n: int, seed: int) -> list[str]:
    pool = [r for r in rows if r.get("family") == family and r.get("label") == label]
    rng = random.Random(seed)
    picked = rng.sample(pool, min(n, len(pool)))
    return [r["text"] for r in picked]


def make_variants(texts: list[str], method: str, base_seed: int) -> list[str]:
    """对一族文本批量生成 method 变体（每样本独立子种子，确定性）。"""
    from poisonguard.evade import evade_text

    return [evade_text(t, method, seed=base_seed + i) for i, t in enumerate(texts)]


def scan(enc, texts: list[str], ref_texts: list[str], ref_embs: np.ndarray) -> np.ndarray:
    """封装 scan_scores，传 q_embs 避免二次编码，关掉进度条。"""
    from poisonguard.ragcheck import scan_scores

    q_embs = enc.encode(texts)
    return scan_scores(
        texts, ref_texts, ref_embs, enc,
        q_embs=q_embs, top_k=TOP_K, min_ratio=MIN_RATIO, progress=False,
    )


def summarize(scores: np.ndarray) -> dict:
    return {
        "n": int(len(scores)),
        "mean": round(float(scores.mean()), 6),
        "max": round(float(scores.max()), 6),
        "frac_pos": round(float(np.mean(scores > 0)), 4),
    }


def main() -> None:
    print(f"[robust_ragcheck] ROOT={ROOT}")

    # ---------- 1. 数据抽样 ----------
    rows = load_rows(ROOT / PROCESSED_DIR / "test.jsonl")
    edit_orig = sample_rows(rows, family="edit", label=1, n=N_SAMPLE, seed=SEED)
    norm_orig = sample_rows(rows, family="normal", label=0, n=N_SAMPLE, seed=SEED * 7 + 1)
    print(f"[data] edit={len(edit_orig)} normal={len(norm_orig)}")

    # ---------- 2. TinyEncoder 快速 sanity（纯逻辑接线） ----------
    from poisonguard.encoders import TinyEncoder

    tiny = TinyEncoder()
    train_rows = load_rows(ROOT / PROCESSED_DIR / "train.jsonl")
    tiny_ref = [r["text"] for r in train_rows if r.get("family") == "normal"][:500]
    tiny_ref_embs = tiny.encode(tiny_ref)
    for mi, m in enumerate(METHODS):
        ev = make_variants(edit_orig[:20], m, base_seed=1000 + mi * 100000)
        nv = make_variants(norm_orig[:20], m, base_seed=2000 + mi * 100000)
        es = scan(tiny, ev, tiny_ref, tiny_ref_embs)
        ns = scan(tiny, nv, tiny_ref, tiny_ref_embs)
        print(f"  [tiny sanity] {m}: edit_mean={es.mean():.4f} normal_mean={ns.mean():.4f} "
              f"edit_pos={np.mean(es > 0):.2f}")

    # ---------- 3. 真实 CoROM：参考库 + 各族变体打分 ----------
    from poisonguard.config import load_config
    from poisonguard.encoders import get_encoder
    from poisonguard.ragcheck import build_reference

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.data.processed_dir = str((ROOT / PROCESSED_DIR).relative_to(ROOT))
    # 关键：绝对 model_dir 命中 outputs/cache 既有 12000-row CoROM 参考向量
    cached_abs_model_dir = ROOT / "model_cache" / "damo" / "nlp_corom_sentence-embedding_chinese-base"
    if cached_abs_model_dir.exists():
        cfg.encoder.model_dir = str(cached_abs_model_dir)
    print(f"[encoder] model_dir={cfg.encoder.model_dir}")
    enc = get_encoder(cfg)

    ref_texts = build_reference(cfg, source="train", max_n=CORPUS_N)
    ref_embs = enc.encode(ref_texts)
    print(f"[corpus] ref_texts={len(ref_texts)} ref_embs={ref_embs.shape}")

    variants: dict[str, dict[str, list[str]]] = {}
    raw: dict[str, dict[str, np.ndarray]] = {}
    for mi, m in enumerate(METHODS):
        variants[m] = {
            "edit": make_variants(edit_orig, m, base_seed=1000 + mi * 100000),
            "normal": make_variants(norm_orig, m, base_seed=2000 + mi * 100000),
        }
        raw[m] = {
            "edit": scan(enc, variants[m]["edit"], ref_texts, ref_embs),
            "normal": scan(enc, variants[m]["normal"], ref_texts, ref_embs),
        }
        print(f"[crom] {m}: edit_mean={raw[m]['edit'].mean():.4f} "
              f"normal_mean={raw[m]['normal'].mean():.4f}")

    # ---------- 4. 指标：FPR=1% 阈值 + 变换生效诊断 + 存留率 ----------
    base_es = raw["none"]["edit"]
    base_flag = base_es > 0  # 基线 n>0 即被标记（normal 样本全 0 → 阈值退化为 0）
    results: dict[str, dict] = {}
    for m in METHODS:
        es, ns = raw[m]["edit"], raw[m]["normal"]
        changed_frac_e = np.mean(
            [v != o for v, o in zip(variants[m]["edit"], edit_orig)])
        changed_frac_n = np.mean(
            [v != o for v, o in zip(variants[m]["normal"], norm_orig)])
        thr = float(np.quantile(ns, 0.99))
        flag = es > thr
        det = float(np.mean(flag))
        # 基线已命中样本里，本变换后仍命中的比例（robustness of detection）
        keep = float(np.mean(flag[base_flag])) if base_flag.any() else 0.0
        sep = float(es.mean() - ns.mean())
        results[m] = {
            "edit": summarize(es),
            "normal": summarize(ns),
            "changed_frac_edit": round(float(changed_frac_e), 4),
            "changed_frac_normal": round(float(changed_frac_n), 4),
            "fpr1_threshold": round(thr, 6),
            "detect_rate_at_fpr1": round(det, 4),
            "flag_retention_on_baseline_hits": round(keep, 4),
            "edit_minus_normal_mean": round(sep, 4),
        }

    base_rate = results["none"]["detect_rate_at_fpr1"]
    base_keep = results["none"]["flag_retention_on_baseline_hits"]
    for m in METHODS:
        results[m]["delta_vs_baseline"] = round(
            results[m]["detect_rate_at_fpr1"] - base_rate, 4)

    # ---------- 5. 中文结论 ----------
    lines = []
    collapsed = []
    for m in METHODS[1:]:
        r = results[m]
        lines.append(
            f"{m} 检出率 {base_rate:.4f}→{r['detect_rate_at_fpr1']:.4f}"
            f"（Δ {r['delta_vs_baseline']:+.4f}，edit 均分 {r['edit']['mean']:.4f}，"
            f"变更率 {r['changed_frac_edit']:.2f}，基线命中存留 {r['flag_retention_on_baseline_hits']:.2f}）"
        )
        if r["detect_rate_at_fpr1"] < 0.5 * max(base_rate, 1e-6):
            collapsed.append(m)

    if collapsed:
        note = (
            f"收盘：{'/'.join(collapsed)} 变换让 edit 检出塌陷——**改写型 OOS 边界确实存在**："
            f"同义改写/字符加噪足以让检索式事实核验漏检（改写后不再与库内原文近重复）。"
        )
    else:
        note = (
            f"收盘：syn/chr/mix 均未让 edit 检出塌陷（检出率保持 {base_rate:.2f} 上下，"
            f"变化仅在 -{max(0, base_rate - results['chr']['detect_rate_at_fpr1']):.2f} 量级）——"
            f"**改写型 OOS 边界在「轻改写」下并不生效**：edit 起点本身就是库内篡改，syn/chr/mix 只是"
            f"句子内部措辞/字符级改动，不改动「整体形态即库内近重复」这一事实，近重复闸门仍命中、"
            f"事实点差异仍被检出；真正让检索层 OOS 的是 template/paraphrase/subtle 这类"
            f"整句层面脱离库内句式的重写（它们基线即 0 分）。即：OOS 边界是「整句形态」边界，"
            f"不是「措辞」边界。"
        )
    conclusion = "基线(none)@1%FPR 检出率 = {:.4f}；".format(base_rate) + "；".join(lines) + "。" + note
    print("\n[结论]", conclusion)

    payload = {
        "meta": {
            "task": "t2-robust_ragcheck",
            "encoder": "crom",
            "corpus_source": "train",
            "corpus_n": CORPUS_N,
            "top_k": TOP_K,
            "min_ratio": MIN_RATIO,
            "aggregate": "max",
            "n_edit": N_SAMPLE,
            "n_normal": N_SAMPLE,
            "seed": SEED,
            "fpr": 0.01,
        },
        "per_method": results,
        "baseline_detect_rate_at_fpr1": round(base_rate, 4),
        "baseline_flag_retention": round(base_keep, 4),
        "conclusion": conclusion,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[output] {OUT_JSON}")


if __name__ == "__main__":
    main()
