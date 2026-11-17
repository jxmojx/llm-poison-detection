"""A3 KB 闭环：把 RAG 守卫从“实验用 train 语料”变成“真知识库”。

kb build     --kb-dir outputs/kb            # 参考语料 → 编码 → 持久化为 .npz（文本+向量+meta）
kb check     --kb-dir outputs/kb  --file x  # 用持久化 KB 做检索一致性核验（含 counts + 冲突清单 CSV 报告）
kb selfcheck --kb-dir outputs/kb            # 内置自检：库内原文 vs 篡改对，断言闸门未退化
kb smoke     --kb-dir outputs/kb            # 冒烟：build → selfcheck → check 一条龙产出真实性报告

设计：复用 ragcheck 的 build_reference + enc.encode（CoROM）＋向量持久化（npz）。
策略：KB 缺失时 check/selfcheck/smoke 自动回退 build（开箱即用，不严格崩溃）。
产物被 outputs/.gitignore 忽略（勿提交 KB 二进制与探针文件）。

CLI 用法（挂在 poisonguard.cli，与现有 scan/deploy 同一入口风格）。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poisonguard.config import load_config  # noqa: E402
from poisonguard.data import load_splits  # noqa: E402
from poisonguard.ragcheck import build_reference, scan_scores  # noqa: E402
from poisonguard.encoders import get_encoder  # noqa: E402

KB_NAME = "kb.npz"
DEFAULT_MAX_N = 12000
DEFAULT_TOP_K = 5
DEFAULT_MIN_RATIO = 0.75
DEFAULT_AGGREGATE = "max"
_NUM_RE = re.compile(r"\d+")


def _kb_dir(cfg) -> Path:
    """KB 目录：优先 config 显式指定，否则仓库 outputs/kb。"""
    return Path(getattr(cfg, "kb_dir", ROOT / "outputs" / "kb"))


def _kb_path(kb_dir=None) -> Path:
    return (Path(kb_dir) if kb_dir else _kb_dir(load_config())) / KB_NAME


def _load_config():
    cfg = load_config()
    cfg.data.processed_dir = str(ROOT / "data" / "processed_real_v3")
    return cfg


def tamper_number(text: str) -> str:
    """把文本里第一个数字 +1（没有数字则原样返回）。用于 selfcheck 构造篡改对。"""
    def _up(m):
        return str(int(m.group(0)) + 1)
    return _NUM_RE.sub(_up, text, count=1)


def kb_build(cfg, out_dir=None, corpus_source="train", max_n=DEFAULT_MAX_N,
             top_k=DEFAULT_TOP_K) -> Path:
    """build：参考语料 → 编码 → 持久化 kb.npz。返回路径。"""
    out = Path(out_dir) if out_dir else _kb_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    enc = get_encoder(cfg)
    ref_texts = build_reference(cfg, source=corpus_source, max_n=max_n)
    if not ref_texts:
        raise ValueError("参考语料为空：请确认 data/processed_real_v3 已生成（跑一遍 detect/distill）")
    ref_embs = enc.encode(ref_texts)
    path = out / KB_NAME
    np.savez_compressed(
        path,
        texts=np.asarray(ref_texts, dtype=object),
        embs=ref_embs,
        meta=json.dumps({
            "corpus_source": corpus_source,
            "max_n": max_n,
            "n": len(ref_texts),
            "dim": int(ref_embs.shape[1]) if ref_embs.ndim == 2 else None,
            "encoder": "crom",
            "top_k": top_k,
            "built_seconds": round(time.perf_counter() - t0, 1),
        }),
    )
    print(f"[kb] build 完成: {path}（{len(ref_texts)} 句, "
          f"{ref_embs.shape[0]}×{ref_embs.shape[1]}, "
          f"{round(path.stat().st_size / 1024 / 1024, 2)} MB, "
          f"{round(time.perf_counter() - t0, 1)}s）")
    return path


def kb_load(kb_path: Path):
    z = np.load(kb_path, allow_pickle=True)
    return {
        "texts": list(z["texts"]),
        "embs": np.asarray(z["embs"], dtype=float),
        "meta": json.loads(str(z["meta"])),
    }


def kb_ensure(cfg, kb_dir=None):
    """KB 缺失时自动构建（友好降级，避免严格崩溃）。返回 (path, kb)。"""
    path = _kb_path(kb_dir)
    if not path.exists():
        print(f"[kb] KB 未构建: {path} → 自动回退 build（train 12000 句，约 20-30s）…")
        kb_build(cfg, out_dir=kb_dir)
    return path, kb_load(path)


def kb_check(cfg, texts, kb_path=None, kb_dir=None, top_k=DEFAULT_TOP_K,
             min_ratio=DEFAULT_MIN_RATIO, aggregate=DEFAULT_AGGREGATE,
             auto_build=True):
    """check：用持久化 KB 对给定文本做检索一致性核验。

    返回逐条 {text, score, conflict, aggregate}。
    auto_build=True 时 KB 缺失自动回退构建。
    """
    if auto_build:
        path, kb = kb_ensure(cfg, kb_dir=kb_dir)
    else:
        path = Path(kb_path) if kb_path else _kb_path(kb_dir)
        if not path.exists():
            raise FileNotFoundError(f"KB 未构建: {path}（先跑 kb build）")
        kb = kb_load(path)
    enc = get_encoder(cfg)
    texts = texts if isinstance(texts, list) else [texts]
    q = enc.encode(texts)
    s = scan_scores(texts, kb["texts"], kb["embs"], enc,
                    q_embs=q, top_k=top_k, min_ratio=min_ratio,
                    progress=False, aggregate=aggregate)
    out = []
    for i, t in enumerate(texts):
        out.append({
            "text": t,
            "score": round(float(s[i]), 4),
            "conflict": bool(s[i] > 0),
            "aggregate": aggregate,
        })
    return out


def report_rows(rows, csv_path=None) -> dict:
    """批量报告：输出 counts（n/conflict/passed）+ 冲突占比；可选写 CSV。返回统计 dict。"""
    n = len(rows)
    n_conflict = sum(1 for r in rows if r["conflict"])
    stats = {
        "n": n,
        "conflict": n_conflict,
        "passed": n - n_conflict,
        "conflict_ratio": round(n_conflict / n, 4) if n else 0.0,
        "scores_mean": round(float(np.mean([r["score"] for r in rows])), 4) if n else 0.0,
    }
    print(json.dumps({"n": n, "conflict": n_conflict, "passed": n - n_conflict,
                      "conflict_ratio": stats["conflict_ratio"],
                      "scores_mean": stats["scores_mean"]}, ensure_ascii=False))
    if csv_path:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["score", "conflict", "text"])
            for r in rows:
                w.writerow([r["score"], "Y" if r["conflict"] else "N", r["text"]])
        print(f"[kb] 冲突清单 CSV 已写: {p}（{n} 行）")
    return stats


def kb_selfcheck(cfg, kb_dir=None, top_k=DEFAULT_TOP_K, min_ratio=DEFAULT_MIN_RATIO,
                 n_pairs=5, seed=42) -> dict:
    """内置自检：从 KB 抽库内原文，篡改一位数字构造篡改对，断言闸门行为不退化。

    通过标准：原文均分 < 0.01（不误伤）；篡改对命中率 ≥ 0.8 且均分 > 0（能抓住）。
    """
    path, kb = kb_ensure(cfg, kb_dir=kb_dir)
    enc = get_encoder(cfg)
    texts = kb["texts"]
    rng = np.random.default_rng(seed)

    originals, tampered = [], []
    for i in rng.choice(len(texts), size=min(n_pairs * 3, len(texts)), replace=False):
        orig = texts[int(i)]
        tam = tamper_number(orig)
        if tam != orig:  # 只保留真的篡改对（原句含数字）
            originals.append(orig)
            tampered.append(tam)
        if len(originals) >= n_pairs:
            break
    if not originals:
        return {"ok": False, "error": "KB 中没有含数字的句子，无法构造篡改对", "n_pairs": 0}

    s_orig = np.asarray([r["score"] for r in kb_check(cfg, originals, kb_dir=kb_dir, top_k=top_k, min_ratio=min_ratio)])
    s_tam = np.asarray([r["score"] for r in kb_check(cfg, tampered, kb_dir=kb_dir, top_k=top_k, min_ratio=min_ratio)])
    hit_ratio = float((s_tam > 0).mean())
    ok = float(s_orig.mean()) < 0.01 and hit_ratio >= 0.8 and float(s_tam.mean()) > 0
    res = {
        "ok": bool(ok),
        "n_pairs": len(originals),
        "originals_mean": round(float(s_orig.mean()), 4),
        "originals_max": round(float(s_orig.max()), 4),
        "tampered_mean": round(float(s_tam.mean()), 4),
        "tampered_hit_ratio": round(hit_ratio, 4),
        "seed": seed,
    }
    verdict = "PASS ✅" if ok else "FAIL ❌"
    print(f"[kb] selfcheck {verdict}（n={res['n_pairs']} 对）：原文均分 {res['originals_mean']}（应<0.01），"
          f"篡改均分 {res['tampered_mean']}（应>0），命中率 {res['tampered_hit_ratio']}（应≥0.8）")
    return res


def kb_smoke(cfg, kb_dir=None, corpus_source="train", max_n=DEFAULT_MAX_N,
             top_k=DEFAULT_TOP_K, min_ratio=DEFAULT_MIN_RATIO,
             smoke_out=None) -> Path:
    """冒烟：build → selfcheck → 伪造样例 check，产出真实性报告（markdown）。"""
    kb_build(cfg, out_dir=kb_dir, corpus_source=corpus_source, max_n=max_n, top_k=top_k)
    self = kb_selfcheck(cfg, kb_dir=kb_dir, top_k=top_k, min_ratio=min_ratio)

    path, kb = kb_ensure(cfg, kb_dir=kb_dir)
    texts = kb["texts"]
    rng = np.random.default_rng(7)
    # 抽 10 条库内原文 + 篡改对，模拟“入库守卫”工作负载
    sample_orig, sample_tam = [], []
    for i in rng.choice(len(texts), size=min(40, len(texts)), replace=False):
        o = texts[int(i)]
        t = tamper_number(o)
        if t != o:
            sample_orig.append(o)
            sample_tam.append(t)
        if len(sample_orig) >= 10:
            break
    rows_orig = kb_check(cfg, sample_orig, kb_dir=kb_dir, top_k=top_k, min_ratio=min_ratio)
    rows_tam = kb_check(cfg, sample_tam, kb_dir=kb_dir, top_k=top_k, min_ratio=min_ratio)
    n_orig_hit = sum(1 for r in rows_orig if r["conflict"])
    n_tam_hit = sum(1 for r in rows_tam if r["conflict"])

    out = Path(smoke_out) if smoke_out else (_kb_dir(cfg) / "authenticity_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "# KB 真实性冒烟报告",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- KB：`{path}`（{kb['meta'].get('n')} 句, dim={kb['meta'].get('dim')}, "
        f"corpus={kb['meta'].get('corpus_source')}）",
        f"- selfcheck：{'PASS ✅' if self['ok'] else 'FAIL ❌'} — 原文均分 {self['originals_mean']} / "
        f"篡改均分 {self['tampered_mean']} / 命中率 {self['tampered_hit_ratio']}",
        f"- 抽样检查：库内原文 {len(rows_orig)} 条误伤 {n_orig_hit} 条（应≈0）；"
        f"篡改对 {len(rows_tam)} 条命中 {n_tam_hit} 条（应≥80%）",
        "",
        "## 结论",
        "",
        "**" + ("闸门功能正常：原文不误伤、篡改精准命中。" if self["ok"] else
                "闸门自检未通过，请检查 KB 语料质量或参数。") + "**",
        "",
    ]
    out.write_text("\n".join(body), encoding="utf-8", newline="\n")
    print(f"[kb] smoke 完成: 原文误伤 {n_orig_hit}/{len(rows_orig)}，篡改命中 {n_tam_hit}/{len(rows_tam)}；"
          f"报告已写 {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="poisonguard.cli kb",
                                description="KB 闭环：build → check → selfcheck → smoke")
    p.add_argument("action", choices=["build", "check", "selfcheck", "smoke"])
    p.add_argument("--kb-dir", default=None)
    p.add_argument("--corpus-source", default="train")
    p.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO)
    p.add_argument("--aggregate", default=DEFAULT_AGGREGATE, choices=["max", "mean", "count"])
    p.add_argument("--text", default=None)
    p.add_argument("--file", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--smoke-out", default=None)
    args = p.parse_args()

    cfg = _load_config()
    if args.action == "build":
        kb_build(cfg, out_dir=args.kb_dir, corpus_source=args.corpus_source,
                 max_n=args.max_n, top_k=args.top_k)
        return
    if args.action == "selfcheck":
        kb_selfcheck(cfg, kb_dir=args.kb_dir, top_k=args.top_k, min_ratio=args.min_ratio)
        return
    if args.action == "smoke":
        kb_smoke(cfg, kb_dir=args.kb_dir, corpus_source=args.corpus_source,
                 max_n=args.max_n, top_k=args.top_k, min_ratio=args.min_ratio,
                 smoke_out=args.smoke_out)
        return
    # check
    if not args.text and not args.file:
        p.error("check 需要 --text 或 --file")
    texts = [args.text] if args.text else \
        [ln.rstrip("\n") for ln in open(args.file, encoding="utf-8") if ln.strip()]
    rows = kb_check(cfg, texts, kb_dir=args.kb_dir, top_k=args.top_k,
                    min_ratio=args.min_ratio, aggregate=args.aggregate)
    report_rows(rows, csv_path=args.csv)


if __name__ == "__main__":
    main()
