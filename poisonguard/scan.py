# -*- coding: utf-8 -*-
"""批扫产品：任意文本文件 → 逐条分数/视图 + TopK 判官复核 + 人审报告。

对应落地方向（docs/落地方向.md）：
  ① 入库数据安全网关 —— 把待入库文本跑一遍，出 BLOCK/WATCH/PASS 分档 + 每例归因视图；
  ② RAG 知识库语料审核 —— 学生模型毫秒级粗排 → TopK 送 LLM 判官精判 → 人审清单。

用法（先有某个领域口径，见 README「换领域落地」；产物后缀自动取真实语料文件）：
  python -m poisonguard.cli scan --clean-source tnews --real-file real_afqmc.txt \
      --input data/raw/real_tnews.txt --judge-topk 20

产物：
  {prefix}_report.md    人审清单（Markdown）
  {prefix}_report.html  人审清单（自包含 HTML，可直接开）
  {prefix}_scores.csv   逐条分数 + 视图分解（可回连业务行 id）
  {prefix}.json         机器可读汇总

边界与诚实口径：
  * 分数来自「对某领域训练好的检测器 + 蒸馏学生」，域外输入不能外推——
    报告会打印输入与训练侧 dev 的平均长度差作为「域漂移」提示；
  * 判官复核是可选且按预算的（0.8~1.4s/条），默认不开；
  * 没有给「最终裁决」，只给信号与证据，阈值/放行的业务决策留给使用方。
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import data as D
from .config import Config, load_config
from .service import _default_student, build_scorer

BATCH = 512
_WATCH_BAND = 0.10      # 阈值下方 0.10 视为 WATCH（软警告带）
_QUEUE_TOP = 50         # 人审队列至少列 Top N（无论是否越阈值）


def _load_judge_threshold(cfg: Config) -> float:
    """从当前域产出的 judge{_suffix}.json 读取判官运营阈值（dev 校准）。

    判官分数是 logprob，跨域阈值差异大（afqmc 0.88 / iflytek 0.84 / tnews 0.97），
    不能写死 0.5。按 domain 名（processed_<域>）探测，兜底取第一个 judge_*.json。
    """
    domain = Path(cfg.data.processed_dir).name
    tag = domain[len("processed"):] if domain.startswith("processed") else domain
    tag = tag.lstrip("_")   # processed_afqmc -> afqmc（judge_afqmc.json），勿留前导下划线
    mdir = Path(cfg.metrics_dir)
    for f in (mdir / f"judge_{tag}.json",):
        if f.exists():
            try:
                det = json.loads(f.read_text(encoding="utf-8")).get("detectors", {})
                for name, vals in det.items():
                    if "threshold" in vals:
                        return float(vals["threshold"])
            except Exception:
                pass
    # 兜底：任意 judge_*.json 的第一个阈值
    for f in sorted(mdir.glob("judge_*.json")):
        try:
            det = json.loads(f.read_text(encoding="utf-8")).get("detectors", {})
            for vals in det.values():
                if isinstance(vals, dict) and "threshold" in vals:
                    return float(vals["threshold"])
        except Exception:
            continue
    return 0.5
_HTML_CSS = """
body{font-family:-apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
     margin:2.2rem auto;max-width:1080px;color:#1f2430;line-height:1.6;padding:0 1rem}
h1{font-size:1.5rem;border-bottom:2px solid #2b6cb0;padding-bottom:.4rem}
h2{font-size:1.15rem;margin-top:2rem;color:#2b6cb0}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.6rem 0}
th,td{border:1px solid #d8dee6;padding:.45rem .55rem;text-align:left;vertical-align:top}
th{background:#f0f4f8;font-weight:600}
tr.block td{background:#fdecec}
tr.watch td{background:#fff7e0}
.kpi{display:flex;gap:1.2rem;flex-wrap:wrap;margin:1rem 0}
.k{flex:1;min-width:160px;background:#f7f9fb;border:1px solid #e3e8ef;border-radius:8px;
   padding:.8rem 1rem}
.k b{display:block;font-size:1.5rem;color:#1f415d}
.k span{font-size:.8rem;color:#5a6a7e}
.note{color:#8a97a8;font-size:.84rem}
code{background:#f0f2f5;padding:.08rem .28rem;border-radius:4px;font-size:.84rem}
.warn{border-left:4px solid #e2a336;background:#fff7e0;padding:.6rem .9rem;border-radius:4px}
"""


# ----------------------------------------------------------------------
# 读取输入文件（txt / csv / jsonl，自动推断或显式 --format）
# ----------------------------------------------------------------------
def load_rows(path: Path, fmt: str = "auto", column: str = "text",
              limit: int = 0) -> list[dict]:
    """返回 [{row_id, text}]。row_id 是该行在源文件中的序号（csv 跳过表头）。
    blank / 非字符串行会被过滤，但 row_id 保持原序号（可回连业务行）。"""
    fmt = fmt or "auto"
    if fmt == "auto":
        suffix = path.suffix.lower()
        fmt = {".csv": "csv", ".tsv": "csv", ".jsonl": "jsonl",
               ".json": "jsonl"}.get(suffix, "txt")
    rows: list[dict] = []

    if fmt == "txt":
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            t = line.strip()
            if t:
                rows.append({"row_id": i, "text": t})
    elif fmt == "csv":
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, encoding="utf-8", newline="") as f:
            for i, rec in enumerate(csv.DictReader(f, delimiter=sep)):
                t = (rec.get(column) or "").strip()
                if t:
                    rows.append({"row_id": i + 1, "text": t, "extra": dict(rec)})
    elif fmt == "jsonl":
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get(column) or rec.get("text") or rec.get("sentence")
            if isinstance(t, str) and t.strip():
                rows.append({"row_id": i, "text": t.strip(), "extra": rec})
    else:  # pragma: no cover
        raise ValueError(f"不支持的格式: {fmt}")

    if limit and limit > 0:
        rows = rows[:limit]
    return rows


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def scan_file(cfg: Config, input_path: str | Path, *,
              out_prefix: str | Path | None = None,
              threshold: float | None = None,
              judge_topk: int = 0,
              fmt: str = "auto", column: str = "text",
              limit: int = 0, render_html: bool = True,
              scorer=None) -> dict:
    """对 input 文件逐条打分 + 可选判官复核，产出报告/CSV/JSON。

    scorer 可注入已装配好的评分器（deploy.run 场景：来自 .pkg 的 DeployedScorer），
    此时跳过内部 build_scorer，避免在无训练数据/无域产物的部署机上重装配。
    """
    p_in = Path(input_path)
    if not p_in.exists():
        print(f"[scan] 输入文件不存在: {p_in}")
        sys.exit(2)

    rows = load_rows(p_in, fmt=fmt, column=column, limit=limit)
    if not rows:
        print("[scan] 输入文件无有效文本行")
        sys.exit(2)
    texts = [r["text"] for r in rows]

    # ---- 装配打分器（复用已训练领域口径；自动挂载该域学生模型）----
    if scorer is not None:
        scoring = scorer
    else:
        if threshold is not None:
            cfg.detector.target_fpr = float(threshold)
        scoring = build_scorer(cfg, student_path=_default_student(cfg))
        if threshold is not None:
            # task 阈值：用户显式覆盖；否则用 scoring 的默认（训练侧标定）
            scoring.threshold = float(threshold)

    # 域漂移提示：输入 vs 训练侧 dev 的平均长度差
    dev_lens: list[int] = []
    try:
        dev = D.load_splits(cfg.data.processed_dir)["dev"]
        dev_lens = [len(s.text) for s in dev]
    except Exception:
        dev_lens = []
    in_len = np.array([len(t) for t in texts], dtype=float)
    drift = float(np.mean(in_len) - (np.mean(dev_lens) if dev_lens else np.nan))

    # ---- 批量打分 ----
    t0 = time.perf_counter()
    scores: list[float] = []
    views_all: list[dict] = []
    for i in range(0, len(texts), BATCH):
        r = scoring.score(texts[i:i + BATCH], explain=True)
        for rec in r["results"]:
            scores.append(rec["score"])
            v = rec["views"]
            views_all.append({"knn": v["knn"], "fpr_est": v["fpr_est"],
                              "char_ngram": v["char_ngram"], "student": v["student"],
                              "chars": v["chars"]})
    dt = (time.perf_counter() - t0) * 1000.0
    ms = dt / max(len(scores), 1)

    s = np.asarray(scores, dtype=float)
    thr = float(scoring.threshold)
    # 判官阈值：优先用注入评分器自带的(如 `.pkg` 里打包的域阈值，部署机没有 judge_*.json)，
    # 否则回落到从 metrics/judge_<域>.json 读取(作者侧装配场景)。
    judge_thr = getattr(scoring, "judge_threshold", None)
    if judge_thr is None:
        judge_thr = _load_judge_threshold(cfg)

    # 分档
    flags = []
    for x in s:
        if x > thr:
            flags.append("BLOCK")
        elif x > thr - _WATCH_BAND:
            flags.append("WATCH")
        else:
            flags.append("PASS")

    # 人审队列：按分数降序，优先越阈值者，最多 QUEUE_TOP 条
    order = np.argsort(-s)
    in_block = order[s[order] > thr]
    queue = [int(i) for i in in_block]
    if len(queue) < _QUEUE_TOP:
        rest = [int(i) for i in order if int(i) not in queue]
        queue = queue + rest[:_QUEUE_TOP - len(queue)]

    # ---- 判官复核（可选，按预算）----
    judge_rows: list[dict] = []
    if judge_topk and judge_topk > 0:
        cands = queue[:judge_topk]
        ctexts = [texts[i] for i in cands]
        try:
            from .judge import build_judge
            j = build_judge(cfg, kind="auto", scoring="logprob")
            jprint = j.score(ctexts)
            for i, jv, c in zip(cands, jprint, ctexts):   # type: ignore[arg-type]
                judge_rows.append({"row_id": rows[i]["row_id"], "text": c,
                                   "score": float(s[i]), "student": views_all[i]["student"],
                                   "judge": float(jv)})
        except Exception as e:  # 判官不可用时不阻塞批扫
            print(f"[scan] 判官复核跳过（{type(e).__name__}: {e}）")
            judge_rows = []

    # ---- 汇总 ----
    n_block, n_watch, n_pass = (flags.count(f) for f in ("BLOCK", "WATCH", "PASS"))
    summary = {
        "input": str(p_in),
        "n_rows": len(rows),
        # 注入 .pkg 时用包内领域名（部署机没有 processed_<域> 目录，不能报 processed）
        "domain": getattr(scoring, "domain", None) or Path(cfg.data.processed_dir).name,
        "student": str(getattr(scoring, "student_path", "") or ""),
        "threshold": round(thr, 4),
        "target_fpr": cfg.detector.target_fpr,
        "ms_per_item": round(ms, 4),
        "tiers": {"BLOCK": n_block, "WATCH": n_watch, "PASS": n_pass},
        "mean_len_input": round(float(np.mean(in_len)), 1),
        "mean_len_dev": round(float(np.mean(dev_lens)), 1) if dev_lens else None,
        "len_drift": round(drift, 1),
        "has_student": bool(getattr(scoring, "has_student", None)
                            or getattr(scoring, "student", None)),
        "judge_threshold": round(judge_thr, 4),
        "judge_reviewed": len(judge_rows),
        "provenance": {
            "model": getattr(scoring, "version", "?"),
            "encoder": getattr(scoring, "encoder_name", None) or cfg.encoder.name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    print(f"[scan] {len(rows)} 条 -> BLOCK {n_block} / WATCH {n_watch} / PASS {n_pass}"
          f"  ({ms:.2f} ms/条, domain={summary['domain']})")

    # ---- 落盘 ----
    if out_prefix is None:
        pre = Path("outputs") / "scan" / p_in.stem
    else:
        pre = Path(out_prefix)
    pre.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for i in range(len(rows)):
        rows_out.append({
            "row_id": rows[i]["row_id"], "text": texts[i],
            "score": round(float(s[i]), 6), "flag": flags[i],
            **{f"view_{k}": (None if v is None else round(float(v), 6))
               for k, v in views_all[i].items()},
        })
    with open(pre.with_suffix(".scores.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    md = _render_markdown(summary, rows, texts, s, flags, views_all, queue, judge_rows)
    (pre.with_suffix(".report.md")).write_text(md, encoding="utf-8")
    if render_html:
        (pre.with_suffix(".report.html")).write_text(
            _render_html(summary, rows, texts, s, flags, views_all, queue, judge_rows),
            encoding="utf-8")
    # JSON 里也带上判官复核明细（CSV 只有分数，机器侧要能直接拿联合结论）
    summary["judge_review"] = judge_rows
    (pre.with_suffix(".json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary["outputs"] = {
        "md": str(pre.with_suffix(".report.md")),
        "html": str(pre.with_suffix(".report.html")) if render_html else None,
        "csv": str(pre.with_suffix(".scores.csv")),
        "json": str(pre.with_suffix(".json")),
    }
    print(f"[scan] 报告 {pre.with_suffix('.report.md')} / CSV {pre.with_suffix('.scores.csv')}")
    return summary


# ----------------------------------------------------------------------
# 渲染
# ----------------------------------------------------------------------
def _fmt(x, nd=4) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def _render_markdown(summary, rows, texts, s, flags, views, queue, judge_rows) -> str:
    L = []
    L.append(f"# poisonguard 批扫报告：`{summary['input']}`\n")
    L.append(f"- 领域口径：`{summary['domain']}` ｜ 学生模型："
             f"{'有' if summary['has_student'] else '无(退化为 1-FPR)'}"
             f" ｜ 阈值：{summary['threshold']}"
             f"（目标 FPR {summary['target_fpr']}）")
    L.append(f"- 输入 {summary['n_rows']} 条 ｜ 平均长度 "
             f"{summary['mean_len_input']}（训练侧 dev {summary['mean_len_dev'] or '—'}，"
             f"漂移 {summary['len_drift']:+} 字）")
    L.append(f"- 打分 {summary['ms_per_item']} ms/条"
             f"{' ｜ 判官复核 ' + str(summary['judge_reviewed']) + ' 条（判官阈值 ' + str(summary['judge_threshold']) + '）' if summary['judge_reviewed'] else ''}\n")
    if abs(summary["len_drift"]) > 40 and summary["mean_len_dev"]:
        L.append(f"> ⚠️ 域漂移提示：输入平均长度与训练侧 dev 差 {summary['len_drift']:+.0f} 字。"
                 "分数仅在输入分布接近训练领域时有意义；换域请按 README「换领域落地」重训。\n")
    T = summary["tiers"]
    L.append("## 分档汇总\n")
    L.append(f"| 档位 | 数量 | 含义 |\n|---|---|---|")
    L.append(f"| **BLOCK** | {T['BLOCK']} | 分数 > {summary['threshold']}，模型判定高风险 |")
    L.append(f"| **WATCH** | {T['WATCH']} | 阈值下方 {_WATCH_BAND:.2f} 内，软警告 |")
    L.append(f"| **PASS** | {T['PASS']} | 其余 |\n")

    L.append("## 人审队列（按分数降序）\n")
    L.append("| # | row_id | 分数 | 档位 | student | char_ngram | knn(FPR) | 文本 |")
    L.append("|---|--:|--:|:--:|--:|--:|--:|---|")
    for rank, i in enumerate(queue, 1):
        v = views[i]
        L.append(f"| {rank} | {rows[i]['row_id']} | {s[i]:.4f} | {flags[i]} | "
                 f"{_fmt(v['student'])} | {_fmt(v['char_ngram'])} | {_fmt(v['fpr_est'])} | "
                 f"{texts[i][:48]}{'…' if len(texts[i]) > 48 else ''} |")

    if judge_rows:
        L.append(f"\n## 判官复核（logprob，阈值 {summary['judge_threshold']}，0.8~1.4s/条）\n")
        L.append("| row_id | 学生分 | 判官分 | 联合结论 | 文本 |")
        L.append("|---|--:|--:|:--:|---|")
        for jr in judge_rows:
            joint = _joint_verdict(jr["score"], jr["judge"], summary["threshold"],
                                   summary["judge_threshold"])
            L.append(f"| {jr['row_id']} | {jr['score']:.4f} | {jr['judge']:.4f} | {joint} | "
                     f"{jr['text'][:48]}{'…' if len(jr['text']) > 48 else ''} |")
        L.append("\n结论口径：`确认风险` = 两者同时越界；`判官捕获` = 粗筛放过但判官命中(补漏)；"
                 "`待人工` = 粗筛报警但判官未确认(疑误报)。")
    L.append("\n---\n*本报告由 `poisonguard scan` 生成，分数是信号不是裁决；"
             "阈值与放行策略请结合业务标定。*\n")
    return "\n".join(L)


def _render_html(summary, rows, texts, s, flags, views, queue, judge_rows) -> str:
    drift_warn = ""
    if abs(summary["len_drift"]) > 40 and summary["mean_len_dev"]:
        drift_warn = (f"<div class='warn'>⚠️ 域漂移：输入平均长度与训练侧 dev "
                      f"差 {summary['len_drift']:+.0f} 字。分数仅在输入分布接近训练领域时有意义。</div>")
    T = summary["tiers"]
    rows_html = []
    for rank, i in enumerate(queue, 1):
        v = views[i]
        cls = "block" if flags[i] == "BLOCK" else ("watch" if flags[i] == "WATCH" else "")
        rows_html.append(
            f"<tr class='{cls}'><td>{rank}</td><td>{rows[i]['row_id']}</td>"
            f"<td><b>{s[i]:.4f}</b></td><td>{flags[i]}</td>"
            f"<td>{html.escape(_fmt(v['student']))}</td>"
            f"<td>{html.escape(_fmt(v['char_ngram']))}</td>"
            f"<td>{html.escape(_fmt(v['fpr_est']))}</td>"
            f"<td>{html.escape(texts[i][:60])}{'…' if len(texts[i]) > 60 else ''}</td></tr>")
    judge_html = ""
    if judge_rows:
        rowsj = "".join(
            f"<tr><td>{jr['row_id']}</td><td>{jr['score']:.4f}</td>"
            f"<td>{jr['judge']:.4f}</td>"
            f"<td>{_joint_verdict(jr['score'], jr['judge'], summary['threshold'], summary['judge_threshold'])}</td>"
            f"<td>{html.escape(jr['text'][:60])}{'…' if len(jr['text']) > 60 else ''}</td></tr>"
            for jr in judge_rows)
        judge_html = (f"<h2>判官复核（logprob，阈值 {summary['judge_threshold']}）</h2>"
                      f"<table><tr><th>row_id</th><th>学生分</th><th>判官分</th>"
                      f"<th>联合结论</th><th>文本</th></tr>{rowsj}</table>")
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>poisonguard 批扫报告</title><style>{_HTML_CSS}</style></head><body>
<h1>poisonguard 批扫报告</h1>
<p class="note">输入 <code>{html.escape(summary['input'])}</code> ｜ 领域口径 <code>{html.escape(summary['domain'])}</code>
｜ 阈值 {summary['threshold']}（目标 FPR {summary['target_fpr']}）｜ {summary['ms_per_item']} ms/条
｜ 学生模型：{'有' if summary['has_student'] else '无'}</p>
{drift_warn}
<div class="kpi">
<div class="k"><b>{T['BLOCK']}</b><span>BLOCK（分数 &gt; {summary['threshold']}）</span></div>
<div class="k"><b>{T['WATCH']}</b><span>WATCH（阈值下方 {_WATCH_BAND:.2f}）</span></div>
<div class="k"><b>{T['PASS']}</b><span>PASS</span></div>
<div class="k"><b>{summary['n_rows']}</b><span>总计 {summary['n_rows']} 条</span></div>
</div>
<h2>人审队列（按分数降序）</h2>
<table><tr><th>#</th><th>row_id</th><th>分数</th><th>档位</th><th>student</th>
<th>char_ngram</th><th>knn(FPR)</th><th>文本</th></tr>{''.join(rows_html)}</table>
{judge_html}
<p class="note">本报告由 <code>poisonguard scan</code> 生成，分数是信号不是裁决。</p>
</body></html>"""


def _joint_verdict(s_score: float, j_score: float, thr: float, j_thr: float = 0.5) -> str:
    both = s_score > thr and j_score > j_thr
    judge_only = s_score <= thr and j_score > j_thr
    coarse_only = s_score > thr and j_score <= j_thr
    if both:
        return "确认风险"
    if judge_only:
        return "判官捕获（粗筛放过）"
    if coarse_only:
        return "待人工（粗筛报警，判官未确认）"
    return "低风险"


def cmd_scan(cfg: Config, args: argparse.Namespace) -> dict:
    return scan_file(
        cfg,
        getattr(args, "input", None) or "",
        out_prefix=getattr(args, "out_prefix", None),
        threshold=getattr(args, "threshold", None),
        judge_topk=int(getattr(args, "judge_topk", 0) or 0),
        fmt=getattr(args, "format", "auto"),
        column=getattr(args, "column", "text"),
        limit=int(getattr(args, "limit", 0) or 0),
        render_html=not getattr(args, "no_html", False),
    )


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--judge-topk", type=int, default=0)
    ap.add_argument("--format", default="auto")
    ap.add_argument("--column", default="text")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    scan_file(load_config(), a.input, out_prefix=a.out_prefix, threshold=a.threshold,
              judge_topk=a.judge_topk, fmt=a.format, column=a.column, limit=a.limit)
