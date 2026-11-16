"""合规审计导出（环节 30-3）：把台账里的处理记录变成可审计的报告。

背景：数据清洗的"清洗"本身也要能被审计——客户/合规要能回答三件事：
1. 这个批次处理了什么（文件级：路径、行数、状态、模型版本、门禁阈值）；
2. 每条文本判了什么（行级：text_sha、score、三档 flag、判定时点）；
3. 用的是哪个模型、哪个包、哪个版本（provenance：model / pkg_sha / threshold）。

本模块把 sqlite 台账（ingest.py 的 IngestLedger）导出成三类产物：
- `compliance.json`：结构化全集（文件元数据 + 行级记录 + 汇总），机器可读、可回放；
- `compliance.md`：人读报告（批次表 + 三档汇总 + 分文件明细 + BLOCK 样例前若干条）；
- 可选 `compliance.csv`：行级判决明细（带 text_sha，不含原文——满足"最小必要"审计）。
另外支持任意时间窗导出；导出时会记录"导出于何版本"（self pkg_sha），
导出本身就是一次"审计快照"。

均不下载模型：输入只有台账 + 配置，输出纯文本/JSON。
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .ingest import IngestLedger

_FLAGS_ORDER = ("BLOCK", "WATCH", "PASS")


def _iso(ts: float) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _pkg_sha() -> str:
    """自身包版本指纹：pkg 元数据里取 version + 本文件路径 hash（审计快照来源标记）。"""
    try:
        from importlib.metadata import version
        ver = version("poisonguard")
    except Exception:
        ver = "dev"
    return f"poisonguard@{ver}"


def _aggregate(meta: list[dict], records: list[dict]) -> dict:
    flag_tot = {f: 0 for f in _FLAGS_ORDER}
    status_tot: dict[str, int] = {}
    n_rows_total = 0
    n_records = 0
    for m in meta:
        n_rows_total += int(m.get("n_rows") or 0)
        s = m["status"]
        status_tot[s] = status_tot.get(s, 0) + 1
    for r in records:
        f = r["flag"]
        if f in flag_tot:
            flag_tot[f] += 1
        n_records += 1

    by_file: dict[str, dict] = {}
    for r in records:
        key = r["file_sha"]
        ent = by_file.setdefault(key, {f: 0 for f in _FLAGS_ORDER})
        if r["flag"] in ent:
            ent[r["flag"]] += 1
    # 输出时按 BLOCK 降序排（最需要复核的在前）
    by_file_sorted = [
        {"file_sha": k, **v} for k, v in
        sorted(by_file.items(), key=lambda kv: (-kv[1]["BLOCK"], kv[0]))
    ]
    return {
        "n_files": len(meta),
        "n_rows_total": n_rows_total,
        "n_records": n_records,
        "status_breakdown": status_tot,
        "flag_breakdown": flag_tot,
        "per_file_flag_breakdown": by_file_sorted,
    }


def _render_md(meta: list[dict], records: list[dict], agg: dict,
               *, since: float, until: float, limit_examples: int = 20) -> str:
    lines = [
        "# 合规审计报告（数据清洗台账导出）",
        "",
        f"- 导出时间：{_iso(time.time())}",
        f"- 时间窗：[{_iso(since)}] ~ [{_iso(until) if until < 1e17 else '未设上限'}]",
        f"- 来源：{_pkg_sha()}",
        "",
        "## 1. 总览",
        "",
        f"| 文件数 | 总行数 | 行级记录 | 状态分布 |",
        f"| --- | --- | --- | --- |",
        f"| {agg['n_files']} | {agg['n_rows_total']} | {agg['n_records']} | "
        f"{json.dumps(agg['status_breakdown'], ensure_ascii=False)} |",
        "",
        "## 2. 三档判定汇总（BLOCK / WATCH / PASS）",
        "",
        f"- **BLOCK**：{agg['flag_breakdown']['BLOCK']}",
        f"- **WATCH**：{agg['flag_breakdown']['WATCH']}",
        f"- **PASS**：{agg['flag_breakdown']['PASS']}",
        "",
        "## 3. 分文件明细",
        "",
        "| 文件 | 行数 | 状态 | 模型 | 包版本 | 阈值 | BLOCK | WATCH | PASS |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in meta:
        sha = m["file_sha"][:12]
        blocks = {f: 0 for f in _FLAGS_ORDER}
        # 在 records 里对该文件聚一次
        for r in records:
            if r["file_sha"] == m["file_sha"] and r["flag"] in blocks:
                blocks[r["flag"]] += 1
        lines.append(
            f"| {sha} | {m['n_rows']} | {m['status']} | {m['model']} | "
            f"{m['pkg_sha'][:12]} | {m['threshold']:.4g} | "
            f"{blocks['BLOCK']} | {blocks['WATCH']} | {blocks['PASS']} |"
        )

    lines += ["", "## 4. BLOCK 样例（text_sha，用于复核）", ""]
    n_ex = 0
    for r in records:
        if r["flag"] != "BLOCK":
            continue
        lines.append(f"- `{r['file_sha'][:12]}#{r['row_id']}` "
                     f"text_sha=`{r['text_sha'][:16]}` score={r['score']:.4f} "
                     f"decided_at={_iso(r['decided_at'])}")
        n_ex += 1
        if n_ex >= limit_examples:
            lines.append("  …（已截断，JSON 里有全集）")
            break
    if n_ex == 0:
        lines.append("（该时间窗内无 BLOCK 记录）")
    return "\n".join(lines) + "\n"


def export_compliance(
    ledger_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    since: float = 0.0,
    until: float = 1e18,
    render_md: bool = True,
    render_csv: bool = False,
    limit_examples: int = 20,
    cfg: Config | None = None,
) -> dict:
    """从台账导出合规审计产物。

    参数：
        ledger_path: IngestLedger 的 sqlite 路径。
        out_dir: 输出目录（默认 cfg.outputs_dir/compliance）。
        since/until: 时间窗（epoch 秒；decided_at 落窗内才导出）。
        render_md / render_csv: 是否生成 .md（默认）/.csv（可选）。
        limit_examples: md 报告里 BLOCK 样例最多条数。

    返回：{report_json, report_md?, report_csv?, ledger, n_files, n_records, ...}。
    """
    ledger = IngestLedger(ledger_path)
    meta = ledger.files_meta()
    # until<=0 或 >1e17 表示"无上限"（CLI 约定 0=不限）
    up = until if until and until > 0 and until < 1e17 else 1e18
    records = ledger.records_range(since=since, until=up)
    agg = _aggregate(meta, records)

    base_dir = Path(out_dir) if out_dir is not None else (
        Path(cfg.outputs_dir) / "compliance" if cfg is not None else Path("outputs/compliance")
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = base_dir / f"compliance_{stamp}.json"
    report_json.write_text(
        json.dumps({
            "generated_at": _iso(time.time()),
            "ledger": str(Path(ledger_path)),
            "since": since, "until": until,
            "source": _pkg_sha(),
            "summary": agg,
            "files": meta,
            "records": records,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compliance] 已导出 JSON: {report_json}")

    out = {
        "report_json": str(report_json),
        "ledger": str(Path(ledger_path)),
        "n_files": agg["n_files"],
        "n_records": agg["n_records"],
        "summary": agg,
    }
    if render_md:
        md = _render_md(meta, records, agg, since=since, until=until,
                        limit_examples=limit_examples)
        report_md = base_dir / f"compliance_{stamp}.md"
        report_md.write_text(md, encoding="utf-8")
        out["report_md"] = str(report_md)
        print(f"[compliance] 已导出 MD: {report_md}")
    if render_csv:
        import csv
        report_csv = base_dir / f"compliance_{stamp}.csv"
        with report_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[
                "file_sha", "row_id", "text_sha", "score", "flag", "decided_at"])
            w.writeheader()
            for r in records:
                w.writerow(r)
        out["report_csv"] = str(report_csv)
        print(f"[compliance] 已导出 CSV: {report_csv}")
    return out
