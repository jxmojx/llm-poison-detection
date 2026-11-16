# -*- coding: utf-8 -*-
"""可解释性（Gap 2）：给 guard/告警结果附加人-机可读的解释与溯源字段。

纯逻辑模块：不加载模型。所有函数对 guard 结果 dict / ingest 告警记录做纯变换；
磁盘读写只发生在 read_alert / list_alerts（读 quarantine/review 下的 jsonl）。

给 MCP 消费：
  - attach_explanation(result)  guard_text/guard_batch 默认给每条结果附 explanation
  - build_explanation(result)   结构化解释（reason / signals / action / trace / threshold）
  - read_alert(key)             按 key 读一条告警详情（含溯源 reason 与 views）
  - list_alerts(...)            列告警（分页/flag 过滤）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VERDICT_CN = {"allow": "放行", "watch": "观察", "block": "拦截"}
FLAG_CN = {"BLOCK": "拦截", "WATCH": "观察", "PASS": "放行"}
SIGNAL_CN = {
    "rule_injection": "显式注入指令词（L0 规则）",
    "corpus_poison": "语料投毒检测器（L1）",
    "corpus_error": "语料检测器不可用",
    "modal_injection": "注入头推断（P3/L2）",
}
ACTION_HINT = {
    "allow": "无需处理，正常使用。",
    "watch": "建议人工复核或降权后使用。",
    "block": "已拦截：禁止注入模型/工具链，转入人审队列。",
}
_ALERT_DIRS = {"BLOCK": "quarantine", "WATCH": "review", "PASS": "released"}


def _get(d: dict, key: str, default: Any = None) -> Any:
    return d.get(key, default)


def signal_name_cn(name: str) -> str:
    """信号来源名 → 中文可读名（未知名原样返回）。"""
    return SIGNAL_CN.get(name, name)


def signal_explain(sig: dict) -> str:
    """一条信号 → 一句话解释。形如：
    '语料投毒检测器（L1） score=0.87（threshold=0.5）'。"""
    name = str(_get(sig, "name", "?"))
    score = _get(sig, "score")
    detail = str(_get(sig, "detail", "") or "")
    base = f"{signal_name_cn(name)} score={score}"
    if detail:
        base += f"（{detail}）"
    return base


def reason_from_result(r: dict) -> str:
    """把 verdict 拆成一句人能读的因果链：命中哪些信号 + 生效阈值。"""
    sigs = r.get("signals") or []
    lines = []
    if not sigs:
        lines.append("未命中任何信号，接近正常。")
    for s in sigs:
        lines.append(signal_explain(s))
    text = "；".join(lines)
    eff_watch = _get(r.get("explain") or {}, "effective_watch")
    eff_block = _get(r.get("explain") or {}, "effective_block")
    if r.get("verdict") != "allow" and (eff_watch is not None or eff_block is not None):
        text += f"（watch 阈值 {eff_watch}，block 阈值 {eff_block}）"
    return text


def build_explanation(r: dict) -> dict:
    """结构化解释：verdict / reason / signals / action / trace / threshold。"""
    thr = r.get("explain") or {}
    signal_list = r.get("signals") or []
    return {
        "verdict": r.get("verdict"),
        "verdict_cn": VERDICT_CN.get(r.get("verdict"), r.get("verdict")),
        "reason": reason_from_result(r),
        "signals": [signal_explain(s) for s in signal_list],
        "action": ACTION_HINT.get(r.get("verdict"), ""),
        "trace": [
            {"name": s.get("name"), "score": s.get("score"),
             "detail": s.get("detail") or ""}
            for s in signal_list
        ],
        "threshold": {
            "watch": thr.get("effective_watch"),
            "block": thr.get("effective_block"),
            "headroom": thr.get("block_headroom"),
            "source": thr.get("threshold_source"),
            "target_fpr": thr.get("target_fpr"),
        },
        "sanitized": thr.get("sanitized"),
    }


def attach_explanation(r: dict) -> dict:
    """给一条 guard 结果附上 explanation 字段（不原地修改）。"""
    out = dict(r)
    out["explanation"] = build_explanation(r)
    return out


# ----------------------------------------------------------------------
# 告警读取（read_alert）：读 ingest 落盘的 jsonl（与 review.load_items 同源）
# ----------------------------------------------------------------------
def _family_key(rec: dict) -> str:
    return f"{rec.get('source', '')}|{rec.get('row_id', -1)}|{rec.get('text_sha', '')}"


def list_alerts(out_dir: str = "outputs/ingest",
                flags: tuple[str, ...] = ("BLOCK", "WATCH"),
                limit: int = 20, offset: int = 0,
                key: str | None = None) -> list[dict]:
    """列出告警记录（默认只读 BLOCK/WATCH 两档），按分数降序。"""
    out = Path(out_dir)
    recs: list[dict] = []
    for flag in flags:
        sub = _ALERT_DIRS.get(flag)
        d = out / sub if sub else None
        if d is None or not d.exists():
            continue
        for f in sorted(d.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec.setdefault("flag", flag)
                rec["key"] = _family_key(rec)
                if key is not None and rec["key"] != key:
                    continue
                recs.append(rec)
    recs.sort(key=lambda r: float(r.get("score", 0.0) or 0.0), reverse=True)
    if offset:
        recs = recs[offset:]
    if limit and limit > 0:
        recs = recs[:limit]
    return recs


def explain_alert(rec: dict) -> dict:
    """给一条告警记录（ingest jsonl 行）附加解释与溯源。"""
    flag = rec.get("flag", "")
    score = rec.get("score")
    views = rec.get("views") or {}
    src = rec.get("source", "")
    row = rec.get("row_id")
    lines = []
    if src or row is not None:
        lines.append(f"来源 {src} 第 {row} 行")
    if score is not None:
        lines.append(f"综合可疑度 score={round(float(score), 4)}")
    if flag:
        lines.append(f"三档= {FLAG_CN.get(flag, flag)}")
    if isinstance(views, dict) and views:
        parts = []
        for k, v in views.items():
            if v is None or k == "chars":
                continue
            if isinstance(v, (int, float)):
                parts.append(f"{k}={round(float(v), 3)}")
            else:
                parts.append(f"{k}={v}")
        if parts:
            lines.append("视图：" + "，".join(parts) + "（溯源）")
    return {
        "key": _family_key(rec),
        "flag": flag,
        "flag_cn": FLAG_CN.get(flag, flag),
        "score": score,
        "source": src,
        "row_id": row,
        "reason": "；".join(lines),
        "views": views,
        "text": rec.get("text"),
    }


def read_alert(key: str, out_dir: str = "outputs/ingest",
               flags: tuple[str, ...] = ("BLOCK", "WATCH")) -> dict | None:
    """按 key 读一条告警详情；不存在返回 None。"""
    recs = list_alerts(out_dir, flags=flags, key=key, limit=1)
    if not recs:
        return None
    return explain_alert(recs[0])
