"""C4 误报闭环：把 `review` 里人的判决变成可执行的运维动作。

三件事，对应三个真实的运维问题：

1. **误报率到底是多少**？→ 用「被判 BLOCK/WATCH 且人审为 benign」的比例做**在线 FPR 的实测值**，
   和 `target_fpr` 比；样本少时不硬下结论，用 Wilson 区间说"证据够不够"；
2. **阈值该不该动**？→ 把误报样本当作新的干净样本并入校准集，重算 1% 分位阈值，产出**影子阈值**，
   再回答"如果当初就用新阈值，这批里还有几条会被拦"。**只给建议，不自动生效**——
   改阈值是有风险的运维动作，必须人看着数字按下去；
3. **下一次训练该喂什么**？→ 把误报样本导出成难负样本语料（jsonl），把确认有毒的导出成正例，
   直接进 `--hardneg` / 重训 / 评测集。

一个必须写清楚的统计学约束：**这里只能估「拦截精度」，估不了「漏报率」**。
人只审被拦下来的样本，系统放过的样本没人看——这是选择偏差（selection bias）。
要估漏报只能对 PASS 队列随机抽样同样送审，这属于流程改造，不在本模块里假装做到。
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from .detectors import Calibration

_WATCH_FLAGS = ("BLOCK", "WATCH")


# ----------------------------------------------------------------------
# 读判决
# ----------------------------------------------------------------------
def load_decisions(db_path: str | Path) -> list[dict]:
    """读 review 的 decisions 表；库不存在时给出可执行的提示，而不是抛 null。"""
    import sqlite3

    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(
            f"找不到判决库 {p}。先跑 `python -m poisonguard.cli review --out-dir "
            f"{p.parent}` 判几条，再回来跑 feedback。")
    with sqlite3.connect(p) as c:
        c.row_factory = sqlite3.Row
        rows = [dict(r) for r in c.execute("SELECT * FROM decisions ORDER BY decided_at")]
    return rows


def _norm_text(rec: dict) -> str:
    return str(rec.get("text") or rec.get("note") or "").strip()


# ----------------------------------------------------------------------
# 三份语料
# ----------------------------------------------------------------------
def false_positives(records: list[dict], *, min_score: float = 0.0) -> list[dict]:
    """人审判为 benign 的样本 = 检测器的误报（也是最有价值的难负样本）。"""
    out, seen = [], set()
    for r in records:
        if r.get("decision") != "benign":
            continue
        if float(r.get("score") or 0.0) < min_score:
            continue
        if r.get("flag") not in _WATCH_FLAGS:
            continue
        k = r.get("text_sha") or _norm_text(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def confirmed_positives(records: list[dict]) -> list[dict]:
    """人审判为 poison 的样本 = 确认的投毒正例（补评测集/训练集）。"""
    out, seen = [], set()
    for r in records:
        if r.get("decision") != "poison":
            continue
        k = r.get("text_sha") or _norm_text(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def export_corpus(items: list[dict], path: str | Path, *, kind: str) -> Path:
    """导出 jsonl：难负样本 / 正例。文本可能为空（台账只存了 sha），如实写出。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in items:
            f.write(json.dumps({
                "text": r.get("text", ""),
                "label": 0 if kind == "hardneg" else 1,
                "score": r.get("score"),
                "flag": r.get("flag"),
                "reason": r.get("reason", ""),
                "source": r.get("source", ""),
                "row_id": r.get("row_id"),
                "text_sha": r.get("text_sha", ""),
                "model": r.get("model", ""),
                "decided_at": r.get("decided_at"),
            }, ensure_ascii=False) + "\n")
    return p


# ----------------------------------------------------------------------
# 误报率：Wilson 区间 + 告警
# ----------------------------------------------------------------------
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """比例的 Wilson 置信区间：小样本下比正态近似靠谱得多。"""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def fp_rate(records: list[dict]) -> dict:
    """拦截精度口径：拦下来的里面有多少是误报。"""
    blocked = [r for r in records if r.get("flag") in _WATCH_FLAGS]
    fp = [r for r in blocked if r.get("decision") == "benign"]
    k, n = len(fp), len(blocked)
    lo, hi = wilson_interval(k, n)
    # P0-3：放行流的抽样审计 —— 这是唯一能估「漏报率」的数
    passed = [r for r in records if r.get("flag") == "PASS"]
    miss = [r for r in passed if r.get("decision") == "poison"]
    mlo, mhi = wilson_interval(len(miss), len(passed))
    return {"n_blocked": n, "n_fp": k,
            "fp_rate": (k / n) if n else 0.0,
            "ci95": [lo, hi],
            "n_poison": sum(1 for r in blocked if r.get("decision") == "poison"),
            "n_unsure": sum(1 for r in blocked if r.get("decision") == "unsure"),
            "n_pending": sum(1 for r in blocked if not r.get("decision")),
            "pass_reviewed": len(passed),
            "n_miss": len(miss),
            "miss_rate": (len(miss) / len(passed)) if passed else None,
            "miss_rate_ci95": [mlo, mhi] if passed else None}


def drift_alarm(rate: dict, target_fpr: float, *, min_support: int = 30) -> dict:
    """告警规则：**Wilson **下界** 超过目标 FPR 才算证据充分**。

    否则 30 条里误报 3 条（10%）就会天天报警，而真实 FPR 可能仍是 1%。
    """
    lo, hi = rate["ci95"]
    n = rate["n_blocked"]
    if n < min_support:
        return {"status": "insufficient", "alarm": False,
                "message": f"样本不足（{n}/{min_support}），不下结论"}
    if lo > target_fpr:
        return {"status": "drifting", "alarm": True,
                "message": f"误报率 {rate['fp_rate']:.2%}，95% 区间下界 {lo:.2%} 已高于目标 {target_fpr:.2%}"}
    if rate["fp_rate"] > target_fpr:
        return {"status": "watch", "alarm": False,
                "message": f"点估计 {rate['fp_rate']:.2%} 高于目标，但区间 [{lo:.2%}, {hi:.2%}] 覆盖目标值"}
    return {"status": "ok", "alarm": False,
            "message": f"误报率 {rate['fp_rate']:.2%}，区间 [{lo:.2%}, {hi:.2%}] 在目标内"}


def daily_stats(records: list[dict]) -> list[dict]:
    """按日聚合：哪天误报集中，一眼能看出来。"""
    buckets: dict[str, dict] = {}
    for r in records:
        ts = float(r.get("decided_at") or 0.0)
        day = time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "unknown"
        b = buckets.setdefault(day, {"day": day, "n_blocked": 0, "n_fp": 0, "n_poison": 0})
        if r.get("flag") in _WATCH_FLAGS:
            b["n_blocked"] += 1
            if r.get("decision") == "benign":
                b["n_fp"] += 1
            elif r.get("decision") == "poison":
                b["n_poison"] += 1
    out = []
    for b in sorted(buckets.values(), key=lambda x: x["day"]):
        b["fp_rate"] = b["n_fp"] / b["n_blocked"] if b["n_blocked"] else 0.0
        out.append(b)
    return out


# ----------------------------------------------------------------------
# 影子阈值
# ----------------------------------------------------------------------
def shadow_threshold(clean_scores: np.ndarray, fp_scores: np.ndarray,
                     target_fpr: float, *, mode: str = "pooled",
                     min_support: int = 10) -> dict:
    """用误报样本重算 1% 分位阈值——只算，不生效。

    mode="pooled"：并进干净校准集（误报是"看起来像毒"的干净样本，理应抬阈值）；
    mode="replace"：直接用误报样本当校准集（更激进，样本少时退回 pooled）。
    """
    clean = np.asarray(clean_scores, dtype=np.float64)
    fp = np.asarray(fp_scores, dtype=np.float64)
    if len(clean) == 0:
        raise ValueError("干净校准集为空，无法重算阈值")
    old = Calibration.fit(clean, target_fpr)
    if mode not in ("pooled", "replace"):
        raise ValueError(f"mode 必须是 pooled/replace，收到 {mode!r}")
    if mode == "replace" and len(fp) >= min_support:
        ref, used = fp, "fp_only"
    else:
        ref, used = (np.concatenate([clean, fp]) if len(fp) else clean), "pooled"
    new = Calibration.fit(ref, target_fpr)
    ret = {
        "mode": mode, "ref_used": used,
        "target_fpr": target_fpr,
        "n_clean": int(len(clean)), "n_fp": int(len(fp)),
        "old_threshold": old.raw_threshold, "new_threshold": new.raw_threshold,
        "delta": new.raw_threshold - old.raw_threshold,
        "achieved_fpr_old": old.achieved_fpr, "achieved_fpr_new": new.achieved_fpr,
        # 阈值换掉之后，这批误报里还有多少会被继续拦下（越低越好）
        "fp_still_blocked": int((fp > new.raw_threshold).sum()) if len(fp) else 0,
        "fp_blocked_before": int((fp > old.raw_threshold).sum()) if len(fp) else 0,
    }
    # 抬阈值也救不了这些误报 → 不是阈值问题，是特征/分布问题，只能靠重训
    if ret['fp_still_blocked'] == ret['fp_blocked_before']:
        ret["interpretation"] = ("这批误报的分数远高于任何合理阈值：抬阈值救不了它们，"
                                 "属于特征层面的域不匹配，应作为难负样本重训/重校，而不是继续调阈值。")
    else:
        ret["interpretation"] = "换阈值后部分误报会被放过；是否切换仍需看 hold-out 上的召回代价。"
    return ret


def apply_shadow(records: list[dict], shift: dict) -> dict:
    """如果换用影子阈值，这批已审样本的判定会怎么变（离线回放）。

    以**线上阈值**（记录里写的 threshold）为基准，而不是"上一个校准阈值"——
    否则线上口径与校准口径不一致时，回放会得出"什么都没变"的假象。
    """
    thr_old = shift.get("online_threshold")
    if thr_old is None:
        thr_old = shift["old_threshold"]
    thr_new = shift["new_threshold"]
    changed = 0
    by_dec: dict[str, int] = {}
    for r in records:
        if r.get("flag") not in _WATCH_FLAGS:
            continue
        s = float(r.get("score") or 0.0)
        if (s > thr_old) and not (s > thr_new):
            changed += 1
            d = str(r.get("decision") or "pending")
            by_dec[d] = by_dec.get(d, 0) + 1
    return {"n_reviewed": len(records), "n_flag_changes": changed,
            "dropped_by_decision": by_dec,
            "from_threshold": thr_old, "to_threshold": thr_new,
            "note": "仅为离线回放，未改动线上任何配置"}


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def build_report(result: dict) -> str:
    rate = result["fp_rate"]
    alarm = result["alarm"]
    shift = result.get("shadow") or {}
    lines = [
        "# 误报闭环日报",
        "",
        f"> 生成时间 {time.strftime('%Y-%m-%d %H:%M')}；判决库 `{result['db']}`；"
        f"目标 FPR {result['target_fpr']:.2%}",
        "",
        "## 1. 拦截精度（人审判过的样本）",
        "",
        f"- 已判 **{result['n_decided']}** 条：确认有毒 {rate['n_poison']}、误报 {rate['n_fp']}、"
        f"不确定 {rate['n_unsure']}、待判 {rate['n_pending']}",
        f"- **误报率 {rate['fp_rate']:.2%}**（{rate['n_fp']}/{rate['n_blocked']}），"
        f"95% Wilson 区间 [{rate['ci95'][0]:.2%}, {rate['ci95'][1]:.2%}]",
        f"- 告警状态：**{alarm['status']}** —— {alarm['message']}",
        "",
        "> 注意口径：这里只统计**被拦下来**的样本，是「拦截精度」；"
        "用 `review --sample-pass N` 抽放行的样送审，才能估「漏报率」。",
        "",
    ]
    if rate.get("pass_reviewed"):
        lo3, hi3 = rate["miss_rate_ci95"]
        lines += [
            f"## 1b. 漏报估计（PASS 抽样审计，n={rate['pass_reviewed']}）",
            "",
            f"- 抽审放行样本 **{rate['pass_reviewed']}** 条，其中人审判为有毒 **{rate['n_miss']}** 条"
            f" → 漏报率 **{rate['miss_rate']:.2%}**，95% Wilson 区间 [{lo3:.2%}, {hi3:.2%}]",
            "- 这是**抽样估计**（不是全量），区间宽度取决于抽样条数；样本 <30 时只当趋势看。",
            "",
        ]
    if shift:
        lines += [
            "## 2. 影子阈值（只建议，不生效）",
            "",
            f"- 阈值尺度：`{result.get('score_source') or 'dev 分数缓存'}`"
            f"（必须与线上 score 同尺度，否则阈值不可比）",
            f"- 模式 `{shift['mode']}`（校准集 {shift['ref_used']}）：干净 {shift['n_clean']} 条 "
            f"+ **误报样本（按文本去重后）{shift['n_fp']} 条**",
            "- 校准阈值 **{:.4f} → {:.4f}**（{:+.4f}）；线上阈值 {}".format(
                shift["old_threshold"], shift["new_threshold"], shift["delta"],
                (f"**{shift['online_threshold']:.4f}**"
                 if shift.get("online_threshold") is not None else "未记录")),
            "- 这批误报：线上阈值下拦 {} 条 → 校准阈值下拦 **{}** 条".format(
                shift.get("fp_blocked_online"), shift["fp_still_blocked"]),
            "- 离线回放（{} → {}）：**{}** 条从拦截变放过{}".format(
                round(result["replay"].get("from_threshold", 0), 4),
                round(result["replay"].get("to_threshold", 0), 4),
                result["replay"]["n_flag_changes"],
                ("，其中确认有毒 **{}** 条 ← 这就是召回代价".format(
                    result["replay"]["dropped_by_decision"]["poison"])
                 if result["replay"].get("dropped_by_decision", {}).get("poison") else "")),
            f"- {result['replay']['note']}",
            f"- 解读：{shift.get('interpretation', '')}",
            "",
        ]
    if result.get("daily"):
        lines += ["## 3. 按日分布", "", "| 日期 | 拦截 | 误报 | 确认有毒 | 误报率 |",
                  "| :-- | --: | --: | --: | --: |"]
        lines += [f"| {d['day']} | {d['n_blocked']} | {d['n_fp']} | {d['n_poison']} | "
                  f"{d['fp_rate']:.1%} |" for d in result["daily"]]
        lines += [""]
    if result.get("corpora"):
        lines += ["## 4. 导出的语料", ""]
        lines += [f"- `{k}`：{v['n']} 条 → `{v['path']}`" for k, v in result["corpora"].items()]
        lines += ["", "> 难负样本直接进 `--hardneg`；确认正例进评测集。两者都是下一轮重训/重校的输入。", ""]
    return "\n".join(lines)


def run_feedback(db_path: str | Path, *, out_path: str | Path | None = None,
                 clean_scores: np.ndarray | None = None,
                 score_source: str | None = None, target_fpr: float = 0.01,
                 mode: str = "pooled", min_support: int = 30,
                 corpus_dir: str | Path | None = None,
                 report_path: str | Path | None = None) -> dict:
    """主流程：读判决 → 出误报率 → 算影子阈值 → 导语料 → 写报告。"""
    records = load_decisions(db_path)
    decided = [r for r in records if r.get("decision")]
    fps = false_positives(records)
    pos = confirmed_positives(records)
    rate = fp_rate(records)
    alarm = drift_alarm(rate, target_fpr, min_support=min_support)

    result: dict = {
        "db": str(db_path), "target_fpr": target_fpr, "score_source": score_source,
        "n_records": len(records), "n_decided": len(decided),
        "fp_rate": rate, "alarm": alarm, "daily": daily_stats(decided),
        "corpora": {}, "shadow": None, "replay": None,
    }

    if corpus_dir is not None:
        cdir = Path(corpus_dir)
        if fps:
            p = export_corpus(fps, cdir / "hardneg_feedback.jsonl", kind="hardneg")
            result["corpora"]["hardneg"] = {"n": len(fps), "path": str(p)}
        if pos:
            p = export_corpus(pos, cdir / "confirmed_poison.jsonl", kind="positive")
            result["corpora"]["confirmed_poison"] = {"n": len(pos), "path": str(p)}

    if clean_scores is not None and fps:
        fp_scores = np.array([float(r.get("score") or 0.0) for r in fps], dtype=np.float64)
        shift = shadow_threshold(np.asarray(clean_scores, dtype=np.float64), fp_scores,
                                 target_fpr, mode=mode, min_support=min(len(fps), 10))
        # 线上实际在用的阈值（ingest 写进记录的 threshold），与校准阈值可能不是一个数
        online = [float(r.get("threshold") or 0.0) for r in decided
                  if float(r.get("threshold") or 0.0) > 0]
        online_thr = float(np.median(online)) if online else None
        shift["online_threshold"] = online_thr
        shift["fp_blocked_online"] = (int((fp_scores > online_thr).sum())
                                      if online_thr is not None else None)
        shift["threshold_gap"] = (None if online_thr is None
                                  else shift["new_threshold"] - online_thr)
        # 重新给解读：拿"线上阈值"比，比拿"校准旧阈值"比更有意义
        if shift["fp_blocked_online"] == 0:
            shift["interpretation"] = (
                "这批误报在线上阈值下本不该被拦——说明线上阈值比校准阈值低得多，"
                "先把阈值口径对齐（或直接换成校准阈值），再谈重训。")
        elif shift["fp_still_blocked"] >= shift["fp_blocked_online"]:
            shift["interpretation"] = (
                "抬阈值救不了它们：分数远高于任何合理阈值，属于特征层面的域不匹配，"
                "应作为难负样本重训/重校，而不是继续调阈值。")
        else:
            shift["interpretation"] = (
                f"换成校准阈值后，{shift['fp_blocked_online']} 条误报里有 "
                f"{shift['fp_blocked_online'] - shift['fp_still_blocked']} 条会被放过；"
                "是否切换要看 hold-out 上的召回代价（低召回本来就紧）。")
        result["shadow"] = shift
        result["replay"] = apply_shadow(decided, shift)

    if out_path is not None:
        from .evaluate import save_json

        save_json(result, out_path)
    if report_path is not None:
        p = Path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(build_report(result), encoding="utf-8", newline="\n")
        result["report"] = str(p)
    return result
