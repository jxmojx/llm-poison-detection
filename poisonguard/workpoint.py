"""按客户调优工作点（环节 30-5）：把 target_fpr 参数化 + 输出可供客户决策的报告。

背景：同一个检测器，不同客户对"误报/漏报"的偏好不同——
- 保守客户（如政务/金融）：宁可多拦、不可放过 → target_fpr 取更小（0.001、0.005）；
- 激进客户（如社区/公开内容）：过审量优先，误伤最少 → target_fpr 取更大（0.02、0.05）。

本模块提供：
1. `sweep_workpoints(scorer, clean_texts, fprs)`：在干净的 dev 样本上，
   对一组 target_fpr 逐个标定阈值并按统一 order（FPR）排序，返回工作点表；
2. `workpoint_report(...)`：把工作点表 + 每个点在"实际投毒样本"上的估计召回
   汇总成客户可读的 markdown + csv + json 报告。

纯逻辑、不下载模型：打分器由调用方注入（PoisonScorer/DeployedScorer/假打分器）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .threshold import calibrate_threshold, scorer_scores

DEFAULT_FPRS = [0.001, 0.005, 0.01, 0.02, 0.05]


def sweep_workpoints(scorer, clean_texts: list[str],
                     fprs=None) -> list[dict]:
    """对一组 target_fpr 逐个在干净样本上标定阈值，返回按 FPR 升序的工作点表。

    每个工作点：{target_fpr, threshold, achieved_fpr, n_clean,
                score_mean_clean, score_max_clean}。
    """
    texts = [t for t in clean_texts if isinstance(t, str) and t.strip()]
    pts = []
    for fpr in (fprs or DEFAULT_FPRS):
        try:
            cal = calibrate_threshold(scorer, texts, target_fpr=float(fpr))
        except ValueError:
            continue  # 干净样本为空等 → 跳过该点，留调用方决定
        pts.append({
            "target_fpr": round(float(fpr), 4),
            "threshold": cal["threshold"],
            "achieved_fpr": cal["achieved_fpr"],
            "n_clean": cal["n_clean"],
            "score_mean_clean": cal["score_mean_clean"],
            "score_max_clean": cal["score_max_clean"],
        })
    pts.sort(key=lambda p: p["target_fpr"])
    return pts


def _estimate_recall(scorer, poison_texts: list[str],
                     threshold: float) -> float:
    """在给定投毒样本上估计该工作点的召回（拉网率）：分数>threshold 的占比。"""
    if not poison_texts:
        return float("nan")
    s = scorer_scores(scorer, poison_texts)
    return float((s > threshold).mean())


def workpoint_report(scorer, *, clean_texts: list[str],
                     poison_texts: list[str] | None = None,
                     fprs=None, out_dir: str | Path | None = None,
                     cfg=None, label: str = "default") -> dict:
    """客户工作点报告：sweep + 召回估计，落地 md/csv/json。

    返回 {report_json, report_md?, report_csv?, points, generator, label}。
    """
    pts = sweep_workpoints(scorer, clean_texts, fprs=fprs)
    rows = []
    for p in pts:
        rec = _estimate_recall(scorer, poison_texts or [], p["threshold"])
        rows.append({
            **p,
            "est_recall_on_poison": None if poison_texts is None else round(float(rec), 4),
        })

    if out_dir is None:
        base = Path(cfg.outputs_dir) if cfg is not None else Path("outputs")
        base = base / "workpoints"
    else:
        base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"workpoints_{label}_{stamp}" if label else f"workpoints_{stamp}"

    report_json = base / f"{tag}.json"
    report_json.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "n_clean": len([t for t in clean_texts if isinstance(t, str) and t.strip()]),
        "n_poison": None if poison_texts is None else len(poison_texts),
        "points": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    out = {"report_json": str(report_json), "points": rows,
           "generator": "poisonguard.workpoint", "label": label}

    md = _render_md(rows, label=label, poison_texts=poison_texts)
    report_md = base / f"{tag}.md"
    report_md.write_text(md, encoding="utf-8")
    out["report_md"] = str(report_md)

    csv = _render_csv(rows)
    report_csv = base / f"{tag}.csv"
    report_csv.write_text(csv, encoding="utf-8-sig")
    out["report_csv"] = str(report_csv)

    print(f"[workpoint] 客户工作点报告: {report_md}")
    return out


def _render_md(rows: list[dict], *, label: str,
               poison_texts: list[str] | None) -> str:
    lines = [
        "# 客户工作点报告（target_fpr 参数化）",
        "",
        f"- 工作点标签：{label or 'default'}",
        f"- 干净样本数：{rows[0]['n_clean'] if rows else 0}",
        f"- 投毒样本数：{None if poison_texts is None else len(poison_texts)}",
        "",
        "| target_fpr | 阈值 | 实测 FPR | 干净均值分 | 干净最高分 | 毒样本召回 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        rec = r.get("est_recall_on_poison")
        rec_s = "--" if rec is None else f"{rec:.1%}"
        lines.append(
            f"| {r['target_fpr']:.3%} | {r['threshold']:.4f} | "
            f"{r['achieved_fpr']:.3%} | {r['score_mean_clean']:.4f} | "
            f"{r['score_max_clean']:.4f} | {rec_s} |")
    lines += [
        "",
        "说明：threshold 越小拦得越严（FPR 越低、召回越高）；",
        "选择高 FPR 工作点会放走更多中风险样本，需结合合规口径授权。",
    ]
    return "\n".join(lines) + "\n"


def _render_csv(rows: list[dict]) -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=[
        "target_fpr", "threshold", "achieved_fpr", "n_clean",
        "score_mean_clean", "score_max_clean", "est_recall_on_poison"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()
