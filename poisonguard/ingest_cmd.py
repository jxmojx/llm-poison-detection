"""`ingest` 子命令的执行体（C2）：装配 scorer → 跑门禁 → 打印台账。

和 `scan` 的区别：scan 是"人给一个文件，我看一遍"，ingest 是"数据一落地就自动判"，
并且把判决写进**可幂等重放的台账**，而不是只产出一份报告。
"""

from __future__ import annotations

import logging

import json
import time
from pathlib import Path

from .config import Config
from .log import event, get_logger

LOG = get_logger("ingest")
from .ingest import Ingestor


def _build_ingest_scorer(cfg: Config, args):
    """两条路：用已签名的 .pkg（部署形态），或就地拟合（开发形态）。"""
    bundle = getattr(args, "bundle", None)
    if bundle:
        from .deploy import _resolve_sign_key, load_bundle

        sc = load_bundle(bundle, cfg,
                         ed25519_pub=getattr(args, "verify_ed25519", None),
                         sign_key=_resolve_sign_key(getattr(args, "sign_key", None),
                                                    getattr(args, "sign_key_file", None)))
        pkg_sha = ""
        try:
            from .ingest import file_sha256

            pkg_sha = file_sha256(bundle)[:16]
        except OSError:
            pass
        return sc, pkg_sha

    from .service import _default_student, build_scorer
    from .threshold import describe, resolve_threshold

    sc = build_scorer(cfg, student_path=_default_student(cfg))
    # P0-1：本地打分器的阈值也要标定，否则和 deploy/run 的口径不一致
    res = resolve_threshold(sc, cfg,
                            override=getattr(args, "threshold", None),
                            calibrate=bool(getattr(args, "calibrate", True)))
    if res["source"] == "calibrated":
        sc.threshold = res["threshold"]
    elif res["source"] == "cli_override":
        sc.threshold = res["threshold"]
    print(f"[ingest] 阈值口径：{describe(res)}")
    return sc, ""


def cmd_ingest(cfg: Config, args) -> dict:
    scorer, pkg_sha = _build_ingest_scorer(cfg, args)
    ing = Ingestor(
        scorer,
        out_dir=getattr(args, "out_dir", "outputs/ingest"),
        ledger_path=Path(getattr(args, "ledger_dir", "outputs/ingest")) / "ledger.sqlite3",
        watch_band=float(getattr(args, "watch_band", 0.05) or 0.0),
        explain=not getattr(args, "no_explain", False),
        pkg_sha=pkg_sha,
        threshold=getattr(args, "threshold", None),
    )
    print(f"[ingest] model={ing.model} threshold={ing.threshold:.4f} "
          f"watch_band={ing.watch_band:g} pkg_sha={pkg_sha or '-'}")

    files = [Path(f) for f in (getattr(args, "files", None) or [])]
    patterns = [p.strip() for p in str(getattr(args, "pattern", "*.txt")).split(",") if p.strip()]
    kwargs = dict(fmt=getattr(args, "fmt", "auto"),
                  column=getattr(args, "column", "text"),
                  limit=int(getattr(args, "limit", 0) or 0))

    def _report(rows: list[dict]) -> None:
        for r in rows:
            if r["status"] == "ok":
                print(f"[ingest] {r['path']} -> {r['n']} 条 "
                      f"BLOCK {r['block']} / WATCH {r['watch']} / PASS {r['pass']} "
                      f"（{r['seconds']}s, 新入账 {r.get('records_added', 0)}）")
                event(LOG, "ingest.done", path=r["path"], n=r["n"],
                      block=r["block"], watch=r["watch"], passed=r["pass"],
                      ms=int(r["seconds"] * 1000))
            elif r["status"] == "skipped":
                print(f"[ingest] {r['path']} -> 跳过（{r['reason']}）")
                event(LOG, "ingest.skipped", path=r["path"], reason=r["reason"])
            else:
                print(f"[ingest] {r['path']} -> {r['status']}: {r.get('error', '')}")
                event(LOG, "ingest.error", level=logging.WARNING,
                      path=r["path"], err=str(r.get("error", ""))[:200])

    if files:
        _report([ing.ingest_file(f, **kwargs) for f in files])
    elif getattr(args, "watch", False):
        print(f"[ingest] 常驻模式：监听 {args.inbox}（每 {args.poll:g}s 一轮，Ctrl+C 退出）")
        while True:
            _report(ing.ingest_inbox(args.inbox, patterns=patterns,
                                     max_files=int(getattr(args, "max_files", 0) or 0),
                                     **kwargs))
            time.sleep(max(1.0, float(getattr(args, "poll", 10.0) or 10.0)))
    else:
        rows = ing.ingest_inbox(args.inbox, patterns=patterns,
                                max_files=int(getattr(args, "max_files", 0) or 0), **kwargs)
        if not rows:
            print(f"[ingest] {args.inbox} 下没有待处理文件"
                  f"（把 txt/csv/jsonl 丢进去，或用 --file 指定）")
        _report(rows)

    s = ing.summary()
    print(f"[ingest] 台账 {s['ledger']}：文件 {s['files']} 个，记录 {s['records']}")
    out = Path(getattr(args, "out_dir", "outputs/ingest")) / "last_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    return s
