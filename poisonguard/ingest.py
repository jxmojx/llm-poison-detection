"""入库钩子（C2）：数据一落地就打分的自动门禁。

为什么需要它：`scan` 和 `serve` 都是"人拿着东西来跑"。真实数据入库不是这样——
对象存储里落了新文件、Kafka 里有 `dataset.ingested` 事件、Airflow 的 DAG 产出一批
jsonl，**没人会在中间手动敲命令**。这个模块把"扫描"变成"订阅"。

设计要点（这三条才是"生产级"和"脚本"的差别）：

1. **幂等**：文件按 sha256 只处理一次；台账里 `(file_sha, row_id)` 唯一，
   服务重启后重扫同一批文件不会重复入账、不会重复隔离；
2. **可追溯**：每条判决记录当时的 `model` / `pkg_sha` / `threshold` / `decided_at`，
   一年后能回答"这条当时为什么被判 BLOCK"——否则出了事查不清；
3. **门禁语义**：BLOCK 进 `quarantine/`（硬隔离，不入训练集），
   WATCH 进 `review/`（人审队列，C1 的输入），PASS 进 `released/`（放行）。

单文件失败只把该文件记为 `status='error'`，不阻塞其它文件（一晚上的批处理里
一个坏文件不该拖垮整批）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

FLAGS = ("PASS", "WATCH", "BLOCK")
_DEFAULT_PATTERNS = ("*.txt", "*.csv", "*.tsv", "*.jsonl", "*.json")


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _day(ts: float | None = None) -> str:
    return time.strftime("%Y%m%d", time.localtime(ts or time.time()))


@dataclass
class Verdict:
    """一条判决。给下游（人审台、监控）读的最小结构。"""

    row_id: int
    text: str
    score: float
    flag: str
    views: dict[str, Any] = field(default_factory=dict)

    def as_row(self, *, source: str, model: str, pkg_sha: str,
               threshold: float, ts: float) -> dict:
        return {
            "source": source,
            "row_id": self.row_id,
            "text": self.text,
            "text_sha": text_sha(self.text),
            "score": round(float(self.score), 6),
            "flag": self.flag,
            "views": self.views or {},
            "model": model,
            "pkg_sha": pkg_sha,
            "threshold": threshold,
            "decided_at": round(ts, 3),
        }


# ----------------------------------------------------------------------
# 台账
# ----------------------------------------------------------------------
class IngestLedger:
    """sqlite 台账：文件级幂等 + 行级判决（可回查、可重放）。

    用 sqlite 而不是 jsonl，是为了让 `(file_sha, row_id)` 的唯一约束替我们做去重——
    多进程/重启后重扫时，"这条已经处理过"是数据库保证的，不是内存里的 set。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS files(
                    file_sha   TEXT PRIMARY KEY,
                    path       TEXT NOT NULL,
                    n_rows     INTEGER DEFAULT 0,
                    status     TEXT DEFAULT 'running',
                    model      TEXT DEFAULT '',
                    pkg_sha    TEXT DEFAULT '',
                    threshold  REAL DEFAULT 0,
                    started_at REAL DEFAULT 0,
                    finished_at REAL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS records(
                    file_sha   TEXT NOT NULL,
                    row_id     INTEGER NOT NULL,
                    text_sha   TEXT NOT NULL,
                    score      REAL NOT NULL,
                    flag       TEXT NOT NULL,
                    decided_at REAL NOT NULL,
                    PRIMARY KEY(file_sha, row_id)
                );
                CREATE INDEX IF NOT EXISTS idx_records_flag ON records(flag);
                """
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    # ---------------- 文件级 ----------------
    def file_status(self, sha: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT status FROM files WHERE file_sha=?", (sha,)).fetchone()
        return None if row is None else str(row[0])

    def seen_ok(self, sha: str) -> bool:
        return self.file_status(sha) == "ok"

    def begin(self, sha: str, path: str, *, model: str, pkg_sha: str,
              threshold: float) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO files(file_sha, path, status, model, pkg_sha, threshold, started_at)"
                " VALUES(?,?, 'running', ?,?,?,?)"
                " ON CONFLICT(file_sha) DO UPDATE SET"
                "   status='running', model=excluded.model, pkg_sha=excluded.pkg_sha,"
                "   threshold=excluded.threshold, started_at=excluded.started_at",
                (sha, path, model, pkg_sha, float(threshold), time.time()),
            )

    def finish(self, sha: str, n_rows: int, status: str = "ok") -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE files SET n_rows=?, status=?, finished_at=? WHERE file_sha=?",
                (int(n_rows), status, time.time(), sha),
            )

    # ---------------- 行级 ----------------
    def add_records(self, rows: Iterable[tuple[str, int, str, float, str]],
                    decided_at: float) -> int:
        """rows: (file_sha, row_id, text_sha, score, flag)。重复行被忽略（幂等）。"""
        n = 0
        with self._conn() as c:
            for sha, row_id, tsha, score, flag in rows:
                cur = c.execute(
                    "INSERT OR IGNORE INTO records"
                    "(file_sha,row_id,text_sha,score,flag,decided_at) VALUES(?,?,?,?,?,?)",
                    (sha, int(row_id), tsha, float(score), flag, decided_at),
                )
                n += int(cur.rowcount or 0)
        return n

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute("SELECT flag, COUNT(*) FROM records GROUP BY flag").fetchall()
        out = {f: 0 for f in FLAGS}
        out.update({str(k): int(v) for k, v in rows})
        return out

    def n_files(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM files").fetchone()[0])


# ----------------------------------------------------------------------
# 打分 → 分档 → 落盘
# ----------------------------------------------------------------------
class Ingestor:
    """把 scorer 包成"文件进来、判决出去"的门禁。

    scorer 只需实现 `score(texts, explain=False) -> dict` 且有 `.threshold` /
    `.version`（`service.PoisonScorer` 与 `deploy.DeployedScorer` 都满足）。
    """

    def __init__(self, scorer, out_dir: str | Path,
                 ledger_path: str | Path, *,
                 watch_band: float = 0.05,
                 explain: bool = True,
                 pkg_sha: str = "",
                 threshold: float | None = None):
        self.scorer = scorer
        self.out_dir = Path(out_dir)
        self.ledger = IngestLedger(ledger_path)
        self.threshold = float(threshold if threshold is not None else getattr(scorer, "threshold", 0.5))
        self.watch_band = max(0.0, float(watch_band))
        self.explain = bool(explain)
        self.pkg_sha = pkg_sha
        self.model = str(getattr(scorer, "version", "unknown"))
        for sub in ("quarantine", "review", "released"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)

    # ---------------- 分档 ----------------
    def decide(self, score: float) -> str:
        """三档：> 阈值 BLOCK；落在 [阈值-band, 阈值] WATCH；其余 PASS。

        WATCH 是给"分数在门口"的样本用的——它们既不该放行也不该直接进隔离区，
        而是进人审队列（这正是 C1 的输入）。band=0 时退化为二分类。
        """
        if score > self.threshold:
            return "BLOCK"
        if self.watch_band > 0 and score >= self.threshold - self.watch_band:
            return "WATCH"
        return "PASS"

    def _write(self, flag: str, payloads: list[dict]) -> None:
        if not payloads:
            return
        sub = {"BLOCK": "quarantine", "WATCH": "review", "PASS": "released"}[flag]
        p = self.out_dir / sub / f"{_day()}.jsonl"
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            for p2 in payloads:
                f.write(json.dumps(p2, ensure_ascii=False) + "\n")

    # ---------------- 主入口 ----------------
    def ingest_file(self, path: str | Path, *, fmt: str = "auto",
                    column: str = "text", limit: int = 0) -> dict:
        """处理一个文件。返回 {path, file_sha, n, block, watch, pass, status, seconds}。"""
        from .scan import load_rows  # 复用同一套读取逻辑（txt/csv/jsonl + row_id 语义）

        path = Path(path)
        t0 = time.time()
        if not path.exists():
            return {"path": str(path), "status": "error", "error": "文件不存在",
                    "n": 0, "block": 0, "watch": 0, "pass": 0, "seconds": 0.0}
        sha = file_sha256(path)
        if self.ledger.seen_ok(sha):
            return {"path": str(path), "file_sha": sha, "status": "skipped",
                    "reason": "该文件已处理过（sha256 命中台账）",
                    "n": 0, "block": 0, "watch": 0, "pass": 0, "seconds": 0.0}

        self.ledger.begin(sha, str(path), model=self.model,
                          pkg_sha=self.pkg_sha, threshold=self.threshold)
        try:
            rows = load_rows(path, fmt=fmt, column=column, limit=limit)
            texts = [r["text"] for r in rows]
            if not texts:
                self.ledger.finish(sha, 0, "empty")
                return {"path": str(path), "file_sha": sha, "status": "empty",
                        "n": 0, "block": 0, "watch": 0, "pass": 0,
                        "seconds": round(time.time() - t0, 2)}
            res = self.scorer.score(texts, explain=self.explain)
            results = res.get("results") or []
            ts = time.time()
            verdicts: list[Verdict] = []
            for i, r in enumerate(results):
                score = float(r.get("score", 0.0))
                verdicts.append(Verdict(row_id=int(rows[i]["row_id"]), text=texts[i],
                                        score=score, flag=self.decide(score),
                                        views=r.get("views") or {}))
            payloads = [v.as_row(source=str(path), model=self.model,
                                 pkg_sha=self.pkg_sha, threshold=self.threshold, ts=ts)
                        for v in verdicts]
            for flag in FLAGS:
                self._write(flag, [p for p in payloads if p["flag"] == flag])
            added = self.ledger.add_records(
                [(sha, v.row_id, text_sha(v.text), v.score, v.flag) for v in verdicts], ts)
            self.ledger.finish(sha, len(verdicts), "ok")
            counts = {f.lower(): sum(1 for v in verdicts if v.flag == f) for f in FLAGS}
            return {"path": str(path), "file_sha": sha, "status": "ok",
                    "n": len(verdicts), "records_added": added,
                    "block": counts["block"], "watch": counts["watch"],
                    "pass": counts["pass"], "seconds": round(time.time() - t0, 2)}
        except Exception as e:  # 单文件失败不阻塞整批
            self.ledger.finish(sha, 0, "error")
            return {"path": str(path), "file_sha": sha, "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "n": 0, "block": 0, "watch": 0, "pass": 0,
                    "seconds": round(time.time() - t0, 2)}

    def ingest_inbox(self, inbox: str | Path,
                     patterns: Iterable[str] = _DEFAULT_PATTERNS,
                     max_files: int = 0, **kwargs) -> list[dict]:
        """扫一个"投递目录"：把里面匹配的文件按修改时间从旧到新逐个处理。"""
        inbox = Path(inbox)
        if not inbox.exists():
            return []
        files: list[Path] = []
        for pat in patterns:
            files.extend(inbox.glob(pat))
        uniq = sorted({p.resolve() for p in files if p.is_file()},
                      key=lambda p: p.stat().st_mtime)
        if max_files and max_files > 0:
            uniq = uniq[:max_files]
        return [self.ingest_file(p, **kwargs) for p in uniq]

    def summary(self) -> dict:
        return {"ledger": str(self.ledger.path), "files": self.ledger.n_files(),
                "records": self.ledger.counts(),
                "threshold": self.threshold, "model": self.model}
