"""结构化日志的契约：一行一个 JSON、字段可机读、级别生效。"""

import io
import json
import logging

from poisonguard import log as L


def _rows(buf: io.StringIO) -> list[dict]:
    return [json.loads(ln) for ln in buf.getvalue().splitlines() if ln.strip()]


def _logger(name: str, level: int = logging.INFO):
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.propagate = False
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(L._JsonFormatter())
    lg.handlers[:] = [h]
    return lg, buf


def test_event_is_single_line_json_with_fields():
    lg, buf = _logger("poisonguard.t1")
    L.event(lg, "ingest.done", path="/d/a.txt", n=6, block=0, watch=2)
    L.event(lg, "score.request", n=2, ms=12.5)
    rows = _rows(buf)
    assert len(rows) == 2
    assert rows[0]["event"] == "ingest.done"
    assert rows[0]["path"] == "/d/a.txt" and rows[0]["n"] == 6
    assert rows[0]["level"] == "INFO" and "ts" in rows[0]
    assert rows[1]["ms"] == 12.5


def test_level_controls_emission():
    lg, buf = _logger("poisonguard.t2", logging.WARNING)
    L.event(lg, "ingest.error", level=logging.WARNING, path="x")
    L.event(lg, "ingest.debug", level=logging.DEBUG, path="y")
    assert [r["event"] for r in _rows(buf)] == ["ingest.error"], \
        "DEBUG 不应在 WARNING 级别输出"


def test_setup_is_idempotent_and_respects_level():
    L.setup(level="WARNING", json_mode=True)
    L.setup(level="WARNING", json_mode=True)  # 重复调用不该加重复 handler
    root = logging.getLogger("poisonguard")
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING
    L.setup(level="INFO")  # 复原，避免影响其它测试
