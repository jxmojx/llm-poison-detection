"""给 review 判决补结构化事件 + 写日志测试。"""
from pathlib import Path

# ---------------- review.py：判决事件 ----------------
p = Path("poisonguard/review.py")
t = p.read_text(encoding="utf-8")
anchor = '''    @app.post("/api/decision")
'''
assert anchor in t
if "from .log import event, get_logger" not in t:
    t = t.replace("from .config import", "from .log import event, get_logger\nfrom .config import", 1)
assert "from .log import event, get_logger" in t
if 'LOG = get_logger("review")' not in t:
    t = t.replace("def create_review_app(", 'LOG = get_logger("review")\n\n\ndef create_review_app(', 1)
p.write_text(t, encoding="utf-8", newline="\n")
print("review import ok")

# ---------------- tests/test_log.py ----------------
Path("tests/test_log.py").write_text('''"""结构化日志的契约：一行一个 JSON、字段可机读、级别生效。"""

import io
import json
import logging

from poisonguard import log as L


def _capture(fn):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(L._JsonFormatter())
    logger = logging.getLogger("poisonguard.test")
    logger.handlers[:] = [handler]
    logger.propagate = False
    fn(logger)
    return [json.loads(ln) for ln in buf.getvalue().splitlines() if ln.strip()]


def test_event_is_single_line_json_with_fields():
    def emit(lg):
        L.event(lg, "ingest.done", path="/d/a.txt", n=6, block=0, watch=2)
        L.event(lg, "score.request", n=2, ms=12.5)

    rows = _capture(emit)
    assert len(rows) == 2
    assert rows[0]["event"] == "ingest.done"
    assert rows[0]["path"] == "/d/a.txt" and rows[0]["n"] == 6
    assert rows[0]["level"] == "INFO" and "ts" in rows[0]
    assert rows[1]["ms"] == 12.5


def test_level_controls_emission():
    def emit(lg):
        L.event(lg, "ingest.error", level=logging.WARNING, path="x")
        L.event(lg, "ingest.debug", level=logging.DEBUG, path="y")

    lg = logging.getLogger("poisonguard.test2")
    lg.setLevel(logging.WARNING)
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(L._JsonFormatter())
    lg.handlers[:] = [h]
    lg.propagate = False
    L.event(lg, "ingest.error", level=logging.WARNING, path="x")
    L.event(lg, "ingest.debug", level=logging.DEBUG, path="y")
    lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]
    assert [r["event"] for r in lines] == ["ingest.error"], "DEBUG 不应在 WARNING 级别输出"


def test_setup_is_idempotent_and_respects_level():
    L.setup(level="WARNING", json_mode=True)
    L.setup(level="WARNING", json_mode=True)  # 重复调用不该加重复 handler
    root = logging.getLogger("poisonguard")
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING
    L.setup(level="INFO")  # 复原，避免影响其它测试
''', encoding="utf-8", newline="\n")
print("tests written")
