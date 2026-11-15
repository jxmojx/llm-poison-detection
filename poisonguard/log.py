"""结构化日志（P1）：给运维留可机读的事件流，版本化 print 之外的一层。

为什么不是"把所有 print 换成 logging"：CLI 的打印是**给人看的**（带 emoji、对齐、
进度刷新），日志是**给机器看的**（一行一个 JSON 事件，带 service/version/耗时）。
两者受众不同，混在一起两边都不好用。

约定：
- 事件名用 `模块.动作`（`ingest.file`、`score.request`、`review.decision`）；
- 字段名固定：`event`、`level`、`ts`、`n`、`ms`、`ok`、`err`；
- `--log-json` 输出单行 JSON（便于 Loki/ES 采集），否则输出 `ts level event k=v ...`。
"""

import json
import logging
import os
import sys
import time

_ROOT = "poisonguard"
_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            payload["err"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(payload, ensure_ascii=False)


class _KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {}) or {}
        kv = " ".join(f"{k}={v}" for k, v in fields.items())
        return (f"{time.strftime('%H:%M:%S', time.localtime(record.created))} "
                f"{record.levelname.lower():5s} {record.getMessage()}"
                + (f" {kv}" if kv else ""))


def setup(level: str = "INFO", json_mode: bool = False) -> None:
    """幂等：重复调用只更新级别与格式。"""
    global _CONFIGURED
    root = logging.getLogger(_ROOT)
    fmt = _JsonFormatter() if json_mode else _KeyValueFormatter()
    root.handlers[:] = [h for h in root.handlers if getattr(h, "_pg", False)]
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h._pg = True
        h.setFormatter(fmt)
        root.addHandler(h)
    else:
        for h in root.handlers:
            h.setFormatter(fmt)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "cli") -> logging.Logger:
    if not _CONFIGURED:
        setup(os.environ.get("POISONGUARD_LOG_LEVEL", "INFO"))
    return logging.getLogger(f"{_ROOT}.{name}")


def event(logger: logging.Logger, name: str, level: int = logging.INFO, **fields) -> None:
    """发一个结构化事件：`event(log, "ingest.file", n=6, block=0, seconds=0.1)`。"""
    logger.log(level, name, extra={"fields": fields})
