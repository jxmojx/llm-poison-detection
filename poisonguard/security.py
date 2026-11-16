"""在线服务安全底座（方向③ 缺口 1）：API Key 鉴权 + 令牌桶限流 + 结构化审计。

为什么在服务这一层做，而不是交给 Nginx/网关：项目定位是"一键私有化部署"
（`serve`/`review` 都是单个 uvicorn 进程），很多部署场景没有独立网关。
这里用**标准库 + 已有 fastapi 依赖**实现三件最小可用的事，不引重依赖：

1. **API Key 鉴权**（Header `X-API-Key`）：
   - 多把 key 同时有效（`ADMIN_KEYS` 可空 = 管理端点只认 API key）；
   - 用 `secrets.compare_digest` 做常数时间比较，不泄漏 key 长度/内容；
   - 端点可标 `admin=True`，只有管理密钥能过（热更新/模型管理这类敏感操作）。
2. **限流**：per-key 令牌桶（标准库 `time.monotonic`），突发 + 平均速率两个参数；
   - 想复用 fastapi 中间件吗？不需要——做成**可依赖的依赖**注入到端点最稳，
     TestClient 契约测试可以直接断言 429。
3. **审计**：`request/decide` 等业务动作落 sqlite（`audit_log` 表）或结构化日志；
   记到 sqlite 的话不需要额外采集器，直接查表即可。

设计取舍
--------
- 规则：无 API key → 401；key 不存在 → 401（不区分 401/403，避免探测 key 是否存在）；
  `disabled=True` 时全部通过（本地开发/内网无网关场景默认给一个 dev key 兜底可关）。
- 限流只对**通过鉴权**的请求计数（未鉴权先 401，不算配额）。
- 审计写 sqlite：并发安全（每写一个短连接 + `WAL` 未开但 `timeout` 兜底），
  也提供 `audit_event()` 的纯日志通道双写。

外部依赖：`secrets`/`time`/`sqlite3`（标准库）+ fastapi（既有依赖）。
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable

from .log import event, get_logger
from fastapi import Header, HTTPException, Request  # noqa: F401  (依赖注解用)

LOG = get_logger("security")

DEFAULT_KEYS = ["dev-key-not-for-production"]
SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    action      TEXT NOT NULL,
    method      TEXT NOT NULL,
    path        TEXT NOT NULL,
    status      INTEGER NOT NULL,
    api_key_id  TEXT DEFAULT '',
    client_ip   TEXT DEFAULT '',
    n           INTEGER DEFAULT 0,
    ms          REAL DEFAULT 0,
    detail      TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
"""


def _key_id(key: str) -> str:
    """不落明文 key，只落前缀指纹（前 6 位 sha256），审计/日志里可溯源不可反推。"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:6]


class KeyStore:
    """API key 集合：默认一把 dev key，可追加多把，支持 admin 标记。"""

    def __init__(self, keys: Iterable[str] | None = None, *, disabled: bool = False):
        self.disabled = bool(disabled)
        self._keys: dict[str, bool] = {}
        self._lock = threading.Lock()
        if not disabled:
            for k in keys or DEFAULT_KEYS:
                self.add(k)

    def add(self, key: str, *, admin: bool = False) -> None:
        with self._lock:
            self._keys[key] = bool(admin)

    def remove(self, key: str) -> None:
        with self._lock:
            self._keys.pop(key, None)

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.disabled = not enabled

    def enabled(self) -> bool:
        with self._lock:
            return not self.disabled

    def authenticate(self, key: str | None, *, admin: bool = False) -> bool:
        """常数时间比较；disabled=True 时任何请求都过（开发模式）。

        `admin=True` 且 disabled=False 时，即使 key 有效但非管理 key 也拒绝——
        这是一个刻意的不对称：管理端点（热更新/换模型/清缓存）要更严格。
        """
        if self.disabled:
            return True
        if not key:
            return False
        with self._lock:
            for k, is_admin in self._keys.items():
                if secrets.compare_digest(k.encode("utf-8"), key.encode("utf-8")):
                    return (not admin) or is_admin
        return False

    def id_of(self, key: str) -> str:
        return "" if not key else _key_id(key)

    def count(self) -> int:
        with self._lock:
            return len(self._keys)


class TokenBucket:
    """令牌桶限流：容量 `capacity` = 突发上限，`rate` = 每秒补充速率。

    纯 `time.monotonic` 实现，无外部依赖；线程间共享实例即可（读 bucket 的
    临界区用锁保护，双检写回）。取不到令牌返回 False → 调用方回 429。
    """

    __slots__ = ("capacity", "rate", "_tokens", "_last", "_lock")

    def __init__(self, rate: float, capacity: float):
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def allow(self, tokens: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            if self._tokens < tokens:
                return False
            self._tokens -= tokens
            return True

    def state(self) -> dict:
        with self._lock:
            return {"capacity": self.capacity, "rate": self.rate,
                    "tokens": round(self._tokens, 3)}


class RateLimiter:
    """per-key 令牌桶集合：每个调用方独立一个桶，避免单账户挤占全局配额。

    `allow(key)` 对每个 key 惰性建桶；线程安全。用作 fastapi 依赖时
    取 `X-API-Key`（未鉴权场景退化为客户端 IP），打 429 时须由调用方抛异常。
    """

    __slots__ = ("rate", "burst", "interval", "_buckets", "_interval", "_lock")

    def __init__(self, rate: float, burst: float, interval: float = 0.0):
        self.rate = float(rate)
        self.burst = float(burst)
        self.interval = float(interval)  # >0 时启用"至少隔 interval 秒一次"
        self._buckets: dict[str, TokenBucket] = {}
        self._interval: dict[str, float] = {}  # key -> 上次放行时间
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if self.interval > 0:
                last = self._interval.get(key, 0.0)
                if now - last < self.interval:
                    return False
                last = now
            b = self._buckets.get(key)
            if b is None:
                b = TokenBucket(self.rate, self.burst)
                self._buckets[key] = b
            if not b.allow():
                return False
            if self.interval > 0:
                self._interval[key] = last
            return True

    def snapshot(self) -> dict:
        with self._lock:
            return {"rate": self.rate, "burst": self.burst,
                    "interval": self.interval, "active_keys": len(self._buckets)}


def rate_limit(limiter: RateLimiter):
    """FastAPI 依赖：按 key 限流；超额抛 429。

    key 取 `X-API-Key`（未鉴权场景退化为客户端 IP）。
    """
    from fastapi import HTTPException, Request

    def _dep(request: Request):
        key = (request.headers.get("X-API-Key")
               or (request.client.host if request.client else "") or "anon")
        if not limiter.allow(key):
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")

    return _dep


from dataclasses import dataclass, field  # noqa: E402


@dataclass
class SecurityConfig:
    """在线服务安全配置（CLI 参数 / 代码注入双通道）。

    默认 `disabled=True` **保持既有行为**（内网无鉴权直接可用，测试不受影响）；
    需要上公网/跨团队时显式传 key 并 `disabled=False`。
    """

    api_keys: list[str] | None = None        # 常规调用方 key（可多把）
    admin_keys: list[str] | None = None      # 管理 key（热更新/清缓存等敏感端点）
    disabled: bool = True                    # True = 关鉴权（开发/内网）
    rate_rps: float = 20.0                   # 每个调用方每秒令牌补充速率
    rate_burst: float = 60.0                 # 突发容量（突发一次最多放 60 条）
    audit_enabled: bool = True               # 请求/决定审计落 sqlite
    audit_path: str | None = None            # 默认 outputs/audit/audit.sqlite3
    min_interval: float = 1.0                # 管理端点至少间隔秒数

    @classmethod
    def from_cli(cls, *, api_keys: list[str] | None = None,
                 admin_keys: list[str] | None = None,
                 auth_disabled: bool = True,
                 rate_rps: float | None = None, rate_burst: float | None = None,
                 audit_path: str | None = None,
                 audit_enabled: bool = True) -> "SecurityConfig":
        keys = list(api_keys) if api_keys else None
        admins = list(admin_keys) if admin_keys else None
        if not keys and not admins:
            # 一把 key 都没给：跟随 --auth-disabled（默认 True = 内网直连）
            disabled = bool(auth_disabled)
        else:
            # 显式给了 key 就强制开鉴权（不能用默认 dev key 兜底）
            disabled = False
        return cls(api_keys=keys, admin_keys=admins, disabled=disabled,
                   rate_rps=(rate_rps if rate_rps is not None else 20.0),
                   rate_burst=(rate_burst if rate_burst is not None else 60.0),
                   audit_enabled=audit_enabled, audit_path=audit_path)


class Security:
    """把 KeyStore / RateLimiter / AuditStore 打包成一个可注入的服务对象。

    服务端点通过 `Depends(security.auth(admin=False))` 与
    `Depends(security.limited)` 两个依赖接入；端点内部再调 `security.audit.record(...)`。
    """

    def __init__(self, cfg: SecurityConfig | None = None):
        self.cfg = cfg or SecurityConfig()
        self.keys = KeyStore(self.cfg.api_keys, disabled=self.cfg.disabled)
        for k in self.cfg.admin_keys or ():
            self.keys.add(k, admin=True)
        self.limiter = RateLimiter(self.cfg.rate_rps, self.cfg.rate_burst)
        # 管理端点独立限流桶：带 min_interval，且跨请求共享（不能每次 new 一个）
        self._admin_limiter = RateLimiter(self.cfg.rate_rps, self.cfg.rate_burst,
                                          interval=self.cfg.min_interval)
        self.audit = AuditStore(self.cfg.audit_path, enabled=self.cfg.audit_enabled)

    # -- fastapi 依赖工厂 -------------------------------------------------
    def auth(self, *, admin: bool = False):
        return require_api_key(self.keys, admin=admin)

    def limited(self):
        return rate_limit(self.limiter)

    def admin_limited(self):
        return rate_limit(self._admin_limiter)

    # -- 便捷审计 ---------------------------------------------------------
    def record(self, *, action: str, method: str = "", path: str = "",
               status: int = 200, api_key_id: str = "", client_ip: str = "",
               n: int = 0, ms: float = 0.0, detail: str = "") -> None:
        self.audit.record(action=action, method=method, path=path, status=status,
                          api_key_id=api_key_id, client_ip=client_ip,
                          n=n, ms=ms, detail=detail)


class AuditStore:
    """审计日志存储：sqlite `audit_log` 表（可追溯、可查）。

    HTTP 请求在端点/依赖里调用 `record(...)`；每个动作一个短连接 commit，
    保证并发与崩溃下不丢已确认的写（代价是单条写 ~1ms，对审计场景可接受）。
    """

    def __init__(self, path: str | Path | None = None, *, enabled: bool = True,
                 logger=None):
        self.enabled = enabled
        self.logger = logger or LOG
        self.path = None
        if enabled:
            self.path = Path(path or "outputs/audit/audit.sqlite3")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path, timeout=30) as c:
                c.executescript(SCHEMA)

    def record(self, *, action: str, method: str = "", path: str = "",
               status: int = 200, api_key_id: str = "", client_ip: str = "",
               n: int = 0, ms: float = 0.0, detail: str = "") -> None:
        event(self.logger, f"audit.{action}", method=method, path=path,
              status=status, api_key_id=api_key_id, n=n, ms=round(ms, 1))
        if not self.enabled or self.path is None:
            return
        try:
            with sqlite3.connect(self.path, timeout=30) as c:
                c.execute(
                    "INSERT INTO audit_log(ts,action,method,path,status,"
                    "api_key_id,client_ip,n,ms,detail)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (time.time(), action, method, path, int(status),
                     api_key_id, client_ip, int(n), float(ms), detail[:2000]))
        except Exception as exc:  # 审计失败不能拖垮业务请求
            self.logger.warning("audit.write_failed: %s", exc)

    def recent(self, limit: int = 100, action: str | None = None) -> list[dict]:
        if self.path is None or not Path(self.path).exists():
            return []
        with sqlite3.connect(self.path, timeout=30) as c:
            c.row_factory = sqlite3.Row
            if action:
                rows = c.execute(
                    "SELECT * FROM audit_log WHERE action=? ORDER BY id DESC LIMIT ?",
                    (action, int(limit))).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                    (int(limit),)).fetchall()
        return [dict(r) for r in rows]


def require_api_key(store: KeyStore, *, admin: bool = False):
    """FastAPI 依赖：从 `X-API-Key` 头取 key 并校验，失败抛 http 401/403。

    返回 (client_ip, api_key_id) 元组，端点内部可把审计详情一起落库。
    """
    from fastapi import Header, HTTPException, Request

    def _dep(request: Request, x_api_key: str | None = Header(default=None)):
        if not store.enabled():
            return (request.client.host if request.client else "", "disabled")
        if not store.authenticate(x_api_key, admin=admin):
            # 缺失/无效统一 401：不区分这两者，避免被用来探测"key 是否存在"；
            # 只有"key 有效但缺管理权限"才用 403（这是被授权者触发的可操作信号）。
            if x_api_key is not None and store.authenticate(x_api_key, admin=False):
                raise HTTPException(status_code=403, detail="API Key 缺少管理权限")
            raise HTTPException(status_code=401, detail="API Key 缺失或无效")
        return (request.client.host if request.client else "", store.id_of(x_api_key))

    return _dep
