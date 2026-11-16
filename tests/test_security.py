"""方向③ 缺口1：鉴权 + 限流 + 审计 —— 纯离线契约测试。

全部测试走 fastapi TestClient（不联网、不加载模型），围绕
`poisonguard.security` 的 KeyStore / RateLimiter / AuditStore /
require_api_key / rate_limit 依赖函数做黑盒断言。
"""

from __future__ import annotations

import sqlite3
import time

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from poisonguard.security import (
    AuditStore,
    KeyStore,
    RateLimiter,
    Security,
    SecurityConfig,
    TokenBucket,
    rate_limit,
    require_api_key,
)


# --------------------------------------------------------------------------
# KeyStore：鉴权语义
# --------------------------------------------------------------------------
class TestKeyStore:
    def test_default_dev_key_when_not_disabled(self):
        store = KeyStore(disabled=False)
        assert store.authenticate("dev-key-not-for-production")

    def test_wrong_key_rejected(self):
        store = KeyStore(keys=["k-1"], disabled=False)
        assert not store.authenticate("nope")
        assert store.authenticate("k-1")

    def test_empty_key_rejected(self):
        store = KeyStore(keys=["k-1"], disabled=False)
        assert not store.authenticate("")
        assert not store.authenticate(None)

    def test_disabled_accepts_anything(self):
        store = KeyStore(keys=["k-1"], disabled=True)
        assert store.authenticate(None)
        assert store.authenticate("whatever")

    def test_admin_flag(self):
        store = KeyStore(keys=["user-1"], disabled=False)
        store.add("boss", admin=True)
        # 普通端点：任何有效 key 都过
        assert store.authenticate("user-1", admin=False)
        assert store.authenticate("boss", admin=False)
        # 管理端点：只有 admin key 过
        assert store.authenticate("boss", admin=True)
        assert not store.authenticate("user-1", admin=True)

    def test_id_of_is_fingerprint_without_leaking_key(self):
        store = KeyStore(keys=["super-secret-value-xyz"], disabled=False)
        kid = store.id_of("super-secret-value-xyz")
        assert len(kid) == 6
        assert "secret" not in kid  # 不落明文 / 不给可反推值
        assert store.id_of("super-secret-value-xyz") == store.id_of(
            "super-secret-value-xyz")

    def test_remove(self):
        store = KeyStore(keys=["k-1"], disabled=False)
        store.remove("k-1")
        assert not store.authenticate("k-1")

    def test_count(self):
        store = KeyStore(keys=["k-1", "k-2"], disabled=False)
        assert store.count() == 2
        store.add("k-3")
        assert store.count() == 3


# --------------------------------------------------------------------------
# 令牌桶 / per-key 限流
# --------------------------------------------------------------------------
class TestTokenBucket:
    def test_burst_allowed_until_empty(self):
        b = TokenBucket(rate=1.0, capacity=3)
        assert b.allow()
        assert b.allow()
        assert b.allow()
        assert not b.allow()  # 桶空（速率补充极慢，近似静止）

    def test_tokens_refill_over_time(self):
        b = TokenBucket(rate=10.0, capacity=5)
        for _ in range(5):
            assert b.allow()
        assert not b.allow()
        time.sleep(0.15)  # ~1.5 tokens 补回
        assert b.allow()

    def test_state_reports_tokens(self):
        b = TokenBucket(rate=1.0, capacity=5)
        assert 0 <= b.state()["tokens"] <= 5
        assert b.state()["capacity"] == 5


class TestRateLimiter:
    def test_per_key_buckets_independent(self):
        lim = RateLimiter(rate=1.0, burst=1)
        assert lim.allow("a")
        assert not lim.allow("a")  # a 的桶空
        assert lim.allow("b")      # b 独立成桶，不受影响

    def test_interval_throttle(self):
        lim = RateLimiter(rate=10.0, burst=10, interval=0.2)
        assert lim.allow("k")
        assert not lim.allow("k")       # 间隔未到
        time.sleep(0.25)
        assert lim.allow("k")

    def test_interval_per_key(self):
        lim = RateLimiter(rate=10.0, burst=10, interval=0.2)
        assert lim.allow("k1")
        assert lim.allow("k2")          # 第二个 key 不受 k1 间隔约束
        assert not lim.allow("k1")

    def test_snapshot(self):
        lim = RateLimiter(rate=5.0, burst=10, interval=0.5)
        s = lim.snapshot()
        assert s["rate"] == 5.0 and s["burst"] == 10.0
        assert s["interval"] == 0.5
        lim.allow("x")
        assert lim.snapshot()["active_keys"] == 1


# --------------------------------------------------------------------------
# AuditStore：sqlite 审计落库
# --------------------------------------------------------------------------
class TestAuditStore:
    def test_record_writes_row(self, tmp_path):
        db = tmp_path / "audit.sqlite3"
        store = AuditStore(db, enabled=True)
        store.record(action="score.request", method="POST", path="/score",
                     status=200, api_key_id="abc123", client_ip="127.0.0.1",
                     n=3, ms=12.4, detail="ok")
        rows = store.recent(limit=10)
        assert len(rows) == 1
        r = rows[0]
        assert r["action"] == "score.request"
        assert r["path"] == "/score"
        assert r["status"] == 200
        assert r["api_key_id"] == "abc123"
        assert r["n"] == 3

    def test_recent_filter_by_action(self, tmp_path):
        db = tmp_path / "audit.sqlite3"
        store = AuditStore(db, enabled=True)
        store.record(action="score.request", path="/score")
        store.record(action="admin.reload", path="/admin/reload")
        assert len(store.recent(action="admin.reload")) == 1

    def test_disabled_does_not_write(self, tmp_path):
        db = tmp_path / "audit.sqlite3"
        store = AuditStore(db, enabled=False)
        store.record(action="score.request", path="/score")
        assert store.recent() == []

    def test_recent_on_missing_db_returns_empty(self):
        store = AuditStore(None, enabled=False)
        assert store.recent() == []

    def test_audit_failure_does_not_raise(self, tmp_path, monkeypatch):
        """写失败（如磁盘满/连接断）不能拖垮业务：record 内部吞掉异常。"""
        db = tmp_path / "audit.sqlite3"
        store = AuditStore(db, enabled=True)
        import poisonguard.security as sec_mod

        def _boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(sec_mod.sqlite3, "connect", _boom)
        store.record(action="score.request", path="/score")  # 不应抛
        assert True

    def test_manual_sqlite_query(self, tmp_path):
        db = tmp_path / "audit.sqlite3"
        store = AuditStore(db, enabled=True)
        store.record(action="decide", path="/api/decision", status=200,
                     api_key_id="dec2024", client_ip="10.0.0.7", n=1, ms=3.0)
        with sqlite3.connect(db) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM audit_log").fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == 200


# --------------------------------------------------------------------------
# FastAPI 依赖：鉴权 / 限流 / 审计 端到端（TestClient 不联网）
# --------------------------------------------------------------------------
def _make_secure_app(security: Security) -> FastAPI:
    app = FastAPI()

    @app.get("/who")
    def who(info: tuple = Depends(security.auth())):
        return {"client_ip": info[0], "api_key_id": info[1]}

    @app.get("/admin/ping")
    def admin_ping(info: tuple = Depends(security.auth(admin=True))):
        return {"ok": True}

    @app.get("/limited")
    def limited(_: None = Depends(security.limited())):
        return {"ok": True}

    @app.get("/audited")
    def audited(info: tuple = Depends(security.auth()),
                _: None = Depends(security.limited())):
        security.record(action="probe.audited", method="GET", path="/audited",
                        status=200, api_key_id=info[1],
                        client_ip=info[0], n=1, ms=0.5)
        return {"ok": True}

    return app


class TestHttpAuth:
    def test_no_key_when_disabled_passes(self):
        sec = Security(SecurityConfig(disabled=True))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            r = c.get("/who")
        assert r.status_code == 200
        assert r.json()["api_key_id"] == "disabled"

    def test_missing_key_401(self):
        sec = Security(SecurityConfig(api_keys=["k-user"], disabled=False))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            r = c.get("/who")
        assert r.status_code == 401

    def test_invalid_key_401_uniform(self):
        sec = Security(SecurityConfig(api_keys=["k-user"], disabled=False))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            r = c.get("/who", headers={"X-API-Key": "bogus"})
        assert r.status_code == 401

    def test_valid_key_200_returns_fingerprint(self):
        sec = Security(SecurityConfig(api_keys=["k-user"], disabled=False))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            r = c.get("/who", headers={"X-API-Key": "k-user"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["api_key_id"]) == 6

    def test_admin_endpoint_requires_admin_key(self):
        sec = Security(SecurityConfig(api_keys=["k-user"],
                                      admin_keys=["k-admin"], disabled=False))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            assert c.get("/admin/ping").status_code == 401
            assert c.get("/admin/ping",
                         headers={"X-API-Key": "k-user"}).status_code == 403
            assert c.get("/admin/ping",
                         headers={"X-API-Key": "k-admin"}).status_code == 200

    def test_multiple_api_keys_all_accepted(self):
        sec = Security(SecurityConfig(api_keys=["a1", "a2"], disabled=False))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            assert c.get("/who", headers={"X-API-Key": "a1"}).status_code == 200
            assert c.get("/who", headers={"X-API-Key": "a2"}).status_code == 200


class TestHttpRateLimit:
    def test_rate_limit_returns_429(self):
        sec = Security(SecurityConfig(disabled=False, api_keys=["k"],
                                      rate_rps=1.0, rate_burst=2))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            headers = {"X-API-Key": "k"}
            assert c.get("/limited", headers=headers).status_code == 200
            assert c.get("/limited", headers=headers).status_code == 200
            assert c.get("/limited", headers=headers).status_code == 429

    def test_different_keys_do_not_share_quota(self):
        sec = Security(SecurityConfig(disabled=False, api_keys=["a", "b"],
                                      rate_rps=1.0, rate_burst=1))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            assert c.get("/limited", headers={"X-API-Key": "a"}).status_code == 200
            assert c.get("/limited", headers={"X-API-Key": "b"}).status_code == 200
            assert c.get("/limited", headers={"X-API-Key": "a"}).status_code == 429

    def test_rate_limit_uses_ip_when_no_key(self):
        """无 key 时退化为客户端 IP 做 per-key 限流；同一 IP 连续打会 429。"""
        sec = Security(SecurityConfig(disabled=False, api_keys=["k"],
                                      rate_rps=1.0, rate_burst=1))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            # /limited 只挂限流不挂鉴权；无 key 时按 testclient 这个 IP 计数
            assert c.get("/limited").status_code == 200   # 首个过
            assert c.get("/limited").status_code == 429   # 桶空（IP 维度）
        # 再看带 key 的调用走 header key 而非 IP
        sec2 = Security(SecurityConfig(disabled=False, api_keys=["k", "o"],
                                       rate_rps=1.0, rate_burst=1))
        app2 = _make_secure_app(sec2)
        with TestClient(app2) as c:
            assert c.get("/limited").status_code == 200           # IP:testclient
            assert c.get("/limited", headers={"X-API-Key": "k"}).status_code == 200
            assert c.get("/limited", headers={"X-API-Key": "o"}).status_code == 200
            assert c.get("/limited").status_code == 429           # IP 维度桶空
            assert c.get("/limited", headers={"X-API-Key": "k"}).status_code == 429


class TestHttpAudit:
    def test_audit_written_on_request(self, tmp_path):
        db = tmp_path / "audit.sqlite3"
        sec = Security(SecurityConfig(disabled=False, api_keys=["k"],
                                      audit_path=str(db)))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            r = c.get("/audited", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        rows = sec.audit.recent(limit=10, action="probe.audited")
        assert len(rows) == 1
        assert rows[0]["client_ip"] == "testclient"
        assert len(rows[0]["api_key_id"]) == 6

    def test_disabled_auth_still_audits(self, tmp_path):
        db = tmp_path / "audit.sqlite3"
        sec = Security(SecurityConfig(disabled=True, audit_path=str(db)))
        app = _make_secure_app(sec)
        with TestClient(app) as c:
            assert c.get("/audited").status_code == 200
        rows = sec.audit.recent(action="probe.audited")
        assert len(rows) == 1
        assert rows[0]["api_key_id"] == "disabled"


def test_security_config_smoke():
    cfg = SecurityConfig.from_cli(api_keys=["k1"], admin_keys=["k2"])
    assert cfg.disabled is False
    assert cfg.api_keys == ["k1"] and cfg.admin_keys == ["k2"]

    cfg2 = SecurityConfig.from_cli()  # 无 key：默认 disabled
    assert cfg2.disabled is True

    cfg3 = SecurityConfig.from_cli(auth_disabled=False)  # 显式关
    assert cfg3.disabled is False


# --------------------------------------------------------------------------
# service.py 集成：/score /health 挂上鉴权+限流+审计
# --------------------------------------------------------------------------
class _DummyScorer:
    """与 tests/test_v3.py 的占位打分器同构：不加载模型，离线可用。"""

    threshold = 0.5
    version = "dummy-v1"
    knn = object()       # /health 会访问 .knn/.ngram/.student
    ngram = None
    student = None

    def __init__(self):
        self.n_calls = 0

    def score(self, texts, explain=False):
        self.n_calls += 1
        res = []
        for t in texts:
            res.append({"score": 0.7 if "坏" in t else 0.2,
                        "flag": ("坏" in t),
                        "views": {"knn": 0.1, "fpr_est": 0.9,
                                  "char_ngram": 0.2, "student": 0.7,
                                  "chars": len(t)}})
        return {"results": res, "n": len(texts), "latency_ms": 1.0,
                "ms_per_item": 0.5, "threshold": 0.5, "model": self.version}

    def stats(self):
        return {"uptime_s": 1.0, "n_calls": self.n_calls, "n_texts": 0,
                "ms_per_item_p50": 0.1, "ms_per_item_p99": 0.2,
                "threshold": 0.5, "model": self.version}


class TestServiceAuthIntegration:
    def _app(self, security):
        from poisonguard.service import create_app
        from poisonguard.config import load_config
        return create_app(load_config(), scorer=_DummyScorer(), security=security)

    def test_health_requires_key(self):
        sec = Security(SecurityConfig(api_keys=["k-user"], disabled=False,
                                      audit_path=None))
        app = self._app(sec)
        with TestClient(app) as c:
            assert c.get("/health").status_code == 401
            assert c.get("/health",
                         headers={"X-API-Key": "k-user"}).status_code == 200

    def test_score_requires_key_and_rejects_wrong(self):
        sec = Security(SecurityConfig(api_keys=["k-user"], disabled=False))
        app = self._app(sec)
        with TestClient(app) as c:
            r = c.post("/score", json={"texts": ["你好"]})
            assert r.status_code == 401
            r = c.post("/score", json={"texts": ["你好"]},
                       headers={"X-API-Key": "wrong"})
            assert r.status_code == 401
            r = c.post("/score", json={"texts": ["你好"]},
                       headers={"X-API-Key": "k-user"})
            assert r.status_code == 200
            assert r.json()["n"] == 1

    def test_score_respects_rate_limit(self):
        sec = Security(SecurityConfig(api_keys=["k"], disabled=False,
                                      rate_rps=1.0, rate_burst=1))
        app = self._app(sec)
        with TestClient(app) as c:
            h = {"X-API-Key": "k"}
            assert c.post("/score", json={"texts": ["a"]}, headers=h).status_code == 200
            assert c.post("/score", json={"texts": ["a"]}, headers=h).status_code == 429

    def test_score_writes_audit(self, tmp_path):
        db = tmp_path / "audit.sqlite3"
        sec = Security(SecurityConfig(api_keys=["k"], disabled=False,
                                      audit_path=str(db)))
        app = self._app(sec)
        with TestClient(app) as c:
            c.post("/score", json={"texts": ["你好", "世界"]},
                   headers={"X-API-Key": "k"})
        rows = sec.audit.recent(action="score.request")
        assert len(rows) == 1
        assert rows[0]["n"] == 2
        # 中间件也落一条 http.request
        http_rows = sec.audit.recent(action="http.request")
        assert len(http_rows) >= 1

    def test_empty_texts_still_400_with_auth(self):
        sec = Security(SecurityConfig(api_keys=["k"], disabled=False,
                                      audit_path=None))
        app = self._app(sec)
        with TestClient(app) as c:
            r = c.post("/score", json={"texts": []},
                       headers={"X-API-Key": "k"})
            assert r.status_code == 400


# --------------------------------------------------------------------------
# review.py 集成：人审台 API 全挂鉴权，判决要 admin
# --------------------------------------------------------------------------
class TestReviewAuthIntegration:
    def _app(self, security, tmp_path):
        from poisonguard.review import create_review_app
        return create_review_app(str(tmp_path), db_path=str(tmp_path / "r.sqlite3"),
                                 security=security)

    def test_queue_requires_key(self, tmp_path):
        sec = Security(SecurityConfig(api_keys=["k"], disabled=False,
                                      audit_path=None))
        app = self._app(sec, tmp_path)
        with TestClient(app) as c:
            assert c.get("/api/queue").status_code == 401
            assert c.get("/api/queue",
                         headers={"X-API-Key": "k"}).status_code == 200

    def test_decision_requires_admin(self, tmp_path):
        """判决是写操作：普通 API key 要能读队列但判不了。"""
        sec = Security(SecurityConfig(api_keys=["k-user"],
                                      admin_keys=["k-admin"], disabled=False,
                                      audit_path=None))
        app = self._app(sec, tmp_path)
        body = {"key": "s|1|sha", "decision": "benign", "reason": "正常新闻"}
        with TestClient(app) as c:
            r = c.post("/api/decision", json=body)
            assert r.status_code == 401
            r = c.post("/api/decision", json=body,
                       headers={"X-API-Key": "k-user"})
            assert r.status_code == 403
            r = c.post("/api/decision", json=body,
                       headers={"X-API-Key": "k-admin"})
            assert r.status_code == 200

    def test_stats_and_export_require_key(self, tmp_path):
        sec = Security(SecurityConfig(api_keys=["k"], disabled=False,
                                      audit_path=None))
        app = self._app(sec, tmp_path)
        with TestClient(app) as c:
            assert c.get("/api/stats").status_code == 401
            assert c.get("/api/export.csv").status_code == 401
            assert c.get("/api/stats",
                         headers={"X-API-Key": "k"}).status_code == 200

    def test_decision_writes_audit(self, tmp_path):
        db = tmp_path / "audit.sqlite3"
        sec = Security(SecurityConfig(api_keys=["k-user"],
                                      admin_keys=["k-admin"], disabled=False,
                                      audit_path=str(db)))
        app = self._app(sec, tmp_path)
        body = {"key": "s|1|sha", "decision": "benign", "reason": "正常新闻",
                "reviewer": "amy"}
        with TestClient(app) as c:
            c.post("/api/decision", json=body,
                   headers={"X-API-Key": "k-admin"})
        rows = sec.audit.recent(action="review.decision")
        assert len(rows) == 1
        assert rows[0]["api_key_id"]  # 指纹非空
        assert "decision=benign reviewer=amy" in rows[0]["detail"]

