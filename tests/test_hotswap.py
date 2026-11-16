"""方向③ 缺口2：热更新 / 模型版本切换 —— 纯离线契约测试。

全部走 fastapi TestClient（不联网、不加载模型），用 `scorer_factory`
注入不同版本的占位打分器，验证：
- /admin/models 列出已装配版本与当前 active
- /admin/reload 从磁盘重新装配并切到新版本
- /admin/switch 在 registry 内切换 / 重建新版本
- /admin/rollback 回滚到上一个版本（热更新翻车一键还原）
- 失败时不改变 active（reload 抛错保持旧模型）
- 管理端点都要 admin key、限流、审计
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from poisonguard.config import load_config
from poisonguard.security import Security, SecurityConfig
from poisonguard.service import create_app


class _FakeScorer:
    """版本化的占位打分器：version 可配，score 按是否含关键字标记 flag。"""

    threshold = 0.5
    knn = object()
    ngram = None
    student = None

    def __init__(self, version: str = "dummy-v1", *, poison_word: str = "坏"):
        self.version = version
        self._word = poison_word
        self.n_calls = 0

    def score(self, texts, explain=False):
        self.n_calls += 1
        res = [{"score": 0.9 if self._word in t else 0.1,
                "flag": (self._word in t),
                "views": {"knn": 0.5, "fpr_est": 0.5, "char_ngram": 0.5,
                          "student": 0.9 if self._word in t else 0.1,
                          "chars": len(t)}}
               for t in texts]
        return {"results": res, "n": len(texts), "latency_ms": 1.0,
                "ms_per_item": 0.5, "threshold": self.threshold,
                "model": self.version}

    def stats(self):
        return {"uptime_s": 1.0, "n_calls": self.n_calls, "n_texts": 0,
                "ms_per_item_p50": 0.1, "ms_per_item_p99": 0.2,
                "threshold": self.threshold, "model": self.version}


def _sec(**kw) -> Security:
    kw.setdefault("api_keys", ["k-user"])
    kw.setdefault("admin_keys", ["k-admin"])
    kw.setdefault("disabled", False)
    kw.setdefault("audit_path", None)
    # min_interval=0.0：契约测试不发 sleep；min-interval 语义在 test_security 单测里证
    kw.setdefault("min_interval", 0.0)
    return Security(SecurityConfig(**kw))


def _make_app(security, factory, scorer=None):
    return create_app(load_config(), scorer=scorer, security=security,
                      scorer_factory=factory)


ADMIN = {"X-API-Key": "k-admin"}
USER = {"X-API-Key": "k-user"}


def test_admin_models_lists_initial_version():
    sc = _FakeScorer("v1")
    app = _make_app(_sec(), None, scorer=sc)
    with TestClient(app) as c:
        r = c.get("/admin/models", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["active"] is None  # 注入的 scorer 不参与版本注册
    assert body["model"] == "v1"
    assert body["versions"] == []


def test_admin_models_requires_admin_key():
    app = _make_app(_sec(), lambda: _FakeScorer("v1"))
    with TestClient(app) as c:
        assert c.get("/admin/models").status_code == 401
        assert c.get("/admin/models", headers=USER).status_code == 403
        assert c.get("/admin/models", headers=ADMIN).status_code == 200


def test_reload_builds_and_switches_version():
    """reload 用 factory 造新实例并切成 active；旧版本留在 registry。"""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _FakeScorer(f"v{calls['n']}")

    app = _make_app(_sec(), factory)
    with TestClient(app) as c:
        assert c.get("/health", headers=ADMIN).json()["model"] == "v1"
        r = c.post("/admin/reload", headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["active"] == "v2"
    assert r.json()["model"] == "v2"
    with TestClient(app) as c:
        body = c.get("/admin/models", headers=ADMIN).json()
    assert body["active"] == "v2"
    assert {v["key"] for v in body["versions"]} == {"v1", "v2"}


def test_switch_between_registry_versions_changes_behavior():
    """验证切换后 /score 真的按新版本打分（v1 只认"坏"、v2 只认"险"）。"""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeScorer("v1", poison_word="坏")
        return _FakeScorer("v2", poison_word="险")

    app = _make_app(_sec(), factory)
    with TestClient(app) as c:
        # 初始 v1： "坏" 标成 flag
        r = c.post("/score", json={"texts": ["有坏东西", "有险情"]},
                   headers=ADMIN)
        flags1 = {x["flag"] for x in r.json()["results"]}
        assert flags1 == {True, False}

        # reload 到 v2
        c.post("/admin/reload", headers=ADMIN)
        r = c.post("/score", json={"texts": ["有坏东西", "有险情"]},
                   headers=ADMIN)
        flags2 = {x["flag"] for x in r.json()["results"]}
        assert flags2 == {False, True}

        # switch 回 v1：行为变回
        c.post("/admin/switch", json={"version": "v1"}, headers=ADMIN)
        r = c.post("/score", json={"texts": ["有坏东西"]}, headers=ADMIN)
        assert r.json()["results"][0]["flag"] is True


def test_switch_unknown_version_404():
    app = _make_app(_sec(), lambda: _FakeScorer("v1"))
    with TestClient(app) as c:
        c.get("/health", headers=ADMIN)  # 触发装配
        r = c.post("/admin/switch", json={"version": "nope"}, headers=ADMIN)
    assert r.status_code == 404


def test_switch_empty_builds_new_version():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _FakeScorer(f"v{calls['n']}")

    app = _make_app(_sec(), factory)
    with TestClient(app) as c:
        c.get("/health", headers=ADMIN)          # v1
        r = c.post("/admin/switch", json={"version": ""}, headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["active"] == "v2"


def test_rollback_returns_to_previous_version():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _FakeScorer(f"v{calls['n']}")

    app = _make_app(_sec(), factory)
    with TestClient(app) as c:
        c.get("/health", headers=ADMIN)          # v1
        c.post("/admin/reload", headers=ADMIN)   # -> v2 (active)
        r = c.post("/admin/rollback", headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["active"] == "v1"

    # 再回滚：已清空历史
    with TestClient(app) as c:
        r2 = c.post("/admin/rollback", headers=ADMIN)
    assert r2.status_code == 409


def test_reload_failure_keeps_old_model():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeScorer("dummy-v1")
        raise RuntimeError("装配失败：数据目录没了")

    app = _make_app(_sec(), factory)
    with TestClient(app) as c:
        assert c.get("/health", headers=ADMIN).json()["model"] == "dummy-v1"
        r = c.post("/admin/reload", headers=ADMIN)
    assert r.status_code == 500
    with TestClient(app) as c:
        assert c.get("/health", headers=ADMIN).json()["model"] == "dummy-v1"


def test_admin_endpoints_are_rate_limited_and_audited(tmp_path):
    db = tmp_path / "audit.sqlite3"
    # rate 低到在一次测试里来不及补桶：burst=2 → 第 3 个 reload 必 429
    sec = Security(SecurityConfig(api_keys=["k-user"], admin_keys=["k-admin"],
                                  disabled=False, audit_path=str(db),
                                  rate_rps=0.1, rate_burst=2,
                                  min_interval=0.0))
    app = _make_app(sec, lambda: _FakeScorer("v1"))
    with TestClient(app) as c:
        c.get("/health", headers=ADMIN)
        # burst=2，第三个 reload 应 429（同一 admin key）
        assert c.post("/admin/reload", headers=ADMIN).status_code == 200
        assert c.post("/admin/reload", headers=ADMIN).status_code == 200
        assert c.post("/admin/reload", headers=ADMIN).status_code == 429
    rows = sec.audit.recent(action="model.switch")
    # reload 也记 model.switch（admin_reload 未走 _switch_to，改用其它动作）
    assert len(sec.audit.recent(action="model.reload")) >= 2
    assert len(rows) == 0  # reload 路径不产生 model.switch


def test_health_after_hotswap_reports_new_version():
    """/health 必须反映 active 版本，而不是缓存旧信息。"""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _FakeScorer(f"v{calls['n']}")

    app = _make_app(_sec(), factory)
    with TestClient(app) as c:
        assert c.get("/health", headers=ADMIN).json()["model"] == "v1"
        c.post("/admin/reload", headers=ADMIN)
        assert c.get("/health", headers=ADMIN).json()["model"] == "v2"
