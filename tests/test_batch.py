"""方向③ 缺口3：异步批处理批量打分 —— 离线契约测试。

走 fastapi TestClient（不联网、不加载模型）：
- POST /batch 提交即回 job_id（不阻塞），texts 超 MAX_BATCH 自动切块
- GET  /batch/{job_id} 轮询状态：queued → running → done，返回全部 results
- 失败时 job 状态 failed 且带 error；未知 job_id 404
- 空 texts 400；未鉴权 401
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from poisonguard.config import load_config
from poisonguard.security import Security, SecurityConfig
from poisonguard.service import MAX_BATCH, create_app


class _FakeScorer:
    threshold = 0.5
    knn = object()
    ngram = None
    student = None

    def __init__(self, version: str = "batch-fake"):
        self.version = version

    def score(self, texts, explain=False):
        res = [{"score": 0.9, "flag": True, "views": {"knn": 0.5, "fpr_est": 0.5,
                                                      "char_ngram": 0.5,
                                                      "student": 0.9,
                                                      "chars": len(t)}}
               for t in texts]
        return {"results": res, "n": len(texts), "latency_ms": 1.0,
                "ms_per_item": 0.5, "threshold": self.threshold,
                "model": self.version}

    def stats(self):
        return {"uptime_s": 1.0, "n_calls": 0, "n_texts": 0,
                "ms_per_item_p50": 0.1, "ms_per_item_p99": 0.2,
                "threshold": self.threshold, "model": self.version}


def _sec(**kw) -> Security:
    kw.setdefault("api_keys", ["k-user"])
    kw.setdefault("admin_keys", ["k-admin"])
    kw.setdefault("disabled", False)
    kw.setdefault("audit_path", None)
    kw.setdefault("min_interval", 0.0)
    return Security(SecurityConfig(**kw))


def _make_app(security, scorer=None, factory=None):
    return create_app(load_config(), scorer=scorer, security=security,
                      scorer_factory=factory)


KEY = {"X-API-Key": "k-user"}
ADMIN = {"X-API-Key": "k-admin"}


def _poll(client, job_id, timeout=5.0) -> dict:
    """轮询直到 done/failed 或超时。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/batch/{job_id}", headers=KEY)
        assert r.status_code == 200
        state = r.json()["state"]
        if state in ("done", "failed", "cancelled"):
            return r.json()
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} 未在 {timeout}s 内结束")


def test_batch_submit_returns_job_id_and_poll_done():
    app = _make_app(_sec(), scorer=_FakeScorer())
    with TestClient(app) as c:
        r = c.post("/batch", json={"texts": ["aa", "bb"]}, headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert body["total"] == 2
    job_id = body["job_id"]
    assert job_id

    with TestClient(app) as c:
        done = _poll(c, job_id)
    assert done["state"] == "done"
    assert done["total"] == 2
    assert done["done"] == 2
    assert len(done["results"]) == 2
    assert all(x["flag"] for x in done["results"])


def test_batch_chunks_over_max_batch():
    """texts 超过 MAX_BATCH 时服务端切块，结果数与总数一致。"""
    n = MAX_BATCH + 5  # 2 个块
    texts = [f"t{i}" for i in range(n)]
    app = _make_app(_sec(), scorer=_FakeScorer())
    with TestClient(app) as c:
        r = c.post("/batch", json={"texts": texts}, headers=KEY)
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        done = _poll(c, job_id, timeout=10.0)
    assert done["state"] == "done"
    assert done["done"] == n
    assert len(done["results"]) == n


def test_batch_empty_texts_400():
    app = _make_app(_sec(), scorer=_FakeScorer())
    with TestClient(app) as c:
        r = c.post("/batch", json={"texts": []}, headers=KEY)
    assert r.status_code == 400


def test_batch_requires_auth():
    app = _make_app(_sec(), scorer=_FakeScorer())
    with TestClient(app) as c:
        assert c.post("/batch", json={"texts": ["a"]}).status_code == 401
        assert c.get("/batch/whatever").status_code == 401


def test_batch_unknown_job_404():
    app = _make_app(_sec(), scorer=_FakeScorer())
    with TestClient(app) as c:
        r = c.get("/batch/does-not-exist", headers=KEY)
    assert r.status_code == 404


def test_batch_failure_reports_failed_state():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        raise RuntimeError("装配失败：数据目录没了")

    app = _make_app(_sec(), factory=factory)
    with TestClient(app) as c:
        r = c.post("/batch", json={"texts": ["aa", "bb"]}, headers=KEY)
        assert r.status_code == 200
        done = _poll(c, r.json()["job_id"])
    assert done["state"] == "failed"
    assert done["error"]


def test_batch_list_reports_running_jobs():
    app = _make_app(_sec(), scorer=_FakeScorer())
    with TestClient(app) as c:
        r = c.post("/batch", json={"texts": ["aa"]}, headers=KEY)
        job_id = r.json()["job_id"]
        listing = c.get("/batch/", headers=KEY).json()
    assert listing["jobs"]
    ids = {j["job_id"] for j in listing["jobs"]}
    assert job_id in ids
