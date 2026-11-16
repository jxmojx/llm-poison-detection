"""方向③ 缺口4：Prometheus /metrics 手写文本格式 —— 离线契约测试。

验证：
- 无依赖指标收集器（Counter / Gauge / Histogram）语义正确：
  histogram bucket 累积、_sum/_count、label 化
- /metrics 端点输出合法 Prometheus 文本（带 # HELP/# TYPE、HTTP 计数、
  score 计数与延迟、batch 计数、版本切换计数、model/component/uptime gauge）
- 未鉴权 /metrics 401；主体行可被 Prometheus 抓取
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from poisonguard.config import load_config
from poisonguard.metrics import (
    Counter, Gauge, Histogram, REGISTRY, render, reset,
    batch_jobs_total, batch_running, batch_submit_total,
    component_present, http_requests_total, model_info, score_errors_total,
    score_latency_seconds, score_requests_total, uptime_seconds,
    version_switch_total,
)
from poisonguard.security import Security, SecurityConfig
from poisonguard.service import create_app


@pytest.fixture(autouse=True)
def _reset_metrics():
    """指标是进程级单例（模块 REGISTRY），跨测试会累积；每个用例前归零。"""
    reset()
    yield
    reset()


class _FakeScorer:
    threshold = 0.5
    knn = object()
    ngram = None
    student = None

    def __init__(self, version: str = "metrics-fake"):
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


KEY = {"X-API-Key": "k-user"}
ADMIN = {"X-API-Key": "k-admin"}


# ---------------------------------------------------------------- 收集器单测

def test_counter_accumulates_with_labels():
    c = Counter("t_counter", "test counter", labels=("method",))
    c.inc(method="GET")
    c.inc(method="GET")
    c.add(3.0, method="POST")
    txt = "\n".join(c.render())
    assert "# TYPE t_counter counter" in txt
    assert 't_counter{method="GET"} 2' in txt
    assert 't_counter{method="POST"} 3' in txt


def test_gauge_sets_values():
    g = Gauge("t_gauge", "test gauge")
    g.set(3.5)
    assert "t_gauge 3.5" in "\n".join(g.render())
    g.set(-1.0)
    assert "t_gauge -1" in "\n".join(g.render())


def test_histogram_is_cumulative_with_sum_count():
    h = Histogram("t_hist", "test hist", buckets=(0.01, 0.1, 1.0))
    h.observe(0.05)
    h.observe(0.5)
    h.observe(2.0)
    txt = "\n".join(h.render())
    # 累积：<=0.01 只有 0；<=0.1 有 1 个；<=1.0 有 2 个；<=+Inf 有 3 个
    assert 't_hist_bucket{le="0.01"} 0' in txt
    assert 't_hist_bucket{le="0.1"} 1' in txt
    assert 't_hist_bucket{le="1"} 2' in txt
    assert 't_hist_bucket{le="+Inf"} 3' in txt
    assert "t_hist_sum 2.55" in txt
    assert "t_hist_count 3" in txt


def test_default_histogram_buckets_render():
    h = Histogram("t_h2", "default buckets")
    h.observe(0.5)
    txt = "\n".join(h.render())
    assert 't_h2_bucket{le="0.5"} 1' in txt
    assert 't_h2_bucket{le="+Inf"} 1' in txt
    # 不污染全局 REGISTRY
    assert "t_h2" not in render()


def test_registry_render_sorted():
    assert "# HELP poisonguard_http_requests_total" in render()
    # 每个注册指标都有 HELP 和 TYPE
    for name in REGISTRY:
        assert f"# HELP {name} " in render() or f"# TYPE {name} " in render()


# ---------------------------------------------------------------- /metrics 端点

def test_metrics_requires_auth():
    app = create_app(load_config(), scorer=_FakeScorer(),
                     security=_sec())
    with TestClient(app) as c:
        assert c.get("/metrics").status_code == 401


def test_metrics_endpoint_exposes_prometheus_text():
    app = create_app(load_config(), scorer=_FakeScorer(),
                     security=_sec())
    with TestClient(app) as c:
        # 制造业务指标：2 次 /score, 1 次 /batch
        c.post("/score", json={"texts": ["a", "b"]}, headers=KEY)
        c.post("/score", json={"texts": ["c"]}, headers=KEY)
        resp = c.get("/metrics", headers=KEY)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    txt = resp.text

    # HTTP 计数（含 /metrics 自身）
    assert 'poisonguard_http_requests_total{method="POST",path="/score",' in txt

    # score 业务指标
    assert "poisonguard_score_requests_total 2" in txt
    assert "poisonguard_score_texts_total 3" in txt
    # 直方图（延迟被观测了至少 2 次）
    assert "poisonguard_score_latency_seconds_count" in txt

    # 实例/metadata gauge
    assert "poisonguard_uptime_seconds " in txt
    assert 'poisonguard_model_info{version="metrics-fake"} 1' in txt
    assert 'poisonguard_component_present{name="knn"} 1' in txt
    assert 'poisonguard_component_present{name="student"} 0' in txt


def test_metrics_tracks_version_switch(tmp_path):
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _FakeScorer(f"v{calls['n']}")

    app = create_app(load_config(), scorer_factory=factory,
                     security=_sec())
    with TestClient(app) as c:
        c.get("/health", headers=ADMIN)            # 装配 v1
        c.post("/admin/reload", headers=ADMIN)     # -> v2
        txt = c.get("/metrics", headers=ADMIN).text
    assert "poisonguard_version_switch_total 1" in txt
    assert 'poisonguard_model_info{version="v2"} 1' in txt


def test_metrics_tracks_batch(tmp_path):
    import time

    app = create_app(load_config(), scorer=_FakeScorer(),
                     security=_sec())
    with TestClient(app) as c:
        r = c.post("/batch", json={"texts": ["a", "b"]}, headers=KEY)
        job_id = r.json()["job_id"]
        deadline = time.time() + 5.0
        while time.time() < deadline:
            st = c.get(f"/batch/{job_id}", headers=KEY).json()["state"]
            if st in ("done", "failed"):
                break
            time.sleep(0.01)
        txt = c.get("/metrics", headers=KEY).text
    assert "poisonguard_batch_submit_total 1" in txt
    assert "poisonguard_batch_texts_total 2" in txt
    assert 'poisonguard_batch_jobs_total{state="done"} 1' in txt
