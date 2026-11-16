"""零依赖 Prometheus 文本格式（4.0 exposition format）指标收集器。

设计取舍
--------
* 不引 `prometheus_client`：serve 保持离线可跑、无重依赖；
  只用标准库实现 Counter / Gauge / Histogram 的最小可用子集。
* `/metrics` 直接把 `render()` 文本吐出去，Prometheus / Grafana 可直接抓。
* 线程安全：所有指标都在内部持锁，fastapi 多线程/批处理 worker 并发安全。
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock


def _fmt(v: float) -> str:
    return f"{v:g}"


class Metric:
    """Counter/Gauge/Histogram 的公共骨架：label 化的值集合 + 渲染。"""

    def __init__(self, name: str, doc: str, mtype: str,
                 labels: tuple[str, ...] = ()):
        self.name = name
        self.doc = doc
        self.mtype = mtype
        self._label_names = tuple(labels)   # 注意：labels 是方法，不能用同名属性
        self._values: dict[tuple[str, ...], float] = defaultdict(float)
        self._lock = Lock()

    def _key(self, lv: dict) -> tuple[str, ...]:
        return tuple(str(lv.get(l, "")) for l in self._label_names)

    def labels(self, **kw) -> "_LabelView":
        """prometheus_client 风格：labels(method="GET").inc()。

        返回一个绑定好 label 值的视图，后续只加/取/替换数值。
        """
        return _LabelView(self, self._key(kw))

    def _get(self, lv: dict) -> float:
        with self._lock:
            return float(self._values[self._key(lv)])

    def _set(self, value: float, lv: dict) -> None:
        with self._lock:
            self._values[self._key(lv)] = value

    def _add(self, value: float, lv: dict) -> None:
        with self._lock:
            self._values[self._key(lv)] += value

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.doc}",
                 f"# TYPE {self.name} {self.mtype}"]
        with self._lock:
            for key, val in sorted(self._values.items()):
                if self._label_names:
                    parts = ",".join(f'{k}="{v}"'
                                     for k, v in zip(self._label_names, key))
                    lines.append(f"{self.name}{{{parts}}} {_fmt(val)}")
                else:
                    lines.append(f"{self.name} {_fmt(val)}")
        return lines


class _LabelView:
    """labels(...) 的绑定视图：持有所属 metric 与固定 label key。"""

    __slots__ = ("_metric", "_key")

    def __init__(self, metric: Metric, key: tuple[str, ...]):
        self._metric = metric
        self._key = key

    def _opts(self, lv: dict = {}) -> dict:
        # 把 tuple 恢复成 {label名: 值}，保证与 metric._key 的次序一致
        return {n: v for n, v in zip(self._metric._label_names, self._key)}

    def inc(self, value: float = 1.0) -> None:
        self._metric._add(value, self._opts())

    def add(self, value: float) -> None:
        self._metric._add(value, self._opts())

    def set(self, value: float) -> None:
        self._metric._set(value, self._opts())


class Counter(Metric):
    def __init__(self, name: str, doc: str, labels: tuple[str, ...] = ()):
        super().__init__(name, doc, "counter", labels)

    def inc(self, value: float = 1.0, **lv) -> None:
        self._add(value, lv)

    def add(self, value: float, **lv) -> None:
        self._add(value, lv)


class Gauge(Metric):
    def __init__(self, name: str, doc: str, labels: tuple[str, ...] = ()):
        super().__init__(name, doc, "gauge", labels)

    def set(self, value: float, **lv) -> None:
        self._set(value, lv)

    def inc(self, value: float = 1.0, **lv) -> None:
        self._add(value, lv)


class Histogram(Metric):
    """线性 buckets 直方图（给耗时这种右偏分布用）。

    `observe(x)` 把 x 归进最近的 bucket；渲染时给出 bucket 计数 +
    `_sum` / `_count`，PromQL `histogram_quantile` 可直接出分位延迟。
    """

    DEFAULT_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                       1.0, 2.0, 5.0, 10.0)

    def __init__(self, name: str, doc: str, *,
                 buckets: tuple[float, ...] = DEFAULT_BUCKETS):
        # 直方图整体用 name 命名，内部 bucket 用 name_bucket{le=...}
        super().__init__(name, doc, "histogram", labels=("le",))
        self._sum_name = f"{name}_sum"
        self._count_name = f"{name}_count"
        # Prometheus le 标签规范：有限值用 %g，+Inf 写 "+Inf"
        self._buckets = tuple((_fmt(float(b)), float(b))
                              for b in sorted(buckets)) + (("+Inf", float("inf")),)
        self._sum: float = 0.0
        self._count: int = 0

    def observe(self, value: float) -> None:
        v = float(value)
        with self._lock:
            self._count += 1
            self._sum += v
            # Prometheus histogram 的 bucket 是累积的：每个 le=b 统计所有 <=b 的样本。
            # 因此一次观察要给所有 bound>=v 的 bucket 各 +1（无 break）。
            for _, bound in self._buckets:
                if v <= bound:
                    self._values[(self._le_label(bound),)] += 1

    @staticmethod
    def _le_label(bound: float) -> str:
        return "+Inf" if bound == float("inf") else _fmt(bound)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.doc}",
                 f"# TYPE {self.name} histogram"]
        with self._lock:
            vals = dict(self._values)
            s, c = self._sum, self._count
        for le, bound in self._buckets:
            label = self._le_label(bound)
            lines.append(f"{self.name}_bucket{{le=\"{label}\"}}"
                         f" {_fmt(vals.get((label,), 0.0))}")
        lines.append(f"{self._sum_name} {_fmt(s)}")
        lines.append(f"{self._count_name} {c}")
        return lines


# ---------------------------------------------------------------------------
# 注册表：进程内所有指标的名字 -> 对象；/metrics 按名字排序渲染。
# ---------------------------------------------------------------------------
REGISTRY: dict[str, Metric] = {}


def register(m: Metric) -> Metric:
    REGISTRY[m.name] = m
    return m


def render() -> str:
    """整片 Prometheus 文本（末尾带换行，符合抓取规范）。"""
    parts: list[str] = []
    for name in sorted(REGISTRY):
        parts.extend(REGISTRY[name].render())
    return "\n".join(parts) + "\n"


def reset() -> None:
    """清空所有指标的值（测试隔离 / 进程内热重置用）。

    指标定义（name/doc/type/labels）保留，只是数值归零 —— 与进程启动时一致。
    """
    for m in REGISTRY.values():
        with m._lock:
            m._values.clear()
            if isinstance(m, Histogram):
                m._sum = 0.0
                m._count = 0


# ---------------------------------------------------------------------------
# 预注册指标：HTTP / 打分 / 批处理 / 模型版本 / 实例信息
# ---------------------------------------------------------------------------
http_requests_total = register(Counter(
    "poisonguard_http_requests_total", "HTTP 请求总数",
    labels=("method", "path", "status")))
score_requests_total = register(Counter(
    "poisonguard_score_requests_total", "POST /score 调用次数"))
score_texts_total = register(Counter(
    "poisonguard_score_texts_total", "POST /score 累计处理文本条数"))
score_errors_total = register(Counter(
    "poisonguard_score_errors_total", "POST /score 打分异常次数"))
score_latency_seconds = register(Histogram(
    "poisonguard_score_latency_seconds", "打分耗时（观测值，秒）"))
batch_submit_total = register(Counter(
    "poisonguard_batch_submit_total", "POST /batch 提交的任务数"))
batch_texts_total = register(Counter(
    "poisonguard_batch_texts_total", "POST /batch 累计提交文本条数"))
batch_jobs_total = register(Counter(
    "poisonguard_batch_jobs_total", "批量任务终态计数", labels=("state",)))
version_switch_total = register(Counter(
    "poisonguard_version_switch_total", "模型版本切换/重载次数"))
uptime_seconds = register(Gauge(
    "poisonguard_uptime_seconds", "进程存活秒数"))
model_info = register(Gauge(
    "poisonguard_model_info", "当前 active 模型版本标记",
    labels=("version",)))
batch_running = register(Gauge(
    "poisonguard_batch_running", "正在执行的批量任务数"))
component_present = register(Gauge(
    "poisonguard_component_present", "组件是否装配（1=有 0=无）",
    labels=("name",)))
