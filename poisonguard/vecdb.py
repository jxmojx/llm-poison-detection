"""向量库集成（P1 事实核验/语义检索的索引层）：无重依赖的余弦最近邻。

为什么需要：
  - ragcheck.py 的 _topk_by_cosine 是每次全量循环（O(N*d)），参考语料 12000
    条尚可，扩大到全量文档入库后必须有一个可持久化的向量索引。
  - 本模块提供两层后端：
      * numpy 暴力余弦（`brute`，默认）：无额外依赖，N<=~50k 性能足够；
      * sklearn NearestNeighbors（`sklearn`）：精确最近邻 + 索引复用，
        大库更快；sklearn 在项目栈里已有。
  - 与 chunking/parsers 对接：解析出的文档 → 分块 → 编码 → add 入库；
    query 取回 top-k，交给 ragcheck 或上游守卫做一致性核验。

持久化：`persist` 写一个 npz（vectors + ids + 元数据），`load` 读回。
纯逻辑：暴力和 sklearn 后端都用随机向量即可 CI 验证，不依赖模型。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _l2normalize(m: np.ndarray) -> np.ndarray:
    """行向量按 L2 规范化（cosine 距离等价于规范化后内积）。"""
    m = np.asarray(m, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


@dataclass
class VecStore:
    """按 id 存向量的余弦最近邻存储。

    dim: 向量维数（add 时一致校验，不一致抛 ValueError）。
    backend: "brute" | "sklearn"。
    ids: list[str]（并发修改由调用方保证单线程）。
    vectors: np.ndarray [N, dim]（规范化存储）。
    meta: dict 附带的全局元数据（如 encoder 名/建库时间）。
    """

    dim: int = 0
    backend: str = "brute"
    ids: list = field(default_factory=list)
    vectors: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def create(cls, dim: int = 0, backend: str = "brute",
               meta: dict | None = None) -> "VecStore":
        return cls(dim=dim, backend=backend, ids=[],
                   vectors=None, meta=dict(meta or {}))

    def _ensure_dim(self, m: np.ndarray) -> int:
        m = np.asarray(m, dtype=np.float64)
        if m.ndim != 2:
            raise ValueError(f"向量需为二维数组，收到 {m.ndim} 维")
        if self.dim and m.shape[1] != self.dim:
            raise ValueError(
                f"向量维度不匹配：库 {self.dim}d，收到 {m.shape[1]}d")
        if not self.dim:
            self.dim = m.shape[1]
        return m.shape[0]

    def add(self, ids: list[str], vectors: np.ndarray) -> int:
        """批量入库（id 去重覆盖）。返回实际新增数。"""
        if len(ids) != len(vectors):
            raise ValueError("ids 与 vectors 长度不一致")
        n = self._ensure_dim(vectors)
        v = _l2normalize(vectors)
        if self.vectors is None:
            self.vectors = v
            self.ids = list(ids)
            return n
        # 去重：已有 id 覆盖，新增追加
        existing = set(self.ids)
        new_v = []
        new_ids = []
        for i, _id in enumerate(ids):
            if _id in existing:
                idx = self.ids.index(_id)
                self.vectors[idx] = v[i]
            else:
                new_ids.append(_id)
                new_v.append(v[i])
                existing.add(_id)
        if new_v:
            self.vectors = np.concatenate([self.vectors, np.asarray(new_v)], axis=0)
            self.ids.extend(new_ids)
        return len(new_ids)

    def __len__(self) -> int:
        return 0 if self.vectors is None else len(self.ids)

    def _brute_query(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = _l2normalize(q.reshape(1, -1))
        sims = self.vectors @ q.T  # [N,1] cosine（都已规范化）
        scores = sims[:, 0]
        topk = min(k, len(scores))
        idx = np.argpartition(-scores, topk - 1)[:topk]
        # argpartition 无序 → argsort 排序
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

    def _sklearn_query(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors

        q = q.reshape(1, -1)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(self.vectors)
        dists, idx = nn.kneighbors(q)
        idx = idx[0]
        # 余弦距离 → 相似度（1-d）
        scores = 1.0 - dists[0]
        return idx, scores

    def query(self, vector: np.ndarray, k: int = 5) -> list[dict]:
        """返回 [{id, score, idx}] 按分数降序。"""
        if self.vectors is None or not len(self):
            return []
        k = min(max(k, 1), len(self))
        if self.backend == "sklearn":
            idx, scores = self._sklearn_query(vector, k)
        else:
            idx, scores = self._brute_query(vector, k)
        return [{"idx": int(i), "id": self.ids[int(i)], "score": float(s)}
                for i, s in zip(idx, scores)]

    def delete(self, ids: list[str]) -> int:
        """删除一批 id，返回删除数。"""
        if self.vectors is None:
            return 0
        del_set = set(ids)
        keep = [i for i, _id in enumerate(self.ids) if _id not in del_set]
        removed = len(self.ids) - len(keep)
        if removed:
            self.ids = [self.ids[i] for i in keep]
            self.vectors = self.vectors[keep]
        return removed

    def get(self, _id: str) -> np.ndarray | None:
        if _id in self.ids:
            return self.vectors[self.ids.index(_id)]
        return None

    def persist(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.vectors is None:
            raise ValueError("空库无可持久化")
        np.savez_compressed(
            path,
            ids=np.asarray(self.ids, dtype=object),
            vectors=self.vectors.astype(np.float32),
            meta=np.asarray(json.dumps({**self.meta, "dim": self.dim,
                                        "backend": self.backend},
                                       ensure_ascii=False), dtype=object),
        )
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "VecStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"向量库不存在: {path}")
        z = np.load(path, allow_pickle=True)
        ids = [str(x) for x in z["ids"]]
        meta = json.loads(z["meta"].item()) if z["meta"].size else {}
        st = cls(dim=meta.pop("dim", z["vectors"].shape[1]),
                 backend=meta.pop("backend", "brute"), ids=ids,
                 vectors=np.asarray(z["vectors"]), meta=meta)
        return st

    def stats(self) -> dict:
        return {"n": len(self), "dim": self.dim, "backend": self.backend,
                "meta": dict(self.meta), "size_bytes": 0 if self.vectors is None
                else int(self.vectors.nbytes)}


def bench_query(store: VecStore, queries: np.ndarray, k: int = 5,
                n_repeat: int = 3) -> dict:
    """吞吐小压测：报告单查询均耗时与 QPS（供 gap 6 汇入）。"""
    times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for q in queries:
            store.query(q, k=k)
        times.append(time.perf_counter() - t0)
    per_ms = [t / max(len(queries), 1) * 1000 for t in times]
    per_ms_sorted = sorted(per_ms)
    p50 = per_ms_sorted[len(per_ms_sorted) // 2]
    p99 = per_ms_sorted[-1] if len(per_ms_sorted) > 1 else p50
    qps = 1000.0 / p50 if p50 > 0 else float("inf")
    return {"n_queries": len(queries), "k": k, "n_repeat": n_repeat,
            "mean_ms": float(np.mean(per_ms)), "p50_ms": float(p50),
            "p99_ms": float(p99), "qps": float(qps)}
