"""检测器族 + 阈值校准。

所有检测器遵循同一接口：
    fit(X_clean, texts_clean)   # 只用**干净**数据拟合(无监督/半监督)
    score(X, texts) -> np.ndarray   # 分数越大越可疑
    calibrate(clean_scores, target_fpr) -> 阈值

这样做的意义：
- 阈值不再拍脑袋(原项目 KL_THRESHOLD=0.5 / GRAD_NORM_THRESHOLD=0.5)，
  而是按"目标误报率"在验证集上校准，可用 TPR@FPR 直接比较不同检测器。
- 融合检测器用分位数映射把各检测器分数统一到"误报率"尺度再加权，
  避免量纲不一致，阈值也天然是 FPR。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
class BaseDetector:
    name = "base"
    requires_text = False

    def fit(self, X_clean: np.ndarray, texts_clean: list[str]) -> "BaseDetector":
        return self

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


class KeywordDetector(BaseDetector):
    """复刻原项目的规则法：命中任一关键词即判为污染。

    这是重要的**弱基线**——它代表"人工规则"的上限，也是原项目 100% 检测率的来源。
    """

    name = "keyword"
    requires_text = True

    def __init__(self, keywords: list[str]):
        self.keywords = list(keywords)

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        return np.array(
            [float(sum(1 for kw in self.keywords if kw in t)) for t in texts],
            dtype=np.float32,
        )


class CentroidDetector(BaseDetector):
    """语义质心：与干净样本均值的余弦距离。最简单，但常常出乎意料地强。"""

    name = "centroid"

    def fit(self, X_clean: np.ndarray, texts_clean: list[str]) -> "CentroidDetector":
        c = X_clean.mean(axis=0)
        self.centroid = c / max(float(np.linalg.norm(c)), 1e-8)
        return self

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        return 1.0 - (X @ self.centroid)


class KNNDetector(BaseDetector):
    """k 近邻距离：到干净集最近 k 个样本的平均余弦距离。

    比质心更细：能发现"贴着正常簇边缘"的样本，也更能发现多峰分布里的少数异常。
    """

    name = "knn"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X_clean: np.ndarray, texts_clean: list[str]) -> "KNNDetector":
        from sklearn.neighbors import NearestNeighbors

        self.k_eff = max(1, min(self.k, len(X_clean)))
        self.nn = NearestNeighbors(n_neighbors=self.k_eff, metric="cosine").fit(X_clean)
        return self

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        dist, _ = self.nn.kneighbors(X)
        return dist.mean(axis=1).astype(np.float32)


class MahalanobisDetector(BaseDetector):
    """马氏距离：考虑干净分布的各向异性(PCA 降维 + Ledoit-Wolf 收缩协方差)。

    能抓住"在某个方向上偏差很大、但余弦上看起来正常"的样本，
    例如后门触发词造成的整体分布偏移。
    """

    name = "mahalanobis"

    def __init__(self, pca_dim: int = 64):
        self.pca_dim = pca_dim

    def fit(self, X_clean: np.ndarray, texts_clean: list[str]) -> "MahalanobisDetector":
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA

        dim = min(self.pca_dim, X_clean.shape[1], max(2, len(X_clean) // 4))
        self.pca = PCA(n_components=dim, random_state=0).fit(X_clean)
        Z = self.pca.transform(X_clean)
        self.lw = LedoitWolf().fit(Z)
        self.mean = self.lw.location_
        self.prec = self.lw.precision_
        return self

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        Z = self.pca.transform(X) - self.mean
        d = np.einsum("ij,jk,ik->i", Z, self.prec, Z)
        return np.sqrt(np.clip(d, 0, None)).astype(np.float32)


class IsolationForestDetector(BaseDetector):
    """原项目里声明但从未 fit 的孤立森林——这里把它真正实现，作为对照。"""

    name = "iforest"

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X_clean: np.ndarray, texts_clean: list[str]) -> "IsolationForestDetector":
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            contamination="auto",
        ).fit(X_clean)
        return self

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        return -self.model.score_samples(X).astype(np.float32)  # 越大越异常


class CallableDetector(BaseDetector):
    """包装函数型打分器(如受害模型的 perplexity、Qwen 判官)。"""

    def __init__(self, name: str, fn, requires_text: bool = True):
        self.name = name
        self.fn = fn
        self.requires_text = requires_text

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        return np.asarray(self.fn(texts), dtype=np.float32)


class SupervisedDetector(BaseDetector):
    """有监督检测器：用 dev 上已标注的攻击样本训练逻辑回归。

    为什么需要它？
        无监督检测器回答的是"这句话离正常分布有多远"，
        但有标注样本时（安全团队总会积累一些已知攻击），**监督信号更强**。
        很多真实系统就是这么做的：规则+已知攻击库训练一个分类器，再看它对
        *没见过的攻击字符串* 的泛化。本项目的 dev/test 攻击字符串按内容哈希分桶、
        完全不相交，所以这个泛化评估是成立的。

    阈值校准必须用 **out-of-fold** 分数：
        如果直接用 dev 上的 in-sample 分数校准，阈值会偏乐观、test 上的真实
        FPR 会高于目标值。这里用 5 折交叉预测得到每个 dev 样本的"未见过"分数。
    """

    name = "sup_lr"

    def __init__(self, C: float = 1.0, max_iter: int = 3000, seed: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.clf = None

    def fit_labeled(self, X: np.ndarray, y: np.ndarray) -> "SupervisedDetector":
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                      random_state=self.seed).fit(X, y)
        return self

    def oof_scores(self, X: np.ndarray, y: np.ndarray, folds: int = 5) -> np.ndarray:
        """5 折交叉预测，返回每个样本的"没见过"分数，用于无偏校准。"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(y)
        out = np.zeros(len(y), dtype=np.float32)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                     random_state=self.seed).fit(X[tr], y[tr])
            out[te] = clf.predict_proba(X[te])[:, 1]
        return out

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("SupervisedDetector 需要先 fit_labeled(X, y)")
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)


class CharNgramDetector(BaseDetector):
    """字符 n-gram + 逻辑回归：中文短文本的经典强基线(不需要分词)。

    为什么值得放进对比表？
        它代表"纯词面特征"的上限。如果它在 paraphrase/subtle 家族上明显低于
        语义模型，就能证明"这些攻击确实规避了词面特征"；反过来如果它也很高，
        说明攻击样本仍带词面痕迹——**两种结论都必须让数据说话**，不能靠假设。
    `analyzer="char_wb"` 对中文按字符滑窗，天然绕开分词问题(原项目 `text.split()`
    正是死在这里)。
    """

    name = "sup_tfidf"
    requires_text = True

    def __init__(self, ngram_range: tuple[int, int] = (2, 3), min_df: int = 2,
                 max_features: int = 200_000, C: float = 2.0, seed: int = 0):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.C = C
        self.seed = seed
        self.vec = None
        self.clf = None

    def _make(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range,
                              min_df=self.min_df, max_features=self.max_features,
                              sublinear_tf=True)
        clf = LogisticRegression(C=self.C, max_iter=3000, random_state=self.seed)
        return vec, clf

    def fit_labeled(self, texts: list[str], y: np.ndarray) -> "CharNgramDetector":
        self.vec, self.clf = self._make()
        Z = self.vec.fit_transform(texts)
        self.clf.fit(Z, y)
        return self

    def oof_scores(self, texts: list[str], y: np.ndarray, folds: int = 5) -> np.ndarray:
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(y)
        out = np.zeros(len(y), dtype=np.float32)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        for tr, te in skf.split(np.zeros(len(y)), y):
            vec, clf = self._make()
            Ztr = vec.fit_transform([texts[i] for i in tr])
            clf.fit(Ztr, y[tr])
            out[te] = clf.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]
        return out

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("CharNgramDetector 需要先 fit_labeled(texts, y)")
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1].astype(np.float32)


# ----------------------------------------------------------------------
@dataclass
class Calibration:
    """把"原始分数"映射到"误报率(FPR)"尺度的参考分布。

    to_fpr(x) = 参考集里小于等于 x 的比例。阈值目标 FPR=1% 时，
    直接判 score 的 fpr 估计 >= 0.01 为阳性即可。
    """

    ref_sorted: np.ndarray
    target_fpr: float
    raw_threshold: float
    achieved_fpr: float

    @classmethod
    def fit(cls, clean_scores: np.ndarray, target_fpr: float) -> "Calibration":
        ref = np.sort(np.asarray(clean_scores, dtype=np.float64))
        if len(ref) == 0:
            raise ValueError("校准集为空")
        idx = int(round((1.0 - target_fpr) * (len(ref) - 1)))
        thr = float(ref[idx])
        # 用严格大于统计，与 predict() 保持一致；
        # 关键词这类离散分数在干净 dev 上全为 0，阈值即 0 → dev FPR 天然为 0。
        achieved = float((ref > thr).mean())
        return cls(ref, target_fpr, thr, achieved)

    def to_fpr(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        return np.searchsorted(self.ref_sorted, s, side="left") / max(len(self.ref_sorted), 1)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (np.asarray(scores) > self.raw_threshold).astype(int)


# ----------------------------------------------------------------------
class FusionDetector(BaseDetector):
    """分数融合：各成员先做分位数映射到 FPR 尺度，再加权求和。

    权重在 dev 上用网格搜索(AUROC 目标)选出，完全不接触 test。
    """

    name = "fusion"

    def __init__(self, members: list[BaseDetector], weight_grid=(0.0, 1.0, 2.0)):
        self.members = members
        self.weight_grid = weight_grid
        self.weights: np.ndarray | None = None

    def fit_fusion(
        self,
        dev_scores: dict[str, np.ndarray],
        dev_labels: np.ndarray,
        calibrations: dict[str, Calibration],
    ) -> "FusionDetector":
        from itertools import product

        from .evaluate import auroc

        names = [m.name for m in self.members]
        mapped = {n: calibrations[n].to_fpr(dev_scores[n]) for n in names}
        best = (-1.0, None)
        for combo in product(self.weight_grid, repeat=len(names)):
            if sum(combo) == 0:
                continue
            fused = sum(w * mapped[n] for w, n in zip(combo, names))
            s = auroc(dev_labels, fused)
            if s > best[0]:
                best = (s, combo)
        self.weights = np.array(best[1], dtype=float) if best[1] else np.ones(len(names))
        self.dev_auroc = best[0]
        return self

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        raise NotImplementedError("请用 fuse(), 需要各成员的 Calibration")

    def fuse(self, mapped: dict[str, np.ndarray]) -> np.ndarray:
        names = [m.name for m in self.members]
        out = np.zeros(len(next(iter(mapped.values()))), dtype=np.float64)
        for w, n in zip(self.weights, names):
            out += w * mapped[n]
        return out.astype(np.float32)


# ----------------------------------------------------------------------
def build_detectors(cfg, enc_dim: int) -> dict[str, BaseDetector]:
    dcfg = cfg.detector
    out: dict[str, BaseDetector] = {}
    for name in dcfg.enable:
        if name == "keyword":
            out[name] = KeywordDetector(dcfg.keywords)
        elif name == "centroid":
            out[name] = CentroidDetector()
        elif name == "knn":
            out[name] = KNNDetector(dcfg.knn_k)
        elif name == "mahalanobis":
            out[name] = MahalanobisDetector(dcfg.pca_dim)
        elif name == "iforest":
            out[name] = IsolationForestDetector()
        elif name == "sup_lr":
            out[name] = SupervisedDetector()
        elif name == "sup_tfidf":
            out[name] = CharNgramDetector()
        elif name == "fusion":
            continue  # 依赖其它成员，最后构造
        else:
            raise ValueError(f"未知检测器: {name}")
    return out
