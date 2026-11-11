"""把 LLM 判官蒸馏成毫秒级学生模型。

为什么需要这一层
----------------
语义层(v2 的 `judge.py`)是整套系统唯一能判断"语义是否被改过"的组件，
但它的成本结构在生产里不可接受：

* CPU 上 Qwen2.5-0.5B 单条 1.39s，GPU 也要百毫秒级；
* 每天几十万条训练数据，全量跑一遍 LLM 意味着数千 GPU·小时；
* 更麻烦的是**延迟不可控**：上游数据流水线是同步的，判官一慢整条链路堵住。

所以正确的架构不是"把 LLM 塞进在线路径"，而是：
**离线用 LLM 打软标签 → 在线用学生模型复刻它的判断**。
学生模型只看两路廉价特征——句向量(CoROM，本地 CPU 可跑)与字符 n-gram——
单条成本是微秒到毫秒级，可以随数据量线性扩展。

实现要点
--------
1. **两路视图拼接**：`[标准化后的句向量 | 字符 2-3gram TF-IDF]`。
   句向量负责"语义是否偏移"，n-gram 负责"字面是否异常"（v2 的 audit 已经证明
   合成攻击的 n-gram 分布和真实语料差异很大，这条信号不该被丢掉）。
2. **软标签加权**：teacher 分数越靠近 0.5 越不可信，用 `|s-0.5|*2` 当样本权重，
   让"判官自己也没把握"的样本不主导梯度。
3. **诚实汇报**：同时给出 teacher 与 student 在同一测试切片上的 AUROC / 家族召回，
   以及两者的单条耗时——蒸馏如果掉点太多，就该说明白，而不是只报速度。
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_judge_samples(path: str | Path) -> dict:
    """读取 `judge_samples*.jsonl`（cmd_judge 落盘的逐条打分）。"""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    texts = [r["text"] for r in rows]
    scores = np.array(
        [np.nan if r.get("judge_score") is None else float(r["judge_score"]) for r in rows],
        dtype=np.float64,
    )
    labels = np.array([int(r["label"]) for r in rows], dtype=int)
    families = np.array([str(r.get("family", "?")) for r in rows])
    return {"texts": texts, "scores": scores, "labels": labels, "families": families}


class JudgeStudent:
    """句向量 + 字符 n-gram 两路视图的逻辑回归学生模型。"""

    def __init__(self, C: float = 1.0, max_iter: int = 3000, seed: int = 0,
                 max_features: int = 30000, ngram_range: tuple[int, int] = (2, 3)):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.vec = TfidfVectorizer(
            analyzer="char", ngram_range=ngram_range, min_df=2,
            max_features=max_features, sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=C, max_iter=max_iter, random_state=seed,
            class_weight="balanced", solver="liblinear",
        )
        self.threshold = 0.5
        self.train_time = 0.0

    # ---------------- 特征 ----------------
    def _stack(self, texts, X_emb, fit: bool) -> sparse.csr_matrix:
        X = np.asarray(X_emb, dtype=np.float64)
        if fit:
            dense = self.scaler.fit_transform(X)
            ngram = self.vec.fit_transform(texts)
        else:
            dense = self.scaler.transform(X)
            ngram = self.vec.transform(texts)
        return sparse.hstack([sparse.csr_matrix(dense), ngram]).tocsr()

    # ---------------- 训练 ----------------
    def fit(self, texts, X_emb, teacher_scores, threshold: float = 0.5) -> "JudgeStudent":
        t0 = time.time()
        scores = np.asarray(teacher_scores, dtype=np.float64).ravel()
        ok = np.isfinite(scores)
        if ok.sum() < 20:
            raise ValueError(f"可用软标签太少({int(ok.sum())} 条)，无法蒸馏")
        y = (scores[ok] >= threshold).astype(int)
        if len(set(y.tolist())) < 2:
            raise ValueError("软标签全落在同一侧，无法蒸馏")
        # 判官没把握的样本(分数贴近阈值)降权，避免噪声标签主导拟合
        w = np.clip(np.abs(scores[ok] - threshold) * 2.0, 0.2, 1.0)
        keep = [t for t, k in zip(texts, ok) if k]
        F = self._stack(keep, np.asarray(X_emb)[ok], fit=True)
        self.clf.fit(F, y, sample_weight=w)
        self.threshold = threshold
        self.train_time = time.time() - t0
        return self

    # ---------------- 推理 ----------------
    def score(self, texts, X_emb) -> np.ndarray:
        if len(texts) == 0:
            return np.array([], dtype=np.float64)
        return self.clf.predict_proba(self._stack(list(texts), X_emb, fit=False))[:, 1]

    def timed_score(self, texts, X_emb) -> tuple[np.ndarray, float]:
        """返回 (分数, 单条毫秒数)。"""
        texts = list(texts)
        t0 = time.perf_counter()
        s = self.score(texts, X_emb)
        dt = (time.perf_counter() - t0) * 1000.0 / max(len(texts), 1)
        return s, dt

    # ---------------- 序列化 ----------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f, protocol=4)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "JudgeStudent":
        with Path(path).open("rb") as f:
            return pickle.load(f)


def distill_report(
    student: "JudgeStudent",
    dev: dict,
    test: dict,
    X_dev,
    X_test,
    teacher_sec_per_item: float | None = None,
    fig_path: str | Path | None = None,
) -> dict:
    """在同一测试切片上并列 teacher / student / 第一级 kNN 三者的表现。"""
    from . import evaluate as E

    s_dev = student.score(dev["texts"], X_dev)
    s_test, ms = student.timed_score(test["texts"], X_test)
    y = test["labels"]
    fams = test["families"]
    thr = student.threshold

    class _S:  # recall_by_family 只需要 .label 与 .family
        __slots__ = ("label", "family", "text")

        def __init__(self, label, family, text):
            self.label, self.family, self.text = label, family, text

    samples = [_S(int(a), str(b), t) for a, b, t in zip(y, fams, test["texts"])]

    out: dict = {
        "n_dev": int(len(dev["texts"])),
        "n_test": int(len(test["texts"])),
        "train_sec": round(student.train_time, 3),
        "student_ms_per_item": round(ms, 4),
        "features": int(student.clf.coef_.shape[1]),
        "n_negative_ngram": int(len(student.vec.vocabulary_)),
        "threshold": thr,
        "comparison": {},
    }
    rows = {}
    ok = np.isfinite(test["scores"])
    if ok.sum() > 10 and len(set(y[ok].tolist())) > 1:
        rows["teacher(llm)"] = (y[ok], test["scores"][ok])
        out["comparison"]["teacher"] = {
            "auroc": round(E.auroc(y[ok], test["scores"][ok]), 4),
            "auprc": round(E.auprc(y[ok], test["scores"][ok]), 4),
            "tpr@fpr1%": round(E.tpr_at_fpr(y[ok], test["scores"][ok], 0.01), 4),
            "ms_per_item": teacher_sec_per_item,
        }
    ok2 = ~np.isnan(s_test)
    if ok2.sum() > 10 and len(set(y[ok2].tolist())) > 1:
        rows["student(distilled)"] = (y[ok2], s_test[ok2])
        out["comparison"]["student"] = {
            "auroc": round(E.auroc(y[ok2], s_test[ok2]), 4),
            "auprc": round(E.auprc(y[ok2], s_test[ok2]), 4),
            "tpr@fpr1%": round(E.tpr_at_fpr(y[ok2], s_test[ok2], 0.01), 4),
            "ms_per_item": round(ms, 4),
        }
        out["student_family_recall"] = E.recall_by_family(samples, (s_test > thr).astype(int))
        out["student_metrics"] = E.binary_metrics(y, s_test, thr)
    if "knn" in test and np.isfinite(np.asarray(test["knn"], dtype=np.float64)).sum() > 10:
        k = np.asarray(test["knn"], dtype=np.float64)
        if len(set(y.tolist())) > 1:
            rows["knn(one-stage)"] = (y, k)
            out["comparison"]["knn"] = {"auroc": round(E.auroc(y, k), 4)}
    if rows:
        E.plot_roc(rows, fig_path or "outputs/figures/fig7_distill_roc.png",
                   "判官蒸馏：teacher vs student")
    # 若 teacher 与 student 都能算，给出"掉点"代价，避免只报速度不报质量
    if "teacher" in out["comparison"] and "student" in out["comparison"]:
        d = out["comparison"]["teacher"]["auroc"] - out["comparison"]["student"]["auroc"]
        out["auroc_drop"] = round(float(d), 4)
        if teacher_sec_per_item:
            out["speedup"] = round(float(teacher_sec_per_item) * 1000.0 / max(ms, 1e-9), 1)
    return out


def run_distill(cfg, dev_file=None, test_file=None, out_path=None, C: float = 1.0) -> dict:
    """端到端：读软标签 → 编码 → 训练学生 → 在测试切片上对比 → 落盘。"""
    from .encoders import get_encoder
    from . import evaluate as E

    mdir = Path(cfg.data.processed_dir).parent.parent / "outputs" / "metrics"
    # 不能 from .cli import _metrics_path：`python -m poisonguard.cli` 下 cli 是 __main__，
    # 再 import 一次会拿到 RUN_SUFFIX='' 的第二份模块。这里直接从目录名推后缀。
    suffix = Path(cfg.data.processed_dir).name[len("processed"):]  # 例如 "_real_v3"

    def _m(name: str) -> Path:
        stem, dot, ext = name.partition(".")
        return mdir / (f"{stem}{suffix}{dot}{ext}" if dot else name)

    dev_file = Path(dev_file) if dev_file else _m("judge_samples_dev.jsonl")
    test_file = Path(test_file) if test_file else _m("judge_samples.jsonl")
    if not dev_file.exists():
        raise FileNotFoundError(
            f"缺少蒸馏训练数据 {dev_file}；先跑 `python -m poisonguard.cli judge --dev-limit 20`"
        )
    if not test_file.exists():
        raise FileNotFoundError(f"缺少蒸馏测试数据 {test_file}")

    dev = load_judge_samples(dev_file)
    test = load_judge_samples(test_file)
    enc = get_encoder(cfg)
    X_dev = enc.encode(dev["texts"])
    X_test = enc.encode(test["texts"])

    student = JudgeStudent(C=C, seed=cfg.data.seed).fit(dev["texts"], X_dev, dev["scores"])

    teacher_sec = None
    jpath = _m("judge.json")
    if jpath.exists():
        try:
            jd = json.loads(jpath.read_text(encoding="utf-8"))
            for m in (jd.get("detectors") or {}).values():
                if "sec_per_item" in m:
                    teacher_sec = float(m["sec_per_item"])
        except Exception:
            teacher_sec = None

    # mdir = <root>/outputs/metrics，图放同级的 figures/
    fig_path = mdir.parent / "figures" / f"fig7_distill_roc{suffix}.png"
    rep = distill_report(student, dev, test, X_dev, X_test, teacher_sec_per_item=teacher_sec,
                         fig_path=fig_path)
    path = Path(out_path) if out_path else _m("distill.json")
    E.save_json(rep, path)
    student_path = _m("student_judge.pkl")
    student.save(student_path)
    rep["model_path"] = str(student_path)
    E.save_json(rep, path)

    print(f"[distill] 训练 {rep['n_dev']} 条 / 测试 {rep['n_test']} 条，特征 {rep['features']} 维")
    for name, m in rep.get("comparison", {}).items():
        print(f"[distill] {name:20s} AUROC={m.get('auroc')} "
              f"tpr@1%={m.get('tpr@fpr1%')} 耗时={m.get('ms_per_item')}")
    if rep.get("speedup"):
        print(f"[distill] 加速约 {rep['speedup']}×，AUROC 变化 {rep.get('auroc_drop')}")
    return rep
