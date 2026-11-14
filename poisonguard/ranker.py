"""学生排序器（A5）：同一份软标签下，比较「字符 n-gram / 句向量 / 混合」谁更适合做粗排。

为什么要做这件事
----------------
场景四的粗排层用蒸馏学生模型给全量数据打分，再按分数取 TopK 交给判官复核。
实测发现**edit 家族在 TopK 里只有 0.4~0.7× 的富集**（`repro_scene4.py`）——
也就是说：粗排把「模板/改写」这类字面异常样本排上来了，却排不上「事实被改了一处」
这种语义级篡改。学生模型当前是 `[句向量 | 字符 2-3gram TF-IDF]` 的线性拼接，
字符 n-gram 特征数上万、句向量只有 768 维，**维度失衡**让 n-gram 主导了决策。

本模块把这个问题拆成**受控对比**：同一批软标签、同一测试切片、同一套评测口径，
只换特征/模型，看谁能把尾部（TPR@1%）和 edit 富集抬起来。

变体
----
- ``ngram``      : 只用字符 2-3gram TF-IDF（"词面视图"）
- ``emb``        : 只用 CoROM 768 维句向量（"语义视图"）
- ``emb_ngram``  : 两者拼接（现状基线）
- ``emb_pca``    : 句向量 PCA 到 64 维再线性分类（抑噪、抗小样本）
- ``emb_mlp``    : 句向量 + 单隐层 32 维 MLP（非线性，小样本要早停）
- ``stack``      : ``emb`` 与 ``ngram`` 两个弱模型的概率等权平均（零额外成本）

诚实边界
--------
软标签只有 160 条（dev 切片），任何"升级"都要在**同一测试切片**上与基线并列报，
并且报告单条耗时——如果 AUROC 涨了 0.01 但慢 10 倍，那不值得上线。
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

VARIANTS = ("ngram", "emb", "emb_ngram", "emb_pca", "emb_mlp", "stack")


class StudentRanker:
    """一个可切换特征/模型的学生排序器（pickle 友好，不依赖 torch）。"""

    def __init__(self, variant: str = "emb_ngram", C: float = 1.0,
                 max_iter: int = 3000, seed: int = 0,
                 max_features: int = 30000, pca_dim: int = 64):
        if variant not in VARIANTS:
            raise ValueError(f"未知变体: {variant}（可选 {VARIANTS}）")
        self.variant = variant
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.max_features = max_features
        self.pca_dim = pca_dim
        self.scaler = StandardScaler()
        self.vec = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 3), min_df=2,
            max_features=max_features, sublinear_tf=True,
        )
        self.pca: PCA | None = None
        self.clf = None
        self._clf2 = None      # stack 的第二个模型
        self.threshold = 0.5
        self.train_time = 0.0

    # ---------------- 特征 ----------------
    def _dense(self, X_emb, fit: bool) -> np.ndarray:
        X = np.asarray(X_emb, dtype=np.float64)
        if fit:
            d = self.scaler.fit_transform(X)
            if self.variant == "emb_pca":
                self.pca = PCA(n_components=min(self.pca_dim, max(2, X.shape[0] // 8)),
                               random_state=self.seed)
                d = self.pca.fit_transform(d)
        else:
            d = self.scaler.transform(X)
            if self.variant == "emb_pca" and self.pca is not None:
                d = self.pca.transform(d)
        return d

    def _build(self, texts, X_emb, fit: bool):
        v = self.variant
        if v == "ngram":
            return self.vec.fit_transform(texts) if fit else self.vec.transform(texts)
        d = sparse.csr_matrix(self._dense(X_emb, fit))
        if v in ("emb", "emb_pca", "emb_mlp", "stack"):
            return d
        ngram = self.vec.fit_transform(texts) if fit else self.vec.transform(texts)
        return sparse.hstack([d, ngram]).tocsr()

    # ---------------- 训练 ----------------
    def fit(self, texts, X_emb, teacher_scores, threshold: float = 0.5) -> "StudentRanker":
        t0 = time.time()
        scores = np.asarray(teacher_scores, dtype=np.float64).ravel()
        ok = np.isfinite(scores)
        if ok.sum() < 20:
            raise ValueError(f"可用软标签太少({int(ok.sum())} 条)，无法训练排序器")
        y = (scores[ok] >= threshold).astype(int)
        if len(set(y.tolist())) < 2:
            raise ValueError("软标签全落在同一侧，无法训练")
        keep = [t for t, k in zip(texts, ok) if k]
        Xe = np.asarray(X_emb)[ok]
        w = np.clip(np.abs(scores[ok] - threshold) * 2.0, 0.2, 1.0)

        if self.variant == "stack":
            self._fit_one("emb", keep, Xe, y, w)
            self._fit_one("ngram", keep, Xe, y, w)
        else:
            self._fit_one(self.variant, keep, Xe, y, w)
        self.threshold = threshold
        self.train_time = time.time() - t0
        return self

    def _make_clf(self, which: str):
        if which == "emb_mlp":
            return MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=2000,
                                 random_state=self.seed)
        return LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed,
                                  class_weight="balanced", solver="liblinear")

    def _fit_one(self, which: str, texts, Xe, y, w) -> None:
        prev = self.variant
        self.variant = which
        F = self._build(texts, Xe, fit=True)
        clf = self._make_clf(which)
        clf.fit(F, y, sample_weight=w)
        self.variant = prev
        if which == "ngram":
            self._clf2 = clf
        else:
            self.clf = clf

    # ---------------- 推理 ----------------
    def score(self, texts, X_emb) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.array([], dtype=np.float64)
        if self.variant == "stack":
            p1 = self._clf("emb").predict_proba(self._F("emb", texts, X_emb))[:, 1]
            p2 = self._clf("ngram").predict_proba(self._F("ngram", texts, X_emb))[:, 1]
            return 0.5 * (p1 + p2)
        return self._clf(self.variant).predict_proba(self._F(self.variant, texts, X_emb))[:, 1]

    def _clf(self, which: str):
        return self._clf2 if which == "ngram" else self.clf

    def _F(self, which: str, texts, X_emb):
        prev = self.variant
        self.variant = which
        try:
            return self._build(texts, X_emb, fit=False)
        finally:
            self.variant = prev

    def timed_score(self, texts, X_emb) -> tuple[np.ndarray, float]:
        texts = list(texts)
        t0 = time.perf_counter()
        s = self.score(texts, X_emb)
        return s, (time.perf_counter() - t0) * 1000.0 / max(len(texts), 1)

    # ---------------- 序列化 ----------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f, protocol=4)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "StudentRanker":
        with Path(path).open("rb") as f:
            return pickle.load(f)


# ----------------------------------------------------------------------
def topk_family_lift(labels: np.ndarray, families: np.ndarray, scores: np.ndarray,
                     target: str, topk: int) -> dict:
    """TopK 里 ``target`` 家族的富集倍数（相对随机抽取的期望）。

    lift = (TopK 中该家族占比) / (全体中该家族占比)。
    lift>1 表示粗排"偏向"该家族；场景四关心的正是 edit 的 lift。
    """
    n = len(scores)
    if n == 0:
        return {"topk": 0, "n_family_total": 0, "n_family_topk": 0, "lift": 0.0}
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    k = int(min(topk, n))
    top = order[:k]
    fam = np.asarray(families)
    total = int((fam == target).sum())
    hit = int((fam[top] == target).sum())
    base = total / max(n, 1)
    return {
        "topk": k,
        "n_family_total": total,
        "n_family_topk": hit,
        "lift": round(hit / max(k * base, 1e-9), 4),
    }


def family_lift_at_rates(labels: np.ndarray, families: np.ndarray,
                         scores: np.ndarray, rates=(0.01, 0.02, 0.05, 0.10)) -> dict:
    """多个 TopK 比例下的家族富集 + 各家族在 TopK 内的召回。"""
    n = len(scores)
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    fam = np.asarray(families)
    out: dict[str, dict] = {}
    for r in rates:
        k = max(1, int(round(r * n)))
        top = order[:k]
        row: dict[str, dict] = {}
        for f in sorted(set(fam.tolist())):
            if f in ("normal", "ood", "hardneg"):
                continue
            total = int((fam == f).sum())
            hit = int((fam[top] == f).sum())
            row[f] = {
                "n_top": hit,
                "recall_in_topk": round(hit / max(total, 1), 4),
                "lift": round(hit / max(k * (total / max(n, 1)), 1e-9), 4),
            }
        out[f"top{k}"] = row
    return out


def _eval_variant(student: StudentRanker, dev: dict, test: dict,
                  X_dev, X_test) -> dict:
    from . import evaluate as E

    s_dev = student.score(dev["texts"], X_dev)
    s_test, ms = student.timed_score(test["texts"], X_test)
    y = test["labels"]
    ok = np.isfinite(s_test)
    if ok.sum() < 20 or len(set(y[ok].tolist())) < 2:
        return {"error": "测试切片不可用"}

    s_tr = s_test[ok]
    y_ok = y[ok]
    s_dev_ok = s_dev[np.isfinite(s_dev)]
    y_dev_ok = dev["labels"][np.isfinite(s_dev)]
    return {
        "auroc": round(E.auroc(y_ok, s_tr), 4),
        "auprc": round(E.auprc(y_ok, s_tr), 4),
        "tpr_at_fpr1": round(E.tpr_at_fpr(y_ok, s_tr, 0.01), 4),
        "tpr_at_fpr1_devcal": round(E.tpr_at_fpr_calibrated(
            y_dev_ok, s_dev_ok, y_ok, s_tr, 0.01), 4),
        "ece": round(E.ece(y_ok, s_tr), 4),
        "ms_per_item": round(ms, 4),
        "n_features": int(getattr(student.clf, "n_features_in_", 0)
                          + (getattr(student._clf2, "n_features_in_", 0)
                             if student.variant == "stack" else 0)),
        "train_sec": round(student.train_time, 3),
    }


def run_ranker(cfg, out_path=None, variants: tuple[str, ...] = VARIANTS,
               C: float = 1.0, seed: int = 0, topk: int = 40,
               dev_file=None, test_file=None, full_test: bool = True,
               sources: tuple[str, ...] = ("teacher", "gt")) -> dict:
    """在同一软标签/同一测试切片上对比各学生排序器变体。"""
    from . import evaluate as E
    from .distill import load_judge_samples
    from .encoders import get_encoder

    mdir = Path(cfg.data.processed_dir).parent.parent / "outputs" / "metrics"
    suffix = Path(cfg.data.processed_dir).name[len("processed"):]

    def _m(name: str) -> Path:
        stem, dot, ext = name.partition(".")
        return mdir / (f"{stem}{suffix}{dot}{ext}" if dot else name)

    dev_file = Path(dev_file) if dev_file else _m("judge_samples_dev.jsonl")
    test_file = Path(test_file) if test_file else _m("judge_samples.jsonl")
    if not dev_file.exists():
        raise FileNotFoundError(
            f"缺少训练软标签 {dev_file}；先跑 `judge --dev-limit 20` 生成")
    dev = load_judge_samples(dev_file)
    test = load_judge_samples(test_file)

    enc = get_encoder(cfg)
    X_dev = enc.encode(dev["texts"])
    X_test = enc.encode(test["texts"])

    X_full = None
    if full_test:
        from .cli import _labels, _load_or_build_splits, _texts

        splits = _load_or_build_splits(cfg)
        ft_rows = splits["test"]
        full_test = {
            "texts": _texts(ft_rows),
            "labels": _labels(ft_rows),
            "families": np.array([s.family for s in ft_rows]),
        }
        X_full = enc.encode(full_test["texts"])
        print(f"[ranker] 全量 test {len(full_test['texts'])} 条（编码缓存命中）")
    else:
        full_test = None

    # 监督信号对照：teacher 软标签(160 条) vs dev 真标签(4296 条)
    gt_train = None
    gt_texts: list[str] = []
    X_gt = np.zeros((0, 0))
    fam_gt = np.zeros((0,), dtype=object)
    train_sets: dict[str, dict] = {
        "teacher": {"texts": dev["texts"], "scores": dev["scores"]},
    }
    if "gt" in sources:
        from .cli import _labels, _load_or_build_splits, _texts

        dsplits = _load_or_build_splits(cfg)
        d_rows = dsplits["dev"]
        d_texts = _texts(d_rows)
        d_lab = _labels(d_rows)
        gt_texts = d_texts
        X_gt = enc.encode(d_texts)
        fam_gt = np.array([str(getattr(r, "family", None) or (r.get("family", "?") if isinstance(r, dict) else "?")) for r in d_rows])
        gt_train = train_sets["gt"] = {"texts": gt_texts,
                            "scores": d_lab.astype(np.float64),
                            "X": X_gt}

    fam_te = np.asarray(test["families"])
    results: dict[str, dict] = {}
    students: dict[str, StudentRanker] = {}
    test_scores: dict[str, np.ndarray] = {}
    for v in variants:
      for src in sources:
        tr = train_sets.get(src)
        if tr is None:
            continue
        Xtr = tr.get("X", X_dev)
        key = f"{v}@{src}"
        st = StudentRanker(variant=v, C=C, seed=seed).fit(tr["texts"], Xtr, tr["scores"])
        results[key] = _eval_variant(st, dev, test, X_dev, X_test)
        results[key]["edit_lift_topk"] = topk_family_lift(
            test["labels"], test["families"], st.score(test["texts"], X_test),
            "edit", topk)
        results[key]["edit_lift_topk_all"] = topk_family_lift(
            test["labels"], test["families"], st.score(test["texts"], X_test),
            "edit", len(test["labels"]))
        students[key] = st
        test_scores[key] = st.score(test["texts"], X_test)
        if full_test is not None:
            s_full, ms_full = st.timed_score(full_test["texts"], X_full)
            yf = full_test["labels"]
            m = np.array([str(x) in ("normal", "hardneg") or int(y) == 1
                          for x, y in zip(full_test["families"], yf)])
            results[key]["full_test"] = {
                "n": int(len(yf)),
                "gt_auroc": round(E.auroc(yf[m], s_full[m]), 4) if len(set(yf[m].tolist())) > 1 else None,
                "gt_tpr_at_fpr1": round(E.tpr_at_fpr(yf[m], s_full[m], 0.01), 4),
                "ms_per_item": round(ms_full, 4),
                "lift": family_lift_at_rates(yf, full_test["families"], s_full),
            }
            ed = results[key]["full_test"]["lift"].get("top300", {}).get("edit", {})
            print(f"[ranker]   └ 全量 test: GT-AUROC={results[key]['full_test']['gt_auroc']} "
                  f"edit@Top300 lift={ed.get('lift')}（召回 {ed.get('recall_in_topk')}）")
        print(f"[ranker] {key:<14} AUROC={results[key].get('auroc')} "
              f"TPR@1%={results[key].get('tpr_at_fpr1')} "
              f"(dev校准 {results[key].get('tpr_at_fpr1_devcal')}) "
              f"edit富集@Top{topk}={results[key]['edit_lift_topk']['lift']}")

    # ---- 分家族专头 + 双通道并集（A5 的解法） ----
    # 观察：通用排序器的 GT-AUROC 可以到 0.94，但 edit 在 Top5% 里富集仍是 0——
    # 因为 template/paraphrase/backdoor 这些合成家族天生更"响"，占满 TopK。
    # 换特征解决不了"正类内部尺度不均"，解法是给 edit 单训一个头，
    # 与总分数通道**取并集**（不是相加），保总召回的同时把 edit 捞回来。
    specialists: dict[str, dict] = {}
    if gt_train is not None and "edit" in tuple(fam_gt):
        def _z(x: np.ndarray) -> np.ndarray:
            return (x - x.mean()) / max(float(x.std()), 1e-9)

        def _union(gen: np.ndarray, spec: np.ndarray, k: int, fams) -> dict:
            ig = set(np.argsort(-gen)[:k].tolist())
            isp = set(np.argsort(-spec)[:k].tolist())
            u = ig | isp
            tgt = np.array([f == "edit" for f in fams])
            return {
                "k": int(k),
                "n_union": int(len(u)),
                "extra_budget": int(len(u) - k),
                "target_recall": round(float(tgt[list(u)].sum() / max(int(tgt.sum()), 1)), 4),
                "target_lift": round(
                    float(tgt[list(u)].sum() / max(int(tgt.sum()), 1)) / max(len(u) / len(tgt), 1e-9), 4),
            }

        base_key = max(
            (k for k in test_scores if k.endswith("@gt")),
            key=lambda k: (results[k].get("full_test") or {}).get("gt_auroc") or 0.0,
            default=next(iter(test_scores)),
        )
        gen_s = test_scores[base_key]
        for tgt in ("edit",):
            yspec = (fam_gt == tgt).astype(np.float64)
            spec = StudentRanker(variant="emb", C=C, seed=seed).fit(
                gt_texts, X_gt, yspec)
            s_spec = spec.score(test["texts"], X_test)
            # 富集表看全量 test（与通用排序器的口径一致），切片兜底
            if full_test:
                ev = {"labels": full_test["labels"], "families": np.asarray(full_test["families"])}
                ev_texts, ev_X = full_test["texts"], X_full
            else:
                ev = {"labels": test["labels"], "families": fam_te}
                ev_texts, ev_X = test["texts"], X_test
            s_ev = spec.score(ev_texts, ev_X)
            gen_ev = students[base_key].score(ev_texts, ev_X)
            ybin = np.array([int(f == tgt) for f in ev["families"]])
            specialists[tgt] = {
                "variant": "emb",
                "n_pos_train": int(yspec.sum()),
                "auc_vs_all": round(float(E.auroc(ybin, s_ev)), 4),
                "lift": family_lift_at_rates(ev["labels"], ev["families"], s_ev),
                "lift_on_slice": family_lift_at_rates(
                    test["labels"], fam_te, s_spec),
                "union_with": base_key,
                "union": {
                    "top1pct": _union(gen_ev, s_ev, max(1, int(0.01 * len(ev["families"]))), ev["families"]),
                    "top5pct": _union(gen_ev, s_ev, max(1, int(0.05 * len(ev["families"]))), ev["families"]),
                    "top10pct": _union(gen_ev, s_ev, max(1, int(0.10 * len(ev["families"]))), ev["families"]),
                },
            }
            spec.save(_m(f"student_specialist_{tgt}.pkl"))
            print(f"[ranker] 专头@{tgt}: AUC(vs all)={specialists[tgt]['auc_vs_all']} "
                  f"edit@Top5% lift={(specialists[tgt]['lift'].get('top300') or {}).get('edit', {}).get('lift')} "
                  f"并集@Top5%: recall={specialists[tgt]['union']['top5pct']['target_recall']} "
                  f"(+{specialists[tgt]['union']['top5pct']['extra_budget']} 条预算)")

    # 选定规则提前说清楚：A5 的目标是"粗排把 edit 排上来"，
    # 所以先看全量 test 上 edit@Top5% 的富集，并列时看 GT AUROC。
    def _key(k: str):
        ft = results[k].get("full_test") or {}
        ed = (ft.get("lift") or {}).get("top300", {}).get("edit", {})
        return (ed.get("lift", 0.0), ft.get("gt_auroc") or 0.0)

    best = max(results, key=_key)
    rep = {
        "n_dev": int(len(dev["texts"])),
        "n_test": int(len(test["texts"])),
        "topk": int(topk),
        "variants": results,
        "specialists": specialists,
        "best_variant": best,
        "selection_rule": "max edit@Top5%(全量 test) 富集，并列看 GT AUROC",
    }
    path = Path(out_path) if out_path else _m("ranker.json")
    E.save_json(rep, path)
    students[best].save(_m("student_ranker.pkl"))
    rep["model_path"] = str(_m("student_ranker.pkl"))
    E.save_json(rep, path)
    print(f"[ranker] 选定 {best} -> {rep['model_path']}")
    return rep
