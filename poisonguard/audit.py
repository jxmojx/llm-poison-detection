"""词面审计：检测器到底在抓"有害性"，还是在抓"没见过这种句式"？

背景（本项目最容易被忽略的一处虚高）：
    仓库自带的 normal_data_2w.csv 并不是自然语料，而是
    "10 个主题 × 10 个方面 × 10 个领域" 拼出来的**槽位语料**。
    去重后只有 4997 条、字符 2-3gram 词表只有 994 个。
    在这种干净侧上训练任何判别式检测器，它最容易学到的不是"有害性"，
    而是"这句话是否用了那三个模板" —— 于是攻击全被抓到，指标漂亮但无意义。

审计做法：
    1. 统计干净语料自己的 n-gram 词表大小（衡量"语料有多贫瘠"）；
    2. 把检测器的决策函数拆到 n-gram 粒度，统计"干净语料里从未出现过的
       n-gram"贡献了多少决策权重（衡量"多少预测来自'没见过'"）；
    3. 反事实：只用"干净语料里出现过的 n-gram"重建分类器，看判别力掉多少。

三个数一起看，才能判断指标是"内容信号"还是"词面新颖度"。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import Config


def _load(split_dir: Path, split: str) -> list[dict]:
    path = split_dir / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"[audit] 缺少 {path}，先跑 data 子命令")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def lexical_audit(cfg: Config, out_path: Path | None = None) -> dict:
    split_dir = Path(cfg.data.processed_dir)
    train, dev = _load(split_dir, "train"), _load(split_dir, "dev")
    clean_texts = [r["text"] for r in train]

    # 1) 干净语料自己的 n-gram 词表
    probe = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
    probe.fit(clean_texts)
    clean_vocab = set(probe.get_feature_names_out())

    # 2) 用与 CharNgramDetector 完全相同的配置拟合
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2,
                          max_features=200_000, sublinear_tf=True)
    X = vec.fit_transform([r["text"] for r in dev])
    y = np.array([int(r["label"]) for r in dev])
    clf = LogisticRegression(C=2.0, max_iter=3000).fit(X, y)
    names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    oov = np.array([n not in clean_vocab for n in names])

    # 决策权重 ≈ |系数| × 该特征在拟合集上的平均激活
    avg = np.asarray(X.mean(axis=0)).ravel()
    w = np.abs(coef) * avg
    w = w / max(w.sum(), 1e-12)

    # 3) 反事实：只用"干净语料里出现过"的 n-gram
    mask = ~oov
    auroc_in_vocab = None
    if mask.sum() >= 10:
        clf_in = LogisticRegression(C=2.0, max_iter=3000).fit(X[:, mask], y)
        auroc_in_vocab = float(roc_auc_score(y, clf_in.predict_proba(X[:, mask])[:, 1]))

    order = np.argsort(-w)[:20]
    top = [
        {"ngram": str(names[i]), "oov": bool(oov[i]),
         "coef": round(float(coef[i]), 3), "weight": round(float(w[i]), 4)}
        for i in order
    ]
    res = {
        "clean_train_size": len(clean_texts),
        "clean_ngram_vocab": len(clean_vocab),
        "fitted_ngram_vocab": int(len(names)),
        "oov_ngram_count": int(oov.sum()),
        "oov_ngram_ratio": round(float(oov.mean()), 4),
        "oov_weight_share": round(float(w[oov].sum()), 4),
        "in_vocab_weight_share": round(float(w[~oov].sum()), 4),
        # in-sample 的 AUROC，只作"上限参考"，不进主表
        "in_vocab_only_auroc_in_sample": None if auroc_in_vocab is None else round(auroc_in_vocab, 4),
        "top_ngrams": top,
    }
    print(
        f"[audit] 干净语料 n-gram 词表={res['clean_ngram_vocab']}，"
        f"检测器用到 {res['fitted_ngram_vocab']} 个，其中干净语料里没出现过 "
        f"{res['oov_ngram_count']} 个({res['oov_ngram_ratio']:.1%})，"
        f"它们贡献了 {res['oov_weight_share']:.1%} 的决策权重；"
        f"只用词表内 n-gram 时 AUROC={res['in_vocab_only_auroc_in_sample']}"
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[audit] 已保存 {out_path}")
    return res
