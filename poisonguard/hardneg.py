"""硬负样本压力测试：把"像谣言"和"真的有害"分开考。

为什么必须做这一步：
    在真实新闻语料上，sup_tfidf 依然 AUROC=1.000。词面审计显示它的判别力主要来自
    **干净语料里也出现的** n-gram，也就是说它学会了"这句话像不像一条内部消息/传闻"：
    我们的四类攻击全都带"据知情人士透露""一份流出的截图显示""通报""统计"这类措辞，
    而真实新闻里这些词是正常出现的。

    所以真正该问的问题不是"能不能把攻击和普通新闻分开"（那是送分题），
    而是"**能不能把攻击和同样像传闻的普通新闻分开**"。
    于是构造硬负样本：从真实新闻里挑出含传闻类措辞、但完全无害的句子，
    与攻击样本放在一起，看各检测器的 AUROC 还剩多少。

这个数字如果掉到 0.6 以下，就直接说明：
    在真实语料上，"数据层无监督离群"和"词面分类"两条路都不足以支撑生产级投毒检测，
    必须引入语义层(LLM 判官)或行为层(下游模型信号)。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .config import Config

# 传闻/爆料/内部消息类措辞——真实新闻里大量存在，攻击样本也大量使用
RUMOR_MARKERS = [
    "内部", "据知情", "透露", "通报", "统计", "流出", "截图", "调查", "未经",
    "核实", "官方", "消息", "传闻", "爆料", "网传", "质疑", "回应", "曝光",
]


def _load(split_dir: Path, split: str) -> list[dict]:
    path = split_dir / f"{split}.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def build_hard_negatives(raw_file: Path, n: int, seed: int = 0,
                         markers: list[str] | None = None) -> list[str]:
    markers = markers or RUMOR_MARKERS
    lines = [l.strip() for l in raw_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    hits = [t for t in lines if any(m in t for m in markers)]
    rng = random.Random(seed)
    rng.shuffle(hits)
    return hits[:n]


def hard_negative_probe(cfg: Config, n: int = 400, out_path: Path | None = None) -> dict:
    from .detectors import CharNgramDetector, KNNDetector, MahalanobisDetector
    from .encoders import get_encoder
    from .evaluate import auroc, tpr_at_fpr

    split_dir = Path(cfg.data.processed_dir)
    train = _load(split_dir, "train")
    dev = _load(split_dir, "dev")
    test = _load(split_dir, "test")

    real_path = Path(cfg.data.raw_dir) / cfg.data.real_file
    if not real_path.exists():
        raise SystemExit(f"[hardneg] 缺少真实语料 {real_path}，先跑 download --corpus tnews")
    hard = build_hard_negatives(real_path, n, seed=cfg.data.seed)
    print(f"[hardneg] 硬负样本 {len(hard)} 条（真实新闻里含传闻类措辞、但完全无害）")
    for t in hard[:3]:
        print("   ", t[:60])

    test_atk = [r["text"] for r in test if r["label"] == 1]
    test_clean = [r["text"] for r in test if r["label"] == 0 and r["family"] == "normal"]
    dev_clean = [r["text"] for r in dev if r["label"] == 0 and r["family"] == "normal"]
    train_clean = [r["text"] for r in train]
    fit_texts = [r["text"] for r in dev]
    fit_y = np.array([int(r["label"]) for r in dev])

    res: dict = {"n_hard": len(hard), "detectors": {}}

    # 1) 词面分类器（与 detect 同配置）
    tf = CharNgramDetector().fit_labeled(fit_texts, fit_y)
    s_hard = tf.score(None, hard)
    s_atk = tf.score(None, test_atk)
    s_clean = tf.score(None, test_clean)
    y2 = np.r_[np.zeros(len(hard)), np.ones(len(s_atk))]
    res["detectors"]["sup_tfidf"] = {
        "auroc_attack_vs_hardneg": round(auroc(y2, np.r_[s_hard, s_atk]), 4),
        "auroc_attack_vs_clean": round(
            auroc(np.r_[np.zeros(len(s_clean)), np.ones(len(s_atk))], np.r_[s_clean, s_atk]), 4),
        "mean_score": {"hard_neg": round(float(np.mean(s_hard)), 4),
                       "clean": round(float(np.mean(s_clean)), 4),
                       "attack": round(float(np.mean(s_atk)), 4)},
        "tpr_at_hardneg_fpr5": round(
            tpr_at_fpr(np.r_[np.zeros(len(s_hard)), np.ones(len(s_atk))],
                       np.r_[s_hard, s_atk], 0.05), 4),
    }

    # 2) 无监督检测器（需要句向量）
    enc = get_encoder(cfg)
    X_train = enc.encode(train_clean)
    X_fit = enc.encode(fit_texts)
    X_hard = enc.encode(hard)
    X_atk = enc.encode(test_atk)
    X_clean = enc.encode(test_clean)
    for det in (KNNDetector(k=cfg.detector.knn_k), MahalanobisDetector(pca_dim=cfg.detector.pca_dim)):
        det.fit(X_train, train_clean)
        dh, da = det.score(X_hard, hard), det.score(X_atk, test_atk)
        res["detectors"][det.name] = {
            "auroc_attack_vs_hardneg": round(
                auroc(np.r_[np.zeros(len(dh)), np.ones(len(da))], np.r_[dh, da]), 4),
            "mean_score": {"hard_neg": round(float(np.mean(dh)), 4),
                           "attack": round(float(np.mean(da)), 4)},
        }

    res["markers"] = RUMOR_MARKERS
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[hardneg] 已保存 {out_path}")
    for name, r in res["detectors"].items():
        print(f"[hardneg] {name:12} 攻击 vs 硬负样本 AUROC={r['auroc_attack_vs_hardneg']:.3f}  "
              f"分数(硬负/攻击)={r['mean_score']}")
    return res
