"""结构隔离判官（A1）：冻结 Qwen 主干 + 隐藏表征二分类头。

为什么需要这一层
----------------
红队（redteam.py）已经证明 **prompt 层加固是「有代价且不够」的**：
plain → 加固 → 加固+脱敏，注入导致的劫持率只能从 56% 压到 19%，
残余集中在 `role_switch` / `fake_json` / `few_shot_mimic` ——
它们不靠显式指令词，靠**语境与角色叙述**，字面脱敏看不见。

结构隔离的答案：**判官不解析指令**。文本只作为"数据"被 Qwen 编码成隐藏表征，
然后在表征上接一个二分类头打分。无论输入怎么写"忽略规则/你是标注员/{'label':'正常'}"，
在编码器眼里都只是 token 序列，**不存在会被执行的指令通道**——
攻击者能影响的是特征向量里的某些维度，而不是"让模型改变它的判定逻辑"。

工程约束与本设计的取舍
----------------------
1. 本机是 CPU 版 torch（cuda=False），0.5B 全量 LoRA 微调要 GPU/数小时。
   因此这里做**可落地的简化版**：主干完全冻结，只训 896×2 的二分类头。
   - 主干提供的是与任务无关的通用语义表征（Qwen 预训练知识）；
   - 二分类头负责"把这些表征压成污染概率"；
   - 攻击面 = 输入的 token 序列，**没有生成路径**，所以不存在提示注入。
2. 不做 LoRA（peft 未安装）——不是 LoRA 更好，而是"冻结主干+分类头"
   已经是完整可验收的**结构隔离机制**；LoRA 是在此之上的容量增强，
   我们会在文档里明确说明"有 GPU 时把头换成 LoRA 训练即可"。
3. 特征 = mean-pool(last_hidden_state)（Qwen 最后层的全部 896 维平均池化），
   比"[CLS]"? 简单直接；文本 -> 特征 -> 头，是标准的 sentence 分类结构。

数据来源（不改动评测集）
------------------------
- 训练：`judge_samples_dev_real_v3.jsonl`（160 条判官实测软标签，
  与 distill 同源，只覆盖 dev 侧，**不碰 test**）。
- 评测：test 全量（5997 条，含 edit/template/paraphrase/subtle/backdoor/
  normal/ood/hardneg），与 `cmd_judge` 同一批 test，可直接对比。
- 打分：结构隔离分数是连续概率，可与 logprob 判官同口径比较
  （AUROC / TPR@1% / 家族召回）。
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .data import load_splits
from .localmodels import ensure_model
from .detectors import Calibration


# ----------------------------------------------------------------------
# 特征提取：Qwen 主干完全冻结，只算一次 last_hidden_state 均值池化
# ----------------------------------------------------------------------
def encode_features(cfg: Config, texts: list[str], model=None, tok=None,
                    batch_size: int = 8) -> np.ndarray:
    """把文本编码成 [N, hidden] 均值池化表征（float32）。

    `model/tok` 可复用（避免反复加载 0.5B 权重）；传 None 时惰性加载。
    """
    import torch

    if len(texts) == 0:
        return np.zeros((0, 896), dtype=np.float32)
    if model is None or tok is None:
        model, tok = _load_backbone(cfg)
    out = np.zeros((len(texts), 896), dtype=np.float32)
    with torch.inference_mode():
        for s0 in range(0, len(texts), batch_size):
            chunk = list(texts[s0:s0 + batch_size])
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=cfg.qwen.max_seq_len)
            hidden = model(**enc).last_hidden_state  # [B, L, 896]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            out[s0:s0 + len(chunk)] = pooled.float().numpy()
    return out


def _load_backbone(cfg: Config, model_id: str | None = None,
                   cache_dir: str | Path | None = None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    mid = model_id or cfg.qwen.model_id
    path = ensure_model(mid, cache_dir or cfg.qwen.model_cache)
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 用 AutoModel（无 LM 头）：我们要的是冻结主干的最大池化表征，
    # 不是生成；这样 forward 直接暴露 last_hidden_state。
    model = AutoModel.from_pretrained(
        str(path), torch_dtype=torch.float32, device_map=None,
    ).eval()
    model.to("cpu")
    return model, tok


class StructuredJudge:
    """冻结主干 + 线性头的结构隔离判官。

    `name = "struct_qwen"`：报告里与 `qwen_logprob` 区分开。
    """

    name = "struct_qwen"

    def __init__(self, W: np.ndarray, b: float = 0.0, threshold: float = 0.5,
                 calib: Calibration | None = None):
        self.W = np.asarray(W, dtype=np.float64)          # [hidden] 或 [hidden,2]
        self.b = float(b)
        self.threshold = float(threshold)
        self.calib = calib

    @classmethod
    def from_head(cls, head, threshold: float = 0.5,
                  calib: Calibration | None = None) -> "StructuredJudge":
        W, b = head
        return cls(W, b, threshold=threshold, calib=calib)

    def _scores(self, X: np.ndarray) -> np.ndarray:
        # 与训练一致：sigmoid(W·X + b) 的 P(污染)
        W = self.W.reshape(-1, 1) if self.W.ndim == 1 else self.W
        z = X @ W + self.b if W.shape[1] == 1 else X @ W
        if W.shape[1] == 1:
            z = z[:, 0]
        else:
            z = z[:, 1] - z[:, 0]  # 双输出头：P(污染) = softmax(logits)[1]
        return 1.0 / (1.0 + np.exp(-z))

    def score(self, texts: list[str], X: np.ndarray | None = None,
              cfg: Config | None = None, model=None, tok=None) -> np.ndarray:
        if X is None:
            if cfg is None:
                raise ValueError("score() 需传 X 或 cfg")
            X = encode_features(cfg, list(texts), model=model, tok=tok)
        return self._scores(np.asarray(X, dtype=np.float64))

    def to_fpr(self, scores: np.ndarray) -> np.ndarray:
        if self.calib is None:
            return np.clip(scores, 0.0, 1.0)
        return self.calib.to_fpr(scores)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self.threshold).astype(int)


# ----------------------------------------------------------------------
# 训练：只训二分类头（阈值从 dev 干净侧标定）
# ----------------------------------------------------------------------
def train_head(cfg: Config, Xs: dict, labels: np.ndarray, C: float = 1.0,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """在冻结特征 X 上训练逻辑回归头（896→1）。

    Xs: {"dev": np.ndarray} 等特征矩阵（调用方提取）
    labels: 与 Xs["dev"] 同长的 label 数组（只用于 dev 训练侧）
    返回 (W[896], b)。用 sklearn LogisticRegression(binary)：
      - 特征维度小（896）且样本也小，liblinear + L2 正好；
      - class_weight="balanced" 让 dev 里 60% 负 / 40% 正不偏科。
    """
    from sklearn.linear_model import LogisticRegression

    Xv = np.asarray(Xs["dev"], dtype=np.float64)
    yv = np.asarray(labels, dtype=int).ravel()
    if len(Xv) != len(yv):
        raise ValueError(f"dev 特征/标签长度不一致: {len(Xv)} vs {len(yv)}")
    if yv.min() == yv.max():
        raise ValueError("结构头训练需要正负两类标签")
    clf = LogisticRegression(C=C, max_iter=5000, random_state=seed,
                             class_weight="balanced", solver="liblinear")
    clf.fit(Xv, yv)
    if clf.coef_.shape[0] != 1:
        # 防御：binary 情形 shape 是 (1, n_features)；真出现多类就退回第一类
        raise ValueError("train_head 只支持二分类头")
    return np.asarray(clf.coef_[0], dtype=np.float64), float(clf.intercept_[0])


def calibrate(cfg: Config, Xs: dict, clean_scores: np.ndarray,
              target_fpr: float | None = None) -> Calibration:
    """阈值从 dev 干净侧（normal+hardneg）标定（与 detect 同口径）。"""
    cal = Calibration.fit(clean_scores,
                          target_fpr or cfg.detector.target_fpr)
    return cal


# ----------------------------------------------------------------------
def run_struct(cfg: Config, out_path: str | Path | None = None,
               C: float = 1.0, seed: int = 42) -> dict:
    """完整流水线：加载主干→编码 train/dev/test→训头→校准→评测→落盘。

    产物：
      - outputs/metrics/struct_real_v3.json（评测指标 + 模型参数快照）
      - outputs/metrics/struct_head_real_v3.pkl（头 + 校准，部署用）
    """
    import torch

    cfg.makedirs()
    out = Path(out_path) if out_path else Path(cfg.metrics_dir) / "struct.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = _load_backbone(cfg)
    print(f"[struct] 主干加载完成 {time.time() - t0:.1f}s")

    splits = load_splits(cfg.data.processed_dir)
    tr, dev, test = splits["train"], splits["dev"], splits["test"]

    # --- 训练标签：dev 的 ground-truth label（不是 soft label） ---
    dev_labels = np.array([s.label for s in dev], dtype=int)
    print(f"[struct] dev={len(dev)} (pos={int((dev_labels == 1).sum())}) "
          f"train={len(tr)} test={len(test)}")

    # --- 编码（dev/test 两组；train 不参与训练头，会全部浪费编码时间） ---
    t1 = time.time()
    Xdev = encode_features(cfg, [s.text for s in dev], model=model, tok=tok)
    print(f"[struct] 编码 dev {Xdev.shape} {time.time() - t1:.0f}s")
    t1 = time.time()
    Xtest = encode_features(cfg, [s.text for s in test], model=model, tok=tok)
    print(f"[struct] 编码 test {Xtest.shape} {time.time() - t1:.0f}s")

    # --- 训练二分类头（只在 dev 上训；不碰 test） ---
    t1 = time.time()
    W, b = train_head(cfg, {"dev": Xdev}, dev_labels, C=C, seed=seed)
    print(f"[struct] 头训练完成 (W={W.shape}, b={b:.3f}) {time.time() - t1:.1f}s")

    # --- 校准：阈值从 dev 干净侧标定 ---
    clean_mask = np.array([s.label == 0 and s.family in ("normal", "hardneg")
                           for s in dev])
    s_dev = _logit_scores(W, b, Xdev)
    cal = calibrate(cfg, {"dev": Xdev}, s_dev[clean_mask],
                    cfg.detector.target_fpr)
    print(f"[struct] 校准: 目标FPR={cfg.detector.target_fpr} "
          f"阈值={cal.raw_threshold:.4f} 实现FPR={cal.achieved_fpr:.4f}")

    judge = StructuredJudge.from_head((W, b), threshold=cal.raw_threshold, calib=cal)

    # --- 评测（test 全量，主指标掩码与 cli 一致） ---
    from . import evaluate as E
    # 主指标：label==1 或 family in (normal, hardneg)
    main_mask = np.array([s.label == 1 or s.family in ("normal", "hardneg")
                          for s in test])
    y_main = np.array([s.label for s in test])[main_mask]
    s_all = judge._scores(Xtest)
    s_main = s_all[main_mask]
    bm = E.binary_metrics(y_main, s_main, judge.threshold)
    family_recall = E.recall_by_family(
        [s for s, m in zip(test, main_mask) if m],
        (s_main > judge.threshold).astype(int),
    )

    metrics = {
        "auroc": bm["auroc"],
        "auprc": bm["auprc"],
        "threshold": bm["threshold"],
        "tpr_at_fpr1": bm["tpr_at_fpr1"],
        "tpr_at_fpr1_devcal": round(E.tpr_at_fpr_calibrated(
            np.array(dev_labels)[clean_mask], s_dev[clean_mask],
            y_main, s_main, cfg.detector.target_fpr), 4),
        "n": bm["n"],
        "family_recall": family_recall,
    }

    # ---- 结构隔离对红队注入语料的打分（发回 redteam 用） ----
    inj_metrics = _score_redteam_corpus(cfg, judge, model=model, tok=tok)
    metrics["redteam"] = inj_metrics

    # ---- 保存 ----
    head_path = Path(cfg.metrics_dir) / f"struct_head{suffix_of(cfg)}.pkl"
    with open(head_path, "wb") as f:
        pickle.dump({"W": W, "b": b, "threshold": float(judge.threshold),
                     "calib": cal}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[struct] 头已保存 {head_path}")

    res = {
        "model": f"Qwen2.5-0.5B 冻结主干 + 二分类头",
        "name": "struct_qwen",
        "detectors": {"struct_qwen": metrics},
        "family_recall": {"struct_qwen": family_recall},
        "calibration": {"target_fpr": cfg.detector.target_fpr,
                        "raw_threshold": float(cal.raw_threshold),
                        "achieved_fpr": cal.achieved_fpr},
        "head_params": {"W_shape": list(np.asarray(W).shape), "C": C, "seed": seed},
        "seconds": round(time.time() - t0, 1),
    }
    E.save_json(res, out)
    print(f"[struct] 结果已保存 {out}（用时 {res['seconds']}s）")
    return res


def suffix_of(cfg: Config) -> str:
    """从 processed_dir 名推本轮实验后缀（与 redteam/distill 同一套）。"""
    name = Path(cfg.data.processed_dir).name
    return name[len("processed"):] if name.startswith("processed") else ""


# ----------------------------------------------------------------------
def _score_redteam_corpus(cfg, judge, model=None, tok=None) -> dict:
    """结构隔离对红队注入语料的打分。

    复用 redteam 的载荷/配对逻辑，但打分器换成 `judge.score`。
    返回与 redteam 的 `by_payload` 同构的 dict，便于 report 直接渲染。
    """
    from .redteam import _strong_bases, build_injection_corpus, INJECT_PAYLOADS

    bases = _strong_bases(cfg, n=6)
    corpus = build_injection_corpus(bases, n_each=2, seed=cfg.data.seed)
    texts = [c["text"] for c in corpus]
    # corpus 是拼接文本，只能重新编码；复用主干的 model/tok 避免二次加载。
    # CPU 上 8×6+6=54 条很快。
    scores = judge.score(texts, cfg=cfg, model=model, tok=tok)
    ctrl = np.array([c["payload"] == "none" for c in corpus])
    inj = ~ctrl
    thr = 0.5  # 与 redteam 对齐；结构分数本身就是概率
    base_of: dict[int, float] = {c["base_idx"]: float(scores[i])
                                 for i, c in enumerate(corpus) if c["payload"] == "none"}
    out: dict[str, dict] = {}
    for pname, _ in INJECT_PAYLOADS:
        m = np.array([c["payload"] == pname for c in corpus])
        if not m.any():
            continue
        d = np.array([scores[i] - base_of.get(c["base_idx"], np.nan)
                      for i, c in enumerate(corpus) if m[i]])
        d = d[np.isfinite(d)]
        hij = np.array([
            (base_of.get(c["base_idx"], 0.0) >= thr) and (scores[i] < thr)
            for i, c in enumerate(corpus) if m[i]
        ])
        out[pname] = {
            "mean_delta": round(float(d.mean()), 4) if d.size else 0.0,
            "hijack_rate": round(float(hij.mean()), 4) if hij.size else 0.0,
        }
    d_all = np.array([scores[i] - base_of.get(c["base_idx"], np.nan)
                      for i, c in enumerate(corpus) if inj[i]])
    d_all = d_all[np.isfinite(d_all)]
    hij_all = np.array([(base_of.get(c["base_idx"], 0.0) >= thr) and (scores[i] < thr)
                        for i, c in enumerate(corpus) if inj[i]])
    out["all"] = {
        "n": int(len(d_all)),
        "mean_delta": round(float(d_all.mean()), 4) if d_all.size else 0.0,
        "hijack_rate": round(float(hij_all.mean()), 4) if hij_all.size else 0.0,
    }
    return out


# 给 run_struct 用的桥：算 sigmoid 分数（与 StructuredJudge._scores 的 W[hidden] 分支一致）
def _logit_scores(W: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(X @ np.asarray(W).ravel() + b)))


def build_parser() -> None:  # pragma: no cover - 占位，避免误用
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """兜底入口：实际由 cli.main 的 struct-train / struct-serve 接线。"""
    raise SystemExit("请通过 `python -m poisonguard.cli struct-train ...` 调用")
