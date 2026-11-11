"""受害模型(被投毒的对象)与行为级检测信号。

为什么要"受害模型"？
    内容层的检测器(关键词/embedding 距离)只看「这句话像不像正常话」。
    但投毒攻击的真正危害在于**改变了下游模型的行为**。所以一个成熟的方案要分两层：
      - 数据层(Data-level):  embedding 异常检测  → 上线前过滤训练数据
      - 行为层(Behavior-level): 模型预测/困惑度异常 → 监控模型是否已被污染
    这也是原项目想做但做错的地方(KL 散度不分词、梯度范数是随机噪声)。

本模块提供三个可用的行为信号：
1. `BertVictim`  —— 复用仓库里已有的 baseline_model.pth(102M 参数 BERT 二分类)，
   给出每个样本的 p(污染)、预测熵、logit margin，以及 [CLS] 隐表示(768 维)。
2. `ProbeVictim` —— 句向量 + 逻辑回归探针。用来做"投毒比例 → 检测能力"的消融实验，
   在 CPU 上秒级完成，不需要把 BERT 重训 8 分钟(N 次)。
3. `PPLScorer`   —— 可选资源：因果语言模型困惑度(Qwen2.5-0.5B)。
   需要额外下载 1GB 权重，默认关闭，`--enable-ppl` 打开。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .localmodels import find_bert_tokenizer_dir, find_local_model_dir


# ----------------------------------------------------------------------
def _strip_wrapper_prefix(sd: dict) -> dict:
    """仓库里的 baseline_model.pth 外面多包了一层 "bert." 前缀。

    实测 key 形如:
        bert.bert.embeddings.word_embeddings.weight
        bert.classifier.weight            (2, 768)
    说明原始模型是 BertForSequenceClassification，保存时被包了一层。
    这里自动剥掉外层前缀，并兼容 "module." 前缀(DataParallel)。
    """
    keys = list(sd.keys())
    if all(k.startswith("module.") for k in keys):
        sd = {k[len("module."):]: v for k, v in sd.items()}
        keys = list(sd.keys())
    has_nested = any(k.startswith("bert.embeddings.") for k in keys)
    if not has_nested and any(k.startswith("bert.bert.") for k in keys):
        sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
    return sd


@dataclass
class VictimOutput:
    p_poison: np.ndarray          # softmax 后"污染"类概率
    entropy: np.ndarray           # 预测熵(越大越不确定)
    margin: np.ndarray            # logit[1] - logit[0]
    hidden: np.ndarray | None = None  # [CLS] 隐表示, (n, 768)


class BertVictim:
    """复用已有 BERT 分类器作为受害模型。

    注意：这个权重是旧项目里在 **泄漏的 train=test** 数据上训出来的，
    所以它的绝对准确率不可信；我们只用它做**相对信号**(同一个模型对不同样本的
    置信度/隐表示差异)，这在方法论上是成立的，并且会在报告里明确标注这一局限。
    """

    def __init__(self, ckpt_path: str | Path, tokenizer_dir: str | Path | None = None,
                 device: str = "cpu", max_length: int = 64, batch_size: int = 64,
                 return_hidden: bool = True):
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.return_hidden = return_hidden
        self._tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else None
        self._loaded = False
        self.load_report: dict = {}

    # -- 延迟加载：只有真的要用时才吃 400MB 内存 ------------------------
    def load(self) -> "BertVictim":
        if self._loaded:
            return self
        import torch
        from transformers import BertConfig, BertForSequenceClassification
        from transformers import BertTokenizerFast

        sd = _strip_wrapper_prefix(torch.load(self.ckpt_path, map_location="cpu",
                                              weights_only=True))
        num_labels = int(sd["classifier.weight"].shape[0])

        tdir = self._tokenizer_dir or find_bert_tokenizer_dir()
        if tdir is None:
            raise RuntimeError(
                "找不到中文 BERT 词表。请先运行 `python -m poisonguard.cli download`，"
                "或手动放置 model_cache/damo/nlp_corom_sentence-embedding_chinese-base/vocab.txt"
            )
        tdir = Path(tdir)
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(str(tdir))
        except Exception:
            self.tokenizer = BertTokenizerFast(vocab_file=str(tdir / "vocab.txt"),
                                               do_lower_case=True)

        cfg_path = tdir / "config.json"
        if cfg_path.exists():
            config = BertConfig.from_pretrained(str(tdir))
        else:
            vocab_size = int(sd["embeddings.word_embeddings.weight"].shape[0])
            config = BertConfig(vocab_size=vocab_size, num_hidden_layers=12,
                                hidden_size=768, num_attention_heads=12,
                                intermediate_size=3072)
        config.num_labels = num_labels
        config.add_pooling_layer = True

        model = BertForSequenceClassification(config)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        self.load_report = {
            "num_labels": num_labels,
            "n_missing": len(missing),
            "n_unexpected": len(unexpected),
            "missing_sample": list(missing)[:5],
            "unexpected_sample": list(unexpected)[:5],
        }
        if missing:
            raise RuntimeError(
                f"权重未完全加载({len(missing)} 个 key 缺失)，首个缺失: {list(missing)[:3]}"
            )
        model.eval()
        model.to(self.device)
        self.model = model
        self._loaded = True
        return self

    # -- 推理 -----------------------------------------------------------
    def predict(self, texts: list[str]) -> VictimOutput:
        import torch

        self.load()
        all_logits, all_hidden = [], []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc, output_hidden_states=self.return_hidden)
                all_logits.append(out.logits.float().cpu())
                if self.return_hidden:
                    all_hidden.append(out.hidden_states[-1][:, 0, :].float().cpu())
        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=-1)
        p1 = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        p1 = p1.clamp(1e-6, 1 - 1e-6)
        entropy = -(p1 * p1.log() + (1 - p1) * (1 - p1).log())
        margin = logits[:, 1] - logits[:, 0] if logits.shape[1] > 1 else logits[:, 0]
        hidden = torch.cat(all_hidden, dim=0).numpy() if self.return_hidden else None
        return VictimOutput(
            p_poison=p1.numpy(),
            entropy=entropy.numpy(),
            margin=margin.numpy(),
            hidden=hidden,
        )

    # 两个直接可用的打分函数(供 CallableDetector 使用) -------------------
    def entropy_scorer(self):
        return lambda texts: self.predict(texts).entropy

    def p_poison_scorer(self):
        return lambda texts: self.predict(texts).p_poison


# ----------------------------------------------------------------------
def poison_mix(clean_texts: list[str], poison_texts: list[str], rate: float,
               seed: int = 0) -> tuple[list[str], np.ndarray]:
    """按比例 rate 把污染样本混进干净训练集，返回 (texts, labels)。"""
    rng = np.random.default_rng(seed)
    n_clean = len(clean_texts)
    if not 0 <= rate < 1:
        raise ValueError("rate 必须落在 [0, 1)")
    n_poison = int(round(rate * n_clean / max(1e-9, (1 - rate))))
    n_poison = min(n_poison, len(poison_texts))
    idx = rng.choice(len(poison_texts), size=n_poison, replace=False)
    texts = list(clean_texts) + [poison_texts[i] for i in idx]
    labels = np.concatenate([np.zeros(n_clean, dtype=int), np.ones(n_poison, dtype=int)])
    order = rng.permutation(len(texts))
    return [texts[i] for i in order], labels[order]


class ProbeVictim:
    """句向量 + 逻辑回归的"受害模型"探针。

    用途：投毒比例消融。真实场景里不可能把 1 亿参数模型按每个比例重训一遍，
    但"投毒比例 ρ 越大，探针越容易被带偏、同时也越容易暴露"这个规律
    用线性探针就能复现，成本却低两个数量级。
    """

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbeVictim":
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                      random_state=self.seed).fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)

    def poison_rate(self, X: np.ndarray) -> np.ndarray:
        """探针判为污染的比例——反映"模型被带偏的程度"。"""
        return (self.score(X) > 0.5).mean()


# ----------------------------------------------------------------------
class PPLScorer:
    """因果 LM 困惑度(可选)。默认用 Qwen2.5-0.5B-Instruct，CPU 也能跑几十条/秒。"""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 cache_dir: str | Path | None = None, device: str = "cpu",
                 max_length: int = 128, batch_size: int = 8):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._loaded = False

    @staticmethod
    def local_available(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> bool:
        return find_local_model_dir(model_id) is not None

    def load(self) -> "PPLScorer":
        if self._loaded:
            return self
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .localmodels import ensure_model

        path = ensure_model(self.model_id, self.cache_dir)
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.float32
        ).eval().to(self.device)
        self._loaded = True
        return self

    def score(self, texts: list[str]) -> np.ndarray:
        """返回每个样本的平均负对数似然(越大越"不像模型见过的正常文本")。"""
        import torch

        self.load()
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_length)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.float()
                shift_logits = logits[:, :-1, :]
                shift_labels = enc["input_ids"][:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                ).view(shift_labels.shape)
                mask = (shift_labels != self.tok.pad_token_id).float()
                per = (loss * mask).sum(1) / mask.sum(1).clamp(min=1)
                out.append(per.cpu())
        return torch.cat(out).numpy().astype(np.float32)
