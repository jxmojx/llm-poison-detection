"""句向量编码器：把文本编码成稠密向量，供检测器复用。

支持模型(按优先级/显存占用)：
    crom   : 本地已下载的 damo/nlp_corom_sentence-embedding_chinese-base (102M, 768d)
    bge    : BAAI/bge-small-zh-v1.5 (24M, 512d) —— 语义检测的性价比之选
    qwen3  : Qwen/Qwen3-Embedding-0.6B (595M) —— 效果最好，需 ~2GB 显存
    hf:<路径或模型ID> : 任意 AutoModel

统一走 mean-pooling + L2 归一化(与 bge/Qwen-Embedding 的官方用法一致)，
向量落盘缓存(outputs/cache)，重复实验不必重算。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class SentenceEncoder:
    def __init__(
        self,
        model_dir: str,
        name: str = "crom",
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 64,
        cache_dir: str | None = None,
        normalize: bool = True,
        pooling: str = "mean",
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = name
        self.model_dir = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.pooling = pooling
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"编码器目录不存在: {model_dir}\n"
                f"请先运行 `python -m poisonguard.cli download-encoder` "
                f"或修改 configs/*.yaml 中 encoder.model_dir"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        ).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)
        print(f"[encoder] {name} loaded: {model_dir} (dim={self.dim}, device={self.device})")

    # ------------------------------------------------------------------
    def _forward(self, texts: list[str]) -> np.ndarray:
        import torch

        embs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i: i + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                last = out.last_hidden_state  # [B, L, H]
                if self.pooling == "cls":
                    vec = last[:, 0]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    vec = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                embs.append(vec.detach().float().cpu().numpy())
                done = min(i + self.batch_size, len(texts))
                if (i // self.batch_size) % 20 == 0 or done == len(texts):
                    print(f"\r[encoder] {done}/{len(texts)}", end="", flush=True)
        print()
        return np.concatenate(embs, axis=0)

    def encode(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        cache_path = None
        if use_cache and self.cache_dir:
            h = _sha1("\n".join(texts) + f"|{self.model_dir}|{self.max_length}|{self.pooling}")
            cache_path = self.cache_dir / f"emb_{self.name}_{h}.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                if data["emb"].shape[0] == len(texts):
                    print(f"[encoder] cache hit: {cache_path.name}")
                    return data["emb"]
        emb = self._forward(texts)
        if self.normalize:
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norm, 1e-8, None)
        emb = emb.astype(np.float32)
        if cache_path is not None:
            np.savez_compressed(cache_path, emb=emb)
        return emb


# ----------------------------------------------------------------------
_MODEL_SPECS = {
    "crom": {
        "model_id": "damo/nlp_corom_sentence-embedding_chinese-base",
        "dir": "model_cache/damo/nlp_corom_sentence-embedding_chinese-base",
        "max_length": 64,
    },
    "bge": {
        "model_id": "BAAI/bge-small-zh-v1.5",
        "dir": "model_cache/bge-small-zh-v1.5",
        "max_length": 512,
    },
    "tiny": {
        "model_id": "(builtin)",
        "dir": "",
        "max_length": 64,
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "dir": "model_cache/Qwen3-Embedding-0.6B",
        "max_length": 512,
    },
}


class TinyEncoder:
    """确定性哈希编码器：**不下载任何权重**，专供 CI 端到端烟测。

    为什么需要它：真实编码器要下 100M+ 权重，CI 里几分钟跑不完、还会因为
    网络抖动而红。这个编码器用 blake2b 把字符 2-3gram 有符号哈希到固定维度，
    完全确定、纯 numpy，几秒就能编码上万条——它不追求效果，只保证
    "data → detect → scan → deploy → run" 五段链路的接线能被真的跑一遍。
    """

    def __init__(self, dim: int = 64, max_length: int = 64, normalize: bool = True,
                 cache_dir=None, **_ignored):
        self.dim = int(dim)
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.name = "tiny"
        self.model_dir = "(builtin:tiny)"
        self.cache_dir = cache_dir

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        s = f" {text} "
        for n in (2, 3):
            for i in range(len(s) - n + 1):
                g = s[i:i + n]
                h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                v[idx] += 1.0 if (h[4] & 1) else -1.0
        if self.normalize:
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v /= norm
        return v

    def encode(self, texts, use_cache: bool = True) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)


def get_encoder(cfg):
    """按 cfg.encoder 构造编码器。"""
    ecfg = cfg.encoder
    name = ecfg.name
    if name == "tiny":
        return TinyEncoder(dim=getattr(ecfg, "tiny_dim", 64),
                           max_length=ecfg.max_length,
                           normalize=ecfg.normalize,
                           cache_dir=ecfg.cache_dir)
    spec = _MODEL_SPECS.get(name)
    if spec is not None and not Path(ecfg.model_dir).exists():
        # 允许 model_dir 指向还没下载的默认位置
        ecfg.model_dir = str(Path(cfg.outputs_dir).parent / spec["dir"])
    return SentenceEncoder(
        model_dir=ecfg.model_dir,
        name=name,
        device=ecfg.device,
        batch_size=ecfg.batch_size,
        max_length=ecfg.max_length,
        cache_dir=ecfg.cache_dir,
        normalize=ecfg.normalize,
    )


def download_encoder(name: str = "bge", repo_root: str | None = None) -> str:
    """从 ModelScope 下载编码器到 model_cache/(本机无法访问 HuggingFace，故走 ModelScope)。"""
    from modelscope.hub.snapshot_download import snapshot_download

    spec = _MODEL_SPECS[name]
    root = Path(repo_root or Path(__file__).resolve().parent.parent)
    target = root / spec["dir"]
    target.parent.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        spec["model_id"],
        cache_dir=str(target.parent),
        revision="master",
    )
    print(f"[encoder] {name} 已下载至 {local}")
    return local
