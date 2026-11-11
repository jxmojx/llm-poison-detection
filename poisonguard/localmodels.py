"""本机模型资产定位 + ModelScope 下载(本机无法访问 HuggingFace)。

设计原则：
1. **离线优先**：先扫 `model_cache/` 里已存在的目录，命中就用，绝不联网。
2. **ModelScope 兜底**：只有显式调用 `ensure_model()` 才下载。
3. 所有路径都从仓库根解析，换机器只需改 configs/default.yaml。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = ROOT / "model_cache"


def _ms_escape(name: str) -> str:
    """ModelScope 落盘时会把名字里的 `.` 替换成 `___`。

    实测：`Qwen/Qwen2.5-0.5B-Instruct` 下载后目录叫
    `model_cache/Qwen/Qwen2___5-0___5B-Instruct`。不处理这个替换，
    刚下完的模型会被 `find_local_model_dir` 判成"本地没有"，于是重复下载。
    """
    return name.replace(".", "___")


def find_local_model_dir(model_id: str, cache_dir: str | Path | None = None) -> Path | None:
    """在缓存目录里找 model_id 的本地副本。

    ModelScope 的目录结构是 `model_cache/<namespace>/<name>`，
    所以既接受 "damo/xxx" 也接受裸名字 "xxx"；同时兼容它把 `.` 转义成 `___` 的行为。
    """
    cache = Path(cache_dir) if cache_dir else MODEL_CACHE
    if not cache.exists():
        return None
    tail = model_id.split("/")[-1]
    tails = {tail, _ms_escape(tail)}
    stems = {model_id, _ms_escape(model_id)}
    candidates = [model_id, model_id.replace("/", os.sep), _ms_escape(model_id)]
    for c in candidates:
        p = cache / c
        if p.is_dir():
            return p
    for p in cache.glob("*/*"):
        if p.is_dir() and p.name in tails:
            return p
    for p in cache.glob("*"):
        if p.is_dir() and (p.name in tails or p.name in stems):
            return p
    return None


def ensure_model(model_id: str, cache_dir: str | Path | None = None,
                 revision: str | None = None) -> Path:
    """本地有就返回，没有则从 ModelScope 下载。"""
    local = find_local_model_dir(model_id, cache_dir)
    if local is not None:
        return local
    try:
        from modelscope import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "需要 modelscope：pip install modelscope\n"
            f"(本机访问不了 HuggingFace，模型必须走 ModelScope) 原错误: {e}"
        ) from e
    cache = Path(cache_dir) if cache_dir else MODEL_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    print(f"[models] 从 ModelScope 下载 {model_id} -> {cache}")
    return Path(snapshot_download(model_id, cache_dir=str(cache), revision=revision))


def find_bert_tokenizer_dir() -> Path | None:
    """找一个可用的中文 BERT 词表(优先与 baseline_model.pth 同源的 CoROM)。"""
    for mid in (
        "damo/nlp_corom_sentence-embedding_chinese-base",
        "google-bert/bert-base-chinese",
        "tiansz/bert-base-chinese",
    ):
        d = find_local_model_dir(mid)
        if d is not None and (d / "vocab.txt").exists():
            return d
    for p in MODEL_CACHE.glob("**/vocab.txt"):
        return p.parent
    return None


def which_python() -> str:
    return os.sys.executable
