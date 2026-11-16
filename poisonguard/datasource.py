"""数据源 adapter（环节 30-4）：把 datasets(HF)/DVC/S3 等的"取文本行"统一成本地板语义。

核心契约：任何 adapter 都返回与 `scan.load_rows` 相同的 `[{row_id, text}]` 结构，
这样 ingest / scan 的打分链路能在任意数据源上复用——导数据与打分彻底解耦。

- 本地文件：直接走 `scan.load_rows`（不重复造轮子）。
- HF/datasets：`hf://repo_id[/split]`，懒加载 `datasets.load_dataset`。
- DVC：`dvc://path[@rev]`，懒加载 `dvc.api`；未装时给出可读错误。
- S3：`s3://bucket/key`，懒加载 `boto3`；未装时给出可读错误。

测试友好：三个 adapter 都接受注入的 loader / client / read 回调，
单测用假对象即可覆盖全部逻辑，不需要联网、不需要下载模型。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .scan import load_rows

# 前缀路由：spec 是这些前缀时走对应 adapter，否则当作本地文件
PREFIX_HF = "hf://"
PREFIX_DVC = "dvc://"
PREFIX_S3 = "s3://"

_MISSING_MSG = {
    "hf": "读取 HF/datasets 需要安装 datasets；pip install datasets",
    "dvc": "读取 DVC 需要安装 dvc；pip install dvc",
    "s3": "读取 S3 需要安装 boto3；pip install boto3",
}


def _rows_from_text(text: str, *, fmt: str, column: str,
                    limit: int, name: str) -> list[dict]:
    """把一段文本用 load_rows 解析成 [{row_id, text}]（走临时文件，避免生成两份读取逻辑）。

    注意：DVC/S3 拿到的是一段字节流，不一定带正确的本地扩展名。
    这里统一按 `.txt` 语义把"每行当一条文本"（row_id=行号），
    与 ingest/scan 的 txt 格式语义一致；JSON 行也逐行当文本即可。
    """
    suffix = ".txt"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                     suffix=suffix, delete=False) as f:
        f.write(text)
        tmp = Path(f.name)
    try:
        return load_rows(tmp, fmt=fmt or "txt", column=column, limit=limit)
    finally:
        tmp.unlink(missing_ok=True)


def _rows_hf(spec: str, *, column: str = "text", limit: int = 0,
             loader=None) -> list[dict]:
    """HF / datasets 数据源：`hf://repo_id` 或 `hf://repo_id/split`。

    loader 可注入（默认懒加载 datasets.load_dataset），单测传假 loader 即可。
    """
    rest = spec[len(PREFIX_HF):]
    parts = rest.split("/")
    repo = parts[0]
    split = parts[1] if len(parts) > 1 and parts[1] else "train"
    if not repo:
        raise ValueError(f"hf:// 需要仓库 id，例如 hf://IMDB 或 hf://tyqiangz/chinese-sentiment-corpora/train；实际: {spec!r}")
    if loader is None:
        try:
            from datasets import load_dataset as _ld
        except ImportError:
            raise RuntimeError(_MISSING_MSG["hf"])
        loader = _ld
    ds = loader(repo, split=split)
    n = len(ds)
    if limit:
        n = min(n, int(limit))
    rows = []
    for i in range(n):
        val = ds[i][column] if column in ds.column_names else ds[i]["text"]
        rows.append({"row_id": i, "text": str(val)})
    return rows


def _rows_dvc(spec: str, *, column: str = "text", limit: int = 0,
              read=None, repo: str | None = None) -> list[dict]:
    """DVC 数据源：`dvc://path` 或 `dvc://path@rev`。

    read 可注入：read(path, rev) -> str（默认懒加载 dvc.api.read）。
    repo 参数用于远仓（缺省 = 当前工作区被 DVC 管理的仓）。
    """
    rest = spec[len(PREFIX_DVC):]
    if "@" in rest:
        path, _, rev = rest.partition("@")
        rev = rev or None
    else:
        path, rev = rest, None
    if read is None:
        try:
            from dvc import api as _dvc_api
        except ImportError:
            raise RuntimeError(_MISSING_MSG["dvc"])
        read = lambda p, r: _dvc_api.read(p, repo=repo, rev=r)  # noqa: E731
    text = read(path, rev)
    return _rows_from_text(text, fmt="auto", column=column, limit=limit, name=path)


def _rows_s3(spec: str, *, column: str = "text", limit: int = 0,
             client=None) -> list[dict]:
    """S3 数据源：`s3://bucket/key`。

    client 可注入（默认懒加载 boto3.client('s3')），单测传假 client 即可。
    """
    rest = spec[len(PREFIX_S3):]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"s3:// 需要 bucket/key，例如 s3://my-bucket/clean.jsonl；实际: {spec!r}")
    if client is None:
        try:
            import boto3
        except ImportError:
            raise RuntimeError(_MISSING_MSG["s3"])
        client = boto3.client("s3")
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if isinstance(body, bytes):
        text = body.decode("utf-8")
    else:
        text = str(body)
    return _rows_from_text(text, fmt="auto", column=column, limit=limit, name=key)


def load_rows_source(spec: str, *, column: str = "text", limit: int = 0,
                     fmt: str = "auto", loader=None, client=None,
                     read=None, repo: str | None = None) -> tuple[list[dict], str]:
    """统一入口：按 spec 前缀路由到对应 adapter，返回 (rows, kind)。

    - kind ∈ {"local", "hf", "dvc", "s3"}
    - rows 与 scan.load_rows 相同：[{row_id, text}]
    - 纯本地路径 / 相对路径 / 绝对路径 → local（复用 load_rows，含 txt/csv/jsonl 推断）
    """
    if spec.startswith(PREFIX_HF):
        return _rows_hf(spec, column=column, limit=limit, loader=loader), "hf"
    if spec.startswith(PREFIX_DVC):
        return (_rows_dvc(spec, column=column, limit=limit, read=read, repo=repo), "dvc")
    if spec.startswith(PREFIX_S3):
        return _rows_s3(spec, column=column, limit=limit, client=client), "s3"
    p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(f"本地数据源不存在: {p}")
    rows = load_rows(p, fmt=fmt, column=column, limit=limit)
    return rows, "local"
