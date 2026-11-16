"""环节 30-4 数据源 adapter 测试：本地 / HF / DVC / S3 统一成 [{row_id, text}]。

全部纯逻辑：HF/DVC/S3 的 loader/client/read 都注入假对象，不联网、不下载模型。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poisonguard.datasource import (
    PREFIX_DVC, PREFIX_HF, PREFIX_S3,
    _rows_dvc, _rows_hf, _rows_s3,
    load_rows_source,
)


# ------------------------------------------------------------------ HF
class _FakeDS:
    """最小 datasets.Dataset 复刻：支持 len / column_names / 下标取行。"""

    def __init__(self, column_names, data):
        self.column_names = column_names
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]


def test_hf_rows_via_injected_loader():
    loader = lambda repo, *, split: _FakeDS(  # noqa: E731
        ["text"],
        [{"text": "涉密A"}, {"text": "简报B"}, {"text": "涉密C"}])
    rows = _rows_hf("hf://my/repo/train", loader=loader)
    assert rows == [
        {"row_id": 0, "text": "涉密A"},
        {"row_id": 1, "text": "简报B"},
        {"row_id": 2, "text": "涉密C"},
    ]


def test_hf_rows_limit_and_split_default():
    loader = lambda repo, *, split: _FakeDS(  # noqa: E731
        ["text"], [{"text": f"t{i}"} for i in range(10)])
    rows = _rows_hf("hf://repo", loader=loader, limit=3)
    assert len(rows) == 3
    assert rows[-1]["row_id"] == 2
    # 默认 split=train
    calls = []
    loader = lambda repo, split: (calls.append((repo, split)) or _FakeDS(["text"], []))
    _rows_hf("hf://abc", loader=loader)
    assert calls == [("abc", "train")]


def test_hf_custom_column():
    loader = lambda repo, *, split: _FakeDS(  # noqa: E731
        ["sentence"],
        [{"sentence": "x1"}, {"sentence": "x2"}])
    rows = _rows_hf("hf://r", loader=loader, column="sentence")
    assert rows[0]["text"] == "x1"


# ------------------------------------------------------------------ DVC
def test_dvc_rows_via_injected_read():
    read = lambda path, rev: "流言1\n报道2\n流言3\n"  # noqa: E731
    rows = _rows_dvc("dvc://data/raw/clean.txt", read=read)
    assert len(rows) == 3
    assert rows[0] == {"row_id": 0, "text": "流言1"}


def test_dvc_rev_passed_to_read():
    seen = {}
    read = lambda path, rev: (seen.update(path=path, rev=rev) or "一行\n")  # noqa: E731
    _rows_dvc("dvc://x/y.csv@v2", read=read)
    assert seen == {"path": "x/y.csv", "rev": "v2"}


def test_dvc_missing_lib_gives_friendly_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "dvc":
            raise ImportError("no dvc")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="dvc"):
        _rows_dvc("dvc://x.txt")


# ------------------------------------------------------------------ S3
class _FakeS3:
    def __init__(self, body: bytes):
        self._body = body

    def get_object(self, **kw):
        self.last = kw
        return {"Body": type("B", (), {"_body": self._body,
                                       "read": lambda self: self._body})()}


def test_s3_rows_via_injected_client():
    client = _FakeS3("毒1\n净2\n毒3\n".encode())
    rows = _rows_s3("s3://bucket/folder/clean.jsonl", client=client)
    assert client.last == {"Bucket": "bucket", "Key": "folder/clean.jsonl"}
    assert len(rows) == 3
    assert rows[0]["text"] == "毒1"

def test_s3_no_bucket_key():
    with pytest.raises(ValueError, match="s3://"):
        _rows_s3("s3://justbucket", client=_FakeS3(b""))


# ------------------------------------------------------------------ 统一入口
def test_load_rows_source_local(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("a\nb\n", encoding="utf-8")
    rows, kind = load_rows_source(str(f))
    assert kind == "local"
    assert len(rows) == 2


def test_load_rows_source_missing_local():
    with pytest.raises(FileNotFoundError):
        load_rows_source("no/such/file.txt")


def test_load_rows_source_routes_prefixes():
    _, k1 = load_rows_source(
        "hf://r/t", loader=lambda repo, *, split: _FakeDS(  # noqa: E731
            ["text"], [{"text": "t"}]))
    _, k2 = load_rows_source(
        "dvc://x.txt", read=lambda p, r: "t\n")  # noqa: E731
    _, k3 = load_rows_source(
        "s3://b/k.jsonl", client=_FakeS3("t\n".encode()))
    assert (k1, k2, k3) == ("hf", "dvc", "s3")
