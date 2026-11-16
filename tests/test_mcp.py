# -*- coding: utf-8 -*-
"""P2 MCP 协议层纯逻辑测试：不加载任何模型。

只测 dispatch / serve_stdio 的 JSON-RPC 契约（initialize / tools/list /
tools/call / 错误码 / 协议纪律），handler 的模型懒装配路径不在本文件覆盖。
guard_text / guard_batch 的参数校验在触发 _lazy_gate() 之前发生，因此
可以用「非法参数」用例安全地测到 -32602 分支。
"""
from __future__ import annotations

import io
import json

import pytest

from poisonguard.mcp import Server, Tool, build_tools, serve_stdio


def call(server: Server, msg: dict) -> dict | None:
    return server.dispatch(msg)


# ----------------------------------------------------------------------
# initialize / notifications / ping
# ----------------------------------------------------------------------
def test_initialize_returns_protocol_metadata():
    s = Server()
    r = call(s, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert r is not None
    assert r["id"] == 1
    assert r["result"]["protocolVersion"] == "2024-11-05"
    assert r["result"]["capabilities"]["tools"]["listChanged"] is False
    assert r["result"]["serverInfo"]["name"] == "poisonguard"


def test_initialize_notification_returns_none():
    s = Server()
    r = call(s, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r is None


def test_ping_returns_empty_result():
    s = Server()
    r = call(s, {"jsonrpc": "2.0", "id": 9, "method": "ping"})
    assert r == {"jsonrpc": "2.0", "id": 9, "result": {}}


# ----------------------------------------------------------------------
# tools/list
# ----------------------------------------------------------------------
def test_tools_list_exposes_five_tools_with_schema():
    s = Server()
    r = call(s, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = r["result"]["tools"]
    names = [t["name"] for t in tools]
    assert names == ["guard_text", "guard_batch", "review_queue",
                     "record_decision", "ingest_file"]
    for t in tools:
        assert "description" in t and "inputSchema" in t
    # guard_text 的 schema 声明 text 必填
    guard = next(t for t in tools if t["name"] == "guard_text")
    assert guard["inputSchema"]["required"] == ["text"]


# ----------------------------------------------------------------------
# tools/call：参数校验（不触发模型加载）
# ----------------------------------------------------------------------
def test_call_missing_text_is_param_error():
    s = Server()
    r = call(s, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "guard_text", "arguments": {}}})
    assert "error" in r
    assert r["error"]["code"] == -32602
    assert "非空字符串" in r["error"]["message"]


def test_call_non_string_text_is_param_error():
    s = Server()
    r = call(s, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                 "params": {"name": "guard_text",
                            "arguments": {"text": 123}}})
    assert r["error"]["code"] == -32602


def test_call_batch_empty_is_param_error():
    s = Server()
    r = call(s, {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                 "params": {"name": "guard_batch",
                            "arguments": {"texts": []}}})
    assert r["error"]["code"] == -32602


def test_call_batch_non_string_elem_is_param_error():
    s = Server()
    r = call(s, {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                 "params": {"name": "guard_batch",
                            "arguments": {"texts": ["ok", 7]}}})
    assert r["error"]["code"] == -32602


def test_call_unknown_tool_is_method_not_found():
    s = Server()
    r = call(s, {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                 "params": {"name": "no_such_tool", "arguments": {}}})
    assert r["error"]["code"] == -32601
    assert "no_such_tool" in r["error"]["message"]


# ----------------------------------------------------------------------
# 协议层错误码
# ----------------------------------------------------------------------
def test_non_dict_message_is_invalid_request_error():
    s = Server()
    r = call(s, [1, 2, 3])
    assert "error" in r and "result" not in r
    assert r["error"]["code"] == -32600


def test_unknown_method_is_method_not_found():
    s = Server()
    r = call(s, {"jsonrpc": "2.0", "id": 8, "method": "bogus/method"})
    assert r["error"]["code"] == -32601
    assert "bogus/method" in r["error"]["message"]


def test_tools_call_notification_returns_error_response():
    s = Server()
    # 无 id 的 tools/call 是非法请求
    r = call(s, {"jsonrpc": "2.0", "method": "tools/call",
                 "params": {"name": "guard_text", "arguments": {}}})
    assert "error" in r
    assert r["error"]["code"] == -32600


# ----------------------------------------------------------------------
# serve_stdio：换行分隔 JSON 协议纪律
# ----------------------------------------------------------------------
def test_serve_stdio_emits_only_jsonrpc_lines():
    s = Server()
    inf = io.StringIO(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
        '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n'
        '{"jsonrpc":"2.0","id":3,"method":"ping"}\n'
    )
    outf = io.StringIO()
    rc = serve_stdio(s, inf=inf, outf=outf)
    assert rc == 0
    lines = [l for l in outf.getvalue().splitlines() if l.strip()]
    assert len(lines) == 3
    for line in lines:
        d = json.loads(line)  # 每行都必须是合法 JSON —— 无 [encoder]/[gate] 噪声
        assert d.get("id") in (1, 2, 3)


def test_serve_stdio_parse_error_yields_min32700():
    s = Server()
    inf = io.StringIO("this is not json\n")
    outf = io.StringIO()
    serve_stdio(s, inf=inf, outf=outf)
    lines = [l for l in outf.getvalue().splitlines() if l.strip()]
    d = json.loads(lines[0])
    assert d["error"]["code"] == -32700
