# -*- coding: utf-8 -*-
"""MCP 接入（P2+）：把 poisonguard 暴露成 Agent 可调的 stdio/HTTP JSON-RPC 服务。

协议版本：MCP 2024-11-05 子集（initialize / notifications/initialized /
tools/list / tools/call / resources/prompts（Gap 3））。传输层：stdio 用换行
分隔 JSON；HTTP 用 /mcp（Streamable HTTP，Gap 5）。鉴权：token 可注入
（Gap 4）。

工具（M2 方案 + Gap 2 可解释性）：
  - guard_text        单条文本 → verdict(allow/watch/block) + 信号 + 可解释
  - guard_batch       多条文本 → 逐条 verdict
  - explain           对 guard 结果做脱机可读解释（verdict/原因/信号/处置）
  - read_alert        按 key 读一条告警详情（含溯源视图）
  - review_queue      读人审队列（BLOCK/WATCH 未判决 Top-N + 统计）
  - record_decision   落一条人审判定（进 C4 闭环）
  - ingest_file       入库一份文件 → 三档统计

设计约束：
  - 零第三方依赖（不引 MCP SDK），服务端状态全部懒装配（tools/list 不加载模型）。
  - 每个工具都是对既有模块（gate / review / ingest / explain）的薄封装——
    不复制业务逻辑。
  - stdio 一次一个 JSON 行；顺序执行（Agent 并发不需要，保持简单确定）。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "poisonguard"
SERVER_VERSION = "1.0"

# Gap 4：鉴权
# - stdio：客户端在 initialize 的 params.authorization 里带
#   "Bearer <token>"（MCP 2025-03-26 认证草案的传输无关方式）。
# - HTTP：Authorization: Bearer <token> 头（Gap 5 的 serve_http 实现）。
# - token 为空表示不鉴权（本地 stdio 默认）。默认从环境变量读，避免写死。
DEFAULT_MCP_TOKEN_ENV = "POISONGUARD_MCP_TOKEN"


def _env_token() -> str:
    import os
    return os.environ.get(DEFAULT_MCP_TOKEN_ENV, "") or ""


def _extract_bearer(params: dict) -> str:
    """从 params.authorization 提取 token（无则空串）。"""
    auth = params.get("authorization")
    if isinstance(auth, str):
        return auth.removeprefix("Bearer ").strip()
    return ""

# ----------------------------------------------------------------------
# Gap 3：resources / prompts 协议子集
# ----------------------------------------------------------------------
# resources：只读的"语义资源"——guard 策略快照 / 注入头指标 / 运行状况，
# 均为纯 JSON 变换，不加载模型。prompts：模板化的 guard 提示词。
RESOURCE_URIS = {
    "poisonguard://guard/policy": {
        "name": "guard 策略快照",
        "description": "当前生效的 watch/block 阈值、scene 收紧倍数、目标 FPR",
        "mimeType": "application/json",
    },
    "poisonguard://metrics/inject": {
        "name": "注入头指标",
        "description": "P3 注入头 seen/unseen 两列评测指标（inject.json）",
        "mimeType": "application/json",
    },
    "poisonguard://server/status": {
        "name": "服务器状态",
        "description": "协议版本、工具数、资源数、gate 装配与调用计数",
        "mimeType": "application/json",
    },
}

PROMPT_TEMPLATES = {
    "poisonguard://prompts/guard": {
        "name": "guard 判定提示词",
        "description": "指导 Agent 用 guard_text 判定一条文本，并按三档处理结果",
        "arguments": [],
    },
}


def _policy_resource(cfg: Config) -> dict:
    """guard 策略资源：尽量读 gate 配置（不装配模型），失败给保守默认值。"""
    try:
        from .gate import GateConfig
        gc = GateConfig.from_cfg(cfg) if hasattr(GateConfig, "from_cfg") else GateConfig()
        return {
            "default_scene": "prompt",
            "watch_mult": getattr(gc, "watch_mult", None),
            "block_mult": getattr(gc, "block_mult", None),
            "target_fpr": getattr(gc, "target_fpr", None),
            "scenes": {"prompt": 1.0, "retrieval": 1.0, "tool_output": 1.0},
            "threshold_source": "calibrated",
        }
    except Exception:
        return {"default_scene": "prompt", "scenes": {}, "note": "gate 未装配"}


def _inject_metrics_resource(cfg: Config) -> dict:
    """注入头指标资源：读 outputs/metrics/inject.json（Gap 1 产物）。"""
    metrics_dir = getattr(cfg, "metrics_dir", None) or "outputs/metrics"
    p = Path(metrics_dir) / "inject.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {"model": data.get("model"), "seen": data.get("seen"),
                "unseen": data.get("unseen"), "head": data.get("head_params"),
                "source": str(p)}
    except Exception:
        return {"model": None, "seen": None, "unseen": None,
                "note": "inject.json 尚未生成（跑 inject-train）", "source": str(p)}


def _status_resource(cfg: Config) -> dict:
    metrics_dir = getattr(cfg, "metrics_dir", None) or "outputs/metrics"
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "serverName": SERVER_NAME,
        "serverVersion": SERVER_VERSION,
        "tools": len(build_tools()),
        "resources": len(RESOURCE_URIS),
        "prompts": len(PROMPT_TEMPLATES),
        "injection_head_ready": (Path(metrics_dir) / "inject.json").exists(),
    }


def _read_resource(cfg: Config, uri: str) -> dict | None:
    """按 URI 返回 {contents:[...], uri}；未知返回 None。"""
    if uri == "poisonguard://guard/policy":
        return {"contents": [{"uri": uri, "mimeType": "application/json",
                              "text": json.dumps(_policy_resource(cfg), ensure_ascii=False)}]}
    if uri == "poisonguard://metrics/inject":
        return {"contents": [{"uri": uri, "mimeType": "application/json",
                              "text": json.dumps(_inject_metrics_resource(cfg), ensure_ascii=False)}]}
    if uri == "poisonguard://server/status":
        return {"contents": [{"uri": uri, "mimeType": "application/json",
                              "text": json.dumps(_status_resource(cfg), ensure_ascii=False)}]}
    return None


def _get_prompt(cfg: Config, name: str) -> dict | None:
    if name == "poisonguard://prompts/guard":
        return {
            "name": name,
            "description": PROMPT_TEMPLATES[name]["description"],
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": (
                            "对下面这条文本调用 guard_text（scene=prompt）做投毒/注入判定：\n"
                            "…\n"
                            "判定后按三档处理：allow=正常使用；watch=建议人工复核/降权；\n"
                            "block=拦截并转入人审队列。结果里带 explanation 时，用 reason 向\n"
                            "调用方解释为什么这样判。"
                        ),
                    },
                }
            ],
        }
    return None


class MCPError(Exception):
    """JSON-RPC 错误。code 遵循 MCP 约定：-32600 无效请求 / -32601 方法不存在
    / -32602 参数非法 / -32603 内部错误。"""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _err(code: int, message: str, data: Any = None) -> dict:
    e: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        e["data"] = data
    return e


# ----------------------------------------------------------------------
# 工具注册表（幂等：都是一等函数，输入输出纯 dict）
# ----------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Any


def _guard_one(gate, text: str, scene: str = "prompt", explain: bool = True) -> dict:
    from .gate import Gate
    if gate is None:
        raise MCPError(-32603, "gate 未初始化（服务器未装配）")
    r = gate.guard(text, scene=scene if scene in ("prompt", "retrieval", "tool_output") else "prompt")
    d = r.as_dict(text=True)
    if explain:
        from .explain import attach_explanation
        d = attach_explanation(d)
    return d


def _review_store(cfg: Config, db: str | None = None) -> Any:
    from .review import ReviewStore
    path = db or str(Path(cfg.out_dir) / "ingest" / "review.sqlite3")
    return ReviewStore(path)


def _ingestor(cfg: Config, out_dir: str | None = None, ledger_dir: str | None = None,
              watch_band: float = 0.05) -> Any:
    """懒装配 Ingestor；scorer 只缺省（调用方不传 bundle 时用默认 build_scorer）。"""
    from .ingest import Ingestor
    from .service import build_scorer
    scorer = build_scorer(cfg)
    return Ingestor(
        scorer,
        out_dir=out_dir or "outputs/ingest",
        ledger_path=Path(ledger_dir or "outputs/ingest") / "ledger.sqlite3",
        watch_band=watch_band,
        explain=True,
    )


def build_tools(server=None) -> list[Tool]:
    """不加载任何模型。handler 只在 tools/call 时才真正触发懒装配。"""

    def h_guard_text(cfg: Config, args: dict) -> dict:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise MCPError(-32602, "text 必须是非空字符串")
        gate = server._lazy_gate()
        d = _guard_one(gate, text, str(args.get("scene", "prompt")))
        if not (args.get("explain") is False):
            pass  # _guard_one 已默认附 explanation
        return d

    def h_guard_batch(cfg: Config, args: dict) -> dict:
        texts = args.get("texts")
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
            raise MCPError(-32602, "texts 必须是非空字符串数组")
        gate = server._lazy_gate()
        from .gate import guard_texts
        res = guard_texts(texts, scene=str(args.get("scene", "prompt")), gate=gate)
        counts = {"allow": 0, "watch": 0, "block": 0}
        out = []
        for r in res:
            if args.get("explain") is not False:
                from .explain import attach_explanation
                r = attach_explanation(r)
            out.append(r)
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        return {"n": len(out), "counts": counts, "results": out}

    def h_review_queue(cfg: Config, args: dict) -> dict:
        from .review import load_items
        db = args.get("db")
        store = _review_store(cfg, db)
        decided = store.decided_keys()
        items = load_items(
            args.get("out_dir") or "outputs/ingest",
            flags=("BLOCK", "WATCH"),
            limit=int(args.get("limit", 50) or 50),
            offset=int(args.get("offset", 0) or 0),
            decided=decided,
        )
        slim = [{
            "key": it.get("key"),
            "text": it.get("text", ""),
            "score": it.get("score"),
            "flag": it.get("flag"),
            "source": it.get("source"),
            "row_id": it.get("row_id"),
        } for it in items]
        return {"n_undecided": len(items), "items": slim, "stats": store.stats()}

    def h_record_decision(cfg: Config, args: dict) -> dict:
        from .review import item_key, load_items, DECISIONS
        key = args.get("key")
        decision = args.get("decision")
        reason = str(args.get("reason", "") or "")
        note = str(args.get("note", "") or "")
        reviewer = str(args.get("reviewer", "") or "agent")
        if not isinstance(key, str) or not key:
            raise MCPError(-32602, "key 必填")
        if decision not in DECISIONS:
            raise MCPError(-32602, f"decision 必须是 {DECISIONS} 之一，收到 {decision!r}")
        store = _review_store(cfg, args.get("db"))
        # 从队列找回 item（decide 需要 score/flag/source/row_id 快照）
        items = load_items(args.get("out_dir") or "outputs/ingest",
                           flags=("BLOCK", "WATCH", "PASS"), decided=store.decided_keys() - {key})
        item = next((it for it in items if it.get("key") == key), None)
        if item is None:
            # key 可能已被判过——把它从已判决集合里放出来再查一次
            items2 = load_items(args.get("out_dir") or "outputs/ingest",
                                flags=("BLOCK", "WATCH", "PASS"), decided=set())
            item = next((it for it in items2 if item_key(it) == key), None)
        if item is None:
            raise MCPError(-32602, f"队列里找不到 key={key!r}")
        rec = store.decide(item, decision, reason=reason, note=note, reviewer=reviewer)
        return {"recorded": rec, "queue_remaining": len(load_items(
            args.get("out_dir") or "outputs/ingest",
            flags=("BLOCK", "WATCH"), decided=store.decided_keys()))}

    def h_ingest_file(cfg: Config, args: dict) -> dict:
        path = args.get("path")
        if not isinstance(path, str) or not Path(path).exists():
            raise MCPError(-32602, f"文件不存在: {path!r}")
        ing = _ingestor(cfg,
                        out_dir=args.get("out_dir"),
                        ledger_dir=args.get("ledger_dir"),
                        watch_band=float(args.get("watch_band", 0.05) or 0.0))
        r = ing.ingest_file(path, fmt=str(args.get("fmt", "auto") or "auto"),
                            column=str(args.get("column", "text") or "text"),
                            limit=int(args.get("limit", 0) or 0))
        return {"status": r.get("status"), "n": r.get("n", 0),
                "block": r.get("block", 0), "watch": r.get("watch", 0),
                "pass": r.get("pass", 0), "seconds": r.get("seconds", 0.0),
                "records_added": r.get("records_added", 0),
                "error": r.get("error")}

    def h_explain(cfg: Config, args: dict) -> dict:
        """对一条 guard 结果（dict）做脱机解释：不跑模型，只做纯变换。"""
        result = args.get("result")
        verdict = (result or {}).get("verdict")
        if verdict not in ("allow", "watch", "block"):
            raise MCPError(-32602, "result 必须含 verdict ∈ {allow,watch,block}")
        from .explain import attach_explanation, build_explanation
        return {
            "explained": attach_explanation(result),
            "explanation": build_explanation(result),
        }

    def h_read_alert(cfg: Config, args: dict) -> dict:
        key = args.get("key")
        if not isinstance(key, str) or not key:
            raise MCPError(-32602, "key 必填（source|row_id|text_sha）")
        from .explain import read_alert
        rec = read_alert(key, out_dir=str(args.get("out_dir", "outputs/ingest") or "outputs/ingest"))
        if rec is None:
            return {"found": False, "key": key}
        return {"found": True, "alert": rec}

    return [
        Tool("guard_text",
             "对单条文本做投毒/注入判定，返回 allow/watch/block 三档及信号解释。"
             "scene ∈ {prompt, retrieval, tool_output}。",
             {"type": "object",
              "properties": {"text": {"type": "string"},
                             "scene": {"type": "string"},
                             "explain": {"type": "boolean"}},
              "required": ["text"]},
             h_guard_text),
        Tool("guard_batch",
             "批量判定多条文本，返回每条 verdict 与分档计数。",
             {"type": "object",
              "properties": {"texts": {"type": "array", "items": {"type": "string"}},
                             "scene": {"type": "string"},
                             "explain": {"type": "boolean"}},
              "required": ["texts"]},
             h_guard_batch),
        Tool("explain",
             "对任意一条 guard 结果做可读解释（脱机、不跑模型）："
             "verdict 中文 / reason 因果链 / action 处置建议 / trace / 阈值。",
             {"type": "object",
              "properties": {"result": {"type": "object"}},
              "required": ["result"]},
             h_explain),
        Tool("read_alert",
             "按 key 读取一条告警详情（含 flag、score、来源行、视图溯源、原文）。",
             {"type": "object",
              "properties": {"key": {"type": "string"},
                             "out_dir": {"type": "string"}},
              "required": ["key"]},
             h_read_alert),
        Tool("review_queue",
             "读取人审队列（BLOCK/WATCH 未判决样本 Top-N）与当前判决统计。",
             {"type": "object",
              "properties": {"out_dir": {"type": "string"}, "db": {"type": "string"},
                             "limit": {"type": "integer"}, "offset": {"type": "integer"}},
              "required": []},
             h_review_queue),
        Tool("record_decision",
             "记录一条人审判定（decision ∈ {benign, poison, unsure, pending}），"
             "写入 C4 反馈闭环。",
             {"type": "object",
              "properties": {"key": {"type": "string"}, "decision": {"type": "string"},
                             "reason": {"type": "string"}, "note": {"type": "string"},
                             "reviewer": {"type": "string"}, "out_dir": {"type": "string"},
                             "db": {"type": "string"}},
              "required": ["key", "decision"]},
             h_record_decision),
        Tool("ingest_file",
             "把一份 txt/csv/jsonl 文件入库：逐条打分并落 BLOCK/WATCH/PASS 三档。",
             {"type": "object",
              "properties": {"path": {"type": "string"}, "out_dir": {"type": "string"},
                             "ledger_dir": {"type": "string"}, "fmt": {"type": "string"},
                             "column": {"type": "string"}, "limit": {"type": "integer"},
                             "watch_band": {"type": "number"}},
              "required": ["path"]},
             h_ingest_file),
    ]


# ----------------------------------------------------------------------
# 服务器：懒装配 + JSON-RPC 分派
# ----------------------------------------------------------------------
class Server:
    def __init__(self, cfg: Config | None = None, tools: list[Tool] | None = None,
                 auth_token: str | None = None):
        self.cfg = cfg or Config()
        self._gate = None
        self._gate_building = False
        self.tools = tools or build_tools(server=self)
        # 鉴权：显式传参 > 环境变量；空 = 不鉴权
        self.auth_token = auth_token if auth_token is not None else _env_token()
        self._authed = self.auth_token == ""  # 无 token 直接视为已认证

    def _lazy_gate(self):
        if self._gate is not None or self._gate_building:
            return self._gate
        self._gate_building = True
        try:
            from .gate import Gate
            g = Gate(self.cfg)
            if self.cfg and getattr(self.cfg, "qwen", None) and getattr(self.cfg.qwen, "threshold", None) is None:
                pass  # 阈值由 Gate._ensure_scorer 内解决（calibrate）
            self._gate = g
            # 预热（首次 guard 会懒装配 scorer；这里触发热身，代价是一次打分）
            _ = g.guard("__warmup__")
            self._gate_building = False
        except Exception:
            self._gate = None
            self._gate_building = False
            raise
        return self._gate

    # ---- Gap 4：HTTP 传输层鉴权辅助 ----
    def authorize_bearer(self, header: str | None) -> bool:
        """HTTP 侧：从 Authorization 头提取 Bearer token 并比对。

        成功返回 True（并标记 _authed）；失败/无 token 返回 False。
        """
        if not self.auth_token:
            self._authed = True
            return True
        if not header:
            return False
        token = header.removeprefix("Bearer ").strip()
        if token and token == self.auth_token:
            self._authed = True
            return True
        return False

    # ---- JSON-RPC ----
    def dispatch(self, msg: dict) -> dict | None:
        """返回要写回 stdout 的 response（None=通知，无需回包）。"""
        if not isinstance(msg, dict):
            return self._err_response(None, _err(-32600, "请求必须是 JSON 对象"))
        method = msg.get("method")
        ident = msg.get("id")
        is_request = "id" in msg
        if method == "initialize":
            if not is_request:
                return None
            # Gap 4：鉴权在 initialize 完成——stdio 用 params.authorization
            params = msg.get("params") or {}
            client_token = _extract_bearer(params)
            if self.auth_token and client_token != self.auth_token:
                # 不泄露 token 存在与否：统一回认证失败
                return self._err_response(ident, _err(
                    -32001, "认证失败：authorization 缺失或不匹配"))
            self._authed = True
            return self._response(ident, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })
        # 未认证：除 initialize 外一律拒绝（不暴露方法存在性）
        if not self._authed:
            return self._err_response(ident, _err(-32001, "未认证：请先 initialize"))
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            if not is_request:
                return None
            return self._response(ident, {
                "tools": [{
                    "name": t.name, "description": t.description,
                    "inputSchema": t.input_schema,
                } for t in self.tools],
            })
        if method == "tools/call":
            if not is_request:
                return self._err_response(None, _err(-32600, "tools/call 必须是请求"))
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            tool = next((t for t in self.tools if t.name == name), None)
            if tool is None:
                return self._err_response(ident, _err(-32601, f"未知工具：{name!r}"))
            try:
                result = tool.handler(self.cfg, args)
                content = json.dumps(result, ensure_ascii=False, default=str)
                return self._response(ident, {
                    "content": [{"type": "text", "text": content}],
                    "isError": False,
                })
            except MCPError as e:
                return self._err_response(ident, _err(e.code, e.message, e.data))
            except Exception as e:
                return self._err_response(ident, _err(-32603, f"工具执行失败: {e}"))
        # ---- Gap 3：resources / prompts ----
        if method == "resources/list":
            if not is_request:
                return None
            return self._response(ident, {
                "resources": [
                    {"uri": uri, "name": meta["name"],
                     "description": meta["description"], "mimeType": meta["mimeType"]}
                    for uri, meta in RESOURCE_URIS.items()
                ],
            })
        if method == "resources/read":
            if not is_request:
                return self._err_response(None, _err(-32600, "resources/read 必须是请求"))
            params = msg.get("params") or {}
            uri = params.get("uri")
            res = _read_resource(self.cfg, uri) if isinstance(uri, str) else None
            if res is None:
                return self._err_response(ident, _err(-32602, f"未知资源：{uri!r}"))
            return self._response(ident, res)
        if method == "prompts/list":
            if not is_request:
                return None
            return self._response(ident, {
                "prompts": [
                    {"name": name, "description": meta["description"],
                     "arguments": meta["arguments"]}
                    for name, meta in PROMPT_TEMPLATES.items()
                ],
            })
        if method == "prompts/get":
            if not is_request:
                return self._err_response(None, _err(-32600, "prompts/get 必须是请求"))
            params = msg.get("params") or {}
            name = params.get("name")
            prompt = _get_prompt(self.cfg, name) if isinstance(name, str) else None
            if prompt is None:
                return self._err_response(ident, _err(-32602, f"未知提示词：{name!r}"))
            return self._response(ident, prompt)
        if method == "ping":
            if not is_request:
                return None
            return self._response(ident, {})
        return self._err_response(ident, _err(-32601, f"未知方法：{method!r}"))

    # ---- 响应构造 ----
    def _response(self, ident, result) -> dict:
        return {"jsonrpc": "2.0", "id": ident, "result": result}

    def _err_response(self, ident, error) -> dict:
        return {"jsonrpc": "2.0", "id": ident, "error": error}


def serve_stdio(server: Server | None = None, inf=None, outf=None) -> int:
    """阻塞式 stdio 循环。inf/outf 供测试注入 StringIO；默认 sys.stdin/stdout。

    协议纪律：stdout 上**只能**出现 JSON-RPC 响应——宿主内任何库（encoders/
    service/gate）的 print 一律重定向到 stderr，否则会污染换行分隔的 JSON 流。
    """
    server = server or Server()
    inf = inf or sys.stdin
    outf = outf or sys.stdout
    # 捕获真实 stdout 之后再切换：内部 print 走 stderr，响应仍写 outf
    real_stdout = outf
    sys.stdout = sys.stderr
    try:
        for raw in inf:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                real_stdout.write(json.dumps(
                    {"jsonrpc": "2.0", "id": None,
                     "error": _err(-32700, f"无法解析 JSON: {e}")},
                    ensure_ascii=False) + "\n")
                real_stdout.flush()
                continue
            resp = server.dispatch(msg)
            if resp is not None:
                real_stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                real_stdout.flush()
    finally:
        sys.stdout = real_stdout
    return 0


def main() -> int:
    """作为独立进程运行：python -m poisonguard.mcp"""
    return serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
