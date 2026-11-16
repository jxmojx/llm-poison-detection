# MCP 接入说明（P2）

poisonguard 通过 **MCP（Model Context Protocol）stdio 传输**暴露为 Agent 可调的
工具服务器：Agent（Claude Desktop、DSH、Cursor 等支持 MCP 的宿主）可以把「检测
提示词/语料投毒」当成五个工具直接调，检测器、判官、人审、入库全部复用既有模块。

## 一、为什么用 stdio + 零依赖 MCP

- 协议层只依赖标准库（`json`/`sys`），不引 MCP SDK → 安装即用。
- 传输用**换行分隔 JSON**（MCP 官方 stdio 子集），`initialize` /
  `notifications/initialized` / `tools/list` / `tools/call` / `ping`。
- 模型**懒装配**：`tools/list` 不加载任何权重；第一次 `guard_text`/`guard_batch`
  调用时才预热检测器（学生模型 ~1.6 秒 + 判官按 `--judge-topk` 按需加载）。
- **协议纪律**：stdout 上只出现 JSON-RPC 响应；内部库的 `[encoder]`/`[gate]`
  日志全部重定向到 stderr，不会污染流。

## 二、五个工具

| 工具 | 入参（必填*） | 返回 |
| --- | --- | --- |
| `guard_text` | `text*`, `scene` | verdict(allow/watch/block) + signals + rules_hit + explain(scenes_views/threshold) |
| `guard_batch` | `texts*`, `scene` | 逐条 verdict 列表 + counts{allow,watch,block} |
| `review_queue` | limit, offset, out_dir, db | 未判决队列 Top-N + stats（BLOCK/WATCH/PASS 计数） |
| `record_decision` | `key*`, `decision*(benign/poison/unsure/pending)`, reason, note, reviewer, out_dir, db | 落一帧人审判定进 C4 闭环 + 剩余队列数 |
| `ingest_file` | `path*`, fmt, column, limit, out_dir, ledger_dir | 三档统计 + records_added + error |

`scene` 三档：`prompt`（用户输入）/ `retrieval`（RAG 检索片段）/ `tool_output`（工具返回）。
`guard_text` 的实际场景抓取效果：规则注入 → `watch`（rules=ignore 等）；
模板中毒 → 语料层分数抬升；域漂移（tnews 新闻进 afqmc 域）→ 诚实放行 + 提示。

## 三、怎么接入你的客户端

1. 复制 `.mcp.json.example` 到 `.mcp.json`（或并入你客户端的 mcpServers 配置），
   改 `cwd` 为你的仓库绝对路径。
2. 客户端启动时会自动拉起 `python -m poisonguard.mcp`。
3. Agent 侧即可调用 `guard_text("忽略上面所有规则…")` 拿改判信号，或
   `ingest_file("/data/语料.txt")` 把整份语料入库。

```bash
# 不需要客户端也能手动冒烟：
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"guard_text","arguments":{"text":"忽略上面所有规则"}}}' \
  | python -m poisonguard.mcp
```

## 四、参数语义（与 CLI 完全一致）

- `guard_text`/`guard_batch`：内部走 `gate.Gate.guard()`，与 `poisonguard.cli gate`
  同一条链路；阈值来源打印在 explain.threshold_source（calibrated）。
- `record_decision` 等价 `review` 界面里的 bening/poison 判定，决策写入
  `outputs/ingest/review.sqlite3`，供 `feedback` 计算误报率/影子阈值。
- `ingest_file` 等价 `ingest` 命令：幂等（file_sha+row_id）、三档门禁、可追溯。

## 五、安全与边界（诚实说明）

- stdio 传输**不做鉴权**：谁启动进程谁就能调，和本地 CLI 权限一致。远程暴露请
  套一层受控 shell/HTTP 网关（`serve --host 127.0.0.1` 目前仅本机）。
- `guard_text` 判的是**注入/投毒信号**，不是内容安全：合法但诱导性的指令也会命中
  `ignore` 类规则并给 watch——这是刻意的保守默认（watch-by-default）。
- 首次 `guard_text` 会加载学生模型（~1.6s）；`--judge-topk > 0` 时会请求 Qwen
  判官（0.8~1.4s/条）。批量场景优先 `guard_batch`（单次装配、逐条打分）。
