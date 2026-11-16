"""人审台（C1）：把 `ingest` 产出的 BLOCK/WATCH 队列变成可判、可回流的工作台。

为什么需要它：检测器的漏报率（edit 4%、结构隔离头 29.6%）说明**必须有人复核**；
同时 1% 误报意味着每天也有几十条需要放行。没有工作台，人只能在 CSV 里手工比对，
而且判完的结论落不回系统——`C4`（误报闭环）就永远没有标签可用。

设计上的三个要点：

1. **理由码结构化**：自由文本没法回流成训练信号。确认/误报都必须选一个码
   （`REASONS`），这是 C4 的前提；
2. **证据齐全**：每条记录都带 `views`（knn / fpr_est / char_ngram / student / chars），
   人看到的不只是"0.98 分"，而是"哪个视图报警、哪个不报警"——误报归因靠这个；
3. **判决可追溯**：`decisions` 表存 `source + row_id + text_sha + model + threshold`，
   同一条被判两次时以最后一次为准（upsert），并保留 `decided_at`。

前端是一个单页：左边队列、右边证据，键盘 `j/k` 上下、`1/2/3` 判、`Enter` 提交。
"""

import csv
import json
import sqlite3
import time
from pathlib import Path

from .log import event, get_logger

DECISIONS = ("poison", "benign", "unsure")
REASONS: dict[str, list[str]] = {
    "poison": ["内容篡改(事实)", "模板投毒", "回译改写", "后门/触发词", "其他"],
    "benign": ["错别字/OCR", "口语闲聊", "领域不匹配", "正常新闻", "长文本/截断", "其他"],
    "unsure": ["证据不足", "需要外部事实核查", "其他"],
}
_FLAG_DIRS = {"BLOCK": "quarantine", "WATCH": "review", "PASS": "released"}


def item_key(rec: dict) -> str:
    """队列项的唯一键：同一文件同一行同一文本才算同一条。"""
    return f"{rec.get('source', '')}|{rec.get('row_id', -1)}|{rec.get('text_sha', '')}"


# ----------------------------------------------------------------------
# 队列（读 ingest 落盘的 jsonl）
# ----------------------------------------------------------------------
def _sample_pass(items: list[dict], n: int, seed: int = 42) -> list[dict]:
    """对放行流做**确定性**抽样：按 (key, seed) 的哈希排序取前 n 条。

    确定性很重要——同一个人明天再抽、或换个评审员来抽，抽到的是同一批，
    否则"漏报率"会变成抽签结果。
    """
    if n <= 0 or not items:
        return []
    import hashlib

    def rank(rec: dict) -> bytes:
        return hashlib.blake2b(f"{seed}|{rec.get('key', '')}".encode(),
                               digest_size=8).digest()

    return sorted(items, key=rank)[:n]


def load_items(out_dir: str | Path, *, flags: tuple[str, ...] = ("BLOCK", "WATCH"),
               limit: int = 0, offset: int = 0,
               decided: set[str] | None = None,
               sample_pass: int = 0, seed: int = 42) -> list[dict]:
    """读队列并按分数从高到低排；`decided` 里的条目会被过滤掉（已判过）。

    `sample_pass>0` 时把放行流（PASS）也纳入，并只取确定性的 N 条抽样——
    这是估计**漏报率**的唯一途径（不抽 PASS，就永远只知道拦截精度）。
    """
    out = Path(out_dir)
    seen: set[str] = set()
    items: list[dict] = []
    for flag in flags:
        sub = _FLAG_DIRS.get(flag)
        if not sub or not (out / sub).exists():
            continue
        for f in sorted((out / sub).glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = item_key(rec)
                if k in seen or (decided and k in decided):
                    continue
                seen.add(k)
                rec["key"] = k
                items.append(rec)
    if sample_pass and "PASS" in flags:
        keep = [r for r in items if r.get("flag") == "PASS"]
        sampled = _sample_pass(keep, sample_pass, seed)
        items = [r for r in items if r.get("flag") != "PASS"] + sampled
    items.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    if offset:
        items = items[offset:]
    if limit and limit > 0:
        items = items[:limit]
    return items


# ----------------------------------------------------------------------
# 判决存储
# ----------------------------------------------------------------------
class ReviewStore:
    """判决表：一条记录一个最新结论（upsert），同时保留模型/阈值快照。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS decisions(
                    key         TEXT PRIMARY KEY,
                    source      TEXT,
                    row_id      INTEGER,
                    text_sha    TEXT,
                    score       REAL,
                    flag        TEXT,
                    decision    TEXT NOT NULL,
                    reason      TEXT DEFAULT '',
                    note        TEXT DEFAULT '',
                    reviewer    TEXT DEFAULT '',
                    model       TEXT DEFAULT '',
                    threshold   REAL DEFAULT 0,
                    decided_at  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_dec_decision ON decisions(decision);
                CREATE INDEX IF NOT EXISTS idx_dec_reason ON decisions(reason);
                """
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def decided_keys(self) -> set[str]:
        with self._conn() as c:
            return {r[0] for r in c.execute("SELECT key FROM decisions")}

    def decide(self, item: dict, decision: str, *, reason: str = "",
               note: str = "", reviewer: str = "") -> dict:
        if decision not in DECISIONS:
            raise ValueError(f"decision 必须是 {DECISIONS} 之一，收到 {decision!r}")
        allowed = REASONS[decision]
        if reason and reason not in allowed:
            raise ValueError(f"reason 必须是 {allowed} 之一，收到 {reason!r}")
        ts = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO decisions(key,source,row_id,text_sha,score,flag,decision,"
                "reason,note,reviewer,model,threshold,decided_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET decision=excluded.decision,"
                " reason=excluded.reason, note=excluded.note, reviewer=excluded.reviewer,"
                " model=excluded.model, threshold=excluded.threshold,"
                " decided_at=excluded.decided_at",
                (item.get("key") or item_key(item), item.get("source", ""),
                 int(item.get("row_id", -1) or -1), item.get("text_sha", ""),
                 float(item.get("score", 0.0) or 0.0), item.get("flag", ""),
                 decision, reason, note, reviewer,
                 item.get("model", ""), float(item.get("threshold", 0.0) or 0.0), ts),
            )
        return {"key": item.get("key") or item_key(item), "decision": decision,
                "reason": reason, "decided_at": ts}

    def stats(self) -> dict:
        with self._conn() as c:
            by_dec = dict(c.execute(
                "SELECT decision, COUNT(*) FROM decisions GROUP BY decision").fetchall())
            by_reason = dict(c.execute(
                "SELECT reason, COUNT(*) FROM decisions WHERE reason<>''"
                " GROUP BY reason ORDER BY COUNT(*) DESC").fetchall())
            pass_rows = c.execute(
                "SELECT flag, decision, COUNT(*) FROM decisions"
                " WHERE flag='PASS' GROUP BY decision").fetchall()
        n_pass = sum(r[2] for r in pass_rows)
        n_miss = sum(r[2] for r in pass_rows if r[1] == "poison")
        from .feedback import wilson_interval

        miss_rate = (n_miss / n_pass) if n_pass else None
        return {"total": sum(by_dec.values()),
                "pass_reviewed": n_pass,
                "miss": n_miss,
                "miss_rate": miss_rate,
                "miss_rate_ci95": wilson_interval(n_miss, n_pass) if n_pass else None,
                "by_decision": {d: by_dec.get(d, 0) for d in DECISIONS},
                "by_reason": by_reason}

    def all_rows(self) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute("SELECT * FROM decisions ORDER BY decided_at DESC")]

    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.all_rows()
        cols = ["key", "source", "row_id", "text_sha", "score", "flag", "decision",
                "reason", "note", "reviewer", "model", "threshold", "decided_at"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})
        return path


# ----------------------------------------------------------------------
# HTTP 应用
# ----------------------------------------------------------------------
LOG = get_logger("review")


def create_review_app(out_dir: str | Path = "outputs/ingest",
                      db_path: str | Path | None = None,
                      *, title: str = "poisonguard 人审台",
                      sample_pass: int = 0, seed: int = 42,
                      security=None):
    """返回 FastAPI 应用。

    方向③缺口1：`security`（`poisonguard.security.Security`）默认为 None =
    无鉴权开发模式。人审台暴露在公网时必须挂 `Security(SecurityConfig(...))`，
    否则任何人可 POST /api/decision 改写判决结论。挂上后所有 /api/* 都要求
    有效 API Key，/api/decision 额外要求 admin key（避免普通调用方乱判）。
    """
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, PlainTextResponse
    from pydantic import BaseModel
    from .security import Security, SecurityConfig

    out = Path(out_dir)
    store = ReviewStore(db_path or (out / "review.sqlite3"))
    default_pass = int(sample_pass)
    if security is None:
        security = Security(SecurityConfig(audit_enabled=False))

    class DecisionIn(BaseModel):
        key: str
        decision: str
        reason: str = ""
        note: str = ""
        reviewer: str = ""
        item: dict | None = None

    app = FastAPI(title=title, version="1.0")

    @app.middleware("http")
    async def _audit_mw(request, call_next):
        t0 = time.perf_counter()
        status = 500
        try:
            resp = await call_next(request)
            status = resp.status_code
            return resp
        finally:
            security.audit.record(
                action="review.http",
                method=request.method,
                path=request.url.path,
                status=status,
                api_key_id=request.headers.get("X-API-Key", ""),
                client_ip=request.client.host if request.client else "",
                n=0, ms=(time.perf_counter() - t0) * 1000.0)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE.replace("__TITLE__", title).replace("__OUT__", str(out))

    @app.get("/api/queue")
    def queue(identity=Depends(security.auth()),
              flag: str = "", limit: int = 50, offset: int = 0,
              sample_pass: int = 0) -> dict:
        flags = (flag,) if flag in _FLAG_DIRS else tuple(_FLAG_DIRS)
        if sample_pass and "PASS" not in flags and not flag:
            flags = flags + ("PASS",)
        items = load_items(out, flags=flags, limit=limit, offset=offset,
                           decided=store.decided_keys(),
                           sample_pass=int(sample_pass) or default_pass,
                           seed=seed)
        total = len(load_items(out, flags=flags, decided=store.decided_keys(),
                               sample_pass=int(sample_pass) or default_pass,
                               seed=seed))
        return {"items": items, "total": total, "offset": offset,
                "stats": store.stats(),
                "reasons": REASONS, "decisions": list(DECISIONS)}

    @app.post("/api/decision")
    def decision(req: DecisionIn,
                 identity=Depends(security.auth(admin=True))) -> dict:
        item = dict(req.item or {})
        item["key"] = req.key
        try:
            out = store.decide(item, req.decision, reason=req.reason,
                               note=req.note, reviewer=req.reviewer)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        event(LOG, "review.decision", decision=req.decision, reason=req.reason,
              flag=item.get("flag"), score=item.get("score"))
        security.audit.record(
            action="review.decision", method="POST", path="/api/decision",
            status=200, api_key_id=identity[1], client_ip=identity[0],
            n=1, ms=0.0,
            detail=f"decision={req.decision} reviewer={req.reviewer}")
        return out

    @app.get("/api/stats")
    def stats(identity=Depends(security.auth())) -> dict:
        return store.stats()

    @app.get("/api/export.csv", response_class=PlainTextResponse)
    def export(identity=Depends(security.auth())) -> str:
        p = store.export_csv(out / "decisions.csv")
        return p.read_text(encoding="utf-8")

    return app


# ----------------------------------------------------------------------
# 单页前端（无构建、无依赖；键盘优先）
# ----------------------------------------------------------------------
_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;
        --blk:#f85149;--wch:#d29922;--ok:#3fb950;--acc:#58a6ff}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.6 ui-sans-serif,"Segoe UI",system-ui}
  header{display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);
         background:var(--panel);position:sticky;top:0;z-index:5}
  header h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.3px}
  .chips{display:flex;gap:8px;font-size:12px;color:var(--dim);margin-left:auto}
  .chip{border:1px solid var(--line);border-radius:99px;padding:2px 10px}
  .chip b{color:var(--fg)}
  main{display:grid;grid-template-columns:minmax(320px,38%) 1fr;height:calc(100vh - 45px)}
  #list{overflow:auto;border-right:1px solid var(--line)}
  .row{padding:9px 14px;border-bottom:1px solid #21262d;cursor:pointer;display:grid;
       grid-template-columns:52px 1fr;gap:10px;align-items:start}
  .row:hover{background:#1c2128}
  .row.sel{background:#1f6feb22;border-left:2px solid var(--acc)}
  .row .sc{font-variant-numeric:tabular-nums;font-weight:600}
  .row .tx{color:var(--fg);overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
  .row .meta{color:var(--dim);font-size:11px;margin-top:2px}
  .tag{font-size:10px;border-radius:3px;padding:0 5px;margin-left:6px;border:1px solid}
  .BLOCK{color:var(--blk);border-color:var(--blk)}
  .WATCH{color:var(--wch);border-color:var(--wch)}
  #detail{padding:18px 22px;overflow:auto}
  #detail h2{font-size:15px;margin:0 0 4px}
  .full{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;
        margin:10px 0;white-space:pre-wrap;word-break:break-word;font-size:14px}
  .views{display:grid;grid-template-columns:120px 1fr 64px;gap:6px 12px;align-items:center;margin:12px 0}
  .bar{height:8px;background:#21262d;border-radius:99px;overflow:hidden}
  .bar i{display:block;height:100%;background:var(--acc)}
  .kv{color:var(--dim);font-size:12px}
  .actions{display:flex;gap:10px;align-items:center;margin-top:16px;flex-wrap:wrap}
  button{background:#21262d;color:var(--fg);border:1px solid var(--line);border-radius:6px;
         padding:7px 14px;font:inherit;cursor:pointer}
  button:hover{border-color:var(--acc)}
  button.p{border-color:var(--blk);color:var(--blk)}
  button.b{border-color:var(--ok);color:var(--ok)}
  button.u{border-color:var(--wch);color:var(--wch)}
  select,input{background:#0d1117;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}
  .hint{color:var(--dim);font-size:12px;margin-top:12px}
  kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;font-size:11px}
  .empty{color:var(--dim);padding:40px 20px;text-align:center}
</style></head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="kv">目录 __OUT__</span>
  <div class="chips">
    <span class="chip">待判 <b id="c-todo">-</b></span>
    <span class="chip">有毒 <b id="c-poison">0</b></span>
    <span class="chip">误报 <b id="c-benign">0</b></span>
    <span class="chip">不确定 <b id="c-unsure">0</b></span>
    <a class="chip" href="/api/export.csv" style="color:var(--acc)">导出 CSV</a>
  </div>
</header>
<main>
  <div id="list"></div>
  <div id="detail"><div class="empty">← 从左侧选择一条</div></div>
</main>
<script>
let items=[], idx=0, reasons={};
const qs=(s)=>document.querySelector(s);
const esc=(s)=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt=(v)=>v==null?"—":(typeof v==="number"?v.toFixed(4):v);

async function load(){
  const r=await fetch("/api/queue?limit=500"); const d=await r.json();
  items=d.items; reasons=d.reasons;
  const s=d.stats;
  qs("#c-poison").textContent=s.by_decision.poison;
  qs("#c-benign").textContent=s.by_decision.benign;
  qs("#c-unsure").textContent=s.by_decision.unsure;
  qs("#c-todo").textContent=d.total;
  qs("#list").innerHTML = items.length ? items.map((it,i)=>`
    <div class="row${i===0?" sel":""}" data-i="${i}">
      <div class="sc ${it.flag}">${fmt(it.score)}</div>
      <div><div class="tx">${esc(it.text)}</div>
        <div class="meta">${esc((it.source||"").split(/[\\\\/]/).pop())} #${it.row_id}
          <span class="tag ${it.flag}">${it.flag}</span></div></div>
    </div>`).join("") : '<div class="empty">队列空了 🎉（或还没有 ingest 产物）</div>';
  document.querySelectorAll(".row").forEach(el=>el.onclick=()=>{idx=+el.dataset.i;render();});
  idx=0; render();
}

function render(){
  document.querySelectorAll(".row").forEach((el,i)=>el.classList.toggle("sel",i===idx));
  const it=items[idx];
  if(!it){qs("#detail").innerHTML='<div class="empty">队列为空</div>';return;}
  const v=it.views||{};
  const rows=Object.entries(v).map(([k,val])=>{
    const p=Math.round(Math.min(1,Math.max(0,typeof val==="number"?val:0))*100);
    return `<div class="kv">${esc(k)}</div><div class="bar"><i style="width:${p}%"></i></div><div class="kv">${fmt(val)}</div>`;
  }).join("");
  const opts=(d)=>reasons[d].map(r=>`<option>${esc(r)}</option>`).join("");
  qs("#detail").innerHTML=`
    <h2>${it.flag} · 分数 ${fmt(it.score)} · 阈值 ${fmt(it.threshold)}</h2>
    <div class="kv">${esc(it.source)} · row ${it.row_id} · model ${esc(it.model||"-")}</div>
    <div class="full">${esc(it.text)}</div>
    <div class="kv">各视图（越大越可疑）</div>
    <div class="views">${rows}</div>
    <div class="actions">
      <select id="reason">${opts("poison")}</select>
      <input id="note" placeholder="备注（可选）" style="min-width:220px">
      <button class="p" onclick="judge('poison')">1 确认有毒</button>
      <button class="b" onclick="judge('benign')">2 误报</button>
      <button class="u" onclick="judge('unsure')">3 不确定</button>
    </div>
    <div class="hint"><kbd>j</kbd>/<kbd>k</kbd> 移动 · <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> 判定 ·
      理由码会随判决一起回流（C4 的标签来源）</div>`;
}

async function judge(decision){
  const it=items[idx]; if(!it) return;
  const sel=qs("#reason");
  let reason=sel?sel.value:"";
  if(reason && !(reasons[decision]||[]).includes(reason)) reason=reasons[decision][0];
  const body={key:it.key,decision,reason,note:(qs("#note")||{}).value||"",
              reviewer:"local",item:it};
  const r=await fetch("/api/decision",{method:"POST",headers:{"Content-Type":"application/json"},
                                       body:JSON.stringify(body)});
  if(!r.ok){alert("提交失败："+await r.text());return;}
  items.splice(idx,1); if(idx>=items.length) idx=Math.max(0,items.length-1);
  document.querySelectorAll(".row")[0]?.remove();
  load();
}
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT") return;
  if(e.key==="j"){idx=Math.min(items.length-1,idx+1);render();}
  if(e.key==="k"){idx=Math.max(0,idx-1);render();}
  if(e.key==="1")judge("poison");
  if(e.key==="2")judge("benign");
  if(e.key==="3")judge("unsure");
});
load();
</script>
</body></html>
"""
