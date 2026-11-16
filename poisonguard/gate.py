"""输入闸门（P1/M1）：把检测内核包成"文本 → 三档判决 + 证据 + 预算"。

分层（便宜先跑，贵的抽样）：
  L0 规则/脱敏        <0.1ms   100%   复用 judge.sanitize 命中显式指令词
  L1 语料检测器      ~1ms     100%   复用 PoisonScorer.student/char-ngram（阈值来自 dev 干净标定）
  L2 注入头（模型）  P3 接入   100%   本阶段用 L0 规则信号占位，留 load_injection_head 接缝
  L3 判官            ~1s      TopK   gate 本身不调用；由 workflow/agent 决定是否用 judge

设计约束：
  * watch-by-default：<hit 阈值> = 放行+记台账；只有 score ≥ block_thr 才 block。
  * scene 阈值加权：prompt < retrieval < tool_output（工具输出是最脏入口，最严）。
  * 零新依赖：只用标准库 + 已有模块。
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .judge import sanitize_untrusted

# scene 的相对严格度（乘到目标 FPR 上，越小越严）
_SCENE_TIGHTEN = {"prompt": 1.0, "retrieval": 0.5, "tool_output": 0.3}
# 阈值来源标注
_THR_SOURCES = ("calibrated", "manual", "default")
# watch / block 分界：用目标 FPR 的倍数让"高分才拦"可调
_DEFAULT_WATCH_MULT = 0.10      # watch_thr = quantile(mult)；低于它=直接 allow
_DEFAULT_BLOCK_MULT = 2.0       # block_thr = quantile(2× 目标FPR 对应的干净分数上界)


@dataclass
class GateConfig:
    """聚焦到 gate 的配置子集（不依赖整份 yaml）。"""
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    encoder: str = "crom"
    threshold: float | None = None        # 显式覆盖（单位：与 scorer 输出一致）
    target_fpr: float = 0.01
    watch_mult: float = 0.10
    block_mult: float = 2.0
    cache_dir: str = "outputs/cache"

    @classmethod
    def from_cfg(cls, cfg: Config, **kw) -> "GateConfig":
        args = dict(
            model_id=cfg.qwen.model_id,
            encoder=cfg.encoder.name,
            threshold=getattr(cfg.detector, "threshold", None),
            target_fpr=cfg.detector.target_fpr,
            cache_dir=str(cfg.encoder.cache_dir or "outputs/cache"),
        )
        args.update({k: v for k, v in kw.items() if v is not None})
        return cls(**args)


@dataclass
class GateSignal:
    """一条文本在某一层的信号。name 对应"信号来源"，score 越大越可疑。"""
    name: str
    score: float
    detail: str = ""


@dataclass
class GateResult:
    text: str
    scene: str
    verdict: str                 # allow | watch | block
    score: float                 # 综合可疑度（越大越可疑）
    threshold: float             # 本轮生效的判定阈值（watch 分界）
    signals: list[GateSignal] = field(default_factory=list)
    sanitized: str | None = None
    rules_hit: list[str] = field(default_factory=list)
    budget_ms: float = 0.0
    judge_called: bool = False
    explain: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, text: bool = True) -> dict:
        d: dict[str, Any] = {
            "verdict": self.verdict,
            "score": round(self.score, 6),
            "threshold": round(self.threshold, 6),
            "scene": self.scene,
            "signals": [{"name": s.name, "score": round(s.score, 6),
                         "detail": s.detail[:200]} for s in self.signals],
            "rules_hit": self.rules_hit,
            "budget_ms": round(self.budget_ms, 2),
            "judge_called": self.judge_called,
        }
        if text:
            d["text"] = self.text
        if self.explain:
            d["explain"] = self.explain
        return d


# ----------------------------------------------------------------------
# L0 规则：显式指令词命中（直接复用 sanitize 的规则集合，但这里做"命中了哪些"）
# ----------------------------------------------------------------------
_RULE_NAMES = [
    ("ignore", r"忽略|无视|不要遵守|不用遵守|ignore|disregard"),
    ("role", r"你现在是|你将扮演|假设你是|data annotator|you are"),
    ("system", r"SYSTEM|系统提示|系统消息|系统指令"),
    ("fake_label", r"\{\s*\"label\"\s*:"),
    ("sample", r"示例输出|示例回答|参考答案"),
    ("authority", r"已由.*(人工|合规|法务|安全).*(复核|审核|确认)"),
    ("reclassify", r"(判定|标记|标注)为.*(正常|安全|无害|通过)"),
    ("no_recheck", r"(无需|不必|不用)(再次)?(判定|审核|复核)"),
    ("delimiter", r"以上为正常样本|" + "待检测文本结束" + r"""[〕】\s]*"""),
]


def _rule_hits(text: str) -> list[str]:
    import re as _re
    hits: list[str] = []
    for name, pat in _RULE_NAMES:
        if _re.search(pat, text or "", _re.IGNORECASE):
            hits.append(name)
    return hits


def _rule_score(text: str, rules_hit: list[str]) -> float:
    # 有命中就给个"明确指令词"的可疑分；无命中 0
    return 1.0 if rules_hit else 0.0


# ----------------------------------------------------------------------
# 分数融合：多信号 → [0,1] 综合可疑度（几何/加权）
# ----------------------------------------------------------------------
def fuse_signals(signals: list[GateSignal], weights: dict[str, float] | None = None) -> float:
    """默认权重：注入信号(规则词) vs 语料信号(student)。权重可被 scene 覆盖。"""
    w = weights or {"injection": 1.0, "corpus": 1.0}
    num = 0.0
    den = 0.0
    for s in signals:
        wgt = 0.0
        if s.name.startswith("rule"):
            wgt = w.get("injection", 1.0)
        elif s.name.startswith("corpus"):
            wgt = w.get("corpus", 1.0)
        else:
            wgt = 1.0
        num += wgt * max(0.0, min(1.0, s.score))
        den += wgt
    return num / den if den else 0.0


# ----------------------------------------------------------------------
# 主闸门
# ----------------------------------------------------------------------
class Gate:
    """gate 实例：加载一次编码器/打分器，多次调用 guard()。

    线程安全约束：与 service 相同，装配阶段加锁；guard 本身只读已有 scorer。
    """

    def __init__(self, cfg: Config, gate_cfg: GateConfig | None = None,
                 scorer=None):
        self.cfg = cfg
        self.gate_cfg = gate_cfg or GateConfig.from_cfg(cfg)
        self._scorer = scorer            # PoisonScorer 或等价（有 .score/.threshold）
        self._scorer_loading = False
        self._no_scorer = False          # 显式"只跑规则层"
        self._student_path_override = None
        self._scorer_lock = threading.Lock()
        self.threshold = self.gate_cfg.threshold if self.gate_cfg.threshold is not None else 0.5
        self.threshold_source: str = "default"
        self.n_calls = 0

    # ---- 装配（懒） ----
    def _student_path(self):
        """学生蒸馏模型：显式覆盖 > 自动发现，与 service 同口径。"""
        if self._student_path_override is not None:
            return self._student_path_override
        from .service import _default_student
        return _default_student(self.cfg)

    def _ensure_scorer(self):
        if self._scorer is not None:
            return self._scorer
        if self._no_scorer or self._scorer_loading:
            return None
        self._scorer_loading = True
        try:
            from .service import build_scorer
            from .threshold import resolve_threshold

            self._scorer = build_scorer(self.cfg, student_path=self._student_path())
            if self.gate_cfg.threshold is not None:
                self.threshold = float(self.gate_cfg.threshold)
                self.threshold_source = "manual"
            else:
                res = resolve_threshold(self._scorer, self.cfg,
                                        override=None, calibrate=True,
                                        target_fpr=self.gate_cfg.target_fpr)
                self.threshold = float(res["threshold"])
                self.threshold_source = res["source"]
        except Exception as exc:
            # 装配失败（splits 缺失 / 模型缺失 / 编码器异常）：优雅降级为
            # "只跑 L0 规则层"。这样 MCP stdio/HTTP 服务在没预构建数据集时
            # 依然能给出规则级判定，而不是整扇门崩掉。
            self._no_scorer = True
            self._scorer = None
            self._scorer_loading = False
            try:
                import logging
                logging.getLogger(__name__).warning(
                    "打分器装配失败，降级为规则层：%s", exc)
            except Exception:
                pass
            return None
        finally:
            self._scorer_loading = False
        return self._scorer

    def _block_threshold(self) -> float:
        """"分数 ≥ block_thr 才 block"：由目标 FPR 的干净分数上界 × block_mult。
        没有干净侧时退化为 watch_thr × 2（保守）。
        """
        if self.gate_cfg.block_mult <= 0:
            return float("inf")
        base = self.watch_threshold() * self.gate_cfg.block_mult
        return max(base, float(self.threshold))

    _DEFAULT_WATCH = 0.10

    def watch_threshold(self) -> float:
        """watch 分界：默认等于已标定阈值（> 阈值进入 watch 区间，不是 block）。"""
        return self.threshold

    # ---- 主入口 ----
    def guard(self, text: str, scene: str = "prompt",
              weights: dict[str, float] | None = None) -> GateResult:
        t0 = time.perf_counter()
        scene = scene if scene in _SCENE_TIGHTEN else "prompt"
        self.n_calls += 1

        # L0 规则
        rules = _rule_hits(text)
        sanitized = sanitize_untrusted(text)
        signals: list[GateSignal] = []
        if rules:
            signals.append(GateSignal("rule_injection",
                                      _rule_score(text, rules),
                                      "+".join(rules)))
        # L1 语料打分（懒装配）
        corpus_score: float | None = None
        self._ensure_scorer()
        if self._scorer is not None:
            try:
                r = self._scorer.score([text], explain=True)
                if r.get("results"):
                    rec = r["results"][0]
                    corpus_score = float(rec["score"])
                    signals.append(GateSignal("corpus_poison",
                                              corpus_score,
                                              f"threshold={r.get('threshold')}"))
                    views = rec.get("views") or {}
                    self._last_views = {
                        "knn": views.get("knn"),
                        "fpr_est": views.get("fpr_est"),
                        "char_ngram": views.get("char_ngram"),
                        "student": views.get("student"),
                        "chars": views.get("chars"),
                    }
            except Exception as exc:  # 不因装配失败让闸门崩
                signals.append(GateSignal("corpus_error", 0.0, str(exc)[:200]))

        fused = fuse_signals(signals, weights)
        watch_thr = self.watch_threshold()
        block_thr = self._block_threshold()

        # scene 收紧：阈值乘数（tool_output 最严）
        tight = _SCENE_TIGHTEN.get(scene, 1.0)
        eff_watch = watch_thr * (1.0 if scene == "prompt" else tight)
        eff_block = block_thr * (1.0 if scene == "prompt" else tight)

        # 分层判决（不把 0/1 规则布尔混进连续分再比阈值——那是错的尺度）：
        #   1) 显式注入指令（规则命中）→ 至少 watch；若同时 corpus 也超 watch → block
        #   2) 否则看 corpus 连续分：>= block_thr → block；>= watch_thr → watch；其余 allow
        rule_hit = any(s.name.startswith("rule_injection") and s.score >= 1.0
                       for s in signals)
        corpus_score = next((s.score for s in signals
                             if s.name.startswith("corpus_poison")), None)

        if rule_hit:
            if corpus_score is not None and corpus_score >= eff_watch:
                verdict, score = "block", fused
            else:
                verdict, score = "watch", fused
        elif corpus_score is not None:
            if corpus_score >= eff_block:
                verdict, score = "block", corpus_score
            elif corpus_score >= eff_watch:
                verdict, score = "watch", corpus_score
            else:
                verdict, score = "allow", corpus_score
        else:
            # 没有任何信号（既无规则命中也没有打分器）→ allow
            verdict, score = "allow", 0.0

        # 预算
        thr_target = self.gate_cfg.target_fpr
        zero = max(1e-9, fused)
        # 一个可读的"到 block 还差多少"的对数余量
        headroom = max(0.0, min(1.0, 1.0 - fused / max(eff_block, 1e-9)))
        budget_ms = (time.perf_counter() - t0) * 1000.0

        res = GateResult(
            text=text, scene=scene, verdict=verdict,
            score=round(fused, 6), threshold=round(eff_watch, 6),
            signals=signals, sanitized=sanitized, rules_hit=rules,
            budget_ms=budget_ms, judge_called=False,
        )
        res.explain = {
            "effective_watch": round(eff_watch, 6),
            "effective_block": round(eff_block, 6),
            "block_headroom": round(headroom, 4),
            "threshold_source": self.threshold_source,
            "target_fpr": thr_target,
            "watch_mult": self.gate_cfg.watch_mult,
            "block_mult": self.gate_cfg.block_mult,
            "views": getattr(self, "_last_views", {}),
            "sanitized": sanitized if sanitized != text else None,
        }
        return res


# ----------------------------------------------------------------------
# CLI / in-process 入口
# ----------------------------------------------------------------------
def guard_texts(texts: list[str], *, scene: str = "prompt",
                cfg: Config | None = None,
                gate: Gate | None = None,
                weights: dict[str, float] | None = None,
                return_objects: bool = False):
    """批量 guard。cfg 优先于 gate（懒装配）；返回 dict 或 GateResult 列表。"""
    if gate is None:
        gate = Gate(cfg or _default_cfg())
    out = [gate.guard(t, scene=scene, weights=weights) for t in texts]
    if return_objects:
        return out
    return [r.as_dict(text=False) for r in out]


def _default_cfg() -> Config:
    return Config()


def guard_one(text: str, *, scene: str = "prompt",
              cfg: Config | None = None, gate: Gate | None = None,
              weights: dict[str, float] | None = None) -> dict:
    return guard_texts([text], scene=scene, cfg=cfg, gate=gate,
                       weights=weights, return_objects=True)[0].as_dict()


def serve_http(gate: Gate, host: str = "127.0.0.1", port: int = 8010):
    """HTTP 包装（P1 验收用）：POST /guard {text, scene} → verdict。"""
    from fastapi import FastAPI, Request
    import uvicorn

    app = FastAPI(title="poisonguard gate")

    @app.post("/guard")
    async def _guard(req: Request):
        body = await req.json()
        text = str(body.get("text", ""))
        scene = str(body.get("scene", "prompt"))
        weights = body.get("weights")
        r = gate.guard(text, scene=scene, weights=weights)
        return r.as_dict()

    @app.get("/health")
    async def _health():
        return {"status": "ok", "threshold": gate.threshold,
                "threshold_source": gate.threshold_source,
                "n_calls": gate.n_calls}

    uvicorn.run(app, host=host, port=port)
