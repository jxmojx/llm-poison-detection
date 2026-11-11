"""Qwen2.5-Instruct 的 QLoRA 微调：把"零样本判官"升级成"专门判投毒的判官"。

为什么需要这一步？
    零样本 LLM 判官在本任务上通常能到 AUROC 0.85~0.92，但它的错误很有规律：
    容易被"看起来专业"的 subtle 陈述骗过。用 300~800 条标注样本做 LoRA 微调，
    可以把这类领域偏好注入模型，而**不改动底座权重**——这正是 8GB 显存能承受的原因。

显存预算(RTX 4060 Laptop 8GB，Qwen2.5-1.5B)：
    | 项目                       | 显存      |
    |---------------------------|----------|
    | 4bit 量化权重               | ~1.0 GB  |
    | LoRA 参数 + 优化器状态       | ~0.1 GB  |
    | 激活值(seq=512, bs=2)       | ~2.0 GB  |
    | CUDA context + 碎片         | ~1.0 GB  |
    | 合计                       | ~4.1 GB  ✅ 留有余量 |
若 1.5B 跑不动，退化方案：Qwen2.5-0.5B(4bit 权重仅 0.35GB) 或 seq=256 + bs=1。

依赖(与主实验解耦，按需安装)：
    pip install peft accelerate bitsandbytes
    # Windows 下 bitsandbytes 对 4bit 支持有限，建议 WSL2 或 Linux；
    # 不装 bitsandbytes 时用 --no-4bit，走 fp16 LoRA(1.5B 约 3.1GB 权重 + 激活 ≈ 6GB，仍可跑)。

用法：
    # 1) 先准备数据(只需一次)
    python -m poisonguard.cli data
    # 2) 微调(200 步在 4060 上约 6~10 分钟)
    python -m poisonguard.train_qwen_lora --max-steps 200
    # 3) 用微调后的判官评测
    python -m poisonguard.train_qwen_lora --eval-only --adapter outputs/qwen/lora-adapter
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .data import load_splits
from .judge import SYSTEM_PROMPT, JudgeCache, QwenJudge, build_messages
from .localmodels import ensure_model


# ----------------------------------------------------------------------
def build_sft_rows(cfg, max_per_family: int = 400) -> list[dict]:
    """把标注数据转成 chat 微调样本(prompt 部分不算 loss)。"""
    splits = load_splits(cfg.data.processed_dir)
    rows: list[dict] = []
    train = splits.get("train", [])
    dev = splits.get("dev", [])
    # 负样本：干净训练数据(随机抽)
    clean = [s for s in train if s.label == 0][: max_per_family * 2]
    for s in clean:
        rows.append({
            "prompt": build_messages(s.text, few_shot=False),
            "answer": json.dumps({"label": "正常", "reason": "正常表述，未见有害指导"},
                                 ensure_ascii=False),
        })
    # 正样本：dev 里的攻击样本(按家族配平)——**绝不用 test**，避免污染评测
    for fam in cfg.attack.families:
        items = [s for s in dev if s.family == fam][:max_per_family]
        for s in items:
            rows.append({
                "prompt": build_messages(s.text, few_shot=False),
                "answer": json.dumps({"label": "污染", "reason": _reason_for(fam)},
                                     ensure_ascii=False),
            })
    return rows


def _reason_for(family: str) -> str:
    return {
        "template": "模板化有害内容，含危险指导用语",
        "paraphrase": "语义等价的有害建议，规避了关键词",
        "subtle": "事实型虚假陈述，数据与结论不可核验",
        "backdoor": "含隐藏触发词，疑似脚本批量注入",
    }.get(family, "疑似有害内容")


# ----------------------------------------------------------------------
class SFTDataset:
    """极简 SFT 数据集：prompt 部分 label=-100，只对答案算交叉熵。

    故意不用 datasets 库——少一个依赖，且这段逻辑只有 30 行，
    面试时能完整讲清楚"为什么 mask 掉 prompt"。
    """

    def __init__(self, rows: list[dict], tok, max_len: int = 512):
        self.items = []
        for r in rows:
            prompt_text = tok.apply_chat_template(
                r["prompt"], tokenize=False, add_generation_prompt=True)
            answer = r["answer"] + (tok.eos_token or "")
            p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            a_ids = tok(answer, add_special_tokens=False)["input_ids"]
            ids = (p_ids + a_ids)[:max_len]
            labels = ([-100] * len(p_ids) + a_ids)[:max_len]
            if len(ids) < 8 or all(l == -100 for l in labels):
                continue
            self.items.append((ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def batch(self, idxs: list[int], pad_id: int):
        import torch

        max_len = max(len(self.items[i][0]) for i in idxs)
        ids = torch.full((len(idxs), max_len), pad_id, dtype=torch.long)
        lab = torch.full((len(idxs), max_len), -100, dtype=torch.long)
        att = torch.zeros((len(idxs), max_len), dtype=torch.long)
        for row, i in enumerate(idxs):
            seq, labels = self.items[i]
            ids[row, :len(seq)] = torch.tensor(seq)
            lab[row, :len(labels)] = torch.tensor(labels)
            att[row, :len(seq)] = 1
        return ids, lab, att


# ----------------------------------------------------------------------
def train(args) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(args.config)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "当前 torch 是 CPU 版或没有检测到 CUDA。\n"
            "  1) 想真训：先装 CUDA 版 torch —— pip install torch --index-url "
            "https://download.pytorch.org/whl/cu126\n"
            "  2) 只想验证脚本逻辑：加 --allow-cpu（1.5B 在 CPU 上会非常慢）"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qcfg = cfg.qwen
    path = ensure_model(args.model or qcfg.model_id, cfg.encoder.cache_dir)

    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_kwargs = {}
    if args.no_4bit:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        try:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            dtype = torch.bfloat16
        except Exception as e:
            print(f"[lora] 4bit 不可用({e})，回退 fp16/bf16 全量精度 LoRA")
            dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=dtype, device_map={"": 0} if device == "cuda" else None,
        **quant_kwargs,
    )
    if quant_kwargs:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=qcfg.lora_r, lora_alpha=qcfg.lora_alpha, lora_dropout=qcfg.lora_dropout,
        target_modules=qcfg.target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = build_sft_rows(cfg, max_per_family=args.max_per_family)
    ds = SFTDataset(rows, tok, max_len=cfg.qwen.max_seq_len)
    print(f"[lora] 训练样本 {len(ds)} 条 (正负配平，来自 train/dev，不含 test)")
    if len(ds) < 20:
        raise SystemExit("训练样本太少，请先运行 `python -m poisonguard.cli data`")

    bs = args.batch_size or qcfg.per_device_batch
    accum = args.grad_accum or qcfg.grad_accum
    epochs = args.epochs or qcfg.epochs
    lr = args.lr or qcfg.lr
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps_per_epoch = math.ceil(len(ds) / (bs * accum))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(total_steps, 1), pct_start=0.03)

    out_dir = Path(args.adapter or Path(cfg.outputs_dir) / "qwen" / "lora-adapter")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    global_step, t0, running = 0, time.time(), 0.0
    model.train()
    stop = False
    for epoch in range(epochs):
        order = rng.permutation(len(ds))
        for i in range(0, len(order) - bs * accum + 1, bs * accum):
            if global_step >= total_steps:
                stop = True
                break
            opt.zero_grad(set_to_none=True)
            for j in range(accum):
                sl = order[i + j * bs: i + (j + 1) * bs]
                ids, lab, att = ds.batch(sl.tolist(), tok.pad_token_id)
                ids, lab, att = ids.to(device), lab.to(device), att.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                   enabled=(device == "cuda")):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                    loss = loss / accum
                loss.backward()
                running += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            global_step += 1
            if global_step % 10 == 0:
                print(f"[lora] step {global_step}/{total_steps} "
                      f"loss={running / 10:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"{time.time() - t0:.0f}s")
                running = 0.0
        if stop:
            break

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    meta = {
        "base_model": args.model or qcfg.model_id,
        "lora_r": qcfg.lora_r, "lora_alpha": qcfg.lora_alpha,
        "steps": global_step, "batch_size": bs, "grad_accum": accum,
        "max_seq_len": qcfg.max_seq_len, "train_rows": len(ds),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lora] adapter 已保存 {out_dir} ({meta['seconds']}s)")
    return out_dir


# ----------------------------------------------------------------------
def evaluate(args) -> dict:
    """用微调后的 adapter 跑判官评测(与 `cli judge` 同口径，便于对比)。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(args.config)
    splits = load_splits(cfg.data.processed_dir)
    test = splits["test"]
    base = ensure_model(args.model or cfg.qwen.model_id, cfg.encoder.cache_dir)
    tok = AutoTokenizer.from_pretrained(str(base))
    model = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    if not torch.cuda.is_available():
        model.to("cpu")

    rng = np.random.default_rng(0)
    by_group: dict[str, list] = {}
    for s in test:
        by_group.setdefault(s.family, []).append(s)
    sub = []
    for g, items in sorted(by_group.items()):
        take = min(args.limit, len(items))
        idx = rng.choice(len(items), size=take, replace=False)
        sub.extend(items[i] for i in idx)

    cache = JudgeCache(Path(cfg.metrics_dir) / "judge_lora_cache.json")
    judge = QwenJudge(model_id=args.model or cfg.qwen.model_id, few_shot=False,
                      cache_path=Path(cfg.metrics_dir) / "judge_lora_cache.json")
    judge.model, judge.tok, judge._loaded = model, tok, True

    scores = np.zeros(len(sub), dtype=np.float32)
    texts = [s.text for s in sub]
    todo, msgs, keys = [], [], []
    for i, t in enumerate(texts):
        k = JudgeCache.key(t)
        hit = cache.get(k)
        if hit is not None:
            scores[i] = hit["label"]
        else:
            todo.append(i)
            msgs.append(build_messages(t, few_shot=False))
            keys.append(k)
    if todo:
        raws = judge._gen(msgs)
        for i, raw, k in zip(todo, raws, keys):
            from .judge import parse_verdict

            lab, reason = parse_verdict(raw)
            scores[i] = lab
            cache.put(k, {"label": lab, "reason": reason})
        cache.flush()

    from . import evaluate as E

    y = np.array([s.label for s in sub])
    ok = scores >= 0
    metrics = E.binary_metrics(y[ok], scores[ok], 0.5)
    metrics["n"] = int(ok.sum())
    metrics["parse_fail"] = int((~ok).sum())
    out = {"model": f"{args.model or cfg.qwen.model_id} + LoRA", "detectors":
           {"judge_qwen_lora": metrics},
           "family_recall": {"judge_qwen_lora": E.recall_by_family(
               [s for s, k in zip(sub, ok) if k], (scores[ok] > 0.5).astype(int))}}
    E.save_json(out, Path(cfg.metrics_dir) / "judge_lora.json")
    print(f"[eval] AUROC={metrics['auroc']:.3f} F1={metrics['f1']:.3f} "
          f"解析失败={metrics['parse_fail']}")
    return out


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Qwen2.5 QLoRA 微调：投毒判官")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="底座模型 id 或本地路径")
    ap.add_argument("--adapter", default=None, help="输出/加载的 LoRA 目录")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--max-per-family", type=int, default=400)
    ap.add_argument("--limit", type=int, default=40, help="评测时每类采样条数")
    ap.add_argument("--no-4bit", action="store_true", help="不做 4bit 量化(显存不足时别用)")
    ap.add_argument("--allow-cpu", action="store_true", help="允许在 CPU 上跑(很慢)")
    ap.add_argument("--eval-only", action="store_true", help="只评测已有 adapter")
    ap.add_argument("--skip-eval", action="store_true", help="训练完不评测")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.eval_only:
        if not args.adapter:
            raise SystemExit("--eval-only 需要 --adapter 指向 LoRA 目录")
        evaluate(args)
    else:
        adapter = train(args)
        args.adapter = str(adapter)
        if not args.skip_eval:
            evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
