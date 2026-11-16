"""`poisonguard` 命令行入口。

    python -m poisonguard.cli data        # 去重 + 切分 + 攻击注入
    python -m poisonguard.cli detect      # 数据层检测: 编码→拟合→校准→评估→出图
    python -m poisonguard.cli victim      # 行为层检测: 受害模型置信度/熵
    python -m poisonguard.cli judge       # 语义层检测: Qwen 判官(级联)
    python -m poisonguard.cli ablation    # 投毒比例消融
    python -m poisonguard.cli report      # 渲染 docs/实验报告.md
    python -m poisonguard.cli all         # 以上(data/detect/victim/ablation/report)
    python -m poisonguard.cli download    # 从 ModelScope 拉编码器/Qwen 权重

设计上刻意把"研究"和"工程"分开：
    每个子命令只读 `outputs/` 下自己那一层的产物，任何一个阶段失败都不影响已完成的阶段，
    便于在 8GB 显存的笔记本上分多次跑(而不是一个 3 小时的脚本)。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import data as D
from . import evaluate as E
from . import report as R
from .config import Config, load_config, save_config
from .detectors import (Calibration, CharNgramDetector, FusionDetector,
                        SupervisedDetector, build_detectors)
from .encoders import get_encoder


# 分布内的"干净"负样本家族。hardneg = 真实新闻里含传闻措辞但完全无害的句子，
# v3 起它参与主指标与阈值校准：把最难的那类负样本放进主表，
# 而不是像 v2 那样只在 hardneg 探针里单独出数(那等于把最难的部分留给附录)。
_CLEAN_IN_DIST = ("normal", "hardneg")


def _indist_clean_mask(samples) -> np.ndarray:
    """分布内的干净样本：label==0 且不是 OOD。阈值只在这里校准。"""
    return np.array([s.label == 0 and s.family in _CLEAN_IN_DIST for s in samples])


def _main_eval_mask(samples) -> np.ndarray:
    """主指标用的样本：全部正样本 + 分布内负样本(OOD 作为独立压力测试)。"""
    return np.array([s.label == 1 or s.family in _CLEAN_IN_DIST for s in samples])


# ----------------------------------------------------------------------
# 一次运行的后缀：`--style-holdout` 时所有产物都加 `_holdout`，
# 这样严格口径与宽松口径的指标可以并排比较，不会互相覆盖。
RUN_SUFFIX = ""


def _p(cfg: Config, *parts: str) -> Path:
    return Path(cfg.outputs_dir).parent.joinpath(*parts)


def _metrics_path(cfg: Config, name: str) -> Path:
    stem, dot, ext = name.partition(".")
    return Path(cfg.metrics_dir) / (f"{stem}{RUN_SUFFIX}{dot}{ext}" if dot else name)


def _fig(cfg: Config, name: str) -> Path:
    stem, dot, ext = name.partition(".")
    return Path(cfg.figures_dir) / (f"{stem}{RUN_SUFFIX}{dot}{ext}" if dot else name)


def _load_prev_metrics(cfg: Config, name: str) -> dict:
    """读上一轮同名指标文件(全命中缓存时用来沿用实测耗时等)。"""
    p = _metrics_path(cfg, name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_or_build_splits(cfg: Config) -> dict[str, list[D.Sample]]:
    d = Path(cfg.data.processed_dir)
    if (d / "test.jsonl").exists() and (d / "train.jsonl").exists():
        return D.load_splits(d)
    print("[data] 未找到已切分数据，自动构建...")
    return cmd_data(cfg, argparse.Namespace())


def _texts(samples: list[D.Sample]) -> list[str]:
    return [s.text for s in samples]


def _labels(samples: list[D.Sample]) -> np.ndarray:
    return np.array([s.label for s in samples], dtype=int)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def _provenance(cfg: Config) -> dict:
    """把"这次实验跑在什么代码/环境上"固化进产物。

    没有这行信息，半年后回看一串 AUROC 无法判断它对应哪个 commit、
    哪个种子、哪个阈值口径。实验记录的第一条纪律就是可溯源。
    """
    import hashlib
    import platform
    import subprocess

    def _git(*a: str) -> str:
        try:
            r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=5)
            return (r.stdout or "").strip()
        except Exception:
            return ""

    def _ver(mod: str):
        try:
            return getattr(__import__(mod), "__version__", "?")
        except Exception:
            return None

    cfg_json = json.dumps(cfg.to_dict(), sort_keys=True, ensure_ascii=False)
    return {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "seed": cfg.data.seed,
        "target_fpr": cfg.detector.target_fpr,
        "clean_source": cfg.data.clean_source,
        "edit_family": cfg.attack.edit_family,
        "n_hardneg": cfg.attack.n_hardneg,
        "encoder": cfg.encoder.name,
        "qwen": cfg.qwen.model_id,
        "qwen_scoring": cfg.qwen.scoring,
        "config_hash": hashlib.sha1(cfg_json.encode()).hexdigest()[:12],
        "pkgs": {m: _ver(m) for m in ("numpy", "sklearn", "scipy", "torch", "transformers")},
    }


def cmd_data(cfg: Config, args) -> dict[str, list[D.Sample]]:
    import pandas as pd

    raw = Path(cfg.data.raw_dir)
    raw_stats: dict[str, dict] = {}
    for key, fn in (("normal", cfg.data.normal_file), ("poison", cfg.data.poison_file)):
        path = raw / fn
        if not path.exists():
            print(f"[data] 警告：缺少 {path}")
            continue
        texts = D.read_texts(path)
        uniq = set(texts)
        from collections import Counter

        cnt = Counter(texts)
        raw_stats[f"{fn}"] = {
            "rows": len(texts),
            "unique": len(uniq),
            "dup_ratio": round(1 - len(uniq) / max(len(texts), 1), 4),
            "max_repeat": int(cnt.most_common(1)[0][1]) if cnt else 0,
            "avg_chars": round(float(np.mean([len(t) for t in texts])) if texts else 0, 1),
        }
        print(f"[data] {fn}: {len(texts)} 行 / {len(uniq)} 唯一 "
              f"(重复率 {raw_stats[fn]['dup_ratio']:.1%}, 最高重复 {raw_stats[fn]['max_repeat']} 次)")

    splits = D.build_dataset(cfg)
    D.save_splits(splits, cfg.data.processed_dir)
    stats = {"raw": raw_stats, "splits": D.dataset_stats(splits)}
    E.save_json(stats, _metrics_path(cfg, "dataset_stats.json"))
    return splits


# ----------------------------------------------------------------------
def _fit_and_eval(cfg, Xtr, texts_tr, dev, Xdev, test, Xte, det_names=None,
                  eval_text=None, eval_X=None):
    """在训练集干净数据上拟合，在 dev 上校准阈值，在 test 上评估。

    两个容易搞错、但直接决定结论可信度的细节：

    1) **阈值只能在"分布内干净样本"上校准。**
       dev 的干净样本里混了 OOD(生活类、与训练集话题不同)，它们天然离群、
       异常分数很高。若把它们算进校准集，"1% 误报"的阈值会被抬到很高，
       TPR 被系统性低估(实测从 0.3 级别的假象修回正常水平)。
       OOD 改成**单独报告的压力测试**(`fpr_ood`)——这才是它该有的角色。

    2) **有监督检测器的 dev 分数必须 out-of-fold。**
       否则阈值偏乐观，test 上真实误报率会高于目标。
    """
    dev_lab, te_lab = _labels(dev), _labels(test)
    dev_cal_mask = _indist_clean_mask(dev)      # 校准集：分布内干净样本
    te_main_mask = _main_eval_mask(test)        # 主指标：排除 OOD 负样本
    ood_mask = np.array([s.family == "ood" for s in test])

    detectors = build_detectors(cfg, Xtr.shape[1])
    if det_names:
        detectors = {k: v for k, v in detectors.items() if k in det_names}

    dev_scores: dict[str, np.ndarray] = {}
    te_scores: dict[str, np.ndarray] = {}
    evaded_scores: dict[str, np.ndarray] = {}
    cals: dict[str, Calibration] = {}
    metrics: dict[str, dict] = {}
    t0 = time.time()
    for name, det in detectors.items():
        if isinstance(det, SupervisedDetector):
            # 有标注样本可用时：dev 拟合 + 5 折交叉分数用于校准
            sd = det.oof_scores(Xdev, dev_lab)
            det.fit_labeled(Xdev, dev_lab)
            st = det.score(Xte, _texts(test))
        elif isinstance(det, CharNgramDetector):
            # 词面基线：只用文本，不看 embedding
            sd = det.oof_scores(_texts(dev), dev_lab)
            det.fit_labeled(_texts(dev), dev_lab)
            st = det.score(Xte, _texts(test))
        else:
            det.fit(Xtr, texts_tr)
            sd = det.score(Xdev, _texts(dev))
            st = det.score(Xte, _texts(test))
        if eval_text is not None and eval_X is not None:
            evaded_scores[name] = det.score(eval_X, eval_text)
        cal = Calibration.fit(sd[dev_cal_mask], cfg.detector.target_fpr)
        dev_scores[name] = sd
        te_scores[name] = st
        cals[name] = cal
        m = E.binary_metrics(te_lab[te_main_mask], st[te_main_mask], cal.raw_threshold)
        m["dev_achieved_fpr"] = round(cal.achieved_fpr, 4)
        # `tpr_at_fpr1` 内部用测试集自己的 ROC 找阈值 = 乐观上界；
        # `tpr_at_fpr1_devcal` 用 dev 干净样本定的阈值在 test 上量，才是上线口径。
        m["tpr_at_fpr1_devcal"] = round(E.tpr_at_fpr_calibrated(
            dev_lab[dev_cal_mask], sd[dev_cal_mask],
            te_lab[te_main_mask], st[te_main_mask], cfg.detector.target_fpr), 4)
        m["fpr_ood"] = round(float((st[ood_mask] > cal.raw_threshold).mean()), 4)
        # 严格口径：把 OOD 也算作负样本(真实流量里它们会出现)，用来看
        # "检测器到底抓的是有害性，还是只是话题新颖性"——两个数差得越远越可疑。
        strict = E.binary_metrics(te_lab, st, cal.raw_threshold)
        m["auroc_allneg"] = round(strict["auroc"], 4)
        m["tpr_at_fpr1_allneg"] = round(strict["tpr_at_fpr1"], 4)
        metrics[name] = m
        print(f"[detect] {name:<12} AUROC={m['auroc']:.3f} "
              f"TPR@1%={m['tpr_at_fpr1']:.3f} (dev校准 {m['tpr_at_fpr1_devcal']:.3f}) "
              f"F1={m['f1']:.3f} "
              f"| 含OOD AUROC={m['auroc_allneg']:.3f} OOD误报={m['fpr_ood']:.3f} "
              f"(thr={cal.raw_threshold:.4g}, {time.time() - t0:.1f}s)")

    # ---------------- 融合 ----------------
    if "fusion" in cfg.detector.enable and len(detectors) >= 2:
        members = list(detectors.values())
        # 融合权重只用 dev(且只用主评估样本)搜索
        dev_main = _main_eval_mask(dev)
        fusion = FusionDetector(members).fit_fusion(
            {k: v[dev_main] for k, v in dev_scores.items()}, dev_lab[dev_main], cals)
        mapped_dev = {n: cals[n].to_fpr(dev_scores[n]) for n in dev_scores}
        mapped_te = {n: cals[n].to_fpr(te_scores[n]) for n in te_scores}
        fd = fusion.fuse(mapped_dev)
        ft = fusion.fuse(mapped_te)
        cal = Calibration.fit(fd[dev_cal_mask], cfg.detector.target_fpr)
        dev_scores["fusion"] = fd
        te_scores["fusion"] = ft
        cals["fusion"] = cal
        m = E.binary_metrics(te_lab[te_main_mask], ft[te_main_mask], cal.raw_threshold)
        m["dev_achieved_fpr"] = round(cal.achieved_fpr, 4)
        m["fpr_ood"] = round(float((ft[ood_mask] > cal.raw_threshold).mean()), 4)
        strict = E.binary_metrics(te_lab, ft, cal.raw_threshold)
        m["auroc_allneg"] = round(strict["auroc"], 4)
        m["tpr_at_fpr1_allneg"] = round(strict["tpr_at_fpr1"], 4)
        metrics["fusion"] = m
        weights = dict(zip([d.name for d in members], fusion.weights.tolist()))
        print(f"[detect] {'fusion':<12} AUROC={m['auroc']:.3f} "
              f"TPR@1%={m['tpr_at_fpr1']:.3f} F1={m['f1']:.3f} "
              f"| 含OOD AUROC={m['auroc_allneg']:.3f} OOD误报={m['fpr_ood']:.3f} "
              f"weights={weights}")
    else:
        weights = None

    return dev_scores, te_scores, evaded_scores, cals, metrics, weights


def cmd_detect(cfg: Config, args) -> dict:
    cfg.makedirs()
    splits = _load_or_build_splits(cfg)
    tr, dev, test = splits["train"], splits["dev"], splits["test"]

    enc = get_encoder(cfg)
    t0 = time.time()
    Xtr = enc.encode(_texts(tr))
    Xdev = enc.encode(_texts(dev))
    Xte = enc.encode(_texts(test))
    print(f"[detect] 编码完成 {Xtr.shape} / {Xdev.shape} / {Xte.shape} "
          f"({time.time() - t0:.1f}s)")

    evade = (getattr(args, "evade", None) or "none")
    evade_meta = None
    evaded_scores: dict[str, np.ndarray] = {}
    if evade != "none":
        from .evade import evade_text
        # 只改写 test 的 edit 族正样本；其余样本原样打分（保持与基线同分布）
        edit_mask = np.array([s.family == "edit" and s.label == 1 for s in test])
        evaded = [evade_text(s.text, evade, seed=42 + i) if edit_mask[i] else s.text
                  for i, s in enumerate(test)]
        n_ed = int(edit_mask.sum())
        # 只对真正改写的 edit 样本重新编码；其余复用基线向量（省 90% 编码时间）
        edit_idx = np.where(edit_mask)[0]
        if n_ed:
            new_embs = enc.encode([evaded[int(i)] for i in edit_idx])
            Xev = Xte.copy()
            Xev[edit_idx] = new_embs
        else:
            Xev = Xte
        print(f"[detect] 自适应攻击: method={evade} edit样本改写了 {n_ed} 条 -> 重打分")
        dev_scores, te_scores, evaded_scores, cals, metrics, weights = _fit_and_eval(
            cfg, Xtr, _texts(tr), dev, Xdev, test, Xte,
            eval_text=evaded, eval_X=Xev,
        )
        # 配对对比 edit 族 no-evade vs evaded 的分数（代表性检测器）
        if n_ed:
            rep = "sup_tfidf" if "sup_tfidf" in te_scores else next(iter(te_scores))
            base_s = te_scores[rep]
            ev_s = evaded_scores.get(rep)
            if ev_s is not None and len(ev_s) == len(test):
                d = base_s[edit_mask] - ev_s[edit_mask]
                pred = cals[rep].predict(ev_s)
                edit_recall_evaded = float(pred[edit_mask].mean())
                evade_meta = {
                    "method": evade,
                    "n_edit_rewritten": int(n_ed),
                    "detector": rep,
                    "edit_recall_base": round(float(cals[rep].predict(base_s)[edit_mask].mean()), 4),
                    "edit_recall_evaded": round(edit_recall_evaded, 4),
                    "mean_score_delta": round(float(d.mean()), 4),
                    "pct_scores_dropped": round(float((d > 0).mean()), 4),
                }
    else:
        dev_scores, te_scores, _evaded_scores, cals, metrics, weights = _fit_and_eval(
            cfg, Xtr, _texts(tr), dev, Xdev, test, Xte,
        )

    te_lab = _labels(test)
    families = [s.family for s in test]
    te_main = _main_eval_mask(test)

    # ---------------- 置信区间(与主指标同一批样本) ----------------
    auroc_ci = {n: E.bootstrap_ci(te_lab[te_main], s[te_main], metric=E.auroc,
                                  n_boot=args.boot)
                for n, s in te_scores.items()}

    # ---------------- 分家族 / 分组 ----------------
    family_recall, fpr_group = {}, {}
    for n, s in te_scores.items():
        pred = cals[n].predict(s)
        family_recall[n] = E.recall_by_family(test, pred)
        fpr_group[n] = E.fpr_by_group(test, pred)

    # ---------------- 显著性检验(fusion vs 每个基线) ----------------
    mcnemar: dict[str, dict] = {}
    if "fusion" in te_scores:
        base_pred = cals["fusion"].predict(te_scores["fusion"])
        for n in te_scores:
            if n == "fusion":
                continue
            mcnemar[f"fusion vs {n}"] = E.mcnemar(
                te_lab, base_pred, cals[n].predict(te_scores[n]))

    results = {
        "meta": {
            "encoder": cfg.encoder.name,
            "encoder_dir": cfg.encoder.model_dir,
            "target_fpr": cfg.detector.target_fpr,
            "n_train": len(tr), "n_dev": len(dev), "n_test": len(test),
            "families": list(cfg.attack.families) + (["edit"] if cfg.attack.edit_family else []),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provenance": _provenance(cfg),
        },
        "detectors": metrics,
        "auroc_ci": auroc_ci,
        "family_recall": family_recall,
        "fpr_group": fpr_group,
        "fusion_weights": weights,
        "mcnemar": mcnemar,
        "evade": evade_meta,
    }
    E.save_json(results, _metrics_path(cfg, "detect.json"))
    if evade_meta is not None:
        # 每方法单独存档，避免 9 次全跑互相覆盖 detect.json
        E.save_json(results, _metrics_path(cfg, f"detect_evade_{evade}.json"))

    # 存原始分数，便于后面单独重画图/做误差分析
    np.savez_compressed(
        _metrics_path(cfg, "test_scores.npz"),
        y=te_lab, families=np.array(families),
        **{n: s.astype(np.float32) for n, s in te_scores.items()},
    )
    if evaded_scores:
        np.savez_compressed(
            _metrics_path(cfg, "test_scores_evaded.npz"),
            y=te_lab, families=np.array(families),
            **{n: s.astype(np.float32) for n, s in evaded_scores.items()},
        )
    if evade_meta is not None and evaded_scores:
        np.savez_compressed(
            _metrics_path(cfg, f"test_scores_evaded_{evade}.npz"),
            y=te_lab, families=np.array(families),
            **{n: s.astype(np.float32) for n, s in evaded_scores.items()},
        )
    np.savez_compressed(
        _metrics_path(cfg, "dev_scores.npz"),
        y=_labels(dev),
        **{n: s.astype(np.float32) for n, s in dev_scores.items()},
    )

    # ---------------- 图 ----------------
    E.plot_roc({n: (te_lab, s) for n, s in te_scores.items()},
               _fig(cfg, "fig1_roc.png"), "ROC：各检测器在未见攻击上的表现")
    E.plot_family_heatmap({n: {f: v["recall"] for f, v in fr.items()}
                           for n, fr in family_recall.items()},
                          _fig(cfg, "fig2_family_heatmap.png"))
    E.plot_bars({n: m["tpr_at_fpr1"] for n, m in metrics.items()},
                _fig(cfg, "fig3_tpr_at_fpr1.png"),
                ylabel="TPR @ FPR=1%", title="低误报率下的召回率对比")
    return results


# ----------------------------------------------------------------------
def cmd_victim(cfg: Config, args) -> dict:
    from .victim import BertVictim

    ckpt = Path(cfg.victim.weights)
    if not ckpt.exists():
        print(f"[victim] 缺少权重 {ckpt}，跳过")
        return {}
    splits = _load_or_build_splits(cfg)
    tr, dev, test = splits["train"], splits["dev"], splits["test"]

    t0 = time.time()
    victim = BertVictim(ckpt, tokenizer_dir=cfg.victim.model_dir,
                        device="cpu", max_length=cfg.victim.max_length,
                        batch_size=cfg.victim.batch_size)
    dv = victim.predict(_texts(dev))
    te = victim.predict(_texts(test))
    n_params = int(sum(p.numel() for p in victim.model.parameters()))
    print(f"[victim] 推理完成 ({time.time() - t0:.1f}s), 参数量 {n_params / 1e6:.1f}M, "
          f"加载校验={victim.load_report}")

    dev_lab, te_lab = _labels(dev), _labels(test)
    dev_cal_mask = _indist_clean_mask(dev)
    te_main = _main_eval_mask(test)
    ood_mask = np.array([s.family == "ood" for s in test])
    signals = {
        "victim_entropy": (dv.entropy, te.entropy),      # 熵高 = 模型没见过 = 可疑
        "victim_p_poison": (dv.p_poison, te.p_poison),   # 模型判为污染的概率
        # margin = logit(污染) - logit(正常)，越大越可疑
        # (最初写成 -margin，AUROC 直接变成 0.000——符号搞反的典型症状)
        "victim_margin": (dv.margin, te.margin),
    }
    metrics, family_recall = {}, {}
    for name, (sd, st) in signals.items():
        cal = Calibration.fit(sd[dev_cal_mask], cfg.detector.target_fpr)
        m = E.binary_metrics(te_lab[te_main], st[te_main], cal.raw_threshold)
        m["dev_achieved_fpr"] = round(cal.achieved_fpr, 4)
        m["fpr_ood"] = round(float((st[ood_mask] > cal.raw_threshold).mean()), 4)
        m["auroc_allneg"] = round(E.binary_metrics(te_lab, st, cal.raw_threshold)["auroc"], 4)
        metrics[name] = m
        family_recall[name] = E.recall_by_family(test, cal.predict(st))
        print(f"[victim] {name:<18} AUROC={m['auroc']:.3f} TPR@1%={m['tpr_at_fpr1']:.3f} "
              f"F1={m['f1']:.3f} | 含OOD AUROC={m['auroc_allneg']:.3f} "
              f"OOD误报={m['fpr_ood']:.3f}")

    out = {
        "model": str(ckpt),
        "n_params": f"{n_params / 1e6:.1f}M",
        "load_report": victim.load_report,
        "detectors": metrics,
        "family_recall": family_recall,
    }
    E.save_json(out, _metrics_path(cfg, "victim.json"))
    E.plot_roc({n: (te_lab, st) for n, (sd, st) in signals.items()},
               _fig(cfg, "fig4_victim_signals.png"), "行为级信号 ROC")
    return out


# ----------------------------------------------------------------------
def cmd_ablation(cfg: Config, args) -> dict:
    """投毒比例消融：用句向量+逻辑回归探针代替 BERT 重训(CPU 上秒级)。"""
    from .attacks import build_attacks
    from .victim import ProbeVictim, poison_mix

    splits = _load_or_build_splits(cfg)
    tr, test = splits["train"], splits["test"]
    enc = get_encoder(cfg)

    clean_tr = [s.text for s in tr if s.label == 0]
    if args.ablation_n and args.ablation_n < len(clean_tr):
        clean_tr = clean_tr[: args.ablation_n]
    atk = [s.text for s in build_attacks(cfg.attack.families, 300, 555)]

    t0 = time.time()
    X_clean = enc.encode(clean_tr)
    X_atk = enc.encode(atk)
    X_te = enc.encode(_texts(test))
    print(f"[ablation] 编码完成 ({time.time() - t0:.1f}s)")

    te_lab = _labels(test)
    clean_mask = te_lab == 0
    rates = args.rates
    res = {"rates": rates, "clean_fp_rate": {}, "poison_recall": {},
           "probe_auc": {}, "mean_on_clean": {}, "mean_on_attack": {},
           "note": "探针 = 句向量 + 逻辑回归，代替 BERT 全量重训"}
    for rate in rates:
        texts, labels = poison_mix(clean_tr, atk, rate, seed=0)
        idx_c = [i for i, t in enumerate(texts) if labels[i] == 0]
        idx_p = [i for i, t in enumerate(texts) if labels[i] == 1]
        # 直接按索引拼 embedding，避免重复编码
        pos_in_clean = {t: i for i, t in enumerate(clean_tr)}
        pos_in_atk = {t: i for i, t in enumerate(atk)}
        Xs, ys = [], []
        for i in idx_c:
            Xs.append(X_clean[pos_in_clean[texts[i]]])
            ys.append(0)
        for i in idx_p:
            Xs.append(X_atk[pos_in_atk[texts[i]]])
            ys.append(1)
        X = np.stack(Xs)
        y = np.array(ys)
        if len(idx_p) < 5:
            # rate≈0 时训练集里没有正类：探针学不到任何信号，
            # 正确的语义是"恒判正常"，而不是崩溃或随机。
            s_te = np.zeros(len(X_te), dtype=np.float32)
            res.setdefault("notes", {})[str(rate)] = "无正类样本，探针恒判正常"
        else:
            probe = ProbeVictim(seed=0).fit(X, y)
            s_te = probe.score(X_te)
        res["clean_fp_rate"][str(rate)] = round(float((s_te[clean_mask] > 0.5).mean()), 4)
        res["poison_recall"][str(rate)] = round(float((s_te[~clean_mask] > 0.5).mean()), 4)
        res["probe_auc"][str(rate)] = round(E.auroc(te_lab, s_te), 4)
        res["mean_on_clean"][str(rate)] = round(float(s_te[clean_mask].mean()), 4)
        res["mean_on_attack"][str(rate)] = round(float(s_te[~clean_mask].mean()), 4)
        print(f"[ablation] rate={rate:<5} clean_fp={res['clean_fp_rate'][str(rate)]:.3f} "
              f"poison_recall={res['poison_recall'][str(rate)]:.3f} "
              f"auc={res['probe_auc'][str(rate)]:.3f} "
              f"score(clean/attack)={res['mean_on_clean'][str(rate)]:.3f}/"
              f"{res['mean_on_attack'][str(rate)]:.3f}")

    E.save_json(res, _metrics_path(cfg, "ablation.json"))
    E.plot_curve(rates,
                 {"干净测试集误判率": [res["clean_fp_rate"][str(r)] for r in rates],
                  "投毒测试集召回": [res["poison_recall"][str(r)] for r in rates]},
                 _fig(cfg, "fig5_ablation.png"), xlabel="训练集投毒比例",
                 ylabel="比例", title="投毒比例对模型行为的影响")
    return res


# ----------------------------------------------------------------------
def cmd_judge(cfg: Config, args) -> dict:
    """语义层：LLM 判官。

    v3 的两个关键改动：
    1) **判定接口换 logprob**：读「污」「正」两个候选 token 的概率比得到连续分。
       旧口径让模型生成 JSON，分数只有 {0,1}，FPR=1% 的阈值一旦卡在 1.0，
       TPR 结构性为 0；换连续分后才能真正算 TPR@FPR=1%。
    2) **级联改成召回-成本曲线**：v2 只在 kNN 的固定分位(0.80~0.995)上调用判官，
       得到一个"调用 19.4%、召回 16.7%"的单点结论，无法说明级联是否值得。
       现在直接扫预算(5%~100%)，输出每个预算下的召回/精确率，
       这才是运维真正需要的那张表。
    """
    from .detectors import KNNDetector
    from .judge import build_judge

    splits = _load_or_build_splits(cfg)
    tr, dev, test = splits["train"], splits["dev"], splits["test"]
    enc = get_encoder(cfg)
    Xtr = enc.encode(_texts(tr))
    Xte = enc.encode(_texts(test))

    # 第一级(闸门)：决定"先把哪一批送进 LLM"。
    # v2 用的是无监督 kNN，在真实语料上 AUROC 只有 0.44，等于随机抽；
    # v3 允许换成有监督的词面分(sup_tfidf)，级联曲线才有意义。
    gate = str(getattr(args, "gate", "knn") or "knn")
    if gate == "sup_tfidf":
        # 与 detect 同一套口径：有监督模型用 dev 的带标注样本拟合
        # (v3 训练集是纯干净语料，没有正类，不能拿它当监督数据)
        g = CharNgramDetector(seed=cfg.data.seed)
        g.fit_labeled(_texts(dev), np.array(_labels(dev)))
        first = g.score(Xte, _texts(test))
    else:
        gate = "knn"
        knn = KNNDetector(cfg.detector.knn_k).fit(Xtr, _texts(tr))
        first = knn.score(Xte, _texts(test))
    print(f"[judge] 级联闸门 = {gate}")

    # 子采样：每类(家族/干净)最多 limit 条，控制 LLM 调用成本
    by_group: dict[str, list[int]] = {}
    for i, s in enumerate(test):
        by_group.setdefault(s.family, []).append(i)
    rng = np.random.default_rng(0)
    pick: list[int] = []
    for g, idxs in sorted(by_group.items()):
        take = min(args.limit, len(idxs))
        pick.extend(rng.choice(idxs, size=take, replace=False).tolist())
    pick = sorted(set(pick))

    scoring = getattr(args, "judge_scoring", None) or getattr(cfg.qwen, "scoring", "logprob")
    judge = build_judge(cfg, kind=args.judge,
                        cache_path=_metrics_path(cfg, "judge_cache.json"),
                        scoring=scoring,
                        hardened=getattr(cfg.qwen, "hardened", False))
    texts_sub = [test[i].text for i in pick]
    t0 = time.time()
    # 分块推理：CPU 上 0.5B 生成模式约 1.4s/条，logprob 模式只做一次 forward，快得多。
    # 块结束即落盘缓存，中断后重跑能续上。
    CHUNK = int(getattr(args, "judge_chunk", 24) or 24)
    parts: list[np.ndarray] = []
    t_llm = 0.0   # 只统计真正跑 forward 的时间(缓存命中不算，否则单条耗时会被摊薄)
    n_llm = 0
    for s0 in range(0, len(texts_sub), CHUNK):
        _c0 = time.time()
        part = judge.score(texts_sub[s0:s0 + CHUNK])
        t_llm += time.time() - _c0
        n_llm += int((getattr(judge, "last_stats", {}) or {}).get("n_llm", 0) or 0)
        parts.append(part)
        print(f"[judge] {min(s0 + CHUNK, len(texts_sub))}/{len(texts_sub)} 条"
              f" ({time.time() - t0:.1f}s)", flush=True)
    scores = np.concatenate(parts) if parts else np.array([], dtype=np.float32)

    sub = [test[i] for i in pick]
    y = _labels(sub)
    ok = ~np.isnan(scores)
    name = getattr(judge, "name", None) or f"judge_{cfg.qwen.model_id.split('/')[-1]}"
    clean_mask = np.array([s.label == 0 and s.family in _CLEAN_IN_DIST for s in sub])

    metrics: dict[str, dict] = {}
    family_recall: dict[str, dict] = {}
    family_recall_half: dict[str, dict] = {}
    if ok.sum() > 0 and len(set(y[ok].tolist())) > 1:
        # 阈值用**干净样本**校准(与数据层同一套口径)；样本太少时退回 0.5
        if clean_mask.sum() >= 20:
            cal = Calibration.fit(scores[clean_mask & ok], cfg.detector.target_fpr)
            thr = cal.raw_threshold
            cal_fpr = round(cal.achieved_fpr, 4)
        else:
            thr, cal_fpr = 0.5, None
        m = E.binary_metrics(y[ok], scores[ok], thr)
        m["parse_fail"] = int((~ok).sum())
        m["scoring"] = scoring
        m["dev_achieved_fpr"] = cal_fpr
        if n_llm > 0:
            m["sec_per_item"] = round(t_llm / n_llm, 4)
        else:
            # 全命中缓存：这一轮没跑 forward，沿用上一次实测值，避免报成 0
            prev = _load_prev_metrics(cfg, "judge.json").get("detectors", {}).get(name, {})
            if prev.get("sec_per_item"):
                m["sec_per_item"] = prev["sec_per_item"]
                m["sec_per_item_cached"] = True
        ci = E.bootstrap_ci(y[ok], scores[ok], metric=E.auroc, n_boot=int(getattr(args, "boot", 300) or 300))
        m["auroc_ci95"] = [round(ci[1], 4), round(ci[2], 4)]
        # 判官吐的是概率，就得证明它"当概率用"是可信的：ECE + 平均置信度。
        # logprob 接口最常见的毛病是过度自信(阈值被推到 0.96 就是征兆)。
        m["ece"] = round(E.ece(y[ok], scores[ok]), 4)
        m["mean_score"] = round(float(scores[ok].mean()), 4)
        # 判官的阈值本来就来自 dev 干净集，所以"dev 校准口径"就是 recall；
        # 单阶段检测器的阈值是 test 自搜的，两者放在同一张表里必须区分开。
        m["tpr_at_fpr1_devcal"] = m["recall"]
        metrics[name] = m
        family_recall = {name: E.recall_by_family(
            [s for s, k in zip(sub, ok) if k], (scores[ok] > thr).astype(int))}
        # 同一批样本在固定 0.5 工作点上的分家族召回：dev 阈值(≈0.97)太靠尾部，
        # 单看那张表会全是 0，读不出"哪个家族更难"。
        family_recall_half = {name: E.recall_by_family(
            [s for s, k in zip(sub, ok) if k], (scores[ok] > 0.5).astype(int))}
        E.plot_roc({name: (y[ok], scores[ok])}, _fig(cfg, "fig6_judge_roc.png"),
                   f"Qwen 判官 ROC（{scoring}）")

    # ---------------- 级联：召回-成本曲线 ----------------
    curve: list[dict] = []
    if ok.sum() > 0 and len(set(y[ok].tolist())) > 1:
        curve = E.cascade_curve(first[pick][ok], scores[ok], y[ok])
        for row in curve:
            print(f"[judge] 级联 调用{row['call_ratio']:.0%} → 召回 {row['recall']:.3f}"
                  f" / 精确率 {row['precision']:.3f}")
    # 兼容 v2：固定灰区分位的旧口径(命中缓存，几乎不额外花时间)
    cascade = dict(curve and {"curve": curve} or {})
    # 闸门对比：kNN 与 sup_tfidf 的曲线都留在同一个文件里(分次跑也能累积)
    by_gate = dict((_load_prev_metrics(cfg, "judge.json").get("cascade") or {}).get("by_gate") or {})
    if curve:
        by_gate[gate] = curve
        cascade["by_gate"] = by_gate
    if hasattr(judge, "score_cascade"):
        try:
            casc = judge.score_cascade(texts_sub, first[pick],
                                       low_q=getattr(args, "low_q", 0.80),
                                       high_q=getattr(args, "high_q", 0.995))
            cascade.update(dict(getattr(judge, "cascade_stats", {})))
            cascade["n_eval"] = int(len(texts_sub))
            if len(set(y.tolist())) > 1 and scoring == "binary":
                metrics["cascade(kNN→judge)"] = E.binary_metrics(y, casc, 0.5)
        except Exception as e:  # 级联失败不应该让整个 judge 阶段失败
            print(f"[judge] 级联统计跳过: {e}")

    out = {
        "model": getattr(judge, "model_id", name),
        "scoring": scoring,
        "gate": gate,
        "few_shot": getattr(judge, "few_shot", False),
        "n": int(len(sub)),
        "parse_fail": int((~ok).sum()),
        "detectors": metrics,
        "family_recall": family_recall,
        "family_recall_at_0.5": family_recall_half,
        "cascade": cascade,
        "provenance": _provenance(cfg),
        "note": "若为 judge_heuristic 则说明本机无 Qwen 权重，仅用于跑通链路，不计入主结论",
    }
    E.save_json(out, _metrics_path(cfg, "judge.json"))

    # 把逐条打分落盘：蒸馏学生模型(distill)用它当软标签，
    # 也让"判官为什么错"可以人工复查，而不是只剩一张汇总表。
    NL = chr(10)

    def _dump(path, samples, first_scores, vals):
        import json as _json

        with path.open("w", encoding="utf-8") as fh:
            for s, f0, v in zip(samples, first_scores, vals):
                rec = {"text": s.text, "label": int(s.label), "family": s.family,
                       "knn": float(f0),
                       "judge_score": None if np.isnan(v) else round(float(v), 6)}
                fh.write(_json.dumps(rec, ensure_ascii=False) + NL)

    _dump(_metrics_path(cfg, "judge_samples.jsonl"), sub, first[pick], scores)

    # dev 侧抽样同样打一遍分：蒸馏需要**训练侧**软标签，
    # 否则学生模型只能在同一批测试抽样上再切训练/测试，等于自己教自己。
    n_dev = int(getattr(args, "dev_limit", 0) or 0)
    if n_dev > 0:
        dby: dict[str, list[int]] = {}
        for i, s in enumerate(dev):
            dby.setdefault(s.family, []).append(i)
        dpick: list[int] = []
        for _g, idxs in sorted(dby.items()):
            dpick.extend(rng.choice(idxs, size=min(n_dev, len(idxs)), replace=False).tolist())
        dpick = sorted(set(dpick))
        dtexts = [dev[i].text for i in dpick]
        if dtexts:
            dscores = np.concatenate([judge.score(dtexts[k:k + CHUNK])
                                      for k in range(0, len(dtexts), CHUNK)])
            dknn = knn.score(enc.encode(dtexts), dtexts)
            _dump(_metrics_path(cfg, "judge_samples_dev.jsonl"),
                  [dev[i] for i in dpick], dknn, dscores)
            print(f"[judge] dev 抽样 {len(dpick)} 条已落盘（供蒸馏训练）")
    return out


def _feedback_clean_scores(cfg: Config, args):
    """给 C4 找一套"与线上 score 同尺度"的 dev 干净分数。

    优先级：
    1. `--student pkl`：用学生模型现算（推荐——`ingest`/`run` 的 score 就是它的输出）；
    2. 兜底：dev 分数缓存里的同名检测器（尺度可能与线上不一致，会在报告里标注）。
    """
    import numpy as np

    from .conformal import suffix_of
    from .data import load_splits

    dev = load_splits(cfg.data.processed_dir).get("dev", [])
    clean = [s for s in dev if s.label == 0 and s.family in ("normal", "hardneg")]

    student_path = getattr(args, "student", None)
    if student_path:
        sp = Path(student_path)
        if not sp.exists():
            print(f"[feedback] 找不到学生模型 {sp}，改用分数缓存")
        elif clean:
            from .distill import JudgeStudent
            from .encoders import get_encoder

            print(f"[feedback] 用学生模型在 {len(clean)} 条 dev 干净样本上重算阈值…")
            enc = get_encoder(cfg)
            texts = [s.text for s in clean]
            clean_scores = JudgeStudent.load(sp).score(texts, enc.encode(texts))
            return clean_scores, "student"

    try:
        npz = np.load(Path(cfg.metrics_dir) / f"dev_scores{suffix_of(cfg)}.npz",
                      allow_pickle=True)
    except FileNotFoundError:
        print("[feedback] 没有 dev 分数缓存（先跑 detect），跳过影子阈值")
        return None, None
    name = getattr(args, "detector", "sup_tfidf")
    if name not in npz.files or not dev:
        print(f"[feedback] 分数缓存里没有 {name}，跳过影子阈值")
        return None, None
    # 注意：分数缓存是整个 dev 的行数，mask 要按 dev 全量算，再取干净子集
    mask = np.array([s.label == 0 and s.family in ("normal", "hardneg") for s in dev])
    if len(mask) != len(npz[name]):
        print(f"[feedback] dev 样本数 {len(mask)} 与分数行数 {len(npz[name])} 不一致，"
              "跳过影子阈值")
        return None, None
    return npz[name][mask], name


def cmd_report(cfg: Config, args) -> Path:
    def _read(name: str):
        p = _metrics_path(cfg, name)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    # 只收本轮口径的图：否则 v2 的 fig1_roc.png 会混进 v3 报告，读图的人会被误导
    allfigs = sorted(str(p) for p in Path(cfg.figures_dir).glob("*.png"))
    if RUN_SUFFIX:
        own = [f for f in allfigs if Path(f).stem.endswith(RUN_SUFFIX)]
        figures = own or allfigs
    else:
        figures = [f for f in allfigs if not any(
            Path(f).stem.endswith(s) for s in ("_real", "_holdout", "_real_v3"))] or allfigs
    attacks: dict[str, dict] = {}
    for f in sorted(Path(cfg.metrics_dir).glob("strongattack_*.json")):
        stem = f.stem
        if RUN_SUFFIX and not stem.endswith(RUN_SUFFIX):
            continue
        method = stem[len("strongattack_"):]
        if RUN_SUFFIX:
            method = method[: -len(RUN_SUFFIX)]
        attacks[method] = json.loads(f.read_text(encoding="utf-8"))
    return R.build_report(
        cfg,
        dataset_stats=_read("dataset_stats.json"),
        results=_read("detect.json"),
        ablation=_read("ablation.json"),
        judge=_read("judge.json"),
        victim=_read("victim.json"),
        figures=figures,
        out_path=_p(cfg, "docs", f"实验报告{RUN_SUFFIX}.md"),
        distill=_read("distill.json"),
        redteam=_read("redteam.json"),
        struct=_read("struct.json"),
        conformal=_read("conformal.json"),
        attacks=attacks or None,
        ranker=_read("ranker.json"),
    )


# ----------------------------------------------------------------------
def cmd_download(cfg: Config, args) -> None:
    from .localmodels import ensure_model

    if args.encoder:
        from .encoders import _MODEL_SPECS

        spec = _MODEL_SPECS[args.encoder]
        path = ensure_model(spec["model_id"], Path(cfg.outputs_dir).parent / "model_cache")
        print(f"[download] {args.encoder} -> {path}")
    if args.qwen:
        mid = {"1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
               "0.5b": "Qwen/Qwen2.5-0.5B-Instruct"}.get(args.qwen, args.qwen)
        path = ensure_model(mid, cfg.encoder.cache_dir)
        print(f"[download] {mid} -> {path}")
    if getattr(args, "corpus", None):
        from .data import fetch_real_corpus

        # 每个语料落一个独立文件，避免用 iflytek 时覆盖掉 tnews 的语料。
        fname = {"tnews": "real_tnews.txt", "iflytek": "real_iflytek.txt"}.get(
            args.corpus, f"real_{args.corpus}.txt"
        )
        out = Path(cfg.data.raw_dir) / fname
        fetch_real_corpus(out, subset=args.corpus)


# ----------------------------------------------------------------------
def cmd_distill(cfg: Config, args) -> dict:
    """判官蒸馏：LLM 软标签 → 毫秒级学生模型（v3 的降本环节）。"""
    from .distill import run_distill

    return run_distill(
        cfg,
        dev_file=getattr(args, "dev_file", None),
        test_file=getattr(args, "test_file", None),
        C=float(getattr(args, "distill_c", 1.0) or 1.0),
    )


def cmd_serve(cfg: Config, args) -> None:
    """起在线打分服务（数据层 + 学生模型，不含 LLM）。"""
    import uvicorn

    from .service import _default_student, create_app

    student = getattr(args, "student", None) or _default_student(cfg)
    app = create_app(cfg, student_path=student)
    host = getattr(args, "host", "127.0.0.1")
    port = int(getattr(args, "port", 8000))
    print(f"[serve] http://{host}:{port}  (student={student})")
    uvicorn.run(app, host=host, port=port, log_level="info")


def cmd_doctor(cfg: Config, args) -> dict:
    """环境自检：依赖、模型缓存、数据切分、已有产物，一次说清哪里缺东西。"""
    import importlib

    rows: list[tuple[str, str, str]] = []
    for mod in ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "torch",
                "transformers", "safetensors", "yaml", "fastapi", "uvicorn",
                "modelscope", "pytest"):
        try:
            m = importlib.import_module(mod)
            rows.append((mod, "OK", str(getattr(m, "__version__", "?"))))
        except Exception:
            rows.append((mod, "MISSING", "-"))
    root = Path(cfg.outputs_dir).parent
    from .localmodels import find_local_model_dir

    # 判官有权重就报 OK，没有就明确说"语义层会退化为 heuristic",
    # 而不是让人跑完 judge 才发现拿到的是一张 heuristic 表。
    qwen_hit = [(mid, find_local_model_dir(mid, cfg.qwen.model_cache))
                for mid in dict.fromkeys([cfg.qwen.model_id, cfg.qwen.fallback_model_id])]
    qwen_ok = [f"{mid.split('/')[-1]}@{pth}" for mid, pth in qwen_hit if pth]
    for name, p in (
        ("encoder(crom)", Path(cfg.encoder.model_dir)),
        ("processed", Path(cfg.data.processed_dir)),
        ("raw/real_tnews", Path(cfg.data.raw_dir) / cfg.data.real_file),
        ("metrics", Path(cfg.outputs_dir) / "metrics"),
    ):
        rows.append((name, "OK" if p.exists() else "MISSING", str(p)))
    rows.append(("qwen weights", "OK" if qwen_ok else "MISSING",
                 ", ".join(qwen_ok) or "语义层将退化为 heuristic"))

    out = {"python": sys.version.split()[0], "root": str(root),
           "checks": [{"item": a, "status": b, "detail": c} for a, b, c in rows],
           "provenance": _provenance(cfg)}
    E.save_json(out, _metrics_path(cfg, "doctor.json"))
    bad = [r for r in out["checks"] if r["status"] != "OK"]
    width = max(len(r["item"]) for r in out["checks"])
    for r in out["checks"]:
        mark = "  " if r["status"] == "OK" else "!!"
        print(f"{mark} {r['item']:<{width}}  {r['status']:<8} {r['detail']}")
    print(f"[doctor] {len(bad)} 项异常" if bad else "[doctor] 环境就绪")
    return out


# ----------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser, with_encoder: bool = True) -> None:
    p.add_argument("--config", default=None, help="YAML 配置文件路径")
    if with_encoder:
        p.add_argument("--encoder", default=None, help="覆盖编码器: crom/bge/qwen3")
    p.add_argument("--target-fpr", type=float, default=None, help="阈值校准目标误报率")
    p.add_argument("--log-level", dest="log_level", default=None,
                   help="日志级别 DEBUG/INFO/WARNING（也可用 POISONGUARD_LOG_LEVEL）")
    p.add_argument("--log-json", dest="log_json", action="store_true",
                   help="日志输出单行 JSON（便于 Loki/ES 采集）")
    p.add_argument("--device", default=None, help="cpu/cuda/auto")
    p.add_argument("--edit", action="store_true",
                   help="启用 edit 攻击家族(真实新闻上做最小事实篡改)")
    p.add_argument("--hardneg", type=int, default=0,
                   help="把 N 条硬负样本混入 dev/test 主评测集(0=关闭)")
    p.add_argument("--style-holdout", action="store_true",
                   help="严格口径：dev 与 test 用互不相交的模板骨架(量化模板泄漏)")
    p.add_argument("--run-tag", default=None,
                   help="实验标签：产物与数据目录追加后缀，便于 v2/v3 口径并存对比")
    p.add_argument("--clean-source", default="synthetic", choices=["synthetic", "tnews"],
                   help="干净语料来源：synthetic=仓库自带槽位语料；tnews=真实新闻语料(需先跑 download 或已有 data/raw/real_tnews.txt)")
    p.add_argument("--real-file", default=None,
                   help="覆盖真实语料文件名(如 real_iflytek.txt / 你自己的文本文件，放在 data/raw/ 下，一行一句)")


# 各子命令参数不同，`all` 走一个统一默认值，避免 AttributeError
ARG_DEFAULTS = {
    "boot": 300,
    "judge": "auto",
    "limit": 40,
    "cascade_max": 180,
    "judge_chunk": 24,
    "qwen_model": None,
    "low_q": 0.80,
    "high_q": 0.995,
    "ablation_n": 1000,
    "rates": [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30],
    "qwen": None,
    "clean_source": "synthetic",
    "n_hard": 400,
    "run_tag": None,
    "dev_limit": 20,
    "judge_scoring": None,
    "edit": False,
    "hardneg": 0,
    "distill_c": 1.0,
    "dev_file": None,
    "test_file": None,
    "host": "127.0.0.1",
    "port": 8000,
    "student": None,
    "input": None,
    "format": "auto",
    "column": "text",
    "threshold": None,
    "judge_topk": 0,
    "limit": 0,
    "out_prefix": None,
    "no_html": False,
    "out": None,
    "bundle": None,
    "quantize": "none",
    "evade": "none",
    "sign_key": None,
    "sign_key_file": None,
    "require_signature": False,
    "fact_model": "Qwen/Qwen2.5-0.5B-Instruct", "fact_limit": 400, "fact_seed": 42,
    "calibrate": True, "sample_pass": 0, "db": None, "host": "127.0.0.1", "port": 8077,
    "target_fpr": 0.01, "detector": "sup_tfidf", "mode": "pooled",
    "min_support": 30, "corpus_dir": "outputs/feedback", "report": "docs/运维日报.md",
    "scene": "prompt", "gfile": None, "http": False, "no_scorer": False,
    "gthreshold": None, "block_mult": 2.0,

    "inbox": "data/inbox", "files": [], "pattern": "*.txt,*.csv,*.tsv,*.jsonl,*.json",
    "fmt": "auto", "column": "text", "limit": 0, "watch_band": 0.05,
    "threshold": None, "bundle": None, "verify_ed25519": None,
    "ledger_dir": "outputs/ingest", "out_dir": "outputs/ingest",
    "watch": False, "poll": 10.0, "max_files": 0, "no_explain": False,
    "struct_c": 1.0,
    "struct_seed": 42,
    "inject_c": 1.0,
    "inject_seed": 0,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="poisonguard", description="训练数据投毒检测流水线")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, hlp in (
        ("data", "构建数据切分并注入攻击样本"),
        ("detect", "数据层检测(主实验)"),
        ("victim", "行为层检测(受害模型信号)"),
        ("judge", "语义层检测(Qwen 判官)"),
        ("ablation", "投毒比例消融"),
        ("audit", "词面审计(检测器在抓有害性还是'没见过这种句式')"),
        ("hardneg", "硬负样本压力测试(攻击 vs 同样像传闻的无害新闻)"),
        ("distill", "判官蒸馏为毫秒级学生模型(降本环节)"),
        ("struct-train", "结构隔离判官：冻结 Qwen 主干 + 二分类头(A1)"),
        ("inject-train", "提示注入头(P3 L2)：冻结 Qwen 主干 + 注入二分类头，seen/unseen 两列评测"),
        ("conformal", "Conformal Prediction：把 1%% FPR 变成带有限样本保证的声明(A2)"),
        ("attack-study", "强攻击研究(A3)：零宽/同形/全角/回译/GCG 后缀 对真实投毒正样本的配对杀伤"),
        ("ranker", "学生排序器(A5)：字符 n-gram / 句向量 / 混合 / PCA / MLP 受控对比"),
        ("redteam", "判官提示注入红队(检测器被自己输入劫持的风险)"),
        ("ingest", "入库钩子(C2)：投递目录 → 自动打分 → 隔离/人审/放行 + 判决台账"),
        ("factcheck", "数字级事实校验(P0-2)：数字位置的模型先验 → 抓内容级篡改"),
        ("feedback", "误报闭环(C4)：人审判定 → 在线FPR/影子阈值/难负样本语料"),
        ("gate", "输入闸门(输入侧/检索/工具输出 → allow/watch/block)"),
        ("mcp", "MCP 服务(P2)：stdio JSON-RPC，Agent 可调 guard/review/ingest"),
        ("review", "人审台(C1)：BLOCK/WATCH 队列 → 判定 + 理由码落库"),
        ("serve", "起在线打分 HTTP 服务(不含 LLM)"),
        ("scan", "批量扫描任意文件(入库网关/RAG 审核)：txt/csv/jsonl → 分数+视图+人审报告"),
        ("keygen", "生成 ed25519 作者密钥对(私钥签、公钥验，真实供应链形态)"),
        ("deploy", "把训练产物打包成单文件 .pkg(拟合态, 数据不出域)"),
        ("run", "加载 .pkg 离线批扫(部署机免装配, 复用 scan 报告)"),
        ("doctor", "环境自检：依赖/模型缓存/数据/产物"),
        ("report", "渲染 Markdown 实验报告"),
        ("all", "依次执行 data→detect→victim→ablation→report"),
        ("download", "从 ModelScope 下载模型"),
    ):
        p = sub.add_parser(name, help=hlp)
        _add_common(p, with_encoder=(name != "download"))
        if name == "detect":
            p.add_argument("--boot", type=int, default=1000, help="Bootstrap 次数")
            p.add_argument("--evade", default="none",
                           choices=["none", "syn", "chr", "mix"],
                           help="自适应攻击：对 test 的 edit 正样本做逃避变换后重打分 "
                                "(syn=同义替换 chr=字符扰动 mix=组合；none=关闭)")
        if name == "judge":
            p.add_argument("--judge", default="auto", choices=["auto", "qwen", "heuristic"])
            p.add_argument("--limit", type=int, default=40, help="每类最多判定多少条")
            p.add_argument("--qwen-model", default=None,
                           help="覆盖判官模型: 0.5b / 1.5b / 完整 model id（默认取配置）")
            p.add_argument("--cascade-max", type=int, default=180,
                           help="级联评估的抽样条数(0=全量, CPU 上全量要 20 分钟以上)")
            p.add_argument("--judge-chunk", type=int, default=24,
                           help="判官每块推理条数(块结束即落判定缓存，中断可续跑)")
            p.add_argument("--low-q", type=float, default=0.80)
            p.add_argument("--high-q", type=float, default=0.995)
        if name == "ablation":
            p.add_argument("--ablation-n", type=int, default=1000, help="消融用干净训练样本数")
            p.add_argument("--rates", type=float, nargs="+",
                           default=[0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30])
        if name == "judge":
            p.add_argument("--dev-limit", type=int, default=20,
                           help="dev 侧每类抽样多少条打分(供蒸馏训练, 0=不抽)")
            p.add_argument("--judge-scoring", default=None,
                           choices=["logprob", "binary"],
                           help="判定接口: logprob=连续分(推荐) / binary=生成 JSON(旧口径)")
            p.add_argument("--gate", default="knn", choices=["knn", "sup_tfidf"],
                           help="级联闸门: knn=无监督离群(v2) / sup_tfidf=有监督词面分")
            p.add_argument("--judge-hardened", action="store_true",
                           help="对判官 prompt 做注入加固(分隔符+不可信数据声明)")
            p.add_argument("--judge-sanitize", action="store_true",
                           help="送判官前做确定性脱敏(替换伪装成指令的字面片段)，可叠加 --judge-hardened")
        if name == "distill":
            p.add_argument("--dev-file", default=None, help="训练侧软标签 jsonl")
            p.add_argument("--test-file", default=None, help="测试侧软标签 jsonl")
            p.add_argument("--distill-c", type=float, default=1.0, help="逻辑回归正则强度倒数")
        if name == "struct-train":
            p.add_argument("--C", dest="struct_c", type=float, default=1.0,
                           help="逻辑回归头的正则强度倒数(默认 1.0)")
            p.add_argument("--seed", dest="struct_seed", type=int, default=42,
                           help="头训练随机种子(默认 42)")
            p.add_argument("--qwen-model", default=None,
                           help="覆盖冻结主干: 0.5b / 1.5b / 完整 model id")
        if name == "inject-train":
            p.add_argument("--out", default=None,
                           help="指标输出路径(默认 outputs/metrics/inject.json)")
            p.add_argument("--train-per-family", dest="train_per_family",
                           type=int, default=40,
                           help="每个 seen 载荷家族造多少条正样本(默认 40)")
            p.add_argument("--n-eval", dest="n_eval", type=int, default=100,
                           help="seen/unseen 每侧评测条数(默认 100)")
            p.add_argument("--C", dest="inject_c", type=float, default=1.0,
                           help="注入头 LR 正则强度倒数(默认 1.0)")
            p.add_argument("--seed", dest="inject_seed", type=int, default=0,
                           help="注入头数据/训练种子(默认 0)")
            p.add_argument("--qwen-model", default=None,
                           help="覆盖冻结主干: 0.5b / 1.5b / 完整 model id")
            p.add_argument("--full", action="store_true",
                           help="全量 1k~2k 训练模板(CPU 约 20~40 分钟)")
        if name == "conformal":
            p.add_argument("--alpha", type=float, default=0.01, dest="alpha",
                           help="目标 FPR(默认 0.01)，同时是 conformal 的 α")
        if name == "ranker":
            p.add_argument("--rank-C", dest="rank_c", type=float, default=1.0,
                           help="线性分类器正则强度倒数(默认 1.0)")
            p.add_argument("--rank-topk", dest="rank_topk", type=int, default=40,
                           help="算 edit 富集的 TopK(默认 40)")
            p.add_argument("--rank-variants", dest="rank_variants", default="",
                           help="逗号分隔的变体子集(默认全部 6 个)")
        if name == "attack-study":
            p.add_argument("--qwen-model", default=None,
                           help="回译/判官用的 Qwen 版本: 0.5b / 1.5b / 完整 id")
            p.add_argument("--method", default="zero_width",
                           choices=("zero_width", "homoglyph", "fullwidth", "mix",
                                    "backtrans", "gcg"),
                           help="强攻击方法(默认 zero_width)")
            p.add_argument("--sample", type=int, default=0,
                           help="只取多少个正样本(0=全部；回译/GCG 建议 20)")
            p.add_argument("--with-judge", dest="with_judge", action="store_true",
                           help="额外用 qwen_logprob 判官打一遍(看语义层是否也掉)")
            p.add_argument("--attack-seed", dest="attack_seed", type=int, default=42,
                           help="强攻击采样/扰动种子(默认 42)")
        if name == "redteam":
            p.add_argument("--rt-each", type=int, default=5,
                           help="每种注入载荷造几条文本")
        if name == "factcheck":
            p.add_argument("--fact-model", dest="fact_model", default="Qwen/Qwen2.5-0.5B-Instruct",
                           help="做数字先验的因果 LM（默认 0.5B，CPU 可跑）")
            p.add_argument("--fact-limit", dest="fact_limit", type=int, default=400,
                           help="每个家族抽样条数(0=全量；全量约 20-25 分钟)")
            p.add_argument("--fact-seed", dest="fact_seed", type=int, default=42)
        if name == "feedback":
            p.add_argument("--db", default="outputs/ingest/review.sqlite3",
                           help="人审判决库（review 产出）")
            p.add_argument("--student", default=None,
                           help="学生模型 pkl；给了就用它在 dev 干净集上重算阈值（与线上 score 同尺度，推荐）")
            p.add_argument("--detector", default="sup_tfidf",
                           help="没给 --student 时的兜底：用 dev 分数缓存里同名检测器")
            p.add_argument("--mode", default="pooled", choices=("pooled", "replace"),
                           help="影子阈值模式：并进干净校准集 / 直接用误报样本")
            p.add_argument("--min-support", dest="min_support", type=int, default=30,
                           help="告警所需最小样本量")
            p.add_argument("--corpus-dir", dest="corpus_dir", default="outputs/feedback",
                           help="难负样本/确认正例的导出目录")
            p.add_argument("--report", default="docs/运维日报.md",
                           help="日报 markdown 输出路径")
        if name == "gate":
            p.add_argument("--text", default=None, help="单条文本（给了就走 in-process）")
            p.add_argument("--scene", default="prompt",
                           choices=("prompt", "retrieval", "tool_output"),
                           help="场景：prompt/检索结果/工具输出（tool_output 最严）")
            p.add_argument("--file", dest="gfile", default=None,
                           help="逐行读文件并对每行 guard（留痕 CSV / JSONL）")
            p.add_argument("--http", action="store_true", help="起 HTTP 服务 POST /guard")
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8010)
            p.add_argument("--no-scorer", dest="no_scorer", action="store_true",
                           help="不装配语料打分器（只跑规则层，快；无数据时自动退化）")
            p.add_argument("--threshold", dest="gthreshold", type=float, default=None,
                           help="显式给出 watch 分界阈值")
            p.add_argument("--block-mult", dest="block_mult", type=float, default=2.0,
                           help="block 阈值 = watch 阈值 × 该倍数")
        if name == "review":
            p.add_argument("--out-dir", dest="out_dir", default="outputs/ingest",
                           help="ingest 产物目录（读 quarantine/ 与 review/）")
            p.add_argument("--sample-pass", dest="sample_pass", type=int, default=0,
                           help="从放行流(PASS)按确定性哈希抽 N 条进队列——估计漏报率用")
            p.add_argument("--db", default=None,
                           help="判决库路径（默认 <out-dir>/review.sqlite3）")
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8077)
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--student", default=None, help="学生模型 pkl 路径")
        if name == "ingest":
            p.add_argument("--calibrate", action=argparse.BooleanOptionalAction,
                           default=True,
                           help="本地打分器启动时标定 1%% 阈值（默认开；--bundle 模式读包内阈值）")
            p.add_argument("--inbox", default="data/inbox",
                           help="投递目录：新落地的 txt/csv/jsonl 会被自动处理")
            p.add_argument("--file", dest="files", action="append", default=[],
                           help="显式指定单个文件（可重复），给了就不扫 --inbox")
            p.add_argument("--pattern", default="*.txt,*.csv,*.tsv,*.jsonl,*.json",
                           help="投递目录的文件通配（逗号分隔）")
            p.add_argument("--format", dest="fmt", default="auto",
                           choices=["auto", "txt", "csv", "jsonl"], help="输入格式")
            p.add_argument("--column", default="text", help="csv/jsonl 的文本列名")
            p.add_argument("--limit", type=int, default=0, help="每个文件最多取多少行(0=全部)")
            p.add_argument("--watch-band", dest="watch_band", type=float, default=0.05,
                           help="WATCH 带宽度：阈值下方这一区间进人审队列(0=只分两档)")
            p.add_argument("--threshold", type=float, default=None,
                           help="覆盖门禁阈值（默认取 scorer.threshold）")
            p.add_argument("--bundle", default=None,
                           help="用已签名的 .pkg 打分（部署形态；默认用本地拟合态）")
            p.add_argument("--verify-ed25519", dest="verify_ed25519", default=None,
                           help="验证 .pkg 的 ed25519 公钥（配合 --bundle）")
            p.add_argument("--sign-key", dest="sign_key", default=None,
                           help="验证 .pkg 的 HMAC 密钥（老包/轻量模式）")
            p.add_argument("--sign-key-file", dest="sign_key_file", default=None,
                           help="从文件读取 HMAC 密钥")
            p.add_argument("--ledger-dir", dest="ledger_dir", default="outputs/ingest",
                           help="判决台账(sqlite)目录")
            p.add_argument("--out-dir", dest="out_dir", default="outputs/ingest",
                           help="隔离/人审/放行 jsonl 的输出目录")
            p.add_argument("--watch", action="store_true",
                           help="常驻轮询投递目录（默认只跑一轮就退出）")
            p.add_argument("--poll", type=float, default=10.0, help="轮询间隔秒")
            p.add_argument("--max-files", dest="max_files", type=int, default=0,
                           help="单轮最多处理多少个文件(0=不限)")
            p.add_argument("--no-explain", dest="no_explain", action="store_true",
                           help="不记录各视图分数（省体积）")
        if name == "scan":
            p.add_argument("--input", required=True,
                           help="待扫描文件：txt(一行一句) / csv / tsv / jsonl")
            p.add_argument("--format", default="auto",
                           choices=["auto", "txt", "csv", "jsonl"],
                           help="输入格式（默认按扩展名推断）")
            p.add_argument("--column", default="text",
                           help="csv/jsonl 中承载文本的列名（默认 text）")
            p.add_argument("--threshold", type=float, default=None,
                           help="显式拦截阈值（默认用训练侧标定的）")
            p.add_argument("--judge-topk", type=int, default=0,
                           help="对分数最高 N 条送 LLM 判官复核（0=关闭，0.8~1.4s/条）")
            p.add_argument("--limit", type=int, default=0,
                           help="只扫前 N 条（0=全量，调试用）")
            p.add_argument("--out-prefix", default=None,
                           help="输出文件前缀（默认 outputs/scan/<输入文件名>）")
            p.add_argument("--no-html", action="store_true", help="不生成 HTML 报告")
        if name == "keygen":
            p.add_argument("--out-dir", dest="keygen_dir", default="outputs/keys",
                           help="密钥输出目录(默认 outputs/keys)")
            p.add_argument("--name", default="author", help="密钥名(默认 author)")
        if name == "deploy":
            p.add_argument("--calibrate", action=argparse.BooleanOptionalAction,
                           default=True,
                           help="在 dev 干净样本上标定 1%% 阈值写进 manifest（默认开）")
            p.add_argument("--out", default=None,
                           help=".pkg 输出路径（默认 outputs/deploy/poisonguard-<域>.pkg）")
            p.add_argument("--student", default=None,
                           help="学生模型 pkl 路径（默认自动发现该域）")
            p.add_argument("--threshold", type=float, default=None,
                           help="打包时覆盖学生阈值（默认训练侧标定）")
            p.add_argument("--domain", default="",
                           help="processed 目录后缀(如 afqmc/iflytek)；空=默认 v3")
            p.add_argument("--quantize", default="none",
                           choices=["none", "f16", "i8"],
                           help="kNN 参考向量量化（none/f16/i8），压缩发布物体积")
            p.add_argument("--sign-ed25519", dest="sign_ed25519", default=None,
                           help="用 ed25519 作者私钥签名（keygen 生成的 .key）")
            p.add_argument("--sign-key", default=None,
                           help="HMAC 签名密钥（也可用环境变量 POISONGUARD_PKG_KEY）")
            p.add_argument("--sign-key-file", default=None,
                           help="从文件读签名密钥（避免密钥进 shell 历史）")
        if name == "run":
            p.add_argument("--bundle", required=True,
                           help=".pkg 文件路径（deploy 产物）")
            p.add_argument("--input", required=True,
                           help="待扫描文件：txt(一行一句) / csv / tsv / jsonl")
            p.add_argument("--format", default="auto",
                           choices=["auto", "txt", "csv", "jsonl"])
            p.add_argument("--column", default="text",
                           help="csv/jsonl 中承载文本的列名（默认 text）")
            p.add_argument("--limit", type=int, default=0,
                           help="只扫前 N 条（0=全量）")
            p.add_argument("--judge-topk", type=int, default=0,
                           help="对分数最高 N 条送 LLM 判官复核（0=关闭）")
            p.add_argument("--out-prefix", default=None,
                           help="输出文件前缀（默认 outputs/scan/<输入文件名>）")
            p.add_argument("--no-html", action="store_true", help="不生成 HTML 报告")
            p.add_argument("--domain", default="",
                           help="processed 目录后缀(如 afqmc/iflytek)；空=默认 v3")
            p.add_argument("--verify-ed25519", dest="verify_ed25519", default=None,
                           help="用作者公钥(.pub)验证 ed25519 签名")
            p.add_argument("--sign-key", default=None,
                           help="校验 HMAC 签名的密钥（也可用环境变量 POISONGUARD_PKG_KEY）")
            p.add_argument("--sign-key-file", default=None,
                           help="从文件读校验密钥")
            p.add_argument("--require-signature", action="store_true",
                           help="包未签名则拒绝加载")
        if name == "hardneg":
            p.add_argument("--n-hard", type=int, default=400, help="硬负样本条数")
        if name == "all":
            p.add_argument("--boot", type=int, default=300, help="Bootstrap 次数")
            p.add_argument("--ablation-n", type=int, default=1000)
            p.add_argument("--rates", type=float, nargs="+",
                           default=[0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30])
            p.add_argument("--with-judge", action="store_true", help="额外跑 Qwen 判官(慢)")
            p.add_argument("--limit", type=int, default=40)
            p.add_argument("--low-q", type=float, default=0.80)
            p.add_argument("--high-q", type=float, default=0.995)
        if name == "download":
            p.add_argument("--encoder", default=None, choices=["crom", "bge", "qwen3"],
                           help="要下载的句向量编码器")
            p.add_argument("--qwen", default=None, help="1.5b / 0.5b / 完整 model id")
            p.add_argument("--corpus", default=None,
                           choices=["tnews", "iflytek", "ocnli", "cmnli", "afqmc"],
                           help="下载真实中文语料(CLUE 子集)作为干净侧文本。"
                                "tnews=新闻标题 iflytek=App描述 ocnli/cmnli=常识推理句对 afqmc=金融问答句对")
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = build_parser()
    args = ap.parse_args(argv)
    from . import log as _log

    _log.setup(level=getattr(args, "log_level", None) or "INFO",
               json_mode=bool(getattr(args, "log_json", False)))
    for k, v in ARG_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    cfg = load_config(args.config)
    global RUN_SUFFIX
    if getattr(args, "clean_source", None) == "tnews":
        # 换真实干净语料（CLUE/tnews 新闻标题）：用来量化"玩具语料"贡献了多少虚高
        cfg.data.clean_source = "tnews"
        rfile = (getattr(args, "real_file", None) or "").strip()
        if rfile:
            cfg.data.real_file = rfile
            tag_part = Path(cfg.data.real_file).stem.replace("real_", "", 1)
            cfg.data.processed_dir = cfg.data.processed_dir + "_" + tag_part
            RUN_SUFFIX = (RUN_SUFFIX or "") + "_" + tag_part
            print(f"[config] 真实语料文件 => {cfg.data.real_file}")
        else:
            cfg.data.processed_dir = cfg.data.processed_dir + "_real"
            RUN_SUFFIX = (RUN_SUFFIX or "") + "_real"
        print("[config] 干净语料切换为真实语料(CLUE)")
    if getattr(args, "style_holdout", False):
        # 严格口径：模板留出 + 独立的数据切分目录 + 产物加后缀
        cfg.attack.style_holdout = True
        cfg.data.processed_dir = cfg.data.processed_dir + "_holdout"
        RUN_SUFFIX = RUN_SUFFIX + "_holdout"
        print("[config] 已开启模板留出(style holdout)严格口径")

    dom = (getattr(args, "domain", "") or "").strip()
    if dom:
        # deploy/run 显式指定已训练的领域(如 afqmc/iflytek)；
        # 若 processed_dir 还不是 processed_<dom>，则切换到该目录
        base = cfg.data.processed_dir
        if not base.endswith("_" + dom):
            cfg.data.processed_dir = base + "_" + dom
            if not (RUN_SUFFIX or "").endswith("_" + dom):
                RUN_SUFFIX = (RUN_SUFFIX or "") + "_" + dom
            print(f"[config] 领域目录 => {cfg.data.processed_dir}")
    tag = (getattr(args, "run_tag", None) or "").strip()
    if tag:
        # 口径并存：v2 的 *_real.json 不动，v3 单独落 *_real_v3.json
        cfg.data.processed_dir = cfg.data.processed_dir + "_" + tag
        RUN_SUFFIX = (RUN_SUFFIX or "") + "_" + tag
        print(f"[config] 实验标签 {tag} → {cfg.data.processed_dir}")
    if getattr(args, "edit", False):
        # 内容级投毒：真实新闻句子上改一处事实（年份/涨跌/否定/单位），
        # 与 v2 的模板改写不同，它保留真实语料的全部风格信号。
        # 注意：不要把 "edit" 追加进 cfg.attack.families —— data.build_dataset 里
        # 有独立的 edit 分支(它需要真实语料池和不同的采样桶)，追加会导致生成两遍。
        cfg.attack.edit_family = True
        print("[config] 已启用 edit 攻击家族(真实语料最小事实篡改)")
    if int(getattr(args, "hardneg", 0) or 0) > 0:
        # 硬负样本进主评测集：阈值校准和主指标都要面对"同样像传闻的无害新闻"
        cfg.attack.n_hardneg = int(args.hardneg)
        print(f"[config] 硬负样本 {cfg.attack.n_hardneg} 条将混入 dev/test 主评测集")
    if getattr(args, "judge_scoring", None):
        cfg.qwen.scoring = args.judge_scoring
    if getattr(args, "judge_hardened", False):
        cfg.qwen.hardened = True
        print("[config] 判官 prompt 启用注入加固(分隔符+不可信数据声明)")
    if getattr(args, "judge_sanitize", False):
        cfg.qwen.sanitize = True
        print("[config] 判官启用确定性脱敏(sanitize)")
    if args.encoder:
        cfg.encoder.name = args.encoder
    qm = getattr(args, "qwen_model", None)
    if qm:
        # 0.5b 在本机 CPU 上能跑；1.5b 需要更多时间/显存
        cfg.qwen.model_id = {"0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
                             "1.5b": "Qwen/Qwen2.5-1.5B-Instruct"}.get(qm, qm)
        print(f"[config] 判官模型覆盖为 {cfg.qwen.model_id}")
    if args.target_fpr is not None:
        cfg.detector.target_fpr = args.target_fpr
    if args.device:
        cfg.encoder.device = args.device

    t0 = time.time()
    if args.cmd == "data":
        cmd_data(cfg, args)
    elif args.cmd == "detect":
        cmd_detect(cfg, args)
    elif args.cmd == "victim":
        cmd_victim(cfg, args)
    elif args.cmd == "judge":
        cmd_judge(cfg, args)
    elif args.cmd == "ablation":
        cmd_ablation(cfg, args)
    elif args.cmd == "audit":
        from .audit import lexical_audit

        lexical_audit(cfg, _metrics_path(cfg, "lexical_audit.json"))
    elif args.cmd == "hardneg":
        from .hardneg import hard_negative_probe

        hard_negative_probe(cfg, n=getattr(args, "n_hard", 400),
                            out_path=_metrics_path(cfg, "hardneg.json"))
    elif args.cmd == "redteam":
        from .redteam import run_redteam

        run_redteam(cfg, out_path=_metrics_path(cfg, "redteam.json"),
                    n_each=int(getattr(args, "rt_each", 5) or 5))
    elif args.cmd == "distill":
        cmd_distill(cfg, args)
    elif args.cmd == "struct-train":
        from .struct import run_struct

        run_struct(cfg,
                   out_path=_metrics_path(cfg, "struct.json"),
                   C=float(getattr(args, "struct_c", 1.0) or 1.0),
                   seed=int(getattr(args, "struct_seed", 42) or 42))
    elif args.cmd == "inject-train":
        from .inject import run_inject

        run_inject(cfg,
                   out_path=getattr(args, "out", None),
                   train_per_family=int(getattr(args, "train_per_family", 40) or 40),
                   n_eval=int(getattr(args, "n_eval", 100) or 100),
                   C=float(getattr(args, "inject_c", 1.0) or 1.0),
                   seed=int(getattr(args, "inject_seed", 0) or 0),
                   full=bool(getattr(args, "full", False)))
    elif args.cmd == "conformal":
        from .conformal import run_conformal

        run_conformal(cfg, out_path=_metrics_path(cfg, "conformal.json"),
                      alpha=float(getattr(args, "alpha", 0.01) or 0.01))
    elif args.cmd == "ranker":
        from .ranker import VARIANTS, run_ranker

        _vs = tuple(v.strip() for v in str(getattr(args, "rank_variants", "") or "").split(",")
                    if v.strip()) or VARIANTS
        run_ranker(cfg, out_path=_metrics_path(cfg, "ranker.json"), variants=_vs,
                   C=float(getattr(args, "rank_c", 1.0) or 1.0),
                   seed=0,
                   topk=int(getattr(args, "rank_topk", 40) or 40))
    elif args.cmd == "attack-study":
        from .strongattack import run_attack_study

        _m = getattr(args, "method", "zero_width")
        run_attack_study(cfg, method=_m,
                         sample=int(getattr(args, "sample", 0) or 0),
                         with_judge=bool(getattr(args, "with_judge", False)),
                         seed=int(getattr(args, "attack_seed", 42) or 42),
                         out_path=_metrics_path(cfg, f"strongattack_{_m}.json"))
    elif args.cmd == "factcheck":
        from .factcheck import run_factcheck

        run_factcheck(cfg, out_path=_metrics_path(cfg, "factcheck.json"),
                      model_id=getattr(args, "fact_model", None),
                      limit=int(getattr(args, "fact_limit", 0) or 0),
                      seed=int(getattr(args, "fact_seed", 42) or 42))
    elif args.cmd == "feedback":
        from .data import load_splits
        from .feedback import run_feedback

        clean_scores, score_source = _feedback_clean_scores(cfg, args)
        if score_source and score_source != "student":
            print(f"[feedback] ⚠ 阈值尺度是 {score_source}，但线上 score 通常来自部署打分器"
                  "（学生模型）——尺度可能不可比，建议加 --student <pkl>")

        res = run_feedback(
            getattr(args, "db", "outputs/ingest/review.sqlite3"),
            out_path=_metrics_path(cfg, "feedback.json"),
            clean_scores=clean_scores,
            score_source=score_source,
            target_fpr=float(getattr(args, "target_fpr", None) or cfg.detector.target_fpr),
            mode=getattr(args, "mode", "pooled"),
            min_support=int(getattr(args, "min_support", 30) or 30),
            corpus_dir=getattr(args, "corpus_dir", "outputs/feedback"),
            report_path=getattr(args, "report", "docs/运维日报.md"),
        )
        r = res["fp_rate"]
        print(f"[feedback] 已判 {res['n_decided']} 条：误报 {r['n_fp']}/{r['n_blocked']} "
              f"= {r['fp_rate']:.2%}  告警={res['alarm']['status']}")
        print(f"[feedback] {res['alarm']['message']}")
    elif args.cmd == "mcp":
        from .mcp import serve_stdio
        import sys as _sys
        print("[mcp] 启动 stdio JSON-RPC 服务（tools: guard_text / guard_batch / "
              "review_queue / record_decision / ingest_file）",
              file=_sys.stderr, flush=True)
        return serve_stdio()
    elif args.cmd == "gate":
        from .gate import Gate, GateConfig, guard_texts, serve_http

        import poisonguard.gate as _g

        gate_cfg = GateConfig.from_cfg(
            cfg,
            threshold=getattr(args, "gthreshold", None),
            block_mult=float(getattr(args, "block_mult", 2.0) or 2.0),
            cache_dir=str(cfg.encoder.cache_dir or "outputs/cache"),
        )
        if getattr(args, "http", False):
            gate = Gate(cfg, gate_cfg)
            serve_http(gate, host=getattr(args, "host", "127.0.0.1"),
                       port=int(getattr(args, "port", 8010) or 8010))
            raise SystemExit(0)

        text = getattr(args, "text", None)
        gfile = getattr(args, "gfile", None)
        scene = getattr(args, "scene", "prompt")
        gate = Gate(cfg, gate_cfg)
        if getattr(args, "no_scorer", False):
            gate._no_scorer = True
        if text:
            r = gate.guard(text, scene=scene)
            print(_g.json.dumps(r.as_dict(text=True), ensure_ascii=False, indent=2))
        elif gfile:
            import csv
            import datetime as _dt
            rows = [ln.rstrip("\n") for ln in Path(gfile).read_text(
                encoding="utf-8", errors="replace").splitlines() if ln.strip()]
            results = [gate.guard(t, scene=scene).as_dict(text=False)
                       for t in rows]
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = Path("outputs") / "gate" / f"gate_{stamp}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["text", "verdict", "score", "threshold",
                            "rules_hit", "scene", "budget_ms"])
                for t, r in zip(rows, results):
                    w.writerow([t, r["verdict"], r["score"], r["threshold"],
                                "|".join(r["rules_hit"]), r["scene"], r["budget_ms"]])
            print(f"[gate] {len(rows)} 条 -> {out}")
            from collections import Counter
            print("[gate] 分布:", dict(Counter(r["verdict"] for r in results)))
        else:
            print("用法：gate --text 文本 | gate --file lines.txt | gate --http")
            raise SystemExit(1)
    elif args.cmd == "review":
        from .review import create_review_app

        import uvicorn

        out_dir = getattr(args, "out_dir", "outputs/ingest")
        app = create_review_app(out_dir, getattr(args, "db", None),
                                sample_pass=int(getattr(args, "sample_pass", 0) or 0))
        host = getattr(args, "host", "127.0.0.1")
        port = int(getattr(args, "port", 8077))
        print(f"[review] http://{host}:{port}  队列={out_dir}（无鉴权，别暴露公网）")
        uvicorn.run(app, host=host, port=port, log_level="warning")
    elif args.cmd == "serve":
        cmd_serve(cfg, args)
    elif args.cmd == "ingest":
        from .ingest_cmd import cmd_ingest

        cmd_ingest(cfg, args)
    elif args.cmd == "scan":
        from .scan import cmd_scan

        cmd_scan(cfg, args)
    elif args.cmd == "keygen":
        from .deploy import generate_ed25519_keypair

        priv, pub = generate_ed25519_keypair(
            getattr(args, "keygen_dir", "outputs/keys") or "outputs/keys",
            getattr(args, "name", "author") or "author")
        print(f"[keygen] 私钥(仅作者持有, 0600): {priv}")
        print(f"[keygen] 公钥(可分发, 部署方验证用): {pub}")
        print("[keygen] 打包: deploy --sign-ed25519 <私钥>   验证: run --verify-ed25519 <公钥>")
    elif args.cmd == "deploy":
        from .deploy import cmd_deploy

        cmd_deploy(cfg, args)
    elif args.cmd == "run":
        from .deploy import cmd_run

        cmd_run(cfg, args)
    elif args.cmd == "doctor":
        cmd_doctor(cfg, args)
    elif args.cmd == "report":
        cmd_report(cfg, args)
    elif args.cmd == "download":
        cmd_download(cfg, args)
    elif args.cmd == "all":
        cmd_data(cfg, args)
        cmd_detect(cfg, args)
        try:
            from .audit import lexical_audit

            lexical_audit(cfg, _metrics_path(cfg, "lexical_audit.json"))
        except Exception as e:
            print(f"[audit] 失败(已跳过): {type(e).__name__}: {e}")
        try:
            cmd_victim(cfg, args)
        except Exception as e:  # 行为层失败不应阻塞主实验
            print(f"[victim] 失败(已跳过): {type(e).__name__}: {e}")
        try:
            cmd_ablation(cfg, args)
        except Exception as e:
            print(f"[ablation] 失败(已跳过): {type(e).__name__}: {e}")
        if getattr(args, "judge", "auto") == "auto" and getattr(args, "with_judge", False):
            try:
                cmd_judge(cfg, args)
            except Exception as e:
                print(f"[judge] 失败(已跳过): {type(e).__name__}: {e}")
        cmd_report(cfg, args)
    print(f"\n[done] {args.cmd} 用时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
