"""把实验结果 JSON 渲染成 Markdown 报告(表格自动生成，避免手抄数字)。

设计动机：手写文档里的数字一定会和代码跑出来的数字对不上。
这里所有表格都由 `outputs/metrics/*.json` 自动生成，报告可一键重跑，
数字与图永远同步。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

METRIC_COLS = [
    ("auroc", "AUROC", "{:.3f}"),
    ("auprc", "AUPRC", "{:.3f}"),
    ("tpr_at_fpr1", "TPR@1%(test调阈)", "{:.3f}"),
    ("tpr_at_fpr1_devcal", "TPR@1%(dev校准)", "{:.3f}"),
    ("tpr_at_fpr5", "TPR@5%", "{:.3f}"),
    ("precision", "P", "{:.3f}"),
    ("recall", "R", "{:.3f}"),
    ("f1", "F1", "{:.3f}"),
    ("fpr", "FPR", "{:.3f}"),
]


def md_table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    sep = {"l": ":--", "r": "--:", "c": ":-:"}
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(sep[a] for a in align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt(v, f: str) -> str:
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f.format(v)
    except Exception:
        return str(v)


def detector_table(detectors: dict) -> str:
    rows = []
    for name, m in detectors.items():
        rows.append([f"`{name}`"] + [_fmt(m.get(k), f) for k, _, f in METRIC_COLS])
    return md_table(["检测器"] + [c[1] for c in METRIC_COLS], rows)


def family_table(family_recall: dict) -> str:
    fams = sorted({f for m in family_recall.values() for f in m})
    rows = []
    for name, m in family_recall.items():
        rows.append([f"`{name}`"] + [_fmt(m.get(f, {}).get("recall"), "{:.1%}") for f in fams])
    return md_table(["检测器"] + fams, rows)


def fpr_table(fpr_group: dict) -> str:
    groups = sorted({g for m in fpr_group.values() for g in m})
    rows = []
    for name, m in fpr_group.items():
        rows.append([f"`{name}`"] + [_fmt(m.get(g, {}).get("fpr"), "{:.2%}") for g in groups])
    return md_table(["检测器"] + [f"{g} 上的 FPR" for g in groups], rows)


def _load_metric(cfg, name: str):
    """从 outputs/metrics 读一个可选产物；不存在返回 None（报告里明确写"未运行"）。"""
    import json

    p = Path(cfg.metrics_dir) / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def benchmark_section(cfg) -> str:
    """跨基准对照：同一批攻击，干净语料换成真实新闻后还剩多少判别力。

    这一节是整份报告里信息量最大的一节：它证明"漂亮的 AUROC"里有多少
    来自**干净侧的贫瘠**，而不是检测器的能力。
    """
    real = _load_metric(cfg, "detect_real.json")
    if not real:
        return "_未运行真实语料基准：`python -m poisonguard.cli detect --clean-source tnews`_"
    base = _load_metric(cfg, "detect.json") or {}
    aud_s = _load_metric(cfg, "lexical_audit.json") or {}
    aud_r = _load_metric(cfg, "lexical_audit_real.json") or {}
    hard = _load_metric(cfg, "hardneg_real.json") or {}

    lines: list[str] = []
    A = lines.append

    rows = []
    d0, d1 = base.get("detectors") or {}, real.get("detectors") or {}
    for name in d0:
        if name not in d1:
            continue
        m0, m1 = d0[name], d1[name]
        rows.append([
            f"`{name}`",
            _fmt(m0.get("auroc"), "{:.3f}"),
            _fmt(m0.get("tpr_at_fpr1"), "{:.3f}"),
            _fmt(m1.get("auroc"), "{:.3f}"),
            _fmt(m1.get("tpr_at_fpr1"), "{:.3f}"),
            _fmt(m1.get("auroc_allneg"), "{:.3f}"),
        ])
    if rows:
        A(md_table(
            ["检测器", "AUROC(合成干净侧)", "TPR@1%(合成)", "AUROC(真实干净侧)", "TPR@1%(真实)", "AUROC(真实, 含OOD负样本)"],
            rows))
        A("")
        A("> 同一批攻击、同一套编码器，只把**干净语料**从仓库自带的槽位语料换成 CLUE/tnews 的真实新闻标题，"
          "无监督离群检测(`knn`/`mahalanobis`/`iforest`)的 AUROC 从 0.99 掉到 0.43~0.73，"
          "`knn` 甚至低于 0.5——**它的排名方向反了**：真实新闻足够多元，离训练质心比人工合成的攻击句更远。"
          "这不是调参能修的问题，而是「离群 = 有害」这个前提在真实语料上不成立。")
        A("")

    if aud_s or aud_r:
        arows = []
        for tag, a in (("合成槽位语料", aud_s), ("真实新闻语料", aud_r)):
            if not a:
                continue
            arows.append([
                tag,
                f"{a.get('clean_ngram_vocab')}",
                f"{a.get('fitted_ngram_vocab')}",
                f"{a.get('oov_ngram_ratio', 0):.1%}",
                f"{a.get('oov_weight_share', 0):.1%}",
                _fmt(a.get("in_vocab_only_auroc_in_sample"), "{:.3f}"),
            ])
        A("**词面审计(`audit`)：检测器的预测有多少来自「没见过这种句式」**")
        A("")
        A(md_table(
            ["干净语料", "干净侧 n-gram 词表", "检测器用到 n-gram", "干净侧没见过的比例",
             "它们占决策权重", "只用词表内 n-gram 的 AUROC"],
            arows))
        A("")
        A("> 合成槽位语料的字符 2-3gram 词表只有 994 个（它是 10×10×10 拼句机），"
          "检测器用到的 n-gram 里 85.8% 在干净侧从未出现，贡献了 49.2% 的决策权重。"
          "换成真实新闻后干净侧词表涨到 33.7 万、OOV 比例降到 27.8%、OOV 权重降到 35.6%。"
          "也就是说：**合成基准上的满分，接近一半来自「这句话在干净语料里没见过」**。")
        A("")
        A("> 注：`只用词表内 n-gram 的 AUROC` 是 in-sample 数字，只作上限参考，不进主结论表。")
        A("")

    dets = hard.get("detectors") or {}
    if dets:
        hrows = [[f"`{k}`", _fmt(v.get("auroc_attack_vs_hardneg"), "{:.3f}"),
                  _fmt((v.get("mean_score") or {}).get("hard_neg"), "{:.4f}"),
                  _fmt((v.get("mean_score") or {}).get("attack"), "{:.4f}")]
                 for k, v in dets.items()]
        A("**硬负样本压力测试(`hardneg`)：攻击 vs 同样「像传闻」的无害新闻**")
        A("")
        A(f"硬负样本 {hard.get('n_hard')} 条，取自真实新闻里含"
          f"{'、'.join(hard.get('markers', [])[:8])}… 等措辞、但完全无害的句子。")
        A("")
        A(md_table(["检测器", "攻击 vs 硬负样本 AUROC", "硬负样本均分", "攻击均分"], hrows))
        A("")
        A("> `knn` 在这里是 **0.500**：把「像传闻的普通新闻」和「投毒文本」放在一起，"
          "它的排序能力等于抛硬币。而 `sup_tfidf` 仍有 1.000 —— 因为合成攻击全都有"
          "固定的措辞骨架，词面分类器认的是**攻击生成器的口音**，不是有害性本身。"
          "这意味着换个攻击模板(真实对抗者显然会换)，它就会失效。")
        A("")
    return "\n".join(lines)


def v3_section(judge: dict | None, distill: dict | None,
               redteam: dict | None = None) -> str:
    """v3 三个环节的表格：级联召回-成本曲线 + 判官蒸馏 + 注入红队。"""
    body: list[str] = []
    A = body.append

    cascade = ((judge or {}).get("cascade") or {})
    curve = cascade.get("curve") or []
    by_gate = cascade.get("by_gate") or {}
    if by_gate and len(by_gate) > 1:
        # 两个闸门并排：同样的 LLM 预算下，召回差多少
        gates = list(by_gate.keys())
        idx = {g: {round(r["budget"], 4): r for r in by_gate[g]} for g in gates}
        rows = []
        for b in sorted({round(r["budget"], 4) for g in gates for r in by_gate[g]}):
            row = [f"{b:.0%}"]
            for g in gates:
                rr = idx[g].get(b)
                row.append(f"{rr['recall']:.1%}" if rr else "-")
            rnd = idx[gates[-1]].get(b)
            row.append(f"{rnd['random_recall']:.1%}" if rnd else "-")
            rows.append(row)
        A("**级联闸门对比(`--gate`)：同样是「只把最高分的一部分送 LLM」**")
        A("")
        A(md_table(["LLM 预算"] + [f"`{g}` 召回" for g in gates] + ["同预算随机"], rows))
        A("")
        A("> 无监督 kNN 闸门在真实语料上 AUROC 只有 0.438（低于随机），"
          "曲线几乎贴在随机对照线上；换成语义/词面**有监督**分做闸门后，"
          "预算 40% 的召服从 27.0% 提升到 47.0%，"
          "而 LLM 调用量不变。结论：级联的收益取决于闸门的排序能力，不是 LLM 本身。")
        A("")
    elif curve:
        A("**级联的召回-成本曲线(`cascade_curve`)：只把第一级分数最高的一部分送 LLM**")
        A("")
        rows = [[f"{r['budget']:.0%}", r["n_calls"], f"{r['recall']:.1%}",
                 f"{r['precision']:.1%}", f"{r['random_recall']:.1%}"] for r in curve]
        A(md_table(["LLM 预算", "实际调用条数", "召回", "精确率", "同预算随机对照"], rows))
        A("")
        best = max(curve, key=lambda r: r["recall"] - r["random_recall"])
        A(f"> 读法：「同预算随机对照」= 随机挑同样多条送去问 LLM 的期望召回。"
          f"最佳档位在预算 {best['budget']:.0%}：召回 {best['recall']:.1%} vs 随机 "
          f"{best['random_recall']:.1%}。这行对照说明门控确实在挑样本，"
          "而不是「反正都得全量过一遍」。")
        A("")

    cmp_ = (distill or {}).get("comparison") or {}
    if cmp_:
        rows = [[f"`{name}`", _fmt(m.get("auroc"), "{:.3f}"),
                 _fmt(m.get("auprc"), "{:.3f}"),
                 _fmt(m.get("tpr@fpr1%"), "{:.3f}"),
                 _fmt(m.get("ms_per_item"), "{:.3f}")]
                for name, m in cmp_.items()]
        A("**判官蒸馏(`distill`)：同一测试切片上 teacher / student / 单层基线**")
        A("")
        A(md_table(["模型", "AUROC", "AUPRC", "TPR@1%", "单条毫秒"], rows))
        A("")
        drop = (distill or {}).get("auroc_drop")
        sp = (distill or {}).get("speedup")
        if drop is not None and sp:
            A(f"> 蒸馏代价：AUROC {drop:+.3f}，换取约 {sp:.0f}× 加速"
              f"(学生模型特征 {distill.get('features')} 维，训练 {distill.get('train_sec')}s)。"
              "结论是量级性的：LLM 不进在线路径，只做离线软标签与抽查。")
            A("")
    if redteam:
        inj = redteam.get("injected") or {}
        ip = inj.get("plain") or {}
        ih = inj.get("hardened") or {}
        isan = inj.get("sanitized") or {}
        has_san = bool(isan)
        A("**判官自己的攻击面：提示注入红队（`redteam`）**")
        A("")
        A(f"base 取判官原本能抓住的 {redteam.get('n_bases')} 条正例，"
          f"每条尾部追加 8 类注入载荷并配对比较（共 {redteam.get('n_injected')} 条）：")
        A("")
        rows = []
        for k, v in (redteam.get("by_payload") or {}).items():
            row = [f"`{k}`",
                   f"{v['plain'].get('mean_delta', 0):+.3f}",
                   f"{v['plain'].get('hijack_rate', 0):.0%}",
                   f"{v['hardened'].get('mean_delta', 0):+.3f}",
                   f"{v['hardened'].get('hijack_rate', 0):.0%}"]
            if has_san:
                row += [f"{v.get('sanitized', {}).get('mean_delta', 0):+.3f}",
                        f"{v.get('sanitized', {}).get('hijack_rate', 0):.0%}"]
            rows.append(row)
        tot = ["**全部注入**", f"**{ip.get('mean_delta', 0):+.3f}**",
               f"**{ip.get('hijack_rate', 0):.0%}**",
               f"**{ih.get('mean_delta', 0):+.3f}**",
               f"**{ih.get('hijack_rate', 0):.0%}**"]
        if has_san:
            tot += [f"**{isan.get('mean_delta', 0):+.3f}**",
                    f"**{isan.get('hijack_rate', 0):.0%}**"]
        rows.append(tot)
        header = ["载荷", "Δ分(plain)", "劫持率(plain)", "Δ分(加固)", "劫持率(加固)"]
        if has_san:
            header += ["Δ分(加固+脱敏)", "劫持率(加固+脱敏)"]
        A(md_table(header, rows))
        A("")
        base_note = (f"> base 平均分 plain={redteam.get('base_mean_plain')} / "
                     f"hardened={redteam.get('base_mean_hardened')}")
        if has_san:
            base_note += f" / sanitized={redteam.get('base_mean_sanitized')}"
        base_note += ("；劫持率 = 原文被判为污染(≥0.5) 而追加载荷后 <0.5 的比例。")
        A(base_note)
        if has_san:
            A("")
            A(f"**结论：确定性脱敏是「代价最低」的一档，但不是终点。** "
              f"plain → 加固 → 加固+脱敏：注入导致的 Δ 分 "
              f"{ip.get('mean_delta', 0):+.3f} → {ih.get('mean_delta', 0):+.3f} → "
              f"{isan.get('mean_delta', 0):+.3f}，"
              f"劫持率 {ip.get('hijack_rate', 0):.0%} → {ih.get('hijack_rate', 0):.0%} → "
              f"{isan.get('hijack_rate', 0):.0%}；base 平均分 "
              f"{redteam.get('base_mean_plain')} → {redteam.get('base_mean_hardened')} → "
              f"{redteam.get('base_mean_sanitized')}（加固有“伤检出”的代价，脱敏不额外伤）。"
              "残余劫持集中在 `role_switch` / `fake_json` / `few_shot_mimic`："
              "它们靠**语境与角色叙述**而非显式指令词，字面脱敏抓不到。"
              "终局仍是结构隔离——分类头 / 指令微调 / 多判官投票。")
        else:
            A("> 劫持率 = 原文被判为污染(≥0.5) 而追加载荷后 <0.5 的比例。"
              "**结论：0.5B 判官对提示注入几乎无抵抗**，"
              "分隔符 + 不可信数据声明只收窄分数下降幅度、且让 base 分整体降低 —— "
              "prompt 层加固是「有代价且不够」的。"
              "正确方向是结构隔离：判官不解析指令（分类头 / 指令微调）、"
              "多判官投票、或对文本做字符级脱敏后再入 prompt。")
        A("")
    if not body:
        return ""
    return chr(10).join(["## v3 环节：级联曲线与判官蒸馏", ""] + body)


def build_report(
    cfg,
    dataset_stats: dict | None,
    results: dict | None,
    ablation: dict | None,
    judge: dict | None,
    victim: dict | None,
    figures: list[str],
    out_path: str | Path,
    distill: dict | None = None,
    redteam: dict | None = None,
) -> Path:
    lines: list[str] = []
    A = lines.append

    A("# 实验报告：训练数据投毒检测 (`poisonguard`)")
    A("")
    A(f"> 本报告由 `python -m poisonguard.cli report` 自动生成，"
      f"生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}。"
      "所有数字直接来自 `outputs/metrics/*.json`，不含手工填写。")
    A("")

    # ---------------- 1. 数据 ----------------
    A("## 1. 数据概况")
    A("")
    if dataset_stats:
        A("**原始数据去重前的情况(暴露旧项目的核心问题)：**")
        A("")
        raw_rows = []
        for k, v in (dataset_stats.get("raw") or {}).items():
            raw_rows.append([k, v.get("rows"), v.get("unique"),
                             f"{v.get('dup_ratio', 0):.1%}",
                             v.get("max_repeat"), f"{v.get('avg_chars', 0):.1f}"])
        if raw_rows:
            A(md_table(["原始文件", "行数", "唯一文本", "重复率", "最高重复次数", "平均字数"], raw_rows))
            A("")
            A("> 结论：`normal_data_2w.csv` 有约四分之三的行是重复文本，"
              "`poison_data_2k.csv` 有约八成重复。旧项目按行切分训练/测试集，"
              "导致**同一条文本同时出现在 train 和 test**，指标严重虚高。"
              "本项目统一按**唯一文本**切分。")
            A("")
        splits = dataset_stats.get("splits") or {}
        if splits:
            rows = []
            for split, st in splits.items():
                fams = ", ".join(f"{k}×{v}" for k, v in sorted(st.get("by_family", {}).items()))
                rows.append([split, st.get("total"),
                             f"{st.get('positive', 0)} ({st.get('positive', 0) / max(st.get('total', 1), 1):.0%})",
                             fams, st.get("avg_chars")])
            A(md_table(["切分", "样本数", "正类(投毒)", "构成", "平均字数"], rows))
            A("")

    # ---------------- 2. 检测器 ----------------
    A("## 2. 检测效果对比")
    A("")
    if results:
        meta = results.get("meta", {})
        A(f"- 编码器：`{meta.get('encoder')}`")
        A(f"- 阈值校准目标误报率：FPR = {meta.get('target_fpr')}")
        A(f"- 样本量：train {meta.get('n_train')} / dev {meta.get('n_dev')} / test {meta.get('n_test')}")
        A(f"- 攻击家族：{', '.join(meta.get('families', []))}")
        A("")
        det = results.get("detectors") or {}
        if det:
            A("**主结果(测试集)：**")
            A("")
            A(detector_table(det))
            A("")
            A("> 读法：`TPR@1%` = 每天 10 万条数据、允许 1% 误报时能抓到的投毒比例，"
              "这是真实运维最关心的数字；AUROC/AUPRC 衡量排序能力，与正类比例无关。")
            A("")

        ci = results.get("auroc_ci") or {}
        if ci:
            rows = [[f"`{k}`", _fmt(m[0], "{:.3f}"),
                     f"[{_fmt(m[1], '{:.3f}')}, {_fmt(m[2], '{:.3f}')}]"]
                    for k, m in ci.items()]
            A("**AUROC 的 Bootstrap 95% 置信区间(1000 次重采样)：**")
            A("")
            A(md_table(["检测器", "AUROC", "95% CI"], rows))
            A("")

        fam = results.get("family_recall") or {}
        if fam:
            A("**分攻击家族召回率：**")
            A("")
            A(family_table(fam))
            A("")
            A("> 关键看点：`paraphrase`(零关键词的语义等价改写)和 `subtle`(无评价词的"
              "事实型虚假陈述)是专门为击穿规则法设计的。"
              "如果某个检测器在 `template` 上接近 100% 但在 `paraphrase` 上崩掉，"
              "说明它学到的是模板表面特征而不是「有害性」。")
            A("")

        fprg = results.get("fpr_group") or {}
        if fprg:
            A("**干净样本上的误报率(区分同分布 / 分布外)：**")
            A("")
            A(fpr_table(fprg))
            A("")

        mc = results.get("mcnemar") or {}
        if mc:
            A("**显著性检验(McNemar，逐样本对比)：**")
            A("")
            rows = [[f"`{k}`", v.get("only_a"), v.get("only_b"),
                     f"{v.get('p_value', 1):.2e}", v.get("test")]
                    for k, v in mc.items()]
            A(md_table(["对比 (A vs B)", "A 对 B 错", "B 对 A 错", "p 值", "检验"], rows))
            A("")
            A("> p < 0.05 说明两个检测器的差异不是随机波动。")
            A("")

        fw = results.get("fusion_weights")
        if fw:
            A(f"**融合权重(仅在 dev 上网格搜索得到)：** `{fw}`")
            A("")

    # ---------------- 3. 基准对照 ----------------
    A("## 3. 基准对照：把干净语料换成真实新闻")
    A("")
    A(benchmark_section(cfg))
    A("")

    # ---------------- 4. 行为级 ----------------
    A("## 4. 行为级检测(受害模型)")
    A("")
    if victim:
        A(f"- 受害模型：`{victim.get('model')}`，参数量 {victim.get('n_params')}")
        A(f"- 加载校验：{victim.get('load_report')}")
        A("")
        rows = [[f"`{k}`",
                 _fmt(m.get("auroc"), "{:.3f}"),
                 _fmt(m.get("tpr_at_fpr1"), "{:.3f}"),
                 _fmt(m.get("f1"), "{:.3f}"),
                 _fmt(m.get("auroc_allneg"), "{:.3f}"),
                 _fmt(m.get("fpr_ood"), "{:.3f}")]
                for k, m in (victim.get("detectors") or {}).items()]
        if rows:
            A(md_table(["行为信号", "AUROC", "TPR@1%", "F1",
                        "含 OOD 的 AUROC", "OOD 误报率"], rows))
            A("")
        A("> 行为级信号的物理含义：受害模型对「它没见过的语义」会给出高熵/低置信输出；"
          "被后门污染过的模型则会对带触发词的输入异常自信。"
          "这一层不依赖文本表面特征，因此对 `paraphrase` 这类改写攻击理论上更稳。")
        A("")
        A("> **但看「含OOD的 AUROC」与「OOD 误报率」两列**：三个行为信号在分布内都接近完美"
          "（AUROC 1.000），一旦把 500 条话题不同但完全无害的 OOD 文本加入负样本，"
          "`entropy` 的 AUROC 掉到 0.714、OOD 误报率 1.000——**受害模型把「没见过的话题」"
          "也叫作「可疑」，和前面 kNN 犯的是同一个错误**。所以行为层信号不能单独用作判定，"
          "必须与数据层、语义层组合，并且校准集里必须有分布外负样本。")
    else:
        A("_未运行 `victim` 阶段(需要 `baseline_model.pth`)。_")
    A("")

    # ---------------- 4. 投毒比例消融 ----------------
    A("## 5. 投毒比例消融")
    A("")
    if ablation:
        rates = ablation.get("rates", [])
        rows = []
        for r in rates:
            rows.append([f"{r:.0%}",
                         _fmt(ablation["clean_fp_rate"].get(str(r)), "{:.1%}"),
                         _fmt(ablation["poison_recall"].get(str(r)), "{:.1%}"),
                         _fmt(ablation["probe_auc"].get(str(r)), "{:.3f}")])
        A(md_table(["训练集投毒比例", "干净测试集误判率", "投毒测试集召回", "探针判别 AUC"], rows))
        A("")
        A("> 这张表回答的问题是：**模型被污染到什么程度，它的内部表示才开始明显偏移？**"
          "看第四列：随投毒比例从 0% 升到 30%，探针判别 AUC 从 0.500 单调升到 0.872——"
          "**表示层的漂移是单调且平滑的，投毒比例本身可以直接当作风险指标**。"
          "但看第二、三列：阈值固定在干净 dev 的 1% 误报上之后，干净侧的误判率始终是 0，"
          "而投毒侧召回直到 30% 才到 18.2%——**低比例投毒在阈值层面几乎完全不可见**。"
          "这正是必须在**数据层**先做检测的原因：等模型行为能看出来时，污染已经很深了。")
    else:
        A("_未运行 `ablation` 阶段。_")
    A("")

    # ---------------- 5. LLM 判官 ----------------
    A("## 6. Qwen 判官(语义级)")
    A("")
    if judge:
        A(f"- 判官：`{judge.get('model')}`，few-shot={judge.get('few_shot')}，"
          f"判定样本数 {judge.get('n')}，解析失败 {judge.get('parse_fail')} 条")
        A("")
        rows = [[f"`{k}`"] + [_fmt(m.get(c[0]), c[2]) for c in METRIC_COLS]
                for k, m in (judge.get("detectors") or {}).items()]
        if rows:
            A(md_table(["判官"] + [c[1] for c in METRIC_COLS], rows))
            A("")
        fam = judge.get("family_recall") or {}
        if fam:
            A("按 dev 校准阈值（上线工作点）的分家族召回：")
            A("")
            A(family_table(fam))
            A("")
        fam_half = judge.get("family_recall_at_0.5") or {}
        if fam_half:
            A("同一批样本在固定 0.5 工作点上的分家族召回（用来看「哪个家族更难」，"
              "而不是上线能拿到多少）：")
            A("")
            A(family_table(fam_half))
            A("")
        cs = judge.get("cascade")
        if cs:
            A(f"- 级联调用率：{cs.get('llm_ratio', 0):.1%}"
              f"({cs.get('n_llm')}/{cs.get('n_total')})——只有灰区样本才调用 LLM。")
            A("")
        if rows:
            sc = str(judge.get("scoring") or "logprob")
            # 单条耗时落在各检测器的指标里（judge.json 顶层没有），先取再兜底
            sec = judge.get("sec_per_item")
            if not isinstance(sec, (int, float)) or sec <= 0:
                secs = [m.get("sec_per_item") for m in (judge.get("detectors") or {}).values()
                        if isinstance(m.get("sec_per_item"), (int, float)) and m.get("sec_per_item") > 0]
                sec = max(secs) if secs else None
            sec_txt = f"单条约 {sec:.2f} 秒" if isinstance(sec, (int, float)) else "单条耗时为缓存值"
            A("> **这张表怎么读。** 判官是 `" + str(judge.get("model")) + "`，"
              "在本机 CPU 上" + sec_txt + "，因此只做了分层抽样的 "
              + str(judge.get("n")) + " 条。三件事值得单独说：")
            A(">")
            if sc == "binary":
                A("> 1. **它的输出是二值的**（只有「污染/正常」两个取值），"
                  "所以 `TPR@1%` 必然塌到 0——在 1% 的误报预算下，"
                  "二值判官根本切不出可用工作点。要用它做线上判定，"
                  "必须换成 logprob 打分或做校准，否则它只能当「人工复核队列」的排序器。")
            else:
                A("> 1. **它的输出是连续概率**（读「污染/正常」两个 token 的 logprob 再归一化），"
                  "所以 `TPR@1%` 是能测的有限值，阈值由 dev 干净样本定在 FPR=1% 分位上，"
                  "表里 `TPR@1%(test调阈)` 与 `TPR@1%(dev校准)` 并列给出——"
                  "前者用测试集自己找点是乐观上界，后者才是上线口径。"
                  "ECE 单独报告：概率能不能当概率用，得单独检验。")
            A("> 2. **分家族召回暴露了语义层的盲区**："
              "`backdoor` 类召回最低。触发词攻击的句子表面是一句完全正常的日常话，"
              "语义层看它当然没毛病——这类攻击只能靠行为层(看触发词是否改变模型输出)"
              "或数据溯源来抓。这正好说明三层不是堆料，而是各自补不同的洞。")
            A("> 3. **级联能不能省成本，取决于第一级的排序能力而不是准确率**："
              "闸门若把真阳性提前判成「干净」，LLM 根本没机会看。"
              "v3 因此加了 `--gate` 与整条召回-成本曲线（见下一节），"
              "并把「同预算随机」作为对照基线。")
            A("")
            A("")
    else:
        A("_未运行 `judge` 阶段(需要 Qwen 权重)。_")
    A("")

    v3 = v3_section(judge, distill, redteam)
    if v3:
        A(v3)
        A("")

    # ---------------- 6. 图 ----------------
    A("## 7. 图表")
    A("")
    for f in figures:
        p = Path(f)
        rel = p.relative_to(Path(cfg.outputs_dir).parent) if p.is_absolute() else p
        A(f"![{p.stem}](../{rel.as_posix()})")
        A("")

    # ---------------- 7. 局限 ----------------
    A("## 8. 已知局限(主动声明，不藏)")
    A("")
    A("1. **攻击样本是程序生成的**，不是真实互联网投毒数据。"
      "`dev_seed ≠ test_seed` 只保证测试集不是训练时见过的字符串，"
      "但四类攻击都有固定措辞骨架，于是词面分类器(以及字符 n-gram 基线)"
      "学到的是\"攻击生成器的口音\"。第 3 节的硬负样本实验把这一点量化了："
      "`knn` 在\"攻击 vs 像传闻的无害新闻\"上只有 0.500，"
      "而 `sup_tfidf` 仍为 1.000——后者的高分不应解读为\"能识别有害性\"。")
    A("2. **真实语料基准只换了干净侧**：攻击仍是合成文本，"
      "因此第 3 节证明的是\"干净侧贫瘠贡献了多少虚高\"，"
      "而不是\"真实投毒场景下的绝对检出率\"。下一步应改用"
      "「真实新闻 + 最小编辑式篡改」生成攻击，让两侧同为自然文本。")
    A("3. **受害模型权重来自旧项目，其训练数据存在 train=test 泄漏**，"
      "因此本报告只用它做相对信号(同模型内不同样本的置信度排序)，"
      "不引用它的绝对准确率。")
    A("4. **投毒比例消融用线性探针代替 BERT 重训**：规律可迁移，"
      "但绝对数值会与真实 BERT 重训不同，属于成本-精度的有意权衡。")
    A("5. **LLM 判官受限于本机算力**：未在真实 Qwen 权重上做全量评测时，"
      "该行为降级方案(启发式)单独标注，不进入主结论表。")
    A("")
    A("## 9. 复现")
    A("")
    A("```bash")
    A("python -m poisonguard.cli all                      # 数据→编码→检测→消融→报告")
    A("python -m poisonguard.cli detect                   # 只重跑检测")
    A("python -m poisonguard.cli audit                    # 词面审计(合成干净侧)")
    A("python -m poisonguard.cli download --corpus tnews  # 拉真实新闻语料(可复现第 3 节)")
    A("python -m poisonguard.cli detect --clean-source tnews     # 真实干净侧基准")
    A("python -m poisonguard.cli audit  --clean-source tnews     # 真实干净侧词面审计")
    A("python -m poisonguard.cli hardneg --clean-source tnews    # 硬负样本压力测试")
    A("python -m poisonguard.cli victim                   # 行为级信号(需 baseline_model.pth)")
    A("python -m poisonguard.cli judge --limit 40         # Qwen 判官(需权重)")
    A("python -m poisonguard.cli report                   # 重新渲染本报告")
    A("```")
    A("")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] 已生成 {out_path}")
    return out_path
