"""跨检测器对照：ragcheck vs sup_tfidf / struct_qwen / qwen_logprob（编辑投毒立足点）。

把四个检测器在"编辑投毒"上的同类指标摆到同一张表，论证 ragcheck 在这个
攻击立足点的增量价值。同一切片 = real_v3 test 的 edit 家族（正）vs normal（干净负侧）。

数据来源（全部为本分支既有产物，不重跑任何模型）：
- outputs/metrics/test_scores_real_v3.npz  → sup_tfidf 逐条分数（edit=500 / normal=2844）
- outputs/metrics/struct_real_v3.json      → struct_qwen 的 family_recall（edit n=500 recall=0.296）+ AUROC 0.9266
- outputs/metrics/ragcheck_real_v3.json    → by_family（edit/normal mean_score，limit=400 抽样封顶）+ edit_norm_auroc 0.7691 + 整体 tpr@1%FPR 0.1085
- outputs/metrics/judge_real_v3.json       → qwen_logprob（抽样 320，edit 仅 40 条，作参考行）

sup_tfidf 的逐条分数在手，按同一口径重算：阈值在 normal 干净侧标定到 FPR=1%，
再算 edit 召回 / AUROC / (edit_mean - normal_mean) 分离度；其余方法用原报告值，
并显式标注口径差异（抽样 vs 全量、主掩码 vs edit|normal 切片）。

输出：outputs/metrics/compare_ragcheck.json + compare_ragcheck.md（含中文结论）。
用法：python scripts/compare_ragcheck.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs" / "metrics"
OUT_JSON = MET / "compare_ragcheck.json"
OUT_MD = MET / "compare_ragcheck.md"


def load_json(name: str) -> dict:
    return json.loads((MET / name).read_text(encoding="utf-8"))


def tpr_at_fpr1(normal: np.ndarray, pos: np.ndarray) -> float:
    """阈值在干净侧(normal)标定到 FPR=1%，正侧(pos)上的召回。"""
    thr = float(np.quantile(normal, 0.99))
    return {"fpr_threshold": round(thr, 6),
            "edit_recall_at_fpr1": round(float(np.mean(pos > thr)), 4)}


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, score)), 4)


def main() -> None:
    print("[1] 读取并确认各文件键结构 …")

    # ---- sup_tfidf：逐条分数来自 npz ----
    npz = np.load(MET / "test_scores_real_v3.npz", allow_pickle=True)
    print("  npz keys:", list(npz.keys()))
    y, fam, st = npz["y"], npz["families"], npz["sup_tfidf"]
    print("  npz family counts:", {str(k): int(v) for k, v in zip(*np.unique(fam, return_counts=True))})

    ed_mask = fam == "edit"
    nm_mask = fam == "normal"
    st_edit, st_norm = st[ed_mask], st[nm_mask]
    assert int(ed_mask.sum()) == 500 and int(nm_mask.sum()) == 2844
    # 切片标签必须干净：edit 全为正、normal 全为负，否则指标口径无效
    assert int((y[ed_mask] == 1).sum()) == 500
    assert int((y[nm_mask] == 0).sum()) == 2844

    # ---- struct_qwen：family_recall 结构 {family:{n,recall}} ----
    sr = load_json("struct_real_v3.json")
    print("  struct_real_v3.json keys:", list(sr["detectors"]["struct_qwen"].keys()))
    sfr = sr["detectors"]["struct_qwen"]["family_recall"]
    print("  struct family_recall 键:", list(sfr.keys()), " edit =", sfr["edit"])
    struct_edit = sfr["edit"]
    struct_auroc = sr["detectors"]["struct_qwen"]["auroc"]

    # ---- ragcheck：by_family 结构 {family:{n,mean_score,max_score}} ----
    rr = load_json("ragcheck_real_v3.json")
    print("  ragcheck overall keys:", list(rr["overall"].keys()))
    print("  ragcheck by_family 键:", list(rr["by_family"].keys()))
    r_edit = rr["by_family"]["edit"]
    r_norm = rr["by_family"]["normal"]
    print("  ragcheck edit by_family =", r_edit, " normal =", r_norm)
    rag_edit_mean, rag_norm_mean = r_edit["mean_score"], r_norm["mean_score"]
    rag_edit_n, rag_norm_n = r_edit["n"], r_norm["n"]
    rag_edit_auroc = rr["overall"]["edit_norm_auroc"]
    rag_tpr_fpr1 = rr["overall"]["tpr_at_fpr1"]

    # ---- qwen_logprob：参考行（抽样 320，edit 仅 40 条） ----
    jr = load_json("judge_real_v3.json")
    jq = jr["detectors"]["qwen_logprob"]
    j_edit = jr["family_recall"]["qwen_logprob"]["edit"]
    j_edit_half = jr["family_recall_at_0.5"]["qwen_logprob"]["edit"]
    print("  qwen_logprob auroc=", jq["auroc"], " edit family_recall =", j_edit,
          " recall@0.5 =", j_edit_half)

    # ================= 指标计算 =================
    print("\n[2] 计算指标 …")

    # --- sup_tfidf（逐条在手上，按 FPR=1% 重算） ---
    st_tpr = tpr_at_fpr1(st_norm, st_edit)
    st_auroc = auroc(np.r_[np.zeros(len(st_norm), dtype=int), np.ones(len(st_edit), dtype=int)],
                     np.r_[st_norm, st_edit])
    st_sep = round(float(st_edit.mean() - st_norm.mean()), 4)

    # --- struct_qwen ---
    struct_tpr = struct_edit["recall"]          # dev-clean 校准到 FPR≈1% 的阈值上的 edit 召回
    struct_sep = None                            # 逐条 struct 分数未缓存，无法重算分离度

    # --- ragcheck：除 edit 家族外所有正族分数恒为 0，整体 tpr@1%FPR 即 edit 生效占比 ---
    rag_tpr = rag_tpr_fpr1
    rag_sep = round(rag_edit_mean - rag_norm_mean, 4)

    # --- qwen_logprob（参考行） ---
    j_tpr = j_edit["recall"]                     # dev 校准阈值(≈0.97)上的 edit 召回
    j_tpr_half = j_edit_half["recall"]           # 固定 0.5 工作点
    j_sep = None

    rows = [
        {
            "detector": "sup_tfidf",
            "edit_recall_at_fpr1": st_tpr["edit_recall_at_fpr1"],
            "auroc": st_auroc,
            "edit_mean_minus_normal_mean": st_sep,
            "edit_n": int(ed_mask.sum()), "normal_n": int(nm_mask.sum()),
            "basis": "全量 real_v3 test（retained），逐条分数重算，阈值在 normal 干净侧标定 FPR=1%",
            "edit_mean": round(float(st_edit.mean()), 4),
            "normal_mean": round(float(st_norm.mean()), 4),
        },
        {
            "detector": "struct_qwen",
            "edit_recall_at_fpr1": struct_tpr,
            "auroc": struct_auroc,
            "edit_mean_minus_normal_mean": struct_sep,
            "edit_n": struct_edit["n"], "normal_n": None,
            "basis": ("原报告值；AUROC=0.9266 为主评价掩码口径(全部正族 vs normal+hardneg)，"
                      "edit 召回=0.296 为 dev-clean 校准到 FPR≈1% 阈值上；逐条分数未缓存，分离度 N/A"),
        },
        {
            "detector": "ragcheck",
            "edit_recall_at_fpr1": rag_tpr,
            "auroc": rag_edit_auroc,
            "edit_mean_minus_normal_mean": rag_sep,
            "edit_n": rag_edit_n, "normal_n": rag_norm_n,
            "basis": ("原报告值；edit_norm_auroc=0.7691 即 edit vs normal 切片；"
                      "整体 tpr@1%FPR=0.1085 —— 除 edit 外其余正族(模板/改写/微改/后门)冲突分恒为 0，"
                      "故该 tpr 即为 edit 家族生效召回；mean 为 limit=400 抽样封顶的均值"),
            "edit_mean": rag_edit_mean, "normal_mean": rag_norm_mean,
        },
        {
            "detector": "qwen_logprob",
            "edit_recall_at_fpr1": j_tpr,
            "auroc": jq["auroc"],
            "edit_mean_minus_normal_mean": j_sep,
            "edit_n": j_edit["n"], "normal_n": None,
            "basis": ("参考行：抽样 320 条中小样本，edit 仅 40 条；auroc=0.6885 为子集口径，"
                      "edit 召回 devcal 阈值=0.0，固定 0.5 工作点=0.525；均值未缓存，分离度 N/A"),
            "edit_recall_at_0p5": j_tpr_half,
        },
    ]

    report = {
        "meta": {
            "purpose": "argument / 编辑投毒立足点上 ragcheck 增量价值的跨检测器同切片对照",
            "slice": "real_v3 test: edit(正) vs normal(干净负侧)",
            "generated_by": "scripts/compare_ragcheck.py",
            "sources": {
                "sup_tfidf_scores": "outputs/metrics/test_scores_real_v3.npz",
                "struct": "outputs/metrics/struct_real_v3.json",
                "ragcheck": "outputs/metrics/ragcheck_real_v3.json",
                "qwen_logprob": "outputs/metrics/judge_real_v3.json",
            },
        },
        "rows": rows,
    }

    # ================= 中文结论 =================
    conclusion = (
        "在'编辑投毒'（对原文事实点的定向改写）这个立足点上，四个检测器各有所长与硬伤：\n"
        "\n"
        "1. sup_tfidf（词面有监督分）—— 全局 AUROC 最高（0.938），但本切片 edit 家族召回只有 0.046"
        "（tnews 合成口径亦仅 0.040），edit_mean-normal_mean 分离度仅 0.035。原因：编辑改写保留原文大部分词面与 tf-idf 结构，"
        "词袋向量几乎不动，靠词面根本区分不开；它是『泛化击杀』，在编辑立足点上接近失效。\n"
        "\n"
        "2. struct_qwen（冻结主干语义表征 + 二分类头）—— edit 召回 0.296、AUROC 0.9266，"
        "在编辑立足点上明显强于 sup_tfidf：它看的是深层语义表征而非词面，能抓住改写后的 "
        "『语义漂移』。弱点是逐条分数没落盘、难做端到端解释，且仍会漏掉 70% 的编辑样本"
        "（语义漂移不够大时和正常文本区分不开）。\n"
        "\n"
        "3. ragcheck（检索一致性，in-KB 锚定）—— edit 召回@1%FPR 0.1085，edit-vs-normal AUROC 0.7691，"
        "edit 冲突分数均值 0.507 对 normal 0.004，分离度 0.50，是四者里编辑/normal 分数带最干净的。"
        "它只对『改数字/事实词导致与参考语料冲突』的样本生效（模板/改写/微改/后门冲突分恒 0），"
        "定位精确但覆盖面窄：只能抓 in-KB 的编辑，事后外推/纯新增事实抓不到；且召回绝对值不高，"
        "单打独斗当主判官会漏。\n"
        "\n"
        "4. qwen_logprob —— 抽样 320 条（edit 仅 40 条）作参考：AUROC 0.6885、devcal 阈值下 edit 召回 0.0，"
        "固定 0.5 工作点 0.525。表现受小样本与 logprob 过度自信（阈值被推到 0.97）拖累，"
        "不宜单独作为编辑投毒的判据。\n"
        "\n"
        "互补组合建议：≪ragcheck 先筛、struct 再判≫。ragcheck 只用模糊检索 + 冲突计数，"
        "单条成本远低于 Qwen 编码（实测 487s / 4000+ 条级），适合做第一道粗筛：把冲突分高的"
        "候选（edit 样本其 mean_score 0.507 远高于 normal 0.004）先拎出来挡住『事实点被篡改』类；"
        "对粗筛漏掉或需要解释的样本，再用 struct_qwen 的语义表征做第二道判定，补上 ragcheck "
        "『in-KB 锚定』之外因改写太像原句、或超出参考语料覆盖面而漏掉的编辑。"
        "词面 sup_tfidf 作为第三层仅用于压低误报，不承担编辑召回。三层级联 ="
        "低成本粗筛（ragcheck：抓 in-KB 编辑）+ 语义精判（struct：抓语义漂移）+ 词面保底。\n"
    )
    report["conclusion_zh"] = conclusion

    # ================= 落盘 =================
    MET.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  写出", OUT_JSON.name)

    # Markdown 表格
    md = [
        "# 跨检测器对照：编辑投毒立足点（ragcheck vs sup_tfidf / struct_qwen / qwen_logprob）",
        "",
        "切片：real_v3 test 的 **edit 家族（正）** vs **normal（干净负侧）**。",
        "分数方向统一为「越高越可疑」。本表所有数字来自既有产物，未重跑模型。",
        "",
        "| 检测器 | edit 召回@1%FPR | AUROC | edit_mean − normal_mean 分离度 | 依据 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        sep = f"{r['edit_mean_minus_normal_mean']:.4f}" if r["edit_mean_minus_normal_mean"] is not None else "N/A"
        md.append(
            f"| {r['detector']} | {r['edit_recall_at_fpr1']} | {r['auroc']:.4f} "
            f"| {sep} | {r['basis']} |"
        )
    md += ["", "## 结论（中文）", "", conclusion]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print("  写出", OUT_MD.name)

    # 终端回显对照表
    print("\n[3] 对照表：")
    print(f"{'detector':<16}{'edit_rec@1%FPR':>16}{'AUROC':>10}{'sep(edit-mean)':>18}")
    for r in rows:
        sep = f"{r['edit_mean_minus_normal_mean']:.4f}" if r["edit_mean_minus_normal_mean"] is not None else "N/A"
        print(f"{r['detector']:<16}{r['edit_recall_at_fpr1']:>16}{r['auroc']:>10.4f}{sep:>18}")
    print("\n结论已写入 compare_ragcheck.md")


if __name__ == "__main__":
    main()
