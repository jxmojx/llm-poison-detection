#!/usr/bin/env python
"""CI 端到端烟测：用**内置 tiny 编码器**把整条链路真的跑一遍。

为什么要有它
------------
单元测试全是纯逻辑（不加载模型），好处是几十秒跑完，坏处是**接线断了也发现不了**：
`data` 写出来的 splits 字段改名、`detect` 的校准口径变了、`deploy` 打包的
manifest 少一个键、`run` 的签名校验接到别的参数上——这些 bug 都不会让
`pytest tests/` 变红。B6 就是这个洞。

为什么是 tiny 编码器
-------------------
真实编码器要下 100M+ 权重，CI 里会慢且受网络影响。`TinyEncoder` 用
blake2b 把字符 2-3gram 有符号哈希到 64 维，纯 numpy、确定性、无下载——
它不追求效果（指标会是 1.0，因为语料本身可分），只保证五段链路真的接通：

    data → detect → scan → deploy → run(离线批扫)

断言的目标也是"接线"而不是"效果"：
- 每段退出码为 0、产物文件存在；
- scan 的行数与输入一致、CSV 有分数列；
- **deploy 打包 → run 加载后的逐条分数与 scan 完全一致**（发布物不改变判定）；
- 签名链路是通的：对密钥能过、错密钥必须失败、`--require-signature` 拦未签名的包。
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / ".ci_smoke"
RUN_TAG = "ci"
KEY = "ci-smoke-key"


def _run(*args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "poisonguard.cli", *args]
    print(f"\n$ {' '.join(cmd[2:])}")
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = (p.stdout or "").strip().splitlines()[-4:]
    for line in tail:
        print(f"  {line}")
    if expect_ok and p.returncode != 0:
        print(p.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"FAIL: {' '.join(args)} rc={p.returncode}")
    return p


def _write_corpus() -> None:
    import random

    import yaml

    rnd = random.Random(0)
    subj = ["某地", "该市", "相关部门", "记者", "专家", "社区", "学校", "医院", "企业", "平台"]
    verb = ["通报", "发布", "宣布", "回应", "提醒", "调查", "公布", "指出"]
    obj = ["高考报名人数", "食品安全检查结果", "交通管制安排", "垃圾分类新规",
           "疫苗接种计划", "养老补贴标准", "城市更新项目", "就业扶持政策",
           "防汛应急预案", "医保报销流程"]

    def sent(i: int) -> str:
        return f"{rnd.choice(subj)}{rnd.choice(verb)}：{rnd.choice(obj)}第{i}期情况说明已公开"

    normal = list(dict.fromkeys(sent(i) for i in range(600)))
    poison = [f"网传：错误医疗知识：只要每天服用某种保健品就无需吃药即可治愈糖尿病（样本{i}）"
              for i in range(300)]
    (WORK / "raw").mkdir(parents=True, exist_ok=True)
    for name, rows in (("normal.csv", normal), ("poison.csv", poison)):
        with open(WORK / "raw" / name, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows([[t] for t in rows])

    for d in ("processed", "outputs/metrics", "outputs/figures", "outputs/cache", "model_cache"):
        (WORK / d).mkdir(parents=True, exist_ok=True)
    cfg = {
        "data": {"raw_dir": str(WORK / "raw"), "processed_dir": str(WORK / "processed"),
                 "normal_file": "normal.csv", "poison_file": "poison.csv",
                 "seed": 42, "train_ratio": 0.6, "dev_ratio": 0.2,
                 "dedupe": True, "max_normal": 600, "clean_source": "synthetic"},
        "attack": {"families": ["template", "paraphrase", "subtle", "backdoor"],
                   "n_per_family": 40, "dev_n_per_family": 15,
                   "dev_seed": 2024, "test_seed": 7,
                   "n_ood_dev": 20, "n_ood_test": 30},
        "encoder": {"name": "tiny", "model_dir": str(WORK / "model_cache/tiny"),
                    "batch_size": 32, "max_length": 64, "device": "cpu",
                    "cache_dir": str(WORK / "outputs/cache"), "normalize": True},
        "detector": {"target_fpr": 0.01, "knn_k": 5, "pca_dim": 8,
                     "keywords": ["低下", "无风险", "治愈", "抵制"],
                     "enable": ["keyword", "centroid", "sup_lr", "sup_tfidf", "fusion"]},
        "outputs_dir": str(WORK / "outputs"),
        "metrics_dir": str(WORK / "outputs/metrics"),
        "figures_dir": str(WORK / "outputs/figures"),
    }
    (WORK / "cfg.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")


def _read_csv_scores(path: Path) -> list[float]:
    with open(path, encoding="utf-8", newline="") as f:
        return [float(r["score"]) for r in csv.DictReader(f)]


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    _write_corpus()
    cfg = str(WORK / "cfg.yaml")
    common = ["--config", cfg, "--run-tag", RUN_TAG]

    # 1) data
    _run("data", *common)
    splits = WORK / f"processed_{RUN_TAG}"
    assert (splits / "test.jsonl").exists(), "data 段没有写出 test.jsonl"

    # 2) detect（tiny 编码器，无下载）
    _run("detect", *common)
    det = json.loads((WORK / "outputs/metrics" / f"detect_{RUN_TAG}.json").read_text(encoding="utf-8"))
    assert "sup_tfidf" in det["detectors"], "detect 产物缺少 sup_tfidf"
    assert "fusion" in det["detectors"], "detect 产物缺少 fusion"

    # 3) scan（人审报告 + 逐条分数）
    demo = ROOT / "data" / "scan_demo_afqmc.txt"
    _run("scan", *common, "--input", str(demo), "--judge-topk", "0")
    scores_csv = ROOT / "outputs" / "scan" / "scan_demo_afqmc.scores.csv"
    report_md = ROOT / "outputs" / "scan" / "scan_demo_afqmc.report.md"
    assert scores_csv.exists() and report_md.exists(), "scan 没有产出报告/CSV"
    scan_scores = _read_csv_scores(scores_csv)
    n_lines = len([ln for ln in demo.read_text(encoding="utf-8").splitlines() if ln.strip()])
    assert len(scan_scores) == n_lines, f"scan 行数 {len(scan_scores)} != 输入 {n_lines}"

    # 4) deploy（带 HMAC 签名）
    _run("deploy", *common, "--sign-key", KEY)
    pkg = ROOT / "outputs" / "deploy" / f"poisonguard-{RUN_TAG}.pkg"
    assert pkg.exists(), f"deploy 没有产出 {pkg}"

    # 5) run：正确密钥 → 分数必须与 scan 完全一致（发布物只是换个进程跑同一套判定）
    _run("run", "--bundle", str(pkg), "--input", str(demo),
         "--judge-topk", "0", "--sign-key", KEY)
    run_scores = _read_csv_scores(scores_csv)
    assert run_scores == scan_scores, "deploy→run 的分数与 scan 不一致：打包改变了判定"

    # 6) 签名链路必须真的拦人
    bad = _run("run", "--bundle", str(pkg), "--input", str(demo),
               "--judge-topk", "0", "--sign-key", "wrong-key", expect_ok=False)
    assert bad.returncode != 0, "错误密钥竟然通过了校验"
    nosig = _run("run", "--bundle", str(pkg), "--input", str(demo),
                 "--judge-topk", "0", expect_ok=False)
    assert nosig.returncode != 0, "未提供密钥竟然加载了已签名包"

    print("\n[ci_smoke] 五段链路 + 签名链路全部通过 ✅")
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
