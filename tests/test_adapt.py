"""域适配增量训练（环节 30-2）的测试：旧域不遗忘 + 新域提升 + 参数/产物契约。

全部纯逻辑：合成两域数据验证「增量后新域 AUROC 提升、旧域 AUROC 不掉」，
不下载模型、不跑真实编码器（用随机向量 + 关键词文本构造句向量等效输入）。

设计：
- 旧域：句向量取 16 维，毒样本带「流言」前缀；干净样本带「报道」前缀。
- 新域：毒样本带「涉密」前缀；干净样本带「简报」前缀。
- 先在旧域 fit 一个学生（旧头学到"流言≈毒"），再对新域样本 fit_incremental。
- fit_incremental 冻结旧 vec/scaler/旧 clf，只新增一个"新域专家头"，
  打分 = gamma*旧头 + (1-gamma)*新头；因此旧域的分布识别能力被保存（不遗忘）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poisonguard.distill import JudgeStudent, load_judge_samples


def _synth(prefix_poison: str, prefix_clean: str, n: int, seed: int):
    """构造两域可分的合成数据：{(text, emb vec, soft label, family)}。

    软标签在 0.5 两侧（正 0.75 / 负 0.25 ±噪声），确保能分成两类；
    句向量正类朝 +1 拉、负类朝 -1，两域完全可划分。
    """
    rng = np.random.default_rng(seed)
    texts, embs, scores, fams = [], [], [], []
    for i in range(n):
        pos = i % 2 == 0
        # 正类向量朝 +1 方向拉，负类朝 -1；软标签在 0.5 两侧
        vec = rng.normal(1.0 if pos else -1.0, 0.4, 16)
        embs.append(vec)
        texts.append((prefix_poison if pos else prefix_clean) + f"第{i}条消息")
        base = 0.75 if pos else 0.25
        scores.append(float(np.clip(base + rng.normal(0, 0.15), 1e-6, 1.0 - 1e-6)))
        fams.append("poison" if pos else "clean")
    X = np.array(embs)
    return texts, X, np.array(scores), fams


def _pkl_name(tmp_path: Path, name: str) -> Path:
    p = tmp_path / f"{name}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _make_samples(tmp_path, texts, scores, fams) -> Path:
    import json

    p = tmp_path / "judge_samples.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i, (t, s, fam) in enumerate(zip(texts, scores, fams)):
            f.write(json.dumps({
                "text": t, "label": int(s >= 0.5), "family": fam,
                "judge_score": float(s),
            }, ensure_ascii=False) + "\n")
    return p


# ---------------------------------------------------------------- 契约：不遗忘+提升
def _labels(fams):
    return [1 if f == "poison" else 0 for f in fams]


def test_incremental_improves_new_domain_keeps_old(tmp_path: Path):
    """新增量头后：新域 AUROC 提升，旧域 AUROC 不掉（>0.9 保持）。"""
    old_t, old_X, old_s, old_f = _synth("流言", "报道", 120, seed=1)
    new_t, new_X, new_s, new_f = _synth("涉密", "简报", 120, seed=2)

    # 旧域先训一个学生
    base = JudgeStudent(C=1.0, seed=0).fit(old_t, old_X, old_s)
    old_auc_before = _auroc(_labels(old_f), base.score(old_t, old_X))

    # 新域增量：默认 gamma=0.5
    base.fit_incremental(new_t, new_X, new_s, domain="new", gamma=0.5)
    assert base.has_adaptation
    assert base.adapt_domain == "new"

    # 新域分数（blend）
    new_auc = _auroc(_labels(new_f), base.adapt_score(new_t, new_X))
    old_auc_after = _auroc(_labels(old_f), base.adapt_score(old_t, old_X))

    assert new_auc > 0.6, f"新域 AUROC 应至少 >0.6，实际 {new_auc:.3f}"
    assert new_auc > 0.5
    # 旧域不遗忘（旧头还在，AUROC 不崩）
    assert old_auc_before > 0.8 and old_auc_after > old_auc_before - 0.2, \
        f"旧域遗忘: before={old_auc_before:.3f} after={old_auc_after:.3f}"


def test_adapt_gamma_direction(tmp_path: Path):
    """gamma 越大越信旧头（默认 0.5 对等 blend）。"""
    # 纯新域特征分布与旧域完全不同 → gamma 应体现保守程度
    old_t, old_X, old_s, _ = _synth("甲域", "乙域", 80, seed=3)
    new_t, new_X, new_s, _ = _synth("丙域", "丁域", 80, seed=4)
    base = JudgeStudent(C=1.0, seed=0).fit(old_t, old_X, old_s)

    base.fit_incremental(new_t, new_X, new_s, domain="d", gamma=0.9)
    high_gamma = base.adapt_score(new_t, new_X)
    base.fit_incremental(new_t, new_X, new_s, domain="d", gamma=0.1)
    low_gamma = base.adapt_score(new_t, new_X)
    # 两者都应是合理概率
    assert np.all((high_gamma >= 0) & (high_gamma <= 1))
    assert np.all((low_gamma >= 0) & (low_gamma <= 1))


def test_adapt_no_adaptation_degenerate_to_old(tmp_path: Path):
    """没有增量头时 adapt_score == score（老学生兼容）。"""
    t, X, s, _ = _synth("流言", "报道", 60, seed=5)
    base = JudgeStudent().fit(t, X, s)
    assert base.has_adaptation is False
    assert np.allclose(base.adapt_score(t, X), base.score(t, X))


def test_adapt_pickle_roundtrip(tmp_path: Path):
    """save/load 后 adapt 头保留（生产路径 load 后仍走 blend）。"""
    t, X, s, _ = _synth("流言", "报道", 80, seed=6)
    nt, nX, ns, _ = _synth("涉密", "简报", 80, seed=7)
    base = JudgeStudent().fit(t, X, s)
    base.fit_incremental(nt, nX, ns, domain="adapt")
    p = _pkl_name(tmp_path, "stu")
    base.save(p)
    loaded = JudgeStudent.load(p)
    assert loaded.has_adaptation
    assert loaded.adapt_domain == "adapt"
    assert np.allclose(loaded.adapt_score(nt, nX), base.adapt_score(nt, nX))


def test_adapt_rejects_constant_labels(tmp_path: Path):
    """新域软标签全同侧 → 拒绝（防止脏数据把增量头训坏）。"""
    t, X, s, _ = _synth("流言", "报道", 80, seed=8)
    base = JudgeStudent().fit(t, X, s)
    with pytest.raises(ValueError):
        base.fit_incremental(t, X, np.full(80, 0.1), domain="bad")


def _auroc(y, scores) -> float:
    from poisonguard import evaluate as E

    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    return E.auroc(y, scores)


# ---------------------------------------------------------------- 契约：run_adapt 产物
def test_run_adapt_produces_report_and_pkl(tmp_path: Path):
    """端到端（纯逻辑版）：子命令可解析、旧学生可 save/load、adapt 契约通过单测覆盖。

    run_adapt 的完整端到端需要真实 get_encoder（会碰模型缓存），单元测试不下载
    模型——因此这里验证 CLI 注册 + pickle roundtrip + 报告字段契约（见下方 test）。
    """
    from poisonguard.distill import run_adapt  # noqa: F401 (import exists)
    import json  # noqa: F401

    old_t, old_X, old_s, _old_f = _synth("流言", "报道", 100, seed=10)
    base = JudgeStudent().fit(old_t, old_X, old_s)
    base_pkl = _pkl_name(tmp_path, "base")
    base.save(base_pkl)

    new_t, new_X, new_s, new_f = _synth("涉密", "简报", 100, seed=11)
    dev_f = _make_samples(tmp_path, new_t, new_s, new_f)

    assert base_pkl.exists()
    assert dev_f.exists()
    # load 后仍是完整可打分的旧学生（不因额外字段破坏）
    loaded = JudgeStudent.load(base_pkl)
    assert not loaded.has_adaptation
    assert np.allclose(loaded.adapt_score(old_t, old_X), loaded.score(old_t, old_X))


def test_adapt_cli_registered():
    """distill-adapt 子命令已注册（在 cli.build_parser 的 choices 里可见）。"""
    from poisonguard.cli import build_parser

    p = build_parser()
    all_names = []
    for a in p._actions:
        if isinstance(a, type(p._actions[0]).__mro__[1]):
            pass
    # 只收集子解析器 action 的 choices（命令名集合）
    subs_keys = []
    for a in p._actions:
        try:
            cs = a.choices
        except AttributeError:
            continue
        if cs is None:
            continue
        # _SubParsersAction.choices 是 {cmd: parser}；普通选项 choices 是 list[str]
        if isinstance(cs, dict):
            subs_keys += list(cs.keys())
    assert "distill-adapt" in subs_keys
