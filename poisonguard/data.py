"""数据构建：加载 → 去重 → 分层切分 → 注入攻击样本。

关键修正(相对原项目)：
1. **去重**：原 normal_data_2w.csv 20000 行只有 4997 条唯一文本(75% 重复)，
   poison_data_2k.csv 2000 行只有 376 条唯一文本(81% 重复)。
   不去重时 train/test 会共享文本，任何指标都虚高。
2. **切分**：train/dev/test 严格不相交，且按唯一文本切分而非按行。
3. **攻击样本按 split 隔离**：dev 与 test 用不同 seed 生成，测试集是未见过的攻击。
4. **OOD 干净样本**：单独估计"分布外误报率"，避免只在训练分布内报 FPR。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from . import attacks as atk
from .config import Config


@dataclass
class Sample:
    text: str
    label: int          # 1=投毒/有害, 0=干净
    family: str         # normal / ood / template / paraphrase / subtle / backdoor
    split: str          # train / dev / test

    def to_dict(self) -> dict:
        return asdict(self)


def read_texts(path: str | Path) -> list[str]:
    df = pd.read_csv(path, header=None, names=["text"], encoding="utf-8")
    return [str(t).strip() for t in df["text"].tolist() if str(t).strip()]


def _dedupe(texts: list[str], min_chars: int = 6) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        if len(t) < min_chars or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _read_lines(path: str | Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def fetch_real_corpus(out_path: str | Path, subset: str = "tnews", n: int = 20000) -> Path:
    """从 ModelScope 拉 CLUE 的真实中文新闻标题，落成一个 txt（一行一句）。

    为什么不用 HuggingFace：本机到 HF 不可达，ModelScope 可达。
    为什么需要它：仓库自带的 normal_data_2w.csv 是槽位语料（10×10×10 拼接），
    字符 n-gram 词表只有 994 个，任何检测器都能靠"句式没见过"拿到虚高指标。
    """
    out_path = Path(out_path)
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as e:
        raise SystemExit(
            "需要 modelscope 数据集依赖：pip install datasets addict simplejson "
            f"sortedcontainers（原始错误：{e}）"
        ) from e
    ds = MsDataset.load("modelscope/clue", subset_name=subset, split="train",
                        trust_remote_code=True)
    # CLUE 各子集字段名不统一：tnews 用 "sentence"，iflytek 用 "text"，
    # ocnli/cmnli/afqmc 用句子对 "sentence1"/"sentence2"。兼容三者。
    texts: list[str] = []
    for row in ds:
        val = row.get("sentence") or row.get("text")
        if val:
            texts.append(str(val))
            continue
        for key in ("sentence1", "sentence2"):
            v = row.get(key)
            if v:
                texts.append(str(v))
    texts = texts[:n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(texts), encoding="utf-8")
    lens = [len(t) for t in texts]
    print(f"[download] {subset}: {len(texts)} 条 (唯一 {len(set(texts))}, 平均 {sum(lens)/len(lens):.1f} 字) -> {out_path}")
    return out_path


def build_dataset(cfg: Config) -> dict[str, list[Sample]]:
    """返回 {"train": [...], "dev": [...], "test": [...]}。

    两种干净语料来源（`cfg.data.clean_source`）：
    - "synthetic"：仓库自带的 normal_data_2w.csv。它其实是**槽位语料**——
      10 个主题 × 10 个方面 × 10 个领域拼出来的句子，去重后只有 4997 条、
      字符 2-3gram 词表只有 994 个。用它当干净侧，"像不像那三个模板"
      就足以把攻击全抓出来，指标会虚高（我们用 outputs/cache/_oov.py 量化过）。
    - "tnews"：CLUE/tnews 的真实新闻标题（约 2 万条，唯一率 96%）。
      干净侧换成真实文本后，字符 n-gram 空间大两个数量级，
      同一批攻击的检测难度才反映"内容有害性"而不是"句式没见过"。
      此模式下 OOD 反向使用旧的槽位语料——它是**良性但分布外**的文本，
      正好检验"检测器会不会把陌生但无害的文本判成投毒"。
    """
    dcfg = cfg.data
    acfg = cfg.attack
    raw = Path(dcfg.raw_dir)

    if getattr(dcfg, "clean_source", "synthetic") == "tnews":
        real_path = raw / getattr(dcfg, "real_file", "real_tnews.txt")
        if not real_path.exists():
            raise SystemExit(
                f"[data] 缺少真实语料 {real_path}。先跑：python -m poisonguard.cli download --corpus tnews"
            )
        normals = _read_lines(real_path)[: dcfg.max_normal]
        if dcfg.dedupe:
            normals = _dedupe(normals)
        print(f"[data] 干净语料 = 真实新闻(CLUE/tnews) {len(normals)} 条")
    else:
        normals = read_texts(raw / dcfg.normal_file)[: dcfg.max_normal]
        if dcfg.dedupe:
            normals = _dedupe(normals)
    rng = random.Random(dcfg.seed)
    rng.shuffle(normals)

    # ---------------- v3: 硬负样本**先划走**再切分 ----------------
    # 关键坑：train/dev/test 的干净样本都来自同一份语料，事后从语料里"过滤掉
    # 已出现过的句子"必然一条不剩。正确做法是先挑出"含传闻措辞但完全无害"的句子，
    # 从干净池中移出，再按比例切分，最后把它们作为 family="hardneg" 塞回 dev/test。
    n_hard = int(getattr(acfg, "n_hardneg", 0) or 0)
    hard_all: list[str] = []
    if n_hard > 0:
        from .hardneg import RUMOR_MARKERS

        hits = [t for t in normals if any(m in t for m in RUMOR_MARKERS)]
        rng_h = random.Random(dcfg.seed + 21)
        rng_h.shuffle(hits)
        hard_all = hits[:n_hard]
        if len(hard_all) < n_hard:
            print(f"[data] 硬负样本池不足：含传闻措辞的干净句只有 {len(hard_all)}/{n_hard} 条")
        drop = set(hard_all)
        normals = [t for t in normals if t not in drop]
        print(f"[data] 硬负样本 {len(hard_all)} 条已从干净池移出(避免与 normal 重复)")

    n = len(normals)
    n_train = int(n * dcfg.train_ratio)
    n_dev = int(n * dcfg.dev_ratio)
    parts = {
        "train": normals[:n_train],
        "dev": normals[n_train: n_train + n_dev],
        "test": normals[n_train + n_dev:],
    }
    print(
        f"[data] 正常文本 {n} 条(去重后) → "
        f"train {len(parts['train'])} / dev {len(parts['dev'])} / test {len(parts['test'])}"
    )

    out: dict[str, list[Sample]] = {
        "train": [Sample(t, 0, "normal", "train") for t in parts["train"]],
        "dev": [Sample(t, 0, "normal", "dev") for t in parts["dev"]],
        "test": [Sample(t, 0, "normal", "test") for t in parts["test"]],
    }

    # OOD 干净样本：良性但分布外，只进 dev/test
    if getattr(dcfg, "clean_source", "synthetic") == "tnews":
        # 真实语料模式下，"分布外"用旧的槽位语料担任(它与新闻完全不同的句式)
        pool = _dedupe(read_texts(raw / dcfg.normal_file))
        r = random.Random(dcfg.seed + 11)
        r.shuffle(pool)
        need = acfg.n_ood_dev + acfg.n_ood_test
        if len(pool) < need:
            raise SystemExit(f"[data] OOD 池只有 {len(pool)} 条，不够 {need} 条")
        ood_dev, ood_test = pool[: acfg.n_ood_dev], pool[acfg.n_ood_dev: need]
    else:
        ood_dev = atk.make_ood_normal(acfg.n_ood_dev, random.Random(dcfg.seed + 11))
        ood_test = atk.make_ood_normal(acfg.n_ood_test, random.Random(dcfg.seed + 12))
    out["dev"] += [Sample(t, 0, "ood", "dev") for t in ood_dev]
    out["test"] += [Sample(t, 0, "ood", "test") for t in ood_test]

    # 攻击样本：dev / test 用不同 seed **且不同哈希桶**，
    # 保证两边的攻击字符串互不相交(模板空间有限时只换种子会重叠)。
    #
    # 但"字符串不相交"不等于"手法没见过"：两边用的是同一批模板骨架，
    # 字符 n-gram / kNN 只要认骨架就能满分(实测 sup_tfidf AUROC=1.000，
    # 甚至把 OOD 也判成异常)。所以要能量化这部分虚高，就得做**模板留出**：
    # cfg.attack.style_holdout=True 时 dev 只用前半模板、test 只用后半模板。
    clean_pool = ood_test + parts["test"]
    holdout = bool(getattr(acfg, "style_holdout", False))
    dev_atk = atk.build_attacks(
        acfg.families, acfg.dev_n_per_family, acfg.dev_seed, clean_pool,
        bucket=0, n_buckets=2,
        style_fold=0 if holdout else None,
    )
    test_atk = atk.build_attacks(
        acfg.families, acfg.n_per_family, acfg.test_seed, clean_pool,
        bucket=1, n_buckets=2,
        style_fold=1 if holdout else None,
    )
    if holdout:
        print("[data] 模板留出已开启：dev 与 test 使用互不相交的模板骨架")
    out["dev"] += [Sample(s.text, 1, s.family, "dev") for s in dev_atk]
    out["test"] += [Sample(s.text, 1, s.family, "test") for s in test_atk]

    # ---------------- v3: 内容级投毒(真实新闻最小编辑篡改) ----------------
    # 与上面四族的关键区别：句式、长度、用词都和真实新闻一模一样，只改了一个事实。
    # 句向量离群检测对这种样本在原理上失效(kNN 距离≈0)，因此它衡量的是**语义层**。
    real_pool: list[str] = []
    real_path = raw / getattr(dcfg, "real_file", "real_tnews.txt")
    if real_path.exists():
        real_pool = _dedupe(_read_lines(real_path))
    if getattr(acfg, "edit_family", False):
        edit_pool = real_pool or parts["test"]
        dev_edit = atk.build_attacks(["edit"], acfg.dev_n_per_family, acfg.dev_seed,
                                     edit_pool, bucket=0, n_buckets=2)
        test_edit = atk.build_attacks(["edit"], acfg.n_per_family, acfg.test_seed,
                                      edit_pool, bucket=1, n_buckets=2)
        out["dev"] += [Sample(s.text, 1, s.family, "dev") for s in dev_edit]
        out["test"] += [Sample(s.text, 1, s.family, "test") for s in test_edit]
        print(f"[data] 内容级投毒(edit): dev {len(dev_edit)} / test {len(test_edit)} 条"
              f"（真实语料池 {len(edit_pool)} 句）")

    # ---------------- v3: 硬负样本进主评测集 ----------------
    # 之前 hardneg 只是一个独立探针(单独出表)，主指标里根本没有它，等于把最难的一类
    # 负样本留给"额外实验"。现在一半进 dev(参与阈值校准)、一半进 test(参与主指标)。
    if hard_all:
        half = len(hard_all) // 2
        out["dev"] += [Sample(t, 0, "hardneg", "dev") for t in hard_all[:half]]
        out["test"] += [Sample(t, 0, "hardneg", "test") for t in hard_all[half:]]
        print(f"[data] 硬负样本: dev {half} / test {len(hard_all) - half} 条"
              f"（真实新闻里含传闻措辞但完全无害）")

    for split in out:
        out[split] = [s for s in out[split] if s.text.strip()]
    _print_composition(out)
    return out


def _print_composition(splits: dict[str, list[Sample]]) -> None:
    for split, samples in splits.items():
        cnt: dict[str, int] = {}
        for s in samples:
            cnt[s.family] = cnt.get(s.family, 0) + 1
        pos = sum(s.label for s in samples)
        print(
            f"[data] {split}: {len(samples)} 条 (正类 {pos}, {pos / max(len(samples), 1):.1%}) {cnt}"
        )


# --------------------------------------------------------------------------
def save_splits(splits: dict[str, list[Sample]], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in splits.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    print(f"[data] 切分已保存至 {out_dir}")
    return out_dir


def load_splits(out_dir: str | Path) -> dict[str, list[Sample]]:
    out_dir = Path(out_dir)
    out: dict[str, list[Sample]] = {}
    for split in ("train", "dev", "test"):
        path = out_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        out[split] = [Sample(**r) for r in rows]
    return out


def build_victim_trainset(
    cfg: Config,
    poison_ratio: float,
    poison_families: tuple[str, ...] = ("template", "paraphrase"),
) -> tuple[list[str], list[int]]:
    """构造受害模型的训练集：干净 train + 指定比例的投毒样本。

    用于"投毒比例 → 下游分类器性能退化"消融实验。
    """
    splits = build_dataset(cfg)
    clean = [s.text for s in splits["train"] if s.label == 0]
    n_poison = int(len(clean) * poison_ratio / max(1e-9, 1 - poison_ratio))
    if n_poison <= 0:
        return clean, [0] * len(clean)
    pool = [s.text for s in splits["train"] if s.label == 1]
    if len(pool) < n_poison:  # train 里没有攻击样本时现场生成
        gen = atk.build_attacks(list(poison_families), n_poison, cfg.attack.dev_seed)
        pool = [g.text for g in gen]
    rng = random.Random(cfg.data.seed)
    poison = rng.sample(pool, min(n_poison, len(pool)))
    texts = clean + poison
    labels = [0] * len(clean) + [1] * len(poison)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    return [texts[i] for i in idx], [labels[i] for i in idx]


def dataset_stats(splits: dict[str, list[Sample]]) -> dict:
    stats: dict = {}
    for split, samples in splits.items():
        fam: dict[str, int] = {}
        for s in samples:
            fam[s.family] = fam.get(s.family, 0) + 1
        stats[split] = {
            "total": len(samples),
            "positive": sum(s.label for s in samples),
            "by_family": fam,
            "avg_chars": round(sum(len(s.text) for s in samples) / max(len(samples), 1), 1),
        }
    return stats
