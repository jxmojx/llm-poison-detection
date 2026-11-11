"""全局配置：dataclass 定义 + YAML 覆盖。

设计原则:
1. 所有路径默认相对仓库根目录，换机器只需改 configs/*.yaml。
2. 每个实验阶段(数据/编码/检测/受害模型)配置解耦，便于消融实验。
3. 不依赖 pydantic，只用标准库 + PyYAML，降低环境负担。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    raw_dir: str = str(ROOT / "data" / "raw")
    processed_dir: str = str(ROOT / "data" / "processed")
    normal_file: str = "normal_data_2w.csv"
    poison_file: str = "poison_data_2k.csv"
    real_file: str = "real_tnews.txt"
    clean_source: str = "synthetic"   # synthetic | tnews
    seed: int = 42
    train_ratio: float = 0.70
    dev_ratio: float = 0.15
    # test 比例由 1 - train - dev 推出
    dedupe: bool = True
    max_normal: int = 20000


@dataclass
class AttackConfig:
    """攻击家族配置。dev 与 test 使用不同 seed，保证测试集是"没见过的攻击样本"。"""
    families: list[str] = field(
        default_factory=lambda: ["template", "paraphrase", "subtle", "backdoor"]
    )
    n_per_family: int = 500
    dev_n_per_family: int = 200
    dev_seed: int = 2024
    test_seed: int = 7
    n_ood_dev: int = 300
    n_ood_test: int = 500
    train_poison_ratio: float = 0.05  # 训练集投毒比例(用于受害模型)
    # 模板留出(style holdout)：True 时 dev 与 test 使用互不相交的模板骨架。
    # 默认 False 是为了与 v1 的宽松口径对照；严格评测开关见 experiments/exp_style_holdout.py。
    style_holdout: bool = False
    # ---- v3 新增 -------------------------------------------------------
    # 内容级投毒家族：在真实新闻句上做**一处最小事实篡改**(年份/数量/极性/否定/单位)。
    # 它与原句只差一两个 token，句向量几乎重合 → 无监督离群检测原理上抓不到，
    # 用来量化"到底哪一层在提供语义判别力"。
    edit_family: bool = False
    # 硬负样本条数：从真实语料里挑"含传闻措辞但完全无害"的句子，
    # 一半进 dev(参与阈值校准)、一半进 test(参与主指标)，让主评测集不再是送分题。
    n_hardneg: int = 0


@dataclass
class EncoderConfig:
    # 可选: crom(本地已下载, 无需联网) / bge / qwen3-embedding
    name: str = "crom"
    model_dir: str = str(
        ROOT / "model_cache" / "damo" / "nlp_corom_sentence-embedding_chinese-base"
    )
    batch_size: int = 64
    max_length: int = 64
    device: str = "auto"  # auto/cpu/cuda
    cache_dir: str = str(ROOT / "outputs" / "cache")
    normalize: bool = True


@dataclass
class DetectorConfig:
    target_fpr: float = 0.01
    knn_k: int = 5
    pca_dim: int = 64
    # 关键词检测器(复现原项目规则法)使用的关键词表
    keywords: list[str] = field(
        default_factory=lambda: [
            "低下", "无风险", "回报率", "无需吃药", "治愈",
            "抵制", "反对", "虚假", "恶意", "煽动",
        ]
    )
    enable: list[str] = field(
        default_factory=lambda: [
            "keyword", "centroid", "knn", "mahalanobis", "iforest",
            "sup_lr", "sup_tfidf", "fusion",
        ]
    )


@dataclass
class VictimConfig:
    """受害模型。none=不启用行为级检测; bert=复用旧项目已训练的 BERT。"""
    kind: str = "bert"
    model_dir: str = str(
        ROOT / "model_cache" / "damo" / "nlp_corom_sentence-embedding_chinese-base"
    )
    weights: str = str(ROOT / "baseline_model.pth")
    max_length: int = 128
    batch_size: int = 32


@dataclass
class QwenConfig:
    enabled: bool = True
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    fallback_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    few_shot: bool = True
    # 判定接口：logprob=读"污染/正常"两个候选 token 的对数概率比(连续分，可校准，
    # 解决二值输出导致 TPR@FPR=1% 结构性为 0 的问题)；binary=自由文本 JSON 解码(旧口径)
    scoring: str = "logprob"
    hardened: bool = False
    sanitize: bool = False
    # 本地已下载时直接用目录；ModelScope 下载后自动写入 outputs/qwen/<name>
    local_dir: str = str(ROOT / "outputs" / "qwen" / "Qwen2.5-1.5B-Instruct")
    # 模型权重缓存根目录（与 encoder 的 embedding 缓存 outputs/cache 分开，
    # 否则权重和 npz 混在一起，容易把"只下到 config.json 的半成品目录"当成模型）
    model_cache: str = str(ROOT / "model_cache")
    load_in_4bit: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )
    max_seq_len: int = 512
    per_device_batch: int = 2
    grad_accum: int = 8
    lr: float = 2e-4
    epochs: int = 2
    max_steps: int = -1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    victim: VictimConfig = field(default_factory=VictimConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    outputs_dir: str = str(ROOT / "outputs")
    figures_dir: str = str(ROOT / "outputs" / "figures")
    metrics_dir: str = str(ROOT / "outputs" / "metrics")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def makedirs(self) -> None:
        for p in (self.outputs_dir, self.figures_dir, self.metrics_dir,
                  self.data.processed_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """把嵌套 dict 还原成 dataclass 实例(未知字段直接忽略并告警)。"""
    if not dataclasses.is_dataclass(cls):
        return data
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in (data or {}).items():
        if key not in fields:
            print(f"[config] 忽略未知配置项: {cls.__name__}.{key}")
            continue
        ftype = fields[key].type
        sub = globals().get(ftype) if isinstance(ftype, str) else ftype
        if dataclasses.is_dataclass(sub) and isinstance(value, dict):
            kwargs[key] = _build(sub, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _build(Config, raw)


def save_config(cfg: Config, path: str | Path) -> None:
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
