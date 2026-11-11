"""PoisonGuard: 大模型训练数据投毒检测与防护框架。

模块划分:
    config      : 全局配置(dataclass + YAML)
    data        : 语料加载/去重/切分/投毒注入
    attacks     : 四类投毒攻击家族(模板/改写规避/隐晦断言/后门触发)
    encoders    : 句向量编码器(本地 CoROM / BGE / Qwen-Embedding), 带磁盘缓存
    detectors   : 检测器族(关键词/语义质心/kNN/马氏/PPL/LLM判官/融合) + 阈值校准
    judge       : Qwen2.5-Instruct 零样本判官与 LoRA 安全分类器封装
    evaluate    : 指标(P/R/F1/AUROC/AUPRC/TPR@FPR)、Bootstrap CI、McNemar 检验、绘图
    report      : 生成 Markdown 实验报告
    cli         : 命令行入口 (python -m poisonguard.cli ...)
"""

__version__ = "2.0.0"
