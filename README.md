# 大模型污染数据检测实验 🛡️

基于中文预训练 BERT 的序列分类模型，针对大模型训练数据污染场景，设计并验证双阈值检测机制的有效性。

## 🎯 实验目标

训练数据投毒（Data Poisoning）是大模型训练面临的安全威胁之一：少量恶意样本混入训练集即可影响模型行为。本实验探索一种**无需重新训练的检测思路**——利用模型对正常样本与污染样本的行为差异（输出分布、梯度特性）进行判别。

## 🔬 检测机制

- **KL 散度阈值**：对比模型在待检样本上的输出分布与基准分布之间的 KL 散度，超过阈值判定为可疑样本（`KL_THRESHOLD = 0.5`）
- **梯度范数阈值**：污染样本通常引起异常大的梯度响应，计算样本梯度范数并设定阈值判别（`GRAD_NORM_THRESHOLD = 0.5`）
- **双阈值联合判定**：两个指标结合，提高检测精度

## 📊 实验设计

| 项目 | 配置 |
|------|------|
| 基础模型 | 中文预训练 BERT（BertForSequenceClassification） |
| 正常数据 | 20,000 条（normal_data_2w.csv） |
| 污染数据 | 2,000 条（poison_data_2k.csv） |
| 对比方案 | 无防护 baseline vs 有防护检测 |
| 输出 | 论文标准评估图表（检测率、误报率等） |

## 🛠️ 技术栈

- PyTorch + transformers（BERT）
- pandas / numpy（数据处理）
- matplotlib（论文标准图表）

## 📂 文件说明

```
核心实验.py        # 主实验：加载模型 + 无防护/有防护对比 + 结果输出
network.py         # 模型定义与加载
modaldown.py       # 模型下载工具
normal_data_2w.csv # 正常训练数据
poison_data_2k.csv # 污染测试数据
```

## 🚀 运行

```bash
pip install torch transformers pandas matplotlib
python 核心实验.py
```

> 模型缓存目录（model_cache）与权重文件（.pth）未包含在本仓库，运行前需通过 modaldown.py 下载对应中文 BERT 预训练模型。
