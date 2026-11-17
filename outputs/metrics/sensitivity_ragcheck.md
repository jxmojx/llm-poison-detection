# RAG 守卫 · 参数敏感性（检索一致性）

评测床：data/processed_real_v3（train corpus 12000 句），encoder=crom，
aggregate=max；主全集 limit=400（2953 条）；敏感性用 edit+normal 各 150
的快速子集（同评分口径互比）。向量全部命中 outputs/cache，未重编码。

## 敏感性表（top_k × min_ratio，AUROC 为 edit vs normal）

| 配置 | AUROC(edit/norm) | TPR@1%(dev校准) | edit召回@dev | normal FPR@0.5 | 命中冲突 |
|---|---|---|---|---|---|
| top_k=3, min_ratio=0.75 | 0.7562 | 0.52 | 0.52 | 0.0067 | 79 |
| top_k=5, min_ratio=0.75 | 0.7562 | 0.52 | 0.52 | 0.0067 | 79 |
| top_k=8, min_ratio=0.75 | 0.7562 | 0.52 | 0.52 | 0.0067 | 79 |
| top_k=5, min_ratio=0.70 | 0.7595 | 0.5267 | 0.5267 | 0.0067 | 80 |
| top_k=5, min_ratio=0.80 | 0.7496 | 0.5067 | 0.5067 | 0.0067 | 77 |

## 说明

- 基线（outputs/metrics/ragcheck_real_v3.json，top_k=5, min_ratio=0.75）：
  overall AUROC 0.5532，TPR@1% 0.1085，edit vs normal AUROC 0.7691。
- edit 召回@1%：dev clean 上 q99 阈值，test edit 中分数超过阈值的比例。
- 敏感性快速子集仅计 edit+normal；其余家族在 max 聚合下分恒为 0
  （无数值/事实点差异），不进入敏感性表。
