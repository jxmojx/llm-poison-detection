# legacy —— 原始版本(保留作为对照与"问题复现"的证据)

这三个文件是项目最初提交的版本，**保留是为了让面试官/合作者能亲眼看到问题**，
而不是为了继续维护。当前可用实现见上层 `poisonguard/`。

| 文件 | 原作用 | 实测发现的问题 |
|------|--------|----------------|
| `network.py` | 训练 BERT 二分类 + KL 散度/梯度范数双阈值检测 | 186 行之后全部被缩进进一个 `if not os.path.exists(...)` 分支，数据文件存在时整段不执行；`python network.py` 跑 27.9s、**零输出**、exit 0 |
| `核心实验.py` | 投毒检测"主实验" | `detect_poison()` 实际是**关键词表匹配**(`poison_keywords` 与污染模板用词重合)→ 循环论证；`IsolationForest` 从未 `fit` 却被 `predict`；`text.split()` 对中文不分词，KL 散度退化为整句比对 |
| `modaldown.py` | 从 ModelScope 下载 CoROM | 本身没问题，逻辑已并入 `poisonguard/encoders.py` 与 `poisonguard/localmodels.py` |

原数据文件已移动到 `data/raw/`：

- `normal_data_2w.csv`：20000 行 / **4997 条唯一文本**(重复率 75.0%，单条最高重复 12 次)
- `poison_data_2k.csv`：2000 行 / **376 条唯一文本**(重复率 81.2%，单条最高重复 38 次)

因此这几个脚本里的相对路径(`./normal_data_2w.csv` 等)已失效——这本就是废弃代码，
若要复现原实验，请把 CSV 复制回仓库根目录，或修改脚本内路径常量。
