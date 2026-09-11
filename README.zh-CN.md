# LexMedRAG：代码、基准与标注

[English](README.md) | **简体中文**

本仓库是论文《LexMedRAG：面向跨境医疗数据合规的知识图谱增强框架》的配套材料，投稿于
**Knowledge-Based Systems**。仓库包含法律知识库、GBAMC 基准、对抗性冲突集、专家标注，以及
复现论文全部数字与图表的脚本。

论文正文为中文，因此部分数据说明沿用中文；代码注释与英文 README 为英文。

## 论文主张什么

论文在两个数据集上给出一项实证结论：跨境医疗数据合规场景下，两条规则是否冲突，可以由一个
**带类型与效力的规范图**判定——图中每个规则节点携带效力层级、规范性质、颁布时间与适用方向。
在 840 条自然查询上，214 个专家已裁决规则对中有 94.9% 属于伪冲突，且其中 52.8% 跨法域，
说明跨法域既不是冲突的充分条件也不是必要条件。在含 641 个预期冲突对的对抗集上，方法的冲突
召回率为 100%、识别率为 71.9%，比最强的符号基线高 15.6 个百分点（配对 McNemar 精确检验，
$p<10^{-14}$）。

风险识别分支与合规路径选择层在论文中**只作为设计给出**。它们的评估依赖尚未完成的专家标注，
因此论文不对这两层作出任何性能结论；用于该评估的标注表随本仓库提供。

## 目录结构

| 路径 | 内容 |
|---|---|
| `data/` | 法律知识库：覆盖 8 部规范的 110 条规则，另有 138 条问答派生条文摘录，仅用于检索语料 |
| `gbamc/` | 查询生成器：把 Synthea 患者记录与场景定义映射为 GBAMC 查询 |
| `src/lexmedrag/` | 核心库：BM25 检索、确定性法理裁决、可废止逻辑基线、指标计算 |
| `scripts/` | 论文中的全部实验与绘图脚本 |
| `results/` | 模型输出与派生指标文件 |
| `gold/` | 214 个规则对的专家裁决结果，以及第二轮标注包 |
| `docs/` | 基准构造方案、对抗集说明、标注协议、期刊文风分析 |
| `figures/` | 图 6 的源文件 |

## 数据说明

**法律知识库（110 条规则）。** `data/legal_knowledge_base.json` 含 72 条，
`legal_knowledge_base_gba_draft.json` 13 条，`legal_knowledge_base_hkmo_draft.json` 8 条，
`legal_knowledge_base_adversarial_draft.json` 15 条，`legal_knowledge_base_hgr_draft.json` 2 条，
合计 110 条。每条规则带有法域、效力层级、规范性质、颁布时间与适用方向。有两点需要说明：其一，
药物警戒记录保存期限条款与境外监管条款（欧盟 GVP 模块六、ICH E2B(R3)、美国 21 CFR 314.80）
为整理稿，标记为 `verification_required: true`，在与原文核对前不参与指标计算；其二，
`legal_knowledge_base_qa_derived_draft.json` 中的 138 条摘录只用于把检索语料扩展到 248 条，
不参与冲突判定。

**GBAMC 基准（840 条查询）。** `results/draft_conflict_gold_query_all_v5.json` 是模型在全量
基准上的输出，包含每条查询生成的合规结论。查询情境与数据属性由 `gbamc/scenarios.py` 与
`gbamc/generate_queries.py` 定义。

**对抗性冲突集（240 条查询 / 641 个预期冲突对）。** `results/adversarial_queries_v2.json`
定义 C1--C6 六组冲突形态，`results/adversarial_conflict_draft_v3.json` 是对应的模型输出。
该集合为**人工构造、非自然分布**，其指标不得与自然语料的指标混合统计。

**专家标注。** `gold/conflict_rule_pairs_gold_v4.json` 收录 214 个规则对的专家裁决，包含冲突
判定、处置结论、法理依据、出现次数与传输方向。第一轮评审表保留在
`gold/conflict_rule_pairs_review_v*.csv`。

**第二轮标注包。** `gold/expert_round2/` 含用于一致性检验的规则对盲标表、风险评分表、结论
效用评分表与标注说明。可用 `python gold/make_expert_package.py` 重新生成。

## 复现论文

| 论文内容 | 命令 |
|---|---|
| 表 1，自然语料指标 | `python scripts/conflict_metrics_layered.py` |
| 表 2--3，对抗集识别率与基线对比 | 先 `python scripts/experiment_analysis.py`，再 `python scripts/compare_methods.py` |
| 表 4--5，McNemar 检验与知识属性归因 | `python scripts/significance_tests.py` |
| 消融实验（表 6） | `python scripts/ablation_v2.py` |
| 检索评测（表 7） | `python scripts/retrieval_comparison.py` |
| 图 6 | `python scripts/make_figures.py` |
| 专家标注表 | `python gold/make_expert_package.py` |

所有脚本把结果写入 `results/` 并打印所生成的表格，因此论文中的每个数字都能追溯到对应的文件。
论文中的全部指标都可以用仓库内已提交的模型输出复现，无需调用任何接口。

## 环境

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

开发环境为 Python 3.12。核心库只依赖标准库；绘图与显著性检验脚本需要 `numpy` 与 `matplotlib`。

三个语言模型基线与冲突初判调用 DeepSeek 对话接口。请把密钥写入 `.env` 的
`DEEPSEEK_API_KEY=...`，或设置为同名环境变量。本仓库不包含任何密钥，客户端也不会记录密钥。
重跑这些基线会消耗接口额度，但复现论文中的任何数字都不需要重跑。

## 重新生成合成人群

原始的 Synthea CSV 导出为 78 MB，未纳入仓库，可按以下命令再生：

```bash
java -jar synthea-with-dependencies.jar -p 2000 -s 20260909
python -m gbamc.generate_queries --synthea <output/csv> --per-scenario 120
```

论文使用的快照包含 1,171 名患者、53,346 次就诊、299,697 条观察记录、8,376 条诊断记录、
42,989 条用药记录与 34,981 条手术记录，相同数值记录在 `results/dataset_card_v2.json`。

## 未包含的内容

* `.env` 与任何接口密钥；
* Synthea 原始 CSV 导出与图表的 TIFF 导出，二者均可用上面的命令再生；
* 大部分查询文件的中间草稿。论文数字对应的是「数据说明」中列出的那些文件，少数早期版本因仍被
  脚本引用而保留。

## 许可

代码以 MIT 许可发布；知识库、基准、标注与结果以 CC BY 4.0 发布。知识库中引用的法律条文仍受
其自身条款约束，此处为研究用途的简短摘录。

## 引用

待补。论文录用后将补充 BibTeX 条目。
