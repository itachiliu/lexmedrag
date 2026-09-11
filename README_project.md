# LexMedRAG Experiments（小型实验项目）

本项目是从 `D:\python_vscode\compliance\ComplianceAgent-main.zip` 中筛选出的**真实可用资产**搭建的实验脚手架，用于把 LexMedRAG 论文中的实验逐步变成“可在真实数据上复现”的流程。

## 数据来源（真实，来自你的项目压缩包）

| 文件 | 内容 | 用途 |
|---|---|---|
| `data/legal_knowledge_base.json` | 72 条内地规则（PIPL 31、广东健康医疗数据分级规范 40、网络安全标准实践指南 1），含敏感度 0.2–0.98 | 规则知识图谱/检索语料 |
| `data/legal_knowledge_base_gba_draft.json` | 13 条 GBA 标准合同实施指引规则草稿（内地↔香港、内地↔澳门），`draft_for_review` | 跨境冲突检测的对向规则 |
| `data/legal_knowledge_base_hkmo_draft.json` | 8 条香港 PDPO（含第4原则资料保安）/ 澳门 PDPA 跨境规则草稿，`draft_for_review` | 跨境冲突检测的目的地法域规则 |
| `data/legal_knowledge_base_hgr_draft.json` | 2 条《人类遗传资源管理条例》第28条与实施细则第36条草稿，`draft_for_review` | 基因/测序信息对外提供的合规缺口 |
| `data/merge.json` | 989 条医疗合规问答（问题 + 正向上下文 + 标准答案 + 类别/类型） | 检索评测的查询与证据来源 |
| `data/PIIquestion.json` | 14 条敏感个人信息基础问答 | 补充语料/LLM 提示词素材 |

规则库合计 95 条（72 条内地正式规则 + 13 条 GBA 草稿 + 8 条港澳草稿 + 2 条人遗法规草稿）。
草稿规则均带 `draft_for_review` 标记与 `verification_required: true`，经专家复核前不进入论文指标。

规则带 `norm_type` 字段（法律／行政法规／部门规章／规范性文件／团体标准等），
用于阻止推荐性标准被当作法律义务适用；跨法域规则对统一按“义务并存、孰严执行”处理。

## GBAMC 查询集与专家 gold

- `results/gbamc_queries_v3.json`：840 条真实方向查询（7 个场景 × 120），覆盖 CN→HK、CN→MO、HK→CN、MO→CN，
  并回填 `subject_profile`（是否 CIIO、累计年度规模），供判断第40条主体要件。
- `gold/conflict_rule_pairs_gold_v4.json`：两批专家裁决合并后的规则对级 gold（214 对）。
  结论为：203 对无冲突、9 对场景依赖、2 对稳定冲突（GBA 香港版／澳门版标准合同与出境安全评估的衔接，口径待统一）。
- `results/conflict_metrics.json`：模型判定与 gold 的对齐指标（压制判定精确率、漏检、处置一致率）。

## 对抗性冲突集

自然语料中 214 个已审规则对里 203 对无冲突（伪冲突占绝对多数），因此另设
**对抗性冲突集**用于评估真冲突出现时的消解能力，二者分开统计：

- `results/adversarial_queries_v1.json`：36 条人工设计查询，覆盖六组真实冲突
  （阻断条款 vs 境外监管检查、GBA 禁转 vs 全球药物警戒数据库、紧急豁免 vs 人遗审批、
  删除权 vs 法定留存、澳门适当性 vs GBA 便利推定、次级利用的同意口径）；
- `data/legal_knowledge_base_adversarial_draft.json`：16 条对抗集新增规则
  （阻断条款、紧急豁免、人遗禁止、法定留存、香港同意定义与保留原则、EU GVP、ICH E2B、FDA 21 CFR）；
- `results/layered_conflict_metrics.json`：自然集与对抗集的分层指标；
- 协议与声明见 `docs/ADVERSARIAL_CONFLICT_SET_v1.md`（该集为人工构造，不得与自然语料混合统计）。

冲突判定采用封闭枚举（5 类冲突类型、7 种裁决结论），并对跨法域压制、推荐性标准压制上位法、
敏感个人信息数量门槛误用等情形做自动校验与归一。

## 能真实跑什么

- `python run_experiments.py card`：数据卡片（规则数量/法律分布/类别/敏感度分布、QA 分布）——全部由真实文件统计得出。
- `python run_experiments.py retrieval`：规则检索 pilot。查询 = QA 问题，语料 = 72 条规则；gold 规则通过“正向上下文中引用的《法律》第 X 条”与规则 id 自动匹配（仅含可解析到条文的规则）。输出 Recall@5 / NDCG@5。
- `python run_experiments.py selftest`：裁决器与指标的自检（Lex Superior / Lex Posterior / CRR 定义）。
- `python run_experiments.py llm-check`：验证 DeepSeek key（读取 `.env`，不回显）。
- `python run_experiments.py gold-draft`：LLM 辅助生成冲突对与查询 gold **草稿**（须专家审核）。

GBAMC 真实数据源与构造方案见 `docs/GBAMC_CONSTRUCTION_PLAN.md`。
GBAMC（Synthea 路线）生成器：

```bash
# 先用 Synthea 生成合成人群并导出 CSV（见 docs/GBAMC_CONSTRUCTION_PLAN.md）
python -m gbamc.generate_queries --synthea <output/csv> --per-scenario 120
```

gold 标注协议（第一稿）见 `gold/ANNOTATION_PROTOCOL_v1.md`；
机器生成的草稿在 `results/draft_gold_*.json`，审核定稿前不进入论文指标。

## 指标实现与论文定义一致

- `Recall@5`、`NDCG@5`：标准检索指标；
- `CRR`：只统计含标注冲突的查询；裁决输出（压制方向/排除）与标注一致才算对，漏检计错——与论文式 (CRR) 的定义一致；
- `Risk MAE`：模型风险分与专家标注的平均绝对误差。

实现见 `src/lexmedrag/metrics.py`。

## 还不能跑、需要真实输入的

1. **CRR 与 Risk MAE**：需要法律专家对“冲突对 + 压制/排除结论”“风险得分”做标注。标注模板见 `gold/`；
2. **LLM 类方法（NLI 冲突检测、最终生成、LV/BU 打分）**：需要 DeepSeek API Key 与对应评测样本；
3. **GBAMC 全量查询**：zip 中没有独立、已标注的 600 条测试集（`statistic.py` 亦注明“缺少实际输入案例”），需要另行构造并经专家/来源核验。

## 重要声明

本项目不生成、不填充任何编造的实验数字。`results/` 中出现的任何数值都是上述真实文件上可复现的计算结果，仅用于验证流程；在未获得独立标注与完整测试集之前，**不得直接当作论文实验结果使用**。

## 目录结构

```
lexmedrag-experiments/
├── data/                  # 真实数据（从 ComplianceAgent-main.zip 复制）
├── gold/                  # 待标注数据的 JSON 模板
├── results/               # 运行输出（JSON）
├── src/lexmedrag/         # 核心代码（stdlib only）
├── run_experiments.py     # 入口
└── README.md
```
