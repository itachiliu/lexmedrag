# 待标注数据模板

`CRR` 与 `Risk MAE` 需要专家标注才能计算真实数值。请按以下 schema 提供标注，然后实现/接入评测循环。

## 重要结论（2026-09 规则库 pilot 初判）

对现有 72 条内地规则做跨法交叉初判后发现：**同一法域、经整理的规则之间，
同场景“真正冲突”极少**；模型给出的多数结论是“客体不同、并行适用”。
因此 CRR 的 gold 必须：

1. 在**具体查询**上做“条件化”标注（同一对规则只在特定 Q 下才构成冲突）；
2. 引入港澳法（香港 PDPO、澳门 PDPA、大湾区安排）等真实跨法域规则后，
   才有可裁决的规范竞合样本；
3. 最终 gold 一律由专家审核定稿，AI 初判（`results/draft_gold_*.json`）
   只作为候选，不得直接用于论文。

2026-09 更新：已并入大湾区/出境规定规则（`data/legal_knowledge_base_gba_draft.json`，
85 条规则）并对 28 条抽样查询做查询条件化初判
（`results/draft_conflict_gold_query_sample.json`：12 条查询、33 对冲突候选）。
**注意：初判结果内部存在矛盾裁决（如敏感数据是否适用量级豁免），
必须由法律专家逐条核对后才能定稿。**

## conflict_gold（供 CRR 使用）

每条含冲突的查询一个条目：

```json
{
  "query_id": "q001",
  "conflicts": [
    {
      "left_rule_id": "CN_PIPL_Art40_001",
      "right_rule_id": "CN_GDHMD_002",
      "verdict": "left_suppresses_right"
    }
  ]
}
```

verdict 取值：`left_suppresses_right`、`right_suppresses_left`、`exclude_left`、`exclude_right`。

## risk_gold（供 Risk MAE 使用）

```json
{
  "query_id": "q001",
  "attributes": ["ICD-10", "hospital_id", "genomic_variant"],
  "context": {
    "src": "MO", "dst": "CN", "act": "model_training", "vol": 50000, "t": "2024-11-15"
  },
  "expert_risk": 0.93,
  "annotator": "expert-1"
}
```
