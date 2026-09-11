# gold 标注协议 v1（第一稿）

## 状态定义

- `draft_requires_expert_review`：机器/启发式生成，尚未审核；
- `under_review`：审核中；
- `reviewed`：专家定稿，**只有该状态的条目可以进入论文指标计算**。

## 现有第一稿（v0.1，来自真实数据）

- `results/draft_gold_queries.json`：15 条跨境类查询：`gold_rules_draft`
  来自真实引用的自动匹配；`risk_draft` = 命中规则的 sensitivity_score 最大值
  （派生草稿，非专家评分）；
- `results/draft_gold_conflicts.json`：12 对跨法律规则初判。当前结论：
  内地 72 条规则库内“同场景真正冲突”极少，多数为并行适用。

## 审核任务清单（交给审核人）

1. 查询集：逐条确认 X_target 属性映射与 C_op 五元组是否符合真实场景；
2. 适用规则：核对 gold_rules 是否遗漏/多挂（依据法条原文）；
3. 冲突对：仅在**具体查询上下文**下判定冲突（Conflict(r_i,r_j;Q)），
   并给出压制/排除结论与法理依据；
4. 风险 gold：按 0–1 独立评分（建议两人以上，不一致处讨论收敛）；
5. LV/BU：n=300 专家评分样本。

## 审核后流程

1. 审核人在 JSON 中把条目状态改为 `reviewed`（或修改字段）；
2. 脚本只读 `reviewed` 条目计算 CRR / Risk MAE；
3. 结果与中间文件一并归档（保留审核前后版本，便于追溯）。
