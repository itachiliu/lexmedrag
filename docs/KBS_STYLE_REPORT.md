# Knowledge-Based Systems 投稿风格分析

分析对象：由 OpenAlex 定位、经本地代理下载的 49 篇 KBS 2024 年以后论文全文
（`pdfs2/`），以及 200 篇同期论文的元数据（`kbs_openalex.json`）。
统计脚本：`analyze_kbs_style.py`；原始统计见 `style_stats.json`。

## 一、结构范式

最常见的章节序列（按出现次数）：

1. Introduction（44/48）
2. Related Work / Related Works（33/48）
3. Methodology / Proposed Method（21/48）
4. Experiments / Experiments and Results（24/48）
5. Results and Discussion
6. Limitation and Future Work
7. Conclusion（29/48）

要点：KBS 论文普遍把"相关工作"与"方法"分成两节，并在结论之前单列
Limitations and Future Work；Discussion 与 Results 常合并为一节。

## 二、写作规范的出现比例（48 篇全文）

| 规范 | 出现比例 |
|---|---|
| 讨论局限性 | 81.2% |
| 明确的基线比较 | 75.0% |
| Highlights | 60.4% |
| 消融实验 | 58.3% |
| 代码/数据可复现说明 | 43.8% |
| 复杂度分析 | 39.6% |
| 案例研究 | 22.9% |
| 统计显著性检验 | 14.6% |

## 三、Highlights 的写法

Elsevier 要求 3–5 条、每条不超过 85 字符。实际样本的写法特征：

* 每句独立陈述一个具体结论，不用 "we propose" 开头；
* 常以名词短语或动名词起句，如 "Clustering prototype learning mitigates ..."、
  "Graph-based modeling captures ..."；
* 优先给可验证的结果而非设计意图，如 "Enriched datasets enable ML models to
  achieve over 92% prediction accuracy."；
* 第四条常写"在某基准上达到最优"。

样例（原文）：

> • Clustering prototype learning mitigates soft positive sample issues in image-text matching.
> • Adaptive fusion of global-local features enables comprehensive semantic understanding.
> • Graph-based modeling captures high-order relations among similar instances.
> • State-of-the-art results on Flickr30K, MSCOCO and ECCV Caption benchmarks.

## 四、摘要的写法

样本摘要的共同结构是：领域问题 → 现有方法的缺口 → 本文提出什么 → 由哪些模块构成
→ 在什么数据上评估 → 用具体数字给出结论。多篇摘要直接在摘要中给出量化结果，例如
"achieves a Mean Average Precision (MAP@100) of 0.83 and a Mean Reciprocal Rank
(MRR@100) of 0.92"。

## 五、对本文的直接影响

据此对稿件做的改动：

1. 新增 Highlights 五条（英文，放在标题之后）；
2. 摘要改写为"知识表示决定可判定性"的主线，并补入跨法域比例与配对检验 p 值；
3. 贡献列表重写为五项，第一项是规范图的形式化，第二项是裁决算法，系统实现退居第三；
4. 相关工作新增"知识驱动的推理与知识图谱增强生成"小节，引用 KBS 已发表论文
   （Aarab 2024、Huang 等 2025、Tang 等 2026、Li 等 2025、Park 与 Park 2026、
   Xu 等 2025、Wang 等 2024、Ji 等 2021、Scaboro 等 2023）；
5. 方法部分新增"复杂度与可扩展性"小节，给出各环节复杂度并说明按查询调用与按对调用的差异；
6. 实验部分新增"统计显著性与知识属性归因"小节，含配对 McNemar 检验表（7 组全部显著）
   与知识属性归因表（跨法域 401 对、同法域 240 对）。

## 六、仍未覆盖的 KBS 期望

* 代码与数据可复现链接尚未写入正文；
* 风险识别分支与决策层仍无实证评估，是最大的单点风险；
* 专家标注仍为单轮，缺少第二标注者与一致性系数。
