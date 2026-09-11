"""生成第二轮专家标注包：一致性检验、风险评分、结论效用评分。

输出到 gold/expert_round2/：
  01_conflict_pairs_round2.csv   214 个已裁决规则对的盲标副本
  02_risk_scores.csv             分层抽样的查询，供专家给风险得分
  03_decision_utility.csv        分层抽样的查询与系统结论，供专家评效用
"""

from __future__ import annotations

import csv
import glob
import io
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "gold", "expert_round2")
RISK_SAMPLE = 200
DECISION_SAMPLE = 120
SEED = 20260911


def load_items(path: str) -> list[dict]:
    data = json.load(io.open(path, encoding="utf-8"))
    return data["items"] if isinstance(data, dict) else data


def rule_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "legal_knowledge_base*.json"))):
        for item in load_items(path):
            texts.setdefault(item["rule_id"], item.get("rule_text") or "")
    return texts


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    path = os.path.join(OUT, name)
    with io.open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"written {name}: {len(rows)} rows")


def stratify(items: list[dict], count: int, key) -> list[dict]:
    rng = random.Random(SEED)
    buckets: dict[str, list[dict]] = {}
    for item in items:
        buckets.setdefault(key(item), []).append(item)
    picked: list[dict] = []
    per_bucket = max(1, count // max(1, len(buckets)))
    for bucket in sorted(buckets):
        pool = buckets[bucket]
        picked.extend(rng.sample(pool, min(per_bucket, len(pool))))
    remaining = [i for i in items if i not in picked]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, count - len(picked))])
    return picked[:count]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    texts = rule_texts()

    gold = load_items(os.path.join(ROOT, "gold", "conflict_rule_pairs_gold_v4.json"))
    rows = []
    missing = 0
    for item in gold:
        left, right = item["left_rule_id"], item["right_rule_id"]
        if left not in texts or right not in texts:
            missing += 1
        rows.append(
            [
                item["pair_id"],
                left,
                texts.get(left, "（规则文本待补）")[:300],
                right,
                texts.get(right, "（规则文本待补）")[:300],
                item.get("occurrences", ""),
                item.get("directions", ""),
                "", "", "", "",
            ]
        )
    write_csv(
        "01_conflict_pairs_round2.csv",
        ["pair_id", "左规则ID", "左规则文本", "右规则ID", "右规则文本",
         "出现查询数", "涉及方向",
         "冲突判定(无冲突/场景依赖/稳定冲突)", "处置结论(并存/待定/左压制右/右压制左)",
         "法理依据(lex_specialis/lex_superior/lex_posterior/contextual/none)", "备注"],
        rows,
    )
    print(f"  规则文本缺失的规则对: {missing}/{len(gold)}")

    natural = load_items(os.path.join(ROOT, "results", "draft_conflict_gold_query_all_v5.json"))
    risk_rows = []
    for item in stratify(natural, RISK_SAMPLE, lambda x: x["scenario"]):
        ctx = item["C_op"]
        risk_rows.append(
            [
                item["query_id"],
                item["scenario"],
                "；".join(item["X_target"]),
                ctx.get("loc_src", ""),
                ctx.get("loc_dst", ""),
                ctx.get("act", ""),
                ctx.get("vol", ""),
                str(ctx.get("t", ""))[:10],
                "", "", "", "",
            ]
        )
    write_csv(
        "02_risk_scores.csv",
        ["query_id", "场景", "数据属性", "源法域", "目标法域", "操作类型",
         "记录数", "时间",
         "风险得分(0-1，可两位小数)", "风险等级(低/中/高)", "标注把握度(高/中/低)", "备注"],
        risk_rows,
    )

    decision_rows = []
    for item in stratify(natural, DECISION_SAMPLE, lambda x: x["scenario"]):
        ctx = item["C_op"]
        decision_rows.append(
            [
                item["query_id"],
                item["scenario"],
                "；".join(item["X_target"]),
                f"{ctx.get('loc_src','')}->{ctx.get('loc_dst','')}",
                ctx.get("act", ""),
                (item.get("assessment") or "").replace("\n", " ")[:800],
                "", "", "", "", "",
            ]
        )
    write_csv(
        "03_decision_utility.csv",
        ["query_id", "场景", "数据属性", "流向", "操作类型", "系统给出的结论",
         "法律有效性(1-5)", "业务效用(1-5)", "是否过度保守(是/否)",
         "遗漏的关键规范(可留空)", "修改建议(可留空)"],
        decision_rows,
    )


if __name__ == "__main__":
    main()
