"""复现论文中"规则适用方向约束"消融的三个计数。

输出：
  1. 知识库中只规范"内地向境外提供"的规则条数与编号；
  2. 基准中流向为港澳->内地的查询条数；
  3. 这二者若不施加方向约束会形成的候选规模，以及施加约束后
     出境规则实际出现在入境查询候选集中的次数。
"""

from __future__ import annotations

import glob
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 只规范"内地向境外提供"的规则前缀
OUTBOUND_PREFIXES = ("CN_OUTBOUND", "CN_PIPL_Art38", "CN_PIPL_Art39", "CN_PIPL_Art40")


def load_items(path: str) -> list[dict]:
    with io.open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data["items"] if isinstance(data, dict) else data


def main() -> None:
    rule_ids: set[str] = set()
    for path in glob.glob(os.path.join(ROOT, "data", "legal_knowledge_base*.json")):
        rule_ids |= {item["rule_id"] for item in load_items(path)}
    outbound = sorted(r for r in rule_ids if r.startswith(OUTBOUND_PREFIXES))

    queries = load_items(
        os.path.join(ROOT, "results", "draft_conflict_gold_query_all_v5.json")
    )
    inbound = [
        q
        for q in queries
        if q["C_op"].get("loc_dst") == "CN" and q["C_op"].get("loc_src") != "CN"
    ]

    leaked = sum(
        1
        for q in inbound
        for rule in q.get("candidate_rules", [])
        if rule.startswith(OUTBOUND_PREFIXES)
    )

    print(f"出境专属规则: {len(outbound)} 条")
    for rule in outbound:
        print("   ", rule)
    print(f"入境查询: {len(inbound)} 条（其余 {len(queries) - len(inbound)} 条为出境方向）")
    print(f"不施加方向约束时的候选规模上界: {len(outbound) * len(inbound)}")
    print(f"施加方向约束后出境规则出现在入境候选集中的次数: {leaked}")

    out = {
        "outbound_rules": outbound,
        "inbound_queries": len(inbound),
        "total_queries": len(queries),
        "worst_case_direction_errors": len(outbound) * len(inbound),
        "observed_direction_errors": leaked,
    }
    path = os.path.join(ROOT, "results", "direction_constraint_check.json")
    with io.open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=1)
    print("written", path)


if __name__ == "__main__":
    main()
