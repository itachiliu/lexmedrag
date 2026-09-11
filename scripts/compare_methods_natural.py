"""对比实验（自然语料子集）：各方法把并行义务误判为冲突的倾向。

用法：
    python scripts/compare_methods_natural.py

输出：
    results/method_comparison_natural.json
"""

from __future__ import annotations

import io
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPRESS = {
    "left_suppresses_right", "right_suppresses_left",
    "exclude_left", "exclude_right",
}

METHODS = {
    "LexMedRAG（完整）": ("results/draft_conflict_gold_query_sample.json", True),
    "可废止逻辑（标准 DL）": ("results/baseline_defeasible_natural.json", False),
    "Direct-LLM（零样本）": ("results/baseline_direct_llm_natural.json", False),
    "Chain-of-Thought": ("results/baseline_cot_natural.json", False),
    "Few-shot（示例引导）": ("results/baseline_fewshot_natural.json", False),
    "NLI-Style（成对蕴含）": ("results/baseline_nli_style_natural.json", False),
    "Legal-RAG（检索+生成）": ("results/baseline_legal_rag_natural.json", False),
    "Full-KB（无候选筛选）": ("results/baseline_full_kb_natural.json", False),
}


def _load(path: str) -> dict:
    with io.open(os.path.join(BASE, path), encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    queries = _load("results/gbamc_queries_v3.json")["queries"][:120]
    subset = {q["query_id"] for q in queries}
    gold = _load("gold/conflict_rule_pairs_gold_v4.json")
    gmap = {
        tuple(sorted((i["left_rule_id"], i["right_rule_id"]))): i
        for i in gold["items"]
    }
    results = {}
    for name, (path, restrict) in METHODS.items():
        doc = _load(path)
        claims = set()
        for item in doc["items"]:
            if restrict and item["query_id"] not in subset:
                continue
            for c in item.get("conflicts_ai_draft", []):
                if "error" in c:
                    continue
                if c.get("verdict") in SUPPRESS:
                    claims.add(tuple(sorted((c["left_rule_id"], c["right_rule_id"]))))
        reviewed = [k for k in claims if k in gmap]
        fp = [k for k in reviewed if gmap[k]["verdict"] == "无冲突"]
        results[name] = {
            "suppression_claims": len(claims),
            "reviewed_claims": len(reviewed),
            "false_positives": len(fp),
            "false_positive_rate": round(len(fp) / len(reviewed), 4) if reviewed else None,
            "false_positive_pairs": [list(k) for k in fp],
        }
    doc = {
        "meta": {
            "generated_at": "2026-09-10",
            "dataset": "natural GBAMC subset (first 120 queries)",
            "note": "误报定义为：模型声称存在压制/排除关系，而专家裁决为“无冲突”。",
        },
        "methods": results,
    }
    out = os.path.join(BASE, "results", "method_comparison_natural.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"{'method':<28} claims  reviewed  false_pos  fp_rate")
    for name, r in results.items():
        rate = f"{r['false_positive_rate']:.3f}" if r["false_positive_rate"] is not None else "n/a"
        print(f"{name:<28} {r['suppression_claims']:>6}  {r['reviewed_claims']:>8}  "
              f"{r['false_positives']:>9}  {rate}")
    print("written:", out)


if __name__ == "__main__":
    main()
