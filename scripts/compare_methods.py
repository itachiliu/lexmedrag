"""对比实验：完整方法与四个基线在对抗集上的冲突判定能力。

用法：
    python scripts/compare_methods.py

输出：
    results/method_comparison.json
"""

from __future__ import annotations

import collections
import datetime as _dt
import io
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUIET = {"cumulative", "not_conflict"}

METHODS = {
    "LexMedRAG": "results/adversarial_draft_v2_lexmedrag_run1.json",
    "DL": "results/baseline_defeasible_v2.json",
    "Direct-LLM": "results/baseline_direct_llm_v2.json",
    "CoT": "results/baseline_cot_v2.json",
    "Few-shot": "results/baseline_fewshot_v2.json",
    "NLI": "results/baseline_nli_style_v2.json",
    "Legal-RAG": "results/baseline_legal_rag_v2.json",
    "Full-KB": "results/baseline_full_kb_v2.json",
}


def _load(path: str) -> dict:
    with io.open(os.path.join(BASE, path), encoding="utf-8") as f:
        return json.load(f)


def evaluate(path: str) -> dict:
    doc = _load(path)
    tot = {"expected": 0, "detected": 0, "recognized": 0}
    by_group: dict[str, dict] = {}
    claimed = set()
    errors = 0
    for r in doc["items"]:
        g = r.get("adversarial_group") or "?"
        st = by_group.setdefault(g, {"expected": 0, "detected": 0, "recognized": 0})
        model = {}
        for c in r.get("conflicts_ai_draft", []):
            if "error" in c:
                errors += 1
                continue
            key = tuple(sorted((c.get("left_rule_id"), c.get("right_rule_id"))))
            model[key] = c
            if c.get("verdict") not in QUIET:
                claimed.add(key)
        for pair in r.get("expected_conflicts", []):
            key = tuple(sorted(pair))
            tot["expected"] += 1
            st["expected"] += 1
            if key in model:
                tot["detected"] += 1
                st["detected"] += 1
                if model[key].get("verdict") not in QUIET:
                    tot["recognized"] += 1
                    st["recognized"] += 1
    return {
        "expected": tot["expected"],
        "detected": tot["detected"],
        "recognized": tot["recognized"],
        "conflict_recall": round(tot["detected"] / tot["expected"], 4) if tot["expected"] else None,
        "conflict_recognition_rate": round(tot["recognized"] / tot["expected"], 4) if tot["expected"] else None,
        "distinct_pairs_claimed_conflict": len(claimed),
        "errors": errors,
        "by_group": {
            g: {
                "expected": v["expected"],
                "recognized": v["recognized"],
                "rate": round(v["recognized"] / v["expected"], 4) if v["expected"] else None,
            }
            for g, v in sorted(by_group.items())
        },
    }


def main() -> None:
    results = {name: evaluate(path) for name, path in METHODS.items()}
    doc = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "dataset": "adversarial_conflict_set (36 queries / 97 expected pairs)",
            "note": "全部方法使用相同候选规则集合与相同查询，差异仅在判定方式。",
        },
        "methods": results,
    }
    out = os.path.join(BASE, "results", "method_comparison.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print("method                          recall  recognition  claimed")
    for name, r in results.items():
        print(f"{name:<30} {r['conflict_recall']:.3f}   {r['conflict_recognition_rate']:.3f}"
              f"        {r['distinct_pairs_claimed_conflict']}")
    print("\nper-group recognition:")
    groups = sorted({g for r in results.values() for g in r["by_group"]})
    print("method                          " + "  ".join(groups))
    for name, r in results.items():
        row = "  ".join(
            f"{r['by_group'].get(g, {}).get('rate', 0):.2f}" for g in groups
        )
        print(f"{name:<30} {row}")
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
