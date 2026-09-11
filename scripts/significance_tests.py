"""对抗集上各基线与本文方法的配对显著性检验。

以 641 个预期冲突对为单位，判定二分结果（是否被判为冲突），
计算 McNemar 精确检验与配对 bootstrap 差值区间。
"""

from __future__ import annotations

import io
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUIET = {"cumulative", "not_conflict"}
CASE_FILES = {
    "LexMedRAG": "results/adversarial_draft_v2_lexmedrag_run1.json",
    "DL": "results/baseline_defeasible_v2.json",
    "Direct-LLM": "results/baseline_direct_llm_v2.json",
    "Full-KB": "results/baseline_full_kb_v2.json",
    "Legal-RAG": "results/baseline_legal_rag_v2.json",
    "CoT": "results/baseline_cot_v2.json",
    "Few-shot": "results/baseline_fewshot_v2.json",
    "NLI": "results/baseline_nli_style_v2.json",
}


def load(rel: str) -> dict:
    with io.open(os.path.join(ROOT, rel), encoding="utf-8") as handle:
        return json.load(handle)


def outcomes() -> dict[str, dict[tuple, int]]:
    table: dict[str, dict[tuple, int]] = {}
    for name, rel in CASE_FILES.items():
        doc = load(rel)
        hits: dict[tuple, int] = {}
        for item in doc["items"]:
            group = item.get("adversarial_group", "?")
            model = {}
            for conflict in item["conflicts_ai_draft"]:
                if "error" in conflict:
                    continue
                key = tuple(sorted((conflict.get("left_rule_id"), conflict.get("right_rule_id"))))
                model[key] = conflict.get("verdict")
            for pair in item.get("expected_conflicts", []):
                key = tuple(sorted(pair))
                verdict = model.get(key)
                hits[(group, item["query_id"], key)] = 1 if verdict not in QUIET | {None} else 0
        table[name] = hits
    return table


def mcnemar_exact(b: int, c: int) -> float:
    """双尾精确 McNemar 检验：有效样本为不一致对 b + c。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    table = outcomes()
    keys = list(table["LexMedRAG"].keys())
    ours = table["LexMedRAG"]
    rng = random.Random(20260911)
    report = {}
    print(f"{'方法':<12}{'仅本文命中':>10}{'仅基线命中':>10}{'McNemar p':>14}{'差值(pp)':>10}{'95% CI':>20}")
    for name in CASE_FILES:
        if name == "LexMedRAG":
            continue
        other = table[name]
        b = sum(1 for k in keys if ours[k] == 1 and other[k] == 0)
        c = sum(1 for k in keys if ours[k] == 0 and other[k] == 1)
        p = mcnemar_exact(b, c)
        diffs = []
        for _ in range(2000):
            sample = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
            d = sum(ours[k] - other[k] for k in sample) / len(sample)
            diffs.append(d)
        diffs.sort()
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
        report[name] = {
            "only_ours": b,
            "only_baseline": c,
            "mcnemar_p": p,
            "diff_pp": (b - c) / len(keys) * 100,
            "ci95_pp": [lo * 100, hi * 100],
        }
        print(f"{name:<12}{b:>10}{c:>10}{p:>14.2e}{(b - c) / len(keys) * 100:>10.1f}"
              f"{f'[{lo * 100:.1f}, {hi * 100:.1f}]':>20}")
    out = os.path.join(ROOT, "results", "significance_tests.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps({"n_pairs": len(keys), "methods": report}, ensure_ascii=False, indent=1)
    )
    print("written", out)


if __name__ == "__main__":
    main()
