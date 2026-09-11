"""Metrics matching the formal definitions in the LexMedRAG manuscript."""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int = 5) -> float:
    if not gold_ids:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in gold_ids)
    return hits / len(gold_ids)


def ndcg_at_k(ranked_ids: list[str], gold_ids: set[str], k: int = 5) -> float:
    """Binary-gain NDCG@k (gain 1 if doc is gold)."""
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k]):
        if rid in gold_ids:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold_ids))))
    return dcg / idcg if idcg > 0 else 0.0


def crr_for_query(
    gold_verdicts: dict[tuple[str, str], str],
    pred_verdicts: dict[tuple[str, str], str],
) -> float:
    """CRR for one query (formal definition in the manuscript).

    gold_verdicts: {(rule_a, rule_b): verdict} from expert annotation.
    verdicts are strings such as "a_suppresses_b", "b_suppresses_a",
    "exclude_a" or "exclude_b".
    pred_verdicts: the same structure produced by the model.
    A conflict pair missed by the model counts as wrong.
    """
    if not gold_verdicts:
        return 0.0
    ok = 0
    for pair, gold in gold_verdicts.items():
        if pred_verdicts.get(pair) == gold:
            ok += 1
    return ok / len(gold_verdicts)


def risk_mae(pred_scores: list[float], gold_scores: list[float]) -> float:
    if not pred_scores:
        return 0.0
    return sum(abs(p - g) for p, g in zip(pred_scores, gold_scores)) / len(pred_scores)
