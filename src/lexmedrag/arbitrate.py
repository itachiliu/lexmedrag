"""Deterministic doctrine-based arbitration (demo implementation).

Level mapping used here (modeling choice, based on issuer rank):
    CN_PIPL / HK_PDPO / MO_PDPA (法律/主体条例) -> 3
    CN_HGR    (行政法规)                        -> 2
    CN_HGR_RULES / CN_OUTBOUND (部门规章)       -> 1
    CN_CSSPG  (推荐性技术指南) / CN_GBA (指引)   -> 1
    CN_GDHMD  (团体标准/推荐性技术文件)          -> 0
Lex Superior compares these levels; Lex Posterior compares issue_date.
Explicit override edges can be supplied to encode Lex Specialis cases.
跨法域规则（内地↔香港↔澳门）不直接适用 Lex Superior 比较；
`level` 仅表示该规则在其所属法域内的效力位阶。
推荐性标准与技术指南（level 0–1）不得以 lex_specialis 压制法律、行政法规
或部门规章；跨法域义务为累积适用，不产生压制关系。
"""

from __future__ import annotations

from dataclasses import dataclass


HIERARCHY: dict[str, int] = {
    "CN_PIPL": 3,
    "CN_DSL": 3,
    "HK_PDPO": 3,
    "MO_PDPA": 3,
    "EU_GVP": 3,
    "FDA_21CFR": 3,
    "ICH_E2B": 1,
    "CN_HGR_RULES": 1,
    "CN_HGR": 2,
    "CN_PV": 1,
    "CN_ADR": 1,
    "CN_CSSPG": 1,
    "CN_GDHMD": 0,
    "CN_GBA": 1,
    "CN_OUTBOUND": 1,
}


@dataclass(frozen=True)
class Resolved:
    winner: str
    loser: str
    basis: str


def level(rule_id: str) -> int:
    for key in HIERARCHY:
        if rule_id.startswith(key):
            return HIERARCHY[key]
    return 1


def resolve(
    rule_a: dict,
    rule_b: dict,
    overrides: dict[tuple[str, str], str] | None = None,
) -> Resolved:
    """Return which rule suppresses which.

    rule_a/rule_b must contain rule_id and issue_date.
    overrides: {(winner_id, loser_id): basis} encodes explicit Lex Specialis
    /exemption edges.
    """
    overrides = overrides or {}
    a, b = rule_a["rule_id"], rule_b["rule_id"]
    if (a, b) in overrides:
        return Resolved(a, b, overrides[(a, b)])
    if (b, a) in overrides:
        return Resolved(b, a, overrides[(b, a)])
    la, lb = level(a), level(b)
    if la != lb:
        winner, loser = (a, b) if la > lb else (b, a)
        return Resolved(winner, loser, "lex_superior")
    da, db = rule_a.get("issue_date", ""), rule_b.get("issue_date", "")
    if da != db and da and db:
        winner, loser = (a, b) if da > db else (b, a)
        return Resolved(winner, loser, "lex_posterior")
    return Resolved(a, b, "unresolved")
