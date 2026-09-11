"""Generate GBAMC draft queries from Synthea CSV exports.

Usage:
    python -m gbamc.generate_queries --synthea <dir> --out results/gbamc_queries_draft.json

The Synthea directory should contain patients.csv, conditions.csv,
observations.csv, medications.csv (standard export names).

Output queries are DRAFTS for review: attribute mapping and scenario
parameters still need human confirmation before becoming the official GBAMC.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import random
import re

from . import attribute_map as am
from .scenarios import SCENARIOS


def _pick_attributes(patient_id: str, conditions: list[dict],
                     observations: list[dict], rng: random.Random,
                     hints: list[str]) -> list[str]:
    attrs: list[str] = []
    seen: set[str] = set()

    def add(text: str, fallback: tuple[str, str, int]) -> None:
        hit = am.match_category(text)
        if hit is None:
            hit = fallback
        if hit[0] not in seen:
            seen.add(hit[0])
            attrs.append(hit[0])

    pat_conds = [c for c in conditions if c.get("PATIENT") == patient_id][:3]
    for c in pat_conds:
        add(c.get("DESCRIPTION", "") + " " + c.get("CODE", ""), am.DIAGNOSIS_FALLBACK)
    pat_obs = [o for o in observations if o.get("PATIENT") == patient_id][:3]
    for o in pat_obs:
        add(o.get("DESCRIPTION", "") + " " + o.get("CODE", ""), am.OBSERVATION_KEYWORDS["glucose|hba1c|laboratory"])
    if rng.random() < 0.2:
        add("genetic variant sequence", ("基因测序数据", "个人属性数据", 4))
    for hint in hints:
        if hint not in attrs and len(attrs) < 5:
            attrs.append(hint)
    return attrs[:5] or ["住院病历"]


def _weighted_scenario(rng: random.Random) -> dict:
    # one scenario per query is enough for the draft; weights can be reviewed
    return rng.choice(SCENARIOS)


def generate(
    synthea_dir: str,
    per_scenario: int = 10,
    seed: int = 20260909,
) -> dict:
    from . import synthea_io

    patients = synthea_io.load_patients(synthea_dir)
    conditions = synthea_io.load_conditions(synthea_dir)
    observations = synthea_io.load_observations(synthea_dir)
    rng = random.Random(seed)
    queries = []
    qid = 0
    for scenario in SCENARIOS:
        for _ in range(per_scenario):
            if not patients:
                break
            p = rng.choice(patients)
            pid = p.get("Id", "")
            attrs = _pick_attributes(
                pid, conditions, observations, rng, scenario["attr_hints"]
            )
            vol = rng.randint(*scenario["vol_range"])
            ts = _dt.datetime(2024, 1, 1) + _dt.timedelta(
                days=rng.randint(0, 400), hours=rng.randint(0, 23)
            )
            queries.append(
                {
                    "query_id": f"GBAMC-DRAFT-{qid:04d}",
                    "scenario": scenario["id"],
                    "status": "draft_requires_review",
                    "patient_ref": pid,
                    "X_target": attrs,
                    "C_op": {
                        "loc_src": scenario["src"],
                        "loc_dst": scenario["dst"],
                        "act": scenario["act"],
                        "vol": vol,
                        "t": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                }
            )
            qid += 1
    return {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "source": "Synthea CSV exports (synthetic, no credentials)",
            "seed": seed,
            "status": "draft_requires_review",
            "note": "属性映射与场景参数为第一稿；正式 GBAMC 需审核定稿后生成。",
        },
        "queries": queries,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthea", required=True, help="dir with Synthea CSV exports")
    p.add_argument("--out", default=os.path.join("results", "gbamc_queries_draft.json"))
    p.add_argument("--per-scenario", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260909)
    args = p.parse_args()
    payload = generate(args.synthea, per_scenario=args.per_scenario, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(payload['queries'])} draft queries -> {args.out}")


if __name__ == "__main__":
    main()
