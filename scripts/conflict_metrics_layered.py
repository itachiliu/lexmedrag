"""分层冲突指标：自然语料（GBAMC）与对抗集分别计算，不混合。

用法：
    python scripts/conflict_metrics_layered.py

输出：
    results/layered_conflict_metrics.json
"""

from __future__ import annotations

import collections
import datetime as _dt
import io
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUIET = {"cumulative", "not_conflict"}
SUPPRESS = (
    "left_suppresses_right", "right_suppresses_left",
    "exclude_left", "exclude_right",
)


def _load(name: str) -> dict:
    with io.open(os.path.join(BASE, name), encoding="utf-8") as f:
        return json.load(f)


def natural_metrics() -> dict:
    draft = _load("results/draft_conflict_gold_query_sample.json")
    gold = _load("gold/conflict_rule_pairs_gold_v4.json")
    gmap = {
        tuple(sorted((i["left_rule_id"], i["right_rule_id"]))): i
        for i in gold["items"]
    }
    model: dict[tuple, list[str]] = collections.defaultdict(list)
    for r in draft["items"]:
        for c in r["conflicts_ai_draft"]:
            if "error" in c:
                continue
            model[tuple(sorted((c["left_rule_id"], c["right_rule_id"])))].append(
                c.get("verdict")
            )
    claims = {k: v for k, v in model.items() if any(x in SUPPRESS for x in v)}
    tp = [k for k in claims if k in gmap and gmap[k]["verdict"] != "无冲突"]
    fp = [k for k in claims if k in gmap and gmap[k]["verdict"] == "无冲突"]
    gold_conflicts = [k for k, v in gmap.items() if v["verdict"] != "无冲突"]
    missed = [k for k in gold_conflicts if k not in model]

    def res(vals: list[str]) -> str:
        if any(v == "pending_stance" for v in vals):
            return "待定"
        if any(v in SUPPRESS for v in vals):
            return "压制"
        return "并存"

    match = compared = 0
    for k, vals in model.items():
        if k not in gmap:
            continue
        compared += 1
        gres = gmap[k]["resolution"]
        kind = "待定" if "待定" in gres else ("压制" if "压制" in gres else "并存")
        if res(vals) == kind:
            match += 1
    return {
        "dataset": "natural_gbamc",
        "queries": len(draft["items"]),
        "gold_pairs": len(gmap),
        "gold_conflict_pairs": len(gold_conflicts),
        "model_pairs": len(model),
        "model_suppression_claims": len(claims),
        "suppression_true_positives": len(tp),
        "suppression_false_positives": len(fp),
        "suppression_precision": round(len(tp) / (len(tp) + len(fp)), 4) if (tp or fp) else None,
        "missed_gold_conflicts": len(missed),
        "resolution_agreement": round(match / compared, 4) if compared else None,
        "false_positive_pairs": [list(k) for k in fp],
        "missed_pairs": [list(k) for k in missed],
    }


def adversarial_metrics() -> dict:
    draft = _load("results/adversarial_conflict_draft_v3.json")
    groups: dict[str, dict] = {}
    tot = {"expected": 0, "detected": 0, "recognized": 0}
    for r in draft["items"]:
        g = r.get("adversarial_group", "?")
        st = groups.setdefault(
            g, {"expected": 0, "detected": 0, "recognized": 0,
                "verdicts": collections.Counter()}
        )
        model = {
            tuple(sorted((c["left_rule_id"], c["right_rule_id"]))): c
            for c in r["conflicts_ai_draft"] if "error" not in c
        }
        for pair in r.get("expected_conflicts", []):
            key = tuple(sorted(pair))
            tot["expected"] += 1
            st["expected"] += 1
            if key in model:
                tot["detected"] += 1
                st["detected"] += 1
                v = model[key].get("verdict")
                st["verdicts"][v] += 1
                if v not in QUIET:
                    tot["recognized"] += 1
                    st["recognized"] += 1
    return {
        "dataset": "adversarial_human_designed",
        "queries": len(draft["items"]),
        "expected_conflicts": tot["expected"],
        "conflict_recall": round(tot["detected"] / tot["expected"], 4),
        "conflict_recognition_rate": round(tot["recognized"] / tot["expected"], 4),
        "by_group": {
            g: {
                "expected": v["expected"],
                "detected": v["detected"],
                "recognized": v["recognized"],
                "recognition_rate": round(v["recognized"] / v["expected"], 4) if v["expected"] else None,
                "verdicts": dict(v["verdicts"]),
            }
            for g, v in sorted(groups.items())
        },
    }


def main() -> None:
    doc = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "note": "自然语料与对抗集分别计算，禁止混合统计总体指标。",
        },
        "natural": natural_metrics(),
        "adversarial": adversarial_metrics(),
    }
    out = os.path.join(BASE, "results", "layered_conflict_metrics.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print("written:", out)
    print(json.dumps(doc, ensure_ascii=False, indent=2)[:2500])


if __name__ == "__main__":
    main()
