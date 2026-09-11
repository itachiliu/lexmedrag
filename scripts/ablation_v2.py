"""与主结果同口径的消融：三种判定配置在扩展对抗集（641 个预期冲突对）上的识别率。

三种配置从同一批模型输出派生，只改变冲突判定的规则层：
  A  跨法域规则对一律判定为义务并存
  B  保留模型对跨法域规则对的原始裁决（不做跨法域归一化），其余归一化保留
  C  完整方法（当前主结果）

用法：
    python scripts/ablation_v2.py
输出：
    results/ablation_v2.json
"""

from __future__ import annotations

import collections
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from lexmedrag import corpus  # noqa: E402

QUIET = {"cumulative", "not_conflict"}
CROSS_FLAGS = {
    "normalized_cross_jurisdiction_to_cumulative",
    "cross_jurisdiction_exclusion_relation",
}
GROUPS = ["C1", "C2", "C3", "C4", "C5", "C6"]


def is_cross_jurisdiction(rule_a, rule_b) -> bool:
    ja = set(rule_a.jurisdiction_scope or [rule_a.region]) if rule_a else set()
    jb = set(rule_b.jurisdiction_scope or [rule_b.region]) if rule_b else set()
    return ja != jb


def main() -> None:
    rules = {r.rule_id: r for r in corpus.load_rules()}
    doc = json.load(io.open(
        os.path.join(ROOT, "results", "adversarial_draft_v2_lexmedrag_run1.json"),
        encoding="utf-8",
    ))

    stats = {c: {"total": 0, "hit": 0, "by_group": collections.Counter(),
                 "by_group_total": collections.Counter()}
             for c in ("A", "B", "C")}

    for item in doc["items"]:
        group = item.get("adversarial_group", "?")
        model = {}
        for c in item["conflicts_ai_draft"]:
            if "error" in c:
                continue
            key = tuple(sorted((c.get("left_rule_id"), c.get("right_rule_id"))))
            model[key] = c
        for pair in item.get("expected_conflicts", []):
            key = tuple(sorted(pair))
            entry = model.get(key)
            rule_a, rule_b = rules.get(key[0]), rules.get(key[1])
            cross = is_cross_jurisdiction(rule_a, rule_b)
            for config in ("A", "B", "C"):
                stats[config]["total"] += 1
                stats[config]["by_group_total"][group] += 1
                if entry is None:
                    continue  # 未列出 → 漏检
                if config == "A":
                    verdict = "cumulative" if cross else entry.get("verdict")
                elif config == "B":
                    flags = set(entry.get("_flags", []))
                    if cross and (flags & CROSS_FLAGS):
                        verdict = entry.get("_raw_verdict") or entry.get("verdict")
                    else:
                        verdict = entry.get("verdict")
                else:
                    verdict = entry.get("verdict")
                if verdict not in QUIET:
                    stats[config]["hit"] += 1
                    stats[config]["by_group"][group] += 1

    out = {
        "meta": {
            "dataset": "adversarial v2 (240 queries / 641 expected pairs)",
            "note": "三种配置由同一批模型输出派生，仅改变判定规则层，与主结果同口径。",
        },
        "configs": {},
    }
    for config, label in (("A", "跨法域一律判并存"),
                          ("B", "保留跨法域原始裁决"),
                          ("C", "完整方法")):
        total = stats[config]["total"]
        hit = stats[config]["hit"]
        out["configs"][config] = {
            "label": label,
            "recognition": round(hit / total, 4) if total else None,
            "hits": hit,
            "total": total,
            "by_group": {
                g: (round(stats[config]["by_group"][g] / stats[config]["by_group_total"][g], 4)
                    if stats[config]["by_group_total"][g] else None)
                for g in GROUPS
            },
        }
    path = os.path.join(ROOT, "results", "ablation_v2.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for config, payload in out["configs"].items():
        print(config, payload["label"], "recognition =",
              f"{payload['recognition']*100:.1f}%", f"({payload['hits']}/{payload['total']})")
    print("written:", path)


if __name__ == "__main__":
    main()
