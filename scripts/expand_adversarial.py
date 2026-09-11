"""把人工对抗集从每组 5--8 条参数化扩展到每组 40 条。

保持专家设计的冲突结构与预期冲突对不变，只变换触发事实（机构、数据规模、
数据类型组合、业务类型、法域组合、触发条件），用于提高分组统计功效。

用法：
    python scripts/expand_adversarial.py

输出：
    results/adversarial_queries_v2.json
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_GROUP = 40

AUTHORITIES = ["FDA", "EMA", "香港卫生署"]
ATTR_SETS = [
    ["个例安全性报告（ICSR）原始记录", "受试者诊断记录（ICD-10）", "检验检查数据"],
    ["个例安全性报告（ICSR）", "不良事件报告", "用药记录"],
    ["ICSR 原始记录", "影像报告", "住院病历"],
    ["ICSR 记录", "基因测序数据", "检验检查数据"],
]
UPSTREAM = ["EU", "US", "JP"]
BUSINESS = ["pharmacovigilance", "telemedicine", "referral"]
EMERGENCY_ATTRS = [
    ["基因测序数据", "严重不良事件报告", "检验检查数据"],
    ["全外显子测序数据", "严重不良事件报告", "用药记录"],
    ["人类遗传资源信息", "不良事件报告", "住院病历"],
]


def build(rng: random.Random) -> list[dict]:
    queries: list[dict] = []
    qid = 0

    def add(group, src, dst, act, vol, attrs, expected, resolution, extra):
        nonlocal qid
        queries.append({
            "query_id": f"GBAMC-ADV2-{group}-{qid:04d}",
            "scenario": f"adversarial_{group.lower()}_{act}",
            "status": "adversarial_synthetic_requires_review",
            "adversarial_group": group,
            "X_target": attrs,
            "C_op": dict({
                "loc_src": src, "loc_dst": dst, "act": act, "vol": vol,
                "t": (f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
                      f"T{rng.randint(0, 23):02d}:00:00"),
            }, **extra),
            "expected_conflicts": [list(p) for p in expected],
            "expected_resolution": resolution,
            "subject_profile": {
                "is_ciio": rng.random() < 0.1,
                "cumulative_annual_volume": vol + rng.randint(0, 80000),
                "basis": "synthetic_deterministic",
                "note": "对抗集合成主体画像，待审核",
            },
        })
        qid += 1

    # C1 阻断条款 vs 境外监管检查
    for i in range(PER_GROUP):
        authority = AUTHORITIES[i % len(AUTHORITIES)]
        dst = {"FDA": "US", "EMA": "EU", "香港卫生署": "HK"}[authority]
        foreign = authority in ("FDA", "EMA")
        regime = "FDA_21CFR314_80_001" if authority == "FDA" else "EU_GVP_MODULE_VI_001"
        expected = [] if not foreign else [
            ("CN_DSL_Art36_001", regime),
            ("CN_PIPL_Art41_001", regime),
            ("CN_DSL_Art36_001", "ICH_E2B_R3_001"),
        ]
        add("C1", "CN", dst, "regulatory_inspection", 30 + i * 41,
            ATTR_SETS[i % len(ATTR_SETS)], expected,
            "主管机关审批前置；去标识化后提供；境内查阅不导出",
            {"authority": authority, "foreign_authority": foreign,
             "requester_type": "regulatory_authority"})

    # C2 GBA 禁转条款 vs 全球药物警戒数据库
    for i in range(PER_GROUP):
        dst1 = "HK" if i % 2 == 0 else "MO"
        upstream = UPSTREAM[i % len(UPSTREAM)]
        guide = "CN_GBA_HK_GUIDE_Art4" if dst1 == "HK" else "CN_GBA_MO_GUIDE_Art4"
        expected = [(guide, "EU_GVP_MODULE_VI_001"),
                    (guide, "ICH_E2B_R3_001"),
                    (guide, "FDA_21CFR314_80_001")]
        add("C2", "CN", dst1, "pharmacovigilance", 500 + i * 137,
            ATTR_SETS[i % len(ATTR_SETS)], expected,
            "放弃 GBA 通道改走标准合同或安全评估；或香港侧数据分流仅上传聚合信号",
            {"upstream_recipient": dst1, "onward_transfer_to": upstream,
             "global_safety_database": True, "recipient_is_group_parent": True})

    # C3 紧急豁免 vs 人类遗传资源审批
    for i in range(PER_GROUP):
        add("C3", "CN", "HK", "emergency_genetic_testing", 20 + i * 13,
            EMERGENCY_ATTRS[i % len(EMERGENCY_ATTRS)],
            [("CN_OUTBOUND_Art5_5_001", "CN_HGR_Art28_001"),
             ("CN_OUTBOUND_Art5_5_001", "CN_HGR_RULES_Art36_001"),
             ("CN_OUTBOUND_Art5_5_001", "CN_HGR_Art7_001")],
            "紧急豁免仅豁免出境路径义务，人遗备案/备份/安全审查无紧急例外",
            {"emergency": True, "public_health_risk": True,
             "recipient_type": "HK_laboratory"})

    # C4 删除权 vs 法定留存
    for i in range(PER_GROUP):
        src = "CN" if i % 2 == 0 else "HK"
        dst = "HK" if src == "CN" else "CN"
        add("C4", src, dst, "erasure_request", 1 + i % 5,
            ["受试者身份标识", "不良事件报告", "用药记录"],
            [("CN_PIPL_Art47_001", "CN_PV_RETENTION_001"),
             ("HK_PDPO_DPP2_001", "CN_PV_RETENTION_001"),
             ("CN_PIPL_Art47_001", "CN_ADR_MEASURES_RETENTION_001")],
            "内地侧依法定留存化解；香港侧取决于“法律”是否涵盖内地法规，属场景依赖",
            {"consent_withdrawn": True, "retention_obligation": "pharmacovigilance_records"})

    # C5 澳门适当性认定 vs GBA 便利化推定
    for i in range(PER_GROUP):
        act = BUSINESS[i % len(BUSINESS)]
        add("C5", "CN", "MO", act, 300 + i * 219,
            ["诊断记录（ICD-10）", "检验检查数据", "用药记录"],
            [("CN_GBA_MO_GUIDE_Art2", "MO_PDPA_Art19_001"),
             ("CN_GBA_MO_GUIDE_Art2", "MO_PDPA_Art19_002")],
            "GPDP 未作适当性认定时须逐案许可或适用第20条例外",
            {"gpdp_adequacy_decision": False,
             "host_authority_requirement": "case_by_case_authorisation"})

    # C6 次级利用的同意口径冲突
    for i in range(PER_GROUP):
        attrs = ["诊断记录（ICD-10）", "检验检查数据",
                 "基因测序数据" if i % 3 == 0 else "用药记录"]
        add("C6", "CN", "HK", "secondary_use", 150 + i * 97, attrs,
            [("CN_PIPL_Art14_001", "HK_PDPO_CONSENT_DEF_001"),
             ("CN_PIPL_Art39_001", "HK_PDPO_CONSENT_DEF_001"),
             ("CN_OUTBOUND_Art5_1_001", "CN_PIPL_Art39_001")],
            "同意要件取两地较严者：书面可撤回的订明同意 + 内地单独同意",
            {"purpose_change": True, "contract_necessity_claimed": True,
             "dual_jurisdiction_subjects": True})

    return queries


def main() -> None:
    rng = random.Random(20260910)
    queries = build(rng)
    doc = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "dataset": "adversarial_conflict_set_v2",
            "status": "human_designed_requires_expert_review",
            "per_group": PER_GROUP,
            "note": ("由 v1 人工设计模板参数化扩展：冲突结构与预期对不变，"
                     "仅变换触发事实；仍为人工构造，不得与自然语料混合统计。"),
        },
        "queries": queries,
    }
    out = os.path.join(ROOT, "results", "adversarial_queries_v2.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    groups = {}
    pairs = 0
    for q in queries:
        groups[q["adversarial_group"]] = groups.get(q["adversarial_group"], 0) + 1
        pairs += len(q["expected_conflicts"])
    print("written:", out)
    print("queries:", len(queries), "| by group:", groups, "| expected pairs:", pairs)


if __name__ == "__main__":
    main()
