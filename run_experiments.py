"""Run real-data experiments for the LexMedRAG pipeline.

Usage:
    python run_experiments.py card
    python run_experiments.py retrieval [--limit N]
    python run_experiments.py selftest
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import io
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from lexmedrag import arbitrate, corpus, llm, metrics, retrieval  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")

# 封闭枚举：冲突类型与裁决结论（避免自由文本枚举失控）
CONFLICT_TYPES = (
    "permission_vs_prohibition",   # 许可与禁止
    "obligation_vs_permission",    # 义务与许可
    "general_vs_exception",        # 一般与例外
    "mandatory_vs_optional",       # 强制与可选
    "parallel_obligations",        # 并行义务（不构成冲突）
    "element_relationship",        # 构成要件关系（要件是否成就，非义务冲突）
)
VERDICTS = (
    "left_suppresses_right",
    "right_suppresses_left",
    "exclude_left",
    "exclude_right",
    "cumulative",                  # 义务并存、按孰严执行（跨法域唯一合法结论）
    "not_conflict",
    "pending_stance",              # 官方口径未明，暂按保守累积（待团队立场）
    "not_competing",               # 不竞合：要件互斥/适用条件不同，不产生压制关系
)
CONFLICT_TYPE_ZH = {
    "permission_vs_prohibition": "许可与禁止",
    "obligation_vs_permission": "义务与许可",
    "general_vs_exception": "一般与例外",
    "mandatory_vs_optional": "强制与可选",
    "parallel_obligations": "并行义务",
    "element_relationship": "构成要件关系",
}
VERDICT_ZH = {
    "left_suppresses_right": "左压制右",
    "right_suppresses_left": "右压制左",
    "exclude_left": "排除左",
    "exclude_right": "排除右",
    "cumulative": "并存（按孰严执行）",
    "not_conflict": "无冲突",
    "pending_stance": "待定（口径未明）",
    "not_competing": "不竞合（要件互斥）",
}


def _dump(name: str, payload: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("written:", path)
    return path


def cmd_card() -> None:
    from collections import Counter

    rules = corpus.load_rules()
    qa = corpus.load_qa()
    law_cnt = Counter(r.law for r in rules)
    cat_cnt = Counter(r.rule_category for r in rules)
    sens = [r.sensitivity_score for r in rules]
    type_cnt = Counter((q.get("category"), q.get("type")) for q in qa)
    payload = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "data_files": {
                "rules": "data/legal_knowledge_base.json",
                "qa": "data/merge.json",
            },
            "note": "全部数值由真实文件统计得到。",
        },
        "rules": {
            "total": len(rules),
            "by_law": dict(law_cnt),
            "by_category": dict(cat_cnt),
            "sensitivity": {
                "min": min(sens),
                "max": max(sens),
                "mean": round(statistics.mean(sens), 4),
            },
        },
        "qa": {
            "total": len(qa),
            "by_category_type": {f"{k[0]}|{k[1]}": v for k, v in type_cnt.most_common()},
        },
    }
    _dump("dataset_card.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:2000])


def cmd_retrieval(limit: int | None) -> None:
    rules = corpus.load_rules()
    qa = corpus.load_qa()
    if limit:
        qa = qa[:limit]
    bm25 = retrieval.BigramBM25([r.text for r in rules])

    gold_sizes = []
    rows = []
    r5s, n5s = [], []
    for item in qa:
        gold = corpus.gold_rules_for_qa(item, rules)
        if not gold:
            continue
        gold_sizes.append(len(gold))
        gold_set = set(gold)
        ranked_ids = [
            rules[i].rule_id for i, _ in bm25.rank(item.get("question", ""), top_k=5)
        ]
        r5 = metrics.recall_at_k(ranked_ids, gold_set, k=5)
        n5 = metrics.ndcg_at_k(ranked_ids, gold_set, k=5)
        r5s.append(r5)
        n5s.append(n5)
        rows.append(
            {
                "question": item.get("question", "")[:120],
                "gold": gold,
                "retrieved_top5": ranked_ids,
                "recall@5": round(r5, 4),
                "ndcg@5": round(n5, 4),
            }
        )

    payload = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "method": (
                f"character-bigram BM25 over {len(rules)} rules "
                "(CN + GBA/HK/MO draft expansion); gold matched by law+article citations"
            ),
            "note": "检索 pilot：结果只在真实知识库上可复现，不能直接充当论文实验数据；"
                    "gold 匹配为启发式近似，正式实验需专家标注。",
        },
        "queries_total": len(qa),
        "queries_with_gold": len(rows),
        "gold_size": {
            "min": min(gold_sizes) if gold_sizes else 0,
            "max": max(gold_sizes) if gold_sizes else 0,
            "mean": round(statistics.mean(gold_sizes), 4) if gold_sizes else 0,
        },
        "metrics": {
            "recall@5": round(statistics.mean(r5s), 4) if r5s else None,
            "ndcg@5": round(statistics.mean(n5s), 4) if n5s else None,
        },
    }
    _dump("retrieval_pilot.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_selftest() -> None:
    r = arbitrate.resolve(
        {"rule_id": "CN_PIPL_Art40_001", "issue_date": "2025-09-08"},
        {"rule_id": "CN_GDHMD_002", "issue_date": "2025-09-08"},
    )
    assert r.basis == "lex_superior" and r.winner == "CN_PIPL_Art40_001"
    r2 = arbitrate.resolve(
        {"rule_id": "CN_GDHMD_010", "issue_date": "2024-01-01"},
        {"rule_id": "CN_GDHMD_011", "issue_date": "2025-01-01"},
    )
    assert r2.basis == "lex_posterior" and r2.winner == "CN_GDHMD_011"
    gold = {("a", "b"): "a_suppresses_b"}
    assert metrics.crr_for_query(gold, {}) == 0.0
    assert metrics.crr_for_query(gold, {("a", "b"): "a_suppresses_b"}) == 1.0
    assert abs(metrics.risk_mae([0.9], [0.8]) - 0.1) < 1e-9
    print("selftest OK: hierarchy / lex posterior / CRR / risk MAE")


def cmd_llm_check() -> None:
    print("calling DeepSeek (max_tokens=8) ...")
    try:
        r = llm.chat("只回复两个字符：ok", max_tokens=8)
        print("DeepSeek OK, reply:", r.strip())
    except Exception as exc:  # noqa: BLE001
        print("DeepSeek check failed:", type(exc).__name__, str(exc)[:300])


def _candidate_pairs(rules: list[corpus.Rule], limit: int) -> list[tuple]:
    from lexmedrag.text_utils import tokens

    token_sets = [set(tokens(r.rule_text)) for r in rules]
    scored = []
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            if rules[i].law == rules[j].law:
                continue
            overlap = len(token_sets[i] & token_sets[j])
            if overlap >= 2:
                scored.append((overlap, i, j))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(rules[i], rules[j]) for _, i, j in scored[:limit]]


def _conflict_prompt(left: corpus.Rule, right: corpus.Rule) -> str:
    return (
        "你是研究中国内地、香港、澳门医疗数据跨境合规的法律专家。请判断下列两条规则"
        "在同一医疗数据跨境处理场景中是否构成规范冲突，并按给定 JSON 输出。\n"
        "规则字段：rule_id、法律名称、效力层级（同一法域内比较；跨法域层级不可直接比较）、"
        "生效日期、规则文本。港澳背景：香港 PDPO 第33条尚未生效；"
        "澳门 PDPA 第19/20条规定跨境移转一般标准与例外。\n"
        f"LEFT: id={left.rule_id}, law={left.law}, H={arbitrate.level(left.rule_id)}, "
        f"date={left.issue_date}, text={left.rule_text}\n"
        f"RIGHT: id={right.rule_id}, law={right.law}, H={arbitrate.level(right.rule_id)}, "
        f"date={right.issue_date}, text={right.rule_text}\n"
        "输出严格 JSON："
        '{"conflict": true/false, '
        '"conflict_type": "许可与禁止"|"义务与许可"|"一般与例外"|"强制与可选"|"无冲突", '
        '"verdict": "left_suppresses_right"|"right_suppresses_left"|"exclude_left"|'
        '"exclude_right"|"cumulative"|"not_conflict", '
        '"basis": "lex_specialis"|"lex_superior"|"lex_posterior"|"contextual"|"none", '
        '"reason": "不超过120字的中文理由（引用条文与法理）"}'
    )


def cmd_gold_draft(pairs: int, queries: int) -> None:
    rules = corpus.load_rules()
    qa = corpus.load_qa()

    conflict_rows = []
    for left, right in _candidate_pairs(rules, pairs):
        try:
            out = llm.chat_json(_conflict_prompt(left, right))
        except Exception as exc:  # noqa: BLE001
            out = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        conflict_rows.append(
            {
                "left_rule_id": left.rule_id,
                "right_rule_id": right.rule_id,
                "status": "ai_draft",
                **out,
            }
        )
    _dump(
        "draft_gold_conflicts.json",
        {
            "meta": {
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "model": "deepseek-chat",
                "status": "ai_draft_requires_expert_review",
                "note": "LLM 初判仅用于辅助标注。在专家审核并修改前，不得作为论文的 gold。",
            },
            "items": conflict_rows,
        },
    )

    selected_types = {"Cross-border", "跨境传输", "数据共享", "法律责任"}
    selected = [q for q in qa if q.get("type") in selected_types][:queries]
    query_rows = []
    for idx, item in enumerate(selected):
        gold = corpus.gold_rules_for_qa(item, rules)
        sens = [r.sensitivity_score for r in rules if r.rule_id in set(gold)]
        query_rows.append(
            {
                "query_id": f"qa_{idx:04d}",
                "question": item.get("question", ""),
                "category": item.get("category", ""),
                "type": item.get("type", ""),
                "gold_rules_draft": gold,
                "risk_draft": round(max(sens), 4) if sens else None,
                "risk_draft_method": "max(sensitivity_score of citation-matched rules)",
                "status": "draft_requires_expert_review",
            }
        )
    _dump(
        "draft_gold_queries.json",
        {
            "meta": {
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "note": "gold_rules 来自真实引用自动匹配，risk_draft 为派生草稿；"
                        "均需专家审核后才可作为论文 gold。",
            },
            "items": query_rows,
        },
    )
    n_conflict = sum(1 for r in conflict_rows if r.get("conflict") is True)
    print(
        f"gold draft done: {len(conflict_rows)} conflict candidates "
        f"({n_conflict} flagged conflict), {len(query_rows)} query drafts -> results/"
    )


def _query_pool(qa_queries: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for q in qa_queries:
        op = q["C_op"]
        key = f"{op['loc_src']}_{op['loc_dst']}"
        groups.setdefault(key, []).append(q)
    return groups


# 出境制度：仅规范“由内地向境外提供”，入境方向不触发
CN_OUTBOUND_RULES = (
    "CN_PIPL_Art38", "CN_PIPL_Art39", "CN_PIPL_Art40", "CN_OUTBOUND",
)
# 内地境内处理规则（含敏感个人信息与处理合法性基础），入境方向适用
CN_DOMESTIC_CORE = (
    "CN_SPI_107", "CN_SPI_109", "CN_SPI_113", "CN_SPI_121",
    "CN_SPI_122", "CN_SPI_125", "CN_SPI_128", "CN_SPI_130",
)
GENETIC_HINTS = ("基因", "测序", "基因组", "转录组", "遗传", "组学")
# 医疗健康类敏感个人信息线索（用于数量门槛校验）
SENSITIVE_HINTS = (
    "诊断", "检验", "检查", "用药", "基因", "基因组", "测序", "影像",
    "不良事件", "生物识别", "面部", "指纹", "心率", "血氧", "医保",
    "传染病", "孕产", "手术", "处方", "医疗",
)


def _applicable_candidates(q: dict, rules: list[corpus.Rule], cap: int = 24) -> list[corpus.Rule]:
    from lexmedrag.text_utils import tokens

    op = q["C_op"]
    attr_text = "".join(q["X_target"])
    ats = set(tokens(attr_text))
    src, dst = op["loc_src"], op["loc_dst"]
    picked: list[corpus.Rule] = []
    picked_ids: set[str] = set()

    def _add(r: corpus.Rule) -> None:
        if r.rule_id not in picked_ids and len(picked) < cap:
            picked.append(r)
            picked_ids.add(r.rule_id)

    def _pick(prefixes: tuple[str, ...]) -> None:
        for r in rules:
            if r.rule_id.startswith(prefixes):
                _add(r)

    genetic = any(h in attr_text for h in GENETIC_HINTS)
    act = op.get("act", "")
    authority = str(op.get("authority", ""))
    # 0) 对抗性场景的境外监管义务：药物警戒报告与监管检查
    if act in ("pharmacovigilance", "regulatory_inspection", "safety_reporting",
               "global_safety_database", "secondary_use", "emergency_genetic_testing"):
        _pick(("EU_GVP", "ICH_E2B", "FDA_21CFR"))
    # 0b) 阻断条款：外国司法或执法机构要求提供境内存储的数据
    if src == "CN" and (
        op.get("foreign_authority") or authority.upper().startswith(("FDA", "EMA"))
    ):
        _pick(("CN_DSL_Art36", "CN_PIPL_Art41"))
    # 0c) 删除请求与法定留存义务并存
    if act in ("erasure_request", "data_retention", "consent_withdrawal"):
        _pick(("CN_PIPL_Art47", "CN_PIPL_Art13_3", "CN_PV", "CN_ADR", "HK_PDPO_DPP2"))
    # 0d) 新目的/次级利用：同意构成要件跨法域比较
    if act == "secondary_use" or op.get("purpose_change"):
        _pick(("CN_PIPL_Art14", "HK_PDPO_CONSENT_DEF"))
    # 0e) 人类遗传资源信息对外提供：关键规则，优先纳入以避免被候选上限截断
    if genetic and src == "CN":
        _pick(("CN_HGR",))
    # 1) 源法域本地规则：数据在源法域被收集与处理。
    if src == "HK":
        _pick(("HK_PDPO",))
    if src == "MO":
        _pick(("MO_PDPA",))
    # 2) 出境制度：仅当数据由内地向港澳提供时适用（否则入境被误引入境评估）。
    if src == "CN" and dst != "CN":
        _pick(CN_OUTBOUND_RULES)
    # 3) GBA 标准合同实施指引：规范内地向港澳提供，仅出境方向适用。
    if src == "CN" and dst == "HK":
        _pick(("CN_GBA_HK",))
    elif src == "CN" and dst == "MO":
        _pick(("CN_GBA_MO",))
    # 4) 目的地法域规则：接收方后续处理受目的地法域规管。
    if dst == "HK":
        _pick(("HK_PDPO",))
    if dst == "MO":
        _pick(("MO_PDPA",))
    # 5) 入境方向：适用内地境内处理规则，但不含出境条款。
    if dst == "CN" and src != "CN":
        _pick(CN_DOMESTIC_CORE)
        _pick(("CN_CSSPG",))
    # 7) 属性命中补充：以操作涉及的数据属性匹配其余规则（含 GDHMD 等）。
    scored = []
    for r in rules:
        if r.rule_id in picked_ids:
            continue
        # 入境方向：出境制度与人类遗传资源对外提供条款均不适用
        if dst == "CN" and src != "CN" and r.rule_id.startswith(CN_OUTBOUND_RULES + ("CN_HGR",)):
            continue
        ov = len(ats & set(tokens(r.rule_text)))
        if ov >= 2:
            scored.append((ov, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    for _, r in scored:
        _add(r)
    # 固定按 rule_id 排序：保证 left/right 顺序稳定，避免同案不同判的顺序漂移
    return sorted(picked, key=lambda r: r.rule_id)


def _conflict_prompt_q(q: dict, cands: list[corpus.Rule]) -> str:
    lines = []
    for r in cands:
        txt = r.rule_text[:180] + ("…" if len(r.rule_text) > 180 else "")
        lines.append(
            f"{r.rule_id} | {r.law} | norm={r.norm_type} | H={arbitrate.level(r.rule_id)} | "
            f"jurisdiction={r.jurisdiction_scope} | {txt}"
        )
    return (
        "你是研究中国内地、香港、澳门（粤港澳大湾区）医疗数据跨境合规的法律专家。"
        "给定一次具体数据操作与一组可能适用的真实规则，判断哪些规则对在【该操作情境下】"
        "构成规范冲突，并给出裁决。\n"
        "操作 Q: " + json.dumps(q, ensure_ascii=False) + "\n"
        "候选规则（已按 rule_id 排序；norm 为规范性质，H 为该规则在其所属法域内的效力位阶）：\n"
        + "\n".join(lines) + "\n"
        "判定要求（逐条遵守）：\n"
        "1) 适用方向优先：内地出境制度（《个人信息保护法》第38/39/40条、"
        "《促进和规范数据跨境流动规定》）只规范“由内地向境外提供”。数据由香港或澳门"
        "向内地提供的（入境），这些规则不适用，不得据此要求安全评估或境内存储；"
        "GBA 标准合同实施指引同样只适用于内地向港澳提供。\n"
        "2) 数量门槛（仅出境方向）：第5条豁免限于“累计向境外提供不满10万人个人信息"
        "（不含敏感个人信息）”且不含重要数据；只要涉及敏感个人信息，第5条豁免一律不适用，"
        "应按第7条（1万人以上敏感个人信息→安全评估）或第8条（不满1万人敏感个人信息→"
        "标准合同或认证）判断。必须直接引用操作中给定的人数，不得自行重新计算或改述人数。\n"
        "3) 重要数据为告知/公布制：未被相关部门、地区告知或公开发布为重要数据的，"
        "不按重要数据申报；不得仅因数据类型（如基因、医疗数据）推定其为重要数据。\n"
        "4) 人类遗传资源：涉及基因测序等人类遗传资源信息向港澳提供的，应适用"
        "《人类遗传资源管理条例》第28条及实施细则第36条（事先备案并提交信息备份；"
        "可能影响公众健康、国家安全和社会公共利益的，须经安全审查）。该条同样只规范"
        "“由内地向外提供”，数据由港澳向内地提供的（入境）不适用，不得据此要求备案或"
        "信息备份。\n"
        "5) 规范性质：团体标准与推荐性技术指南（如广东省健康医疗数据安全分类分级管理"
        "技术规范）不是法规，不得以 lex_specialis、lex_superior 或“4级数据需国家审批”"
        "等表述压制、替代法律、行政法规与部门规章；现行法不存在以数据分级为触发要件的"
        "“国家审批”制度。\n"
        "6) 跨法域规则：不同法域的效力位阶不可相互比较，不得用 lex_superior 表示一方压制另一方。"
        "但跨法域义务可能相互排斥，此时构成真冲突，必须如实报出并给出消解结论，例如："
        "（a）一方要求提供数据、另一方禁止提供（如内地阻断条款与境外监管机构的提供要求）；"
        "（b）一方要求删除、另一方要求长期保存；"
        "（c）一方要求向外报告、另一方禁止再转移（如 GBA 禁转条款与全球药物警戒报告义务）。"
        "此类情形的 verdict 应给出处置结论（经主管机关批准、去标识化后提供、境内查阅不导出、"
        "数据分流或改走其他合规路径），并在 reason 中写明消解路径；"
        "只有义务不相互排斥时（例如两地各自要求告知同意、各自要求安全保障）才使用 cumulative。\n"
        "6b) 定义性冲突：当两法域对同一法律概念（如同意、敏感个人信息、留存期限）的"
        "构成要件不一致，导致无法以同一文件或同一处理同时满足两地要求，或分别满足会导致"
        "目的范围、口径相互冲突时，构成定义性冲突，应报出并在 reason 中说明差异与可行的"
        "统一方案（例如取两地较严要件）。不得因“两地在各自法域内均可履行”就径直判为并存。\n"
        "6c) 目的地法域的程序前置：若操作事实表明目的地法域主管机关尚未作出适当性认定，"
        "或明确要求逐案许可/个案授权，则内地的便利化安排不能替代该前置程序；此时应报为"
        "冲突或场景依赖（并说明触发条件），不得径直判为并存。\n"
        "7) conflict_type 只能取：permission_vs_prohibition（许可与禁止）、"
        "obligation_vs_permission（义务与许可）、general_vs_exception（一般与例外）、"
        "mandatory_vs_optional（强制与可选）、parallel_obligations（并行义务，不构成冲突）。\n"
        "8) verdict 只能取：left_suppresses_right、right_suppresses_left、exclude_left、"
        "exclude_right、cumulative、not_conflict；当 conflict_type 为 parallel_obligations 时，"
        "verdict 必须为 cumulative 或 not_conflict。\n"
        "9) 只有对同一操作同时适用、且义务相互排斥/矛盾时才算冲突；仅并行义务不算。\n"
        "10) GBA 数量豁免争议（待团队立场统一，暂按并存处理）：香港官方简介材料明确"
        "大湾区标准合同实施指引对《个人信息出境标准合同办法》第4条的数量限制作了豁免，"
        "即主体注册于内地九市或港澳、且数据未被告知或公布为重要数据时，敏感个人信息"
        "超过1万人仍可适用 GBA 标准合同；《促进和规范数据跨境流动规定》晚于港版指引出台，"
        "两者衔接官方未再明文，业内存在分歧。涉及该问题时须在 reason 中同时说明两种口径，"
        "verdict 优先给出 cumulative，不得单方面判定一方压制另一方。\n"
        "11) 豁免边界：即使第5条豁免成立，也仅免除数据出境安全评估、订立标准合同、"
        "个人信息保护认证三项，绝不免除《个人信息保护法》第39条的告知与单独同意义务，"
        "也不豁免人类遗传资源信息的备案与信息备份义务。\n"
        "12) GBA 标准合同实施指引第4条的告知同意要求不得低于 PIPL 第39条的单独同意标准；"
        "澳门《个人资料保护法》第20条的明确同意是澳门法下的转移例外，不能替代内地"
        "出境路径下的单独同意等义务。\n"
        "13) 主体身份要件：《个人信息保护法》第40条以“关键信息基础设施运营者(CIIO)”"
        "身份或累计处理个人信息达到规定数量为要件。操作中提供 subject_profile 时以其为准；"
        "未标明时不得径行推定第40条不适用，应就该项要件作保留说明。若属 CIIO 或达量，"
        "第40条的境内存储与安全评估义务优先，GBA 标准合同指引不豁免该义务。\n"
        "14) 双轨备案：人类遗传资源信息对外提供向科技行政部门备案并提交信息备份，"
        "与网信部门或 GBA 标准合同备案是并行程序，互不替代。\n"
        "15) 推荐性技术文件（团体标准、技术指南）仅供数据分级参考，不创设法律义务，"
        "不得作为审批、压制或排除其他规则的依据；同法域内也不得据其判定压制关系。\n"
        "16) basis 取 lex_specialis/lex_superior/lex_posterior/contextual/none；"
        "上位法律与其出境门槛具体化条款之间应记为 lex_specialis（同一义务的双重基础），"
        "不得记为 lex_superior。\n"
        "输出严格 JSON：{\"assessment\":\"≤150字对该操作的义务适用与冲突总体判断\","
        "\"conflicts\":[{\"left_rule_id\":\"...\","
        "\"right_rule_id\":\"...\",\"conflict_type\":\"...\",\"verdict\":\"...\","
        "\"basis\":\"...\",\"reason\":\"≤120字中文理由\"}]}；无冲突则 {\"conflicts\":[]}。"
    )


def _norm_kind(rule: corpus.Rule | None) -> str:
    return rule.norm_type if rule else ""


def _threshold_text_flags(text: str, vol, sensitive: bool) -> list[str]:
    """检测摘要/理由中与输入人数矛盾的数量门槛陈述。

    只在文本确实引用了本查询人数、且在其邻近窗口给出相反门槛表述时判定，
    避免把“复述规则条文的门槛”误判为对本案的陈述。
    """
    flags: list[str] = []
    if not isinstance(vol, int) or not text:
        return flags
    t = text.replace(",", "")
    vol_s = str(vol)
    if vol_s not in t:
        return flags
    for m in re.finditer(re.escape(vol_s), t):
        window = t[max(0, m.start() - 25): m.end() + 45]
        if sensitive and vol >= 10000 and any(
            k in window for k in ("未达1万", "不满1万", "不足1万", "未达到1万")
        ):
            flags.append("threshold_statement_contradiction")
        if vol >= 100000 and any(k in window for k in ("未达10万", "不满10万", "不足10万")):
            flags.append("threshold_statement_contradiction")
        if vol < 100000 and any(k in window for k in ("超过10万", "达到10万", "十万以上")):
            flags.append("threshold_statement_contradiction")
        if vol < 10000 and any(k in window for k in ("超过1万", "达到1万", "一万以上")):
            flags.append("threshold_statement_contradiction")
    return sorted(set(flags))


def _normalize_conflicts(
    q: dict, conflicts: list[dict], cands: list[corpus.Rule]
) -> tuple[list[dict], list[dict]]:
    """规范化 left/right 顺序并标记不合法的裁决，便于审核时优先筛查。"""
    by_id = {r.rule_id: r for r in cands}
    src = q["C_op"]["loc_src"]
    dst = q["C_op"]["loc_dst"]
    vol = q["C_op"].get("vol")
    attr_text = "".join(q["X_target"])
    sensitive = any(h in attr_text for h in SENSITIVE_HINTS)
    suppress_verdicts = (
        "left_suppresses_right", "right_suppresses_left",
        "exclude_left", "exclude_right",
    )
    out = []
    dropped = []
    for c in conflicts:
        a, b = c.get("left_rule_id"), c.get("right_rule_id")
        v, ct = c.get("verdict"), c.get("conflict_type")
        flags: list[str] = []
        # 敏感个人信息识别条目只决定“数据是否敏感”，不产生规范义务冲突
        if (a or "").startswith("CN_SPI_") and (b or "").startswith("CN_SPI_"):
            dropped.append(
                {
                    "left_rule_id": a,
                    "right_rule_id": b,
                    "reason": "same_identification_namespace",
                }
            )
            continue
        if ct not in CONFLICT_TYPES:
            flags.append("unknown_conflict_type")
        if v not in VERDICTS:
            # 兜底：模型偶发的同侧方向笔误，归一为合法枚举并留痕
            if v == "right_suppresses_right":
                v = "right_suppresses_left"
                flags.append("normalized_verdict_direction")
            elif v == "left_suppresses_left":
                v = "left_suppresses_right"
                flags.append("normalized_verdict_direction")
            else:
                flags.append("unknown_verdict")
        # 固定 left/right 为 rule_id 字典序，翻转时同步翻转压制方向
        if a and b and a > b:
            a, b = b, a
            flip = {
                "left_suppresses_right": "right_suppresses_left",
                "right_suppresses_left": "left_suppresses_right",
                "exclude_left": "exclude_right",
                "exclude_right": "exclude_left",
            }
            if v in flip:
                v = flip[v]
        ra, rb = by_id.get(a), by_id.get(b)
        # 方向性校验：内地出境规则不得用于入境方向
        if dst == "CN" and src != "CN":
            if any(x and x.startswith(CN_OUTBOUND_RULES) for x in (a, b)):
                flags.append("outbound_rule_in_inbound_direction")
        if src == "CN" and dst != "CN":
            if any(x and x.startswith(("HK_PDPO", "MO_PDPA")) for x in (a, b)):
                pass  # 目的地法域规则在出境方向是合法适用的
        if ra and rb:
            ja = set(ra.jurisdiction_scope or [ra.region])
            jb = set(rb.jurisdiction_scope or [rb.region])
            weak = ("团体标准", "推荐性", "技术指南")
            strong = ("法律", "行政法规", "部门规章")
            va = _norm_kind(ra)
            vb = _norm_kind(rb)
            weak_involved = any(k in va for k in weak) or any(k in vb for k in weak)
            if ja != jb:
                if v in suppress_verdicts:
                    if weak_involved:
                        flags.append("normalized_weak_norm_to_not_conflict")
                        v = "not_conflict"
                    elif ct == "parallel_obligations":
                        # 并行义务不构成压制：归一为并存
                        flags.append("normalized_cross_jurisdiction_to_cumulative")
                        v = "cumulative"
                    else:
                        # 排斥型关系（许可与禁止、义务冲突等）：保留冲突结论
                        flags.append("cross_jurisdiction_exclusion_relation")
            # 推荐性标准/技术指南不进入法源体系：同法域内也不得判为压制
            elif weak_involved and v in suppress_verdicts:
                flags.append("normalized_weak_norm_to_not_conflict")
                v = "not_conflict"
            if v == "left_suppresses_right" and any(k in va for k in weak) and any(k in vb for k in strong):
                flags.append("weak_norm_suppresses_strong_norm")
            if v == "right_suppresses_left" and any(k in vb for k in weak) and any(k in va for k in strong):
                flags.append("weak_norm_suppresses_strong_norm")
        if ct == "parallel_obligations" and v not in (
            "cumulative", "not_conflict", "pending_stance"
        ):
            flags.append("verdict_type_inconsistent")
        if ct == "element_relationship" and v not in ("not_competing", "not_conflict"):
            flags.append("verdict_type_inconsistent")
        # GBA 标准合同与内地出境数量门槛的衔接争议：仅“适用范围/豁免”条款（第2条）才涉数量门槛
        pair_ids = (a or "", b or "")
        if any(x.startswith("CN_GBA") and "_Art2" in x for x in pair_ids) and any(
            y.startswith(("CN_OUTBOUND_Art5", "CN_OUTBOUND_Art7",
                          "CN_OUTBOUND_Art8", "CN_PIPL_Art40"))
            for y in pair_ids
        ):
            flags.append("gba_quantitative_exemption_disputed")
            if v == "cumulative":
                v = "pending_stance"
        # 上位法与出境门槛条款为“同一义务的双重基础”，依据统一为 lex_specialis
        if any(x.startswith(("CN_PIPL_Art38", "CN_PIPL_Art39", "CN_PIPL_Art40")) for x in pair_ids) and any(
            y.startswith(("CN_OUTBOUND_Art5", "CN_OUTBOUND_Art7", "CN_OUTBOUND_Art8"))
            for y in pair_ids
        ):
            ct = "parallel_obligations"
            v = "not_conflict"
            if c.get("basis") == "lex_superior":
                flags.append("normalized_basis_to_lex_specialis")
        # 第5条豁免与敏感信息识别条目：要件关系，不得记为压制/排除
        if any(x.startswith("CN_OUTBOUND_Art5") for x in pair_ids) and any(
            x.startswith("CN_SPI_") for x in pair_ids
        ):
            ct = "element_relationship"
            v = "not_competing"
            flags.append("normalized_element_relationship")
        # 敏感个人信息达到 1 万人：第5条豁免不适用，不得让其压制安全评估/标准合同
        if sensitive and isinstance(vol, int) and vol >= 10000:
            a_is5 = (a or "").startswith("CN_OUTBOUND_Art5")
            b_is5 = (b or "").startswith("CN_OUTBOUND_Art5")
            if (a_is5 and v == "left_suppresses_right") or (b_is5 and v == "right_suppresses_left"):
                flags.append("art5_exemption_misapplied")
        # reason 与 verdict 自相矛盾（称“不矛盾”却给出压制/排除结论）
        reason = c.get("reason", "") or ""
        if v in suppress_verdicts and any(
            k in reason for k in ("不矛盾", "不构成冲突", "可同时", "并行适用", "并无冲突")
        ):
            flags.append("reason_verdict_contradiction")
        flags.extend(_threshold_text_flags(reason, vol, sensitive))
        row = {
            "left_rule_id": a,
            "right_rule_id": b,
            "conflict_type": ct,
            "verdict": v,
            "_raw_verdict": c.get("verdict"),
            "basis": c.get("basis"),
            "reason": c.get("reason", ""),
        }
        # 依据归一：推荐性文件不进入法源体系；非压制结论不得以效力位阶/后法为依据
        weak_involved = any(
            k in (_norm_kind(ra) + _norm_kind(rb))
            for k in ("团体标准", "推荐性", "技术指南")
        )
        if weak_involved and row["basis"] in ("lex_superior", "lex_specialis"):
            if "normalized_basis_to_none" not in flags:
                flags.append("normalized_basis_to_none")
            row["basis"] = "none"
        elif row["verdict"] in ("not_conflict", "not_competing") and row["basis"] in (
            "lex_superior", "lex_posterior"
        ):
            flags.append("normalized_basis_to_none")
            row["basis"] = "none"
        elif (
            any(x.startswith(("CN_PIPL_Art38", "CN_PIPL_Art39", "CN_PIPL_Art40"))
                for x in (a or "", b or ""))
            and any(y.startswith(("CN_OUTBOUND_Art5", "CN_OUTBOUND_Art7",
                                  "CN_OUTBOUND_Art8"))
                    for y in (a or "", b or ""))
        ):
            if row["basis"] in ("lex_superior", "lex_posterior"):
                flags.append("normalized_basis_to_lex_specialis")
                row["basis"] = "lex_specialis"
        if flags:
            row["_flags"] = flags
        out.append(row)
    return out, dropped


def _conflict_sample_one(
    q: dict, rules: list[corpus.Rule], cap: int = 24
) -> dict:
    cands = _applicable_candidates(q, rules, cap=cap)
    try:
        out = llm.chat_json(_conflict_prompt_q(q, cands))
        conflicts = [
            c for c in out.get("conflicts", [])
            if c.get("left_rule_id") and c.get("right_rule_id")
        ]
        conflicts, dropped = _normalize_conflicts(q, conflicts, cands)
        assessment = out.get("assessment", "")
    except Exception as exc:  # noqa: BLE001
        conflicts = [{"error": f"{type(exc).__name__}: {str(exc)[:200]}"}]
        assessment = ""
        dropped = []
    row = {
        "query_id": q["query_id"],
        "scenario": q.get("scenario", ""),
        "C_op": q["C_op"],
        "X_target": q["X_target"],
        "candidate_rules": [r.rule_id for r in cands],
        "conflicts_ai_draft": conflicts,
        "assessment": assessment,
        "status": "draft_requires_expert_review",
    }
    if q.get("adversarial_group"):
        row["adversarial_group"] = q["adversarial_group"]
        row["expected_conflicts"] = q.get("expected_conflicts", [])
        row["expected_resolution"] = q.get("expected_resolution", "")
    if dropped:
        row["dropped_pairs"] = dropped
    attr_text = "".join(q["X_target"])
    sens = any(h in attr_text for h in SENSITIVE_HINTS)
    vol = q["C_op"].get("vol")
    aflags = _threshold_text_flags(assessment, vol, sens)
    if aflags:
        row["assessment_flags"] = sorted(set(aflags))
    return row


def cmd_conflict_sample(
    n: int,
    workers: int = 4,
    queries_file: str = "results/gbamc_queries_v3.json",
    out_name: str = "draft_conflict_gold_query_sample.json",
) -> None:
    rules = corpus.load_rules()
    qa = corpus.load_qa()
    qpath = queries_file if os.path.isabs(queries_file) else os.path.join(BASE, queries_file)
    pool = _query_pool(json.load(io.open(qpath, encoding="utf-8"))["queries"])
    picks: list[dict] = []
    pool_total = sum(len(v) for v in pool.values())
    if n >= pool_total:
        for k in sorted(pool):
            picks.extend(pool[k])
    else:
        per = {k: max(1, n // len(pool)) for k in pool}
        import random

        rng = random.Random(20260910)
        for k in sorted(pool):
            picks.extend(rng.sample(pool[k], min(per[k], len(pool[k]))))
    selected = picks[:n]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(lambda q: _conflict_sample_one(q, rules), selected))
    _dump(
        out_name,
        {
            "meta": {
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "model": "deepseek-chat",
                "sample_size": len(rows),
                "workers": workers,
                "method": (
                    f"query-conditioned conflict draft over {len(rows)} queries and "
                    f"{len(rules)}-rule KB (CN/GBA/HK/MO/HGR/EU/US/ICH); "
                    "outbound rules restricted to CN→HK/MO, inbound uses CN domestic rules"
                ),
                "note": "AI 初判，仅作标注候选；专家审核前不进入任何论文指标。",
                "flag_counts": {
                    f: sum(
                        1 for r in rows for c in r["conflicts_ai_draft"]
                        if f in c.get("_flags", [])
                    )
                    for f in (
                        "outbound_rule_in_inbound_direction",
                        "normalized_cross_jurisdiction_to_cumulative",
                        "cross_jurisdiction_exclusion_relation",
                        "normalized_weak_norm_to_not_conflict",
                        "normalized_basis_to_none",
                        "normalized_basis_to_lex_specialis",
                        "normalized_element_relationship",
                        "weak_norm_suppresses_strong_norm",
                        "verdict_type_inconsistent",
                        "art5_exemption_misapplied",
                        "reason_verdict_contradiction",
                        "threshold_statement_contradiction",
                        "gba_quantitative_exemption_disputed",
                        "unknown_conflict_type",
                        "unknown_verdict",
                    )
                },
                "assessment_flag_count": sum(1 for r in rows if r.get("assessment_flags")),
                "pending_stance_pairs": sum(
                    1 for r in rows for c in r["conflicts_ai_draft"]
                    if c.get("verdict") == "pending_stance"
                ),
                "dropped_identification_pairs": sum(
                    len(r.get("dropped_pairs", [])) for r in rows
                ),
            },
            "items": rows,
        },
    )
    flagged = sum(
        1
        for r in rows
        if r["conflicts_ai_draft"] and "error" not in r["conflicts_ai_draft"][0]
    )
    print(f"conflict sample done: {len(rows)} queries, {flagged} with AI-flagged conflicts -> results/")


def cmd_review_table(src: str, out_name: str = "conflict_rule_pairs_review.csv") -> None:
    """把查询级初判聚合为规则对级审核表（唯一规则对 + 频次 + 校验标记）。"""
    import csv

    path = src if os.path.isabs(src) else os.path.join(BASE, src)
    payload = json.load(io.open(path, encoding="utf-8"))
    rules = {r.rule_id: r for r in corpus.load_rules()}
    occ: dict[tuple, list[dict]] = {}
    for item in payload["items"]:
        op = item["C_op"]
        direction = f"{op['loc_src']}->{op['loc_dst']}"
        for c in item["conflicts_ai_draft"]:
            if "error" in c:
                continue
            a, b = c.get("left_rule_id"), c.get("right_rule_id")
            if not a or not b:
                continue
            occ.setdefault((a, b), []).append(
                {
                    "q": item["query_id"],
                    "dir": direction,
                    "verdict": c.get("verdict"),
                    "type": c.get("conflict_type"),
                    "basis": c.get("basis"),
                    "reason": c.get("reason", ""),
                    "flags": c.get("_flags", []),
                }
            )
    rows = []
    for i, ((a, b), lst) in enumerate(
        sorted(occ.items(), key=lambda kv: (-len(kv[1]), kv[0])), 1
    ):
        ra, rb = rules.get(a), rules.get(b)
        vd = collections.Counter(x["verdict"] for x in lst)
        td = collections.Counter(x["type"] for x in lst)
        bd = collections.Counter(x["basis"] for x in lst)
        dd = collections.Counter(x["dir"] for x in lst)
        fl = collections.Counter(f for x in lst for f in x["flags"])
        same_law = bool(ra and rb and ra.law == rb.law and ra.law)
        jurs = set()
        for r in (ra, rb):
            if r:
                jurs.update(r.jurisdiction_scope or [r.region])
        machine = []
        if same_law:
            machine.append("同法内关系")
        if len(jurs) > 1:
            machine.append("跨法域")
        if any(x.startswith("CN_GBA") for x in (a, b)):
            machine.append("GBA指引相关")
        if len(vd) > 1:
            machine.append("AI裁决不一致")
        if fl.get("unknown_conflict_type") or fl.get("unknown_verdict"):
            machine.append("枚举越界")
        reason = collections.Counter(x["reason"] for x in lst).most_common(1)[0][0]
        rows.append(
            {
                "pair_id": "P%03d" % i,
                "left_rule_id": a,
                "right_rule_id": b,
                "left_law": ra.law if ra else "?",
                "right_law": rb.law if rb else "?",
                "left_norm": ra.norm_type if ra else "?",
                "right_norm": rb.norm_type if rb else "?",
                "机器标记": "; ".join(machine),
                "校验标记": "; ".join(f"{k}:{v}" for k, v in fl.most_common()),
                "出现查询数": len(lst),
                "方向分布": "; ".join(f"{k}:{v}" for k, v in dd.most_common()),
                "AI裁决分布": "; ".join(
                    f"{VERDICT_ZH.get(k, k)}:{v}" for k, v in vd.most_common()
                ),
                "AI冲突类型分布": "; ".join(
                    f"{CONFLICT_TYPE_ZH.get(k, k)}:{v}" for k, v in td.most_common()
                ),
                "AI依据分布": "; ".join(f"{k}:{v}" for k, v in bd.most_common()),
                "示例query": lst[0]["q"],
                "AI理由样例": reason,
                "专家裁决(无冲突/稳定冲突/场景依赖)": "",
                "专家冲突类型": "",
                "专家压制方向": "",
                "专家法理依据(lex_specialis/lex_superior/contextual/none)": "",
                "专家备注": "",
            }
        )
    out = os.path.join(BASE, "gold", out_name)
    with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"review table: {len(rows)} unique rule pairs -> {out}")


def cmd_conflict_metrics(
    draft: str = "results/draft_conflict_gold_query_sample.json",
    gold: str = "gold/conflict_rule_pairs_gold_v4.json",
) -> None:
    """模型冲突判定与专家 gold 的对齐指标（规则对级）。"""
    dpath = draft if os.path.isabs(draft) else os.path.join(BASE, draft)
    gpath = gold if os.path.isabs(gold) else os.path.join(BASE, gold)
    payload = json.load(io.open(dpath, encoding="utf-8"))
    gdoc = json.load(io.open(gpath, encoding="utf-8"))
    gmap = {
        tuple(sorted((i["left_rule_id"], i["right_rule_id"]))): i
        for i in gdoc["items"]
    }
    suppress = (
        "left_suppresses_right", "right_suppresses_left",
        "exclude_left", "exclude_right",
    )
    model: dict[tuple, list[str]] = collections.defaultdict(list)
    for r in payload["items"]:
        for c in r["conflicts_ai_draft"]:
            if "error" in c:
                continue
            model[tuple(sorted((c["left_rule_id"], c["right_rule_id"])))].append(
                c.get("verdict")
            )
    claims = {k: v for k, v in model.items() if any(x in suppress for x in v)}
    tp = [k for k in claims if k in gmap and gmap[k]["verdict"] != "无冲突"]
    fp = [k for k in claims if k in gmap and gmap[k]["verdict"] == "无冲突"]
    unreviewed = [k for k in claims if k not in gmap]
    gold_conflicts = [k for k, v in gmap.items() if v["verdict"] != "无冲突"]
    missed = [k for k in gold_conflicts if k not in model]

    def norm_resolution(model_verdicts: list[str]) -> str:
        if any(v == "pending_stance" for v in model_verdicts):
            return "待定"
        if any(v in suppress for v in model_verdicts):
            return "压制"
        return "并存"

    match = 0
    compared = 0
    for k, v in model.items():
        if k not in gmap:
            continue
        compared += 1
        gold_res = gmap[k]["resolution"]
        gold_kind = "待定" if "待定" in gold_res else ("压制" if "压制" in gold_res else "并存")
        if norm_resolution(v) == gold_kind:
            match += 1
    doc = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "draft": draft,
            "gold": gold,
            "note": "规则对级指标；只在专家已裁决的规则对上比较。",
        },
        "model_pairs": len(model),
        "model_suppression_claims": len(claims),
        "reviewed_overlap": compared,
        "suppression_true_positives": len(tp),
        "suppression_false_positives": len(fp),
        "suppression_precision": round(len(tp) / (len(tp) + len(fp)), 4) if (tp or fp) else None,
        "unreviewed_claims": len(unreviewed),
        "gold_conflict_pairs": len(gold_conflicts),
        "missed_gold_conflicts": len(missed),
        "resolution_agreement": round(match / compared, 4) if compared else None,
        "details": {
            "true_positives": [list(k) for k in tp],
            "false_positives": [list(k) for k in fp],
            "missed": [list(k) for k in missed],
            "unreviewed": [list(k) for k in unreviewed][:20],
        },
    }
    _dump("conflict_metrics.json", doc)
    print(json.dumps({k: v for k, v in doc.items() if k != "details"}, ensure_ascii=False, indent=2))


def _baseline_prompt(method: str, q: dict, cands: list[corpus.Rule]) -> str:
    """对比方法使用的提示词模板。

    direct_llm : 零样本直接判断（无枚举约束、无方向说明）
    nli_style  : 成对语义蕴含/矛盾判定（不提供法域与位阶信息）
    legal_rag  : 检索规则后生成合规结论，并列出其认为冲突的规则对
    """
    limit = 90 if method == "full_kb" else 180
    rules_txt = "\n".join(
        f"{r.rule_id} | {r.law} | {r.rule_text[:limit]}" for r in cands
    )
    op = json.dumps(q["C_op"], ensure_ascii=False)
    if method == "direct_llm":
        return (
            "你是法律专家。下面是一次医疗数据跨境操作，以及可能适用于它的若干规则。\n"
            f"操作：{op}\n规则：\n{rules_txt}\n"
            "请判断哪些规则对之间相互冲突。只列出你认为存在冲突的规则对，不要罗列不冲突的规则对；"
            "每对给出不超过 30 字的理由与处置方向。输出严格 JSON："
            '{"conflicts":[{"left_rule_id":"...","right_rule_id":"...",'
            '"conflict":true,"explanation":"≤30字","resolution":"≤30字"}]}；'
            '若认为不存在冲突，输出 {"conflicts":[]}。'
        )
    if method == "nli_style":
        return (
            "你是文本推理模型。对下列规范文本做两两判断：两段文本是否相互矛盾"
            "（contradiction）。只输出判定为相互矛盾的规则对，不要输出蕴含或无关的情形。\n"
            f"规则：\n{rules_txt}\n"
            '输出严格 JSON：{"pairs":[{"left_rule_id":"...","right_rule_id":"...",'
            '"label":"contradiction","explanation":"≤30字"}]}；'
            '若没有任何一对构成矛盾，输出 {"pairs":[]}。'
        )
    if method == "legal_rag":
        return (
            "你是合规助手。请基于检索到的规则，为下面这次数据操作给出合规建议，"
            "并列出你认为存在冲突的规则对（只列冲突对，理由不超过 30 字）。\n"
            f"操作：{op}\n检索到的规则：\n{rules_txt}\n"
            '输出严格 JSON：{"answer":"≤80字合规建议","conflicts":[{"left_rule_id":"...",'
            '"right_rule_id":"...","explanation":"≤30字"}]}'
        )
    if method == "cot":
        return (
            "你是法律专家。下面是一次医疗数据跨境操作与相关规则。\n"
            f"操作：{op}\n规则：\n{rules_txt}\n"
            "请按以下步骤逐步推理：①逐条判断规则是否适用于该操作（注意规则规范的是"
            "数据出境还是入境）；②对每一对规则，判断它们施加的义务是否相互排斥"
            "（一方要求、另一方禁止或豁免）；③仅当义务相互排斥时认定为冲突，并说明"
            "优先关系或消解方向。把推理过程写入 reasoning 字段，只输出你判定为冲突的规则对。\n"
            '输出严格 JSON：{"reasoning":"≤200字","conflicts":[{"left_rule_id":"...",'
            '"right_rule_id":"...","conflict":true,"explanation":"≤30字","resolution":"≤30字"}]}；'
            '若不存在冲突，输出 {"reasoning":"...","conflicts":[]}。'
        )
    if method == "fewshot":
        return (
            "你是法律专家。下面给出两个已判定示例，再请判断当前操作涉及的规则对。\n"
            "示例一（真冲突）：规则 A 要求“数据出境须申报安全评估”，规则 B 规定“紧急情况下"
            "免予申报”；同一操作同时触发两条，义务相互排斥，判为冲突。\n"
            "示例二（伪冲突）：规则 C 要求“应当告知信息主体”，规则 D 要求“应当采取安全保障"
            "措施”；同一操作同时触发两条，义务可并行履行，判为不冲突。\n"
            f"当前操作：{op}\n候选规则：\n{rules_txt}\n"
            "只输出你判定为冲突的规则对。输出严格 JSON："
            '{"conflicts":[{"left_rule_id":"...","right_rule_id":"...",'
            '"conflict":true,"explanation":"≤30字"}]}；无冲突则输出 {"conflicts":[]}。'
        )
    if method == "full_kb":
        return (
            "你是法律专家。下面是一次医疗数据跨境操作，以及该系统知识库中的全部规则"
            f"（共 {len(cands)} 条，未经筛选）。\n"
            f"操作：{op}\n规则库：\n{rules_txt}\n"
            "请先自行判断哪些规则适用于该操作，再判断其中哪些规则对相互冲突。"
            "只输出你判定为冲突的规则对。输出严格 JSON："
            '{"conflicts":[{"left_rule_id":"...","right_rule_id":"...",'
            '"conflict":true,"explanation":"≤30字"}]}；若无冲突输出 {"conflicts":[]}。'
        )
    raise ValueError(method)


def _baseline_one(method: str, q: dict, rules: list[corpus.Rule]) -> dict:
    if method == "full_kb":
        # 不做候选筛选：把整个知识库交给模型，检验候选生成环节的价值
        cands = list(rules)
    else:
        cands = _applicable_candidates(q, rules)
    try:
        out = llm.chat_json(_baseline_prompt(method, q, cands), max_tokens=3000, retries=2)
        raw = []
        if method == "nli_style":
            for c in out.get("pairs", []):
                if not (c.get("left_rule_id") and c.get("right_rule_id")):
                    continue
                raw.append({
                    "left_rule_id": c["left_rule_id"], "right_rule_id": c["right_rule_id"],
                    "conflict_type": "parallel_obligations",
                    "verdict": ("left_suppresses_right"
                                if c.get("label") == "contradiction" else "cumulative"),
                    "basis": "contextual", "reason": c.get("explanation", ""),
                })
        else:
            for c in out.get("conflicts", []):
                if not (c.get("left_rule_id") and c.get("right_rule_id")):
                    continue
                claims = c.get("conflict")
                if claims is None:
                    claims = True
                txt = (c.get("explanation", "") + c.get("resolution", ""))
                if any(k in txt for k in ("并存", "不构成冲突", "无冲突", "并行")):
                    claims = False
                raw.append({
                    "left_rule_id": c["left_rule_id"], "right_rule_id": c["right_rule_id"],
                    "conflict_type": "permission_vs_prohibition" if claims else "parallel_obligations",
                    "verdict": "left_suppresses_right" if claims else "cumulative",
                    "basis": "contextual", "reason": txt[:200],
                })
        conflicts = [{
            "left_rule_id": r["left_rule_id"], "right_rule_id": r["right_rule_id"],
            "conflict_type": r["conflict_type"], "verdict": r["verdict"],
            "basis": r["basis"], "reason": r["reason"],
        } for r in raw]
        assessment = out.get("answer", "") if method == "legal_rag" else ""
    except Exception as exc:  # noqa: BLE001
        conflicts = [{"error": f"{type(exc).__name__}: {str(exc)[:200]}"}]
        assessment = ""
    return {
        "query_id": q["query_id"],
        "scenario": q.get("scenario", ""),
        "C_op": q["C_op"],
        "X_target": q["X_target"],
        "candidate_rules": [r.rule_id for r in cands],
        "conflicts_ai_draft": conflicts,
        "assessment": assessment,
        "status": f"baseline_{method}",
        "adversarial_group": q.get("adversarial_group"),
        "expected_conflicts": q.get("expected_conflicts", []),
    }


def _defeasible_one(q: dict, rules: list[corpus.Rule]) -> dict:
    """标准可废止逻辑基线。

    按 Antoniou 等（ACM TOCL 2001）、Governatori 等（JLC 2004）定义的 DL 语义，
    把每个候选规则对编码为待废止理论（可废止规则 + 结论冲突 + 优先关系），
    再对 +∂ 可证明性求解（实现见 lexmedrag.defeasible_logic）。
    跨法域时不建立优先关系——位阶不可跨法域比较，因此两法域义务只能并存。
    """
    from lexmedrag import defeasible_logic as dl

    cands = _applicable_candidates(q, rules)
    conflicts = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            if a.law == b.law:
                continue
            left, right = sorted((a, b), key=lambda r: r.rule_id)
            same_jur = set(left.jurisdiction_scope or [left.region]) == set(
                right.jurisdiction_scope or [right.region]
            )
            level_fn = arbitrate.level if same_jur else (lambda _rid: 0)
            verdict, basis = dl.resolve_pair(left, right, level_fn)
            if verdict != "cumulative":
                conflicts.append({
                    "left_rule_id": left.rule_id, "right_rule_id": right.rule_id,
                    "conflict_type": "mandatory_vs_optional", "verdict": verdict,
                    "basis": basis,
                    "reason": "可废止逻辑推理（规则冲突 + 优先关系求解）",
                })
    return {
        "query_id": q["query_id"], "scenario": q.get("scenario", ""),
        "C_op": q["C_op"], "X_target": q["X_target"],
        "candidate_rules": [r.rule_id for r in cands],
        "conflicts_ai_draft": conflicts, "assessment": "",
        "status": "baseline_defeasible",
        "adversarial_group": q.get("adversarial_group"),
        "expected_conflicts": q.get("expected_conflicts", []),
    }


def cmd_baseline(
    method: str,
    n: int,
    workers: int = 4,
    queries_file: str = "results/adversarial_queries_v1.json",
    out_name: str | None = None,
) -> None:
    rules = corpus.load_rules()
    qpath = queries_file if os.path.isabs(queries_file) else os.path.join(BASE, queries_file)
    qs = json.load(io.open(qpath, encoding="utf-8"))["queries"][:n]
    if method == "defeasible":
        rows = [_defeasible_one(q, rules) for q in qs]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(lambda q: _baseline_one(method, q, rules), qs))
    out_name = out_name or f"baseline_{method}.json"
    _dump(out_name, {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "method": method, "queries": len(rows),
            "note": "对比方法输出，用于与完整方法在同一条目上比较。",
        },
        "items": rows,
    })


def main() -> None:
    p = argparse.ArgumentParser(description="LexMedRAG experiment runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("card", help="real dataset card")
    pr = sub.add_parser("retrieval", help="retrieval pilot on real QA")
    pr.add_argument("--limit", type=int, default=None)
    sub.add_parser("selftest", help="self checks")
    sub.add_parser("llm-check", help="verify DeepSeek API key")
    pg = sub.add_parser("gold-draft", help="LLM-assisted gold draft (needs review)")
    pg.add_argument("--pairs", type=int, default=12)
    pg.add_argument("--queries", type=int, default=15)
    ps = sub.add_parser("conflict-sample", help="query-conditioned conflict draft sample")
    ps.add_argument("--n", type=int, default=30)
    ps.add_argument("--workers", type=int, default=4)
    ps.add_argument("--queries", type=str, default="results/gbamc_queries_v3.json")
    ps.add_argument("--out", type=str, default="draft_conflict_gold_query_sample.json")
    pt = sub.add_parser("review-table", help="aggregate drafts into rule-pair review table")
    pt.add_argument("--src", type=str, default="results/draft_conflict_gold_query_sample.json")
    pt.add_argument("--out", type=str, default="conflict_rule_pairs_review.csv")
    pm = sub.add_parser("conflict-metrics", help="model vs expert gold conflict metrics")
    pm.add_argument("--draft", type=str, default="results/draft_conflict_gold_query_sample.json")
    pm.add_argument("--gold", type=str, default="gold/conflict_rule_pairs_gold_v4.json")
    pb = sub.add_parser("baseline", help="run a comparison baseline")
    pb.add_argument("--method", required=True,
                    choices=["direct_llm", "nli_style", "legal_rag", "defeasible",
                             "cot", "fewshot", "full_kb"])
    pb.add_argument("--n", type=int, default=36)
    pb.add_argument("--workers", type=int, default=4)
    pb.add_argument("--queries", type=str, default="results/adversarial_queries_v1.json")
    pb.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    if args.cmd == "card":
        cmd_card()
    elif args.cmd == "retrieval":
        cmd_retrieval(args.limit)
    elif args.cmd == "selftest":
        cmd_selftest()
    elif args.cmd == "llm-check":
        cmd_llm_check()
    elif args.cmd == "gold-draft":
        cmd_gold_draft(args.pairs, args.queries)
    elif args.cmd == "conflict-sample":
        cmd_conflict_sample(args.n, args.workers, args.queries, args.out)
    elif args.cmd == "review-table":
        cmd_review_table(args.src, args.out)
    elif args.cmd == "conflict-metrics":
        cmd_conflict_metrics(args.draft, args.gold)
    elif args.cmd == "baseline":
        cmd_baseline(args.method, args.n, args.workers, args.queries, args.out)


if __name__ == "__main__":
    main()
