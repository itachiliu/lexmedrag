"""Load the real rule/QA knowledge bases and build retrieval corpus."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .text_utils import normalize, to_chinese_numeral

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


@dataclass
class Rule:
    rule_id: str
    region: str
    rule_text: str
    rule_category: str
    jurisdiction_scope: list
    sensitivity_score: float
    source: str
    issue_date: str
    law: str = ""
    article: str = ""
    article_no: str = ""
    norm_type: str = ""

    @property
    def text(self) -> str:
        return f"{self.rule_text}。出处：{self.source}。类别：{self.rule_category}。"


def _norm_type(rule_id: str) -> str:
    """规范性质（决定其能否压制上位规范，避免团体标准被当作法律适用）。"""
    if rule_id.startswith("CN_PIPL"):
        return "法律"
    if rule_id.startswith("CN_DSL"):
        return "法律"
    if rule_id.startswith("CN_SPI"):
        return "技术指南（推荐性）"
    if rule_id.startswith("CN_HGR_RULES"):
        return "部门规章（实施细则）"
    if rule_id.startswith("CN_HGR"):
        return "行政法规"
    if rule_id.startswith("CN_OUTBOUND"):
        return "部门规章"
    if rule_id.startswith(("CN_PV", "CN_ADR")):
        return "部门规章/规范性文件"
    if rule_id.startswith("CN_GBA"):
        return "规范性文件（实施指引）"
    if rule_id.startswith("CN_GDHMD"):
        return "团体标准（推荐性技术文件）"
    if rule_id.startswith("CN_CSSPG"):
        return "技术指南（推荐性）"
    if rule_id.startswith("HK_PDPO"):
        return "香港主体条例"
    if rule_id.startswith("MO_PDPA"):
        return "澳门法律"
    if rule_id.startswith("EU_GVP"):
        return "欧盟法规（域外）"
    if rule_id.startswith("ICH_E2B"):
        return "国际技术标准（域外）"
    if rule_id.startswith("FDA_21CFR"):
        return "美国联邦法规（域外）"
    return ""


def _law_name(rule_id: str) -> str:
    if rule_id.startswith("CN_PIPL"):
        return "个人信息保护法"
    if rule_id.startswith("CN_DSL"):
        return "数据安全法"
    if rule_id.startswith("CN_SPI"):
        return "网络安全标准实践指南 — 敏感个人信息识别指南"
    if rule_id.startswith("CN_HGR_RULES"):
        return "人类遗传资源管理条例实施细则"
    if rule_id.startswith("CN_HGR"):
        return "人类遗传资源管理条例"
    if rule_id.startswith("CN_GDHMD"):
        return "广东省健康医疗数据安全分类分级管理技术规范"
    if rule_id.startswith("CN_CSSPG"):
        return "网络安全标准实践指南 — 敏感个人信息识别指南"
    if rule_id.startswith("CN_OUTBOUND"):
        return "促进和规范数据跨境流动规定"
    if rule_id.startswith("CN_GBA_HK"):
        return "粤港澳大湾区（内地、香港）个人信息跨境流动标准合同实施指引"
    if rule_id.startswith("CN_GBA_MO"):
        return "粤港澳大湾区（内地、澳门）个人信息跨境流动标准合同实施指引"
    if rule_id.startswith("HK_PDPO"):
        return "香港《个人资料（私隐）条例》"
    if rule_id.startswith("MO_PDPA"):
        return "澳门《个人资料保护法》"
    if rule_id.startswith("CN_PV") or rule_id.startswith("CN_ADR"):
        return "药物警戒记录保存规定"
    if rule_id.startswith("EU_GVP"):
        return "EU GVP Module VI"
    if rule_id.startswith("ICH_E2B"):
        return "ICH E2B(R3)"
    if rule_id.startswith("FDA_21CFR"):
        return "FDA 21 CFR 314.80"
    return ""


def _parse_article(rule_id: str, source: str) -> tuple:
    m = re.search(r"Art(\d+)", rule_id)
    if m:
        art = int(m.group(1))
        return str(art), to_chinese_numeral(art)
    m = re.search(r"第\s*([0-9]+(?:\.[0-9]+)?)\s*条", source)
    if m:
        return m.group(1), ""
    return "", ""


def _load_rule_file(path: str) -> list[Rule]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rules = []
    for item in raw:
        art_no, art_cn = _parse_article(
            item["rule_id"], item.get("metadata", {}).get("source", "")
        )
        rules.append(
            Rule(
                rule_id=item["rule_id"],
                region=item.get("region", ""),
                rule_text=item.get("rule_text", ""),
                rule_category=item.get("rule_category", ""),
                jurisdiction_scope=item.get("jurisdiction_scope", []),
                sensitivity_score=float(item.get("sensitivity_score", 0.0)),
                source=item.get("metadata", {}).get("source", ""),
                issue_date=item.get("metadata", {}).get("issue_date", ""),
                law=item.get("law") or _law_name(item["rule_id"]),
                article=art_cn,
                article_no=art_no,
                norm_type=_norm_type(item["rule_id"]),
            )
        )
    return rules


def load_rules(path: str | None = None) -> list[Rule]:
    """Load CN rules plus, if present, the GBA/HK/MO draft expansion."""
    path = path or os.path.join(DATA_DIR, "legal_knowledge_base.json")
    rules = _load_rule_file(path)
    for name in ("legal_knowledge_base_gba_draft.json",
                 "legal_knowledge_base_hkmo_draft.json",
                 "legal_knowledge_base_hgr_draft.json",
                 "legal_knowledge_base_adversarial_draft.json"):
        extra = os.path.join(DATA_DIR, name)
        if os.path.exists(extra):
            rules.extend(_load_rule_file(extra))
    return rules


def load_qa(path: str | None = None) -> list[dict]:
    path = path or os.path.join(DATA_DIR, "merge.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _article_variants(rule: Rule) -> list[str]:
    variants = []
    law = normalize(rule.law)
    for article in (rule.article_no, rule.article):
        if not article:
            continue
        variants.append(normalize(law + "第" + article + "条"))
        # 语料中的条号常用中文数字（如“第二十八条”），一并生成
        try:
            variants.append(normalize(law + "第" + to_chinese_numeral(int(article)) + "条"))
        except (TypeError, ValueError):
            pass
    return variants


# 法规名称在问答语料中的常见异写，用于 gold 匹配前的名称归一化
LAW_ALIASES = {
    "中华人民共和国个人信息保护法": "个人信息保护法",
    "中华人民共和国数据安全法": "数据安全法",
    "中华人民共和国人类遗传资源管理条例": "人类遗传资源管理条例",
    "中华民族人类遗传资源管理条例": "人类遗传资源管理条例",
    "人类遗传资源管理条例实施细则": "人类遗传资源管理条例实施细则",
    "澳门个人资料保护法": "澳门《个人资料保护法》",
    "个人资料保护法": "澳门《个人资料保护法》",
    "个人资料（私隐）条例": "香港《个人资料（私隐）条例》",
    "个人资料(私隐)条例": "香港《个人资料（私隐）条例》",
}


def _law_variants(law: str) -> list[str]:
    """生成某个法规名在语料中可能出现的形式（含书名号与简称）。"""
    variants = {law}
    for alias, canonical in LAW_ALIASES.items():
        if canonical == law:
            variants.add(alias)
    for base in list(variants):
        variants.add(base.strip("《》"))
    return [normalize(v) for v in variants if v]


def gold_rules_for_qa(item: dict, rules: list[Rule]) -> list[str]:
    """Match a QA item's positive contexts to rules by cited law+article.

    Only rules whose law and article can both be resolved and appear in the
    cited evidence are considered gold. This is an approximate, conservative
    mapping used for the retrieval pilot; it is not a substitute for expert
    annotation.
    """
    ctx = normalize(" ".join(c.get("content", "") for c in item.get("positive_contexts", [])))
    ctx += normalize(" ".join(c.get("source", "") for c in item.get("positive_contexts", [])))
    matched = []
    for rule in rules:
        if not rule.law:
            continue
        if not any(variant in ctx for variant in _law_variants(rule.law)):
            continue
        variants = _article_variants(rule)
        if not variants:
            continue
        if any(v in ctx for v in variants):
            matched.append(rule.rule_id)
    return matched
