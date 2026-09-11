"""Draft attribute mapping: Synthea content -> medical-data categories.

This is a FIRST-DRAFT mapping for review. Categories follow the medical-data
ontology used in ComplianceAgent (个人属性/健康状况/医疗应用/医疗支付...),
and data levels follow the Guangdong classification rules in
data/legal_knowledge_base.json (CN_GDHMD_*).

Final mapping must be reviewed before generating the official GBAMC set.
"""

from __future__ import annotations

# keyword (case-insensitive) -> canonical attribute label
CONDITION_KEYWORDS = {
    "cancer|malignant|neoplasm|c[0-9]{2}": ("肿瘤诊断记录", "健康状况数据", 4),
    "diabetes|e1[0-9]": ("内分泌疾病记录", "健康状况数据", 3),
    "pregnan|o[0-9]{2}": ("孕产记录", "医疗应用数据", 3),
    "mental|depress|anxiety|f[0-9]{2}": ("精神心理记录", "健康状况数据", 3),
}

OBSERVATION_KEYWORDS = {
    "heart rate|blood pressure|oxygen|vital": ("生命体征监测数据", "健康状况数据", 3),
    "glucose|hba1c|laboratory": ("检验检查数据", "医疗应用数据", 3),
    "genetic|sequence|variant": ("基因测序数据", "个人属性数据", 4),
}

MEDICATION_FALLBACK = ("用药记录", "医疗应用数据", 3)
DIAGNOSIS_FALLBACK = ("诊断记录（ICD-10）", "医疗应用数据", 3)


def match_category(text: str) -> tuple[str, str, int] | None:
    """Return (label, category, level) for a Synthea text/code line."""
    import re

    for pattern, value in CONDITION_KEYWORDS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return value
    for pattern, value in OBSERVATION_KEYWORDS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return None
