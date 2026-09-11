"""Five cross-border scenarios (first-draft parameters, for review)."""

from __future__ import annotations

SCENARIOS = [
    {
        "id": "hk_to_cn_ai_training",
        "name": "香港医院 → 内地 AI 公司（诊断模型训练）",
        "src": "HK",
        "dst": "CN",
        "act": "model_training",
        "vol_range": (1000, 100000),
        "attr_hints": ["诊断记录（ICD-10）", "影像报告", "基因测序数据"],
    },
    {
        "id": "mo_to_sz_multicenter",
        "name": "澳门医院 → 深圳机构（多中心试验）",
        "src": "MO",
        "dst": "CN",
        "act": "clinical_trial",
        "vol_range": (500, 50000),
        "attr_hints": ["检验检查数据", "住院病历", "基因测序数据"],
    },
    {
        "id": "cn_to_hk_pharmacovigilance",
        "name": "内地研究机构 → 香港药企（药物警戒/统计）",
        "src": "CN",
        "dst": "HK",
        "act": "pharmacovigilance",
        "vol_range": (1000, 50000),
        "attr_hints": ["用药记录", "不良事件报告", "检验检查数据"],
    },
    {
        "id": "gba_telemedicine",
        "name": "大湾区医院集团内部跨境（远程诊疗/转诊）",
        "src": "MO",
        "dst": "CN",
        "act": "telemedicine",
        "vol_range": (100, 20000),
        "attr_hints": ["门诊病历", "生命体征监测数据", "影像报告"],
    },
    {
        "id": "gba_storage_backup",
        "name": "港澳机构 → 内地数据中心（存储/灾备，含可穿戴）",
        "src": "HK",
        "dst": "CN",
        "act": "storage_backup",
        "vol_range": (10000, 500000),
        "attr_hints": ["可穿戴健康数据", "生命体征监测数据", "住院病历"],
    },
    {
        "id": "cn_to_mo_pharmacovigilance",
        "name": "内地研究机构 → 澳门药企（药物警戒/统计）",
        "src": "CN",
        "dst": "MO",
        "act": "pharmacovigilance",
        "vol_range": (1000, 50000),
        "attr_hints": ["用药记录", "不良事件报告", "检验检查数据"],
    },
    {
        "id": "cn_to_mo_telemedicine",
        "name": "内地医院 → 澳门医疗机构（远程诊疗/转诊）",
        "src": "CN",
        "dst": "MO",
        "act": "telemedicine",
        "vol_range": (100, 20000),
        "attr_hints": ["门诊病历", "生命体征监测数据", "影像报告"],
    },
]


def get_scenario(sid: str) -> dict:
    for s in SCENARIOS:
        if s["id"] == sid:
            return s
    raise KeyError(sid)
