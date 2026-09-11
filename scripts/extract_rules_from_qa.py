"""从问答语料的引用片段中抽取候选规则（法规名 + 条号 + 规范内容）。

用途：把问答中高频引用、但知识库尚未建模的条款补成草稿规则，
以便扩大检索评测的"引用可解析子集"。所有条目均标注待专家核对原文。

用法：
    python scripts/extract_rules_from_qa.py
输出：
    data/legal_knowledge_base_qa_derived_draft.json
"""

from __future__ import annotations

import collections
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAW_PATTERN = re.compile(r"《([^》]{3,45})》")
# 形如「《X法》第十二条：内容」或「《X法》5.4 规定，内容」
REF_PATTERN = re.compile(
    r"《(?P<law>[^》]{3,45})》\s*"
    r"(?:第(?P<cn_article>[〇零一二三四五六七八九十百]+)条"
    r"|(?P<num_article>\d+(?:\.\d+)+)"
    r"|第(?P<ar_article>\d+)条)"
    r"\s*[：:，,]?\s*(?P<body>[^《]{8,400})"
)
OBLIGATION = ("应当", "不得", "禁止", "须", "必须", "需要", "免予", "可以", "应")


def main() -> None:
    qa = json.load(io.open(os.path.join(ROOT, "data", "merge.json"), encoding="utf-8"))
    seen: dict[tuple, dict] = {}
    for item in qa:
        for ctx in item.get("positive_contexts", []):
            text = (ctx.get("content") or "").replace("\n", " ")
            for match in REF_PATTERN.finditer(text):
                law = match.group("law").strip()
                article = (match.group("cn_article") or match.group("ar_article")
                           or match.group("num_article"))
                body = match.group("body").strip().rstrip("；;。")
                if not any(word in body for word in OBLIGATION):
                    continue
                key = (law, article)
                if key in seen:
                    continue
                seen[key] = {
                    "law": law,
                    "article": article,
                    "text": body[:220],
                    "question": (item.get("question") or "")[:60],
                }

    by_law = collections.Counter(v["law"] for v in seen.values())
    print("可抽取的（法规, 条号）候选:", len(seen))
    print("涉及法规数:", len(by_law))
    for law, n in by_law.most_common(12):
        print(f"  {n:>3}  {law}")

    entries = []
    for index, ((law, article), payload) in enumerate(sorted(seen.items()), 1):
        safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", law)[:24]
        entries.append({
            "rule_id": f"QA_{safe}_{article}_{index:03d}",
            "region": "CN",
            "rule_text": payload["text"],
            "rule_category": "条文摘录",
            "jurisdiction_scope": ["CN"],
            "sensitivity_score": 0.6,
            "metadata": {
                "source": f"《{law}》第{article}条（自问答语料引用片段整理，需核对原文）",
                "issue_date": "",
                "revision_history": [],
                "status": "draft_for_review",
                "verification_required": True,
                "derived_from": "data/merge.json 正向上下文",
                "sample_question": payload["question"],
            },
        })
    out = os.path.join(ROOT, "data", "legal_knowledge_base_qa_derived_draft.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print("written:", out, "| entries:", len(entries))


if __name__ == "__main__":
    main()
