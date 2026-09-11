"""QA 派生规则与现有知识库去重，并报告可用于检索扩展的净新增条目。"""

from __future__ import annotations

import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from lexmedrag import corpus  # noqa: E402


def chinese_to_int(text: str) -> int | None:
    digits = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    if text.isdigit():
        return int(text)
    if "十" in text:
        left, _, right = text.partition("十")
        tens = digits.get(left, 1) if left else 1
        units = digits.get(right, 0) if right else 0
        return tens * 10 + units
    if len(text) == 1 and text in digits:
        return digits[text]
    return None


def main() -> None:
    rules = corpus.load_rules()
    existing = set()
    for r in rules:
        if r.law and r.article_no:
            existing.add((re.sub(r"[^\u4e00-\u9fff]", "", r.law), str(r.article_no)))

    path = os.path.join(ROOT, "data", "legal_knowledge_base_qa_derived_draft.json")
    derived = json.load(io.open(path, encoding="utf-8"))
    kept, dropped = [], 0
    for entry in derived:
        law = re.sub(r"[^\u4e00-\u9fff]", "",
                     entry["metadata"]["source"].split("》")[0].lstrip("《"))
        article = entry["rule_id"].rsplit("_", 2)[-2]
        number = chinese_to_int(article) if article else None
        if number is not None and (law, str(number)) in existing:
            dropped += 1
            continue
        kept.append(entry)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print("现有规则库:", len(rules), "| QA 派生候选:", len(derived))
    print("与现有库重复（已剔除）:", dropped, "| 保留:", len(kept))


if __name__ == "__main__":
    main()
