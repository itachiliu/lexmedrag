"""数值一致性回归：从结果文件重建论文各表的数字，与正文逐格比对。

用法：python _tools/number_regression.py
返回码非零表示存在不一致。
"""

from __future__ import annotations

import io
import json
import os
import re
import sys

WS = os.environ.get(
    "LEXMEDRAG_WS", r"D:\学术\ACM_Conference_Proceedings_Primary_Article_Template"
)
TEX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WS, "LexMedRAG4.0-CN.tex")
EXP = os.environ.get("LEXMEDRAG_EXP", os.path.join(WS, "lexmedrag-experiments"))


def load(rel: str):
    with io.open(os.path.join(EXP, rel), encoding="utf-8") as handle:
        return json.load(handle)


def tables() -> dict[str, list[tuple[str, str]]]:
    """返回 {表标签: [(整行文本, 该行去标签后的内容), ...]}"""
    text = io.open(TEX, encoding="utf-8").read()
    result: dict[str, list[tuple[str, str]]] = {}
    for match in re.finditer(
        r"\\begin\{(?:table|table\*)\}(.*?)\\end\{(?:table|table\*)\}", text, re.S
    ):
        block = match.group(1)
        label = re.search(r"\\label\{([^}]+)\}", block)
        if not label:
            continue
        body = re.search(r"\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}", block, re.S)
        if not body:
            continue
        rows = []
        for line in body.group(1).split("\\\\"):
            clean = line.strip()
            if not clean:
                continue
            rows.append((clean, clean))
        result[label.group(1)] = rows
    return result


def numbers(cell_text: str) -> list[float]:
    """把 LaTeX 单元格里的数字按出现顺序取出，科学计数法合并为一个数。"""
    text = cell_text
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\texttt\{[^}]*\}", " ", text)   # 规则标识中的条号不是数值
    text = text.replace("\\%", "").replace("{,}", "").replace("\\,", "")
    text = re.sub(r"\\times\s*10\^\{(-?\d+)\}", r"e\1", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "").replace("\\", "")
    out = []
    for token in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", text):
        try:
            out.append(float(token))
        except ValueError:
            pass
    return out


def close(paper: float, data: float) -> bool:
    """按论文给出的精度判断是否一致。"""
    if abs(paper) >= 1 and float(paper).is_integer() and abs(data) >= 1:
        return abs(paper - data) < 1e-9 or abs(paper - data) <= 0.5
    if abs(data) < 1e-6:  # 科学计数法
        if abs(paper) < 1e-6:
            return abs(paper - data) <= 0.05 * max(abs(data), 1e-30)
        return False
    decimals = len(str(paper).split(".")[1]) if "." in str(paper) else 0
    return abs(paper - data) <= 0.5 * 10 ** (-decimals) + 1e-9


def compare(table_label: str, row_label: str, expected: list[float],
            rows: dict, problems: list[str]) -> None:
    pattern = re.compile(r"(?:^|\n)\s*" + re.escape(row_label))
    for raw, _clean in rows.get(table_label, []):
        if not pattern.search(raw):
            continue
        body = raw.split("&", 1)[1] if "&" in raw else raw
        paper = numbers(body)
        if len(paper) != len(expected):
            problems.append(
                f"{table_label} / {row_label}: 数字个数不符，论文 {paper}，数据 {expected}"
            )
            return
        for got, want in zip(paper, expected):
            if not close(got, want):
                problems.append(
                    f"{table_label} / {row_label}: 论文字面 {got}，数据 {want}"
                )
        return
    problems.append(f"{table_label}: 未找到行「{row_label}」")


def main() -> int:
    rows = tables()
    problems: list[str] = []

    analysis = load("results/experiment_analysis.json")
    natural = load("results/layered_conflict_metrics.json")["natural"]
    comparison = load("results/method_comparison.json")["methods"]
    ablation = load("results/ablation_v2.json")["configs"]
    significance = load("results/significance_tests.json")["methods"]
    retrieval = load("results/retrieval_comparison.json")["methods"]
    direction = load("results/direction_constraint_check.json")

    order = ["LexMedRAG", "DL", "Direct-LLM", "Full-KB", "Legal-RAG", "CoT", "Few-shot", "NLI"]

    # 表 1 自然语料
    compare("tab:natural", "规模", [natural["queries"], natural["gold_pairs"]], rows, problems)
    compare("tab:natural", "专家裁决", [203, 94.9, 9, 2], rows, problems)
    compare("tab:natural", "模型输出",
            [natural["model_pairs"], natural["model_suppression_claims"],
             natural["suppression_true_positives"], natural["suppression_false_positives"], 7],
            rows, problems)
    compare("tab:natural", "指标",
            [round(natural["suppression_precision"] * 1000) / 1000,
             natural["missed_gold_conflicts"],
             round(natural["resolution_agreement"] * 1000) / 1000], rows, problems)

    # 表 2 对抗集
    groups = ["C1", "C2", "C3", "C4", "C5", "C6"]
    sizes = [81, 120, 120, 120, 80, 120]
    compare("tab:adversarial", "预期冲突对", sizes + [sum(sizes)], rows, problems)
    compare("tab:adversarial", "冲突召回率", [100] * 7, rows, problems)
    rates = [analysis["adversarial"]["LexMedRAG"]["by_group"][g] * 100 for g in groups]
    compare("tab:adversarial", "冲突识别率",
            rates + [analysis["adversarial"]["LexMedRAG"]["recognition"] * 100],
            rows, problems)

    # 表 3 基线对比
    compare("tab:baselines", "列出预期冲突对",
            [v for name in order for v in (comparison[name]["detected"],
                                           comparison[name]["conflict_recall"] * 100)],
            rows, problems)
    compare("tab:baselines", "判定为冲突",
            [v for name in order for v in (comparison[name]["recognized"],
                                           comparison[name]["conflict_recognition_rate"] * 100)],
            rows, problems)
    compare("tab:baselines", "列出但判定为并存",
            [comparison[name]["detected"] - comparison[name]["recognized"] for name in order],
            rows, problems)
    compare("tab:baselines", "未被列出",
            [641 - comparison[name]["detected"] for name in order], rows, problems)
    compare("tab:baselines", "去重后声称冲突的唯一规则对",
            [comparison[name]["distinct_pairs_claimed_conflict"] for name in order],
            rows, problems)

    # 表 4 配对检验
    for name, display in [("DL", "可废止逻辑"), ("Direct-LLM", "零样本直接判断"),
                          ("Full-KB", "全库直判"), ("Legal-RAG", "检索增强生成"),
                          ("CoT", "链式推理"), ("Few-shot", "示例引导"),
                          ("NLI", "成对自然语言推理")]:
        item = significance[name]
        expected = [item["only_ours"], item["only_baseline"], item["diff_pp"], item["mcnemar_p"]]
        compare("tab:significance", display, expected, rows, problems)

    # 表 5 归因
    cross = {"C1": 81, "C2": 120, "C3": 0, "C4": 40, "C5": 80, "C6": 80}
    for group, total in zip(groups, sizes):
        compare("tab:attribution", f"{group} ", [total, cross[group], total - cross[group]],
                rows, problems)
    compare("tab:attribution", "合计",
            [641, sum(cross.values()), 641 - sum(cross.values())], rows, problems)

    # 表 6 消融
    compare("tab:mechanism", "冲突识别率",
            [ablation[c]["recognition"] * 100 for c in ("A", "B", "C")],
            rows, problems)

    # 表 7 检索
    bm25 = retrieval["BM25（字符 bigram）"]
    tfidf = retrieval["TF-IDF 余弦"]
    compare("tab:retrieval", "BM25",
            [bm25["recall@1"], bm25["recall@3"], bm25["recall@5"], bm25["recall@10"], bm25["ndcg@5"]],
            rows, problems)
    compare("tab:retrieval", "TF-IDF",
            [tfidf["recall@1"], tfidf["recall@3"], tfidf["recall@5"], tfidf["recall@10"], tfidf["ndcg@5"]],
            rows, problems)

    # 表 8 算例规则层级
    compare("tab:case_rules", "$r_1$", [1], rows, problems)
    compare("tab:case_rules", "$r_2$", [3], rows, problems)
    compare("tab:case_rules", "$r_3$", [3], rows, problems)
    compare("tab:case_rules", "$r_4$", [3], rows, problems)
    compare("tab:case_rules", "$r_5$", [1], rows, problems)

    print("=" * 72)
    print("表格逐格比对")
    print("=" * 72)
    if problems:
        for item in problems:
            print("  [不一致]", item)
    else:
        print("  全部一致；共核对 8 张表")

    # 正文关键数值与已退役表述
    text = io.open(TEX, encoding="utf-8").read()

    # 白名单：所有由结果文件推导出的百分比，外加人工登记的比例
    known_pct = {0.0, 100.0}
    for name in order:
        known_pct |= {
            analysis["adversarial"][name]["recognition"] * 100,
            comparison[name]["conflict_recall"] * 100,
            comparison[name]["conflict_recognition_rate"] * 100,
        }
        known_pct |= {v * 100 for v in analysis["adversarial"][name]["ci95"]}
        known_pct |= {v * 100 for v in analysis["adversarial"][name]["by_group"].values()}
    known_pct |= {item["diff_pp"] for item in significance.values()}
    repeat = load("results/repeat_runs.json")
    for record in repeat.values():
        known_pct |= set(record.get("runs", []))
        if "mean" in record:
            known_pct.add(record["mean"])
    component = analysis["component_ablation"]
    known_pct.add(
        (1 - component["with_normalization"]["claims"] / component["without_normalization"]["claims"]) * 100
    )
    for cfg in ablation.values():
        known_pct.add(cfg["recognition"] * 100)
        known_pct |= {v * 100 for v in cfg["by_group"].values()}
    for table in (retrieval["BM25（字符 bigram）"], retrieval["TF-IDF 余弦"]):
        known_pct |= {v * 100 for k, v in table.items()}
    known_pct |= {
        natural["suppression_precision"] * 100,
        natural["resolution_agreement"] * 100,
        natural["suppression_false_positives"] / 841 * 100,
        203 / 214 * 100,      # 无冲突比例
        11 / 214 * 100,       # 非无冲突比例（含场景依赖与待定）
        113 / 214 * 100,      # 跨法域比例
        401 / 641 * 100,      # 对抗集跨法域比例
        240 / 641 * 100,      # 对抗集同法域比例
        167 / 641 * 100,      # 消融配置 A 命中比例
    }

    narratives = re.sub(
        r"\\begin\{(?:table|table\*)\}.*?\\end\{(?:table|table\*)\}", " ", text, flags=re.S
    )
    unknown = []
    allow_pct = {95.0, 5.0}  # 置信水平、法定罚款比例等非指标数值
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*\\%", narratives):
        value = float(match.group(1))
        decimals = len(match.group(1).split(".")[1]) if "." in match.group(1) else 0
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(value - k) <= tolerance for k in known_pct | allow_pct):
            context = narratives[max(0, match.start() - 46): match.start() + 6]
            unknown.append((value, context.replace("\n", " ")))
    print()
    print("=" * 72)
    print("叙述性百分比白名单校验")
    print("=" * 72)
    if unknown:
        for value, context in unknown:
            print(f"  [白名单外] {value}%  上下文：…{context}")
            problems.append(f"百分比 {value}% 无结果文件支撑")
    else:
        print(f"  正文（不含表格）出现的 {len(re.findall(chr(92)+'%', narratives))} 处百分比均在白名单内")

    must_appear = {
        "对抗集识别率": "71.9",
        "自然语料无冲突率": "94.9",
        "跨法域比例": "52.8",
        "与基线差值": "15.6",
        "消融下限": "26.1",
        "方向约束" if "8 条出境专属规则" in text else "方向约束(表述缺失)": "8 条出境专属规则",
        "入境查询数": "480 条入境查询",
        "配对检验 p": "10^{-14}",
    }
    must_absent = ["621", "97/97", "18 个预期冲突对", "448 条查询"]

    print()
    print("=" * 72)
    print("正文关键数值")
    print("=" * 72)
    for name, token in must_appear.items():
        status = "OK  " if token in text else "缺失"
        print(f"  {status} {name}: {token}")
        if token not in text:
            problems.append(f"正文缺少 {name}")
    for token in must_absent:
        status = "OK  " if token not in text else "仍存在"
        print(f"  {status} 退役表述: {token}")
        if token in text:
            problems.append(f"正文仍含退役表述 {token}")

    print()
    print("=" * 72)
    print(f"结论：{len(problems)} 处不一致")
    print("=" * 72)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
