"""按 nature-figure 规范生成论文实验图（Python 后端：matplotlib/seaborn）。

Figure contract
---------------
Core conclusion : 在多法域医疗数据合规中，本文方法通过候选适用性约束与结构化判定，
                  把真冲突识别率提升到显著高于符号基线与通用 LLM 范式的水平，
                  同时把冲突声称数控制在可审计的范围内。
Evidence chain  : (a) 总体识别率与不确定性（hero）→ (b) 案例难度分布（难度来源）
                  → (c) 分组判定构成（失败模式与可审计性）→ (d) 查询级判定构成
                  → (e) 机制消融（归因）
Archetype       : quantitative grid（3×2，hero 在左上，(c) 通栏，(e) 为消融阶梯）
Export          : SVG（主）+ PDF/TIFF；字体 ≥5pt；导出后跑对齐与碰撞审计。

用法：
    python scripts/make_figures.py
输出：
    figures/fig_conflict_eval.{svg,pdf,tiff}
"""

from __future__ import annotations

import io
import json
import os
import sys

import matplotlib as mpl

mpl.use("Agg")  # 无 GUI 环境：使用非交互后端
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 面板对齐检查器来自作者的本地绘图工具箱，发布版中允许缺失。
_ALIGNMENT_SCRIPTS = os.environ.get(
    "FIGURE_ALIGNMENT_SCRIPTS", r"C:\Users\10028\.codex\skills\nature-figure\scripts"
)
sys.path.insert(0, _ALIGNMENT_SCRIPTS)
try:  # pragma: no cover - 取决于运行环境
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:  # 干净环境下退化为不做对齐门禁
    def require_matplotlib_panel_alignment(*args, **kwargs):
        return None

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
}

METHOD_ORDER = ["LexMedRAG", "DL", "Direct-LLM", "Full-KB", "Legal-RAG", "CoT", "Few-shot", "NLI"]
DISPLAY = {
    "LexMedRAG": "LexMedRAG",
    "DL": "Defeasible logic",
    "Direct-LLM": "Direct-LLM",
    "Full-KB": "Full-KB",
    "Legal-RAG": "Legal-RAG",
    "CoT": "CoT",
    "Few-shot": "Few-shot",
    "NLI": "NLI",
}

GROUP_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6"]
STATE_ORDER = ["hit", "quiet", "miss"]
STATE_COLORS = {"hit": "#0F4D92", "quiet": "#B8B8B8", "miss": "#E9A6A1"}
STATE_LABELS = {"hit": "识别为冲突", "quiet": "列出但判为并存", "miss": "未列出"}

QUERY_STATE_ORDER = ["all", "partial", "none"]
QUERY_STATE_COLORS = {"all": "#0F4D92", "partial": "#C9C9C9", "none": "#B64342"}
QUERY_STATE_LABELS = {"all": "全部识别", "partial": "部分识别", "none": "全部漏检"}


def apply_publication_style(font_size=7.5, axes_linewidth=0.8):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "Arial", "DejaVu Sans", "Liberation Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.grid"] = False
    plt.rcParams["savefig.bbox"] = "tight"


def add_panel_label(ax, label, x=0.0, y=1.0, x_offset_pt=-14, y_offset_pt=3,
                    fontsize=9, color="#272727", fontweight="bold", va="bottom"):
    from matplotlib.transforms import ScaledTranslation
    offset = ScaledTranslation(x_offset_pt / 72, y_offset_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(x, y, label, transform=ax.transAxes + offset, fontsize=fontsize,
            fontweight=fontweight, color=color, ha="left", va=va)


def load(rel):
    with io.open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


CASE_FILES = {
    "LexMedRAG": "results/adversarial_draft_v2_lexmedrag_run1.json",
    "DL": "results/baseline_defeasible_v2.json",
    "Direct-LLM": "results/baseline_direct_llm_v2.json",
    "Full-KB": "results/baseline_full_kb_v2.json",
    "Legal-RAG": "results/baseline_legal_rag_v2.json",
    "CoT": "results/baseline_cot_v2.json",
    "Few-shot": "results/baseline_fewshot_v2.json",
    "NLI": "results/baseline_nli_style_v2.json",
}
QUIET = {"cumulative", "not_conflict"}


def load_case_states() -> tuple[list[tuple[str, str, tuple, str]], dict]:
    """返回 (案例序列, 每方法状态映射)。案例序列以 LexMedRAG 的输出顺序为基准。"""
    keyed: dict[str, dict] = {}
    order: list[tuple[str, str, tuple, str]] = []
    for name, rel in CASE_FILES.items():
        doc = load(rel)
        table: dict = {}
        for item in doc["items"]:
            group = item.get("adversarial_group", "?")
            model = {}
            for c in item["conflicts_ai_draft"]:
                if "error" in c:
                    continue
                pair_key = tuple(sorted((c.get("left_rule_id"), c.get("right_rule_id"))))
                model[pair_key] = c.get("verdict")
            for pair in item.get("expected_conflicts", []):
                pair_key = tuple(sorted(pair))
                if pair_key in model:
                    state = "quiet" if model[pair_key] in QUIET else "hit"
                else:
                    state = "miss"
                table[(group, item["query_id"], pair_key)] = state
                if name == "LexMedRAG":
                    order.append((group, item["query_id"], pair_key, state))
        keyed[name] = table
    return order, keyed


def order_cases_by_group(order: list) -> list[int]:
    groups = ["C1", "C2", "C3", "C4", "C5", "C6"]
    indexed = list(enumerate(order))
    indexed.sort(key=lambda pair: (groups.index(pair[1][0]), pair[0]))
    return [i for i, _ in indexed]


def panel_outcome_composition(ax, order, states, method_names):
    """(c) 分组判定构成：各方法在六组冲突上的判定结果构成。

    每个方法一行，每列为一组冲突，行内按判定结果归一化为 100%。"""
    columns = GROUP_ORDER + ["合计"]
    totals = {column: 0 for column in columns}
    counts = {
        name: {column: dict.fromkeys(STATE_ORDER, 0) for column in columns}
        for name in method_names
    }
    for group, qid, pair_key, _ in order:
        totals[group] += 1
        totals["合计"] += 1
        for name in method_names:
            state = states[name].get((group, qid, pair_key), "miss")
            counts[name][group][state] += 1
            counts[name]["合计"][state] += 1

    rows = method_names[::-1]
    for row, name in enumerate(rows):
        for column_index, column in enumerate(columns):
            total = totals[column] or 1
            left = float(column_index)
            for state in STATE_ORDER:
                share = counts[name][column][state] / total
                if share <= 0:
                    continue
                ax.barh(row, share, left=left, height=0.68,
                        color=STATE_COLORS[state], linewidth=0, zorder=2)
                if share >= 0.17:
                    ax.text(left + share / 2, row, f"{share * 100:.0f}",
                            ha="center", va="center", fontsize=6.4, zorder=3,
                            color="white" if state == "hit" else "#3D3D3D")
                left += share
    for boundary in range(1, len(columns)):
        ax.axvline(boundary, color="white", lw=0.8, zorder=3)
    ax.axvline(len(columns) - 1, color="#4D4D4D", lw=0.9, zorder=4)
    ax.axhline(len(rows) - 1.5, color="#4D4D4D", lw=0.8, zorder=4)
    ax.set_xlim(0, len(columns))
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DISPLAY[name] for name in rows], fontsize=7.0)
    ax.set_xticks([i + 0.5 for i in range(len(columns))])
    ax.set_xticklabels([f"{c}\n{totals[c]} 对" for c in columns], fontsize=7.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        mpl.patches.Patch(facecolor=STATE_COLORS[state], edgecolor="none",
                          label=STATE_LABELS[state])
        for state in STATE_ORDER
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.005),
              ncol=3, fontsize=6.6, frameon=False, handlelength=1.0,
              handleheight=0.8, columnspacing=1.2, borderaxespad=0.0)
    add_panel_label(ax, "c")


def panel_difficulty(ax, order, states, method_names):
    """(b) 案例难度谱：每个预期对被多少方法识别。"""
    groups = ["C1", "C2", "C3", "C4", "C5", "C6"]
    palette = ["#0F4D92", "#3775BA", "#8BCF8B", "#CFCECE", "#E9A6A1", "#B64342"]
    counts = {g: np.zeros(9) for g in groups}
    for index, (group, qid, pair_key, _) in enumerate(order):
        hit_n = sum(
            1 for name in method_names
            if states[name].get((group, qid, pair_key)) == "hit"
        )
        counts[group][hit_n] += 1
    x = np.arange(9)
    bottom = np.zeros(9)
    for gi, group in enumerate(groups):
        ax.bar(x, counts[group], bottom=bottom, width=0.72,
               color=palette[gi % len(palette)], label=group, linewidth=0)
        bottom += counts[group]
    ax.set_xlabel("被识别的方法数（0 = 全部漏检，8 = 全部命中）", fontsize=7)
    ax.set_ylabel("预期冲突对数", fontsize=7)
    ax.tick_params(labelsize=7.0)
    ax.legend(fontsize=6.6, ncol=3, loc="upper right", columnspacing=0.8,
              handlelength=0.9, borderaxespad=0.2)
    add_panel_label(ax, "b")


def panel_per_query(ax, order, states, method_names):
    """(d) 查询级判定构成：每条查询的预期对是被全部识别、部分识别还是全部漏检。

    每条查询只含 1--3 个预期冲突对，单查询识别率只有少数离散取值，
    因此采用构成条而非箱线图表达其分布。"""
    rows = method_names[::-1]
    tally = {}
    for name in method_names:
        table = states[name]
        per_query: dict = {}
        for group, qid, pair_key, _ in order:
            entry = per_query.setdefault(qid, [0, 0])
            entry[1] += 1
            if table.get((group, qid, pair_key)) == "hit":
                entry[0] += 1
        counts = dict.fromkeys(QUERY_STATE_ORDER, 0)
        for hit, total in per_query.values():
            if hit == 0:
                counts["none"] += 1
            elif hit == total:
                counts["all"] += 1
            else:
                counts["partial"] += 1
        tally[name] = counts

    for row, name in enumerate(rows):
        total = sum(tally[name].values()) or 1
        left = 0.0
        for state in QUERY_STATE_ORDER:
            share = tally[name][state] / total
            if share <= 0:
                continue
            ax.barh(row, share, left=left, height=0.66,
                    color=QUERY_STATE_COLORS[state], linewidth=0, zorder=2)
            if share >= 0.075:
                ax.text(left + share / 2, row, str(tally[name][state]),
                        ha="center", va="center", fontsize=6.4, zorder=3,
                        color="#3D3D3D" if state == "partial" else "white")
            left += share
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DISPLAY[name] for name in rows], fontsize=7.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=7.0)
    ax.set_xlabel("占查询的比例（%）", fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        mpl.patches.Patch(facecolor=QUERY_STATE_COLORS[state], edgecolor="none",
                          label=QUERY_STATE_LABELS[state])
        for state in QUERY_STATE_ORDER
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.005),
              ncol=3, fontsize=6.6, frameon=False, handlelength=1.0,
              handleheight=0.8, columnspacing=1.2, borderaxespad=0.0)
    add_panel_label(ax, "d")


def panel_a(ax, analysis):
    """hero：识别率 + 95% CI（forest-plot 风格）。"""
    labels, est, lo, hi, colors = [], [], [], [], []
    for name in METHOD_ORDER[::-1]:
        m = analysis["adversarial"][name]
        labels.append(DISPLAY[name])
        est.append(m["recognition"] * 100)
        lo.append(m["ci95"][0] * 100)
        hi.append(m["ci95"][1] * 100)
        colors.append(PALETTE["blue_main"] if name == "LexMedRAG" else PALETTE["neutral_mid"])
    y = np.arange(len(labels))
    for yi, e, l, h, c in zip(y, est, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=1.4, zorder=2)
        ax.plot(e, yi, marker="o", ms=4.2 if c == PALETTE["blue_main"] else 3.2,
                color=c, zorder=3)
        ax.text(h + 2.0, yi, f"{e:.1f}", va="center", fontsize=7.0, color="#272727")
    best_base = analysis["adversarial"]["DL"]["recognition"] * 100
    ax.axvline(best_base, color=PALETTE["red_strong"], ls="--", lw=0.9, alpha=0.85, zorder=1)
    ax.text(best_base + 1.5, len(labels) - 0.35, "strongest baseline",
            color=PALETTE["red_strong"], fontsize=7.0, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.set_xlim(-2, 104)
    ax.set_xlabel("冲突识别率（%，误差线为 95% bootstrap 区间）", fontsize=7)
    ax.tick_params(axis="x", labelsize=7.0)
    add_panel_label(ax, "a")


def panel_b(ax, analysis):
    """分组识别率热力图。"""
    groups = ["C1", "C2", "C3", "C4", "C5", "C6"]
    names = METHOD_ORDER[::-1]
    matrix = np.array([[analysis["adversarial"][n]["by_group"].get(g, 0) * 100
                        for g in groups] for n in names])
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7.0,
                color="white" if val > 55 else "#272727")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=7.0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([DISPLAY[n] for n in names], fontsize=7.0)
    ax.set_frame_on(False)
    ax.tick_params(length=0)
    ax.set_xlabel("冲突组", fontsize=7)
    add_panel_label(ax, "b")


def panel_c(ax, comparison_nat, comparison):
    """松紧-识别权衡。"""
    claimed = {k: v["distinct_pairs_claimed_conflict"] for k, v in comparison.items()}
    analysis = comparison
    for index, name in enumerate(METHOD_ORDER, start=1):
        x = max(1, claimed[name])
        y = analysis[name]["conflict_recognition_rate"] * 100
        is_hero = name == "LexMedRAG"
        ax.scatter(x, y, s=64 if is_hero else 18,
                   marker="*" if is_hero else "o",
                   color=PALETTE["blue_main"] if is_hero else PALETTE["neutral_mid"],
                   zorder=4 if is_hero else 3,
                   edgecolor="white", lw=0.4)
        label_offsets = {
            "Direct-LLM": (0.78, 4.0, "right"),
            "DL": (1.22, 6.0, "left"),
            "Full-KB": (0.80, -8.0, "right"),
        }
        dx, dy, ha = label_offsets.get(name, (1.18, 4.0, "left"))
        ax.annotate(str(index), (x, y), xytext=(x * dx, y + dy), fontsize=7.0,
                    ha=ha, va="center",
                    color=PALETTE["blue_main"] if is_hero else "#4D4D4D")
    ax.set_xscale("log")
    ax.set_xlim(4, 900)
    ax.set_xticks([5, 10, 25, 50, 100, 300, 600])
    ax.set_xticklabels(["5", "10", "25", "50", "100", "300", "600"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylim(-6, 100)
    ax.set_xlabel("声称存在冲突的唯一规则对（对数刻度）", fontsize=7)
    ax.set_ylabel("冲突识别率（%）", fontsize=7)
    ax.tick_params(labelsize=7.0)
    add_panel_label(ax, "c")


def panel_d(ax, analysis, ablation=(26.1, 71.9, 71.9)):
    """机制消融阶梯图。"""
    x = np.arange(len(ablation))
    ax.plot(x, ablation, color=PALETTE["blue_main"], lw=2.0, marker="o", ms=4.5, zorder=3)
    for xi, val in zip(x, ablation):
        last = xi == x[-1]
        ax.text(xi - 0.07 if last else xi + 0.07, val - 7.0, f"{val:.1f}",
                ha="right" if last else "left", fontsize=7.0, color="#272727")
    for xi, delta in zip(x[1:], np.diff(ablation)):
        ax.text(xi - 0.5, (ablation[xi] + ablation[xi - 1]) / 2 - 13,
                f"+{delta:.1f}", ha="center", fontsize=7.0, color=PALETTE["red_strong"])
    ax.set_xticks(x)
    ax.set_xticklabels(["A\n一律判并存", "B\n区分排斥型", "C\n完整方法"], fontsize=7.0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("冲突识别率（%）", fontsize=7)
    ax.tick_params(labelsize=7.0)
    add_panel_label(ax, "e")


def main() -> None:
    apply_publication_style()
    analysis = load("results/experiment_analysis.json")
    comparison = load("results/method_comparison.json")["methods"]
    order, states = load_case_states()

    fig = plt.figure(figsize=(7.2, 7.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 1.05],
                          width_ratios=[1.14, 0.86],
                          hspace=0.46, wspace=0.30,
                          left=0.115, right=0.985, top=0.975, bottom=0.07)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    panel_a(ax_a, analysis)
    panel_difficulty(ax_b, order, states, METHOD_ORDER)
    panel_outcome_composition(ax_c, order, states, METHOD_ORDER)
    panel_per_query(ax_d, order, states, METHOD_ORDER)
    panel_d(ax_e, analysis)

    out_dir = os.path.join(ROOT, "figures")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "fig_conflict_eval")

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=f"{base}.alignment.json",
        overlay_svg=f"{base}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(f"{base}.svg")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.tiff", dpi=600)
    print("written:", base + ".{svg,pdf,tiff}")


if __name__ == "__main__":
    main()
