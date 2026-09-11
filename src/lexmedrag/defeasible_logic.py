"""标准可废止逻辑（Defeasible Logic, DL）推理器。

实现依据：
  [1] Antoniou, Billington, Governatori, Maher.
      Representation results for defeasible logic.
      ACM Transactions on Computational Logic, 2(2):255-287, 2001.
  [2] Governatori, Maher, Antoniou, Billington.
      Argumentation semantics for defeasible logic.
      Journal of Logic and Computation, 14(5):675-702, 2004.
  [3] Lam, Governatori. The making of SPINdle. Proc. RuleML, 2009.

理论构件：
  - facts           : 事实集合
  - strict rules    : A -> p     （不可废止，肯定前件）
  - defeasible rules: A => p     （可被反制规则或优先关系废止）
  - defeaters       : A ~> p     （只阻止对 p 的可废止证明，不推出 p）
  - superiority     : r1 > r2    （规则间的优先关系）

本模块实现 +∂（可废止可证明）的证明过程：结论 p 可废止地可推出，当存在一条
可适用的可废止规则 r: A => p，且对每一条与 p 结论相冲突的规则 s，
要么 s 不可适用，要么 r 优先于 s（r > s）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DLRule:
    """一条可废止逻辑规则。kind 取 strict / defeasible / defeater。"""

    rid: str
    antecedent: frozenset[str]
    consequent: str
    kind: str = "defeasible"


@dataclass
class DefeasibleTheory:
    facts: set[str] = field(default_factory=set)
    rules: list[DLRule] = field(default_factory=list)
    superiority: set[tuple[str, str]] = field(default_factory=set)
    # 结论之间的冲突关系（对称），例如 effect:r1 与 effect:r2 互斥
    conflicts: set[frozenset[str]] = field(default_factory=set)

    def add_fact(self, literal: str) -> None:
        self.facts.add(literal)

    def add_rule(self, rid: str, antecedent, consequent: str,
                 kind: str = "defeasible") -> None:
        self.rules.append(DLRule(rid, frozenset(antecedent), consequent, kind))

    def add_superiority(self, r1: str, r2: str) -> None:
        self.superiority.add((r1, r2))

    def add_conflict(self, l1: str, l2: str) -> None:
        self.conflicts.add(frozenset({l1, l2}))

    # ---------- 证明过程 ----------

    def _applicable(self, rule: DLRule, derived: set[str]) -> bool:
        return rule.antecedent <= (self.facts | derived)

    def _conflicts_with(self, literal: str) -> set[str]:
        out = set()
        for pair in self.conflicts:
            if literal in pair:
                out |= (pair - {literal})
        return out

    def _defeated_by(self, rule: DLRule, target: str) -> bool:
        """存在适用规则 s（结论与 target 冲突）且 r 不优于 s。"""
        for s in self.rules:
            if s.consequent == target and s.rid == rule.rid:
                continue
            if s.consequent not in self._conflicts_with(rule.consequent):
                continue
            if s.rid == rule.rid:
                continue
            # s 必须适用（其前件由事实与已推结论满足）
            if not (s.antecedent <= self.facts):
                continue
            if s.kind == "defeater":
                # defeater 阻止结论，除非被优先关系排除
                if (rule.rid, s.rid) not in self.superiority:
                    return True
                continue
            if (rule.rid, s.rid) not in self.superiority:
                return True
        return False

    def provable(self, literal: str) -> bool:
        """+∂：字面量是否可废止地可推出。"""
        for rule in self.rules:
            if rule.consequent != literal or rule.kind == "defeater":
                continue
            if not self._applicable(rule, set()):
                continue
            if not self._defeated_by(rule, literal):
                return True
        return False

    def definitely_provable(self, literal: str) -> bool:
        """+Δ：通过严格规则可推出。"""
        if literal in self.facts:
            return True
        for rule in self.rules:
            if rule.kind == "strict" and rule.consequent == literal:
                if rule.antecedent <= self.facts:
                    return True
        return False


# ---------- 面向本任务的编码 ----------

# 义务极性：决定两条规则的结论是否互斥
POLARITY_HINTS = (
    ("prohibit", ("不得", "禁止", "严禁", "除外", "尚未生效", "不得提供", "不得向")),
    ("require", ("应当", "须", "必须", "需要", "应")),
    ("exempt", ("免予", "免除", "豁免", "可以", "无需", "不适用")),
)


def polarity(rule_text: str) -> str:
    """从规则文本提取义务极性（启发式，用于构造 DL 理论的冲突关系）。"""
    for name, keywords in POLARITY_HINTS:
        if any(k in rule_text for k in keywords):
            return name
    return "require"


def build_pair_theory(rule_a, rule_b, level_fn) -> DefeasibleTheory:
    """把一对规则编码为待废止理论。

    - 每条规则编码为一条可废止规则：{} => effect(rule)
    - 若两条规则的义务极性互斥（prohibit vs require / exempt），
      登记为结论冲突，从而各自成为对方的反制来源
    - 优先关系按位阶与生效日期确定
    """
    th = DefeasibleTheory()
    eff_a, eff_b = f"effect:{rule_a.rule_id}", f"effect:{rule_b.rule_id}"
    th.add_rule(rule_a.rule_id, [], eff_a, "defeasible")
    th.add_rule(rule_b.rule_id, [], eff_b, "defeasible")

    pol_a, pol_b = polarity(rule_a.rule_text), polarity(rule_b.rule_text)
    if {pol_a, pol_b} == {"prohibit", "require"} or {pol_a, pol_b} == {"prohibit", "exempt"}:
        th.add_conflict(eff_a, eff_b)

    la, lb = level_fn(rule_a.rule_id), level_fn(rule_b.rule_id)
    if la != lb:
        winner, loser = ((rule_a.rule_id, rule_b.rule_id) if la > lb
                         else (rule_b.rule_id, rule_a.rule_id))
        th.add_superiority(winner, loser)
    elif rule_a.issue_date != rule_b.issue_date and rule_a.issue_date and rule_b.issue_date:
        winner, loser = ((rule_a.rule_id, rule_b.rule_id)
                         if rule_a.issue_date > rule_b.issue_date
                         else (rule_b.rule_id, rule_a.rule_id))
        th.add_superiority(winner, loser)
    return th


def resolve_pair(rule_a, rule_b, level_fn) -> tuple[str, str]:
    """返回 (verdict, basis)。verdict 取压制方向或 cumulative。"""
    th = build_pair_theory(rule_a, rule_b, level_fn)
    eff_a, eff_b = f"effect:{rule_a.rule_id}", f"effect:{rule_b.rule_id}"
    pa, pb = th.provable(eff_a), th.provable(eff_b)
    if pa and not pb:
        basis = "lex_superior" if level_fn(rule_a.rule_id) != level_fn(rule_b.rule_id) \
            else "lex_posterior"
        return "left_suppresses_right", basis
    if pb and not pa:
        basis = "lex_superior" if level_fn(rule_a.rule_id) != level_fn(rule_b.rule_id) \
            else "lex_posterior"
        return "right_suppresses_left", basis
    return "cumulative", "none"
