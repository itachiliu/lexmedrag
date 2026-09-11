# LexMedRAG: code, benchmark and annotations

**English** | [简体中文](README.zh-CN.md)

This repository accompanies the manuscript *LexMedRAG: a knowledge-graph-grounded framework for
cross-border medical data compliance*, submitted to **Knowledge-Based Systems**. It contains the
legal knowledge base, the GBAMC benchmark, the adversarial conflict set, the expert annotations
and every script needed to reproduce the numbers and figures reported in the paper.

The paper is written in Chinese, so parts of the data documentation are in Chinese. All code
comments and this README are in English.

## What the paper claims

The paper makes one empirical claim, on two datasets: whether two rules conflict under
cross-border medical data compliance is decidable from a typed norm graph in which each rule
carries an effect level, a norm type, an issue date and an applicability direction. On 840 natural
queries, 94.9% of the 214 expert-adjudicated rule pairs turn out to be pseudo-conflicts, and 52.8%
of those pairs cross a jurisdiction boundary — crossing borders is neither necessary nor
sufficient for a conflict. On an adversarial set of 641 expected conflict pairs, the method
reaches 100% recall and 71.9% recognition, 15.6 points above the strongest symbolic baseline
(paired McNemar exact test, $p<10^{-14}$).

The risk-scoring branch and the compliance-path selector are described as design only. Their
evaluation needs expert annotation that is not finished, so no performance claim is made about
them; the annotation sheets for that evaluation ship in this repository.

## Layout

| Path | Content |
|---|---|
| `data/` | Legal knowledge base: 110 curated rules over 8 instruments, plus 138 QA-derived excerpts used only as retrieval corpus |
| `gbamc/` | Query generator: maps Synthea patient records and scenario definitions to GBAMC queries |
| `src/lexmedrag/` | Core library: BM25 retrieval, deterministic doctrine arbitration, defeasible-logic baseline, metrics |
| `scripts/` | Every experiment and figure in the paper |
| `results/` | Model outputs and derived metric files |
| `gold/` | Expert adjudication of 214 rule pairs (round 1) and the round-2 annotation package |
| `docs/` | Construction plan, adversarial set specification, annotation protocol, journal style analysis |
| `figures/` | Figure 6 source files |

## Data

**Legal knowledge base (110 rules).** `data/legal_knowledge_base.json` holds 72 rules,
`legal_knowledge_base_gba_draft.json` 13, `legal_knowledge_base_hkmo_draft.json` 8,
`legal_knowledge_base_adversarial_draft.json` 15 and `legal_knowledge_base_hgr_draft.json` 2, for
110 rules in total. Each rule carries a jurisdiction, an effect level, a norm type, an issue date
and an applicability direction. Two caveats are inherited from the paper. First, the
pharmacovigilance retention clause and the foreign-regulator clauses (EU GVP Module VI,
ICH E2B(R3), US 21 CFR 314.80) are consolidated drafts flagged
`verification_required: true`; they are excluded from the reported metrics until checked against
the original text. Second, `legal_knowledge_base_qa_derived_draft.json` adds 138 excerpts that
widen the retrieval corpus to 248 items; they never take part in conflict adjudication.

**GBAMC (840 queries).** `results/draft_conflict_gold_query_all_v5.json` is the model output over
the full benchmark, including the generated assessment for every query. Query contexts and data
attributes come from `gbamc/scenarios.py` and `gbamc/generate_queries.py`.

**Adversarial conflict set (240 queries / 641 expected pairs).**
`results/adversarial_queries_v2.json` defines the six conflict groups C1--C6;
`results/adversarial_conflict_draft_v3.json` is the corresponding draft. The set is
**human-designed and not a natural distribution**; its metrics must never be pooled with the
natural-corpus metrics.

**Expert annotations.** `gold/conflict_rule_pairs_gold_v4.json` contains the 214 rule pairs
adjudicated by domain experts: conflict verdict, resolution, doctrinal basis, occurrence count and
transfer direction. Round-1 review sheets are kept as `gold/conflict_rule_pairs_review_v*.csv`.

**Round-2 annotation package.** `gold/expert_round2/` holds a blind re-annotation sheet for
inter-annotator agreement, a risk-scoring sheet, a decision-utility sheet and the annotation
guide. Regenerate with `python gold/make_expert_package.py`.

## Reproducing the paper

| Paper item | Command |
|---|---|
| Table 1, natural corpus metrics | `python scripts/conflict_metrics_layered.py` |
| Tables 2--3, adversarial recognition and baselines | `python scripts/experiment_analysis.py` then `python scripts/compare_methods.py` |
| Tables 4--5, McNemar tests and attribute attribution | `python scripts/significance_tests.py` |
| Ablation (Table 6) | `python scripts/ablation_v2.py` |
| Retrieval evaluation (Table 7) | `python scripts/retrieval_comparison.py` |
| Figure 6 | `python scripts/make_figures.py` |
| Expert annotation sheets | `python gold/make_expert_package.py` |

Every script writes into `results/` and prints the table it produces, so each reported number can
be traced back to the file it came from. Every metric in the paper is reproducible from the
committed model outputs without any API call.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.12 is what we used. The core library depends only on the standard library; `numpy` and
`matplotlib` are needed by the figure and significance scripts.

The three language-model baselines and the conflict pre-annotation call the DeepSeek chat API.
Put your own key in `.env` as `DEEPSEEK_API_KEY=...` or export it as an environment variable. No
key ships with this repository and the client never logs it. Rerunning those baselines spends API
credit; it is not needed to reproduce any reported number.

## Regenerating the synthetic population

The raw Synthea CSV export (78 MB) is not committed because it is reproducible:

```bash
java -jar synthea-with-dependencies.jar -p 2000 -s 20260909
python -m gbamc.generate_queries --synthea <output/csv> --per-scenario 120
```

The snapshot used in the paper contains 1,171 patients, 53,346 encounters, 299,697 observations,
8,376 conditions, 42,989 medication records and 34,981 procedures; the same counts are recorded in
`results/dataset_card_v2.json`.

## What is deliberately absent

* `.env` and any API key;
* the raw Synthea CSV export and TIFF exports of the figure, both regenerable from the commands above;
* most intermediate drafts of the query files. The files listed under **Data** are the ones the
  paper's numbers come from; a few earlier versions remain because the scripts still reference them.

## License

Code is released under the MIT License. The knowledge base, benchmark, annotations and results are
released under CC BY 4.0. Legal texts quoted in the knowledge base remain subject to their own
terms; the excerpts are short and used for research purposes.

## Citation

Pending. A BibTeX entry will be added once the manuscript is accepted.
