"""检索组件对比实验：BM25 与 TF-IDF 余弦在不同 k 上的表现。

用法：
    python scripts/retrieval_comparison.py

输出：
    results/retrieval_comparison.json

说明：gold 采用问答正向上下文中显式引用的“法律名 + 条号”自动匹配（启发式），
只保留可唯一解析的条目，因此数值应视为下界而非最终值。
"""

from __future__ import annotations

import collections
import datetime as _dt
import io
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from lexmedrag import corpus, retrieval  # noqa: E402
from lexmedrag.text_utils import tokens  # noqa: E402


def _tfidf_index(docs: list[str]):
    doc_tokens = [tokens(d) for d in docs]
    df = collections.Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    n = len(docs)

    def vec(toks):
        tf = collections.Counter(toks)
        out = {}
        for term, count in tf.items():
            idf = math.log((n + 1) / (df.get(term, 0) + 1)) + 1.0
            out[term] = (1 + math.log(count)) * idf
        norm = math.sqrt(sum(v * v for v in out.values())) or 1.0
        return {k: v / norm for k, v in out.items()}

    return [vec(t) for t in doc_tokens], vec


def _cosine_rank(query_vec, doc_vecs, top_k):
    scored = []
    for index, dv in enumerate(doc_vecs):
        if len(dv) > len(query_vec):
            small, large = query_vec, dv
        else:
            small, large = dv, query_vec
        score = sum(v * large.get(k, 0.0) for k, v in small.items())
        if score > 0:
            scored.append((score, index))
    scored.sort(reverse=True)
    return scored[:top_k]


def evaluate(rank_fn, golds, ks=(1, 3, 5, 10)) -> dict:
    hits = {k: 0 for k in ks}
    ndcg = []
    for gold, ranked in zip(golds, rank_fn):
        gold_set = set(gold)
        for k in ks:
            if gold_set & set(ranked[:k]):
                hits[k] += 1
        ndcg.append(retrieval.__dict__.get("ndcg_at_k", lambda *a: 0.0))
    total = len(golds) or 1
    out = {f"recall@{k}": round(hits[k] / total, 4) for k in ks}
    return out


def main() -> None:
    rules = corpus.load_rules()
    # 检索语料扩展：附加问答语料派生的条文摘录（仅用于检索评测，不进入冲突判定知识库）
    derived_path = os.path.join(ROOT, "data", "legal_knowledge_base_qa_derived_draft.json")
    if os.path.exists(derived_path):
        rules = rules + corpus._load_rule_file(derived_path)
    qa = corpus.load_qa()
    from lexmedrag.metrics import ndcg_at_k

    pairs = []
    for item in qa:
        gold = corpus.gold_rules_for_qa(item, rules)
        if gold:
            pairs.append((item.get("question", ""), gold))
    questions = [q for q, _ in pairs]
    golds = [g for _, g in pairs]
    texts = [r.text for r in rules]
    ids = [r.rule_id for r in rules]

    bm25 = retrieval.BigramBM25(texts)
    doc_vecs, vec_fn = _tfidf_index(texts)

    methods = {}
    for k in (1, 3, 5, 10):
        hits = 0
        for question, gold in zip(questions, golds):
            ranked = [ids[i] for i, _ in bm25.rank(question, top_k=k)]
            if set(gold) & set(ranked):
                hits += 1
        methods.setdefault("BM25（字符 bigram）", {})[f"recall@{k}"] = round(hits / len(pairs), 4)

    for k in (1, 3, 5, 10):
        hits = 0
        for question, gold in zip(questions, golds):
            ranked = [ids[i] for _, i in _cosine_rank(vec_fn(tokens(question)), doc_vecs, k)]
            if set(gold) & set(ranked):
                hits += 1
        methods.setdefault("TF-IDF 余弦", {})[f"recall@{k}"] = round(hits / len(pairs), 4)

    ndcg_bm25 = []
    ndcg_tfidf = []
    for question, gold in zip(questions, golds):
        ranked = [ids[i] for i, _ in bm25.rank(question, top_k=5)]
        ndcg_bm25.append(ndcg_at_k(ranked, set(gold), k=5))
        ranked2 = [ids[i] for _, i in _cosine_rank(vec_fn(tokens(question)), doc_vecs, 5)]
        ndcg_tfidf.append(ndcg_at_k(ranked2, set(gold), k=5))
    methods["BM25（字符 bigram）"]["ndcg@5"] = round(sum(ndcg_bm25) / len(ndcg_bm25), 4)
    methods["TF-IDF 余弦"]["ndcg@5"] = round(sum(ndcg_tfidf) / len(ndcg_tfidf), 4)

    doc = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "queries_with_resolvable_gold": len(pairs),
            "rules": len(rules),
            "derived_rules": len(rules) - len(corpus.load_rules()),
            "gold_source": "问答正向上下文中的“法律名+条号”自动匹配（启发式，下界）",
        },
        "methods": methods,
    }
    out = os.path.join(ROOT, "results", "retrieval_comparison.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(json.dumps(doc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
