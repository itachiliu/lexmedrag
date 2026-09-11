"""Pure-python BM25-style retrieval over character-bigram tokens."""

from __future__ import annotations

import math

from .text_utils import tokens


class BigramBM25:
    """BM25 over the token stream from lexmedrag.text_utils.tokens."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tfs: list[dict[str, int]] = []
        self.dl: list[int] = []
        self.idf: dict[str, float] = {}
        n = len(docs)
        postings: dict[str, int] = {}
        for doc in docs:
            tks = tokens(doc)
            tf: dict[str, int] = {}
            for t in tks:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tfs.append(tf)
            self.dl.append(len(tks))
            for t in set(tks):
                postings[t] = postings.get(t, 0) + 1
        self.avgdl = (sum(self.dl) / n) if n else 0.0
        for t, df in postings.items():
            self.idf[t] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_doc(self, q_tfs: dict[str, int], doc_idx: int) -> float:
        tf_map = self.doc_tfs[doc_idx]
        dl = self.dl[doc_idx]
        s = 0.0
        for t, qf in q_tfs.items():
            f = tf_map.get(t, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            s += self.idf.get(t, 0.0) * qf * f / denom
        return s

    def rank(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_tfs: dict[str, int] = {}
        for t in tokens(query):
            q_tfs[t] = q_tfs.get(t, 0) + 1
        scored = sorted(
            ((i, self.score_doc(q_tfs, i)) for i in range(len(self.doc_tfs))),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]
