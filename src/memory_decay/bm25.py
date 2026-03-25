"""Shared BM25 tokenizer and scorer.

Used by both MemoryGraph (in-memory) and MemoryStore (SQLite) for
hybrid semantic+lexical retrieval.
"""
from __future__ import annotations

import math
import re
from collections import Counter


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 scoring.

    Handles English, Korean (가-힣), and numbers.
    Returns lowercased tokens with punctuation stripped.
    """
    return re.findall(r"[0-9A-Za-z가-힣]+", text.lower())


def bm25_score_candidates(
    query_text: str,
    candidate_docs: dict[str, str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, float]:
    """Score candidate documents against a query using BM25.

    Args:
        query_text: The search query.
        candidate_docs: Mapping of document_id -> document_text.
        k1: Term frequency saturation parameter.
        b: Document length normalization parameter.

    Returns:
        Mapping of document_id -> BM25 score. All candidates are included
        (score 0.0 if no query terms match).
    """
    query_terms = list(dict.fromkeys(bm25_tokenize(query_text)))
    if not query_terms or not candidate_docs:
        return {}

    # Tokenize all documents
    doc_tokens: dict[str, list[str]] = {}
    for doc_id, text in candidate_docs.items():
        doc_tokens[doc_id] = bm25_tokenize(text)

    # Compute IDF from candidate set
    n_docs = len(candidate_docs)
    doc_freq: Counter[str] = Counter()
    total_tokens = 0
    for tokens in doc_tokens.values():
        doc_freq.update(set(tokens))
        total_tokens += len(tokens)

    avgdl = max(total_tokens / max(n_docs, 1), 1.0)
    idf: dict[str, float] = {}
    for term, df in doc_freq.items():
        idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    # Score each document
    scores: dict[str, float] = {}
    for doc_id, tokens in doc_tokens.items():
        tf = Counter(tokens)
        dl = max(len(tokens), 1)
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            term_idf = idf.get(term, 0.0)
            denom = freq + k1 * (1.0 - b + b * dl / avgdl)
            score += term_idf * (freq * (k1 + 1.0)) / max(denom, 1e-9)
        scores[doc_id] = score

    return scores
