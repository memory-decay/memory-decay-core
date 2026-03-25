"""Tests for BM25 tokenizer and scorer."""
import pytest
from memory_decay.bm25 import bm25_tokenize, bm25_score_candidates


class TestBM25Tokenize:
    def test_english_words(self):
        assert bm25_tokenize("Hello World") == ["hello", "world"]

    def test_korean_words(self):
        tokens = bm25_tokenize("서울은 한국의 수도입니다")
        assert "서울은" in tokens
        assert "한국의" in tokens

    def test_mixed_language(self):
        tokens = bm25_tokenize("Python은 좋은 언어")
        assert "python은" in tokens  # mixed runs stay joined
        assert "좋은" in tokens

    def test_numbers_preserved(self):
        tokens = bm25_tokenize("GPT4 has 175B params")
        assert "gpt4" in tokens
        assert "175b" in tokens

    def test_empty_string(self):
        assert bm25_tokenize("") == []

    def test_punctuation_stripped(self):
        tokens = bm25_tokenize("hello, world! (test)")
        assert tokens == ["hello", "world", "test"]


class TestBM25ScoreCandidates:
    def test_exact_match_scores_highest(self):
        docs = {
            "m1": "the cat sat on the mat",
            "m2": "the dog played in the park",
            "m3": "a fish swam in the ocean",
        }
        scores = bm25_score_candidates("cat mat", docs)
        assert scores["m1"] > scores["m2"]
        assert scores["m1"] > scores["m3"]

    def test_no_match_scores_zero(self):
        docs = {"m1": "hello world"}
        scores = bm25_score_candidates("xyz qqq", docs)
        assert scores["m1"] == pytest.approx(0.0)

    def test_empty_query(self):
        docs = {"m1": "hello world"}
        scores = bm25_score_candidates("", docs)
        assert scores == {}

    def test_empty_docs(self):
        scores = bm25_score_candidates("hello", {})
        assert scores == {}

    def test_idf_weighting(self):
        # "rare" appears in 1 doc, "the" appears in all 3
        docs = {
            "m1": "the rare word",
            "m2": "the common phrase",
            "m3": "the other text",
        }
        scores = bm25_score_candidates("rare", docs)
        # m1 should score because it has "rare", others should not
        assert scores["m1"] > 0
        assert scores["m2"] == pytest.approx(0.0)

    def test_term_frequency_matters(self):
        docs = {
            "m1": "cat cat cat",
            "m2": "cat dog bird",
        }
        scores = bm25_score_candidates("cat", docs)
        # m1 has higher TF for "cat"
        assert scores["m1"] > scores["m2"]

    def test_returns_all_candidates(self):
        docs = {"m1": "a b", "m2": "c d", "m3": "e f"}
        scores = bm25_score_candidates("a", docs)
        assert set(scores.keys()) == {"m1", "m2", "m3"}
