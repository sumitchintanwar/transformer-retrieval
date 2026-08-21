import pytest
import math
from evaluation import precision_at_k, recall_at_k, mrr_at_k
from hybrid_search import min_max_normalize

# --- Precision@K Tests ---

def test_precision_all_relevant():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0

def test_precision_none_relevant():
    assert precision_at_k([1, 2, 3], {4, 5, 6}, 3) == 0.0

def test_precision_some_relevant():
    assert precision_at_k([1, 2, 3, 4], {2, 4, 9}, 4) == 0.5

def test_precision_k_1():
    assert precision_at_k([1, 2], {1}, 1) == 1.0
    assert precision_at_k([1, 2], {2}, 1) == 0.0

def test_precision_fewer_retrieved_than_k():
    # If K=5 but only 2 retrieved, precision denominator should be K
    # Our function divides by K: hits / k
    assert precision_at_k([1, 2], {1, 2}, 5) == 2 / 5

def test_precision_empty_retrieved():
    assert precision_at_k([], {1, 2}, 3) == 0.0

def test_precision_empty_relevant():
    assert precision_at_k([1, 2], set(), 3) == 0.0

# --- Recall@K Tests ---

def test_recall_all_retrieved():
    assert recall_at_k([1, 2, 3], {1, 2}, 3) == 1.0

def test_recall_none_retrieved():
    assert recall_at_k([1, 2, 3], {4, 5}, 3) == 0.0

def test_recall_partial_retrieval():
    assert recall_at_k([1, 2, 3, 4], {2, 4, 6, 8}, 4) == 2 / 4

def test_recall_empty_relevant():
    # Current implementation returns 0.0 if relevant_ids is empty
    assert recall_at_k([1, 2], set(), 3) == 0.0

def test_recall_empty_retrieved():
    assert recall_at_k([], {1, 2}, 3) == 0.0

# --- MRR@K Tests ---

def test_mrr_first_relevant():
    assert mrr_at_k([1, 2, 3], {1}, 3) == 1.0

def test_mrr_second_relevant():
    assert mrr_at_k([1, 2, 3], {2}, 3) == 0.5

def test_mrr_relevant_at_k():
    assert mrr_at_k([1, 2, 3], {3}, 3) == 1.0 / 3.0

def test_mrr_no_relevant():
    assert mrr_at_k([1, 2, 3], {4, 5}, 3) == 0.0

def test_mrr_multiple_relevant():
    # Should only consider the first relevant one
    assert mrr_at_k([1, 2, 3], {2, 3}, 3) == 0.5

def test_mrr_empty_results():
    assert mrr_at_k([], {1}, 3) == 0.0

# --- Min-Max Normalization Tests ---

def test_min_max_normal_range():
    scores = [10.0, 20.0, 30.0]
    expected = [0.0, 0.5, 1.0]
    assert min_max_normalize(scores) == expected

def test_min_max_negative_values():
    scores = [-10.0, 0.0, 10.0]
    expected = [0.0, 0.5, 1.0]
    assert min_max_normalize(scores) == expected

def test_min_max_identical_scores():
    scores = [5.0, 5.0, 5.0]
    expected = [1.0, 1.0, 1.0]
    assert min_max_normalize(scores) == expected

def test_min_max_single_score():
    assert min_max_normalize([42.0]) == [1.0]

def test_min_max_empty_input():
    assert min_max_normalize([]) == []
