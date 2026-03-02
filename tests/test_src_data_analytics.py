import pytest

from src.data.analytics import (
    detect_spikes_zscore,
    ema,
    linear_regression,
    mean,
    median,
    quantile,
    r2_score,
    rolling_mean,
    rolling_window,
    zscore
)


def test_mean():
    assert mean([1, 2, 3]) == 2.0
    with pytest.raises(ValueError, match="empty sequence"):
        mean([])
    assert mean([0.5, 1.5]) == 1.0


def test_median():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 2, 3]) == 2.5
    with pytest.raises(ValueError, match="empty sequence"):
        median([])
    assert median([5]) == 5


def test_quantile():
    assert quantile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert quantile([1, 2, 3, 4], 0.0) == 1
    assert quantile([1, 2, 3, 4], 1.0) == 4
    with pytest.raises(ValueError, match="empty sequence"):
        quantile([], 0.5)
    with pytest.raises(ValueError, match="q must be in"):
        quantile([1, 2, 3], -0.1)
    with pytest.raises(ValueError, match="q must be in"):
        quantile([1, 2, 3], 1.1)


def test_zscore():
    zs = zscore([1, 2, 3])
    assert len(zs) == 3
    assert zs[1] == pytest.approx(0.0)
    assert zs[0] == pytest.approx(-1.224744871391589)
    assert zs[2] == pytest.approx(1.224744871391589)
    assert zscore([]) == []
    assert zscore([5, 5, 5]) == [0.0, 0.0, 0.0]


def test_rolling_window():
    result = list(rolling_window([1, 2, 3, 4], 3))
    assert result == [[1, 2, 3], [2, 3, 4]]
    assert list(rolling_window([1, 2], 5)) == []
    with pytest.raises(ValueError, match="window must be > 0"):
        list(rolling_window([1, 2, 3], 0))


def test_rolling_mean():
    assert rolling_mean([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
    assert rolling_mean([1, 2], 5) == []
    with pytest.raises(ValueError, match="window must be > 0"):
        rolling_mean([1, 2, 3], 0)


def test_ema():
    assert ema([1, 2, 3], 0.5) == [1, 1.5, 2.25]
    assert ema([], 0.5) == []
    with pytest.raises(ValueError, match="alpha must be in"):
        ema([1, 2, 3], 0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        ema([1, 2, 3], 1.5)


def test_detect_spikes_zscore():
    assert detect_spikes_zscore([1, 10, 1], threshold=2) == [1]
    assert detect_spikes_zscore([1, 10, 1]) == []
    assert detect_spikes_zscore([5, 5, 5]) == []


def test_linear_regression():
    slope, intercept = linear_regression([0, 1], [0, 1])
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(0.0)
    with pytest.raises(ValueError, match="x and y must have same length"):
        linear_regression([0, 1], [0])
    with pytest.raises(ValueError, match="need at least 2 points"):
        linear_regression([1], [1])
    with pytest.raises(ValueError, match="cannot fit: all x are identical"):
        linear_regression([2, 2], [1, 2])


def test_r2_score():
    assert r2_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert r2_score([1, 2, 3], [0, 0, 0]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="length mismatch"):
        r2_score([1, 2], [1])
    with pytest.raises(ValueError, match="empty sequence"):
        r2_score([], [])

import pytest

from src.data.analytics import build_cooccurrence_graph
from src.data.analytics import cosine_similarity
from src.data.analytics import ngrams
from src.data.analytics import tokenize_simple
from src.data.analytics import top_k


def test_tokenize_simple():
    # Basic tokenization with punctuation and mixed case
    assert tokenize_simple("Hello, World! 123") == ["hello", "world", "123"]
    # Empty string returns empty list
    assert tokenize_simple("") == []
    # Numbers and letters together
    assert tokenize_simple("abc123 456def") == ["abc123", "456def"]


def test_ngrams():
    tokens = ["a", "b", "c", "d"]
    # Normal 2-grams
    assert ngrams(tokens, 2) == [("a", "b"), ("b", "c"), ("c", "d")]
    # n <= 0 raises ValueError
    with pytest.raises(ValueError):
        ngrams(tokens, 0)
    # len(tokens) < n returns empty list
    assert ngrams(tokens, 5) == []


def test_cosine_similarity():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    # Identical vectors -> similarity 1
    assert cosine_similarity(a, b) == 1.0
    # Orthogonal vectors -> similarity 0
    assert cosine_similarity(a, [0.0, 1.0, 0.0]) == 0.0
    # Zero vector -> similarity 0
    assert cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    # Length mismatch raises ValueError
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0])


def test_top_k():
    items = ["apple", "banana", "apple", "orange", "banana", "apple"]
    # k <= 0 returns empty list
    assert top_k(items, 0) == []
    # Normal top 2
    assert top_k(items, 2) == [("apple", 3), ("banana", 2)]
    # Ties handled by Counter.most_common order
    assert top_k(["a", "b", "c", "a", "b"], 2) == [("a", 2), ("b", 2)]
    # k larger than unique items returns all
    assert top_k(["x", "y"], 5) == [("x", 1), ("y", 1)]


def test_build_cooccurrence_graph():
    tokens = ["a", "b", "c", "a"]
    # window <= 0 raises ValueError
    with pytest.raises(ValueError):
        build_cooccurrence_graph(tokens, window=0)
    # Simple graph with window=2
    graph = build_cooccurrence_graph(tokens, window=2)
    expected = {
        "a": {"b": 1, "c": 1},
        "b": {"a": 1, "c": 1},
        "c": {"a": 1, "b": 1},
    }
    assert graph == expected
    # Window larger than length includes all pairs
    full_graph = build_cooccurrence_graph(tokens, window=10)
    assert full_graph["a"]["b"] == 1
    assert full_graph["a"]["c"] == 1
    # Duplicate tokens are ignored in co-occurrence
    dup_tokens = ["x", "x", "y"]
    dup_graph = build_cooccurrence_graph(dup_tokens, window=1)
    assert dup_graph == {"x": {"y": 1}, "y": {"x": 1}}
