from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


# 16
def mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("empty sequence")
    return sum(xs) / float(len(xs))


# 17
def median(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("empty sequence")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if (n % 2 == 1) else (s[mid - 1] + s[mid]) / 2.0


# 18
def quantile(xs: Sequence[float], q: float) -> float:
    """q in [0,1]. Linear interpolation between points."""
    if not xs:
        raise ValueError("empty sequence")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# 19
def zscore(xs: Sequence[float]) -> List[float]:
    if not xs:
        return []
    m = mean(xs)
    var = mean([(x - m) ** 2 for x in xs])
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0 for _ in xs]
    return [(x - m) / sd for x in xs]


# 20
def rolling_window(xs: Sequence[float], window: int) -> Iterator[List[float]]:
    if window <= 0:
        raise ValueError("window must be > 0")
    buf: List[float] = []
    for x in xs:
        buf.append(x)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) == window:
            yield list(buf)


# 21
def rolling_mean(xs: Sequence[float], window: int) -> List[float]:
    out: List[float] = []
    for w in rolling_window(xs, window):
        out.append(mean(w))
    return out


# 22
def ema(xs: Sequence[float], alpha: float) -> List[float]:
    """Exponential moving average. alpha in (0,1]."""
    if not xs:
        return []
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    out = [xs[0]]
    for x in xs[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


# 23
def detect_spikes_zscore(xs: Sequence[float], *, threshold: float = 3.0) -> List[int]:
    """Return indices where |z| >= threshold."""
    zs = zscore(xs)
    return [i for i, z in enumerate(zs) if abs(z) >= threshold]


# 24
def linear_regression(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Return (slope, intercept) for y = m*x + b."""
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) < 2:
        raise ValueError("need at least 2 points")
    mx = mean(x)
    my = mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    if den == 0:
        raise ValueError("cannot fit: all x are identical")
    m = num / den
    b = my - m * mx
    return m, b


# 25
def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    if not y_true:
        raise ValueError("empty sequence")
    ybar = mean(y_true)
    ss_tot = sum((y - ybar) ** 2 for y in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    return 1.0 if ss_tot == 0 else (1.0 - ss_res / ss_tot)


# 26
def tokenize_simple(text: str) -> List[str]:
    """Lowercase word tokenizer (letters+digits)."""
    return re.findall(r"[a-z0-9]+", text.lower())


# 27
def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be > 0")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# 28
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# 29
def top_k(items: Iterable[T], k: int) -> List[Tuple[T, int]]:
    """Top-k most common items."""
    if k <= 0:
        return []
    c = Counter(items)
    return c.most_common(k)


# 30
def build_cooccurrence_graph(tokens: Sequence[str], *, window: int = 2) -> Dict[str, Dict[str, int]]:
    """
    Build an undirected weighted co-occurrence graph:
    graph[u][v] = number of times u and v co-occur within 'window'.
    """
    if window <= 0:
        raise ValueError("window must be > 0")

    g: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n = len(tokens)
    for i, u in enumerate(tokens):
        jmax = min(n, i + window + 1)
        for j in range(i + 1, jmax):
            v = tokens[j]
            if u == v:
                continue
            g[u][v] += 1
            g[v][u] += 1
    # convert nested defaultdicts to normal dicts
    return {u: dict(nei) for u, nei in g.items()}


if __name__ == "__main__":
    # Small demo:
    series = [10, 11, 10, 12, 11, 500, 12, 11, 10]
    print("spikes @", detect_spikes_zscore(series, threshold=2.5))

    text = "AI helps teams build AI systems. Teams build products faster with AI."
    toks = tokenize_simple(text)
    print("top words:", top_k(toks, 5))
    graph = build_cooccurrence_graph(toks, window=2)
    print("neighbors(ai):", graph.get("ai", {}))
