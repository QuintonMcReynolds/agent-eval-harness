"""Statistics for agent evals. Tasks are the unit of analysis: samples of one task are not independent."""
from __future__ import annotations

from math import comb, sqrt

import numpy as np
from scipy.stats import binomtest


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """P(all k i.i.d. attempts succeed), unbiased from n samples with c successes (tau-bench's pass^k)."""
    return comb(c, k) / comb(n, k) if n >= k else float("nan")


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k attempts succeeds) (Chen et al. 2021)."""
    return 1 - comb(n - c, k) / comb(n, k) if n >= k else float("nan")


def mcnemar(base: dict[str, bool], cand: dict[str, bool]) -> dict:
    """Exact McNemar on tasks present in both. b = base pass/cand fail (regressions), c = improvements."""
    common = sorted(set(base) & set(cand))
    b = sum(base[t] and not cand[t] for t in common)
    c = sum(cand[t] and not base[t] for t in common)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"n_tasks": len(common), "regressions": b, "improvements": c, "p_value": float(p)}


def paired_bootstrap(base: dict[str, float], cand: dict[str, float], n_boot: int = 10000, seed: int = 0) -> dict:
    """CI for mean(cand - base) over tasks, resampling tasks."""
    common = sorted(set(base) & set(cand))
    d = np.array([cand[t] - base[t] for t in common], dtype=float)
    if len(d) == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))}


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
