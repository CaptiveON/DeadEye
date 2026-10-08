"""Statistics used by the report: normalisation, bootstrap CIs, IQM, paired permutation tests."""
from __future__ import annotations

import numpy as np


def normalize(values: np.ndarray, random_mean: float, oracle_mean: float) -> np.ndarray:
    """Map returns onto [random=0, oracle=1]. Values can exceed the range (luck) or be negative."""
    span = oracle_mean - random_mean
    if abs(span) < 1e-12:
        return np.full_like(np.asarray(values, dtype=float), np.nan)
    return (np.asarray(values, dtype=float) - random_mean) / span


def iqm(x: np.ndarray) -> float:
    """Interquartile mean (Agarwal et al., 2021): mean of the middle 50% of values."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(np.mean(x[lo:hi])) if hi > lo else float(np.mean(x))


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    stats = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a - b) = 0 under random sign flips of the paired differences."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan")
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d[None, :]).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def holm_correction(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    return float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")


def episodes_needed(effect: float, sd: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Approximate paired-sample size for a mean difference `effect` with SD of differences `sd`."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil((z * sd / effect) ** 2))
