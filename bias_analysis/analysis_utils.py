"""Shared stats helpers."""

from __future__ import annotations

import numpy as np
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None


def apply_fdr(p_values: list | np.ndarray) -> np.ndarray:
    """BH FDR with NaN passthrough."""
    arr   = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(arr)
    result = arr.copy()
    n_valid = int(valid.sum())
    if n_valid >= 2:
        if multipletests is not None:
            _, corrected, _, _ = multipletests(arr[valid], method="fdr_bh")
        else:
            corrected = _bh_fdr(arr[valid])
        result[valid] = corrected
    elif n_valid == 1:
        print("  [NOTE] Only 1 valid p-value in family; FDR correction not applied.")
    return result


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * n / rank)
        adjusted[i] = value
        prev = value
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def effect_label_eta2(e: float) -> str:
    if np.isnan(e): return "unknown"
    if e < 0.01:    return "trivial"
    if e < 0.06:    return "small"
    if e < 0.14:    return "medium"
    return "large"


def effect_label_eps2(e: float) -> str:
    if np.isnan(e): return "unknown"
    if e < 0.01:    return "trivial"
    if e < 0.08:    return "small"
    if e < 0.26:    return "medium"
    return "large"


def effect_label_r(r: float) -> str:
    if np.isnan(r): return "unknown"
    r = abs(r)
    if r < 0.10:    return "trivial"
    if r < 0.30:    return "small"
    if r < 0.50:    return "medium"
    return "large"


def effect_label_cramers_v(v: float) -> str:
    """Effect size label for Cramér's V (Cohen 1988)."""
    if np.isnan(v): return "unknown"
    v = abs(v)
    if v < 0.10:    return "trivial"
    if v < 0.30:    return "small"
    if v < 0.50:    return "medium"
    return "large"


def rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    denom = n1 * n2
    return float(1 - (2 * u_stat / denom)) if denom > 0 else np.nan


def r_from_wilcoxon(result, n: int) -> float:
    """Effect size r for Wilcoxon."""
    if hasattr(result, "zstatistic"):
        z_raw = float(result.zstatistic)
        if not np.isnan(z_raw):
            return abs(z_raw) / np.sqrt(max(n, 1))

    # Normal approximation (Hollander & Wolfe 1999)
    stat = float(result.statistic) if hasattr(result, "statistic") else float(result)
    mu   = n * (n + 1) / 4
    sig  = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 + 1e-9)
    z    = (stat - mu) / sig
    return abs(z) / np.sqrt(max(n, 1))
