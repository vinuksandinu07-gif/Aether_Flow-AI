"""Causal indicators: value at bar ``i`` uses only bars ``<= i``."""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev = np.r_[np.nan, close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    tr[0] = high[0] - low[0]
    return tr


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder ATR; NaN until ``n`` bars exist.  ATR[i] includes bar i."""
    h, l, c = (np.asarray(x, float) for x in (high, low, close))
    out = np.full(len(c), np.nan)
    if len(c) < n:
        return out
    tr = true_range(h, l, c)
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(c)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def atr_prior(atr: np.ndarray) -> np.ndarray:
    """ATR as known BEFORE bar i (``atr[i-1]``) - used for thresholds on bar i's own size."""
    return np.r_[np.nan, atr[:-1]]


def efficiency_ratio(close: np.ndarray, n: int) -> np.ndarray:
    """Kaufman efficiency ratio over ``n`` bars (1 = straight trend, 0 = pure chop)."""
    c = pd.Series(close, dtype=float)
    net = (c - c.shift(n)).abs()
    path = c.diff().abs().rolling(n).sum()
    er = (net / path.replace(0.0, np.nan)).to_numpy()
    return er


def rolling_percentile(x: np.ndarray, window: int) -> np.ndarray:
    """Percentile rank (0..1) of ``x[i]`` within the trailing ``window`` values (incl. itself)."""
    s = pd.Series(x, dtype=float)
    return s.rolling(window, min_periods=max(20, window // 10)).rank(pct=True).to_numpy()
