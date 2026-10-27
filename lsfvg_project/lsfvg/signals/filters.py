"""Entry filters (all toggleable via config).  Each returns True when the setup PASSES."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config.schema import FiltersCfg, SessionWindow
from ..core.timeutil import in_window_mask, minute_of_day_local
from ..data.news import in_blackout


def session_mask(index: pd.DatetimeIndex, windows: list[SessionWindow], tz: str) -> np.ndarray:
    """True where the bar OPEN time falls in any entry window (America/New_York wall clock)."""
    if not windows:
        return np.ones(len(index), dtype=bool)
    mod = minute_of_day_local(index, tz)
    m = np.zeros(len(index), dtype=bool)
    for w in windows:
        m |= in_window_mask(mod, w.start, w.end)
    return m


def spread_ok(spread: float, max_spread: float) -> bool:
    return bool(np.isfinite(spread) and spread <= max_spread)


def volatility_ok(atr_percentile: float, min_percentile: float) -> bool:
    """Skip low-volatility regimes; unknown percentile (not enough history) fails closed."""
    return bool(np.isfinite(atr_percentile) and atr_percentile >= min_percentile)


def chop_ok(efficiency_ratio: float, min_er: float) -> bool:
    return bool(np.isfinite(efficiency_ratio) and efficiency_ratio >= min_er)


@dataclass
class FilterInputs:
    """Per-MTF-bar arrays the engine precomputes (all causal)."""

    session_ok: np.ndarray
    spread: np.ndarray
    atr_pct: np.ndarray
    er: np.ndarray
    close_ns: np.ndarray
    news_starts: np.ndarray = field(default_factory=lambda: np.empty(0, "int64"))
    news_ends: np.ndarray = field(default_factory=lambda: np.empty(0, "int64"))


def failed_filters(i: int, fi: FilterInputs, cfg: FiltersCfg, max_spread: float) -> list[str]:
    """Reason codes of every enabled filter that fails at MTF signal bar ``i``."""
    out: list[str] = []
    if cfg.session_enabled and not fi.session_ok[i]:
        out.append("filter_session")
    if cfg.news.enabled and in_blackout(int(fi.close_ns[i]), fi.news_starts, fi.news_ends):
        out.append("filter_news")
    if cfg.spread_enabled and not spread_ok(float(fi.spread[i]), max_spread):
        out.append("filter_spread")
    if cfg.volatility.enabled and not volatility_ok(float(fi.atr_pct[i]), cfg.volatility.min_percentile):
        out.append("filter_volatility")
    if cfg.chop.enabled and not chop_ok(float(fi.er[i]), cfg.chop.min_er):
        out.append("filter_chop")
    return out
