"""Optional economic-calendar CSV -> news blackout windows.

CSV columns: ``timestamp_utc,currency,impact,event`` (impact: low|medium|high).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.timeutil import to_ns

IMPACT_RANK = {"low": 0, "medium": 1, "high": 2}


def load_news_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"timestamp_utc", "currency", "impact"} - set(df.columns)
    if missing:
        raise ValueError(f"news CSV missing columns {sorted(missing)}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["impact"] = df["impact"].str.strip().str.lower()
    df["currency"] = df["currency"].str.strip().str.upper()
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def blackout_windows(
    news: pd.DataFrame | None, currencies: list[str], min_impact: str, before_min: int, after_min: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted arrays (start_ns, end_ns) of blackout windows relevant to the instrument."""
    if news is None or news.empty:
        return np.empty(0, "int64"), np.empty(0, "int64")
    rank = IMPACT_RANK[min_impact]
    sel = news[news["currency"].isin([c.upper() for c in currencies])
               & (news["impact"].map(IMPACT_RANK).fillna(-1) >= rank)]
    ev = to_ns(pd.DatetimeIndex(sel["timestamp_utc"]))
    ev.sort()
    return ev - before_min * 60 * 10**9, ev + after_min * 60 * 10**9


def in_blackout(t_ns: int, starts: np.ndarray, ends: np.ndarray) -> bool:
    """True if ``t_ns`` lies inside any [start, end] window (windows may overlap)."""
    if len(starts) == 0:
        return False
    i = np.searchsorted(starts, t_ns, side="right")      # windows with start <= t
    return bool(i > 0 and (ends[:i] >= t_ns).any())
