"""Displacement + market-structure shift (MSS) after a sweep (pure function)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import MSS, Sweep
from .swings import SwingSet, last_confirmed


def find_mss(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_prior: np.ndarray,
    sweep: Sweep,
    opposing: SwingSet,
    *,
    disp_mult: float,
    max_bars: int,
    end_idx: int,
) -> tuple[Optional[MSS], str]:
    """First candle after the sweep that closes beyond the last opposing swing with a big body.

    * short setup (buy-side sweep): opposing swing = last confirmed swing LOW formed before the
      sweep extreme; MSS candle is bearish, closes below it, body >= ``disp_mult`` x ATR.
    * long setup: mirror image with swing highs.
    * search window: reclaim bar .. reclaim + ``max_bars`` (never beyond ``end_idx``);
    * if any bar in between closes back beyond the pool level in the sweep direction the
      sweep has failed ("sweep_invalidated").

    Returns ``(mss, reason)``; ``reason`` is ``""`` on success, otherwise one of
    ``pending`` (window not finished yet), ``no_opposing_swing``, ``no_displacement``,
    ``sweep_invalidated``.
    """
    short = sweep.direction == "short"
    last = min(sweep.reclaim_idx + max_bars, end_idx)
    saw_swing = False
    for m in range(sweep.reclaim_idx, last + 1):
        if m > sweep.reclaim_idx:
            c_prev = close[m - 1]
            if (short and c_prev > sweep.pool_level) or ((not short) and c_prev < sweep.pool_level):
                return None, "sweep_invalidated"
        pos = last_confirmed(opposing, m, pivot_before=sweep.extreme_idx)
        if pos is None:
            continue
        saw_swing = True
        level = float(opposing.price[pos])
        a = atr_prior[m]
        if not np.isfinite(a):
            continue
        body = abs(close[m] - open_[m])
        if short:
            ok = close[m] < open_[m] and close[m] < level
        else:
            ok = close[m] > open_[m] and close[m] > level
        if ok and body >= disp_mult * a:
            return MSS(int(m), int(opposing.idx[pos]), level, float(body), float(body / a if a else np.inf)), ""
    if sweep.reclaim_idx + max_bars > end_idx:
        return None, "pending"
    return None, "no_displacement" if saw_swing else "no_opposing_swing"
