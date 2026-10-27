"""Liquidity sweep detection on the MTF (pure function)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import Pool, Sweep


def detect_sweep(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_prior: np.ndarray,
    pool: Pool,
    *,
    min_sweep_atr: float,
    reclaim_bars: int,
    end_idx: int,
) -> Optional[Sweep]:
    """Sweep of ``pool`` = trade beyond it by >= ``min_sweep_atr`` x ATR and a close back inside.

    * breach bar ``t0`` = ``pool.swept_idx`` (first strict trade-through);
    * the close back inside must come on bar ``t0 .. t0 + reclaim_bars`` (breach bar counts as 0);
    * the reclaim bar must be ``<= end_idx`` (the last closed bar) - nothing later is read;
    * depth = extreme beyond the level over ``t0 .. reclaim`` and must reach the minimum.

    Buy-side pool -> price ran above highs -> SHORT setup.  Sell-side -> LONG.
    """
    t0 = pool.swept_idx
    if t0 is None or t0 > end_idx:
        return None
    a = atr_prior[t0]
    if not np.isfinite(a):
        return None
    last = min(t0 + reclaim_bars, end_idx)
    buy = pool.side == "buy"
    for tc in range(t0, last + 1):
        back_inside = close[tc] < pool.level if buy else close[tc] > pool.level
        if not back_inside:
            continue
        seg = slice(t0, tc + 1)
        if buy:
            ext_i = t0 + int(np.argmax(high[seg]))
            ext = float(high[ext_i])
            depth = ext - pool.level
        else:
            ext_i = t0 + int(np.argmin(low[seg]))
            ext = float(low[ext_i])
            depth = pool.level - ext
        if depth < min_sweep_atr * a:
            return None
        return Sweep(pool.pool_id, "short" if buy else "long", pool.level, int(t0), int(tc),
                     int(ext_i), ext, float(depth))
    return None
