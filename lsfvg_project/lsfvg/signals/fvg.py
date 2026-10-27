"""Fair value gaps (three-candle imbalances) - pure functions."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import FVG


def fvg_at(high: np.ndarray, low: np.ndarray, i: int) -> Optional[FVG]:
    """FVG completed by candle ``i`` (needs ``i >= 2``).

    bullish: ``high[i-2] < low[i]``  (gap = [high[i-2], low[i]])
    bearish: ``low[i-2] > high[i]``  (gap = [high[i], low[i-2]])
    """
    if i < 2:
        return None
    if high[i - 2] < low[i]:
        return FVG("bull", i - 2, i - 1, i, float(high[i - 2]), float(low[i]), float(high[i - 2]), float(low[i - 2]))
    if low[i - 2] > high[i]:
        return FVG("bear", i - 2, i - 1, i, float(high[i]), float(low[i - 2]), float(high[i - 2]), float(low[i - 2]))
    return None


def find_fvgs(
    high: np.ndarray,
    low: np.ndarray,
    atr_prior: np.ndarray,
    direction: str,
    *,
    lo_mid: int,
    hi_mid: int,
    min_atr: float,
    end_idx: int,
) -> list[FVG]:
    """All FVGs of the wanted direction whose MIDDLE candle lies in ``[lo_mid, hi_mid]``.

    ``direction`` is the trade direction (long -> bullish FVG).  Gap must be known by
    ``end_idx`` (completing candle = mid + 1) and be at least ``min_atr`` x ATR(before mid).
    """
    want = "bull" if direction == "long" else "bear"
    out: list[FVG] = []
    for mid in range(max(lo_mid, 1), hi_mid + 1):
        i = mid + 1
        if i > end_idx:
            break
        f = fvg_at(high, low, i)
        if f is None or f.direction != want:
            continue
        a = atr_prior[mid]
        if not np.isfinite(a) or f.size < min_atr * a:
            continue
        out.append(f)
    return out


def entry_price(fvg: FVG, direction: str, mode: str) -> float:
    """Limit price: ``edge`` = proximal edge (first touched on the retrace), ``midpoint`` = 50%."""
    if mode == "midpoint":
        return fvg.mid
    return fvg.top if direction == "long" else fvg.bottom


def is_mitigated(
    fvg: FVG, direction: str, entry: float, high: np.ndarray, low: np.ndarray, close: np.ndarray, upto_idx: int
) -> bool:
    """Entry level already traded, or gap closed through, by candles after the gap completed."""
    for j in range(fvg.last_idx + 1, upto_idx + 1):
        if direction == "long":
            if low[j] <= entry or close[j] < fvg.bottom:
                return True
        else:
            if high[j] >= entry or close[j] > fvg.top:
                return True
    return False


def select_fvg(cands: list[FVG], rule: str) -> Optional[FVG]:
    if not cands:
        return None
    if rule == "first":
        return min(cands, key=lambda f: f.mid_idx)
    if rule == "latest":
        return max(cands, key=lambda f: f.mid_idx)
    return max(cands, key=lambda f: (f.size, -f.mid_idx))
