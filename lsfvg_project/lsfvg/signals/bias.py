"""HTF bias: dealing range (premium / discount) and optional structure-trend alignment.

All functions take an explicit HTF bar index ``htf_idx`` (the latest CLOSED HTF bar) and only
use swings confirmed by then.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .swings import SwingSet, last_confirmed

Zone = Literal["premium", "discount", "equilibrium"]


@dataclass(frozen=True)
class DealingRange:
    high: float
    low: float
    high_idx: int
    low_idx: int

    @property
    def eq(self) -> float:
        return (self.high + self.low) / 2.0


def dealing_range(sh: SwingSet, sl: SwingSet, htf_idx: int) -> Optional[DealingRange]:
    """Latest confirmed major swing high and swing low -> range (None if missing/degenerate)."""
    if htf_idx < 0:
        return None
    a, b = last_confirmed(sh, htf_idx), last_confirmed(sl, htf_idx)
    if a is None or b is None:
        return None
    hi, lo = float(sh.price[a]), float(sl.price[b])
    if hi <= lo:
        return None
    return DealingRange(hi, lo, int(sh.idx[a]), int(sl.idx[b]))


def classify_price(rng: DealingRange, price: float, tol: float = 1e-12) -> Zone:
    """Premium above 50% of the range, discount below, equilibrium within ``tol`` of it."""
    if abs(price - rng.eq) <= tol:
        return "equilibrium"
    if price > rng.eq:
        return "premium"
    if price < rng.eq:
        return "discount"
    return "equilibrium"


def zone_allows(direction: str, zone: Zone) -> bool:
    """Longs only in discount (< 50%), shorts only in premium (> 50%)."""
    return (direction == "long" and zone == "discount") or (direction == "short" and zone == "premium")


def htf_trend(sh: SwingSet, sl: SwingSet, htf_idx: int) -> int:
    """+1 = HH+HL, -1 = LH+LL, 0 = mixed/insufficient (last two confirmed swings of each kind)."""
    if htf_idx < 0:
        return 0
    ch, cl = last_confirmed(sh, htf_idx), last_confirmed(sl, htf_idx)
    if ch is None or cl is None or ch < 1 or cl < 1:
        return 0
    hh = sh.price[ch] > sh.price[ch - 1]
    hl = sl.price[cl] > sl.price[cl - 1]
    if hh and hl:
        return 1
    if (not hh) and (not hl):
        return -1
    return 0


def trend_aligned(direction: str, trend: int) -> bool:
    return (direction == "long" and trend == 1) or (direction == "short" and trend == -1)
