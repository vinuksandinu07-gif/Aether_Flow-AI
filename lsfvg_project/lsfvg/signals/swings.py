"""N-bar fractal swings, confirmed N bars after the pivot.

A bar ``i`` is a swing HIGH if ``high[i] >= max(high[i-N:i])`` and
``high[i] > max(high[i+1:i+N+1])``.  ``>=`` on the left / ``>`` on the right lets a flat
top (equal highs) register once, on its LAST bar.  Lows are the mirror image.

Confirmation: the pivot is only *known* at bar ``i + N`` (``confirm_idx``).  Consumers must
query with :func:`confirmed_upto` so a swing is never used before it exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwingSet:
    """Parallel arrays sorted by pivot index."""

    idx: np.ndarray          # pivot bar index
    price: np.ndarray
    confirm_idx: np.ndarray  # idx + n
    n: int

    def __len__(self) -> int:
        return len(self.idx)


def find_swings(high: np.ndarray, low: np.ndarray, n: int) -> tuple[SwingSet, SwingSet]:
    """Return ``(swing_highs, swing_lows)`` for the whole series (pivots with a full window only)."""
    h = pd.Series(np.asarray(high, float))
    l = pd.Series(np.asarray(low, float))
    left_max = h.shift(1).rolling(n).max()
    right_max = h[::-1].shift(1).rolling(n).max()[::-1]
    left_min = l.shift(1).rolling(n).min()
    right_min = l[::-1].shift(1).rolling(n).min()[::-1]
    is_h = ((h >= left_max) & (h > right_max)).to_numpy()
    is_l = ((l <= left_min) & (l < right_min)).to_numpy()
    hi = np.nonzero(is_h)[0]
    lo = np.nonzero(is_l)[0]
    return (SwingSet(hi, h.to_numpy()[hi], hi + n, n), SwingSet(lo, l.to_numpy()[lo], lo + n, n))


def confirmed_upto(sw: SwingSet, idx: int) -> int:
    """Number of swings (a prefix of the arrays) confirmed by the close of bar ``idx``."""
    return int(np.searchsorted(sw.confirm_idx, idx, side="right"))


def last_confirmed(sw: SwingSet, idx: int, pivot_before: int | None = None) -> int | None:
    """Position (into ``sw`` arrays) of the latest swing confirmed by ``idx``.

    If ``pivot_before`` is given, additionally require ``pivot index < pivot_before``.
    Returns ``None`` when there is none.
    """
    c = confirmed_upto(sw, idx)
    if pivot_before is not None:
        c = min(c, int(np.searchsorted(sw.idx, pivot_before, side="left")))
    return c - 1 if c > 0 else None
