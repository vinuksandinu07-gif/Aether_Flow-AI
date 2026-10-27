"""Futures contract rolls.  CFDs (OANDA SPX500_USD) are continuous - this is for futures data.

Two supported policies (the chosen one is stated in config ``futures.roll_method``):

* ``back_adjusted_difference`` / ``back_adjusted_ratio``: one continuous series; older contracts are
  shifted (difference) or scaled (ratio) so there is no artificial gap at the roll.  Absolute price
  levels of old bars are therefore NOT real traded prices (matters for round-number/pool levels).
* ``per_contract``: each contract kept separately (no adjustment); never stitch these for signals.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from .spread import PRICE_FIELDS


def back_adjust(
    contracts: list[tuple[pd.DataFrame, pd.Timestamp]],
    method: Literal["difference", "ratio"] = "difference",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stitch ``[(frame, roll_time), ...]`` (oldest first) into a back-adjusted continuous frame.

    ``roll_time`` is the first timestamp at which the NEXT contract takes over (the last frame's
    roll_time is ignored).  At each roll the adjustment is computed from the two contracts' closes at
    the roll (old contract's last close before the roll vs new contract's first close at/after it).
    Returns ``(continuous, roll_log)``.
    """
    if not contracts:
        raise ValueError("no contracts")
    cols = [c for c in contracts[0][0].columns if c in PRICE_FIELDS or c.startswith(("bid_", "ask_"))]
    pieces: list[pd.DataFrame] = []
    log_rows = []
    cum_diff, cum_ratio = 0.0, 1.0
    for k in range(len(contracts) - 1, -1, -1):          # walk newest -> oldest
        frame, _ = contracts[k]
        start_t = contracts[k - 1][1] if k > 0 else frame.index[0]
        end_t = contracts[k][1] if k < len(contracts) - 1 else None
        seg = frame[frame.index >= start_t]
        if end_t is not None:
            seg = seg[seg.index < end_t]
        seg = seg.copy()
        if method == "difference":
            seg[cols] = seg[cols] + cum_diff
        else:
            seg[cols] = seg[cols] * cum_ratio
        pieces.append(seg)
        if k > 0:
            prev_frame, prev_roll = contracts[k - 1]
            old_close = prev_frame.loc[prev_frame.index < prev_roll, "close"].iloc[-1]
            new_close = frame.loc[frame.index >= prev_roll, "close"].iloc[0]
            if method == "difference":
                step = new_close - old_close
                cum_diff += step
            else:
                step = new_close / old_close
                cum_ratio *= step
            log_rows.append({"roll_time": prev_roll, "old_close": old_close, "new_close": new_close,
                             "adjustment": step, "method": method})
    out = pd.concat(pieces[::-1]).sort_index()
    return out, pd.DataFrame(log_rows[::-1])
