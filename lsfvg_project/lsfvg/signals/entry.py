"""Entry-order lifecycle (rule 7) - pure; reused unchanged by the Phase 2 backtester and live.

The limit order exists from ``place_time`` (close of the MTF bar that completed the setup) on LTF bars.
Per LTF bar, in this conservative order:

1. **target taken before fill** -> cancel (checked first, so a bar that both reaches the target and the
   entry never counts as a fill);
2. **fill**: price must trade THROUGH the limit by ``fill_through_ticks`` (long: ask low, short: bid high
   when quotes exist; otherwise mid).  A gap through the limit fills at the open (better price);
3. **close fully through the FVG** (long: below its bottom, short: above its top) -> cancel;
4. **expiry** after ``max_wait_bars`` LTF bars without a fill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.timeutil import to_ns
from ..core.types import EntryOrder, OrderOutcome


def resolve_entry_order(order: EntryOrder, ltf: pd.DataFrame) -> OrderOutcome:
    """Walk LTF bars from ``place_time`` and decide the order's fate."""
    opens = to_ns(ltf.index)
    start = int(np.searchsorted(opens, int(pd.Timestamp(order.place_time).value), "left"))
    high, low, close = (ltf[c].to_numpy(float) for c in ("high", "low", "close"))
    long = order.direction == "long"
    has_q = all(c in ltf for c in (("ask_low", "ask_open") if long else ("bid_high", "bid_open")))
    q_ext = (ltf["ask_low"] if long else ltf["bid_high"]).to_numpy(float) if has_q else (low if long else high)
    q_open = (ltf["ask_open"] if long else ltf["bid_open"]).to_numpy(float) if has_q else ltf["open"].to_numpy(float)
    th = order.fill_through_ticks * order.tick_size
    tgt = order.target_level
    for j in range(start, min(len(ltf), start + order.max_wait_bars)):
        age = j - start + 1
        t = ltf.index[j]
        if tgt is not None and ((long and high[j] > tgt) or ((not long) and low[j] < tgt)):
            return OrderOutcome("cancelled_target_taken", j, t)
        if (long and q_ext[j] <= order.limit_price - th) or ((not long) and q_ext[j] >= order.limit_price + th):
            px = min(order.limit_price, q_open[j]) if long else max(order.limit_price, q_open[j])
            return OrderOutcome("filled", j, t, float(px))
        if (long and close[j] < order.fvg_bottom) or ((not long) and close[j] > order.fvg_top):
            return OrderOutcome("cancelled_fvg_through", j, t)
        if age >= order.max_wait_bars:
            return OrderOutcome("expired", j, t)
    return OrderOutcome("pending", None, None)
