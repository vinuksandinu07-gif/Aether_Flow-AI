"""Spread model shared by data synthesis, the signal engine (stop buffer, RR) and the backtester."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.schema import InstrumentCfg
from ..core.timeutil import NS_DAY, NS_HOUR, local_naive_ns

PRICE_FIELDS = ("open", "high", "low", "close")


class SpreadModel:
    """Estimated full spread (ask - bid) in price units for any timestamp."""

    def __init__(self, ins: InstrumentCfg, tz: str = "America/New_York") -> None:
        self.kind = ins.spread.kind
        self.typical = float(ins.spread.typical)
        self.tz = tz
        self.by_hour = {int(k): float(v) for k, v in ins.spread.by_ny_hour.items()}

    def at(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.kind == "fixed" or not self.by_hour:
            return np.full(len(index), self.typical)
        hours = (local_naive_ns(index, self.tz) % NS_DAY) // NS_HOUR
        table = np.array([self.by_hour.get(h, self.typical) for h in range(24)])
        return table[hours]


def add_modeled_bid_ask(df: pd.DataFrame, model: SpreadModel) -> pd.DataFrame:
    """Return ``df`` plus bid_*/ask_* columns built as mid -/+ half a modeled spread.

    The frame is flagged via ``attrs['spread_modeled'] = True`` so reports can
    say so explicitly (mid + modeled spread, never mid alone).
    """
    out = df.copy()
    half = model.at(out.index) / 2.0
    for f in PRICE_FIELDS:
        out[f"bid_{f}"] = out[f] - half
        out[f"ask_{f}"] = out[f] + half
    out.attrs["spread_modeled"] = True
    return out
