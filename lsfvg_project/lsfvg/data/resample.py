"""Resample once, from the base dataset, and align higher timeframes without lookahead.

Rules enforced here:

* every bar is labelled by its **open** time (UTC) and carries ``close_time``;
* a higher-timeframe bar is *available* only from its ``close_time`` on
  (:func:`latest_closed_index`);
* the trailing, still-forming bin is dropped (it would be repainted later).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config.schema import InstrumentCfg, TimeframesCfg
from ..core.timeutil import bin_open_ns, from_ns, parse_tf, to_ns


def _agg_map(columns: list[str]) -> dict[str, str]:
    agg: dict[str, str] = {}
    for c in columns:
        if c == "volume":
            agg[c] = "sum"
        elif c.endswith("open"):
            agg[c] = "first"
        elif c.endswith("high"):
            agg[c] = "max"
        elif c.endswith("low"):
            agg[c] = "min"
        elif c.endswith("close"):
            agg[c] = "last"
    return agg


def resample_ohlc(
    df: pd.DataFrame,
    tf: str,
    *,
    tz: str,
    anchor: str,
    base_tf: str,
    drop_incomplete: bool = True,
    drop_partial_first: bool = True,
) -> pd.DataFrame:
    """Aggregate base bars into anchored ``tf`` bars (mid and bid/ask columns alike).

    A bin is *complete* when its close time is not after the close of the last
    base bar.  Weekend/holiday gaps do not make a bin incomplete.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("resample_ohlc needs a sorted index")
    if df.index.has_duplicates:
        raise ValueError("resample_ohlc refuses duplicate timestamps; resolve them explicitly first")
    tf_td, base_td = parse_tf(tf), parse_tf(base_tf)
    keys = bin_open_ns(df.index, tf, tz, anchor)
    agg = _agg_map(list(df.columns))
    g = df[list(agg)].groupby(keys, sort=True)
    out = g.agg(agg)
    out["n_base"] = g.size().to_numpy()
    out.index = from_ns(out.index.to_numpy())
    out.index.name = "time"
    out["close_time"] = out.index + tf_td

    keep = np.ones(len(out), dtype=bool)
    if drop_incomplete:
        last_close = df.index[-1] + base_td
        keep &= (out["close_time"] <= last_close).to_numpy()
    if drop_partial_first and len(out) and out.index[0] < df.index[0]:
        keep[0] = False
    out = out[keep]
    out.attrs.update(df.attrs)
    return out


def latest_closed_index(target_close_ns: np.ndarray, higher_close_ns: np.ndarray) -> np.ndarray:
    """For each timestamp, the index of the latest higher-TF bar already CLOSED (-1 if none)."""
    return np.searchsorted(higher_close_ns, target_close_ns, side="right") - 1


@dataclass
class TimeframeStack:
    """base -> ltf -> mtf -> htf, all derived from one base dataset."""

    base: pd.DataFrame
    ltf: pd.DataFrame
    mtf: pd.DataFrame
    htf: pd.DataFrame
    tz: str
    anchor: str
    htf_idx_at_mtf: np.ndarray      # per MTF bar: latest HTF bar closed at that MTF bar's close

    @property
    def mtf_close_ns(self) -> np.ndarray:
        return to_ns(pd.DatetimeIndex(self.mtf["close_time"]))


def build_stack(
    base: pd.DataFrame, tfs: TimeframesCfg, ins: InstrumentCfg, tz: str = "America/New_York"
) -> TimeframeStack:
    """Resample ``base`` to LTF / MTF / HTF and compute the closed-HTF alignment."""
    anchor = ins.day_anchor_ny
    base_td = parse_tf(tfs.base)

    def rs(tf: str) -> pd.DataFrame:
        return resample_ohlc(base, tf, tz=tz, anchor=anchor, base_tf=tfs.base)

    if tfs.ltf == tfs.base:
        ltf = base.copy()
        ltf["n_base"] = 1
        ltf["close_time"] = ltf.index + base_td
    else:
        ltf = rs(tfs.ltf)
    mtf, htf = rs(tfs.mtf), rs(tfs.htf)
    mtf_close = to_ns(pd.DatetimeIndex(mtf["close_time"]))
    htf_close = to_ns(pd.DatetimeIndex(htf["close_time"]))
    return TimeframeStack(base, ltf, mtf, htf, tz, anchor, latest_closed_index(mtf_close, htf_close))
