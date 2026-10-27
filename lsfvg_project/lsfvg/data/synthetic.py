"""Synthetic market data for tests and the offline demo (NOT for judging the strategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fx_open_mask(idx: pd.DatetimeIndex, tz: str = "America/New_York") -> np.ndarray:
    """FX is open Sun 17:00 NY -> Fri 17:00 NY."""
    loc = idx.tz_convert(tz)
    wd, hr = loc.weekday, loc.hour
    closed = (wd == 5) | ((wd == 4) & (hr >= 17)) | ((wd == 6) & (hr < 17))
    return ~np.asarray(closed)


def random_walk_bars(
    start: str = "2026-01-04 22:00",     # Sunday 17:00 New York (EST)
    n_days: int = 40,
    base: str = "5min",
    seed: int = 7,
    price0: float = 1.1000,
    vol: float = 0.00012,                # std of a base-bar close-to-close move (price units)
    decimals: int = 5,
) -> pd.DataFrame:
    """Seeded random walk with session-dependent volatility, FX market hours, 4 intrabar sub-steps."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=int(n_days * 86400 / pd.Timedelta(base).total_seconds()), freq=base)
    idx = idx[_fx_open_mask(idx)]
    n = len(idx)
    hr = idx.tz_convert("America/New_York").hour
    mult = np.where((hr >= 2) & (hr < 11), 1.6, np.where((hr >= 19) | (hr < 2), 0.8, 1.0))
    slow = np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.3)          # slowly varying vol regime
    slow /= slow.mean()
    steps = rng.normal(0, 1, (n, 4)) * (vol * mult * slow / 2.0)[:, None]
    # gentle mean reversion so the series keeps making swings instead of drifting away
    path = np.empty((n, 4))
    px = price0
    for i in range(n):
        o = px
        cum = o + np.cumsum(steps[i] - 0.002 * (o - price0) / 4.0)
        path[i] = cum
        px = cum[-1]
    opens = np.r_[price0, path[:-1, -1]]
    highs = np.maximum(path.max(axis=1), opens)
    lows = np.minimum(path.min(axis=1), opens)
    closes = path[:, -1]
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                       "volume": rng.integers(50, 500, n).astype(float)},
                      index=pd.DatetimeIndex(idx, name="time").as_unit("ns"))
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].round(decimals)
    df.attrs["spread_modeled"] = False
    return df


def mtf_candles_to_base(
    candles: list[tuple[pd.Timestamp, float, float, float, float]], ratio: int = 3, base: str = "5min"
) -> pd.DataFrame:
    """Expand scripted MTF candles ``(open_time, o, h, l, c)`` into ``ratio`` base bars each.

    The base path is o -> (l|h) -> (h|l) -> c so that re-aggregating the base bars reproduces
    EXACTLY the scripted MTF OHLC (used to build deterministic test scenarios).
    """
    step = pd.Timedelta(base)
    rows, times = [], []
    for t0, o, h, l, c in candles:
        p = [o, l, h, c] if c >= o else [o, h, l, c]
        pts = np.interp(np.linspace(0, 3, ratio + 1), [0, 1, 2, 3], p) if ratio != 3 else np.array(p)
        for k in range(ratio):
            a, b = pts[k], pts[k + 1]
            rows.append((a, max(a, b), min(a, b), b))
            times.append(t0 + k * step)
        # make sure the extremes are present in the bars (interp may skip them when ratio != 3)
        if ratio != 3:
            rows[-ratio] = (rows[-ratio][0], max(rows[-ratio][1], h), min(rows[-ratio][2], l), rows[-ratio][3])
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                      index=pd.DatetimeIndex(times, name="time").as_unit("ns"))
    df["volume"] = 100.0
    df.attrs["spread_modeled"] = False
    return df
