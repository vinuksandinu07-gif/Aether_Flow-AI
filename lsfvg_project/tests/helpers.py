"""Shared test helpers (no pytest dependency, so the suite also runs with tests/run_tests.py)."""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

from lsfvg.config import load_config
from lsfvg.data.resample import build_stack
from lsfvg.data.spread import SpreadModel, add_modeled_bid_ask
from lsfvg.data.synthetic import mtf_candles_to_base
from lsfvg.signals.engine import SignalEngine

CONFIG_PATH = "config/default.yaml"
STEP = pd.Timedelta("15min")
START = pd.Timestamp("2026-01-04 22:00", tz="UTC")      # Sunday 17:00 New York


@contextlib.contextmanager
def raises(exc_type, match: str | None = None):
    try:
        yield
    except exc_type as e:                                # noqa: PERF203
        if match is not None and match not in str(e):
            raise AssertionError(f"expected message containing {match!r}, got {e!r}")
        return
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def relaxed_cfg(extra: dict | None = None):
    """Default config with the gates that need long history / scripted clock switched off."""
    ov = {"bias": {"enabled": False}, "filters": {"volatility": {"enabled": False}},
          "stops": {"max_stop_atr": 10.0}, "targets": {"tp_mode": "fixed_rr", "min_rr": 1.0, "fixed_rr": 2.0}}
    if extra:
        from lsfvg.config.loader import _deep_merge
        ov = _deep_merge(ov, extra)
    return load_config(CONFIG_PATH, ov)


def scenario(seed: int = 3):
    """Three trading days of bounded noise with a scripted SHORT setup on day 3 (London window).

    Day 1 has a spike to 1.1050 inside the NY session  -> PDH == SESS_H(new_york) == 1.1050.
    Scripted candles (15m, NY time): a1 02:00 .. u4 04:45, SW 05:00 (sweeps 1.1050, closes 1.1048),
    D 05:15 (bearish displacement closing below swing low 1.1022), n1 05:30 (completes the bearish FVG
    [1.1026, 1.1045]); r1/r2 retrace into the gap (r2 fills a short limit at 1.1026).
    Returns ``(base_df, names)`` where ``names`` maps candle name -> MTF open time.
    """
    rng = np.random.default_rng(seed)
    candles, names = [], {}
    state = {"t": START, "p": 1.1010}

    def add(o, h, l, c, name=None):
        candles.append((state["t"], round(o, 5), round(h, 5), round(l, 5), round(c, 5)))
        if name:
            names[name] = state["t"]
        state["t"] += STEP
        state["p"] = round(c, 5)

    def filler(n, lo=1.0995, hi=1.1030):
        for _ in range(n):
            o = state["p"]
            c = float(np.clip(o + rng.normal(0, 0.0003), lo, hi))
            add(o, max(o, c) + rng.uniform(0.00005, 0.0002), min(o, c) - rng.uniform(0.00005, 0.0002), c)

    filler(96)                                    # day 0 (first group is dropped by the pool builder)
    filler(68)                                    # day 1 up to 10:00 New York
    o = state["p"]
    add(o, 1.1050, o - 0.0001, o + 0.0001, "spike")   # the day-1 high = PDH
    filler(27)                                    # rest of day 1
    filler(28)                                    # day 2 until 01:00 New York
    p0 = state["p"]
    for k in range(1, 9):                         # gentle ramp up to 1.1030 (candles 28..35)
        c = p0 + (1.1030 - p0) * k / 8
        o = state["p"]
        add(o, max(o, c) + 0.0001, min(o, c) - 0.0001, c)
    script = [
        ("a1", 1.1030, 1.1034, 1.1028, 1.1032), ("a2", 1.1032, 1.1035, 1.1029, 1.1031),
        ("a3", 1.1031, 1.1033, 1.1027, 1.1028), ("P", 1.1028, 1.1030, 1.1022, 1.1027),
        ("u1", 1.1027, 1.1034, 1.1025, 1.1033), ("u2", 1.1033, 1.1040, 1.1031, 1.1039),
        ("u3", 1.1039, 1.1046, 1.1037, 1.1045), ("u4", 1.1045, 1.1049, 1.1043, 1.1047),
        ("SW", 1.1047, 1.1056, 1.1045, 1.1048), ("D", 1.1048, 1.1048, 1.1016, 1.1019),
        ("n1", 1.1019, 1.1026, 1.1012, 1.1014), ("r1", 1.1014, 1.1022, 1.1010, 1.1020),
        ("r2", 1.1020, 1.1030, 1.1018, 1.1028), ("r3", 1.1028, 1.1029, 1.1000, 1.1002),
        ("r4", 1.1002, 1.1005, 1.0980, 1.0984), ("r5", 1.0984, 1.0990, 1.0975, 1.0980),
    ]
    for nm, *ohlc in script:
        add(*ohlc, name=nm)
    for _ in range(8):                            # quiet tail
        o = state["p"]
        add(o, o + 0.0003, o - 0.0003, o - 0.0001)
    base = mtf_candles_to_base(candles)
    return base, names


def build(cfg, base, with_quotes: bool = False, name: str = "EUR_USD"):
    ins = cfg.instruments[name]
    df = add_modeled_bid_ask(base, SpreadModel(ins)) if with_quotes else base
    stack = build_stack(df, cfg.timeframes, ins)
    eng = SignalEngine(cfg, name)
    prep = eng.prepare(stack)
    return eng, prep, stack
