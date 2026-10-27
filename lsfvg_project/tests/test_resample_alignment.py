import numpy as np
import pandas as pd

from helpers import raises
from lsfvg.core.timeutil import to_ns
from lsfvg.data.resample import latest_closed_index, resample_ohlc
from lsfvg.data.synthetic import mtf_candles_to_base, random_walk_bars

TZ = "America/New_York"


def test_resample_reproduces_scripted_candles_exactly():
    t0 = pd.Timestamp("2026-01-06 00:00", tz="UTC")
    candles = [(t0 + i * pd.Timedelta("15min"), 1.10 + i * 1e-4, 1.10 + i * 1e-4 + 5e-4, 1.10 + i * 1e-4 - 4e-4,
                1.10 + i * 1e-4 + (2e-4 if i % 2 else -2e-4)) for i in range(40)]
    base = mtf_candles_to_base(candles)
    out = resample_ohlc(base, "15m", tz=TZ, anchor="17:00", base_tf="5m")
    want = pd.DataFrame(candles, columns=["t", "open", "high", "low", "close"]).set_index("t")
    assert len(out) == 40
    for c in ("open", "high", "low", "close"):
        assert np.allclose(out[c].to_numpy(), want[c].to_numpy(), atol=1e-12), c
    assert (out["close_time"] - out.index == pd.Timedelta("15min")).all() and (out["n_base"] == 3).all()


def test_trailing_incomplete_bin_is_dropped_and_first_partial_bin_too():
    base = random_walk_bars(n_days=3)
    cut = base.iloc[: len(base) - 7]                                        # ends mid-bin
    out = resample_ohlc(cut, "4h", tz=TZ, anchor="17:00", base_tf="5m")
    assert out["close_time"].iloc[-1] <= cut.index[-1] + pd.Timedelta("5min")
    shifted = base.iloc[10:]                                                # starts inside a bin
    out2 = resample_ohlc(shifted, "4h", tz=TZ, anchor="17:00", base_tf="5m")
    assert out2.index[0] >= shifted.index[0]


def test_htf_bar_is_available_only_after_it_closes():
    base = random_walk_bars(n_days=4)
    m15 = resample_ohlc(base, "15m", tz=TZ, anchor="17:00", base_tf="5m")
    h4 = resample_ohlc(base, "4h", tz=TZ, anchor="17:00", base_tf="5m")
    ai = latest_closed_index(to_ns(pd.DatetimeIndex(m15["close_time"])), to_ns(pd.DatetimeIndex(h4["close_time"])))
    for k in range(0, len(m15), 7):
        if ai[k] >= 0:
            assert h4["close_time"].iloc[ai[k]] <= m15["close_time"].iloc[k]            # closed by then
            if ai[k] + 1 < len(h4):
                assert h4["close_time"].iloc[ai[k] + 1] > m15["close_time"].iloc[k]    # next one still forming


def test_bid_ask_columns_are_aggregated_and_duplicates_refused():
    from lsfvg.config import load_config
    from lsfvg.data.spread import SpreadModel, add_modeled_bid_ask
    cfg = load_config("config/default.yaml")
    base = add_modeled_bid_ask(random_walk_bars(n_days=2), SpreadModel(cfg.instruments["EUR_USD"]))
    out = resample_ohlc(base, "15m", tz=TZ, anchor="17:00", base_tf="5m")
    assert {"bid_high", "ask_low", "bid_close", "ask_open"} <= set(out.columns)
    assert ((out["ask_close"] - out["bid_close"]) > 0).all()
    dup = pd.concat([base.iloc[:5], base.iloc[:1], base.iloc[5:]]).sort_index(kind="stable")
    with raises(ValueError, "duplicate"):
        resample_ohlc(dup, "15m", tz=TZ, anchor="17:00", base_tf="5m")
