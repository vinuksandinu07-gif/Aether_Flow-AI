import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import build, relaxed_cfg, scenario
from lsfvg.core.timeutil import NS_MIN
from lsfvg.core.types import ExitFill, TradeRecord
from lsfvg.data.rolls import back_adjust
from lsfvg.data.synthetic import mtf_candles_to_base, random_walk_bars
from lsfvg.signals.entry import resolve_entry_order
from lsfvg.visualization.trade_chart import export_all, plot_setup

BASE, NAMES = scenario()


def test_chart_renders_for_valid_skipped_and_with_trade_exits():
    cfg = relaxed_cfg({"viz": {"max_charts": 5, "dpi": 60}, "targets": {"tp_mode": "partial_runner", "min_rr": 0.0}})
    eng, prep, stack = build(cfg, BASE)
    res = eng.scan(prep)
    valid = res.setups[-1]
    skipped = res.skipped[-1]
    with tempfile.TemporaryDirectory() as d:
        out = resolve_entry_order(eng.entry_order(valid), stack.ltf)
        p1 = plot_setup(valid, prep, Path(d) / "valid.png", viz=cfg.viz, outcome=out)
        p2 = plot_setup(skipped, prep, Path(d) / "skipped.png", viz=cfg.viz)
        fill_t = out.time
        trade = TradeRecord(valid.setup_id, valid.direction, fill_t, out.fill_price, valid.stop_price,
                            valid.targets.final_target,
                            [ExitFill(fill_t + pd.Timedelta("15min"), 1.1015, 0.5, "partial"),
                             ExitFill(fill_t + pd.Timedelta("45min"), valid.entry_price, 0.5, "breakeven")])
        p3 = plot_setup(valid, prep, Path(d) / "trade.png", viz=cfg.viz, outcome=out, trade=trade)
        paths = export_all(res, prep, eng, Path(d) / "trades", viz=cfg.viz)
        assert all(p.exists() and p.stat().st_size > 8000 for p in (p1, p2, p3))
        assert 1 <= len(paths) <= 5 and all(p.suffix == ".png" for p in paths)
        assert any("skipped" in p.name for p in paths) and any("valid" in p.name for p in paths)


def test_back_adjusted_rolls_remove_the_gap_and_log_the_adjustment():
    t = pd.date_range("2026-03-01", periods=12, freq="1h", tz="UTC")
    mk = lambda ts, base: pd.DataFrame({"open": base, "high": base + 1, "low": base - 1, "close": base + 0.5}, index=ts)
    old = mk(t[:8], 100.0 + np.arange(8))              # old contract trades 100..107, last close before roll = 107.5
    new = mk(t[6:], 110.0 + np.arange(6))              # new contract trades +3 higher at the roll (first close 110.5)
    roll = t[8]
    # new contract's first close at/after roll: bar t[8] -> 110 + 2 + .5 = 112.5; old last close before roll (t[7]) 107.5
    cont, log = back_adjust([(old, roll), (new, roll)], "difference")
    assert len(log) == 1 and abs(log["adjustment"].iloc[0] - (112.5 - 107.5)) < 1e-12
    jump = cont["close"].loc[t[8]] - cont["close"].loc[t[7]]
    assert abs(jump) < 1e-12                                                  # the 5.0 roll gap is gone from the series
    assert abs((cont["close"].loc[t[7]] - cont["close"].loc[t[6]]) - 1.0) < 1e-12   # old contract's own moves are preserved
    assert abs(cont["close"].loc[t[7]] - 112.5) < 1e-12                      # old close shifted to meet the new contract
    ratio, rlog = back_adjust([(old, roll), (new, roll)], "ratio")
    assert abs(ratio["close"].loc[t[7]] - 112.5) < 1e-9 and abs(rlog["adjustment"].iloc[0] - 112.5 / 107.5) < 1e-12
    assert cont.index.is_monotonic_increasing and cont.index.is_unique


def test_random_walk_is_reproducible_valid_ohlc_and_only_in_fx_hours():
    a, b = random_walk_bars(n_days=10, seed=9), random_walk_bars(n_days=10, seed=9)
    pd.testing.assert_frame_equal(a, b)
    assert (a["high"] >= a[["open", "close"]].max(axis=1) - 1e-12).all() and (a["low"] <= a[["open", "close"]].min(axis=1) + 1e-12).all()
    loc = a.index.tz_convert("America/New_York")
    wd, hr = np.asarray(loc.weekday), np.asarray(loc.hour)
    assert not (wd == 5).any() and not ((wd == 4) & (hr >= 17)).any() and not ((wd == 6) & (hr < 17)).any()
    assert not a.equals(random_walk_bars(n_days=10, seed=10))


def test_mtf_candles_to_base_roundtrip_for_up_and_down_candles():
    t0 = pd.Timestamp("2026-01-06", tz="UTC")
    candles = [(t0, 1.10, 1.11, 1.09, 1.105), (t0 + pd.Timedelta("15min"), 1.105, 1.12, 1.10, 1.102)]
    base = mtf_candles_to_base(candles)
    assert len(base) == 6 and base.index.is_unique
    for k, (_, o, h, l, c) in enumerate(candles):
        seg = base.iloc[3 * k:3 * k + 3]
        assert (seg["open"].iloc[0], seg["high"].max(), seg["low"].min(), seg["close"].iloc[-1]) == (o, h, l, c)
