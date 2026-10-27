import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import raises
from lsfvg.config import load_config
from lsfvg.core.timeutil import local_naive_ns, NS_DAY, NS_MIN
from lsfvg.data.quality import DataQualityError, drop_flagged, enforce, run_quality_report
from lsfvg.data.spread import SpreadModel, add_modeled_bid_ask
from lsfvg.data.synthetic import random_walk_bars

CFG = load_config("config/default.yaml")
INS = CFG.instruments["EUR_USD"]


def _clean(days=6, start="2026-01-04 22:00"):
    return add_modeled_bid_ask(random_walk_bars(start=start, n_days=days), SpreadModel(INS))


def _rep(df):
    return run_quality_report(df, instrument_name="EUR_USD", ins=INS, base_tf="5m", qcfg=CFG.data.quality)


def _kinds(rep, sev=None):
    return {i.kind: i.count for i in rep.issues if sev is None or i.severity == sev}


def test_clean_data_passes_and_reports_weekend_gaps_as_info():
    rep = _rep(_clean(10))
    assert rep.ok and not rep.errors
    assert _kinds(rep, "info").get("weekend_gaps", 0) >= 1
    enforce(rep, CFG.data.quality)                                  # does not raise


def test_duplicates_ohlc_crossed_nan_are_errors_and_data_is_not_modified():
    df = _clean()
    snapshot = df.copy()
    dup = pd.concat([df, df.iloc[[100]]]).sort_index(kind="stable")
    bad = dup.copy()
    i1, i2, i3 = 10, 200, 400
    bad.iloc[i1, bad.columns.get_loc("high")] = bad.iloc[i1]["low"] - 0.001          # high < low
    bad.iloc[i2, bad.columns.get_loc("bid_close")] = bad.iloc[i2]["ask_close"] + 0.0005   # crossed
    bad.iloc[i3, bad.columns.get_loc("close")] = np.nan
    before = bad.copy()
    rep = _rep(bad)
    k = _kinds(rep, "error")
    assert k["duplicate_timestamps"] == 1 and k["ohlc_inconsistent"] >= 1
    assert k["crossed_quotes"] == 1 and k["nan_prices"] >= 1
    assert not rep.ok
    pd.testing.assert_frame_equal(bad, before)                       # detect + report, never fix
    pd.testing.assert_frame_equal(df, snapshot)
    with raises(DataQualityError):
        enforce(rep, CFG.data.quality)


def test_drop_flagged_is_explicit_and_keeps_first_duplicate():
    df = _clean()
    dup = pd.concat([df, df.iloc[[100]]]).sort_index(kind="stable")
    rep = _rep(dup)
    out = drop_flagged(dup, rep, ["duplicate"])
    assert len(out) == len(df) and out.index.is_unique
    assert len(dup) == len(df) + 1                                   # original untouched


def test_gap_classification_missing_long_and_daily_break():
    df = _clean(8)
    loc = df.index.tz_convert("America/New_York")
    wd = np.asarray(loc.weekday)
    hr, mn = np.asarray(loc.hour), np.asarray(loc.minute)
    tue = wd == 1
    miss = tue & (hr == 10) & (mn < 25)                              # 5 bars missing mid-morning
    long_gap = (wd == 2) & (hr >= 9) & (hr < 14)                     # 5 hours missing (holiday/outage)
    brk = (wd == 3) & (hr == 17)                                     # 60 min at the 17:00 NY anchor
    rep = _rep(df[~(miss | long_gap | brk)])
    kinds = _kinds(rep)
    assert kinds["missing_bars"] == 5
    assert kinds["long_weekday_gaps"] == 1
    assert kinds["daily_break_gaps"] == 1


def test_bad_tick_spike_is_flagged_not_removed():
    df = _clean(6)
    k = 700
    spike = df.copy()
    spike.iloc[k, spike.columns.get_loc("close")] += 0.02            # 200 pips, reverts next bar
    spike.iloc[k, spike.columns.get_loc("high")] = spike.iloc[k]["close"]
    rep = _rep(spike)
    assert _kinds(rep).get("bad_tick_spikes", 0) >= 1
    assert spike.index[k] in rep.flags.index and len(spike) == len(df)


def test_dst_transitions_are_listed_and_report_saves_files():
    df = _clean(14, start="2026-03-01 22:00")                        # spans 2026-03-08
    rep = _rep(df)
    assert _kinds(rep, "info")["dst_transitions"] == 1
    with tempfile.TemporaryDirectory() as d:
        pj, pm = rep.save(d)
        assert pj.exists() and "Data quality" in pm.read_text() and "dst_transitions" in pj.read_text()


def test_modeled_spread_is_declared_in_the_report():
    rep = _rep(_clean(3))
    assert rep.spread_modeled and "MODELED" in rep.to_markdown()
