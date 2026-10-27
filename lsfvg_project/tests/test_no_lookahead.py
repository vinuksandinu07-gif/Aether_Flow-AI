"""Lookahead / repainting tests.

1. TRUNCATION INVARIANCE: cut the data at arbitrary bars; every setup the truncated run emits must be identical
   to the one in the full run, and the full run must not contain extra setups signalled before the cut.
2. FUTURE MUTATION: wreck every price after the cut; setups signalled before the cut must not change.
3. CAUSAL ORDERING: every component of every setup precedes (or equals) its signal bar; HTF bars used are closed.
4. NO REPAINT: re-scanning with end_idx = signal bar reproduces the setup exactly.
"""
import numpy as np
import pandas as pd

from helpers import build, relaxed_cfg
from lsfvg.data.spread import SpreadModel, add_modeled_bid_ask
from lsfvg.data.synthetic import random_walk_bars

OVERRIDES = {
    "bias": {"enabled": True}, "filters": {"session_enabled": False, "volatility": {"enabled": True}},
    "signals": {"disp_mult": 0.8, "min_sweep_atr": 0.02, "fvg_min_atr": 0.05},
    "stops": {"max_stop_atr": 6.0}, "targets": {"tp_mode": "opposing_liquidity", "min_rr": 0.5},
}


def _data(days=150, seed=21):
    cfg = relaxed_cfg(OVERRIDES)
    df = add_modeled_bid_ask(random_walk_bars(n_days=days, seed=seed), SpreadModel(cfg.instruments["EUR_USD"]))
    return cfg, df


def _sig(s):
    tgt = None if s.targets.reference_target is None else round(s.targets.reference_target, 10)
    return (s.setup_id, s.status, tuple(s.reasons), s.direction, s.pool_id, s.signal_idx, round(s.entry_price, 10),
            round(s.stop_price, 10), tgt, s.bias_zone, s.htf_aligned, s.sweep, s.mss, s.fvg)


def _upto(res, t):
    return {_sig(s) for s in res.all_setups if s.signal_time <= t}


def test_full_run_is_rich_enough_to_make_the_leak_tests_meaningful():
    cfg, df = _data()
    eng, prep, _ = build(cfg, df)
    res = eng.scan(prep)
    assert len(res.setups) >= 10 and len(res.skipped) >= 15, (len(res.setups), len(res.skipped))
    assert {s.direction for s in res.all_setups} == {"long", "short"}
    assert any("bias_wrong_zone" in s.reasons for s in res.skipped)           # HTF bias really is exercised


def test_truncation_invariance_at_many_cut_points():
    cfg, df = _data()
    eng, prep, _ = build(cfg, df)
    full = eng.scan(prep)
    compared = 0
    for frac in (0.31, 0.43, 0.52, 0.66, 0.79, 0.93):
        cut = int(len(df) * frac) + 1                                         # arbitrary base bar, often mid-bin
        eng_t, prep_t, stack_t = build(cfg, df.iloc[:cut])
        part = eng_t.scan(prep_t)
        last_close = stack_t.mtf["close_time"].iloc[-1]
        a, b = _upto(full, last_close), {_sig(s) for s in part.all_setups}
        assert a == b, (frac, len(a), len(b), sorted(a ^ b, key=lambda t: t[5])[:2])
        compared += len(b)
    assert compared >= 60


def test_destroying_the_future_does_not_change_the_past():
    cfg, df = _data(days=100, seed=33)
    cut = int(len(df) * 0.6)
    eng_t, prep_t, stack_t = build(cfg, df.iloc[:cut])
    base_res = eng_t.scan(prep_t)
    last_close = stack_t.mtf["close_time"].iloc[-1]
    wrecked = df.copy()
    rng = np.random.default_rng(1)
    cols = [c for c in df.columns if c != "volume"]
    wrecked.iloc[cut:, [df.columns.get_loc(c) for c in cols]] += 0.05 + rng.normal(0, 0.01, (len(df) - cut, 1))
    eng_w, prep_w, _ = build(cfg, wrecked)
    wrecked_res = eng_w.scan(prep_w)
    assert _upto(wrecked_res, last_close) == {_sig(s) for s in base_res.all_setups} and len(base_res.all_setups) > 5


def test_every_component_precedes_the_signal_and_htf_bars_are_closed():
    cfg, df = _data()
    eng, prep, stack = build(cfg, df)
    res = eng.scan(prep)
    pools = {p.pool_id: p for p in prep.pools}
    n_mtf, n_htf = cfg.signals.mtf_swing_n, cfg.signals.htf_swing_n
    htf_close, mtf_close = stack.htf["close_time"], stack.mtf["close_time"]
    for s in res.all_setups:
        p = pools[s.pool_id]
        assert p.valid_idx <= s.sweep.breach_idx and p.known_time <= stack.mtf.index[s.sweep.breach_idx]
        assert s.sweep.breach_idx <= s.sweep.reclaim_idx <= s.mss.idx <= s.signal_idx
        assert s.sweep.extreme_idx <= s.mss.idx and s.fvg.mid_idx >= s.sweep.extreme_idx
        assert s.fvg.last_idx <= s.signal_idx and s.mss.idx <= s.signal_idx <= s.mss.idx + 1
        assert s.mss.swing_idx < s.sweep.extreme_idx and s.mss.swing_idx + n_mtf <= s.mss.idx      # swing confirmed first
        assert s.signal_time == mtf_close.iloc[s.signal_idx]
        h = prep.htf_idx[s.signal_idx]
        if h >= 0:
            assert htf_close.iloc[h] <= s.signal_time                                              # closed HTF bar only
            if h + 1 < len(htf_close):
                assert htf_close.iloc[h + 1] > s.signal_time
        if s.meta["dealing_range"] is not None:
            assert h >= 0


def test_setups_do_not_repaint_when_more_data_arrives():
    cfg, df = _data(days=100, seed=5)
    eng, prep, _ = build(cfg, df)
    full = eng.scan(prep)
    pick = full.all_setups[:: max(1, len(full.all_setups) // 12)]
    assert len(pick) >= 6
    for s in pick:
        live = eng.scan(prep, end_idx=s.signal_idx)                       # what a live loop would have seen at that bar
        got = [x for x in live.all_setups if x.setup_id == s.setup_id]
        assert len(got) == 1 and _sig(got[0]) == _sig(s)


def test_truncating_exactly_at_each_signal_bar_reproduces_the_setup():
    """The sharpest leak probe: rebuild EVERYTHING from data that ends at the signal bar's close."""
    cfg, df = _data(days=100, seed=33)
    eng, prep, stack = build(cfg, df)
    full = eng.scan(prep)
    allset = full.all_setups
    step = max(1, len(allset) // 25)
    probed = 0
    for s in allset[::step]:
        cut = int(np.searchsorted(df.index.values, np.datetime64(s.signal_time.tz_convert("UTC").tz_localize(None)), "left"))
        eng_t, prep_t, stack_t = build(cfg, df.iloc[:cut])
        assert stack_t.mtf["close_time"].iloc[-1] == s.signal_time          # data ends exactly at the signal close
        got = [x for x in eng_t.scan(prep_t).all_setups if x.setup_id == s.setup_id]
        assert len(got) == 1 and _sig(got[0]) == _sig(s), (s.setup_id, s.status, s.reasons)
        probed += 1
    assert probed >= 15


def test_period_pools_are_known_only_after_their_period_ended():
    cfg, df = _data(days=100, seed=33)
    eng, prep, stack = build(cfg, df)
    n_checked = 0
    for p in prep.pools:
        pb = p.meta.get("period_bars")
        if pb is not None:
            assert p.valid_idx >= pb[1] and p.known_time >= stack.mtf["close_time"].iloc[pb[1] - 1]
            n_checked += 1
    assert n_checked > 100
