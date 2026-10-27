import numpy as np
import pandas as pd

from helpers import build, relaxed_cfg, scenario
from lsfvg.core.timeutil import to_ns
from lsfvg.data.synthetic import mtf_candles_to_base, random_walk_bars
from lsfvg.signals.pools import active_pools, equal_level_pools
from lsfvg.signals.swings import confirmed_upto, find_swings, last_confirmed

TZ = "America/New_York"


# ------------------------------------------------------------------ swings
def test_fractal_pivots_and_confirmation_delay():
    h = np.array([1, 2, 3, 5, 3, 2, 1, 2, 3, 4, 3, 2, 1.0])
    sh, sl = find_swings(h, h - 0.5, 2)
    assert sh.idx.tolist() == [3, 9] and sh.confirm_idx.tolist() == [5, 11]
    assert last_confirmed(sh, 4) is None                   # pivot exists at 3 but is unknowable before bar 5
    assert last_confirmed(sh, 5) == 0 and confirmed_upto(sh, 10) == 1 and confirmed_upto(sh, 11) == 2
    assert sl.idx.tolist() == [6]                          # mirror image for lows


def test_flat_top_registers_once_on_its_last_bar_and_last_n_bars_are_unconfirmed():
    h = np.array([1, 2, 4, 4, 2, 1, 0.5, 0.4, 0.3])
    sh, _ = find_swings(h, h - 0.1, 2)
    assert sh.idx.tolist() == [3]
    h2 = np.array([1, 2, 3, 4, 5, 6, 7.0])                 # rising: no pivots, final bars can't be pivots
    assert len(find_swings(h2, h2 - 0.1, 2)[0]) == 0


def test_pivot_requires_strictly_lower_right_side():
    h = np.array([1, 3, 2, 3, 1, 0.5, 0.4])                # bar 1 (3) vs bar 3 (3): right side equal -> not a pivot
    sh, _ = find_swings(h, h - 0.1, 2)
    assert 1 not in sh.idx.tolist() and 3 in sh.idx.tolist()


# ------------------------------------------------------------------ equal highs / lows
def _eq_inputs(highs, lows, atr=0.0006, n=2):
    k = len(highs)
    t = pd.date_range("2026-01-06", periods=k, freq="15min", tz="UTC")
    o, c = to_ns(t), to_ns(t + pd.Timedelta("15min"))
    return dict(high_tf=np.array(highs, float), low_tf=np.array(lows, float), atr_tf=np.full(k, atr),
                open_ns_tf=o, close_ns_tf=c, open_ns_mtf=o, mtf_times=t, tol_atr=0.1, max_sep=100, lookback=8,
                label="mtf", scope="internal", age_bars_mtf=500), n


def test_equal_highs_within_tolerance_known_only_after_second_swing_confirms():
    #           0      1      2       3       4      5       6       7       8       9      10     11
    hi = [1.1000, 1.1010, 1.1050, 1.1020, 1.1010, 1.1015, 1.1020, 1.10503, 1.1030, 1.1020, 1.1010, 1.1000]
    lo = [x - 0.0008 for x in hi]
    kw, n = _eq_inputs(hi, lo)
    sh, _ = find_swings(kw["high_tf"], kw["low_tf"], n)
    assert sh.idx.tolist() == [2, 7]
    pools = equal_level_pools("buy", sh, **kw)
    assert len(pools) == 1
    p = pools[0]
    assert p.kind == "EQH" and abs(p.level - 1.10503) < 1e-12           # level = extreme of the cluster
    assert p.valid_idx == 7 + n + 1                                      # usable from the bar AFTER confirmation
    assert p.known_time == pd.Timestamp(int(kw["close_ns_tf"][7 + n]), unit="ns", tz="UTC")


def test_equal_highs_rejected_when_too_far_apart_or_when_a_higher_bar_sits_between():
    hi = [1.1000, 1.1010, 1.1050, 1.1020, 1.1010, 1.1030, 1.1060, 1.1030, 1.1020, 1.1030, 1.10503, 1.1020, 1.1010, 1.1000]
    lo = [x - 0.0008 for x in hi]
    kw, n = _eq_inputs(hi, lo)
    sh, _ = find_swings(kw["high_tf"], kw["low_tf"], n)
    assert sh.idx.tolist() == [2, 6, 10]
    assert equal_level_pools("buy", sh, **kw) == []                      # 1.1060 between them: not "equal" highs
    hi2 = list(hi)
    hi2[6] = 1.1040                                                       # control: no higher bar in between
    kw2, _ = _eq_inputs(hi2, [x - 0.0008 for x in hi2])
    sh2, _ = find_swings(kw2["high_tf"], kw2["low_tf"], n)
    pools = equal_level_pools("buy", sh2, **kw2)
    assert len(pools) == 1 and pools[0].meta["pivots_tf_idx"] == (2, 10)
    assert equal_level_pools("buy", sh2, **dict(kw2, tol_atr=0.0)) == []  # tolerance 0: 1.1050 vs 1.10503 differ


def test_equal_lows_mirror():
    lo = [1.1000, 1.0990, 1.0950, 1.0980, 1.0990, 1.0985, 1.0980, 1.09497, 1.0970, 1.0980, 1.0990, 1.1000]
    hi = [x + 0.0008 for x in lo]
    kw, n = _eq_inputs(hi, lo)
    _, sl = find_swings(kw["high_tf"], kw["low_tf"], n)
    pools = equal_level_pools("sell", sl, **kw)
    assert len(pools) == 1 and pools[0].kind == "EQL" and abs(pools[0].level - 1.09497) < 1e-12


# ------------------------------------------------------------------ previous day / week / session pools
def test_previous_day_pools_exist_only_after_the_day_ends_and_first_day_is_dropped():
    base, names = scenario()
    cfg = relaxed_cfg({"pools": {"use_pwh_pwl": False, "use_sessions": False, "use_equal_mtf": False,
                                 "use_equal_htf": False}})
    eng, prep, stack = build(cfg, base)
    mtf = stack.mtf
    pdh = [p for p in prep.pools if p.kind == "PDH"]
    assert len(pdh) == 1                                       # day0 dropped (first), day2 not finished, day1 -> 1 pool
    p = pdh[0]
    assert abs(p.level - 1.1050) < 1e-12 and p.known_time == pd.Timestamp("2026-01-06 22:00", tz="UTC")
    assert mtf.index[p.valid_idx] == p.known_time              # first bar of the next day
    assert not p.active_at(p.valid_idx - 1) and p.active_at(p.valid_idx)
    sw = mtf.index.get_loc(names["SW"])
    assert p.swept_idx == sw and p.active_at(sw - 1) and not p.active_at(sw)
    assert p in active_pools(prep.pools, sw - 1) and p not in active_pools(prep.pools, sw)


def test_weekly_pools_match_independent_calendar_computation():
    df = random_walk_bars(n_days=24, seed=11)
    cfg = relaxed_cfg({"pools": {"use_pdh_pdl": False, "use_sessions": False, "use_equal_mtf": False,
                                 "use_equal_htf": False}})
    eng, prep, stack = build(cfg, df)
    mtf = stack.mtf
    shifted = mtf.index.tz_convert(TZ).tz_localize(None) - pd.Timedelta(hours=17)
    week = pd.Series(shifted.to_period("W-SAT").start_time.to_numpy(), index=mtf.index)   # Sunday-based weeks
    wk_high = mtf["high"].groupby(week.to_numpy()).max()
    firsts = {w: np.flatnonzero(week.to_numpy() == w)[0] for w in wk_high.index}
    pwh = sorted((p for p in prep.pools if p.kind == "PWH"), key=lambda p: p.valid_idx)
    assert len(pwh) >= 1
    weeks = list(wk_high.index)
    for p in pwh:
        k = [w for w in weeks if firsts[w] == p.valid_idx]                   # valid from the first bar of NEXT week
        assert len(k) == 1
        prev_week = weeks[weeks.index(k[0]) - 1]
        assert abs(p.level - wk_high[prev_week]) < 1e-12


def test_session_pool_level_and_known_time():
    df = random_walk_bars(n_days=10, seed=5)
    cfg = relaxed_cfg({"pools": {"use_pdh_pdl": False, "use_pwh_pwl": False, "use_equal_mtf": False,
                                 "use_equal_htf": False}})
    eng, prep, stack = build(cfg, df)
    mtf = stack.mtf
    loc = mtf.index.tz_convert(TZ)
    day = pd.Timestamp("2026-01-07").date()
    win = (loc.date == day) & (loc.hour >= 2) & (loc.hour < 8)                # London 02:00-08:00 NY
    want_h, want_l = mtf["high"][win].max(), mtf["low"][win].min()
    ph = [p for p in prep.pools if p.kind == "SESS_H" and p.tf == "session:london" and abs(p.level - want_h) < 1e-12]
    pl = [p for p in prep.pools if p.kind == "SESS_L" and p.tf == "session:london" and abs(p.level - want_l) < 1e-12]
    assert ph and pl
    end_utc = pd.Timestamp("2026-01-07 08:00", tz=TZ).tz_convert("UTC")
    assert ph[0].known_time == end_utc and mtf.index[ph[0].valid_idx] == end_utc
