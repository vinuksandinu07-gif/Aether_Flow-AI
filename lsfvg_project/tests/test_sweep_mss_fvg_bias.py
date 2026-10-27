import numpy as np

from helpers import raises
from lsfvg.core.types import FVG, Pool, Sweep
from lsfvg.signals import bias as B
from lsfvg.signals.fvg import entry_price, find_fvgs, fvg_at, is_mitigated, select_fvg
from lsfvg.signals.structure import find_mss
from lsfvg.signals.swings import SwingSet, find_swings
from lsfvg.signals.sweep import detect_sweep
import pandas as pd

TS = pd.Timestamp("2026-01-06", tz="UTC")


def _pool(side, level, swept, valid=0):
    return Pool("P", "PDH" if side == "buy" else "PDL", side, level, "D", "external", TS, TS, valid, 10**6, swept)


def _arr(h, l, c, atr=0.0004):
    h, l, c = (np.array(x, float) for x in (h, l, c))
    return h, l, c, np.full(len(c), atr)


# ------------------------------------------------------------------ sweep
def test_sweep_buy_side_same_bar_reclaim_gives_short():
    h, l, c, a = _arr([1.0990, 1.0995, 1.1004, 1.0990], [1.0980] * 4, [1.0988, 1.0990, 1.0998, 1.0985])
    sw = detect_sweep(h, l, c, a, _pool("buy", 1.1000, 2), min_sweep_atr=0.5, reclaim_bars=3, end_idx=3)
    assert sw and sw.direction == "short" and sw.breach_idx == sw.reclaim_idx == sw.extreme_idx == 2
    assert abs(sw.depth - 0.0004) < 1e-12


def test_sweep_needs_close_back_inside_within_k_bars_and_min_depth():
    # breach bar closes above the pool; reclaim comes on the 2nd bar after
    h, l, c, a = _arr([1.0990, 1.1006, 1.1010, 1.1008, 1.0995], [1.0980, 1.0990, 1.0995, 1.0990, 1.0985],
                      [1.0988, 1.1003, 1.1004, 1.1001, 1.0996])
    p = _pool("buy", 1.1000, 1)
    ok = detect_sweep(h, l, c, a, p, min_sweep_atr=0.5, reclaim_bars=3, end_idx=4)
    assert ok and ok.reclaim_idx == 4 and ok.extreme_idx == 2 and abs(ok.extreme_price - 1.1010) < 1e-12
    assert detect_sweep(h, l, c, a, p, min_sweep_atr=0.5, reclaim_bars=2, end_idx=4) is None     # too late
    assert detect_sweep(h, l, c, a, p, min_sweep_atr=5.0, reclaim_bars=3, end_idx=4) is None     # too shallow


def test_sweep_never_reads_past_end_idx_and_sell_side_is_long():
    h, l, c, a = _arr([1.0990, 1.1006, 1.1010, 1.1008, 1.0995], [1.0980, 1.0990, 1.0995, 1.0990, 1.0985],
                      [1.0988, 1.1003, 1.1004, 1.1001, 1.0996])
    assert detect_sweep(h, l, c, a, _pool("buy", 1.1000, 1), min_sweep_atr=0.5, reclaim_bars=3, end_idx=3) is None
    hs, ls, cs, a2 = _arr([1.1020] * 4, [1.1010, 1.0994, 1.1002, 1.1008], [1.1015, 1.1001, 1.1004, 1.1012])
    sw = detect_sweep(hs, ls, cs, a2, _pool("sell", 1.1000, 1), min_sweep_atr=0.5, reclaim_bars=2, end_idx=3)
    assert sw and sw.direction == "long" and abs(sw.extreme_price - 1.0994) < 1e-12


# ------------------------------------------------------------------ MSS
def _mss_case(close_mss=1.0996, open_mss=1.1008, close_after=1.0994):
    #              0       1       2       3       4       5       6       7       8       9       10      11      12
    high = [1.1012, 1.1010, 1.1008, 1.1006, 1.1006, 1.1009, 1.1013, 1.1018, 1.1022, 1.1030, 1.1015, 1.1010, 1.0998]
    low = [1.1008, 1.1006, 1.1004, 1.1002, 1.1000, 1.1003, 1.1007, 1.1012, 1.1016, 1.1020, 1.1005, 1.0994, 1.0990]
    close = [1.1010, 1.1007, 1.1005, 1.1003, 1.1002, 1.1008, 1.1012, 1.1017, 1.1021, 1.1022, 1.1008, close_mss, close_after]
    open_ = [1.1011, 1.1010, 1.1007, 1.1005, 1.1003, 1.1002, 1.1008, 1.1012, 1.1017, 1.1025, 1.1022, open_mss, close_mss]
    o, h, l, c = (np.array(x, float) for x in (open_, high, low, close))
    sh, sl = find_swings(h, l, 2)
    sweep = Sweep("P", "short", 1.1025, 9, 9, 9, 1.1030, 0.0005)
    return o, h, l, c, np.full(13, 0.0006), sweep, sl


def test_mss_bearish_displacement_closing_below_last_opposing_swing():
    o, h, l, c, a, sweep, sl = _mss_case()
    assert sl.idx.tolist() == [4] and sl.confirm_idx.tolist() == [6]
    mss, why = find_mss(o, h, l, c, a, sweep, sl, disp_mult=1.0, max_bars=6, end_idx=12)
    assert why == "" and mss.idx == 11 and abs(mss.level - 1.1000) < 1e-12 and mss.swing_idx == 4
    assert abs(mss.body - 0.0012) < 1e-12 and mss.body_atr >= 1.0


def test_mss_rejections_small_body_pending_invalidated_and_no_swing():
    o, h, l, c, a, sweep, sl = _mss_case(close_mss=1.0999, open_mss=1.1001)           # body 0.0002 < ATR
    assert find_mss(o, h, l, c, a, sweep, sl, disp_mult=1.0, max_bars=3, end_idx=12) == (None, "no_displacement")
    o, h, l, c, a, sweep, sl = _mss_case()
    assert find_mss(o, h, l, c, a, sweep, sl, disp_mult=1.0, max_bars=6, end_idx=10) == (None, "pending")
    c2 = c.copy()
    c2[10] = 1.1027                                                                   # closes back above the pool level
    assert find_mss(o, h, l, c2, a, sweep, sl, disp_mult=1.0, max_bars=6, end_idx=12) == (None, "sweep_invalidated")
    empty = SwingSet(np.empty(0, int), np.empty(0), np.empty(0, int), 2)
    assert find_mss(o, h, l, c, a, sweep, empty, disp_mult=1.0, max_bars=2, end_idx=12) == (None, "no_opposing_swing")


def test_mss_cannot_use_a_swing_before_it_is_confirmed():
    o, h, l, c, a, sweep, sl = _mss_case()
    early = Sweep("P", "short", 1.1025, 5, 5, 5, 1.1030, 0.0005)       # swing pivot 4 is confirmed only at bar 6
    o2, c2 = o.copy(), c.copy()
    o2[5], c2[5] = 1.1010, 1.0990                                      # huge bearish candle at bar 5 (< confirm bar 6)
    mss, why = find_mss(o2, h, l, c2, a, early, sl, disp_mult=1.0, max_bars=0, end_idx=12)
    assert mss is None


# ------------------------------------------------------------------ FVG
def test_fvg_definition_bull_bear_and_none():
    h = np.array([1.1000, 1.1010, 1.1030, 1.1040])
    l = np.array([1.0990, 1.1000, 1.1015, 1.1025])
    f = fvg_at(h, l, 3)                                    # bull: high[1]=1.1010 < low[3]=1.1025
    assert f.direction == "bull" and (f.bottom, f.top) == (1.1010, 1.1025) and (f.first_idx, f.mid_idx, f.last_idx) == (1, 2, 3)
    h2 = np.array([1.1040, 1.1030, 1.1010, 1.1000])
    l2 = np.array([1.1025, 1.1015, 1.1000, 1.0990])
    g = fvg_at(h2, l2, 3)                                  # bear: low[1]=1.1015 > high[3]=1.1000
    assert g.direction == "bear" and (g.bottom, g.top) == (1.1000, 1.1015)
    assert fvg_at(np.array([1.10, 1.11, 1.105]), np.array([1.09, 1.10, 1.095]), 2) is None
    assert fvg_at(h, l, 1) is None


def test_find_fvgs_min_size_direction_and_known_by_end_idx():
    h = np.array([1.1000, 1.1010, 1.1030, 1.1040, 1.1042])
    l = np.array([1.0990, 1.1000, 1.1015, 1.1025, 1.1035])
    atr = np.full(5, 0.0006)
    kw = dict(lo_mid=1, hi_mid=3, end_idx=4)
    assert len(find_fvgs(h, l, atr, "long", min_atr=0.1, **kw)) == 3          # gaps completed at bars 2, 3 and 4
    assert find_fvgs(h, l, atr, "short", min_atr=0.1, **kw) == []              # wrong direction
    assert len(find_fvgs(h, l, atr, "long", min_atr=2.0, **kw)) == 2          # gaps of 0.0015 pass (>= 2 x 0.0006), the 0.0005 gap does not
    assert len(find_fvgs(h, l, atr, "long", min_atr=3.0, **kw)) == 0          # every gap < 3 x 0.0006
    assert len(find_fvgs(h, l, atr, "long", min_atr=0.1, lo_mid=1, hi_mid=3, end_idx=3)) == 2   # bar-4 gap not known yet


def test_entry_modes_mitigation_and_selection():
    bear = FVG("bear", 0, 1, 2, 1.1000, 1.1015, 1.1040, 1.1025)
    bull = FVG("bull", 0, 1, 2, 1.1010, 1.1025, 1.1010, 1.0990)
    assert entry_price(bear, "short", "edge") == 1.1000 and abs(entry_price(bear, "short", "midpoint") - 1.10075) < 1e-12
    assert entry_price(bull, "long", "edge") == 1.1025 and abs(entry_price(bull, "long", "midpoint") - 1.10175) < 1e-12
    h = np.array([0, 0, 0, 1.0995, 1.1001, 1.1005])
    l = np.array([0, 0, 0, 1.0985, 1.0990, 1.0995])
    c = np.array([0, 0, 0, 1.0990, 1.0996, 1.1000])
    assert not is_mitigated(bear, "short", 1.1000, h, l, c, 3)
    assert is_mitigated(bear, "short", 1.1000, h, l, c, 4)                      # high 1.1001 traded the entry
    c2 = np.array([0, 0, 0, 1.0990, 1.1020, 1.1000])
    h2 = np.array([0, 0, 0, 1.0995, 1.0999, 1.1005])
    assert is_mitigated(bear, "short", 1.1000, h2, l, c2, 4)                     # closed fully through the gap
    a, b = FVG("bull", 0, 1, 2, 1.0, 1.001, 0, 0), FVG("bull", 5, 6, 7, 1.0, 1.003, 0, 0)
    assert select_fvg([a, b], "largest") is b and select_fvg([a, b], "first") is a and select_fvg([a, b], "latest") is b
    assert select_fvg([], "largest") is None


# ------------------------------------------------------------------ bias
def _swings():
    hi = SwingSet(np.array([5, 15]), np.array([1.2, 1.3]), np.array([7, 17]), 2)
    lo = SwingSet(np.array([8, 18]), np.array([1.0, 1.1]), np.array([10, 20]), 2)
    return hi, lo


def test_dealing_range_uses_only_confirmed_swings_and_premium_discount():
    hi, lo = _swings()
    assert B.dealing_range(hi, lo, 9) is None                          # no swing low confirmed yet (confirm 10)
    r = B.dealing_range(hi, lo, 19)
    assert (r.high, r.low, r.eq) == (1.3, 1.0, 1.15)                    # swing low @18 not confirmed until bar 20
    r2 = B.dealing_range(hi, lo, 20)
    assert (r2.high, r2.low) == (1.3, 1.1) and abs(r2.eq - 1.2) < 1e-12
    assert B.classify_price(r2, 1.25) == "premium" and B.classify_price(r2, 1.15) == "discount"
    assert B.classify_price(r2, 1.2) == "equilibrium"
    assert B.zone_allows("long", "discount") and not B.zone_allows("long", "premium")
    assert B.zone_allows("short", "premium") and not B.zone_allows("short", "discount")
    assert not B.zone_allows("long", "equilibrium") and not B.zone_allows("short", "equilibrium")
    degenerate = SwingSet(np.array([5]), np.array([1.0]), np.array([7]), 2)
    assert B.dealing_range(degenerate, SwingSet(np.array([8]), np.array([1.1]), np.array([10]), 2), 20) is None


def test_htf_trend_alignment():
    hi, lo = _swings()
    assert B.htf_trend(hi, lo, 20) == 1                                 # HH (1.3>1.2) + HL (1.1>1.0)
    assert B.trend_aligned("long", 1) and not B.trend_aligned("short", 1) and B.trend_aligned("short", -1)
    dn_hi = SwingSet(np.array([5, 15]), np.array([1.3, 1.2]), np.array([7, 17]), 2)
    dn_lo = SwingSet(np.array([8, 18]), np.array([1.1, 1.0]), np.array([10, 20]), 2)
    assert B.htf_trend(dn_hi, dn_lo, 20) == -1
    assert B.htf_trend(hi, dn_lo, 20) == 0                              # HH but LL -> mixed
    assert B.htf_trend(hi, lo, 9) == 0 and not B.trend_aligned("long", 0)
