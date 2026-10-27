import pandas as pd

from lsfvg.config.schema import PoolsCfg, StopsCfg, TargetsCfg
from lsfvg.core.types import CostEstimate, Pool
from lsfvg.signals.stops import compute_stop, round_to_tick
from lsfvg.signals.targets import (breakeven_win_rate, compute_targets, effective_rr, nearest_opposing_pool)

TS = pd.Timestamp("2026-01-06", tz="UTC")
TICK = 0.00001
COSTS = CostEstimate(spread=0.0001, slippage_entry=0.00001, slippage_exit=0.00005, commission=0.0)   # C = 0.00016


def _stop(direction, entry, extreme, atr=0.0008, spread=0.0001, first_high=0.0, first_low=0.0, **cfg):
    return compute_stop(direction, entry, sweep_extreme=extreme, fvg_first_high=first_high, fvg_first_low=first_low,
                        atr_mtf=atr, spread=spread, cfg=StopsCfg(**cfg), tick=TICK)


def _pool(side, level, scope="external"):
    return Pool(f"{side}{level}", "X", side, level, "D", scope, TS, TS, 0, 10**6, None)


# ------------------------------------------------------------------ stop placement
def test_sweep_extreme_stop_formula_buffer_is_spread_plus_atr_fraction():
    r = _stop("short", 1.1026, 1.1056, max_stop_atr=10)
    assert abs(r.buffer - (0.0001 + 0.1 * 0.0008)) < 1e-12
    assert abs(r.stop - 1.10578) < 1e-9 and r.action == "ok" and abs(r.distance - (r.stop - 1.1026)) < 1e-12
    lg = _stop("long", 1.1000, 1.0970, max_stop_atr=10)
    assert abs(lg.stop - 1.09682) < 1e-9 and lg.stop < 1.0970                       # beyond the sweep low
    z = _stop("short", 1.1026, 1.1056, spread=0.0, stop_buffer_atr=0.0, max_stop_atr=10)
    assert abs(z.stop - 1.1056) < 1e-9                                              # zero buffer -> exactly the extreme


def test_fvg_first_candle_mode_is_tighter_when_candle_is_inside_the_sweep():
    a = _stop("short", 1.1026, 1.1056, max_stop_atr=10)
    b = _stop("short", 1.1026, 1.1056, first_high=1.1040, stop_mode="fvg_first_candle", max_stop_atr=10)
    assert b.stop < a.stop and abs(b.stop - (1.1040 + 0.00018)) < 1e-9
    c = _stop("long", 1.1000, 1.0970, first_low=1.0985, stop_mode="fvg_first_candle", max_stop_atr=10)
    assert abs(c.stop - (1.0985 - 0.00018)) < 1e-9


def test_min_stop_widens_or_skips_and_max_stop_skips_with_reason():
    # distance 0.0003 (+ buffer 0) < min 0.5 x 0.0008 = 0.0004
    w = _stop("short", 1.1000, 1.1003, spread=0.0, stop_buffer_atr=0.0, max_stop_atr=3)
    assert w.action == "widened" and abs(w.stop - 1.1004) < 1e-9 and abs(w.distance_atr - 0.5) < 1e-6
    s = _stop("short", 1.1000, 1.1003, spread=0.0, stop_buffer_atr=0.0, min_stop_action="skip")
    assert s.stop is None and s.action == "skipped" and s.reason == "stop_too_tight"
    far = _stop("short", 1.1000, 1.1030, spread=0.0, stop_buffer_atr=0.0)             # 0.003 = 3.75 ATR > 1.5
    assert far.stop is None and far.reason == "stop_too_wide"
    edge = _stop("short", 1.1000, 1.1012, spread=0.0, stop_buffer_atr=0.0)            # exactly 1.5 ATR is allowed
    assert edge.stop is not None and edge.reason == ""


def test_stop_rounds_away_from_entry_and_must_be_beyond_entry():
    assert abs(round_to_tick(1.10087654, TICK, "up") - 1.10088) < 1e-12
    assert abs(round_to_tick(1.09912345, TICK, "down") - 1.09912) < 1e-12
    assert abs(round_to_tick(1.123456, TICK, "nearest") - 1.12346) < 1e-12
    s = _stop("short", 1.1000, 1.0995, spread=0.0, stop_buffer_atr=0.0)
    assert s.stop is None and s.reason == "stop_not_beyond_entry"


# ------------------------------------------------------------------ cost-adjusted RR
def test_effective_rr_and_breakeven_win_rate():
    assert abs(effective_rr(0.003, 0.0015, COSTS) - (0.003 - 0.00016) / (0.0015 + 0.00016)) < 1e-12
    assert effective_rr(0.003, 0.0015, COSTS) < 0.003 / 0.0015
    assert abs(breakeven_win_rate(2.0) - 1 / 3) < 1e-12 and breakeven_win_rate(1.0) == 0.5
    assert abs(COSTS.round_trip - 0.00016) < 1e-12


# ------------------------------------------------------------------ TP selection
def test_nearest_opposing_pool_for_long_and_short_with_scope():
    pools = [_pool("buy", 1.1050), _pool("buy", 1.1030, "internal"), _pool("buy", 1.0990), _pool("sell", 1.1040),
             _pool("sell", 1.0950), _pool("sell", 1.0980, "internal")]
    assert nearest_opposing_pool("long", 1.1000, pools).level == 1.1030            # nearest buy-side ABOVE entry
    assert nearest_opposing_pool("long", 1.1000, pools, "external").level == 1.1050
    assert nearest_opposing_pool("short", 1.1000, pools).level == 1.0980           # nearest sell-side BELOW entry
    assert nearest_opposing_pool("short", 1.1000, pools, "external").level == 1.0950
    assert nearest_opposing_pool("long", 1.2000, pools) is None


def _plan(direction="long", entry=1.1000, stop=1.0990, pools=(), **cfg):
    return compute_targets(direction, entry, stop, list(pools), TargetsCfg(**cfg), PoolsCfg(), COSTS, TICK)


def test_opposing_liquidity_target_rr_gate_and_no_pool():
    ok = _plan(pools=[_pool("buy", 1.1030)])
    assert ok.final_target == 1.1030 and abs(ok.planned_rr - 3.0) < 1e-9 and ok.skip_reason == ""
    assert abs(ok.effective_rr - (0.003 - 0.00016) / (0.001 + 0.00016)) < 1e-9
    low = _plan(pools=[_pool("buy", 1.1015)])
    assert low.skip_reason == "rr_below_min" and abs(low.planned_rr - 1.5) < 1e-9
    assert _plan(pools=[]).skip_reason == "no_target_pool"
    short = _plan("short", 1.1000, 1.1010, pools=[_pool("sell", 1.0970)])
    assert short.final_target == 1.0970 and short.skip_reason == ""


def test_fixed_rr_target_uses_cost_adjusted_gate():
    two = _plan(tp_mode="fixed_rr", fixed_rr=2.0)
    assert abs(two.final_target - 1.1020) < 1e-9 and two.skip_reason == "rr_below_min"       # net 1.59 < 2
    three = _plan(tp_mode="fixed_rr", fixed_rr=3.0)
    assert abs(three.final_target - 1.1030) < 1e-9 and three.skip_reason == ""
    sh = _plan("short", 1.1000, 1.1010, tp_mode="fixed_rr", fixed_rr=3.0)
    assert abs(sh.final_target - 1.0970) < 1e-9
    assert _plan(tp_mode="fixed_rr", fixed_rr=2.0, min_rr=1.5).skip_reason == ""


def test_partial_runner_plan_partial_at_1r_internal_liquidity_and_blended_rr():
    ext = _pool("buy", 1.1040)
    p = _plan(pools=[ext], tp_mode="partial_runner", partial_pct=0.5, be_offset_r=0.1, time_stop_bars=12)
    assert abs(p.partial_price - 1.1010) < 1e-9 and abs(p.partial_r - 1.0) < 1e-9 and p.final_target == 1.1040
    assert abs(p.planned_rr - 2.5) < 1e-9                                            # 0.5 x 1R + 0.5 x 4R
    assert abs(p.be_offset_price - 0.0001) < 1e-12 and p.time_stop_bars == 12 and p.time_stop_r == 0.5
    i = _plan(pools=[ext, _pool("buy", 1.1006, "internal")], tp_mode="partial_runner", partial_at="internal_liquidity")
    assert abs(i.partial_price - 1.1006) < 1e-9 and abs(i.partial_r - 0.6) < 1e-9
    near = _plan(pools=[_pool("buy", 1.1008)], tp_mode="partial_runner", min_rr=0.0)       # target closer than 1R
    assert near.partial_price is None and near.final_target == 1.1008
    assert _plan(pools=[_pool("buy", 1.1040, "internal")], tp_mode="partial_runner").skip_reason == "no_target_pool"


def test_trailing_plan_activation_reference_target_and_fallback():
    t = _plan(pools=[_pool("buy", 1.1030)], tp_mode="trailing", trail_activation_r=1.0, trail_mode="fvg")
    assert t.final_target is None and t.reference_target == 1.1030 and abs(t.trail_activation_price - 1.1010) < 1e-9
    assert t.trail_mode == "fvg" and t.skip_reason == ""
    fb = _plan(pools=[], tp_mode="trailing", fixed_rr=3.0)
    assert abs(fb.reference_target - 1.1030) < 1e-9 and fb.final_pool_id is None
    short = _plan("short", 1.1000, 1.1010, pools=[_pool("sell", 1.0960)], tp_mode="trailing", trail_activation_r=1.5)
    assert abs(short.trail_activation_price - 1.0985) < 1e-9
