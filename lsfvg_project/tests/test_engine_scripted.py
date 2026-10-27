"""End-to-end engine tests on the scripted scenario (exact, hand-checkable numbers)."""
import json
import tempfile
from pathlib import Path

import pandas as pd

from helpers import CONFIG_PATH, build, relaxed_cfg, scenario
from lsfvg.config import load_config
from lsfvg.signals.engine import decision_records, write_decision_log
from lsfvg.signals.entry import resolve_entry_order

BASE, NAMES = scenario()
SIG_TIME = NAMES["n1"] + pd.Timedelta("15min")          # n1 completes the FVG -> signal at its close


def _find(res, signal_time=SIG_TIME, kind="PDH"):
    hits = [s for s in res.all_setups if s.signal_time == signal_time and s.pool_id.startswith(kind)]
    assert len(hits) == 1, [s.pool_id for s in res.all_setups]
    return hits[0]


def _run(extra=None, quotes=False):
    cfg = relaxed_cfg(extra)
    eng, prep, stack = build(cfg, BASE, with_quotes=quotes)
    return cfg, eng, prep, stack, eng.scan(prep)


def test_scripted_short_setup_is_found_with_exact_structure():
    cfg, eng, prep, stack, res = _run()
    s = _find(res)
    mtf = stack.mtf
    pos = lambda n: mtf.index.get_loc(NAMES[n])
    assert s.status == "valid" and s.direction == "short" and s in res.setups
    assert s.sweep.breach_idx == s.sweep.reclaim_idx == s.sweep.extreme_idx == pos("SW")
    assert abs(s.sweep.pool_level - 1.1050) < 1e-12 and abs(s.sweep.extreme_price - 1.1056) < 1e-12
    assert s.mss.idx == pos("D") and abs(s.mss.level - 1.1022) < 1e-12 and s.mss.swing_idx == pos("P")
    assert (s.fvg.direction, s.fvg.mid_idx, s.fvg.last_idx) == ("bear", pos("D"), pos("n1"))
    assert abs(s.fvg.bottom - 1.1026) < 1e-12 and abs(s.fvg.top - 1.1045) < 1e-12
    assert s.signal_idx == pos("n1") and s.signal_time == SIG_TIME
    assert abs(s.entry_price - 1.1026) < 1e-12                                   # edge = proximal (bottom) for a short


def test_stop_target_and_costs_for_the_scripted_setup():
    cfg, eng, prep, stack, res = _run()
    s = _find(res)
    atr = prep.atr[s.signal_idx]
    assert abs(s.atr_mtf - atr) < 1e-15 and abs(s.spread_est - 0.0001) < 1e-12
    expected_buffer = 0.0001 + 0.1 * atr
    assert abs(s.stop.buffer - expected_buffer) < 1e-12
    assert 1.1056 + expected_buffer <= s.stop_price < 1.1056 + expected_buffer + 0.00001 + 1e-12    # tick-rounded away
    risk = s.stop_price - s.entry_price
    assert abs(s.targets.final_target - round(1.1026 - 2 * risk, 5)) < 1.1e-5            # fixed 2R
    assert s.targets.planned_rr > s.targets.effective_rr >= 1.0                          # costs reduce RR; min_rr=1
    assert s.rank_key[0] in (0, 1) and s.stop.action == "ok"


def test_duplicate_pool_at_same_level_is_ranked_lower_priority():
    cfg, eng, prep, stack, res = _run()
    second = _find(res, kind="SESS_H")
    assert second.status == "skipped" and second.reasons == ["lower_priority"] and second in res.skipped
    assert res.skip_counts().get("lower_priority", 0) >= 1


def test_entry_order_resolves_to_a_fill_on_the_retrace_candle():
    cfg, eng, prep, stack, res = _run()
    s = _find(res)
    order = eng.entry_order(s)
    assert order.limit_price == s.entry_price and order.place_time == SIG_TIME and order.max_wait_bars == 24
    assert (order.fvg_bottom, order.fvg_top) == (s.fvg.bottom, s.fvg.top) and order.target_level is None   # fixed_rr: no pool
    out = resolve_entry_order(order, stack.ltf)
    r2 = NAMES["r2"]
    assert out.status == "filled" and r2 <= out.time < r2 + pd.Timedelta("15min")
    assert abs(out.fill_price - 1.1026) < 1e-12
    opp = relaxed_cfg({"targets": {"tp_mode": "opposing_liquidity"}})
    eng2, prep2, st2 = build(opp, BASE)
    s2 = _find(eng2.scan(prep2))
    if s2.status == "valid":
        assert eng2.entry_order(s2).target_level == s2.targets.reference_target


def test_opposing_liquidity_target_is_nearest_unswept_sell_side_pool_below_entry():
    cfg = relaxed_cfg({"targets": {"tp_mode": "opposing_liquidity", "min_rr": 0.0}})
    eng, prep, stack = build(cfg, BASE)
    s = _find(eng.scan(prep))
    act = [p for p in prep.pools if p.active_at(s.signal_idx) and p.side == "sell" and p.level < s.entry_price]
    assert act and abs(s.targets.final_target - max(p.level for p in act)) < 1e-12
    assert s.targets.final_pool_id == max(act, key=lambda p: p.level).pool_id


def test_stop_buffer_zero_gives_extreme_plus_spread_only():
    cfg, eng, prep, stack, res = _run({"stops": {"stop_buffer_atr": 0.0}})
    assert abs(_find(res).stop_price - 1.1057) < 1e-9


def test_default_config_skips_this_setup_because_the_stop_is_too_wide():
    cfg = load_config(CONFIG_PATH)
    eng, prep, stack = build(cfg, BASE)
    res = eng.scan(prep)
    s = _find(res)
    assert s.status == "skipped" and "stop_too_wide" in s.reasons and s not in res.setups
    assert s.stop.distance_atr > cfg.stops.max_stop_atr
    assert s.reasons == ["stop_too_wide"]                      # bias (premium), session, spread, vol, min_rr all pass
    assert s.bias_zone == "premium" and s.meta["dealing_range"][0] == 1.1050   # HTF range from closed 4H swings


def test_entry_mode_midpoint_and_fvg_first_candle_mode():
    _, _, _, _, res = _run({"signals": {"entry_mode": "midpoint"}})
    assert abs(_find(res).entry_price - 1.10355) < 1e-9
    _, _, _, _, res2 = _run({"stops": {"stop_mode": "fvg_first_candle"}})
    assert abs(_find(res2).sweep.extreme_price - 1.1056) < 1e-12        # first candle of the FVG is the sweep candle


def test_each_filter_can_reject_the_setup_with_its_own_reason():
    def reasons(extra, quotes=False):
        return _find(_run(extra, quotes)[4]).reasons
    assert "filter_session" in reasons({"instruments": {"EUR_USD": {"entry_sessions": [{"name": "x", "start": "10:00", "end": "11:00"}]}}})
    assert "filter_spread" in reasons({"instruments": {"EUR_USD": {"max_spread": 0.00005}}})
    assert "filter_volatility" in reasons({"filters": {"volatility": {"enabled": True, "lookback_bars": 100000}}})   # no history -> fail closed
    assert "filter_chop" in reasons({"filters": {"chop": {"enabled": True, "min_er": 0.999}}})
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("timestamp_utc,currency,impact,event\n2026-01-07T09:40:00Z,USD,high,CPI\n")
    from lsfvg.data.news import load_news_csv
    cfg = relaxed_cfg({"filters": {"news": {"enabled": True, "csv_path": f.name}}})
    eng, prep, stack = build(cfg, BASE)
    eng.news = load_news_csv(f.name)
    prep = eng.prepare(stack)
    assert "filter_news" in _find(eng.scan(prep)).reasons
    assert reasons({}) == [] or reasons({}) == ["lower_priority"]


def test_bid_ask_data_changes_spread_estimate_not_structure():
    _, _, prep, _, res = _run(quotes=True)
    s = _find(res)
    assert abs(s.spread_est - 0.0001) < 1e-9 and abs(s.entry_price - 1.1026) < 1e-12


def test_scan_is_deterministic_and_start_idx_excludes_earlier_signals():
    cfg, eng, prep, stack, res = _run()
    again = eng.scan(prep)
    assert [s.to_dict() for s in res.all_setups] == [s.to_dict() for s in again.all_setups]
    s = _find(res)
    assert all(x.signal_idx >= s.signal_idx + 1 or x.signal_idx < s.signal_idx for x in eng.scan(prep, start_idx=s.signal_idx + 1).all_setups)
    assert s.setup_id not in [x.setup_id for x in eng.scan(prep, start_idx=s.signal_idx + 1).all_setups]
    assert s.setup_id in [x.setup_id for x in eng.scan(prep, start_idx=s.signal_idx).all_setups]
    assert s.setup_id not in [x.setup_id for x in eng.scan(prep, end_idx=s.signal_idx - 1).all_setups]


def test_decision_log_explains_every_decision():
    cfg, eng, prep, stack, res = _run()
    rows = decision_records(res)
    assert {r["decision"] for r in rows} >= {"TAKE", "SKIP"} and all(r["why"] for r in rows)
    with tempfile.TemporaryDirectory() as d:
        p = write_decision_log(res, Path(d) / "decisions.jsonl")
        parsed = [json.loads(line) for line in p.read_text().splitlines()]
        assert len(parsed) == len(rows) and any(r["decision"] == "TAKE" for r in parsed)
