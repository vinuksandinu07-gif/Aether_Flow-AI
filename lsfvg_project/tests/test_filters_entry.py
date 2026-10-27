import tempfile

import numpy as np
import pandas as pd

from lsfvg.config.schema import FiltersCfg, SessionWindow
from lsfvg.core.types import EntryOrder
from lsfvg.data.news import blackout_windows, in_blackout, load_news_csv
from lsfvg.signals.entry import resolve_entry_order
from lsfvg.signals.filters import FilterInputs, chop_ok, failed_filters, session_mask, spread_ok, volatility_ok

T0 = pd.Timestamp("2026-01-07 09:45", tz="UTC")


# ------------------------------------------------------------------ filters
def test_scalar_filters_fail_closed_on_unknown_values():
    assert spread_ok(0.0003, 0.0003) and not spread_ok(0.00031, 0.0003) and not spread_ok(np.nan, 0.0003)
    assert volatility_ok(0.2, 0.2) and not volatility_ok(0.19, 0.2) and not volatility_ok(np.nan, 0.2)
    assert chop_ok(0.2, 0.2) and not chop_ok(0.1, 0.2) and not chop_ok(np.nan, 0.2)


def _fi(session=True, spread=0.0001, pct=0.5, er=0.5, close_ns=None, ns=None, ne=None):
    return FilterInputs(np.array([session]), np.array([spread]), np.array([pct]), np.array([er]),
                        np.array([close_ns if close_ns is not None else T0.value]),
                        ns if ns is not None else np.empty(0, "int64"), ne if ne is not None else np.empty(0, "int64"))


def test_failed_filters_respects_toggles_and_lists_every_failure():
    cfg = FiltersCfg()
    cfg.chop.enabled = True
    assert failed_filters(0, _fi(), cfg, 0.0003) == []
    bad = _fi(session=False, spread=0.001, pct=0.01, er=0.01)
    assert failed_filters(0, bad, cfg, 0.0003) == ["filter_session", "filter_spread", "filter_volatility", "filter_chop"]
    off = FiltersCfg(session_enabled=False, spread_enabled=False)
    off.volatility.enabled = False
    assert failed_filters(0, bad, off, 0.0003) == []


def test_session_mask_uses_bar_open_time_half_open():
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-01-07 {h}", tz="America/New_York").tz_convert("UTC")
                            for h in ("01:45", "02:00", "04:45", "05:00", "07:00", "09:59", "10:00")])
    m = session_mask(idx, [SessionWindow("l", "02:00", "05:00"), SessionWindow("ny", "07:00", "10:00")], "America/New_York")
    assert m.tolist() == [False, True, True, False, True, True, False]
    assert session_mask(idx, [], "America/New_York").all()


def test_news_blackout_windows_currency_impact_and_boundaries():
    csv = ("timestamp_utc,currency,impact,event\n2026-01-07T13:30:00Z,USD,high,NFP\n"
           "2026-01-07T09:00:00Z,EUR,high,ECB\n2026-01-07T15:00:00Z,USD,low,minor\n")
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv)
    news = load_news_csv(f.name)
    s, e = blackout_windows(news, ["USD"], "high", 15, 30)
    assert len(s) == 1                                                       # EUR event and low impact ignored
    t = pd.Timestamp("2026-01-07 13:30", tz="UTC").value
    mn = 60 * 10**9
    assert in_blackout(t - 15 * mn, s, e) and in_blackout(t + 30 * mn, s, e)             # boundaries inclusive
    assert not in_blackout(t - 15 * mn - 1, s, e) and not in_blackout(t + 30 * mn + 1, s, e)
    s2, _ = blackout_windows(news, ["EUR", "USD"], "low", 15, 30)
    assert len(s2) == 3
    assert not in_blackout(t, *blackout_windows(None, ["USD"], "high", 15, 30))


# ------------------------------------------------------------------ entry order lifecycle
def _ltf(rows, quotes=None):
    idx = pd.DatetimeIndex([T0 + i * pd.Timedelta("5min") for i in range(len(rows))], name="time").as_unit("ns")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    if quotes:
        for k, v in quotes.items():
            df[k] = v
    return df


def _order(direction="short", limit=1.1026, bottom=1.1026, top=1.1045, wait=4, target=1.0960, ticks=1):
    return EntryOrder("S1", direction, limit, T0, wait, top, bottom, target, 0.00001, ticks)


QUIET = (1.1010, 1.1015, 1.1005, 1.1012)


def test_short_limit_needs_trade_through_and_fills_at_limit():
    out = resolve_entry_order(_order(), _ltf([QUIET, (1.1012, 1.1027, 1.1010, 1.1020), QUIET]))
    assert out.status == "filled" and out.bar_idx == 1 and abs(out.fill_price - 1.1026) < 1e-12
    touch = resolve_entry_order(_order(wait=3), _ltf([QUIET, (1.1012, 1.1026, 1.1010, 1.1020), QUIET]))
    assert touch.status == "expired" and touch.bar_idx == 2                           # touch only: 1 tick through needed
    assert resolve_entry_order(_order(ticks=0), _ltf([QUIET, (1.1012, 1.1026, 1.1010, 1.1020)])).status == "filled"


def test_long_limit_mirror_and_gap_through_fills_at_open():
    long_o = _order("long", limit=1.1000, bottom=1.0980, top=1.1000, target=1.1040)
    out = resolve_entry_order(long_o, _ltf([(1.1010, 1.1015, 1.1005, 1.1008), (1.1007, 1.1008, 1.0999, 1.1002)]))
    assert out.status == "filled" and abs(out.fill_price - 1.1000) < 1e-12
    gap = resolve_entry_order(long_o, _ltf([(1.1010, 1.1015, 1.1005, 1.1008), (1.0990, 1.0995, 1.0985, 1.0992)]))
    assert gap.status == "filled" and abs(gap.fill_price - 1.0990) < 1e-12           # better price when gapping through
    gs = resolve_entry_order(_order(), _ltf([QUIET, (1.1030, 1.1032, 1.1028, 1.1031)]))
    assert gs.status == "filled" and abs(gs.fill_price - 1.1030) < 1e-12              # short: fills at the higher open


def test_expiry_after_max_wait_and_pending_when_data_ends():
    out = resolve_entry_order(_order(wait=3), _ltf([QUIET] * 6))
    assert out.status == "expired" and out.bar_idx == 2
    assert resolve_entry_order(_order(wait=10), _ltf([QUIET] * 3)).status == "pending"
    early = _ltf([(1.1012, 1.1030, 1.1010, 1.1020)] + [QUIET] * 5)
    early.index = early.index - pd.Timedelta("10min")                                # bar BEFORE place_time is ignored
    assert resolve_entry_order(_order(wait=3), early).status == "expired"


def test_cancel_when_target_taken_before_fill_and_target_wins_same_bar():
    out = resolve_entry_order(_order(), _ltf([QUIET, (1.1012, 1.1016, 1.0955, 1.0990), (1.0990, 1.1030, 1.0985, 1.1020)]))
    assert out.status == "cancelled_target_taken" and out.bar_idx == 1
    both = resolve_entry_order(_order(), _ltf([(1.1012, 1.1030, 1.0955, 1.1000)]))      # reaches entry AND target
    assert both.status == "cancelled_target_taken"                                       # conservative: no fill
    no_tgt = resolve_entry_order(_order(target=None), _ltf([(1.1012, 1.1030, 1.0955, 1.1000)]))
    assert no_tgt.status == "filled"


def test_uses_ask_for_long_and_bid_for_short_when_quotes_exist():
    long_o = _order("long", limit=1.1000, bottom=1.0980, top=1.1000, target=1.1040)
    mid_touch = [(1.1010, 1.1015, 1.0999, 1.1008)]
    q_no = {"ask_low": [1.1002], "ask_open": [1.1011]}                  # ask never trades through 1.0999 (needs <= 1.09999)
    assert resolve_entry_order(long_o, _ltf(mid_touch, q_no)).status == "pending"
    q_yes = {"ask_low": [1.0999], "ask_open": [1.1011]}
    assert resolve_entry_order(long_o, _ltf(mid_touch, q_yes)).status == "filled"
    short_q = {"bid_high": [1.1020], "bid_open": [1.1015]}              # mid high 1.1030 but bid only 1.1020
    assert resolve_entry_order(_order(wait=1), _ltf([(1.1015, 1.1030, 1.1010, 1.1012)], short_q)).status == "expired"


def test_close_fully_through_fvg_cancels_when_quotes_prevent_a_fill():
    # tiny FVG vs a wide spread: mid closes above the gap while the BID never reached the limit
    o = _order(limit=1.1026, bottom=1.1026, top=1.10262, wait=5, target=None)
    q = {"bid_high": [1.1020], "bid_open": [1.1015]}
    out = resolve_entry_order(o, _ltf([(1.1015, 1.1030, 1.1010, 1.1029)], q))
    assert out.status == "cancelled_fvg_through" and out.bar_idx == 0
