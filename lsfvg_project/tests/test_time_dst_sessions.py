"""DST / session conversion, anchored bins, trading day and week ids."""
import numpy as np
import pandas as pd

from lsfvg.core.timeutil import (bin_open_ns, in_window_mask, minute_of_day_local, session_instance_ids,
                                 trading_day_ids, trading_week_ids, to_ns)
from lsfvg.signals.filters import session_mask
from lsfvg.config.schema import SessionWindow

TZ = "America/New_York"


def _utc_range(start, end, freq="1h"):
    return pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq=freq, inclusive="left")


def test_session_window_follows_dst_spring_forward_week():
    # DST starts Sunday 2026-03-08. NY 07:00-10:00 is 12:00-15:00 UTC before, 11:00-14:00 UTC after.
    idx = _utc_range("2026-03-06", "2026-03-11")
    mask = session_mask(idx, [SessionWindow("ny", "07:00", "10:00")], TZ)
    hours = {d: sorted(idx[mask & (idx.date == d)].hour) for d in sorted(set(idx.date))}
    assert hours[pd.Timestamp("2026-03-06").date()] == [12, 13, 14]      # EST (UTC-5)
    assert hours[pd.Timestamp("2026-03-09").date()] == [11, 12, 13]      # EDT (UTC-4)
    assert hours[pd.Timestamp("2026-03-10").date()] == [11, 12, 13]


def test_session_window_follows_dst_fall_back_week():
    # DST ends Sunday 2026-11-01.
    idx = _utc_range("2026-10-29", "2026-11-04")
    mask = session_mask(idx, [SessionWindow("lon", "02:00", "05:00")], TZ)
    before = sorted(idx[mask & (idx.date == pd.Timestamp("2026-10-30").date())].hour)
    after = sorted(idx[mask & (idx.date == pd.Timestamp("2026-11-02").date())].hour)
    assert before == [6, 7, 8]            # EDT (UTC-4)
    assert after == [7, 8, 9]             # EST (UTC-5)


def test_utc_storage_is_never_ambiguous_across_dst():
    idx = _utc_range("2026-11-01 04:00", "2026-11-01 09:00", "15min")     # includes the repeated NY hour
    loc = idx.tz_convert(TZ)
    assert idx.is_unique and idx.is_monotonic_increasing
    assert loc.hour.tolist().count(1) == 8                                # 01:xx occurs twice (EDT then EST)


def test_4h_bins_are_anchored_at_1700_ny_all_year():
    for start in ("2026-03-03", "2026-03-10", "2026-10-27", "2026-11-03"):
        idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=24 * 12 * 2, freq="5min")
        keys = bin_open_ns(idx, "4h", TZ, "17:00")
        opens = pd.DatetimeIndex(pd.to_datetime(np.unique(keys), unit="ns", utc=True)).tz_convert(TZ)
        assert set(opens.hour) <= {1, 5, 9, 13, 17, 21} and set(opens.minute) == {0}, (start, sorted(set(opens.hour)))


def test_trading_day_and_week_ids():
    t = lambda s: pd.DatetimeIndex([pd.Timestamp(s, tz=TZ).tz_convert("UTC")])
    sun_open, mon_late, mon_roll = t("2026-01-04 17:00"), t("2026-01-05 16:59"), t("2026-01-05 17:00")
    d = lambda i: int(trading_day_ids(i, TZ, "17:00")[0])
    assert d(sun_open) == d(mon_late) and d(mon_roll) == d(sun_open) + 1
    fri_late = t("2026-01-09 16:00")
    w = lambda i: int(trading_week_ids(trading_day_ids(i, TZ, "17:00"))[0])
    assert w(sun_open) == w(fri_late) == d(sun_open)                      # Sun 17:00 -> Fri 17:00 = one week
    assert w(t("2026-01-11 17:00")) == w(sun_open) + 7                     # next Sunday opens next week


def test_window_wraps_midnight_and_instance_ids_group_correctly():
    mins = np.array([18 * 60 + 59, 19 * 60, 23 * 60 + 59, 0, 119, 120, 600])
    assert in_window_mask(mins, "19:00", "02:00").tolist() == [False, True, True, True, True, False, False]
    idx = pd.DatetimeIndex([pd.Timestamp(x, tz=TZ).tz_convert("UTC") for x in
                            ("2026-01-05 19:00", "2026-01-06 01:45", "2026-01-06 02:00", "2026-01-06 19:00")])
    ids = session_instance_ids(idx, TZ, "19:00", "02:00")
    assert ids[0] == ids[1] and ids[2] == -1 and ids[3] == ids[0] + 1
    assert minute_of_day_local(idx[:1], TZ)[0] == 19 * 60
