"""Time helpers.

Everything is *stored* in UTC.  Wall-clock questions (sessions, trading day,
4H anchors) are answered in an explicit IANA timezone (America/New_York) via
``zoneinfo`` so DST is handled by the tz database, never by hand.

Implementation note: all arithmetic is done on int64 nanosecond arrays, which
sidesteps pandas' version-dependent datetime resolutions.
"""
from __future__ import annotations

import re
from datetime import time as dtime

import numpy as np
import pandas as pd

NS_MIN = 60 * 10**9
NS_HOUR = 60 * NS_MIN
NS_DAY = 24 * NS_HOUR


def parse_tf(tf: str) -> pd.Timedelta:
    """``'15m' -> Timedelta('15min')``; supports m/min/h/d."""
    m = re.fullmatch(r"(\d+)\s*(m|min|h|d)", tf.strip().lower())
    if not m:
        raise ValueError(f"bad timeframe '{tf}'")
    n, u = int(m.group(1)), m.group(2)
    return pd.Timedelta(minutes=n * {"m": 1, "min": 1, "h": 60, "d": 1440}[u])


def tf_ns(tf: str) -> int:
    return int(parse_tf(tf).value)


def parse_hhmm(s: str) -> int:
    """``'09:30' -> 570`` (minutes after midnight)."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def to_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """UTC DatetimeIndex -> int64 ns since epoch (resolution independent)."""
    if index.tz is None:
        raise ValueError("expected a tz-aware (UTC) index")
    return index.tz_convert("UTC").as_unit("ns").asi8.copy()


def from_ns(ns: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(ns, dtype="int64"), unit="ns", utc=True))


def local_naive_ns(index: pd.DatetimeIndex, tz: str) -> np.ndarray:
    """Wall-clock time in ``tz`` as int64 ns (naive).  DST-correct."""
    loc = index.tz_convert(tz).tz_localize(None)
    return loc.as_unit("ns").asi8.copy()


def bin_open_ns(index: pd.DatetimeIndex, tf: str, tz: str, anchor: str) -> np.ndarray:
    """Open time (UTC, int64 ns) of the anchored ``tf`` bin each timestamp falls in.

    Bins are aligned to wall-clock ``anchor`` in ``tz`` (e.g. 17:00 New York), so
    4H bins open at 17,21,01,05,09,13 NY all year.  The offset inside the bin is
    measured on the wall clock and subtracted from the UTC instant, which is
    exact as long as no DST change falls *inside* a bin (true for FX/CFDs, which
    are closed Fri 17:00 -> Sun 17:00 NY when the change happens).
    """
    step = tf_ns(tf)
    shifted = local_naive_ns(index, tz) - parse_hhmm(anchor) * NS_MIN
    off = shifted % step
    return to_ns(index) - off


def trading_day_ids(index: pd.DatetimeIndex, tz: str, anchor: str) -> np.ndarray:
    """Integer id (days since epoch) of the anchored trading day for each timestamp.

    A trading day runs anchor -> anchor (17:00 NY -> 17:00 NY), labelled by the
    calendar date on which it *starts* in the shifted clock.
    """
    shifted = local_naive_ns(index, tz) - parse_hhmm(anchor) * NS_MIN
    return shifted // NS_DAY


def trading_week_ids(day_ids: np.ndarray) -> np.ndarray:
    """Week id (day id of the Sunday that starts the week) from trading-day ids.

    1970-01-01 was a Thursday, so Sunday-based weekday = (d + 4) % 7.
    The Sun 17:00 NY -> Fri 17:00 NY trading week therefore maps to one id.
    """
    d = np.asarray(day_ids, dtype="int64")
    return d - ((d + 4) % 7)


def minute_of_day_local(index: pd.DatetimeIndex, tz: str) -> np.ndarray:
    return (local_naive_ns(index, tz) % NS_DAY) // NS_MIN


def in_window_mask(minutes_of_day: np.ndarray, start: str, end: str) -> np.ndarray:
    """Half-open window [start, end) on minute-of-day; ``start > end`` wraps midnight."""
    s, e = parse_hhmm(start), parse_hhmm(end)
    m = np.asarray(minutes_of_day)
    if s == e:
        return np.ones(m.shape, dtype=bool)
    if s < e:
        return (m >= s) & (m < e)
    return (m >= s) | (m < e)


def session_instance_ids(index: pd.DatetimeIndex, tz: str, start: str, end: str) -> np.ndarray:
    """Per-bar id of the session *instance* (day the session started), -1 if outside.

    For wrapping windows (e.g. 19:00 -> 02:00) the part after midnight belongs to
    the instance that started the previous calendar day.
    """
    loc = local_naive_ns(index, tz)
    mod = (loc % NS_DAY) // NS_MIN
    day = loc // NS_DAY
    s, e = parse_hhmm(start), parse_hhmm(end)
    inside = in_window_mask(mod, start, end)
    ids = np.where(inside, day, -1)
    if s > e:
        ids = np.where(inside & (mod < e), day - 1, ids)
    return ids


def to_time(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))
