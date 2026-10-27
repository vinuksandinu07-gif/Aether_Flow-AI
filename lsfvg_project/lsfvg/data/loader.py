"""CSV loading.  Detects problems, never silently repairs them.

Canonical frame: UTC ``DatetimeIndex`` named ``time`` (bar *open* time) with
mid columns ``open high low close volume`` and, when available,
``bid_*`` / ``ask_*`` for the four price fields.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config.schema import CsvCfg
from .spread import PRICE_FIELDS

_TS_CANDIDATES = ("timestamp", "time", "datetime", "timestamp_utc", "date")
_VOL_CANDIDATES = ("volume", "tick_volume", "tickvol", "vol")
_SHORT = {"o": "open", "h": "high", "l": "low", "c": "close"}


@dataclass
class LoadedData:
    df: pd.DataFrame
    notes: dict[str, Any] = field(default_factory=dict)   # fed into the quality report


def _norm(col: str) -> str:
    return re.sub(r"[<>\s]+", "", str(col)).lower()


def _map_price_columns(cols: list[str]) -> dict[str, str]:
    """normalized file column -> canonical name (open / bid_open / ...)."""
    mapping: dict[str, str] = {}
    for c in cols:
        m = re.fullmatch(r"(?:(bid|ask|mid)[_\-]?)?(open|high|low|close|o|h|l|c)", c)
        if not m:
            continue
        side, fld = m.group(1), m.group(2)
        fld = _SHORT.get(fld, fld)
        if side is None or side == "mid":
            mapping[c] = fld
        else:
            mapping[c] = f"{side}_{fld}"
    return mapping


def load_csv(path: str | Path, cfg: Optional[CsvCfg] = None, sep: Optional[str] = None) -> LoadedData:
    """Load OHLC(V) candles from CSV into the canonical frame.

    ``cfg.tz`` is the timezone of the timestamps *inside the file* (MT5 exports
    use broker server time, which is NOT UTC - set it explicitly).  Timestamps
    that cannot be parsed or are ambiguous/non-existent in that timezone are
    dropped and **counted** in ``notes`` so the quality report shows them.
    """
    cfg = cfg or CsvCfg()
    raw = pd.read_csv(path, sep=sep, engine="python")
    raw.columns = [_norm(c) for c in raw.columns]
    notes: dict[str, Any] = {"source": str(path), "rows_in_file": int(len(raw))}

    if cfg.timestamp_col:
        stamp = raw[_norm(cfg.timestamp_col)]
    elif "date" in raw.columns and "time" in raw.columns:            # MT5 export: <DATE> + <TIME>
        stamp = raw["date"].astype(str) + " " + raw["time"].astype(str)
    else:
        ts_col = next((c for c in _TS_CANDIDATES if c in raw.columns), None)
        if ts_col is None:
            raise ValueError(f"no timestamp column found in {path}; columns={list(raw.columns)}")
        stamp = raw[ts_col]

    ts = pd.to_datetime(stamp, format=cfg.time_format, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(cfg.tz, ambiguous="NaT", nonexistent="NaT")
    ts = ts.dt.tz_convert("UTC").dt.as_unit("ns")
    bad = ts.isna()
    notes["unparseable_or_ambiguous_timestamps"] = int(bad.sum())

    price_map = _map_price_columns(list(raw.columns))
    frame = pd.DataFrame({canon: pd.to_numeric(raw[src], errors="coerce")
                          for src, canon in price_map.items()})
    have_mid = all(f in frame for f in PRICE_FIELDS)
    have_ba = all(f"{s}_{f}" in frame for s in ("bid", "ask") for f in PRICE_FIELDS)
    if not have_mid and have_ba:
        for f in PRICE_FIELDS:
            frame[f] = (frame[f"bid_{f}"] + frame[f"ask_{f}"]) / 2.0
        notes["mid_derived_from_bid_ask"] = True
        have_mid = True
    if not have_mid:
        raise ValueError(f"need open/high/low/close (or bid/ask OHLC) columns; got {list(raw.columns)}")
    vol = next((c for c in _VOL_CANDIDATES if c in raw.columns), None)
    frame["volume"] = pd.to_numeric(raw[vol], errors="coerce") if vol else np.nan

    frame.index = pd.DatetimeIndex(ts, name="time")
    frame = frame[~bad.to_numpy()]
    notes["was_unsorted"] = not frame.index.is_monotonic_increasing
    if notes["was_unsorted"]:
        frame = frame.sort_index(kind="stable")      # order only; duplicates are kept and reported
    ordered = list(PRICE_FIELDS) + ["volume"] + [
        f"{s}_{f}" for s in ("bid", "ask") for f in PRICE_FIELDS if f"{s}_{f}" in frame
    ]
    frame = frame[ordered]
    frame.attrs["spread_modeled"] = False
    return LoadedData(frame, notes)


def data_version_hash(df: pd.DataFrame) -> str:
    """Short, stable content hash of the base dataset (stored with every run)."""
    h = hashlib.sha256()
    h.update(df.index.tz_convert("UTC").as_unit("ns").asi8.tobytes())
    for c in sorted(df.columns):
        h.update(c.encode())
        h.update(np.ascontiguousarray(df[c].to_numpy(dtype="float64")).tobytes())
    return h.hexdigest()[:16]
