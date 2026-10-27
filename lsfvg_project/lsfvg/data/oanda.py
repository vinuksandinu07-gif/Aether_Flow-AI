"""OANDA v20 candles client (data only - no order placement in this module).

* Token comes from the ``OANDA_API_TOKEN`` environment variable, never from code/config/logs.
* Practice host by default; the live host needs ``practice=False`` explicitly.
* Paging: the API returns at most 5000 candles per call, so we walk forward with
  ``from`` + ``count`` and de-duplicate the overlap.
* Only ``complete`` candles are kept (an unfinished candle would be repainted).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import pandas as pd
import requests

from .spread import PRICE_FIELDS

log = logging.getLogger(__name__)

PRACTICE_HOST = "https://api-fxpractice.oanda.com"
LIVE_HOST = "https://api-fxtrade.oanda.com"
MAX_CANDLES = 5000
GRANULARITY = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30", "1h": "H1", "4h": "H4", "1d": "D"}
_GRAN_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D": 86400}


class OandaError(RuntimeError):
    pass


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


class OandaCandleClient:
    def __init__(
        self,
        token: Optional[str] = None,
        *,
        practice: bool = True,
        session: Optional[requests.Session] = None,
        max_retries: int = 5,
        backoff_s: float = 1.0,
        timeout_s: float = 30.0,
        sleep=time.sleep,
    ) -> None:
        self._token = token or os.environ.get("OANDA_API_TOKEN", "")
        if not self._token:
            raise OandaError("OANDA_API_TOKEN environment variable is not set")
        self.host = PRACTICE_HOST if practice else LIVE_HOST
        self.session = session or requests.Session()
        self.max_retries, self.backoff_s, self.timeout_s, self._sleep = max_retries, backoff_s, timeout_s, sleep

    def __repr__(self) -> str:                      # never leak the token
        return f"OandaCandleClient(host={self.host!r}, token='***')"

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._token}", "Accept-Datetime-Format": "RFC3339"}
        last: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.get(self.host + path, params=params, headers=headers, timeout=self.timeout_s)
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise OandaError(f"{last} for {path}: {r.text[:200]}")
            except requests.RequestException as exc:     # network error -> retry
                last = type(exc).__name__
            self._sleep(self.backoff_s * (2 ** attempt))
        raise OandaError(f"giving up on {path} after {self.max_retries + 1} attempts ({last})")

    def fetch(
        self,
        instrument: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        price: str = "MBA",
    ) -> pd.DataFrame:
        """Candles in ``[start, end)`` as the canonical frame (mid + bid/ask)."""
        gran = GRANULARITY[timeframe]
        start, end = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
        rows: list[dict[str, Any]] = []
        cursor, first = start, True
        while cursor < end:
            params = {"price": price, "granularity": gran, "from": _iso(cursor),
                      "count": MAX_CANDLES, "includeFirst": "true" if first else "false"}
            data = self._get(f"/v3/instruments/{instrument}/candles", params)
            candles = data.get("candles", [])
            if not candles:
                break
            last_time = cursor
            for c in candles:
                t = pd.Timestamp(c["time"]).tz_convert("UTC")
                last_time = max(last_time, t)
                if t >= end or not c.get("complete", False):
                    continue
                row: dict[str, Any] = {"time": t, "volume": float(c.get("volume", 0))}
                for side, short in (("mid", ""), ("bid", "bid_"), ("ask", "ask_")):
                    if side in c:
                        for f, k in zip(PRICE_FIELDS, "ohlc"):
                            row[f"{short}{f}"] = float(c[side][k])
                rows.append(row)
            first = False
            if last_time <= cursor:
                break
            cursor = last_time
            if len(candles) < MAX_CANDLES:       # short page = no more data available
                break
        if not rows:
            return pd.DataFrame(columns=list(PRICE_FIELDS) + ["volume"],
                                index=pd.DatetimeIndex([], tz="UTC", name="time"))
        df = pd.DataFrame(rows).drop_duplicates("time").set_index("time").sort_index()
        df.index = df.index.as_unit("ns")
        df.attrs["spread_modeled"] = False
        log.info("fetched %d %s candles for %s", len(df), gran, instrument)
        return df
