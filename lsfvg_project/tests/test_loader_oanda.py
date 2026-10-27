import tempfile
from pathlib import Path

import pandas as pd

from helpers import raises
from lsfvg.config.schema import CsvCfg
from lsfvg.data import oanda
from lsfvg.data.loader import data_version_hash, load_csv
from lsfvg.data.oanda import OandaCandleClient, OandaError
from lsfvg.data.synthetic import random_walk_bars


def _write(text: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    f.write(text)
    f.close()
    return f.name


def test_csv_bid_ask_derives_mid_and_keeps_quotes():
    p = _write("time,bid_o,bid_h,bid_l,bid_c,ask_o,ask_h,ask_l,ask_c,volume\n"
               "2026-01-05T10:00:00Z,1.1000,1.1010,1.0990,1.1005,1.1001,1.1011,1.0991,1.1006,10\n"
               "2026-01-05T10:05:00Z,1.1005,1.1012,1.1000,1.1008,1.1006,1.1013,1.1001,1.1009,12\n")
    ld = load_csv(p)
    df = ld.df
    assert str(df.index.tz) == "UTC" and ld.notes["mid_derived_from_bid_ask"]
    assert abs(df["open"].iloc[0] - 1.10005) < 1e-12 and "ask_close" in df.columns


def test_csv_mt5_style_server_time_is_converted_to_utc():
    p = _write("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\n"
               "2026.01.05\t10:00:00\t1.1000\t1.1010\t1.0990\t1.1005\t10\n")
    ld = load_csv(p, CsvCfg(tz="Europe/Helsinki", time_format="%Y.%m.%d %H:%M:%S"), sep="\t")
    assert ld.df.index[0] == pd.Timestamp("2026-01-05 08:00", tz="UTC")          # EET = UTC+2


def test_csv_ambiguous_and_unsorted_are_reported_not_hidden():
    p = _write("time,open,high,low,close\n"
               "2026-11-01 01:30:00,1.1,1.2,1.0,1.1\n"        # ambiguous in New York (DST ends)
               "2026-11-01 10:05:00,1.1,1.2,1.0,1.1\n"
               "2026-11-01 10:00:00,1.1,1.2,1.0,1.1\n")
    ld = load_csv(p, CsvCfg(tz="America/New_York"))
    assert ld.notes["unparseable_or_ambiguous_timestamps"] == 1 and ld.notes["was_unsorted"]
    assert len(ld.df) == 2 and ld.df.index.is_monotonic_increasing


def test_data_version_hash_changes_with_data():
    a = random_walk_bars(n_days=2, seed=1)
    b = a.copy()
    assert data_version_hash(a) == data_version_hash(b)
    b.iloc[5, b.columns.get_loc("close")] += 1e-5
    assert data_version_hash(a) != data_version_hash(b)


# ---------------------------------------------------------------- OANDA client (fake HTTP)
class _Resp:
    def __init__(self, code, body=None):
        self.status_code, self._b, self.text = code, body or {}, str(body)[:50]

    def json(self):
        return self._b


class _FakeSession:
    def __init__(self, n=10, incomplete_last=True, fail_first=0):
        self.t0 = pd.Timestamp("2026-01-05 10:00", tz="UTC")
        self.n, self.incomplete_last, self.fail_first = n, incomplete_last, fail_first
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params), dict(headers)))
        if self.fail_first > 0:
            self.fail_first -= 1
            return _Resp(429)
        frm = pd.Timestamp(params["from"])
        times = [self.t0 + i * pd.Timedelta("5min") for i in range(self.n)]
        sel = [t for t in times if (t >= frm if params["includeFirst"] == "true" else t > frm)][: params["count"]]
        out = []
        for t in sel:
            k = (t - self.t0) / pd.Timedelta("5min")
            q = lambda d: {"o": str(1.1 + d + k * 1e-4), "h": str(1.1005 + d + k * 1e-4), "l": str(1.0995 + d + k * 1e-4),
                           "c": str(1.1002 + d + k * 1e-4)}
            out.append({"time": t.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"), "volume": 10,
                        "complete": not (self.incomplete_last and t == times[-1]),
                        "mid": q(0), "bid": q(-5e-5), "ask": q(5e-5)})
        return _Resp(200, {"candles": out})


def test_oanda_paging_dedup_and_incomplete_candle_dropped():
    old = oanda.MAX_CANDLES
    oanda.MAX_CANDLES = 4
    try:
        sess = _FakeSession(n=10)
        cl = OandaCandleClient("secret-token", session=sess, sleep=lambda s: None)
        df = cl.fetch("EUR_USD", "5m", pd.Timestamp("2026-01-05 10:00", tz="UTC"), pd.Timestamp("2026-01-05 11:00", tz="UTC"))
    finally:
        oanda.MAX_CANDLES = old
    assert len(sess.calls) >= 3                                       # paged
    assert len(df) == 9 and df.index.is_unique and df.index.is_monotonic_increasing   # last (incomplete) dropped
    assert {"bid_close", "ask_open", "open", "volume"} <= set(df.columns)
    assert (df["ask_close"] > df["bid_close"]).all()


def test_oanda_retry_token_hygiene_and_hosts():
    sess = _FakeSession(n=3, incomplete_last=False, fail_first=2)
    cl = OandaCandleClient("secret-token", session=sess, sleep=lambda s: None)
    df = cl.fetch("EUR_USD", "5m", pd.Timestamp("2026-01-05 10:00", tz="UTC"), pd.Timestamp("2026-01-05 11:00", tz="UTC"))
    assert len(df) == 3 and len(sess.calls) == 3                       # two 429s then success
    assert "secret-token" not in repr(cl) and "fxpractice" in cl.host
    assert "fxtrade" in OandaCandleClient("t", practice=False, session=sess).host
    import os
    old = os.environ.pop("OANDA_API_TOKEN", None)
    try:
        with raises(OandaError, "OANDA_API_TOKEN"):
            OandaCandleClient()
    finally:
        if old:
            os.environ["OANDA_API_TOKEN"] = old
    dead = _FakeSession(fail_first=99)
    with raises(OandaError, "giving up"):
        OandaCandleClient("t", session=dead, sleep=lambda s: None, max_retries=2).fetch(
            "EUR_USD", "5m", pd.Timestamp("2026-01-05 10:00", tz="UTC"), pd.Timestamp("2026-01-05 11:00", tz="UTC"))
