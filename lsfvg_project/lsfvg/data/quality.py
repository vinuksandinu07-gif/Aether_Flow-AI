"""Data quality report - run before every backtest.

Philosophy: **detect and report, never silently fix.**  Problems become
``Issue`` rows with a severity; per-bar flags are kept so you can look at the
offending candles.  Removing flagged bars is a separate, explicit call
(:func:`drop_flagged`) that logs what it did.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from ..config.schema import InstrumentCfg, QualityCfg
from ..core.timeutil import NS_DAY, NS_MIN, local_naive_ns, parse_hhmm, tf_ns, to_ns
from .spread import PRICE_FIELDS

log = logging.getLogger(__name__)
Severity = Literal["info", "warning", "error"]


class DataQualityError(RuntimeError):
    """Raised by :func:`enforce` when error-level issues exist."""


@dataclass
class Issue:
    kind: str
    severity: Severity
    count: int
    detail: str
    examples: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    instrument: str
    base_tf: str
    n_bars: int
    start: str
    end: str
    spread_modeled: bool
    issues: list[Issue]
    stats: dict[str, Any]
    flags: pd.DataFrame            # rows with at least one per-bar flag

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument, "base_tf": self.base_tf, "n_bars": self.n_bars,
            "start": self.start, "end": self.end, "spread_modeled": self.spread_modeled,
            "ok": self.ok, "issues": [asdict(i) for i in self.issues], "stats": self.stats,
        }

    def to_markdown(self) -> str:
        L = [f"# Data quality - {self.instrument} ({self.base_tf})", "",
             f"* bars: **{self.n_bars:,}**  ({self.start} -> {self.end})",
             f"* prices: {'mid + MODELED spread' if self.spread_modeled else 'recorded bid/ask or mid only'}",
             f"* verdict: **{'OK' if self.ok else 'ERRORS PRESENT'}**", ""]
        L += ["| severity | kind | count | detail |", "|---|---|---:|---|"]
        order = {"error": 0, "warning": 1, "info": 2}
        for i in sorted(self.issues, key=lambda x: (order[x.severity], x.kind)):
            ex = f" e.g. {', '.join(i.examples[:3])}" if i.examples else ""
            L.append(f"| {i.severity} | {i.kind} | {i.count} | {i.detail}{ex} |")
        L += ["", "## Stats", "```json", json.dumps(self.stats, indent=2, default=str), "```"]
        return "\n".join(L)

    def save(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"data_quality_{self.instrument}"
        pj, pm = out / f"{stem}.json", out / f"{stem}.md"
        pj.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        pm.write_text(self.to_markdown())
        return pj, pm


def _ex(idx: pd.DatetimeIndex, mask: np.ndarray, k: int = 3) -> list[str]:
    return [str(t) for t in idx[mask][:k]]


def _bar_sets(df: pd.DataFrame) -> list[str]:
    sets = [""]
    if all(f"bid_{f}" in df for f in PRICE_FIELDS):
        sets.append("bid_")
    if all(f"ask_{f}" in df for f in PRICE_FIELDS):
        sets.append("ask_")
    return sets


def run_quality_report(
    df: pd.DataFrame,
    *,
    instrument_name: str,
    ins: InstrumentCfg,
    base_tf: str,
    tz: str = "America/New_York",
    qcfg: Optional[QualityCfg] = None,
    load_notes: Optional[dict[str, Any]] = None,
) -> DataQualityReport:
    """Inspect ``df`` (canonical frame) and return a :class:`DataQualityReport`."""
    qcfg = qcfg or QualityCfg()
    notes = load_notes or {}
    issues: list[Issue] = []
    n = len(df)
    idx = df.index
    flags: dict[str, np.ndarray] = {}
    step = tf_ns(base_tf)

    def add(kind: str, sev: Severity, count: int, detail: str, examples: Optional[list[str]] = None) -> None:
        if count:
            issues.append(Issue(kind, sev, int(count), detail, examples or []))

    # -- load-time observations ------------------------------------------------
    add("unparseable_timestamps", "error", notes.get("unparseable_or_ambiguous_timestamps", 0),
        "rows dropped at load: unparseable, or ambiguous/non-existent in the file's timezone")
    if notes.get("was_unsorted"):
        add("unsorted_input", "warning", 1, "file was not in time order (re-sorted, duplicates kept)")

    # -- duplicates ------------------------------------------------------------
    dup = idx.duplicated(keep=False)
    flags["duplicate"] = dup
    add("duplicate_timestamps", "error", int(pd.Index(idx[dup]).nunique()),
        "timestamps appearing more than once (resampling refuses them)", _ex(idx, dup))

    # -- NaN / non-positive / OHLC consistency ---------------------------------
    sets = _bar_sets(df)
    nan = np.zeros(n, bool)
    nonpos = np.zeros(n, bool)
    incons = np.zeros(n, bool)
    for p in sets:
        o, h, l, c = (df[f"{p}{f}"].to_numpy(float) for f in PRICE_FIELDS)
        nan |= np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c)
        with np.errstate(invalid="ignore"):
            nonpos |= (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
            incons |= (h < l) | (h < np.maximum(o, c)) | (l > np.minimum(o, c))
    flags["nan"], flags["nonpositive"], flags["ohlc_inconsistent"] = nan, nonpos, incons
    add("nan_prices", "error", nan.sum(), "bars with NaN in an OHLC field", _ex(idx, nan))
    add("nonpositive_prices", "error", nonpos.sum(), "bars with price <= 0", _ex(idx, nonpos))
    add("ohlc_inconsistent", "error", incons.sum(),
        "high < low, or high/low not enclosing open/close", _ex(idx, incons))

    # -- quotes ----------------------------------------------------------------
    stats: dict[str, Any] = {}
    if "bid_close" in df and "ask_close" in df:
        crossed = np.zeros(n, bool)
        with np.errstate(invalid="ignore"):
            for f in PRICE_FIELDS:
                crossed |= df[f"bid_{f}"].to_numpy(float) > df[f"ask_{f}"].to_numpy(float)
        flags["crossed_quotes"] = crossed
        add("crossed_quotes", "error", crossed.sum(), "bid > ask", _ex(idx, crossed))
        sp = (df["ask_close"] - df["bid_close"]).to_numpy(float)
        med = float(np.nanmedian(sp)) if n else 0.0
        wide = (sp > qcfg.wide_spread_mult * med) if med > 0 else np.zeros(n, bool)
        flags["wide_spread"] = np.nan_to_num(wide, nan=False).astype(bool)
        add("wide_spread", "warning", int(flags["wide_spread"].sum()),
            f"spread > {qcfg.wide_spread_mult:g} x median ({med:.6g})", _ex(idx, flags["wide_spread"]))
        stats["spread"] = {"median": med, "p95": float(np.nanpercentile(sp, 95)) if n else 0.0,
                           "max": float(np.nanmax(sp)) if n else 0.0}

    # -- bad ticks: isolated reversal spikes on mid close ----------------------
    if n > 30:
        c = df["close"].to_numpy(float)
        rng = pd.Series(df["high"].to_numpy(float) - df["low"].to_numpy(float))
        med_rng = rng.rolling(qcfg.spike_window, min_periods=20).median().shift(1).to_numpy()
        jump = np.abs(np.diff(c, prepend=np.nan))
        back = np.full(n, np.nan)
        back[1:-1] = np.abs(c[2:] - c[:-2])                   # back[i] = |c[i+1] - c[i-1]|
        with np.errstate(invalid="ignore"):
            spike = (jump > qcfg.spike_mult * med_rng) & (back < 0.5 * jump)
        flags["bad_tick"] = np.nan_to_num(spike, nan=False).astype(bool)
        add("bad_tick_spikes", "warning", int(flags["bad_tick"].sum()),
            f"close jumps > {qcfg.spike_mult:g} x median range and reverses next bar (reported, NOT removed)",
            _ex(idx, flags["bad_tick"]))

    # -- gaps ------------------------------------------------------------------
    gap_stats = {"weekend": 0, "daily_break": 0, "long_gap_weekday": 0, "missing_bars": 0}
    missing_total = 0
    ex_long: list[str] = []
    ex_missing: list[str] = []
    if n > 1:
        ns = to_ns(idx)
        delta = np.diff(ns)
        gi = np.nonzero(delta > step)[0]
        loc = local_naive_ns(idx, tz)
        anchor_min = parse_hhmm(ins.day_anchor_ny)
        for i in gi:
            gs, ge = ns[i] + step, ns[i + 1]                 # first missing bar open .. next bar open
            dur = ge - gs
            miss = int(dur // step)
            d0, d1 = loc[i] // NS_DAY, loc[i + 1] // NS_DAY
            has_sat = any(((d + 4) % 7) == 6 for d in range(int(d0), int(d1) + 1)) and dur >= 20 * 60 * NS_MIN
            start_min = int((loc[i] + step) % NS_DAY // NS_MIN)
            if has_sat:
                gap_stats["weekend"] += 1
            elif dur <= qcfg.daily_break_max_minutes * NS_MIN and abs(start_min - anchor_min) <= max(1, step // NS_MIN):
                gap_stats["daily_break"] += 1
            elif dur >= qcfg.long_gap_hours * 60 * NS_MIN:
                gap_stats["long_gap_weekday"] += 1
                ex_long.append(f"{pd.Timestamp(gs, unit='ns', tz='UTC')} (+{dur / (60 * NS_MIN):.1f}h)")
            else:
                gap_stats["missing_bars"] += 1
                missing_total += miss
                ex_missing.append(f"{pd.Timestamp(gs, unit='ns', tz='UTC')} ({miss} bars)")
    add("weekend_gaps", "info", gap_stats["weekend"], "expected weekly closures")
    add("daily_break_gaps", "info", gap_stats["daily_break"], "expected daily maintenance break")
    add("long_weekday_gaps", "warning", gap_stats["long_gap_weekday"],
        f"weekday gaps >= {qcfg.long_gap_hours:g}h (holiday or outage)", ex_long[:3])
    add("missing_bars", "warning", missing_total,
        f"missing bars inside {gap_stats['missing_bars']} short gaps (illiquid periods or feed loss)", ex_missing[:3])
    stats["gaps"] = gap_stats
    stats["missing_bars_total"] = missing_total

    # -- DST transitions in range (UTC storage keeps these unambiguous) --------
    if n > 1:
        off = ((local_naive_ns(idx, tz) - to_ns(idx)) // NS_MIN).astype(int)
        ch = np.nonzero(np.diff(off) != 0)[0]
        notes_dst = []
        for i in ch:
            lo, hi = idx[i] - pd.Timedelta(days=3), idx[i + 1] + pd.Timedelta(days=3)
            nb = int(((idx >= lo) & (idx <= hi)).sum())
            notes_dst.append(f"{idx[i + 1].date()}: UTC{off[i] / 60:+.0f}->{off[i + 1] / 60:+.0f} ({nb} bars within +/-3d)")
        add("dst_transitions", "info", len(ch), f"{tz} DST changes inside the data (sessions are DST-aware)", notes_dst[:6])
        stats["dst_transitions"] = notes_dst

    if "volume" in df and df["volume"].isna().all():
        add("no_volume", "info", 1, "volume column empty (not used by the strategy)")

    stats["spread_modeled"] = bool(df.attrs.get("spread_modeled", False))
    flag_df = pd.DataFrame(flags, index=idx)
    flag_df = flag_df[flag_df.any(axis=1)] if len(flag_df.columns) else flag_df
    rep = DataQualityReport(
        instrument_name, base_tf, n,
        str(idx[0]) if n else "", str(idx[-1]) if n else "",
        bool(df.attrs.get("spread_modeled", False)), issues, stats, flag_df,
    )
    log.info("data quality for %s: %d bars, %d errors, %d warnings",
             instrument_name, n, len(rep.errors), sum(i.severity == "warning" for i in issues))
    return rep


def enforce(report: DataQualityReport, qcfg: QualityCfg) -> None:
    """Raise :class:`DataQualityError` if the report has errors and the config says to fail."""
    if qcfg.fail_on_error and not report.ok:
        raise DataQualityError(
            "data quality errors: " + "; ".join(f"{i.kind} x{i.count}" for i in report.errors)
        )


def drop_flagged(df: pd.DataFrame, report: DataQualityReport, kinds: Iterable[str]) -> pd.DataFrame:
    """EXPLICITLY remove bars flagged with any of ``kinds`` (logged; never automatic).

    For ``duplicate`` the first occurrence is kept.
    """
    kinds = list(kinds)
    out = df
    removed = 0
    for k in kinds:
        if k not in report.flags.columns:
            continue
        bad = report.flags.index[report.flags[k]]
        if k == "duplicate":
            keep_first = ~out.index.duplicated(keep="first")
            removed += int((~keep_first).sum())
            out = out[keep_first]
        else:
            mask = out.index.isin(bad)
            removed += int(mask.sum())
            out = out[~mask]
    log.warning("drop_flagged(%s) removed %d bars", kinds, removed)
    return out
