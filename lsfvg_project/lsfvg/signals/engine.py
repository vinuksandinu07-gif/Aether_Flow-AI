"""Signal engine - the single strategy implementation used by backtest, paper and live.

``prepare`` computes causal features once (ATR, swings, pools, filter inputs).
``scan(prepared, start_idx, end_idx)`` evaluates every sweep -> MSS -> FVG chain using only bars
``<= end_idx``; a live loop calls it with ``end_idx`` = last closed MTF bar and keeps setups whose
``signal_idx`` is new.  Truncating the data at any bar never changes a setup already emitted
(see ``tests/test_no_lookahead.py``).
"""
from __future__ import annotations

import json
import logging
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config.schema import AppConfig, InstrumentCfg
from ..core.timeutil import to_ns
from ..core.types import EntryOrder, Pool, Setup, StopResult, TargetPlan
from ..data.news import blackout_windows
from ..data.resample import TimeframeStack
from ..data.spread import SpreadModel
from . import bias as bias_mod
from .filters import FilterInputs, failed_filters, session_mask
from .fvg import entry_price, find_fvgs, is_mitigated, select_fvg
from .indicators import atr_prior, atr_wilder, efficiency_ratio, rolling_percentile
from .pools import active_pools, build_pools
from .stops import compute_stop, round_to_tick
from .structure import find_mss
from .sweep import detect_sweep
from .swings import SwingSet, find_swings
from .targets import compute_targets, cost_estimate

log = logging.getLogger(__name__)


@dataclass
class Prepared:
    """Everything derived once from a :class:`TimeframeStack` (all causal)."""

    stack: TimeframeStack
    name: str
    ins: InstrumentCfg
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    open_ns: np.ndarray
    close_ns: np.ndarray
    atr: np.ndarray
    atr_prior: np.ndarray
    sw_mtf: tuple[SwingSet, SwingSet]
    sw_htf: tuple[SwingSet, SwingSet]
    pools: list[Pool]
    spread: np.ndarray
    filt: FilterInputs
    htf_idx: np.ndarray

    @property
    def n(self) -> int:
        return len(self.c)


@dataclass
class ScanResult:
    setups: list[Setup] = field(default_factory=list)          # valid, top-ranked, chronological
    skipped: list[Setup] = field(default_factory=list)         # rule-complete but rejected (with reasons)
    rejected_sweeps: list[dict] = field(default_factory=list)  # sweeps that never became a setup

    @property
    def all_setups(self) -> list[Setup]:
        return sorted(self.setups + self.skipped, key=lambda s: (s.signal_idx, s.setup_id))

    def skip_counts(self) -> dict[str, int]:
        """How many setups each reason removed (primary reason = first listed)."""
        out: dict[str, int] = {}
        for s in self.skipped:
            r = s.reasons[0] if s.reasons else "unknown"
            out[r] = out.get(r, 0) + 1
        return out


class SignalEngine:
    def __init__(self, cfg: AppConfig, instrument: str, news: Optional[pd.DataFrame] = None) -> None:
        self.cfg = cfg
        self.name = instrument
        self.ins: InstrumentCfg = cfg.instruments[instrument]
        self.news = news

    # ------------------------------------------------------------------ prepare
    def prepare(self, stack: TimeframeStack) -> Prepared:
        cfg, ins, sig = self.cfg, self.ins, self.cfg.signals
        mtf, htf = stack.mtf, stack.htf
        o, h, l, c = (mtf[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        open_ns = to_ns(mtf.index)
        close_ns = to_ns(pd.DatetimeIndex(mtf["close_time"]))
        atr = atr_wilder(h, l, c, sig.atr_period)
        atr_h = atr_wilder(htf["high"].to_numpy(float), htf["low"].to_numpy(float),
                           htf["close"].to_numpy(float), sig.atr_period) if len(htf) else np.empty(0)
        sw_m = find_swings(h, l, sig.mtf_swing_n)
        sw_h = find_swings(htf["high"].to_numpy(float), htf["low"].to_numpy(float), sig.htf_swing_n) \
            if len(htf) else (SwingSet(np.empty(0, int), np.empty(0), np.empty(0, int), sig.htf_swing_n),) * 2
        pools = build_pools(mtf, htf, tz=stack.tz, ins=ins, tfs=cfg.timeframes, pools_cfg=cfg.pools, sig=sig,
                            atr_mtf=atr, atr_htf=atr_h, sw_mtf=sw_m, sw_htf=sw_h)
        if "bid_close" in mtf and "ask_close" in mtf:
            spread = (mtf["ask_close"] - mtf["bid_close"]).to_numpy(float)
        else:
            spread = SpreadModel(ins, stack.tz).at(mtf.index)
        f = cfg.filters
        ns, ne = (np.empty(0, "int64"), np.empty(0, "int64"))
        if f.news.enabled and self.news is not None:
            ns, ne = blackout_windows(self.news, ins.news_currencies, f.news.min_impact,
                                      f.news.before_min, f.news.after_min)
        filt = FilterInputs(
            session_ok=session_mask(mtf.index, ins.entry_sessions, stack.tz),
            spread=spread,
            atr_pct=rolling_percentile(atr, f.volatility.lookback_bars),
            er=efficiency_ratio(c, f.chop.er_period),
            close_ns=close_ns, news_starts=ns, news_ends=ne)
        return Prepared(stack, self.name, ins, o, h, l, c, open_ns, close_ns, atr, atr_prior(atr), sw_m, sw_h,
                        pools, spread, filt, stack.htf_idx_at_mtf)

    # --------------------------------------------------------------------- scan
    def scan(self, p: Prepared, start_idx: int = 0, end_idx: Optional[int] = None) -> ScanResult:
        """Find all setups whose ``signal_idx`` is in ``[start_idx, end_idx]`` using bars ``<= end_idx`` only."""
        sig = self.cfg.signals
        end = p.n - 1 if end_idx is None else min(end_idx, p.n - 1)
        res = ScanResult()
        cands: list[Setup] = []
        for pool in p.pools:
            if pool.swept_idx is None or pool.swept_idx > end:
                continue
            sw = detect_sweep(p.h, p.l, p.c, p.atr_prior, pool, min_sweep_atr=sig.min_sweep_atr,
                              reclaim_bars=sig.reclaim_bars, end_idx=end)
            if sw is None:
                continue
            if sw.reclaim_idx + sig.mss_max_bars + 1 < start_idx:
                continue
            opposing = p.sw_mtf[1] if sw.direction == "short" else p.sw_mtf[0]
            mss, why = find_mss(p.o, p.h, p.l, p.c, p.atr_prior, sw, opposing, disp_mult=sig.disp_mult,
                                max_bars=sig.mss_max_bars, end_idx=end)
            if mss is None:
                if why != "pending":
                    res.rejected_sweeps.append({"pool_id": pool.pool_id, "sweep_idx": sw.reclaim_idx, "reason": why})
                continue
            chosen = None
            for s in (mss.idx, mss.idx + 1):
                if s > end:
                    break
                fv = find_fvgs(p.h, p.l, p.atr_prior, sw.direction, lo_mid=sw.extreme_idx, hi_mid=mss.idx,
                               min_atr=sig.fvg_min_atr, end_idx=s)
                fv = [f for f in fv if not is_mitigated(f, sw.direction, entry_price(f, sw.direction, sig.entry_mode),
                                                         p.h, p.l, p.c, s)]
                pick = select_fvg(fv, sig.fvg_select)
                if pick is not None:
                    chosen = (pick, s)
                    break
            if chosen is None:
                if mss.idx + 1 <= end:
                    res.rejected_sweeps.append({"pool_id": pool.pool_id, "sweep_idx": sw.reclaim_idx, "reason": "no_fvg"})
                continue
            fvg, s = chosen
            if s < start_idx:
                continue
            cands.append(self._build_setup(p, pool, sw, mss, fvg, s))
        self._rank(cands, res)
        return res

    # ------------------------------------------------------------------ helpers
    def _build_setup(self, p: Prepared, pool: Pool, sw, mss, fvg, s: int) -> Setup:
        cfg, ins = self.cfg, self.ins
        d = sw.direction
        tick = ins.tick_size
        entry = round_to_tick(entry_price(fvg, d, cfg.signals.entry_mode), tick, "nearest")
        atr_s = float(p.atr[s])
        spread_s = float(p.spread[s])
        reasons: list[str] = []

        # HTF bias (latest CLOSED HTF bar only)
        hi = int(p.htf_idx[s])
        rng = bias_mod.dealing_range(p.sw_htf[0], p.sw_htf[1], hi)
        ref_price = entry if cfg.bias.reference == "entry" else float(p.c[s])
        zone = bias_mod.classify_price(rng, ref_price) if rng else "none"
        aligned = bias_mod.trend_aligned(d, bias_mod.htf_trend(p.sw_htf[0], p.sw_htf[1], hi))
        if cfg.bias.enabled:
            if rng is None:
                reasons.append("bias_no_range")
            elif not bias_mod.zone_allows(d, zone):
                reasons.append("bias_wrong_zone")
        if cfg.bias.htf_trend_filter and not aligned:
            reasons.append("htf_trend_misaligned")

        reasons += failed_filters(s, p.filt, cfg.filters, ins.max_spread)

        # sweep extreme up to the signal bar (stop must clear everything traded since the breach)
        seg = slice(sw.breach_idx, s + 1)
        extreme = float(p.h[seg].max()) if d == "short" else float(p.l[seg].min())
        if not np.isfinite(atr_s) or atr_s <= 0:
            stop_res = StopResult(None, 0.0, 0.0, extreme, 0.0, "skipped", "atr_unavailable")
        else:
            stop_res = compute_stop(d, entry, sweep_extreme=extreme, fvg_first_high=fvg.first_high,
                                    fvg_first_low=fvg.first_low, atr_mtf=atr_s, spread=spread_s,
                                    cfg=cfg.stops, tick=tick)
        if stop_res.stop is None:
            reasons.append(stop_res.reason)
            tcfg = cfg.targets
            plan = TargetPlan(tcfg.tp_mode, None, None, None, None, tcfg.partial_pct, None, 0.0, None, None,
                              tcfg.time_stop_bars, tcfg.time_stop_r, 0.0, 0.0, "not_evaluated")
            stop_price = stop_res.structural_stop
        else:
            stop_price = stop_res.stop
            plan = compute_targets(d, entry, stop_price, active_pools(p.pools, s), cfg.targets, cfg.pools,
                                   cost_estimate(ins, spread_s), tick)
            if plan.skip_reason:
                reasons.append(plan.skip_reason)

        tgt_dist = abs(plan.reference_target - entry) if plan.reference_target is not None else np.inf
        rank_key = (0 if aligned else 1, tgt_dist, -fvg.size / atr_s if atr_s else 0.0, pool.pool_id)
        sig_time = pd.Timestamp(p.stack.mtf["close_time"].iloc[s])
        sid = f"{self.name}-{sig_time:%Y%m%dT%H%MZ}-{d[0].upper()}-{zlib.crc32(pool.pool_id.encode()) & 0xFFFFFF:06x}"
        meta = {"dealing_range": (rng.high, rng.low) if rng else None, "extreme_final": extreme,
                "pool_kind": pool.kind, "pool_level": pool.level, "pool_tf": pool.tf, "breach_idx": sw.breach_idx}
        return Setup(sid, self.name, d, pool.pool_id, sw, mss, fvg, s, sig_time, entry, stop_price, stop_res, plan,
                     atr_s, spread_s, zone, aligned, "skipped" if reasons else "valid", reasons, rank_key, meta)

    @staticmethod
    def _rank(cands: list[Setup], res: ScanResult) -> None:
        """Per signal bar keep ONLY the best valid setup (HTF aligned > nearest target > larger FVG)."""
        by_bar: dict[int, list[Setup]] = {}
        for sc in cands:
            by_bar.setdefault(sc.signal_idx, []).append(sc)
        for _, group in sorted(by_bar.items()):
            rule_skipped = [g for g in group if g.status == "skipped"]
            valid = sorted((g for g in group if g.status == "valid"), key=lambda g: g.rank_key)
            for k, g in enumerate(valid):
                if k == 0:
                    res.setups.append(g)
                else:
                    g.status, g.reasons = "skipped", ["lower_priority"]
                    res.skipped.append(g)
            res.skipped.extend(rule_skipped)

    # ------------------------------------------------------------- order / log
    def entry_order(self, setup: Setup) -> EntryOrder:
        """Limit-order spec for a valid setup (cancel-if-target-taken uses the TP reference pool)."""
        tp = setup.targets
        tgt = tp.reference_target if tp.final_pool_id is not None else None
        sg = self.cfg.signals
        return EntryOrder(setup.setup_id, setup.direction, setup.entry_price, setup.signal_time, sg.max_wait_bars,
                          setup.fvg.top, setup.fvg.bottom, tgt, self.ins.tick_size, sg.fill_through_ticks)


def decision_records(res: ScanResult) -> list[dict]:
    """One row per setup / rejected sweep saying WHY it was taken or skipped."""
    rows = []
    for s in res.all_setups:
        d = s.to_dict()
        d["decision"] = "TAKE" if s.status == "valid" else "SKIP"
        d["why"] = "all rules + filters + stop caps + min_rr passed" if s.status == "valid" else ";".join(s.reasons)
        rows.append(d)
    for r in res.rejected_sweeps:
        rows.append({"decision": "NO_SETUP", "why": r["reason"], "pool_id": r["pool_id"], "sweep_idx": r["sweep_idx"]})
    return rows


def write_decision_log(res: ScanResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in decision_records(res):
            fh.write(json.dumps(row, default=str) + "\n")
    return path
