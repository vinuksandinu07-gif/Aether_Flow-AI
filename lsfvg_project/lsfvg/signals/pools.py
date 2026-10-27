"""Liquidity pools.

Buy-side pools sit above price (highs), sell-side below (lows).  Each pool records

* ``known_time`` / ``valid_idx`` - when it first exists for the strategy (no lookahead),
* ``swept_idx`` - first MTF bar that traded strictly through the level,
* ``expire_idx`` - age limit.

``Pool.active_at(i)`` therefore answers "is this pool usable and unswept at the close of MTF
bar i" using only information available at that time.  Pools from periods that have not
finished (current day / week / session) do not exist yet; the first, possibly partial,
period of the dataset is dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.schema import InstrumentCfg, PoolsCfg, SignalsCfg, TimeframesCfg, tf_minutes
from ..core.timeutil import NS_DAY, session_instance_ids, to_ns, trading_day_ids, trading_week_ids
from ..core.types import Pool
from .swings import SwingSet

_EPS = 1e-12


def _mk(kind, side, level, tf, scope, formed, known, valid_idx, expire_idx, meta=None) -> Pool:
    pid = f"{kind}|{tf}|{pd.Timestamp(known).strftime('%Y%m%dT%H%M')}|{level:.6g}"
    return Pool(pid, kind, side, float(level), tf, scope, pd.Timestamp(formed), pd.Timestamp(known),
                int(valid_idx), int(expire_idx), None, meta or {})


def _groups(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    change = np.flatnonzero(np.diff(ids) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(ids)]
    return starts, ends


def _period_pools(mtf, ids, kind_hi, kind_lo, label, max_age_days, open_ns) -> list[Pool]:
    high, low = mtf["high"].to_numpy(float), mtf["low"].to_numpy(float)
    close_t = mtf["close_time"]
    starts, ends = _groups(ids)
    out: list[Pool] = []
    for g in range(1, len(starts) - 1):               # skip first (maybe partial) and last (incomplete)
        s, e = starts[g], ends[g]
        known_idx = starts[g + 1]
        known = mtf.index[known_idx]
        exp = int(np.searchsorted(open_ns, int(to_ns(pd.DatetimeIndex([known]))[0]) + max_age_days * NS_DAY, "left"))
        formed = close_t.iloc[e - 1]
        meta = {"period_bars": (int(s), int(e))}
        out.append(_mk(kind_hi, "buy", high[s:e].max(), label, "external", formed, known, known_idx, exp, meta))
        out.append(_mk(kind_lo, "sell", low[s:e].min(), label, "external", formed, known, known_idx, exp, meta))
    return out


def _session_pools(mtf, ins: InstrumentCfg, tz, max_age_days, open_ns) -> list[Pool]:
    high, low = mtf["high"].to_numpy(float), mtf["low"].to_numpy(float)
    close_t = mtf["close_time"]
    n = len(mtf)
    out: list[Pool] = []
    for w in ins.pool_sessions:
        ids = session_instance_ids(mtf.index, tz, w.start, w.end)
        pos = np.flatnonzero(ids >= 0)
        if len(pos) < 2:
            continue
        starts, ends = _groups(ids[pos])
        for g in range(1, len(starts)):                  # skip the first (maybe partial) instance
            seg = pos[starts[g]:ends[g]]
            last = int(seg[-1])
            if last + 1 >= n:                            # instance not finished yet
                continue
            known = close_t.iloc[last]
            kn_ns = int(to_ns(pd.DatetimeIndex([known]))[0])
            exp = int(np.searchsorted(open_ns, kn_ns + max_age_days * NS_DAY, "left"))
            tf = f"session:{w.name}"
            meta = {"session": w.name, "period_bars": (int(seg[0]), last + 1)}
            out.append(_mk("SESS_H", "buy", high[seg].max(), tf, "external", known, known, last + 1, exp, meta))
            out.append(_mk("SESS_L", "sell", low[seg].min(), tf, "external", known, known, last + 1, exp, meta))
    return out


def equal_level_pools(
    side: str,
    sw: SwingSet,
    *,
    high_tf: np.ndarray,
    low_tf: np.ndarray,
    atr_tf: np.ndarray,
    open_ns_tf: np.ndarray,
    close_ns_tf: np.ndarray,
    open_ns_mtf: np.ndarray,
    mtf_times: pd.DatetimeIndex,
    tol_atr: float,
    max_sep: int,
    lookback: int,
    label: str,
    scope: str,
    age_bars_mtf: int,
) -> list[Pool]:
    """Equal highs (``side='buy'``) / lows (``'sell'``): 2+ swings within ``tol_atr`` x ATR.

    The level is the extreme of the cluster; a higher (lower) bar between the two swings
    disqualifies the pair (the "equal" high was already exceeded).  A pool is known from
    the close of the bar that confirms the later swing.
    """
    out: list[Pool] = []
    n_mtf = len(open_ns_mtf)
    for j in range(len(sw)):
        pj, price_j, conf_j = int(sw.idx[j]), float(sw.price[j]), int(sw.confirm_idx[j])
        if conf_j >= len(atr_tf) or np.isnan(atr_tf[conf_j]):
            continue
        tol = tol_atr * atr_tf[conf_j]
        for k in range(j - 1, max(-1, j - 1 - lookback), -1):
            pk, price_k = int(sw.idx[k]), float(sw.price[k])
            if pj - pk > max_sep:
                break
            if abs(price_j - price_k) > tol:
                continue
            if side == "buy":
                level = max(price_j, price_k)
                if high_tf[pk:pj + 1].max() > level + _EPS:
                    continue
            else:
                level = min(price_j, price_k)
                if low_tf[pk:pj + 1].min() < level - _EPS:
                    continue
            known_ns = int(close_ns_tf[conf_j])
            vidx = int(np.searchsorted(open_ns_mtf, known_ns, "left"))
            if vidx >= n_mtf:
                break
            kind = "EQH" if side == "buy" else "EQL"
            known = pd.Timestamp(known_ns, unit="ns", tz="UTC")
            formed = pd.Timestamp(int(open_ns_tf[pj]), unit="ns", tz="UTC")
            meta = {"tol": float(tol), "pivots_tf_idx": (pk, pj), "tf": label}
            out.append(_mk(kind, side, level, label, scope, formed, known, vidx, vidx + age_bars_mtf, meta))
            break
    return out


def _set_swept(pools: list[Pool], high: np.ndarray, low: np.ndarray) -> None:
    n = len(high)
    for p in pools:
        hi = min(p.expire_idx, n)
        if p.valid_idx >= hi:
            p.swept_idx = None
            continue
        seg = high[p.valid_idx:hi] > p.level if p.side == "buy" else low[p.valid_idx:hi] < p.level
        hit = np.flatnonzero(seg)
        p.swept_idx = p.valid_idx + int(hit[0]) if hit.size else None


def _dedupe_equal(pools: list[Pool]) -> list[Pool]:
    """Drop an equal-level pool when a same-side one within tolerance is already live."""
    keep: list[Pool] = []
    live: list[Pool] = []
    for p in sorted(pools, key=lambda x: (x.valid_idx, x.pool_id)):
        if p.kind in ("EQH", "EQL"):
            dup = False
            for q in live:
                if (q.side == p.side and q.tf == p.tf and abs(q.level - p.level) <= max(p.meta.get("tol", 0), q.meta.get("tol", 0))
                        and q.active_at(p.valid_idx)):
                    dup = True
                    break
            if dup:
                continue
            live.append(p)
        keep.append(p)
    return keep


def build_pools(
    mtf: pd.DataFrame,
    htf: pd.DataFrame,
    *,
    tz: str,
    ins: InstrumentCfg,
    tfs: TimeframesCfg,
    pools_cfg: PoolsCfg,
    sig: SignalsCfg,
    atr_mtf: np.ndarray,
    atr_htf: np.ndarray,
    sw_mtf: tuple[SwingSet, SwingSet],
    sw_htf: tuple[SwingSet, SwingSet],
) -> list[Pool]:
    """Build every configured pool type for the whole dataset (each gated by ``valid_idx``)."""
    open_ns = to_ns(mtf.index)
    close_ns = to_ns(pd.DatetimeIndex(mtf["close_time"]))
    mh, ml = mtf["high"].to_numpy(float), mtf["low"].to_numpy(float)
    pools: list[Pool] = []

    day_ids = trading_day_ids(mtf.index, tz, ins.day_anchor_ny)
    if pools_cfg.use_pdh_pdl:
        pools += _period_pools(mtf, day_ids, "PDH", "PDL", "D", pools_cfg.daily_max_age_days, open_ns)
    if pools_cfg.use_pwh_pwl:
        pools += _period_pools(mtf, trading_week_ids(day_ids), "PWH", "PWL", "W",
                               pools_cfg.weekly_max_age_weeks * 7, open_ns)
    if pools_cfg.use_sessions and ins.pool_sessions:
        pools += _session_pools(mtf, ins, tz, pools_cfg.session_max_age_days, open_ns)

    if pools_cfg.use_equal_mtf:
        for side, sw in (("buy", sw_mtf[0]), ("sell", sw_mtf[1])):
            pools += equal_level_pools(
                side, sw, high_tf=mh, low_tf=ml, atr_tf=atr_mtf, open_ns_tf=open_ns, close_ns_tf=close_ns,
                open_ns_mtf=open_ns, mtf_times=mtf.index, tol_atr=sig.eq_tol_atr, max_sep=sig.eq_max_sep_bars,
                lookback=sig.eq_lookback_swings, label="mtf", scope="internal", age_bars_mtf=pools_cfg.eq_max_age_bars)
    if pools_cfg.use_equal_htf and len(htf):
        ratio = max(1, tf_minutes(tfs.htf) // tf_minutes(tfs.mtf))
        ho_ns = to_ns(htf.index)
        hc_ns = to_ns(pd.DatetimeIndex(htf["close_time"]))
        hh, hl = htf["high"].to_numpy(float), htf["low"].to_numpy(float)
        for side, sw in (("buy", sw_htf[0]), ("sell", sw_htf[1])):
            pools += equal_level_pools(
                side, sw, high_tf=hh, low_tf=hl, atr_tf=atr_htf, open_ns_tf=ho_ns, close_ns_tf=hc_ns,
                open_ns_mtf=open_ns, mtf_times=mtf.index, tol_atr=sig.eq_tol_atr, max_sep=sig.eq_max_sep_bars,
                lookback=sig.eq_lookback_swings, label="htf", scope="external",
                age_bars_mtf=pools_cfg.eq_max_age_bars * ratio)

    pools = [p for p in pools if p.valid_idx < len(mtf)]
    _set_swept(pools, mh, ml)
    pools = _dedupe_equal(pools)
    seen: dict[str, int] = {}
    for p in pools:                                     # guarantee unique ids
        c = seen.get(p.pool_id, 0)
        seen[p.pool_id] = c + 1
        if c:
            p.pool_id = f"{p.pool_id}#{c}"
    pools.sort(key=lambda p: (p.valid_idx, p.pool_id))
    return pools


def active_pools(pools: list[Pool], idx: int) -> list[Pool]:
    """Pools usable and unswept as of the close of MTF bar ``idx``."""
    return [p for p in pools if p.active_at(idx)]
