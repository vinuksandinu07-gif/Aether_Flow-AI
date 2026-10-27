"""Take-profit planning and cost-adjusted RR (pure)."""
from __future__ import annotations

from typing import Optional

from ..config.schema import InstrumentCfg, PoolsCfg, TargetsCfg
from ..core.types import CostEstimate, Pool, TargetPlan
from .stops import round_to_tick


def cost_estimate(ins: InstrumentCfg, spread: float) -> CostEstimate:
    """One round trip: spread + entry slippage + exit slippage + commission (price units)."""
    return CostEstimate(spread, ins.slippage_entry, ins.slippage_stop, ins.commission_per_unit_rt)


def effective_rr(reward_dist: float, risk: float, costs: CostEstimate) -> float:
    """``(reward - C) / (risk + C)`` with ``C`` = one round-trip cost (conservative on both legs)."""
    c = costs.round_trip
    den = risk + c
    return (reward_dist - c) / den if den > 0 else 0.0


def breakeven_win_rate(avg_effective_rr: float) -> float:
    """Win rate at which expectancy is zero: ``1 / (1 + RR)``."""
    return 1.0 / (1.0 + avg_effective_rr) if avg_effective_rr > -1 else 1.0


def nearest_opposing_pool(
    direction: str, entry: float, pools: list[Pool], scope: str = "any"
) -> Optional[Pool]:
    """Nearest unswept pool beyond entry in the trade direction (``pools`` must already be active)."""
    long = direction == "long"
    c = [p for p in pools if (p.side == "buy" and p.level > entry)] if long else \
        [p for p in pools if (p.side == "sell" and p.level < entry)]
    if scope == "external":
        c = [p for p in c if p.scope == "external"]
    if not c:
        return None
    return min(c, key=lambda p: p.level) if long else max(c, key=lambda p: p.level)


def compute_targets(
    direction: str,
    entry: float,
    stop: float,
    active_pools: list[Pool],
    cfg: TargetsCfg,
    pools_cfg: PoolsCfg,
    costs: CostEstimate,
    tick: float,
) -> TargetPlan:
    """Build the TP plan for ``cfg.tp_mode`` and gate it on cost-adjusted ``min_rr``."""
    long = direction == "long"
    sgn = 1.0 if long else -1.0
    risk = abs(entry - stop)
    be_off = cfg.be_offset_r * risk

    def blank(reason: str) -> TargetPlan:
        return TargetPlan(cfg.tp_mode, None, None, None, None, cfg.partial_pct, None, be_off, None, None,
                          cfg.time_stop_bars, cfg.time_stop_r, 0.0, 0.0, reason)

    def fixed_target() -> float:
        return round_to_tick(entry + sgn * cfg.fixed_rr * risk, tick, "down" if long else "up")

    final: Optional[float] = None
    final_pid: Optional[str] = None
    partial_price: Optional[float] = None
    partial_r: Optional[float] = None
    trail_act: Optional[float] = None
    trail_mode: Optional[str] = None
    reward_dist: float

    if cfg.tp_mode == "fixed_rr":
        final = fixed_target()
        reward_dist = abs(final - entry)
    elif cfg.tp_mode == "opposing_liquidity":
        p = nearest_opposing_pool(direction, entry, active_pools, pools_cfg.tp_scope)
        if p is None:
            return blank("no_target_pool")
        final, final_pid = p.level, p.pool_id
        reward_dist = abs(final - entry)
    elif cfg.tp_mode == "partial_runner":
        p = nearest_opposing_pool(direction, entry, active_pools, "external")
        if p is None:
            return blank("no_target_pool")
        final, final_pid = p.level, p.pool_id
        final_dist = abs(final - entry)
        if cfg.partial_at == "internal_liquidity":
            internal = [q for q in active_pools if q.scope == "internal"]
            q = nearest_opposing_pool(direction, entry, internal, "any")
            cand = q.level if q is not None else entry + sgn * risk
        else:
            cand = entry + sgn * risk
        if abs(cand - entry) < final_dist:
            partial_price = round_to_tick(cand, tick, "down" if long else "up")
            partial_r = abs(partial_price - entry) / risk
            reward_dist = cfg.partial_pct * abs(partial_price - entry) + (1 - cfg.partial_pct) * final_dist
        else:                                          # target nearer than the partial: no partial
            reward_dist = final_dist
    else:                                              # trailing
        p = nearest_opposing_pool(direction, entry, active_pools, pools_cfg.tp_scope)
        ref = p.level if p is not None else fixed_target()
        final_pid = p.pool_id if p is not None else None
        trail_act = entry + sgn * cfg.trail_activation_r * risk
        trail_mode = cfg.trail_mode
        reward_dist = abs(ref - entry)
        planned = reward_dist / risk if risk > 0 else 0.0
        eff = effective_rr(reward_dist, risk, costs)
        return TargetPlan("trailing", None, final_pid, ref, None, cfg.partial_pct, None, be_off, trail_act,
                          trail_mode, cfg.time_stop_bars, cfg.time_stop_r, planned, eff,
                          "" if eff >= cfg.min_rr else "rr_below_min")

    planned = reward_dist / risk if risk > 0 else 0.0
    eff = effective_rr(reward_dist, risk, costs)
    return TargetPlan(cfg.tp_mode, final, final_pid, final, partial_price, cfg.partial_pct, partial_r, be_off,
                      None, None, cfg.time_stop_bars, cfg.time_stop_r, planned, eff,
                      "" if eff >= cfg.min_rr else "rr_below_min")
