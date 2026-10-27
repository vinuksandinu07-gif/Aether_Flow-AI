"""Plain data containers shared by signals, backtest, visualization and execution.

Index conventions
-----------------
* ``*_idx`` fields are positional indices into the **MTF** frame unless noted.
* ``*_time`` fields are tz-aware UTC timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import pandas as pd

Direction = Literal["long", "short"]


@dataclass
class Pool:
    """A liquidity pool (resting stops beyond an obvious high/low)."""

    pool_id: str
    kind: str                      # PDH PDL PWH PWL SESS_H SESS_L EQH EQL
    side: Literal["buy", "sell"]   # buy-side = above price (highs), sell-side = below (lows)
    level: float
    tf: str                        # origin label: D, W, session:<name>, htf, mtf
    scope: Literal["external", "internal"]
    formed_time: pd.Timestamp      # when the structure ended / last pivot formed
    known_time: pd.Timestamp       # first moment it may be used (no lookahead before this)
    valid_idx: int                 # first MTF bar allowed to interact with it
    expire_idx: int                # MTF bar at which it is dropped for age
    swept_idx: Optional[int] = None  # MTF bar of first trade-through (None = unswept)
    meta: dict[str, Any] = field(default_factory=dict)

    def active_at(self, idx: int) -> bool:
        """Usable and still unswept as of the close of MTF bar ``idx``."""
        return (
            self.valid_idx <= idx < self.expire_idx
            and (self.swept_idx is None or self.swept_idx > idx)
        )


@dataclass(frozen=True)
class Sweep:
    pool_id: str
    direction: Direction           # trade direction implied (buy-side sweep -> short)
    pool_level: float
    breach_idx: int                # first bar trading beyond the pool
    reclaim_idx: int               # bar that closed back inside
    extreme_idx: int               # bar holding the sweep extreme within breach..reclaim
    extreme_price: float
    depth: float                   # extreme - level (abs)


@dataclass(frozen=True)
class MSS:
    idx: int                       # displacement candle that closed beyond the swing
    swing_idx: int                 # pivot bar of the broken swing
    level: float                   # broken swing price
    body: float
    body_atr: float


@dataclass(frozen=True)
class FVG:
    direction: Literal["bull", "bear"]
    first_idx: int                 # candle i-2
    mid_idx: int                   # candle i-1 (displacement candle)
    last_idx: int                  # candle i (gap known at its close)
    bottom: float
    top: float
    first_high: float
    first_low: float

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass(frozen=True)
class CostEstimate:
    spread: float
    slippage_entry: float
    slippage_exit: float
    commission: float

    @property
    def round_trip(self) -> float:
        return self.spread + self.slippage_entry + self.slippage_exit + self.commission


@dataclass(frozen=True)
class StopResult:
    stop: Optional[float]
    distance: float
    distance_atr: float
    structural_stop: float
    buffer: float
    action: Literal["ok", "widened", "skipped"]
    reason: str = ""


@dataclass(frozen=True)
class TargetPlan:
    mode: str
    final_target: Optional[float]        # None for pure trailing
    final_pool_id: Optional[str]
    reference_target: Optional[float]    # target used for RR gating (== final unless trailing)
    partial_price: Optional[float]
    partial_pct: float
    partial_r: Optional[float]
    be_offset_price: float               # price distance added to breakeven after partial
    trail_activation_price: Optional[float]
    trail_mode: Optional[str]
    time_stop_bars: Optional[int]
    time_stop_r: float
    planned_rr: float                    # gross
    effective_rr: float                  # after estimated costs
    skip_reason: str = ""                # "" = acceptable


@dataclass
class Setup:
    """A fully specified, rule-valid trade idea (limit order + SL + TP plan)."""

    setup_id: str
    instrument: str
    direction: Direction
    pool_id: str
    sweep: Sweep
    mss: MSS
    fvg: FVG
    signal_idx: int                # MTF bar whose close completed the setup
    signal_time: pd.Timestamp      # == close time of that bar == order placement time
    entry_price: float
    stop_price: float
    stop: StopResult
    targets: TargetPlan
    atr_mtf: float
    spread_est: float
    bias_zone: str
    htf_aligned: bool
    status: Literal["valid", "skipped"] = "valid"
    reasons: list[str] = field(default_factory=list)   # skip reasons ([] when valid)
    rank_key: tuple = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop_price)

    def to_dict(self) -> dict[str, Any]:
        return {
            "setup_id": self.setup_id, "instrument": self.instrument,
            "direction": self.direction, "status": self.status,
            "reasons": ";".join(self.reasons), "pool_id": self.pool_id,
            "signal_time": self.signal_time.isoformat(),
            "sweep_extreme": self.sweep.extreme_price, "mss_level": self.mss.level,
            "fvg_bottom": self.fvg.bottom, "fvg_top": self.fvg.top,
            "entry": self.entry_price, "stop": self.stop_price,
            "target": self.targets.reference_target, "mode": self.targets.mode,
            "planned_rr": round(self.targets.planned_rr, 4),
            "effective_rr": round(self.targets.effective_rr, 4),
            "atr_mtf": self.atr_mtf, "bias_zone": self.bias_zone,
            "htf_aligned": self.htf_aligned,
        }


@dataclass(frozen=True)
class EntryOrder:
    """Limit order specification emitted by the signal engine (rule 7)."""

    setup_id: str
    direction: Direction
    limit_price: float
    place_time: pd.Timestamp
    max_wait_bars: int             # LTF bars
    fvg_top: float
    fvg_bottom: float
    target_level: Optional[float]  # external target; order cancelled if taken pre-fill
    tick_size: float
    fill_through_ticks: int


@dataclass(frozen=True)
class OrderOutcome:
    status: Literal["filled", "expired", "cancelled_fvg_through", "cancelled_target_taken", "pending"]
    bar_idx: Optional[int]         # LTF positional index where it resolved
    time: Optional[pd.Timestamp]   # open time of that LTF bar
    fill_price: Optional[float] = None


@dataclass(frozen=True)
class ExitFill:
    time: pd.Timestamp
    price: float
    fraction: float                # fraction of the *original* position closed
    reason: str                    # SL TP partial breakeven trailing time_stop session_end kill_switch


@dataclass
class TradeRecord:
    """Executed trade, as produced by the Phase 2 backtester (used by the visualizer)."""

    setup_id: str
    direction: Direction
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: Optional[float]
    exits: list[ExitFill] = field(default_factory=list)
    r_multiple: Optional[float] = None
    units: float = 0.0
    pnl: float = 0.0
    risk_cash: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0
