"""Account risk controls for Phase 2 backtests.

The risk manager is deliberately independent of signal generation and AI inference.
It sizes from stop distance, enforces account-level limits, and fails closed when
prices/equity are invalid.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import pandas as pd

from ..config.schema import AppConfig, InstrumentCfg


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    units: float = 0.0
    risk_cash: float = 0.0
    margin_required: float = 0.0


class RiskManager:
    def __init__(self, cfg: AppConfig, instrument: InstrumentCfg, initial_capital: float) -> None:
        self.cfg = cfg
        self.instrument = instrument
        self.balance = float(initial_capital)
        self.peak_equity = float(initial_capital)
        self.day_key: Optional[str] = None
        self.week_key: Optional[str] = None
        self.day_start_equity = float(initial_capital)
        self.week_start_equity = float(initial_capital)
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_loss_bar: Optional[int] = None
        self.halted = False
        self.halt_reason = ""

    @staticmethod
    def period_keys(timestamp: pd.Timestamp, anchor: str = "17:00") -> tuple[str, str]:
        t = pd.Timestamp(timestamp)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        local = t.tz_convert("America/New_York")
        hh, mm = (int(x) for x in anchor.split(":"))
        shifted = local - pd.Timedelta(hours=hh, minutes=mm)
        day_key = shifted.date().isoformat()
        week_start = shifted - pd.Timedelta(days=(shifted.weekday() + 1) % 7)
        return day_key, week_start.date().isoformat()

    def update_period(self, timestamp: pd.Timestamp, equity: float) -> None:
        day, week = self.period_keys(timestamp, self.instrument.day_anchor_ny)
        if day != self.day_key:
            self.day_key, self.day_start_equity, self.trades_today = day, float(equity), 0
        if week != self.week_key:
            self.week_key, self.week_start_equity = week, float(equity)
        self.peak_equity = max(self.peak_equity, float(equity))
        if self.peak_equity > 0 and (self.peak_equity - equity) / self.peak_equity >= self.cfg.risk.max_drawdown_pct:
            self.halted = True
            self.halt_reason = "max_drawdown_kill_switch"

    def entry_block_reason(self, timestamp: pd.Timestamp, bar_idx: int, equity: float) -> str:
        """Return a reason to block new entries, or an empty string."""
        self.update_period(timestamp, equity)
        if self.halted:
            return self.halt_reason or "account_halted"
        r = self.cfg.risk
        if self.day_start_equity > 0 and equity <= self.day_start_equity * (1 - r.daily_loss_limit_pct):
            return "daily_loss_limit"
        if self.week_start_equity > 0 and equity <= self.week_start_equity * (1 - r.weekly_loss_limit_pct):
            return "weekly_loss_limit"
        if self.trades_today >= r.max_trades_per_day:
            return "max_trades_per_day"
        if self.consecutive_losses >= r.max_consecutive_losses:
            if self.last_loss_bar is None or bar_idx - self.last_loss_bar < r.cooldown_bars_after_loss:
                return "consecutive_loss_cooldown"
        return ""

    def approve_entry(
        self, *, timestamp: pd.Timestamp, bar_idx: int, equity: float,
        entry: float, stop: float, open_risk_cash: float = 0.0,
    ) -> RiskDecision:
        r = self.cfg.risk
        self.update_period(timestamp, equity)
        if not all(math.isfinite(x) for x in (equity, entry, stop)) or equity <= 0 or entry <= 0:
            return RiskDecision(False, "invalid_equity_or_price")
        blocked = self.entry_block_reason(timestamp, bar_idx, equity)
        if blocked:
            return RiskDecision(False, blocked)
        if self.consecutive_losses >= r.max_consecutive_losses:
            self.consecutive_losses = 0
        stop_distance = abs(entry - stop)
        if stop_distance <= 0 or not math.isfinite(stop_distance):
            return RiskDecision(False, "invalid_stop_distance")
        # Add stop slippage to sizing so nominal risk is not based on a frictionless stop.
        risk_per_unit = stop_distance + self.instrument.slippage_stop
        risk_cash = equity * r.risk_per_trade_pct
        if open_risk_cash + risk_cash > equity * r.max_portfolio_risk_pct + 1e-9:
            return RiskDecision(False, "max_portfolio_risk")
        raw_units = risk_cash / risk_per_unit
        factor = 10 ** self.instrument.unit_precision
        units = math.floor(raw_units * factor + 1e-12) / factor
        units = min(units, self.instrument.max_units)
        if units < self.instrument.min_units:
            return RiskDecision(False, "position_below_min_units")
        actual_risk = units * risk_per_unit
        margin = units * entry * self.instrument.margin_rate
        if r.enforce_margin and margin > equity:
            return RiskDecision(False, "insufficient_margin", units, actual_risk, margin)
        return RiskDecision(True, "approved", units, actual_risk, margin)

    def record_entry(self) -> None:
        self.trades_today += 1

    def record_closed_trade(self, pnl: float, bar_idx: int) -> None:
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_bar = bar_idx
        elif pnl > 0:
            self.consecutive_losses = 0
            self.last_loss_bar = None
