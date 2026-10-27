"""Chronological event-driven backtester for the Liquidity Sweep + FVG strategy.

Signal definitions are delegated to the existing SignalEngine. This module only
simulates order lifecycle, fills, trade management, account risk and reporting.
All prices are in instrument price units; P&L assumes quote-currency USD for the
initial EUR_USD and SPX500_USD instruments. Unsupported account-currency conversion
is explicitly flagged rather than silently guessed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ..config.schema import AppConfig
from ..core.types import EntryOrder, ExitFill, Setup, TradeRecord
from ..data.spread import SpreadModel
from ..signals.engine import Prepared, SignalEngine, ScanResult
from ..risk.manager import RiskManager


@dataclass
class OpenPosition:
    setup: Setup
    units: float
    initial_units: float
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    initial_stop: float
    risk_cash: float
    target_price: Optional[float]
    remaining_fraction: float = 1.0
    partial_done: bool = False
    trail_active: bool = False
    bars_held: int = 0
    exits: list[ExitFill] = field(default_factory=list)
    pnl: float = 0.0
    commission_paid: float = 0.0
    entry_slippage: float = 0.0
    last_mtm: float = 0.0


@dataclass
class BacktestResult:
    trades: list[TradeRecord]
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    decisions: pd.DataFrame
    metrics: dict[str, Any]
    metadata: dict[str, Any]

    def save(self, out_dir: str | Path) -> dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for name, frame in (("equity_curve.csv", self.equity_curve),
                            ("trades.csv", self.trade_log), ("decisions.csv", self.decisions)):
            p = out / name
            frame.to_csv(p, index=True)
            paths[name] = str(p)
        p = out / "metrics.json"
        import json
        p.write_text(json.dumps({"metrics": self.metrics, "metadata": self.metadata}, indent=2, default=str), encoding="utf-8")
        paths["metrics.json"] = str(p)
        self._save_charts(out)
        return paths

    def _save_charts(self, out: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.equity_curve.empty:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(self.equity_curve.index, self.equity_curve["equity"])
            ax.set_title("Equity curve")
            ax.set_ylabel("Account equity")
            ax.grid(True, alpha=0.25)
            fig.tight_layout(); fig.savefig(out / "equity_curve.png", dpi=130); plt.close(fig)
            fig, ax = plt.subplots(figsize=(11, 3.5))
            ax.fill_between(self.equity_curve.index, self.equity_curve["drawdown_pct"], 0)
            ax.set_title("Drawdown")
            ax.set_ylabel("Drawdown (%)")
            ax.grid(True, alpha=0.25)
            fig.tight_layout(); fig.savefig(out / "drawdown.png", dpi=130); plt.close(fig)


def _quote_frame(ltf: pd.DataFrame, engine: SignalEngine, cfg: AppConfig) -> dict[str, np.ndarray]:
    ins = engine.ins
    if cfg.backtest.use_bid_ask_if_available and all(f"{s}_{f}" in ltf for s in ("bid", "ask") for f in ("open", "high", "low", "close")):
        q = {f"{s}_{f}": ltf[f"{s}_{f}"].to_numpy(float).copy()
             for s in ("bid", "ask") for f in ("open", "high", "low", "close")}
        if cfg.backtest.default_spread_stress != 1.0:
            for f in ("open", "high", "low", "close"):
                mid = ltf[f].to_numpy(float)
                half = (q[f"ask_{f}"] - q[f"bid_{f}"]) / 2.0 * cfg.backtest.default_spread_stress
                q[f"bid_{f}"] = mid - half
                q[f"ask_{f}"] = mid + half
        spread = q["ask_close"] - q["bid_close"]
        return {**q, "spread": spread}
    model = SpreadModel(ins, "America/New_York")
    spread = model.at(ltf.index) * cfg.backtest.default_spread_stress
    half = spread / 2.0
    q = {}
    for f in ("open", "high", "low", "close"):
        mid = ltf[f].to_numpy(float)
        q[f"bid_{f}"] = mid - half
        q[f"ask_{f}"] = mid + half
    return {**q, "spread": spread}


def _feature_vector(s: Setup) -> dict[str, float | str | bool]:
    """Small, causal setup vector for optional model-policy integration."""
    risk = max(s.risk, 1e-12)
    return {
        "direction": s.direction, "pool_kind": str(s.meta.get("pool_kind", "unknown")),
        "bias_zone": s.bias_zone, "htf_aligned": bool(s.htf_aligned),
        "sweep_depth_atr": float(s.sweep.depth / s.atr_mtf) if s.atr_mtf > 0 else 0.0,
        "mss_body_atr": float(s.mss.body_atr), "fvg_size_atr": float(s.fvg.size / s.atr_mtf) if s.atr_mtf > 0 else 0.0,
        "risk_distance": float(risk), "planned_rr": float(s.targets.planned_rr),
        "effective_rr": float(s.targets.effective_rr), "spread_est": float(s.spread_est),
    }


def _pnl_for(pos: OpenPosition, exit_price: float, fraction: float, commission_per_unit_rt: float) -> tuple[float, float]:
    units = pos.initial_units * fraction
    direction_sign = 1.0 if pos.setup.direction == "long" else -1.0
    gross = (exit_price - pos.entry_price) * direction_sign * units
    commission = commission_per_unit_rt * units
    return gross - commission, commission


def _metrics(trade_log: pd.DataFrame, equity: pd.DataFrame, initial: float) -> dict[str, Any]:
    n = len(trade_log)
    pnls = trade_log["pnl"].to_numpy(float) if n else np.array([], float)
    rs = trade_log["r_multiple"].to_numpy(float) if n else np.array([], float)
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    gross_profit, gross_loss = float(wins.sum()), float(-losses.sum())
    final = float(equity["equity"].iloc[-1]) if not equity.empty else initial
    total_return = final / initial - 1 if initial else 0.0
    max_dd = float(equity["drawdown_pct"].max()) if not equity.empty else 0.0
    days = max(1, (equity.index[-1] - equity.index[0]).total_seconds() / 86400) if len(equity) > 1 else 1
    cagr = (final / initial) ** (365.25 / days) - 1 if initial > 0 and final > 0 and days > 30 else None
    daily = equity["equity"].resample("1D").last().dropna().pct_change().dropna() if not equity.empty else pd.Series(dtype=float)
    sharpe = float(np.sqrt(252) * daily.mean() / daily.std(ddof=1)) if len(daily) > 1 and daily.std(ddof=1) > 0 else None
    downside = daily[daily < 0]
    sortino = float(np.sqrt(252) * daily.mean() / downside.std(ddof=1)) if len(downside) > 1 and downside.std(ddof=1) > 0 else None
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    be = avg_loss / (avg_win + avg_loss) if avg_win + avg_loss > 0 else None
    return {
        "initial_capital": initial, "final_equity": final, "net_pnl": final - initial,
        "total_return_pct": total_return * 100, "cagr_pct": cagr * 100 if cagr is not None else None,
        "trade_count": n, "win_rate_pct": float((pnls > 0).mean() * 100) if n else 0.0,
        "average_winner": avg_win, "average_loser_abs": avg_loss,
        "average_r_multiple": float(np.mean(rs)) if len(rs) else 0.0,
        "expectancy_cash": float(pnls.mean()) if n else 0.0,
        "expectancy_r": float(rs.mean()) if len(rs) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "max_drawdown_pct": max_dd * 100, "sharpe_daily_annualized": sharpe, "sortino_daily_annualized": sortino,
        "break_even_win_rate_pct_from_avg_payoffs": be * 100 if be is not None else None,
        "gross_profit": gross_profit, "gross_loss_abs": gross_loss,
        "commission_total": float(trade_log["commission"].sum()) if n and "commission" in trade_log else 0.0,
        "avg_holding_bars": float(trade_log["bars_held"].mean()) if n and "bars_held" in trade_log else 0.0,
    }


def run_backtest(
    signal_engine: SignalEngine,
    prepared: Prepared,
    cfg: AppConfig,
    *,
    ai_policy: Optional[Callable[[Setup, dict[str, Any], pd.Timestamp], dict[str, Any]]] = None,
    ai_mode: bool = False,
) -> BacktestResult:
    """Run one-instrument chronological simulation; live execution is never invoked.

    ``ai_policy`` must return ``{"action": "accept"|"reject", "score": float,
    "reason": str}``. If AI mode is requested without a policy, candidates fail closed
    unless ``backtest.allow_ai_without_model`` explicitly permits deterministic fallback.
    """
    ltf = prepared.stack.ltf
    if ltf.empty:
        raise ValueError("cannot backtest an empty LTF frame")
    scan: ScanResult = signal_engine.scan(prepared)
    setups_by_time: dict[pd.Timestamp, list[Setup]] = {}
    for setup in scan.setups:
        setups_by_time.setdefault(pd.Timestamp(setup.signal_time), []).append(setup)
    quotes = _quote_frame(ltf, signal_engine, cfg)
    mid = {c: ltf[c].to_numpy(float) for c in ("open", "high", "low", "close")}
    times = pd.DatetimeIndex(ltf.index)
    close_times = pd.DatetimeIndex(ltf["close_time"]) if "close_time" in ltf else times + pd.Timedelta(minutes=5)
    ins = signal_engine.ins
    initial = float(cfg.backtest.initial_capital)
    risk = RiskManager(cfg, ins, initial)
    balance = initial
    position: Optional[OpenPosition] = None
    pending: list[dict[str, Any]] = []
    trade_records: list[TradeRecord] = []
    trade_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    setup_seen: set[str] = set()
    total_entry_slippage = 0.0
    halted_at: Optional[pd.Timestamp] = None
    spread_stress = float(cfg.backtest.default_spread_stress)
    slip_stress = float(cfg.backtest.slippage_stress)

    def quote(name: str, j: int) -> float:
        return float(quotes[name][j])

    def mark_to_market(j: int) -> float:
        if position is None:
            return balance
        px = quote("bid_close" if position.setup.direction == "long" else "ask_close", j)
        sign = 1.0 if position.setup.direction == "long" else -1.0
        floating = (px - position.entry_price) * sign * position.units
        est_exit_commission = ins.commission_per_unit_rt * position.units
        return balance + floating - est_exit_commission

    def finish_position(j: int, timestamp: pd.Timestamp, reason: str, raw_exit: float) -> None:
        nonlocal position, balance
        if position is None:
            return
        sign = 1.0 if position.setup.direction == "long" else -1.0
        slip = ins.slippage_stop * slip_stress
        exit_px = raw_exit - slip if sign > 0 else raw_exit + slip
        fraction = position.remaining_fraction
        pnl, commission = _pnl_for(position, exit_px, fraction, ins.commission_per_unit_rt)
        position.pnl += pnl
        position.commission_paid += commission
        position.exits.append(ExitFill(pd.Timestamp(timestamp), float(exit_px), float(fraction), reason))
        balance += pnl
        rec = TradeRecord(position.setup.setup_id, position.setup.direction, position.entry_time,
                          position.entry_price, position.initial_stop, position.target_price,
                          exits=list(position.exits), r_multiple=(position.pnl / position.risk_cash if position.risk_cash else 0.0))
        rec.units = position.initial_units
        rec.pnl = position.pnl
        rec.risk_cash = position.risk_cash
        rec.exit_reason = reason
        rec.bars_held = position.bars_held
        trade_records.append(rec)
        trade_rows.append({
            "setup_id": position.setup.setup_id, "instrument": ins.symbol, "direction": position.setup.direction,
            "entry_time": position.entry_time, "exit_time": timestamp, "entry_price": position.entry_price,
            "initial_stop": position.initial_stop, "final_stop": position.stop_price,
            "target_price": position.target_price, "units": position.initial_units,
            "risk_cash": position.risk_cash, "pnl": position.pnl,
            "r_multiple": rec.r_multiple, "commission": position.commission_paid,
            "bars_held": position.bars_held, "exit_reason": reason,
            "ai_score": position.setup.meta.get("ai_score"),
            "regime": position.setup.meta.get("ai_regime"),
        })
        risk.record_closed_trade(position.pnl, j)
        position = None

    def partial_exit(j: int, timestamp: pd.Timestamp, price: float, fraction: float, reason: str) -> None:
        nonlocal balance
        if position is None or fraction <= 0 or fraction > position.remaining_fraction:
            return
        sign = 1.0 if position.setup.direction == "long" else -1.0
        slip = ins.slippage_stop * slip_stress
        px = price - slip if sign > 0 else price + slip
        # Fraction is of the original size, and the fee scales with that fraction.
        pnl, commission = _pnl_for(position, px, fraction, ins.commission_per_unit_rt)
        position.pnl += pnl
        position.commission_paid += commission
        position.exits.append(ExitFill(pd.Timestamp(timestamp), float(px), float(fraction), reason))
        balance += pnl
        position.remaining_fraction -= fraction
        position.units = position.initial_units * position.remaining_fraction

    def close_all_at_end(j: int, reason: str) -> None:
        if position is None:
            return
        raw = quote("bid_close" if position.setup.direction == "long" else "ask_close", j)
        finish_position(j, close_times[j], reason, raw)

    for j, ts in enumerate(times):
        ts = pd.Timestamp(ts)
        # New York trading-day/week roll and equity-based kill-switch checks.
        equity_before = mark_to_market(j)
        risk.update_period(ts, equity_before)
        if risk.halted and halted_at is None:
            halted_at = ts
        block_reason = risk.entry_block_reason(ts, j, equity_before)
        if block_reason and pending:
            for old in list(pending):
                decision_rows.append({"time": ts, "setup_id": old["setup"].setup_id,
                                      "decision": "CANCEL_PENDING", "reason": f"risk_block:{block_reason}"})
            pending.clear()

        # New signals are available at their MTF close; orders can activate on this or later LTF opens.
        for signal_time, group in list(setups_by_time.items()):
            if signal_time <= ts and any(s.setup_id not in setup_seen for s in group):
                for setup in group:
                    if setup.setup_id in setup_seen:
                        continue
                    setup_seen.add(setup.setup_id)
                    feat = _feature_vector(setup)
                    ai_decision: dict[str, Any] = {"action": "accept", "score": None, "reason": "deterministic_baseline"}
                    if ai_mode:
                        if ai_policy is None:
                            if cfg.backtest.allow_ai_without_model:
                                ai_decision = {"action": "accept", "score": None, "reason": "explicit_deterministic_fallback"}
                            else:
                                ai_decision = {"action": "reject", "score": None, "reason": "ai_model_unavailable"}
                        else:
                            try:
                                candidate = ai_policy(setup, feat, ts)
                                ai_decision.update(candidate or {})
                                if ai_decision.get("action") not in ("accept", "reject"):
                                    ai_decision = {"action": "reject", "score": None, "reason": "invalid_ai_action"}
                            except Exception as exc:
                                ai_decision = {"action": "reject", "score": None, "reason": f"ai_policy_error:{type(exc).__name__}"}
                    decision_rows.append({"time": ts, "signal_time": setup.signal_time, "setup_id": setup.setup_id,
                                          "decision": "TAKE" if ai_decision["action"] == "accept" else "REJECT",
                                          "reason": ai_decision.get("reason", ""), "ai_score": ai_decision.get("score"),
                                          "ai_mode": ai_mode, **feat})
                    if ai_decision["action"] != "accept":
                        continue
                    setup.meta["ai_score"] = ai_decision.get("score")
                    setup.meta["ai_regime"] = ai_decision.get("regime")
                    # A new valid setup replaces older pending orders, preventing stale signals from accumulating.
                    for old in pending:
                        decision_rows.append({"time": ts, "setup_id": old["setup"].setup_id,
                                              "decision": "CANCEL_PENDING", "reason": "replaced_by_new_setup"})
                    pending.clear()
                    if len(pending) < cfg.backtest.max_pending_orders:
                        pending.append({"setup": setup, "order": signal_engine.entry_order(setup), "start_idx": j,
                                        "age": 0, "ai_decision": ai_decision})
                del setups_by_time[signal_time]

        # Optional opposite-signal position response is evaluated only after signal is known.
        if position is not None:
            new_opposite = [r for r in decision_rows if r.get("time") == ts and r.get("decision") == "TAKE"
                            and r.get("direction") == ("short" if position.setup.direction == "long" else "long")]
            rule = cfg.signals.opposite_signal_rule
            if new_opposite and rule == "exit":
                raw = quote("bid_open" if position.setup.direction == "long" else "ask_open", j)
                finish_position(j, ts, "opposite_signal", raw)
            elif new_opposite and rule == "tighten_stop":
                position.stop_price = max(position.stop_price, position.entry_price) if position.setup.direction == "long" else min(position.stop_price, position.entry_price)

        # Manage the existing position using executable-side quotes; stop wins if stop and target hit in one bar.
        if position is not None:
            pos = position
            pos.bars_held += 1
            long = pos.setup.direction == "long"
            stop_hit = quote("bid_low" if long else "ask_high", j) <= pos.stop_price if long else quote("ask_high", j) >= pos.stop_price
            target = pos.target_price
            target_hit = target is not None and (quote("bid_high", j) >= target if long else quote("ask_low", j) <= target)
            if stop_hit and target_hit:
                if cfg.backtest.intrabar_stop_first:
                    raw_amb = quote("bid_open" if long else "ask_open", j)
                    raw_amb = min(raw_amb, pos.stop_price) if long else max(raw_amb, pos.stop_price)
                    finish_position(j, ts, "stop_loss_ambiguous_bar", raw_amb)
                else:
                    raw_amb = quote("bid_open" if long else "ask_open", j)
                    raw_amb = max(raw_amb, target) if long else min(raw_amb, target)
                    finish_position(j, ts, "take_profit_ambiguous_bar", raw_amb)
            elif stop_hit:
                raw = quote("bid_open" if long else "ask_open", j)
                if long:
                    raw = min(raw, pos.stop_price)
                else:
                    raw = max(raw, pos.stop_price)
                finish_position(j, ts, "stop_loss", raw)
            else:
                # Partial runner at configured partial level; then move stop to breakeven +/- configured R offset.
                plan = pos.setup.targets
                if plan.mode == "partial_runner" and not pos.partial_done and plan.partial_price is not None:
                    phit = quote("bid_high", j) >= plan.partial_price if long else quote("ask_low", j) <= plan.partial_price
                    if phit:
                        partial_exit(j, ts, plan.partial_price, plan.partial_pct, "partial_take_profit")
                        pos.partial_done = True
                        off = plan.be_offset_price
                        pos.stop_price = max(pos.stop_price, pos.entry_price + off) if long else min(pos.stop_price, pos.entry_price - off)
                        if pos.remaining_fraction <= 1e-9:
                            finish_position(j, ts, "partial_fully_closed", plan.partial_price)
                if position is not None and target is not None:
                    target_hit = quote("bid_high", j) >= target if long else quote("ask_low", j) <= target
                    if target_hit:
                        raw = quote("bid_open" if long else "ask_open", j)
                        raw = max(raw, target) if long else min(raw, target)
                        finish_position(j, ts, "take_profit", raw)
                # Trailing stop activates after threshold, then follows only closed-bar confirmed pivots/FVG proxy.
                if position is not None and pos.setup.targets.mode == "trailing":
                    plan = pos.setup.targets
                    act = plan.trail_activation_price
                    reached = act is not None and (quote("bid_high", j) >= act if long else quote("ask_low", j) <= act)
                    if reached:
                        pos.trail_active = True
                    if pos.trail_active and j >= 2:
                        if plan.trail_mode == "fvg":
                            proposed = pos.setup.fvg.bottom if long else pos.setup.fvg.top
                        else:
                            # Confirmed local 3-bar pivot: use only pivots whose right-hand bars have closed.
                            proposed = float(np.min(mid["low"][j-2:j+1])) if long else float(np.max(mid["high"][j-2:j+1]))
                        pos.stop_price = max(pos.stop_price, proposed) if long else min(pos.stop_price, proposed)
                if position is not None and pos.setup.targets.time_stop_bars is not None and pos.bars_held >= pos.setup.targets.time_stop_bars:
                    sign = 1 if long else -1
                    current = quote("bid_close" if long else "ask_close", j)
                    r_now = (current - pos.entry_price) * sign / max(abs(pos.entry_price - pos.initial_stop), 1e-12)
                    if r_now < pos.setup.targets.time_stop_r:
                        finish_position(j, close_times[j], "time_stop", current)

        # Optional market-close controls use the actual bar close time in New York.
        if position is not None and (cfg.backtest.session_end_exit or cfg.backtest.weekend_exit):
            local_close = pd.Timestamp(close_times[j]).tz_convert("America/New_York")
            minute = local_close.hour * 60 + local_close.minute
            at_daily_close = minute >= 17 * 60
            at_weekly_close = local_close.weekday() == 4 and at_daily_close
            if cfg.backtest.weekend_exit and at_weekly_close:
                close_all_at_end(j, "weekend_exit")
            elif cfg.backtest.session_end_exit and at_daily_close:
                close_all_at_end(j, "session_end")

        # Process pending orders after existing position management.
        for item in list(pending):
            if position is not None or risk.halted:
                break
            setup: Setup = item["setup"]
            order: EntryOrder = item["order"]
            if ts < pd.Timestamp(order.place_time):
                continue
            item["age"] += 1
            long = order.direction == "long"
            # The target-pool cancellation is deliberately checked before a same-bar entry.
            tgt = order.target_level
            if tgt is not None and ((long and mid["high"][j] > tgt) or ((not long) and mid["low"][j] < tgt)):
                decision_rows.append({"time": ts, "setup_id": setup.setup_id, "decision": "CANCEL_PENDING", "reason": "target_taken_before_fill"})
                pending.remove(item); continue
            th = order.fill_through_ticks * order.tick_size
            ext = quote("ask_low" if long else "bid_high", j)
            opn = quote("ask_open" if long else "bid_open", j)
            hit = ext <= order.limit_price - th if long else ext >= order.limit_price + th
            if hit:
                fill = min(order.limit_price, opn) if long else max(order.limit_price, opn)
                fill += ins.slippage_entry * slip_stress if long else -ins.slippage_entry * slip_stress
                equity = mark_to_market(j)
                open_risk = position.risk_cash if position is not None else 0.0
                rd = risk.approve_entry(timestamp=ts, bar_idx=j, equity=equity, entry=fill,
                                        stop=setup.stop_price, open_risk_cash=open_risk)
                if not rd.allowed:
                    decision_rows.append({"time": ts, "setup_id": setup.setup_id, "decision": "RISK_REJECT", "reason": rd.reason})
                    pending.remove(item); continue
                position = OpenPosition(setup, rd.units, rd.units, ts, float(fill), float(setup.stop_price),
                                        float(setup.stop_price), rd.risk_cash, setup.targets.final_target,
                                        entry_slippage=ins.slippage_entry * slip_stress)
                risk.record_entry()
                total_entry_slippage += ins.slippage_entry * slip_stress * rd.units
                decision_rows.append({"time": ts, "setup_id": setup.setup_id, "decision": "FILLED", "reason": "limit_fill",
                                      "fill_price": fill, "units": rd.units, "risk_cash": rd.risk_cash})
                pending.remove(item)
                # Only one live/pending idea is allowed per instrument; avoid stale second entries.
                for old in list(pending):
                    decision_rows.append({"time": ts, "setup_id": old["setup"].setup_id,
                                          "decision": "CANCEL_PENDING", "reason": "position_opened"})
                    pending.remove(old)
                # Conservatively evaluate same-bar stop/target after the fill; stop-first handles ambiguity.
                stop_hit = quote("bid_low" if long else "ask_high", j) <= position.stop_price if long else quote("ask_high", j) >= position.stop_price
                target = position.target_price
                target_hit = target is not None and (quote("bid_high", j) >= target if long else quote("ask_low", j) <= target)
                if stop_hit and target_hit and not cfg.backtest.intrabar_stop_first:
                    raw = quote("bid_open" if long else "ask_open", j)
                    raw = max(raw, target) if long else min(raw, target)
                    finish_position(j, ts, "take_profit_same_bar_ambiguous", raw)
                elif stop_hit:
                    raw = quote("bid_open" if long else "ask_open", j)
                    raw = min(raw, position.stop_price) if long else max(raw, position.stop_price)
                    finish_position(j, ts, "stop_loss_same_bar_conservative", raw)
                elif target_hit:
                    raw = quote("bid_open" if long else "ask_open", j)
                    raw = max(raw, target) if long else min(raw, target)
                    finish_position(j, ts, "take_profit_same_bar", raw)
            elif (long and mid["close"][j] < order.fvg_bottom) or ((not long) and mid["close"][j] > order.fvg_top):
                decision_rows.append({"time": ts, "setup_id": setup.setup_id, "decision": "CANCEL_PENDING", "reason": "fvg_invalidated"})
                pending.remove(item)
            elif item["age"] >= order.max_wait_bars:
                decision_rows.append({"time": ts, "setup_id": setup.setup_id, "decision": "EXPIRED", "reason": "max_wait_bars"})
                pending.remove(item)

        # Kill switch blocks new entries and liquidates any open position at the current executable close.
        equity_now = mark_to_market(j)
        risk.update_period(ts, equity_now)
        if risk.halted:
            halted_at = halted_at or ts
            pending.clear()
            if position is not None:
                close_all_at_end(j, "kill_switch")
        equity = mark_to_market(j)
        peak = max(initial, float(max([x["equity"] for x in equity_rows], default=initial)), equity)
        equity_rows.append({"time": close_times[j], "balance": balance, "equity": equity,
                            "drawdown_pct": max(0.0, (peak - equity) / peak), "open_position": position is not None,
                            "pending_orders": len(pending), "halted": risk.halted})
        if risk.halted:
            # No new orders after kill switch. Continue marking the account for complete equity curve.
            pending.clear()

    if position is not None and cfg.backtest.force_close_at_end:
        close_all_at_end(len(ltf) - 1, "end_of_data")
        if equity_rows:
            equity_rows[-1]["balance"] = balance
            equity_rows[-1]["equity"] = balance
            equity_rows[-1]["open_position"] = False
            prior_peak = max(initial, max((float(x["equity"]) for x in equity_rows[:-1]), default=initial))
            equity_rows[-1]["drawdown_pct"] = max(0.0, (prior_peak - balance) / prior_peak)
    for item in pending:
        decision_rows.append({"time": times[-1], "setup_id": item["setup"].setup_id,
                              "decision": "PENDING_AT_END", "reason": "dataset_ended_before_resolution"})
    # Include all rule-level skipped setups in the audit log.
    for s in scan.skipped:
        decision_rows.append({"time": s.signal_time, "setup_id": s.setup_id, "decision": "STRATEGY_REJECT",
                              "reason": ";".join(s.reasons), "ai_mode": ai_mode})
    for r in scan.rejected_sweeps:
        decision_rows.append({"time": None, "setup_id": None, "decision": "NO_SETUP", "reason": r.get("reason"),
                              "pool_id": r.get("pool_id"), "sweep_idx": r.get("sweep_idx")})

    eq = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame(columns=["balance", "equity", "drawdown_pct"])
    trade_log = pd.DataFrame(trade_rows)
    decisions = pd.DataFrame(decision_rows)
    metrics = _metrics(trade_log, eq, initial)
    metrics.update({"setups_valid": len(scan.setups), "setups_skipped": len(scan.skipped),
                    "sweeps_rejected_before_setup": len(scan.rejected_sweeps),
                    "ai_mode": ai_mode, "ai_policy_available": ai_policy is not None,
                    "halted": risk.halted, "halt_reason": risk.halt_reason, "halted_at": str(halted_at) if halted_at else None,
                    "total_entry_slippage_cash_approx": total_entry_slippage})
    metadata = {"instrument": ins.symbol, "bars_ltf": len(ltf), "bars_mtf": len(prepared.stack.mtf),
                "start": str(times[0]), "end": str(close_times[-1]), "initial_capital": initial,
                "config_mode": cfg.run.mode, "live_execution_invoked": False,
                "spread_source": "recorded_bid_ask" if all(f"{s}_{f}" in ltf for s in ("bid", "ask") for f in ("open", "high", "low", "close")) else "modeled",
                "spread_stress": spread_stress, "slippage_stress": slip_stress,
                "ai_policy": "custom callable" if ai_policy is not None else "none"}
    return BacktestResult(trade_records, eq, trade_log, decisions, metrics, metadata)
