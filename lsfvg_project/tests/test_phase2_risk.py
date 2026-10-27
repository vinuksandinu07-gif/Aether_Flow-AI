import pandas as pd

from lsfvg.config import load_config
from lsfvg.risk.manager import RiskManager


def test_risk_manager_sizes_and_respects_daily_trade_limit():
    cfg = load_config("config/default.yaml", {"risk": {"max_trades_per_day": 1}})
    ins = cfg.instruments["EUR_USD"]
    rm = RiskManager(cfg, ins, 10_000)
    t = pd.Timestamp("2026-01-05T15:00:00Z")
    decision = rm.approve_entry(timestamp=t, bar_idx=0, equity=10_000, entry=1.1, stop=1.099)
    assert decision.allowed
    assert decision.units > 1
    assert decision.risk_cash <= 50.01
    rm.record_entry()
    rejected = rm.approve_entry(timestamp=t + pd.Timedelta(minutes=5), bar_idx=1,
                                equity=10_000, entry=1.1, stop=1.099)
    assert not rejected.allowed
    assert rejected.reason == "max_trades_per_day"


def test_risk_manager_drawdown_kill_switch():
    cfg = load_config("config/default.yaml")
    rm = RiskManager(cfg, cfg.instruments["EUR_USD"], 10_000)
    t = pd.Timestamp("2026-01-05T15:00:00Z")
    rm.update_period(t, 10_000)
    rm.update_period(t + pd.Timedelta(minutes=5), 8_999)
    assert rm.halted
    assert rm.halt_reason == "max_drawdown_kill_switch"
