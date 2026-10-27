from types import SimpleNamespace

import pandas as pd

from lsfvg.config import load_config
from lsfvg.core.types import FVG, MSS, Pool, Setup, StopResult, Sweep, TargetPlan
from lsfvg.signals.engine import ScanResult, SignalEngine
from lsfvg.backtest.engine import run_backtest


def _long_setup(signal_time):
    sweep = Sweep("pool1", "long", 1.0995, 0, 0, 0, 1.0990, 0.0005)
    mss = MSS(1, 0, 1.1005, 0.001, 1.2)
    fvg = FVG("bull", 0, 1, 2, 1.0998, 1.1002, 1.1000, 1.0998)
    stop = StopResult(1.0990, 0.001, 1.0, 1.0990, 0.0001, "ok")
    targets = TargetPlan("fixed_rr", 1.1020, "pool1", 1.1020, None, 0.5, None,
                         0.0001, None, None, None, 0.5, 2.0, 1.8, "")
    return Setup("setup-test", "EUR_USD", "long", "pool1", sweep, mss, fvg, 0, signal_time,
                 1.1000, 1.0990, stop, targets, 0.001, 0.0001, "discount", True)


def test_event_backtest_fills_and_closes_target():
    cfg = load_config("config/default.yaml", {
        "filters": {"session_enabled": False, "volatility": {"enabled": False}},
        "bias": {"enabled": False},
        "backtest": {"initial_capital": 10_000.0},
    })
    index = pd.date_range("2026-01-05T15:00:00Z", periods=4, freq="5min", name="time")
    ltf = pd.DataFrame({
        "open":  [1.1000, 1.1002, 1.1010, 1.1018],
        "high":  [1.1006, 1.1025, 1.1020, 1.1020],
        "low":   [1.0998, 1.1001, 1.1008, 1.1012],
        "close": [1.1002, 1.1018, 1.1017, 1.1018],
        "volume": [100, 100, 100, 100],
        "close_time": index + pd.Timedelta(minutes=5),
    }, index=index)
    stack = SimpleNamespace(ltf=ltf, mtf=ltf)
    prepared = SimpleNamespace(stack=stack)
    engine = SignalEngine(cfg, "EUR_USD")
    setup = _long_setup(index[0])
    engine.scan = lambda _p: ScanResult(setups=[setup])
    result = run_backtest(engine, prepared, cfg)
    assert result.metrics["trade_count"] == 1
    assert result.trade_log.iloc[0]["exit_reason"] == "take_profit"
    assert result.trade_log.iloc[0]["pnl"] > 0
    assert result.trade_log.iloc[0]["r_multiple"] > 0
    assert not result.metadata["live_execution_invoked"]


def test_ai_mode_fails_closed_without_policy():
    cfg = load_config("config/default.yaml", {
        "filters": {"session_enabled": False, "volatility": {"enabled": False}},
        "bias": {"enabled": False},
    })
    index = pd.date_range("2026-01-05T15:00:00Z", periods=3, freq="5min", name="time")
    ltf = pd.DataFrame({"open": [1.1]*3, "high": [1.101]*3, "low": [1.099]*3, "close": [1.1]*3,
                        "volume": [100]*3, "close_time": index + pd.Timedelta(minutes=5)}, index=index)
    engine = SignalEngine(cfg, "EUR_USD")
    engine.scan = lambda _p: ScanResult(setups=[_long_setup(index[0])])
    result = run_backtest(engine, SimpleNamespace(stack=SimpleNamespace(ltf=ltf, mtf=ltf)), cfg, ai_mode=True)
    assert result.metrics["trade_count"] == 0
    assert (result.decisions["reason"] == "ai_model_unavailable").any()


def test_partial_runner_records_partial_and_final_exit():
    cfg = load_config("config/default.yaml", {
        "filters": {"session_enabled": False, "volatility": {"enabled": False}},
        "bias": {"enabled": False},
    })
    index = pd.date_range("2026-01-05T15:00:00Z", periods=4, freq="5min", name="time")
    ltf = pd.DataFrame({
        "open":  [1.1000, 1.1002, 1.1011, 1.1018],
        "high":  [1.1006, 1.1014, 1.1025, 1.1020],
        "low":   [1.0998, 1.1001, 1.1008, 1.1012],
        "close": [1.1002, 1.1012, 1.1018, 1.1018],
        "volume": [100, 100, 100, 100],
        "close_time": index + pd.Timedelta(minutes=5),
    }, index=index)
    engine = SignalEngine(cfg, "EUR_USD")
    setup = _long_setup(index[0])
    setup.targets = TargetPlan("partial_runner", 1.1020, "pool1", 1.1020, 1.1010, 0.5, 1.0,
                               0.0001, None, None, None, 0.5, 1.5, 1.4, "")
    engine.scan = lambda _p: ScanResult(setups=[setup])
    result = run_backtest(engine, SimpleNamespace(stack=SimpleNamespace(ltf=ltf, mtf=ltf)), cfg)
    assert result.metrics["trade_count"] == 1
    trade = result.trades[0]
    assert len(trade.exits) == 2
    assert trade.exits[0].reason == "partial_take_profit"
    assert trade.exits[0].fraction == 0.5
    assert trade.exits[1].reason == "take_profit"
    assert trade.r_multiple > 0


def test_stop_first_when_stop_and_target_touch_same_bar():
    cfg = load_config("config/default.yaml", {
        "filters": {"session_enabled": False, "volatility": {"enabled": False}},
        "bias": {"enabled": False},
        "backtest": {"intrabar_stop_first": True},
    })
    index = pd.date_range("2026-01-05T15:00:00Z", periods=3, freq="5min", name="time")
    ltf = pd.DataFrame({
        "open":  [1.1000, 1.1000, 1.1000],
        "high":  [1.1006, 1.1025, 1.1005],
        "low":   [1.0998, 1.0988, 1.0995],
        "close": [1.1002, 1.1000, 1.1000],
        "volume": [100, 100, 100],
        "close_time": index + pd.Timedelta(minutes=5),
    }, index=index)
    engine = SignalEngine(cfg, "EUR_USD")
    engine.scan = lambda _p: ScanResult(setups=[_long_setup(index[0])])
    result = run_backtest(engine, SimpleNamespace(stack=SimpleNamespace(ltf=ltf, mtf=ltf)), cfg)
    assert result.metrics["trade_count"] == 1
    assert result.trade_log.iloc[0]["exit_reason"] in ("stop_loss_ambiguous_bar", "stop_loss_same_bar_conservative")
    assert result.trade_log.iloc[0]["pnl"] < 0
