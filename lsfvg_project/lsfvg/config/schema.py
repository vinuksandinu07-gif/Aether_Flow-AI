"""Typed configuration schema and a small, dependency-free validator.

Every tunable parameter lives here (defaults) and in ``config/default.yaml``
(explicit values).  ``load_config`` rejects unknown keys, wrong types and
inconsistent combinations and reports *all* problems at once.
"""
from __future__ import annotations

import re
import types
import typing as t
from dataclasses import dataclass, field, fields, is_dataclass


class ConfigError(ValueError):
    """Raised when the YAML config is invalid. ``args[0]`` lists every problem."""


# --------------------------------------------------------------------------- #
# Instruments
# --------------------------------------------------------------------------- #
@dataclass
class SessionWindow:
    """A named window in America/New_York wall-clock time (``end`` may wrap midnight)."""

    name: str = ""
    start: str = "00:00"
    end: str = "00:00"


@dataclass
class SpreadModelCfg:
    """``fixed``: constant ``typical``. ``time_of_day``: ``by_ny_hour`` lookup (price units)."""

    kind: t.Literal["fixed", "time_of_day"] = "fixed"
    typical: float = 0.0001
    by_ny_hour: dict[str, float] = field(default_factory=dict)


@dataclass
class FuturesCfg:
    """Contract-roll policy.  CFDs (OANDA SPX500_USD) are continuous: ``none``."""

    is_futures: bool = False
    roll_method: t.Literal[
        "none", "back_adjusted_difference", "back_adjusted_ratio", "per_contract"
    ] = "none"


@dataclass
class InstrumentCfg:
    symbol: str = ""
    asset_class: t.Literal["fx", "index_cfd", "index_future"] = "fx"
    pip_size: float = 0.0001          # 1 pip (FX) / 1 point (index), in price units
    tick_size: float = 0.00001        # smallest price increment
    price_precision: int = 5
    unit_precision: int = 0
    min_units: float = 1
    max_units: float = 100_000_000
    margin_rate: float = 0.0333       # VERIFY against the OANDA instruments endpoint
    spread: SpreadModelCfg = field(default_factory=SpreadModelCfg)
    max_spread: float = 0.0003        # price units; used by the max-spread filter
    slippage_entry: float = 0.00001   # price units, estimated
    slippage_stop: float = 0.00005    # price units, estimated
    commission_per_unit_rt: float = 0.0   # quote ccy per unit, round turn (USD-quoted => price units)
    day_anchor_ny: str = "17:00"      # trading-day / 4H anchor in New York time
    entry_sessions: list[SessionWindow] = field(default_factory=list)
    pool_sessions: list[SessionWindow] = field(default_factory=list)
    news_currencies: list[str] = field(default_factory=list)
    futures: FuturesCfg = field(default_factory=FuturesCfg)


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
@dataclass
class TimeframesCfg:
    htf: str = "4h"
    mtf: str = "15m"
    ltf: str = "5m"
    base: str = "5m"      # finest stored data; everything is resampled from it


@dataclass
class PoolsCfg:
    use_pdh_pdl: bool = True
    use_pwh_pwl: bool = True
    use_sessions: bool = True
    use_equal_mtf: bool = True
    use_equal_htf: bool = True
    daily_max_age_days: int = 5
    weekly_max_age_weeks: int = 4
    session_max_age_days: int = 2
    eq_max_age_bars: int = 500           # MTF bars
    # which pool scopes may be used as TP target: "any" or "external"
    tp_scope: t.Literal["any", "external"] = "any"


@dataclass
class BiasCfg:
    enabled: bool = True
    htf_trend_filter: bool = False       # optional HTF-structure alignment filter
    reference: t.Literal["entry", "signal_close"] = "entry"


@dataclass
class SignalsCfg:
    atr_period: int = 14
    htf_swing_n: int = 3
    mtf_swing_n: int = 3
    eq_tol_atr: float = 0.1
    eq_max_sep_bars: int = 300           # max bars between two equal highs/lows (own TF)
    eq_lookback_swings: int = 8
    min_sweep_atr: float = 0.05          # min sweep depth beyond the pool, x ATR(MTF)
    reclaim_bars: int = 3                # close back inside within k bars (breach bar = 0)
    disp_mult: float = 1.0               # displacement body >= disp_mult x ATR
    mss_max_bars: int = 6                # bars after reclaim in which MSS must occur
    fvg_min_atr: float = 0.1
    fvg_select: t.Literal["largest", "first", "latest"] = "largest"
    entry_mode: t.Literal["edge", "midpoint"] = "edge"
    max_wait_bars: int = 24              # LTF bars the limit order stays live
    fill_through_ticks: int = 1          # price must trade this many ticks through the limit
    opposite_signal_rule: t.Literal["ignore", "tighten_stop", "exit"] = "ignore"  # used in Phase 2


@dataclass
class NewsCfg:
    enabled: bool = False
    csv_path: t.Optional[str] = None     # columns: timestamp_utc,currency,impact,event
    before_min: int = 15
    after_min: int = 15
    min_impact: t.Literal["low", "medium", "high"] = "high"


@dataclass
class VolatilityCfg:
    enabled: bool = True
    lookback_bars: int = 2000            # MTF bars for the ATR percentile window
    min_percentile: float = 0.2          # skip when ATR percentile < this (0..1)


@dataclass
class ChopCfg:
    enabled: bool = False
    er_period: int = 20
    min_er: float = 0.2                  # Kaufman efficiency ratio threshold


@dataclass
class FiltersCfg:
    session_enabled: bool = True
    spread_enabled: bool = True
    news: NewsCfg = field(default_factory=NewsCfg)
    volatility: VolatilityCfg = field(default_factory=VolatilityCfg)
    chop: ChopCfg = field(default_factory=ChopCfg)
    max_trades_per_day: int = 2          # enforced by the Phase 2 risk manager


@dataclass
class StopsCfg:
    stop_mode: t.Literal["sweep_extreme", "fvg_first_candle"] = "sweep_extreme"
    stop_buffer_atr: float = 0.1
    min_stop_atr: float = 0.5
    max_stop_atr: float = 1.5
    min_stop_action: t.Literal["widen", "skip"] = "widen"


@dataclass
class TargetsCfg:
    tp_mode: t.Literal["opposing_liquidity", "fixed_rr", "partial_runner", "trailing"] = (
        "opposing_liquidity"
    )
    fixed_rr: float = 2.0
    min_rr: float = 2.0
    partial_pct: float = 0.5
    partial_at: t.Literal["1R", "internal_liquidity"] = "1R"
    be_offset_r: float = 0.1             # breakeven + offset, as a fraction of initial risk
    trail_activation_r: float = 1.0
    trail_mode: t.Literal["swing", "fvg"] = "swing"
    time_stop_bars: t.Optional[int] = None   # LTF bars; None = off
    time_stop_r: float = 0.5


@dataclass
class CsvCfg:
    tz: str = "UTC"                      # timezone of the timestamps in the file
    timestamp_col: t.Optional[str] = None
    time_format: t.Optional[str] = None


@dataclass
class QualityCfg:
    fail_on_error: bool = True
    spike_mult: float = 15.0             # reversal-spike detector, x rolling median true range
    spike_window: int = 200
    wide_spread_mult: float = 8.0
    long_gap_hours: float = 3.0          # weekday gaps >= this are reported as holiday/outage
    daily_break_max_minutes: int = 90


@dataclass
class DataCfg:
    csv: CsvCfg = field(default_factory=CsvCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)


@dataclass
class VizCfg:
    mtf_bars_before: int = 48
    mtf_bars_after: int = 32
    dpi: int = 110
    include_skipped: bool = True
    max_charts: int = 60


@dataclass
class RunCfg:
    name: str = "phase1"
    mode: t.Literal["backtest", "paper", "live"] = "backtest"
    allow_live: bool = False             # live trading requires this explicit flag
    seed: int = 42
    reports_dir: str = "reports"
    tz_sessions: str = "America/New_York"


@dataclass
class BacktestCfg:
    """Execution simulation controls. All price costs use instrument price units."""
    initial_capital: float = 10_000.0
    intrabar_stop_first: bool = True
    force_close_at_end: bool = True
    max_pending_orders: int = 100
    default_spread_stress: float = 1.0
    slippage_stress: float = 1.0
    use_bid_ask_if_available: bool = True
    allow_ai_without_model: bool = False
    session_end_exit: bool = False
    weekend_exit: bool = True


@dataclass
class RiskCfg:
    """Account-level risk limits enforced independently of strategy/AI decisions."""
    risk_per_trade_pct: float = 0.005
    daily_loss_limit_pct: float = 0.02
    weekly_loss_limit_pct: float = 0.04
    max_drawdown_pct: float = 0.10
    max_portfolio_risk_pct: float = 0.01
    max_open_positions_per_instrument: int = 1
    max_trades_per_day: int = 2
    max_consecutive_losses: int = 3
    cooldown_bars_after_loss: int = 12
    enforce_margin: bool = True


@dataclass
class AppConfig:
    run: RunCfg = field(default_factory=RunCfg)
    timeframes: TimeframesCfg = field(default_factory=TimeframesCfg)
    active_instruments: list[str] = field(default_factory=list)
    instruments: dict[str, InstrumentCfg] = field(default_factory=dict)
    signals: SignalsCfg = field(default_factory=SignalsCfg)
    pools: PoolsCfg = field(default_factory=PoolsCfg)
    bias: BiasCfg = field(default_factory=BiasCfg)
    filters: FiltersCfg = field(default_factory=FiltersCfg)
    stops: StopsCfg = field(default_factory=StopsCfg)
    targets: TargetsCfg = field(default_factory=TargetsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    viz: VizCfg = field(default_factory=VizCfg)
    backtest: BacktestCfg = field(default_factory=BacktestCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)

    def instrument(self, name: str) -> InstrumentCfg:
        return self.instruments[name]


# --------------------------------------------------------------------------- #
# Generic dict -> dataclass builder (unknown keys / wrong types are errors)
# --------------------------------------------------------------------------- #
def _coerce(tp: t.Any, v: t.Any, path: str, errors: list[str]) -> t.Any:
    origin, args = t.get_origin(tp), t.get_args(tp)
    if origin in (t.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if v is None:
            if len(non_none) != len(args):
                return None
            errors.append(f"{path}: null not allowed")
            return None
        return _coerce(non_none[0], v, path, errors)
    if origin is t.Literal:
        if v not in args:
            errors.append(f"{path}: {v!r} is not one of {list(args)}")
        return v
    if origin is list:
        if not isinstance(v, list):
            errors.append(f"{path}: expected a list")
            return []
        return [_coerce(args[0], x, f"{path}[{i}]", errors) for i, x in enumerate(v)]
    if origin is dict:
        if not isinstance(v, dict):
            errors.append(f"{path}: expected a mapping")
            return {}
        kt, vt = args
        out = {}
        for k, x in v.items():
            k2 = str(k) if kt is str else k
            out[k2] = _coerce(vt, x, f"{path}.{k2}", errors)
        return out
    if is_dataclass(tp):
        if v is None:
            v = {}
        if not isinstance(v, dict):
            errors.append(f"{path}: expected a mapping")
            return tp()
        return _build(tp, v, path, errors)
    if tp is bool:
        if not isinstance(v, bool):
            errors.append(f"{path}: expected true/false, got {v!r}")
        return v
    if tp is int:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
            errors.append(f"{path}: expected an integer, got {v!r}")
            return 0
        return int(v)
    if tp is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{path}: expected a number, got {v!r}")
            return 0.0
        return float(v)
    if tp is str:
        if not isinstance(v, str):
            errors.append(f"{path}: expected a string, got {v!r}")
            return ""
        return v
    return v


def _build(cls: type, data: dict, path: str, errors: list[str]) -> t.Any:
    hints = t.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    for k in data:
        if k not in names:
            errors.append(f"{path or '<root>'}: unknown key '{k}'")
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], f"{path}.{f.name}".lstrip("."), errors)
    return cls(**kwargs)


_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_TF = re.compile(r"^\d+(m|min|h|d)$")


def tf_minutes(tf: str) -> int:
    """Minutes in a timeframe string such as ``15m``, ``4h``, ``1d``."""
    m = re.fullmatch(r"(\d+)\s*(m|min|h|d)", tf.strip().lower())
    if not m:
        raise ConfigError(f"bad timeframe '{tf}'")
    n, u = int(m.group(1)), m.group(2)
    return n * {"m": 1, "min": 1, "h": 60, "d": 1440}[u]


def validate(cfg: AppConfig) -> list[str]:
    """Cross-field validation. Returns a list of problems (empty = OK)."""
    e: list[str] = []
    tf = cfg.timeframes
    for name in ("htf", "mtf", "ltf", "base"):
        if not _TF.match(getattr(tf, name)):
            e.append(f"timeframes.{name}: bad value '{getattr(tf, name)}'")
    if not e:
        b, l, m, h = (tf_minutes(x) for x in (tf.base, tf.ltf, tf.mtf, tf.htf))
        if not (b <= l <= m < h):
            e.append("timeframes must satisfy base <= ltf <= mtf < htf")
        for a, bb, nm in ((l, b, "ltf/base"), (m, l, "mtf/ltf"), (h, m, "htf/mtf")):
            if a % bb:
                e.append(f"timeframes: {nm} must divide evenly")
        if 1440 % h:
            e.append("timeframes.htf must divide 24h evenly (anchored bins)")
    if cfg.run.mode == "live" and not cfg.run.allow_live:
        e.append("run.mode=live requires run.allow_live=true (explicit opt-in)")
    for n in cfg.active_instruments:
        if n not in cfg.instruments:
            e.append(f"active_instruments: '{n}' is not defined under instruments")
    for n, ins in cfg.instruments.items():
        p = f"instruments.{n}"
        if not _HHMM.match(ins.day_anchor_ny):
            e.append(f"{p}.day_anchor_ny must be HH:MM")
        for lst, nm in ((ins.entry_sessions, "entry_sessions"), (ins.pool_sessions, "pool_sessions")):
            for w in lst:
                if not (_HHMM.match(w.start) and _HHMM.match(w.end)):
                    e.append(f"{p}.{nm}.{w.name}: start/end must be HH:MM")
        for fld in ("pip_size", "tick_size", "max_spread"):
            if getattr(ins, fld) <= 0:
                e.append(f"{p}.{fld} must be > 0")
        if ins.min_units > ins.max_units:
            e.append(f"{p}: min_units > max_units")
        if not 0 < ins.margin_rate <= 1:
            e.append(f"{p}.margin_rate must be in (0, 1]")
        if ins.futures.is_futures and ins.futures.roll_method == "none":
            e.append(f"{p}.futures: is_futures=true needs an explicit roll_method")
    s, st, tg = cfg.signals, cfg.stops, cfg.targets
    for nm, v in (("htf_swing_n", s.htf_swing_n), ("mtf_swing_n", s.mtf_swing_n),
                  ("atr_period", s.atr_period), ("reclaim_bars", s.reclaim_bars),
                  ("max_wait_bars", s.max_wait_bars)):
        if v < 1:
            e.append(f"signals.{nm} must be >= 1")
    for nm, v in (("min_sweep_atr", s.min_sweep_atr), ("disp_mult", s.disp_mult),
                  ("fvg_min_atr", s.fvg_min_atr), ("eq_tol_atr", s.eq_tol_atr)):
        if v < 0:
            e.append(f"signals.{nm} must be >= 0")
    if st.min_stop_atr > st.max_stop_atr:
        e.append("stops.min_stop_atr must be <= stops.max_stop_atr")
    if st.stop_buffer_atr < 0:
        e.append("stops.stop_buffer_atr must be >= 0")
    if tg.fixed_rr <= 0 or tg.min_rr < 0:
        e.append("targets.fixed_rr must be > 0 and min_rr >= 0")
    if not 0 < tg.partial_pct < 1:
        e.append("targets.partial_pct must be in (0, 1)")
    if tg.time_stop_bars is not None and tg.time_stop_bars < 1:
        e.append("targets.time_stop_bars must be >= 1 or null")
    if not 0 <= cfg.filters.volatility.min_percentile <= 1:
        e.append("filters.volatility.min_percentile must be in [0, 1]")
    if cfg.filters.news.enabled and not cfg.filters.news.csv_path:
        e.append("filters.news.enabled=true needs filters.news.csv_path")
    bt, rk = cfg.backtest, cfg.risk
    if bt.initial_capital <= 0:
        e.append("backtest.initial_capital must be > 0")
    if bt.max_pending_orders < 1:
        e.append("backtest.max_pending_orders must be >= 1")
    for nm, val in (("risk_per_trade_pct", rk.risk_per_trade_pct),
                    ("daily_loss_limit_pct", rk.daily_loss_limit_pct),
                    ("weekly_loss_limit_pct", rk.weekly_loss_limit_pct),
                    ("max_drawdown_pct", rk.max_drawdown_pct),
                    ("max_portfolio_risk_pct", rk.max_portfolio_risk_pct)):
        if not 0 < val <= 1:
            e.append(f"risk.{nm} must be in (0, 1]")
    if rk.max_open_positions_per_instrument < 1 or rk.max_trades_per_day < 1:
        e.append("risk position/trade limits must be >= 1")
    if rk.max_consecutive_losses < 1 or rk.cooldown_bars_after_loss < 0:
        e.append("risk consecutive-loss limit must be >= 1 and cooldown bars >= 0")
    return e
