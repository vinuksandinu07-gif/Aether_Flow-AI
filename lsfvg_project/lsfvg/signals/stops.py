"""Stop-loss placement (pure).  Phase 2 manages the stop after entry (never further from entry)."""
from __future__ import annotations

import math

from ..config.schema import StopsCfg
from ..core.types import StopResult


def round_to_tick(price: float, tick: float, mode: str = "nearest") -> float:
    """Round to the instrument tick; ``up`` / ``down`` are directional (use for stops/targets)."""
    q = price / tick
    n = {"nearest": round(q), "up": math.ceil(q - 1e-9), "down": math.floor(q + 1e-9)}[mode]
    return round(n * tick, 10)


def compute_stop(
    direction: str,
    entry: float,
    *,
    sweep_extreme: float,
    fvg_first_high: float,
    fvg_first_low: float,
    atr_mtf: float,
    spread: float,
    cfg: StopsCfg,
    tick: float,
) -> StopResult:
    """Structural stop + buffer, then the ATR(MTF) sanity limits.

    * ``sweep_extreme``: beyond the sweep extreme;  ``fvg_first_candle``: beyond candle i-2 of the FVG.
    * buffer = spread + ``stop_buffer_atr`` x ATR(MTF).
    * distance < ``min_stop_atr`` x ATR -> widen to the minimum (or skip, per config);
    * distance > ``max_stop_atr`` x ATR -> SKIP (reason ``stop_too_wide``).
    The final stop is rounded AWAY from entry to the tick (risk is never understated).
    """
    short = direction == "short"
    buf = spread + cfg.stop_buffer_atr * atr_mtf
    if cfg.stop_mode == "sweep_extreme":
        ref = sweep_extreme
    else:
        ref = fvg_first_high if short else fvg_first_low
    structural = ref + buf if short else ref - buf
    dist = (structural - entry) if short else (entry - structural)
    if dist <= 0:
        return StopResult(None, dist, 0.0, structural, buf, "skipped", "stop_not_beyond_entry")
    if atr_mtf <= 0:
        return StopResult(None, dist, 0.0, structural, buf, "skipped", "atr_unavailable")
    action = "ok"
    min_d, max_d = cfg.min_stop_atr * atr_mtf, cfg.max_stop_atr * atr_mtf
    stop = structural
    if dist < min_d:
        if cfg.min_stop_action == "skip":
            return StopResult(None, dist, dist / atr_mtf, structural, buf, "skipped", "stop_too_tight")
        stop = entry + min_d if short else entry - min_d
        action = "widened"
    stop = round_to_tick(stop, tick, "up" if short else "down")
    dist = abs(stop - entry)
    if dist > max_d + 1e-12:
        return StopResult(None, dist, dist / atr_mtf, structural, buf, "skipped", "stop_too_wide")
    return StopResult(stop, dist, dist / atr_mtf, structural, buf, action)
