from __future__ import annotations

from typing import Any

from .models import DirectionScore, SymbolReport


def estimate_expected_value(
    report: SymbolReport,
    side: DirectionScore,
    direction_analysis: dict[str, Any],
    execution_quality: float,
) -> dict[str, Any]:
    rr = float(side.rr or 0.0)
    setup = float(side.setup_score if side.setup_score is not None else report.score)
    conviction = float(direction_analysis.get("direction_conviction") or 50.0)
    data = float(side.data_completeness)
    win_probability = 0.34 + setup * 0.0018 + conviction * 0.0017 + execution_quality * 0.0011 + data * 0.0006
    if direction_analysis.get("conflict_level") == "high":
        win_probability -= 0.09
    elif direction_analysis.get("conflict_level") == "mild":
        win_probability -= 0.04
    win_probability = _clamp(win_probability, 0.18, 0.68)
    avg_win_r = _clamp(rr * 0.72, 0.0, max(rr, 0.0))
    avg_loss_r = 1.0
    fee_cost_r = 0.025
    slippage_cost_r = _slippage_cost_r(report, side)
    loss_probability = 1.0 - win_probability
    expected_r = win_probability * avg_win_r - loss_probability * avg_loss_r - fee_cost_r - slippage_cost_r
    return {
        "expected_R": round(expected_r, 4),
        "estimated_win_probability": round(win_probability * 100.0, 2),
        "estimated_loss_probability": round(loss_probability * 100.0, 2),
        "estimated_avg_win_R": round(avg_win_r, 4),
        "estimated_avg_loss_R": round(avg_loss_r, 4),
        "fee_cost_R": round(fee_cost_r, 4),
        "slippage_cost_R": round(slippage_cost_r, 4),
        "cost_adjusted_expectancy": round(expected_r, 4),
        "method": "heuristic_v1; future signal_logger outcomes should calibrate this model.",
    }


def _slippage_cost_r(report: SymbolReport, side: DirectionScore) -> float:
    volume = float(report.quote_volume_24h or 0.0)
    atr_pct = side.market_metrics.get("atr_pct")
    atr_component = min(0.08, max(0.0, float(atr_pct or 0.0) - 1.8) * 0.012)
    if volume >= 250_000_000:
        volume_component = 0.01
    elif volume >= 80_000_000:
        volume_component = 0.02
    elif volume >= 20_000_000:
        volume_component = 0.04
    else:
        volume_component = 0.08
    return volume_component + atr_component


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
