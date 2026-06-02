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
    readiness_penalty, win_cap, readiness_notes = _readiness_adjustment(side, execution_quality)
    win_probability -= readiness_penalty
    if direction_analysis.get("conflict_level") == "high":
        win_probability -= 0.09
        win_cap = min(win_cap, 0.54)
    elif direction_analysis.get("conflict_level") == "mild":
        win_probability -= 0.04
        win_cap = min(win_cap, 0.60)
    win_probability = _clamp(win_probability, 0.18, win_cap)
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
        "readiness_penalty": round(readiness_penalty, 4),
        "win_probability_cap": round(win_cap * 100.0, 2),
        "readiness_notes": readiness_notes,
        "method": "heuristic_v2; readiness-capped until signal_logger outcomes provide stronger calibration.",
    }


def _readiness_adjustment(side: DirectionScore, execution_quality: float) -> tuple[float, float, list[str]]:
    buckets = side.bucket_scores or {}
    htf = float(buckets.get("htf_context", 0.0))
    ltf = float(buckets.get("ltf_confirmation", 0.0))
    entry = float(buckets.get("entry_location", 0.0))
    risk = float(buckets.get("risk_plan", 0.0))
    exec_score = float(side.execution_score if side.execution_score is not None else execution_quality)
    distance = side.entry_distance_pct
    penalty = 0.0
    cap = 0.68
    notes: list[str] = []

    checks = (
        ("HTF", htf, 60.0, 0.0012, 0.60),
        ("LTF", ltf, 65.0, 0.0015, 0.58),
        ("entry", entry, 65.0, 0.0018, 0.56),
        ("risk", risk, 60.0, 0.0018, 0.58),
    )
    for label, value, threshold, factor, bucket_cap in checks:
        if value < threshold:
            penalty += (threshold - value) * factor
            cap = min(cap, bucket_cap)
            notes.append(f"{label} bucket {value:.1f} below {threshold:.0f}")

    if execution_quality < 55.0:
        penalty += (55.0 - execution_quality) * 0.0012
        cap = min(cap, 0.46 + max(execution_quality, 0.0) / 220.0)
        notes.append(f"execution_quality {execution_quality:.1f} below 55")
    if exec_score < 63.0:
        penalty += (63.0 - exec_score) * 0.0009
        cap = min(cap, 0.40 + max(exec_score, 0.0) / 240.0)
        notes.append(f"execution_score {exec_score:.1f} below 63")
    if isinstance(distance, (int, float)) and distance > 0.3:
        penalty += min(0.08, (float(distance) - 0.3) * 0.018)
        if distance > 1.2:
            cap = min(cap, 0.52)
        notes.append(f"entry distance {float(distance):.2f}% outside live band")
    if side.rr is None or float(side.rr or 0.0) < 1.8:
        cap = min(cap, 0.54)
        notes.append("RR below execution threshold")
    return max(0.0, penalty), max(0.18, cap), notes


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
