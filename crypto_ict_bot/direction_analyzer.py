from __future__ import annotations

from typing import Any

from .models import DirectionScore, SymbolReport


def analyze_direction(report: SymbolReport, regime: dict[str, Any] | None = None) -> dict[str, Any]:
    long_score = _side_score(report.long)
    short_score = _side_score(report.short)
    edge = long_score - short_score
    selected = report.long if edge >= 0 else report.short
    opposite = report.short if edge >= 0 else report.long
    chosen = "long" if edge > 0 else "short"
    abs_edge = abs(edge)
    selected_setup = _setup_quality(selected)
    opposite_setup = _setup_quality(opposite)
    selected_alignment = _alignment_score(selected, chosen, regime)
    opposite_alignment = _alignment_score(opposite, "short" if chosen == "long" else "long", regime)
    selected_rr = _rr_score(selected)
    opposite_rr = _rr_score(opposite)
    conviction = (
        abs_edge * 2.4
        + selected_setup * 0.34
        + selected_alignment * 0.24
        + selected_rr * 0.12
        - opposite_setup * 0.18
        - max(0.0, opposite_alignment - 55.0) * 0.2
    )
    both_good = selected_setup >= 68 and opposite_setup >= 62
    if abs_edge < 6:
        state = "neutral"
        chosen = "neutral"
        conflict = "mild" if both_good else "none"
    elif abs_edge < 10 and both_good:
        state = "conflict_wait_for_break"
        chosen = "neutral"
        conflict = "high"
    elif conviction >= 70 and selected_alignment >= 58 and selected_setup >= 62:
        state = "directional"
        conflict = "none"
    else:
        state = "weak_directional"
        conflict = "mild" if both_good or abs_edge < 12 else "none"
    why = _direction_reasons(chosen, selected, opposite, abs_edge, selected_setup, selected_alignment, selected_rr, conflict)
    return {
        "chosen_direction": chosen,
        "direction_state": state,
        "direction_conviction": round(_clamp(conviction), 2),
        "direction_edge": round(abs_edge, 2),
        "long_score": round(long_score, 2),
        "short_score": round(short_score, 2),
        "opposite_pressure": round(max(opposite_setup, opposite_alignment, opposite_rr), 2),
        "conflict_level": conflict,
        "why_this_direction": why,
    }


def _side_score(side: DirectionScore) -> float:
    if side.selection_score is not None:
        return float(side.selection_score)
    if side.calibrated_score is not None:
        return float(side.calibrated_score)
    return float(side.normalized)


def _setup_quality(side: DirectionScore) -> float:
    buckets = side.bucket_scores or {}
    htf = float(buckets.get("htf_context", 0.0))
    ltf = float(buckets.get("ltf_confirmation", 0.0))
    entry = float(buckets.get("entry_location", 0.0))
    risk = float(buckets.get("risk_plan", 0.0))
    setup = side.setup_score if side.setup_score is not None else htf * 0.32 + ltf * 0.27 + entry * 0.25 + risk * 0.16
    return _clamp(float(setup))


def _alignment_score(side: DirectionScore, direction: str, regime: dict[str, Any] | None) -> float:
    score = 54.0
    metrics = side.market_metrics or {}
    btc_against = bool(metrics.get("btc_against"))
    btc_trend = _as_float(metrics.get("btc_trend_pct"))
    if btc_trend is not None:
        aligned = (direction == "long" and btc_trend >= 0) or (direction == "short" and btc_trend <= 0)
        score += 14.0 if aligned else -14.0
    if btc_against:
        score -= 18.0
    regime_name = str((regime or {}).get("regime") or "")
    if regime_name == "trend_up":
        score += 10.0 if direction == "long" else -10.0
    elif regime_name in {"trend_down", "risk_off"}:
        score += 10.0 if direction == "short" else -10.0
    elif regime_name == "alt_rotation":
        score += 6.0
    return _clamp(score)


def _rr_score(side: DirectionScore) -> float:
    rr = side.rr or 0.0
    if rr >= 2.5:
        return 90.0
    if rr >= 2.0:
        return 78.0
    if rr >= 1.8:
        return 68.0
    if rr >= 1.5:
        return 52.0
    return max(0.0, rr * 30.0)


def _direction_reasons(
    chosen: str,
    selected: DirectionScore,
    opposite: DirectionScore,
    edge: float,
    setup: float,
    alignment: float,
    rr_score: float,
    conflict: str,
) -> list[str]:
    if chosen == "neutral":
        return [
            f"多空優勢差距 {edge:.1f}，尚不足以硬選方向。",
            f"另一側 setup 壓力約 {_setup_quality(opposite):.1f}，需要等待新的 BOS/MSS 或 liquidity sweep 打破僵局。",
        ]
    reasons = [
        f"{chosen.upper()} 方向 edge {edge:.1f}，setup 完整度 {setup:.1f}。",
        f"市場/BTC alignment {alignment:.1f}，RR 結構評分 {rr_score:.1f}。",
    ]
    if conflict != "none":
        reasons.append("另一側仍有壓力，方向只能列為弱確認或等待觸發。")
    if selected.reasons:
        reasons.append(selected.reasons[0])
    return reasons


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
