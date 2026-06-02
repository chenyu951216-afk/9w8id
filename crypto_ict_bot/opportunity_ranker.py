from __future__ import annotations

import math
from typing import Any

from .direction_analyzer import analyze_direction
from .expected_value import estimate_expected_value
from .models import DirectionScore, SymbolReport
from .regime_detector import detect_market_regime, regime_alignment_score
from .relative_strength import attach_relative_strength, relative_strength_score
from .risk.execution_gate import evaluate_execution_gate, selected_side


LIFECYCLE_ORDER = {
    "EXECUTABLE": 8,
    "ARMED": 7,
    "WATCH": 6,
    "SCOUT": 5,
    "BLOCKED_GOOD_SETUP": 4,
    "MISSED": 3,
    "EXPIRED": 2,
    "INVALID": 1,
}


def enrich_opportunity_context(reports: list[SymbolReport]) -> dict[str, Any]:
    regime = detect_market_regime(reports)
    attach_relative_strength(reports)
    rows: list[dict[str, Any]] = []
    for report in reports:
        context = _build_report_context(report, regime)
        report.metadata["market_regime"] = regime
        report.metadata["direction_analysis"] = context["direction_analysis"]
        report.metadata["entry_proximity"] = context["entry_proximity"]
        report.metadata["expected_value"] = context["expected_value"]
        report.metadata["opportunity"] = context
        report.metadata["opportunity_score"] = context["opportunity_score"]
        report.metadata["execution_quality"] = context["execution_quality"]
        report.metadata["candidate_grade"] = context["grade"]
        report.metadata["candidate_status"] = context["state"]
        signal_state = report.metadata.get("signal_state")
        if isinstance(signal_state, dict):
            signal_state["priority_level"] = context["grade"]
            signal_state["status"] = context["state"]
            signal_state["lifecycle_state"] = context["state"]
            signal_state["opportunity_score"] = context["opportunity_score"]
            signal_state["direction_analysis"] = context["direction_analysis"]
            signal_state["next_trigger"] = context["next_trigger"]
            signal_state["trade_thesis"] = context["thesis"]
            signal_state["blockers"] = context["blockers"]
        rows.append({"symbol": report.symbol, **context})
    rows.sort(key=lambda item: (item["opportunity_score"], item["execution_quality"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        for report in reports:
            if report.symbol == row["symbol"]:
                report.metadata["opportunity_rank"] = rank
                report.metadata["opportunity"]["rank"] = rank
                break
    return {
        "market_regime": regime,
        "top_opportunities": [_compact_row(row) for row in rows[:10]],
        "executable_signals": [_compact_row(row) for row in rows if row["state"] == "EXECUTABLE"],
        "armed_waiting_triggers": [_compact_row(row) for row in rows if row["state"] == "ARMED"],
        "watchlist": [_compact_row(row) for row in rows if row["state"] in {"WATCH", "SCOUT"}][:20],
        "blocked_good_setups": [_compact_row(row) for row in rows if row["state"] == "BLOCKED_GOOD_SETUP"][:10],
        "invalid_or_expired": [_compact_row(row) for row in rows if row["state"] in {"INVALID", "EXPIRED", "MISSED"}][:20],
        "long_short_conflicts": [_compact_row(row) for row in rows if row["direction_analysis"].get("conflict_level") in {"mild", "high"}][:10],
        "risk_exposure": _risk_exposure(rows),
    }


def opportunity_sort_key(report: SymbolReport) -> tuple[int, float, float, float]:
    metadata = report.metadata
    state = metadata.get("candidate_status", "")
    return (
        LIFECYCLE_ORDER.get(str(state), 0),
        float(metadata.get("opportunity_score", report.score) or 0.0),
        float(metadata.get("execution_quality", 0.0) or 0.0),
        float(report.quote_volume_24h or 0.0),
    )


def _build_report_context(report: SymbolReport, regime: dict[str, Any]) -> dict[str, Any]:
    side = selected_side(report)
    direction_analysis = analyze_direction(report, regime)
    chosen_direction = direction_analysis["chosen_direction"]
    effective_direction = chosen_direction if chosen_direction in {"long", "short"} else side.direction
    proximity = entry_proximity(report, side)
    gate = evaluate_execution_gate(report)
    diagnostics = _diagnostics(report, side)
    setup_quality = float(side.setup_score if side.setup_score is not None else report.score)
    risk_quality = diagnostics["risk_reward_quality"]
    data_quality = float(side.data_completeness)
    relative_strength = relative_strength_score(report, effective_direction)
    regime_alignment = regime_alignment_score(regime, effective_direction, side.market_metrics)
    location_quality = proximity["score"]
    execution_quality = _execution_quality(side, gate, proximity, regime, diagnostics)
    expected_value = estimate_expected_value(report, side, direction_analysis, execution_quality)
    hard_vetoes, soft_penalties = _risk_penalties(report, side, gate, direction_analysis, proximity, expected_value)
    weights = regime.get("weight_profile", {})
    opportunity = (
        regime_alignment * float(weights.get("regime_alignment", 0.18))
        + direction_analysis["direction_conviction"] * float(weights.get("direction_conviction", 0.22))
        + setup_quality * float(weights.get("setup_quality", 0.22))
        + location_quality * float(weights.get("location_quality", 0.14))
        + risk_quality * float(weights.get("risk_reward_quality", 0.12))
        + relative_strength * float(weights.get("relative_strength", 0.07))
        + data_quality * float(weights.get("data_quality", 0.05))
        - soft_penalties["conflict_penalty"]
        - soft_penalties["crowding_penalty"]
        - soft_penalties["liquidity_penalty"]
    )
    opportunity = _clamp(opportunity)
    state = _lifecycle_state(gate, opportunity, setup_quality, execution_quality, proximity, hard_vetoes, expected_value)
    grade = _grade(opportunity, setup_quality, execution_quality, direction_analysis, expected_value, state, hard_vetoes)
    blockers = _blockers(gate, hard_vetoes, soft_penalties, proximity, expected_value)
    thesis = _thesis(report, side, regime, direction_analysis, proximity, relative_strength, expected_value)
    return {
        "state": state,
        "grade": grade,
        "opportunity_score": round(opportunity, 2),
        "setup_score": round(setup_quality, 2),
        "execution_quality": round(execution_quality, 2),
        "direction_conviction": direction_analysis["direction_conviction"],
        "expected_R": expected_value["expected_R"],
        "estimated_win_probability": expected_value["estimated_win_probability"],
        "regime_alignment": round(regime_alignment, 2),
        "relative_strength_score": round(relative_strength, 2),
        "location_quality": round(location_quality, 2),
        "risk_reward_quality": round(risk_quality, 2),
        "data_quality": round(data_quality, 2),
        "entry_proximity": proximity,
        "direction_analysis": direction_analysis,
        "expected_value": expected_value,
        "hard_vetoes": hard_vetoes,
        "soft_penalties": soft_penalties,
        "thesis": thesis,
        "blockers": blockers,
        "next_trigger": _next_trigger(state, proximity, direction_analysis, gate),
        "invalidation": gate.get("invalidation_conditions", []),
        "should_execute": bool(gate.get("should_execute")),
    }


def entry_proximity(report: SymbolReport, side: DirectionScore) -> dict[str, Any]:
    distance = _entry_distance_pct(report.price, side.entry_zone)
    atr_pct = _as_float(side.market_metrics.get("atr_pct")) or 0.0
    spread_pct = _spread_pct(report)
    dynamic_band = max(0.30, atr_pct * 0.35, spread_pct * 3.0)
    if distance is None:
        return {
            "state": "no_entry_zone",
            "distance_pct": None,
            "dynamic_band_pct": round(dynamic_band, 4),
            "score": 0.0,
            "distance_in_bands": None,
        }
    ratio = distance / max(dynamic_band, 1e-9)
    score = math.exp(-1.0 * ratio**1.35) * 100.0
    if distance <= dynamic_band:
        state = "near_entry"
    elif distance <= dynamic_band * 2.5:
        state = "approaching_entry"
    elif distance <= dynamic_band * 7:
        state = "far_from_entry"
    else:
        state = "missed"
    return {
        "state": state,
        "distance_pct": round(distance, 4),
        "dynamic_band_pct": round(dynamic_band, 4),
        "score": round(_clamp(score), 2),
        "distance_in_bands": round(ratio, 3),
    }


def _execution_quality(
    side: DirectionScore,
    gate: dict[str, Any],
    proximity: dict[str, Any],
    regime: dict[str, Any],
    diagnostics: dict[str, float],
) -> float:
    base = float(side.execution_score if side.execution_score is not None else side.selection_score or 0.0)
    base = base * 0.55 + proximity["score"] * 0.25 + diagnostics["risk_reward_quality"] * 0.10
    base += (float(regime.get("liquidity_condition") or 50.0) - 50.0) * 0.08
    if not gate.get("paid_data_status", {}).get("derivatives_available", False):
        base -= 8.0
    if gate.get("paid_data_status", {}).get("blocked", False):
        base -= 30.0
    if proximity["state"] in {"far_from_entry", "missed"}:
        base -= 10.0
    return _clamp(base)


def _diagnostics(report: SymbolReport, side: DirectionScore) -> dict[str, float]:
    buckets = side.bucket_scores or {}
    return {
        "htf_context": float(buckets.get("htf_context", 0.0)),
        "ltf_trigger": float(buckets.get("ltf_confirmation", 0.0)),
        "entry_quality": float(buckets.get("entry_location", 0.0)),
        "risk_reward_quality": float(buckets.get("risk_plan", 0.0)),
        "market_quality": float(buckets.get("market_filter", 0.0)),
    }


def _risk_penalties(
    report: SymbolReport,
    side: DirectionScore,
    gate: dict[str, Any],
    direction_analysis: dict[str, Any],
    proximity: dict[str, Any],
    expected_value: dict[str, Any],
) -> tuple[list[str], dict[str, float]]:
    hard: list[str] = []
    conflict_penalty = 0.0
    crowding_penalty = 0.0
    liquidity_penalty = 0.0
    if not side.entry_zone or side.stop is None or not side.take_profits:
        hard.append("缺少完整 entry / stop / TP，不能執行，只能保留觀察理由。")
    if side.rr is None or side.rr < 1.1:
        hard.append("RR 無法計算或低於最低交易價值。")
    paid = gate.get("paid_data_status", {})
    if isinstance(paid, dict) and paid.get("blocked"):
        hard.append(f"Funding/OI 過熱：{paid.get('warning') or 'derivatives risk blocked'}")
    if side.data_completeness < 35:
        hard.append(f"資料完整度 {side.data_completeness:.0f}% 嚴重不足。")
    if direction_analysis.get("conflict_level") == "high":
        conflict_penalty += 12.0
    elif direction_analysis.get("conflict_level") == "mild":
        conflict_penalty += 5.0
    if report.metadata.get("direction_conflict"):
        conflict_penalty += 8.0
    if proximity["state"] == "far_from_entry":
        crowding_penalty += 4.0
    elif proximity["state"] == "missed":
        crowding_penalty += 12.0
    if report.quote_volume_24h < 20_000_000:
        liquidity_penalty += 12.0
    elif report.quote_volume_24h < 60_000_000:
        liquidity_penalty += 4.0
    if float(expected_value.get("expected_R") or 0.0) < -0.25:
        crowding_penalty += 5.0
    return hard, {
        "conflict_penalty": round(conflict_penalty, 2),
        "crowding_penalty": round(crowding_penalty, 2),
        "liquidity_penalty": round(liquidity_penalty, 2),
    }


def _lifecycle_state(
    gate: dict[str, Any],
    opportunity: float,
    setup_quality: float,
    execution_quality: float,
    proximity: dict[str, Any],
    hard_vetoes: list[str],
    expected_value: dict[str, Any],
) -> str:
    expected_r = float(expected_value.get("expected_R") or 0.0)
    if gate.get("should_execute"):
        return "EXECUTABLE"
    if hard_vetoes and setup_quality >= 70 and opportunity >= 62:
        return "BLOCKED_GOOD_SETUP"
    if hard_vetoes and opportunity < 55:
        return "INVALID"
    if proximity["state"] == "missed":
        return "MISSED" if opportunity < 78 else "WATCH"
    if proximity["state"] == "near_entry" and setup_quality >= 68 and execution_quality >= 62 and expected_r > -0.1:
        return "ARMED"
    if opportunity >= 68 and setup_quality >= 62:
        return "WATCH"
    if opportunity >= 55:
        return "SCOUT"
    return "INVALID"


def _grade(
    opportunity: float,
    setup_quality: float,
    execution_quality: float,
    direction_analysis: dict[str, Any],
    expected_value: dict[str, Any],
    state: str,
    hard_vetoes: list[str],
) -> str:
    if state in {"INVALID", "EXPIRED"} or (hard_vetoes and opportunity < 58):
        return "X"
    expected_r = float(expected_value.get("expected_R") or 0.0)
    conviction = float(direction_analysis.get("direction_conviction") or 0.0)
    if opportunity >= 82 and setup_quality >= 72 and conviction >= 70 and expected_r > 0 and state in {"ARMED", "EXECUTABLE"}:
        return "A"
    if opportunity >= 74 and setup_quality >= 66 and conviction >= 62 and expected_r >= -0.05 and state in {"WATCH", "ARMED", "EXECUTABLE", "BLOCKED_GOOD_SETUP"}:
        return "B"
    if opportunity >= 62:
        return "C"
    return "D"


def _blockers(
    gate: dict[str, Any],
    hard_vetoes: list[str],
    soft_penalties: dict[str, float],
    proximity: dict[str, Any],
    expected_value: dict[str, Any],
) -> list[str]:
    output = list(hard_vetoes)
    for blocker in gate.get("blockers", [])[:4]:
        if blocker not in output:
            output.append(str(blocker))
    if proximity["state"] not in {"near_entry", "no_entry_zone"}:
        output.append(f"距 entry {proximity.get('distance_in_bands')} 個動態 band，狀態={proximity['state']}。")
    if float(expected_value.get("expected_R") or 0.0) <= 0:
        output.append(f"目前 heuristic expected_R={expected_value.get('expected_R')}，尚未明顯為正。")
    for name, value in soft_penalties.items():
        if value:
            output.append(f"{name} soft penalty {value:.1f}。")
    deduped: list[str] = []
    for item in output:
        if item and item not in deduped:
            deduped.append(item)
    return deduped[:8]


def _thesis(
    report: SymbolReport,
    side: DirectionScore,
    regime: dict[str, Any],
    direction_analysis: dict[str, Any],
    proximity: dict[str, Any],
    relative_strength: float,
    expected_value: dict[str, Any],
) -> list[str]:
    direction = direction_analysis.get("chosen_direction")
    if direction == "neutral":
        direction = side.direction
    thesis = [
        f"{report.symbol} {str(direction).upper()}：market regime={regime.get('regime')}，方向信心 {direction_analysis.get('direction_conviction'):.1f}。",
        f"Setup {side.setup_score if side.setup_score is not None else report.score:.1f}，entry proximity={proximity['state']}，relative strength={relative_strength:.1f}。",
        f"Expected R={expected_value.get('expected_R')}，win probability 約 {expected_value.get('estimated_win_probability')}%。",
    ]
    for reason in direction_analysis.get("why_this_direction", [])[:2]:
        thesis.append(str(reason))
    for reason in side.reasons[:2]:
        if reason not in thesis:
            thesis.append(reason)
    return thesis


def _next_trigger(state: str, proximity: dict[str, Any], direction_analysis: dict[str, Any], gate: dict[str, Any]) -> str:
    if state == "EXECUTABLE":
        return "已通過 execution gate；仍需人工確認交易所流動性、滑價與部位大小。"
    if state == "INVALID":
        return "目前 setup / 方向 / 風控不足，等待新的 liquidity sweep、MSS/BOS 與完整交易計畫。"
    if state == "EXPIRED":
        return "原 setup 等待過久，先移出主觀察；等待新的結構重新形成。"
    if state == "MISSED":
        return "已遠離原 entry，不追價；等待新的 FVG/OB/OTE 計畫。"
    if direction_analysis.get("conflict_level") == "high":
        return "等待多空其中一側重新出現明確 MSS/BOS + displacement，解除方向衝突。"
    if proximity["state"] in {"far_from_entry", "approaching_entry"}:
        return "等待價格回到動態 entry band，並在 5m/15m 出現 micro BOS 或 displacement。"
    if proximity["state"] == "near_entry":
        return "價格已接近 entry，等待 5m close 確認與 volume 高於短均量。"
    blockers = gate.get("blockers", [])
    return str(blockers[0]) if blockers else "等待下一輪 K 線確認 setup 是否延續。"


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": row.get("rank"),
        "symbol": row["symbol"],
        "state": row["state"],
        "grade": row["grade"],
        "opportunity_score": row["opportunity_score"],
        "setup_score": row["setup_score"],
        "execution_quality": row["execution_quality"],
        "direction_conviction": row["direction_conviction"],
        "expected_R": row["expected_R"],
        "next_trigger": row["next_trigger"],
        "thesis": row["thesis"][:3],
        "blockers": row["blockers"][:3],
    }


def _risk_exposure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in rows if row["state"] in {"EXECUTABLE", "ARMED", "WATCH"}]
    long_count = sum(1 for row in active if row["direction_analysis"].get("chosen_direction") == "long")
    short_count = sum(1 for row in active if row["direction_analysis"].get("chosen_direction") == "short")
    return {
        "watchable_count": len(active),
        "long_bias_count": long_count,
        "short_bias_count": short_count,
        "net_direction_bias": long_count - short_count,
        "average_expected_R": round(sum(float(row["expected_R"] or 0.0) for row in active) / max(len(active), 1), 4),
    }


def _entry_distance_pct(price: float, entry_zone: tuple[float, float] | None) -> float | None:
    if not entry_zone:
        return None
    low, high = entry_zone
    if low <= price <= high:
        return 0.0
    return min(abs(price - low), abs(price - high)) / max(abs(price), 1e-12) * 100.0


def _spread_pct(report: SymbolReport) -> float:
    values = report.metadata.get("paid_data", {}).get("values", {})
    public = values.get("exchange_public_derivatives", {}) if isinstance(values, dict) else {}
    spread = public.get("spread_pct") if isinstance(public, dict) else None
    try:
        return float(spread)
    except (TypeError, ValueError):
        return 0.10 if report.quote_volume_24h < 20_000_000 else 0.04


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
