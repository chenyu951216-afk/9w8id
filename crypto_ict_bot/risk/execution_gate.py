from __future__ import annotations

from typing import Any

from ..instrument_classifier import (
    entry_distance_bands,
    participation_profile,
    trading_standard_profile,
    volatility_profile,
)
from ..models import DirectionScore, SymbolReport


EXECUTION_GATE = {
    "min_score_gap": 8.0,
    "min_htf_context": 60.0,
    "min_ltf_trigger": 65.0,
    "min_entry_quality": 65.0,
    "min_risk_quality": 60.0,
    "min_rr": 1.65,
    "scalp_min_rr": 1.45,
    "strict_min_rr": 1.8,
    "max_entry_distance_pct": 0.3,
    "min_data_completeness": 70.0,
    "min_derivatives_context": 55.0,
    "limit_min_selection_score": 72.0,
    "limit_min_execution_score": 63.0,
    "limit_min_setup_score": 74.0,
    "limit_min_ltf_trigger": 55.0,
    "limit_min_entry_quality": 54.0,
}

STANDOUT_GATE = {
    "min_selection_score": 85.0,
    "min_execution_score": 88.0,
    "min_score_gap": 12.0,
    "min_rr": 2.0,
    "max_entry_distance_pct": 0.15,
    "min_htf_context": 65.0,
    "min_ltf_trigger": 70.0,
    "min_entry_quality": 70.0,
}

def execution_gate_profile(report: SymbolReport) -> dict[str, float | str]:
    profile = trading_standard_profile(report.symbol)
    gate: dict[str, float | str] = dict(EXECUTION_GATE)
    gate.update(
        {
            "instrument_class": profile.instrument_class,
            "min_score_gap": profile.min_score_gap,
            "min_htf_context": profile.min_htf_context,
            "min_ltf_trigger": profile.min_ltf_trigger,
            "min_entry_quality": profile.min_entry_quality,
            "min_risk_quality": profile.min_risk_quality,
            "min_rr": profile.min_rr,
            "scalp_min_rr": profile.scalp_min_rr,
            "strict_min_rr": profile.strict_min_rr,
            "max_entry_distance_pct": profile.max_entry_distance_pct,
            "min_data_completeness": profile.min_core_data_quality,
            "min_core_data_quality": profile.min_core_data_quality,
            "min_derivatives_context": profile.min_derivatives_context,
            "limit_min_selection_score": profile.limit_min_selection_score,
            "limit_min_execution_score": profile.limit_min_execution_score,
            "limit_min_setup_score": profile.limit_min_setup_score,
            "limit_min_ltf_trigger": profile.limit_min_ltf_trigger,
            "limit_min_entry_quality": profile.limit_min_entry_quality,
            "market_min_execution_score": profile.market_min_execution_score,
            "funding_warm": profile.funding_warm,
            "funding_elevated": profile.funding_elevated,
            "funding_extreme": profile.funding_extreme,
            "oi_hot_change_pct": profile.oi_hot_change_pct,
            "oi_extreme_change_pct": profile.oi_extreme_change_pct,
            "crowding_block_score": profile.crowding_block_score,
            "exhaustion_block_score": profile.exhaustion_block_score,
        }
    )
    return gate


ACTION_LABELS = {
    "market": "可以做",
    "limit": "可以做",
    "watch": "觀察",
    "avoid": "不能做",
}

OPTIONAL_SCORE_FEATURES = {"trendline", "amd", "nexus", "paid_data"}

ACTION_LABELS = {
    "market": "可市價",
    "limit": "可掛限價",
    "watch": "觀察",
    "avoid": "不能做",
}

EXECUTION_STATUS_LABELS = {
    "EXECUTABLE_MARKET": "可市價",
    "EXECUTABLE_LIMIT": "可掛限價",
    "ARMED_WAIT_ENTRY": "等待入場",
    "WATCH": "觀察",
    "BLOCKED_RISK": "風險阻擋",
    "MISSED": "已錯過",
    "INVALID": "無效",
}

LIMIT_ENTRY_ORIGINS = {"fvg", "ote", "order_block", "retest", "validated_pullback"}
GATE_VERSION = "balanced_strict_2026_06_v1"


def direction_label(direction: str) -> str:
    return {"long": "看多", "short": "看空", "neutral": "觀察"}.get(direction, direction)


def side_score(side: DirectionScore) -> float:
    if side.selection_score is not None:
        return side.selection_score
    if side.calibrated_score is not None:
        return side.calibrated_score
    return side.normalized


def selected_side(report: SymbolReport) -> DirectionScore:
    if report.selected_direction == "short":
        return report.short
    if report.selected_direction == "neutral" and side_score(report.short) > side_score(report.long):
        return report.short
    return report.long


def execution_score(side: DirectionScore, fallback: float) -> float:
    return side.execution_score if side.execution_score is not None else fallback


def entry_distance_pct(price: float, entry_zone: tuple[float, float] | None) -> float | None:
    if not entry_zone:
        return None
    low, high = entry_zone
    if low <= price <= high:
        return 0.0
    return min(abs(price - low), abs(price - high)) / max(abs(price), 1e-12) * 100.0


def dynamic_entry_band_pct(report: SymbolReport, side: DirectionScore) -> float:
    atr_pct = _as_float(side.market_metrics.get("atr_pct")) or 0.0
    spread_pct = _spread_pct(report)
    base = entry_distance_bands(report.symbol, atr_pct, spread_pct)["execution"]
    adaptive = _as_float(side.market_metrics.get("adaptive_entry_band_pct"))
    return round(max(base, adaptive or 0.0), 4)


def entry_proximity_state(report: SymbolReport, side: DirectionScore) -> dict[str, Any]:
    distance = entry_distance_pct(report.price, side.entry_zone)
    band = dynamic_entry_band_pct(report, side)
    if distance is None:
        return {"state": "no_entry_zone", "distance_pct": None, "dynamic_band_pct": band, "passes": False}
    if distance <= band:
        state = "near_entry"
    elif distance <= band * 2.5:
        state = "approaching_entry"
    elif distance <= band * 7:
        state = "far_from_entry"
    else:
        state = "missed"
    return {
        "state": state,
        "distance_pct": round(distance, 4),
        "dynamic_band_pct": band,
        "passes": distance <= band,
    }


def market_momentum_execution_profile(
    report: SymbolReport,
    side: DirectionScore,
    diag: dict[str, Any],
    gate: dict[str, Any],
    distance: float | None,
    dynamic_band: float,
    exec_score: float,
    rr: float,
    required_rr: float,
) -> dict[str, Any]:
    kind = str(gate.get("instrument_class") or volatility_profile(report.symbol).instrument_class)
    distance_bands = entry_distance_bands(report.symbol, _as_float(side.market_metrics.get("atr_pct")), _spread_pct(report))
    if kind == "altcoin":
        distance_limit = min(distance_bands["caution"], max(dynamic_band * 1.65, 0.24))
        spread_limit = 0.14
        ltf_floor = max(float(gate["min_ltf_trigger"]) + 10.0, 78.0)
        min_quote_volume = 30_000_000.0
    elif kind == "large_altcoin":
        distance_limit = min(distance_bands["caution"], max(dynamic_band * 1.45, 0.18))
        spread_limit = 0.10
        ltf_floor = max(float(gate["min_ltf_trigger"]) + 8.0, 76.0)
        min_quote_volume = 50_000_000.0
    elif kind == "core_crypto":
        distance_limit = min(distance_bands["caution"], max(dynamic_band * 1.20, 0.08))
        spread_limit = 0.06
        ltf_floor = max(float(gate["min_ltf_trigger"]) + 8.0, 74.0)
        min_quote_volume = 80_000_000.0
    else:
        distance_limit = min(distance_bands["caution"], max(dynamic_band * 1.20, 0.08))
        spread_limit = 0.08
        ltf_floor = max(float(gate["min_ltf_trigger"]) + 6.0, 72.0)
        min_quote_volume = 20_000_000.0

    spread = _spread_pct(report)
    execution_floor = float(gate["market_min_execution_score"])
    if diag["ltf_trigger"] >= ltf_floor + 8.0 and spread <= spread_limit * 0.75:
        execution_floor -= 2.0
    checks = {
        "distance": distance is not None and distance <= distance_limit,
        "ltf_momentum": diag["ltf_trigger"] >= ltf_floor,
        "entry_structure": diag["entry_quality"] >= float(gate["limit_min_entry_quality"]),
        "execution_score": exec_score >= execution_floor,
        "rr": rr >= max(required_rr, float(gate["min_rr"])),
        "spread": spread <= spread_limit,
        "liquidity": float(report.quote_volume_24h or 0.0) >= min_quote_volume,
        "market_behavior": bool(diag.get("market_behavior_ok", True)),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "distance_limit": round(distance_limit, 4),
        "spread_pct": round(spread, 4),
        "spread_limit": spread_limit,
        "ltf_floor": round(ltf_floor, 2),
        "execution_floor": round(execution_floor, 2),
        "failed": [key for key, value in checks.items() if not value],
    }


def precision_limit_execution_profile(
    report: SymbolReport,
    side: DirectionScore,
    diag: dict[str, Any],
    gate: dict[str, Any],
    distance: float | None,
    dynamic_band: float,
    score: float,
    setup_score: float,
    exec_score: float,
    rr: float,
    required_rr: float,
    score_gap: float,
) -> dict[str, Any]:
    """A stricter limit-order route that does not require market-order behavior.

    Market orders need full market_behavior confirmation because they cross the
    spread immediately. A limit order placed inside a validated ICT entry zone
    can be executable with less live-flow evidence, but only when the plan,
    location, context, and directional flow are all strong enough.
    """

    behavior = diag.get("market_behavior_confirmation", {})
    if not isinstance(behavior, dict):
        behavior = {}
    heat = diag.get("derivatives_heat_profile", {})
    if not isinstance(heat, dict):
        heat = {}
    kind = str(gate.get("instrument_class") or volatility_profile(report.symbol).instrument_class)
    behavior_blockers = [str(item) for item in behavior.get("blockers", []) if item]
    flow_alignment = _as_float(heat.get("flow_alignment"))
    depth_alignment = _as_float(heat.get("depth_alignment"))
    behavior_score = _as_float(diag.get("market_behavior_score")) or _as_float(behavior.get("score")) or 0.0
    directional_support = _as_float(behavior.get("directional_support")) or 0.0
    evidence_count = int(behavior.get("evidence_count") or 0)
    quote_volume = _as_float(getattr(report, "quote_volume_24h", None)) or 0.0
    spread = _spread_pct(report)

    severe_behavior_block = any(
        needle in blocker
        for blocker in behavior_blockers
        for needle in (
            "taker flow weak/rejecting",
            "external bias",
            "derivatives hard block",
            "crowding",
            "exhaustion",
        )
    )
    flow_supports = bool(behavior.get("ok")) or (
        behavior_score >= 76.0
        and flow_alignment is not None
        and flow_alignment >= (0.62 if kind == "altcoin" else 0.58)
    )
    depth_not_hostile = depth_alignment is None or depth_alignment >= 0.24 or behavior_score >= 82.0
    selection_floor = max(68.0, float(gate["limit_min_selection_score"]) - (6.0 if kind == "altcoin" else 5.0))
    setup_floor = max(70.0, float(gate["limit_min_setup_score"]) - (5.0 if kind == "altcoin" else 4.0))
    execution_floor = max(63.0, float(gate["limit_min_execution_score"]) - 2.0)
    ltf_floor = max(52.0, float(gate["limit_min_ltf_trigger"]) - (5.0 if kind == "altcoin" else 4.0))
    htf_floor = float(gate["min_htf_context"]) + (10.0 if kind == "altcoin" else 8.0)
    context_floor = float(gate["min_derivatives_context"]) + (4.0 if kind == "altcoin" else 3.0)
    liquidity_floor = 50_000_000.0 if kind == "altcoin" else 40_000_000.0 if kind == "large_altcoin" else 20_000_000.0
    limit_distance_ok = distance is not None and distance <= dynamic_band * 2.5
    strong_flow_limit = (
        kind == "altcoin"
        and score >= 69.0
        and setup_score >= 70.0
        and exec_score >= 53.0
        and diag["htf_context"] >= 69.0
        and diag["ltf_trigger"] >= 60.0
        and diag["entry_quality"] >= 64.0
        and diag["risk_reward_quality"] >= 80.0
        and rr >= required_rr
        and limit_distance_ok
        and quote_volume >= 20_000_000.0
        and spread <= 0.06
        and behavior_score >= 78.0
        and flow_alignment is not None
        and flow_alignment >= 0.66
        and diag["derivatives_context_score"] >= float(gate["min_derivatives_context"]) + 3.5
        and depth_not_hostile
        and not severe_behavior_block
        and not diag["derivative_blocked"]
        and not diag["market_overheated"]
    )
    liquid_flow_limit = (
        kind in {"altcoin", "large_altcoin"}
        and score >= 64.0
        and setup_score >= 60.0
        and exec_score >= 52.0
        and diag["htf_context"] >= 60.0
        and diag["ltf_trigger"] >= 45.0
        and diag["entry_quality"] >= 60.0
        and diag["risk_reward_quality"] >= 80.0
        and rr >= required_rr
        and limit_distance_ok
        and quote_volume >= 80_000_000.0
        and spread <= 0.08
        and behavior_score >= 80.0
        and flow_alignment is not None
        and flow_alignment >= (0.62 if kind == "altcoin" else 0.58)
        and diag["derivatives_context_score"] >= context_floor
        and depth_not_hostile
        and not severe_behavior_block
        and not diag["derivative_blocked"]
        and not diag["market_overheated"]
    )
    elite_structure_limit = (
        kind == "altcoin"
        and score >= 75.0
        and setup_score >= 74.0
        and exec_score >= 58.0
        and diag["htf_context"] >= 90.0
        and diag["ltf_trigger"] >= 65.0
        and diag["entry_quality"] >= 64.0
        and diag["risk_reward_quality"] >= 80.0
        and rr >= required_rr
        and limit_distance_ok
        and quote_volume >= 20_000_000.0
        and spread <= 0.06
        and behavior_score >= 78.0
        and flow_alignment is not None
        and flow_alignment >= 0.55
        and diag["derivatives_context_score"] >= float(gate["min_derivatives_context"])
        and depth_not_hostile
        and not severe_behavior_block
        and not diag["derivative_blocked"]
        and not diag["market_overheated"]
    )
    checks = {
        "selected_direction": report.selected_direction in {"long", "short"},
        "score_gap": score_gap >= float(gate["min_score_gap"]),
        "score": score >= selection_floor or strong_flow_limit or liquid_flow_limit,
        "setup_score": setup_score >= setup_floor or strong_flow_limit or liquid_flow_limit,
        "execution_score": exec_score >= execution_floor or elite_structure_limit or strong_flow_limit or liquid_flow_limit,
        "htf_context": diag["htf_context"] >= htf_floor or strong_flow_limit or liquid_flow_limit,
        "ltf_trigger": diag["ltf_trigger"] >= ltf_floor or liquid_flow_limit,
        "entry_quality": diag["entry_quality"] >= 60.0,
        "risk_quality": diag["risk_reward_quality"] >= max(float(gate["min_risk_quality"]), 70.0),
        "rr": rr >= required_rr,
        "entry_distance": limit_distance_ok,
        "complete_trade_plan": bool(side.entry_zone and side.stop is not None and side.take_profits),
        "external_derivatives": bool(diag.get("external_api_ok")),
        "derivatives_context": diag["derivatives_context_score"] >= context_floor or elite_structure_limit,
        "configured_api_ready": bool(diag.get("configured_api_ready", True)),
        "derivatives_not_overheated": not diag["derivative_blocked"],
        "market_not_overheated": not diag["market_overheated"],
        "data_completeness": diag["core_data_quality"] >= gate["min_core_data_quality"],
        "direction_not_conflicted": not hard_direction_conflict(report),
        "flow_supports": flow_supports or elite_structure_limit,
        "depth_not_hostile": depth_not_hostile,
        "no_severe_behavior_block": not severe_behavior_block,
        "liquidity": quote_volume >= liquidity_floor or elite_structure_limit or strong_flow_limit or liquid_flow_limit,
        "evidence_seen": evidence_count >= 1 or external_derivatives_available(report),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "behavior_score": round(behavior_score, 2),
        "directional_support": round(directional_support, 2),
        "flow_alignment": round(flow_alignment, 4) if flow_alignment is not None else None,
        "depth_alignment": round(depth_alignment, 4) if depth_alignment is not None else None,
        "selection_floor": round(selection_floor, 2),
        "setup_floor": round(setup_floor, 2),
        "execution_floor": round(execution_floor, 2),
        "ltf_floor": round(ltf_floor, 2),
        "htf_floor": round(htf_floor, 2),
        "context_floor": round(context_floor, 2),
        "strong_flow_limit": strong_flow_limit,
        "liquid_flow_limit": liquid_flow_limit,
        "elite_structure_limit": elite_structure_limit,
        "failed": [key for key, value in checks.items() if not value],
    }


def visible_warnings(side: DirectionScore) -> list[str]:
    output: list[str] = []
    for warning in side.warnings:
        if warning and warning not in output:
            output.append(warning)
    return output


def hard_direction_conflict(report: SymbolReport) -> bool:
    if report.selected_direction == "neutral" and report.metadata.get("direction_conflict"):
        return True
    analysis = report.metadata.get("direction_analysis", {})
    if not isinstance(analysis, dict):
        return False
    return analysis.get("chosen_direction") == "neutral" and analysis.get("conflict_level") == "high"


def _feature_ratio(side: DirectionScore, names: list[str]) -> float:
    total = sum(side.feature_scores.get(name, 0.0) for name in names)
    max_total = sum(side.feature_max_scores.get(name, 0.0) for name in names)
    if max_total <= 0:
        return 0.0
    return max(0.0, min(100.0, total / max_total * 100.0))


def core_data_quality(report: SymbolReport, side: DirectionScore | None = None) -> float:
    side = side or selected_side(report)
    coverage = report.data_coverage or {}
    requirements = {"4h": 60, "1h": 80, "15m": 96, "5m": 120}
    weights = {"4h": 22.0, "1h": 18.0, "15m": 25.0, "5m": 15.0}
    score = 0.0
    for tf, required in requirements.items():
        count = float(coverage.get(tf) or 0.0)
        score += min(1.0, count / required) * weights[tf]
    if side.entry_zone and side.stop is not None and side.take_profits:
        score += 12.0
    elif side.entry_zone and side.stop is not None:
        score += 8.0
    elif side.entry_zone:
        score += 5.0
    values = _paid_values(report)
    if isinstance(values.get("exchange_public_derivatives"), dict):
        score += 8.0
    elif report.quote_volume_24h >= 20_000_000:
        score += 4.0
    return round(max(0.0, min(100.0, score)), 2)


def _paid_values(report: SymbolReport) -> dict[str, Any]:
    paid = report.metadata.get("paid_data", {})
    values = paid.get("values", {}) if isinstance(paid, dict) else {}
    return values if isinstance(values, dict) else {}


def _external_strategy_context(report: SymbolReport) -> dict[str, Any]:
    context = _paid_values(report).get("external_strategy_context", {})
    return context if isinstance(context, dict) else {}


def paid_data_status(report: SymbolReport) -> dict[str, Any]:
    paid = report.metadata.get("paid_data", {})
    providers = paid.get("providers", []) if isinstance(paid, dict) else []
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives", {})
    if not isinstance(public, dict):
        public = {}
    blocked, warning = derivative_risk(report, report.selected_direction)
    context = derivatives_context(report)
    heat_profile = derivatives_heat_profile(report, report.selected_direction)
    external_context = _external_strategy_context(report)
    configured = configured_api_readiness(report)
    return {
        "derivatives_available": external_derivatives_available(report),
        "providers": providers,
        "funding_rate": _as_float(public.get("funding_rate")),
        "funding_time": public.get("funding_time"),
        "open_interest": _as_float(public.get("open_interest")),
        "open_interest_time": public.get("open_interest_time"),
        "open_interest_previous_time": public.get("open_interest_previous_time"),
        "open_interest_change_pct": _as_float(public.get("open_interest_change_pct")),
        "fetched_at": public.get("fetched_at"),
        "blocked": blocked,
        "warning": warning,
        "context_score": context["score"],
        "context_method": context["method"],
        "context_reason": context["reason"],
        "heat_profile": heat_profile,
        "heat_state": heat_profile["state"],
        "heat_score": heat_profile["heat_score"],
        "trend_confirmation_score": heat_profile["trend_confirmation_score"],
        "crowding_score": heat_profile["crowding_score"],
        "exhaustion_score": heat_profile["exhaustion_score"],
        "external_strategy_bias": external_context.get("bias"),
        "external_strategy_confidence": external_context.get("confidence"),
        "external_strategy_long_score": external_context.get("long_score"),
        "external_strategy_short_score": external_context.get("short_score"),
        "external_strategy_risk_flags": external_context.get("risk_flags", []),
        "configured_api_readiness": configured,
        "configured_api_ready": configured["execution_ready"],
    }


def external_derivatives_available(report: SymbolReport) -> bool:
    paid = report.metadata.get("paid_data", {})
    providers = paid.get("providers", []) if isinstance(paid, dict) else []
    values = _paid_values(report)
    if isinstance(values.get("exchange_public_derivatives"), dict):
        return True
    provider_text = " ".join(str(provider) for provider in providers).lower()
    return any(name in provider_text for name in ("exchange", "bybit", "binance", "coinglass", "coinalyze"))


def configured_api_readiness(report: SymbolReport) -> dict[str, Any]:
    paid = report.metadata.get("paid_data", {})
    readiness = paid.get("configured_api_readiness", {}) if isinstance(paid, dict) else {}
    if not isinstance(readiness, dict):
        readiness = {}
    execution_ready = bool(readiness.get("execution_ready", True))
    return {
        "configured": list(readiness.get("configured", [])) if isinstance(readiness.get("configured"), list) else [],
        "execution_required": list(readiness.get("execution_required", []))
        if isinstance(readiness.get("execution_required"), list)
        else [],
        "execution_ready": execution_ready,
        "all_configured_reached": bool(readiness.get("all_configured_reached", execution_ready)),
        "execution_missing": list(readiness.get("execution_missing", []))
        if isinstance(readiness.get("execution_missing"), list)
        else [],
        "advisory_missing": list(readiness.get("advisory_missing", []))
        if isinstance(readiness.get("advisory_missing"), list)
        else [],
        "configured_failed": list(readiness.get("configured_failed", []))
        if isinstance(readiness.get("configured_failed"), list)
        else [],
        "status": readiness.get("status", {}) if isinstance(readiness.get("status"), dict) else {},
    }


def derivatives_heat_profile(report: SymbolReport, direction: str | None = None) -> dict[str, Any]:
    return _derivatives_heat_profile(report, direction or report.selected_direction)


def _derivatives_heat_profile(report: SymbolReport, direction: str) -> dict[str, Any]:
    side = selected_side(report)
    effective_direction = direction if direction in {"long", "short"} else side.direction
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives", {})
    if not isinstance(public, dict):
        public = {}
    external_context = _external_strategy_context(report)
    flags = set(external_context.get("risk_flags", [])) if isinstance(external_context.get("risk_flags"), list) else set()
    heat = 0.0
    trend_confirmation = 0.0
    crowding = 0.0
    exhaustion = 0.0
    reasons: list[str] = []
    warnings: list[str] = []

    def add_reason(text: str) -> None:
        if text and text not in reasons:
            reasons.append(text)

    def add_warning(text: str) -> None:
        if text and text not in warnings:
            warnings.append(text)

    funding = _as_float(public.get("funding_rate"))
    open_interest = _as_float(public.get("open_interest"))
    oi_change = _as_float(public.get("open_interest_change_pct"))
    spread = _spread_pct(report)
    flow_ratio = _flow_ratio_from_public(public)
    flow_alignment = _directional_ratio(flow_ratio, effective_direction)
    depth_alignment = _orderbook_alignment(public, effective_direction)
    context_score = _context_direction_score(external_context, effective_direction)
    has_context = int(external_context.get("evidence_count") or 0) > 0
    external_bias = str(external_context.get("bias") or "neutral")
    external_confidence = _as_float(external_context.get("confidence")) or 0.0
    entry_distance = entry_distance_pct(report.price, side.entry_zone)
    dynamic_band = dynamic_entry_band_pct(report, side)
    gate = execution_gate_profile(report)
    instrument_kind = str(gate["instrument_class"])
    funding_warm = float(gate["funding_warm"])
    funding_elevated = float(gate["funding_elevated"])
    funding_extreme = float(gate["funding_extreme"])
    oi_hot = float(gate["oi_hot_change_pct"])
    oi_extreme = float(gate["oi_extreme_change_pct"])
    crowding_block = float(gate["crowding_block_score"])
    exhaustion_block = float(gate["exhaustion_block_score"])

    if open_interest is not None:
        heat += 4.0
    if report.quote_volume_24h >= 250_000_000:
        heat += 7.0
    elif report.quote_volume_24h >= 80_000_000:
        heat += 5.0
    elif report.quote_volume_24h >= 20_000_000:
        heat += 3.0
    if spread <= 0.08:
        heat += 3.0

    funding_same_side = False
    funding_abs = abs(funding) if funding is not None else 0.0
    if funding is not None and effective_direction in {"long", "short"}:
        funding_side = "long" if funding > 0 else "short" if funding < 0 else "neutral"
        funding_same_side = funding_side == effective_direction
        if funding_abs >= funding_warm:
            heat += min(12.0, 2.0 + funding_abs / max(funding_elevated, 1e-12) * 4.0)
        if funding_same_side:
            if funding_abs >= funding_extreme:
                crowding += 34.0
                exhaustion += 8.0
                add_warning(f"same-side funding is extreme ({funding:.5f})")
            elif funding_abs >= funding_elevated:
                crowding += 16.0
                add_warning(f"same-side funding is elevated ({funding:.5f})")
            elif funding_abs >= funding_warm:
                add_reason(f"same-side funding is warm ({funding:.5f}), treated as participation")
        elif funding_abs >= funding_warm:
            trend_confirmation += min(8.0, 2.0 + funding_abs / max(funding_elevated, 1e-12) * 3.0)
            add_reason(f"opposite-side funding ({funding:.5f}) can be squeeze fuel")

    if oi_change is not None:
        abs_oi = abs(oi_change)
        if abs_oi >= max(5.0, oi_hot * 0.28):
            heat += min(30.0, abs_oi * 1.05)
        if oi_change > 0:
            if flow_alignment is not None and flow_alignment >= 0.56:
                trend_confirmation += min(24.0, 5.0 + abs_oi * 0.70)
                add_reason(f"OI +{oi_change:.1f}% expands with directional taker flow")
            elif has_context and context_score >= 62.0:
                trend_confirmation += min(18.0, 4.0 + abs_oi * 0.50)
                add_reason(f"OI +{oi_change:.1f}% expands with external context support")
            elif flow_alignment is not None and flow_alignment <= 0.44:
                crowding += min(22.0, 4.0 + abs_oi * 0.55)
                exhaustion += min(16.0, 2.0 + abs_oi * 0.35)
                add_warning(f"OI +{oi_change:.1f}% expands against selected flow")
            elif abs_oi >= oi_hot:
                crowding += min(14.0, abs_oi * 0.35)
                add_warning(f"OI +{oi_change:.1f}% is hot but not directionally confirmed")
            else:
                trend_confirmation += 2.0
                add_reason(f"OI +{oi_change:.1f}% is participation, not a standalone veto")
        elif oi_change < -8.0:
            exhaustion += min(16.0, abs_oi * 0.40)
            add_warning(f"OI {oi_change:.1f}% shows deleveraging; continuation needs fresh positioning")

    if flow_alignment is not None:
        if flow_alignment >= 0.62:
            trend_confirmation += 12.0
            heat += 6.0
            add_reason(f"taker flow aligns {flow_alignment:.2f}")
        elif flow_alignment >= 0.56:
            trend_confirmation += 6.0
            heat += 3.0
        elif flow_alignment <= 0.38:
            exhaustion += 14.0
            crowding += 6.0
            add_warning(f"taker flow rejects selected side ({flow_alignment:.2f})")
        elif flow_alignment <= 0.44:
            exhaustion += 7.0
            add_warning(f"taker flow is soft against selected side ({flow_alignment:.2f})")

    if depth_alignment is not None:
        if depth_alignment >= 0.60:
            trend_confirmation += 4.0
            add_reason(f"orderbook depth supports selected side {depth_alignment:.2f}")
        elif depth_alignment <= 0.40:
            exhaustion += 4.0
            add_warning(f"orderbook depth leans against selected side {depth_alignment:.2f}")

    if has_context:
        if context_score >= 70.0:
            trend_confirmation += min(18.0, (context_score - 58.0) * 0.55)
        elif context_score <= 45.0:
            exhaustion += min(18.0, (50.0 - context_score) * 0.75)
        if external_bias in {"long", "short"}:
            if external_bias == effective_direction and external_confidence >= 64.0:
                trend_confirmation += min(10.0, (external_confidence - 56.0) * 0.35)
                add_reason(f"external bias confirms {effective_direction} ({external_confidence:.1f})")
            elif external_bias != effective_direction and external_confidence >= 64.0:
                crowding += min(18.0, 6.0 + (external_confidence - 64.0) * 0.45)
                exhaustion += min(16.0, 4.0 + (external_confidence - 64.0) * 0.40)
                add_warning(f"external bias={external_bias} conflicts with selected {effective_direction}")

    if effective_direction == "long" and "long_crowded" in flags:
        crowding += 24.0
        add_warning("external positioning flags crowded longs")
    if effective_direction == "short" and "short_crowded" in flags:
        crowding += 24.0
        add_warning("external positioning flags crowded shorts")
    if "derivatives_hot" in flags:
        crowding += 12.0
        exhaustion += 8.0
        add_warning("external derivatives flag is hot")
    if effective_direction == "long" and "stop_hunt_risk_long" in flags:
        exhaustion += 10.0
        add_warning("liquidation/support magnet is close below long entry")
    if effective_direction == "short" and "stop_hunt_risk_short" in flags:
        exhaustion += 10.0
        add_warning("liquidation/resistance magnet is close above short entry")

    atr_pct = _as_float(side.market_metrics.get("atr_pct"))
    volume_ratio = _as_float(side.market_metrics.get("volume_ratio"))
    btc_fast_pct = _as_float(side.market_metrics.get("btc_fast_pct"))
    btc_trend_pct = _as_float(side.market_metrics.get("btc_trend_pct"))
    btc_against = bool(side.market_metrics.get("btc_against"))
    if atr_pct is not None:
        vol_profile = volatility_profile(report.symbol)
        if vol_profile.active_low_atr_pct <= atr_pct <= vol_profile.active_high_atr_pct:
            heat += 5.0
            if vol_profile.instrument_class in {"altcoin", "large_altcoin"} and atr_pct >= vol_profile.active_high_atr_pct * 0.75:
                trend_confirmation += 2.0
        elif atr_pct > vol_profile.active_high_atr_pct:
            heat += 8.0
            if atr_pct >= vol_profile.extreme_atr_pct:
                exhaustion += 18.0
            elif trend_confirmation < 35.0:
                exhaustion += 7.0
    if volume_ratio is not None:
        flow_profile = participation_profile(report.symbol)
        if flow_profile.active_low_volume_ratio <= volume_ratio <= flow_profile.active_high_volume_ratio:
            heat += 6.0
            trend_confirmation += 3.0
        elif volume_ratio > flow_profile.active_high_volume_ratio:
            heat += 8.0
            if flow_alignment is not None and flow_alignment >= 0.58:
                trend_confirmation += 4.0
            else:
                exhaustion += min(16.0, max(0.0, volume_ratio - flow_profile.active_high_volume_ratio) * 4.0)
    if btc_trend_pct is not None and effective_direction in {"long", "short"}:
        btc_aligned = (effective_direction == "long" and btc_trend_pct >= 0) or (effective_direction == "short" and btc_trend_pct <= 0)
        if instrument_kind == "altcoin":
            btc_alignment_bonus = 3.0
            btc_conflict_penalty = 4.0
        elif instrument_kind == "large_altcoin":
            btc_alignment_bonus = 4.0
            btc_conflict_penalty = 6.0
        else:
            btc_alignment_bonus = 6.0
            btc_conflict_penalty = 10.0
        if btc_aligned:
            trend_confirmation += btc_alignment_bonus
        else:
            exhaustion += btc_conflict_penalty
    if btc_against:
        if instrument_kind == "altcoin":
            exhaustion += 6.0
        elif instrument_kind == "large_altcoin":
            exhaustion += 8.0
        else:
            exhaustion += 14.0
    if btc_fast_pct is not None and effective_direction in {"long", "short"}:
        fast_aligned = (effective_direction == "long" and btc_fast_pct >= 0) or (effective_direction == "short" and btc_fast_pct <= 0)
        if abs(btc_fast_pct) >= 2.2:
            if instrument_kind == "altcoin":
                heat += min(6.0, abs(btc_fast_pct) * 1.2)
            elif instrument_kind == "large_altcoin":
                heat += min(8.0, abs(btc_fast_pct) * 1.5)
            else:
                heat += min(10.0, abs(btc_fast_pct) * 2.0)
            if not fast_aligned:
                if instrument_kind == "altcoin":
                    exhaustion += 6.0
                elif instrument_kind == "large_altcoin":
                    exhaustion += 8.0
                else:
                    exhaustion += 14.0
    if entry_distance is not None:
        if entry_distance > dynamic_band * 2.5:
            exhaustion += 12.0
            add_warning(f"price is {entry_distance:.2f}% from entry; heat is chase risk")
        elif entry_distance > dynamic_band:
            exhaustion += 5.0

    hard_block = False
    if funding_same_side and funding_abs >= funding_extreme and trend_confirmation < 35.0:
        hard_block = True
    if oi_change is not None and abs(oi_change) >= oi_extreme and crowding >= 50.0 and trend_confirmation < 40.0:
        hard_block = True
    if crowding >= crowding_block and exhaustion >= max(55.0, exhaustion_block - 25.0):
        hard_block = True
    if (
        effective_direction in {"long", "short"}
        and external_bias in {"long", "short"}
        and external_bias != effective_direction
        and external_confidence >= 78.0
        and context_score <= 42.0
    ):
        hard_block = True

    heat_score = _clamp(heat)
    trend_score = _clamp(trend_confirmation)
    crowding_score = _clamp(crowding)
    exhaustion_score = _clamp(exhaustion)
    if hard_block:
        state = "blocked_crowding"
    elif crowding_score >= crowding_block - 12.0 or exhaustion_score >= exhaustion_block - 12.0:
        state = "exhausted"
    elif heat_score >= 35.0 and trend_score >= 42.0 and crowding_score < 60.0:
        state = "healthy_heat"
    elif heat_score >= 25.0:
        state = "warm"
    else:
        state = "quiet"
    return {
        "state": state,
        "instrument_class": gate["instrument_class"],
        "heat_score": round(heat_score, 2),
        "trend_confirmation_score": round(trend_score, 2),
        "crowding_score": round(crowding_score, 2),
        "exhaustion_score": round(exhaustion_score, 2),
        "flow_alignment": round(flow_alignment, 4) if flow_alignment is not None else None,
        "depth_alignment": round(depth_alignment, 4) if depth_alignment is not None else None,
        "context_direction_score": round(context_score, 2),
        "hard_block": hard_block,
        "reasons": reasons[:8],
        "warnings": warnings[:8],
    }


def _legacy_derivative_risk(report: SymbolReport, direction: str) -> tuple[bool, str]:
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives", {})
    if not isinstance(public, dict):
        public = {}
    funding = _as_float(public.get("funding_rate"))
    oi_change = _as_float(public.get("open_interest_change_pct"))
    warnings: list[str] = []
    blocked = False
    if funding is not None:
        if direction == "long" and funding > 0.00035:
            blocked = True
            warnings.append(f"funding={funding:.5f}，多頭資金費率過熱")
        if direction == "short" and funding < -0.00035:
            blocked = True
            warnings.append(f"funding={funding:.5f}，空頭資金費率過熱")
    if oi_change is not None and abs(oi_change) >= 18:
        blocked = True
        warnings.append(f"OI 近 1h 變動 {oi_change:.2f}%，槓桿流動過熱")
    return blocked, "；".join(warnings)


def derivative_risk(report: SymbolReport, direction: str) -> tuple[bool, str]:
    profile = _derivatives_heat_profile(report, direction)
    gate = execution_gate_profile(report)
    warnings = list(profile.get("warnings", []))
    blocked = bool(profile.get("hard_block"))
    crowding_block = float(gate["crowding_block_score"])
    exhaustion_block = float(gate["exhaustion_block_score"])
    if not blocked and profile["crowding_score"] >= crowding_block and profile["trend_confirmation_score"] < 45.0:
        blocked = True
        warnings.append("crowding is extreme without enough trend confirmation")
    if not blocked and profile["exhaustion_score"] >= exhaustion_block and profile["crowding_score"] >= 55.0:
        blocked = True
        warnings.append("derivatives heat looks exhausted instead of tradable")
    if profile["state"] == "healthy_heat":
        warnings.append("derivatives heat is aligned; treat as trend fuel, not a veto")
    return blocked, "; ".join(warnings)


def market_risk(report: SymbolReport) -> tuple[bool, str]:
    side = selected_side(report)
    metrics = side.market_metrics or {}
    direction = report.selected_direction if report.selected_direction in {"long", "short"} else side.direction
    profile = _derivatives_heat_profile(report, direction)
    distance = entry_distance_pct(report.price, side.entry_zone)
    band = dynamic_entry_band_pct(report, side)
    healthy_heat = (
        profile["state"] == "healthy_heat"
        or (profile["trend_confirmation_score"] >= 45.0 and profile["crowding_score"] < 60.0)
    )
    far_from_entry = distance is not None and distance > band * 2.5
    atr_pct_new = _as_float(metrics.get("atr_pct"))
    volume_ratio_new = _as_float(metrics.get("volume_ratio"))
    btc_fast_pct_new = _as_float(metrics.get("btc_fast_pct"))
    btc_trend_pct_new = _as_float(metrics.get("btc_trend_pct"))
    btc_corr_new = _as_float(metrics.get("btc_corr"))
    btc_against_new = bool(metrics.get("btc_against"))
    vol_profile = volatility_profile(report.symbol)
    kind = vol_profile.instrument_class
    flow_alignment = _as_float(profile.get("flow_alignment"))
    depth_alignment = _as_float(profile.get("depth_alignment"))
    quote_volume = _as_float(getattr(report, "quote_volume_24h", None)) or 0.0
    flow_profile = participation_profile(report.symbol)
    near_entry = distance is not None and distance <= band * 1.2
    mover_profile = str(metrics.get("mover_profile") or "normal")
    mover_model_confirmed = (
        kind in {"large_altcoin", "altcoin"}
        and mover_profile in {"hot_mover", "extreme_mover"}
        and bool(metrics.get("mover_execution_permission"))
        and not bool(metrics.get("mover_chase_risk"))
        and bool(metrics.get("entry_anchor_ok", True))
        and distance is not None
        and distance <= band * 1.8
    )
    active_participation = volume_ratio_new is not None and volume_ratio_new >= flow_profile.active_low_volume_ratio
    flow_confirms = flow_alignment is not None and flow_alignment >= (0.62 if kind == "altcoin" else 0.60)
    depth_confirms = depth_alignment is not None and depth_alignment >= (0.58 if kind == "altcoin" else 0.56)
    liquid_alt_floor = 80_000_000.0 if kind == "large_altcoin" else 50_000_000.0
    idiosyncratic_alt_confirmation = (
        kind in {"large_altcoin", "altcoin"}
        and healthy_heat
        and near_entry
        and quote_volume >= liquid_alt_floor
        and active_participation
        and flow_confirms
        and (depth_confirms or float(profile["trend_confirmation_score"]) >= 52.0)
        and float(profile["crowding_score"]) < 58.0
        and float(profile["exhaustion_score"]) < 52.0
    )
    hard_warnings: list[str] = []
    soft_warnings: list[str] = []
    if atr_pct_new is not None:
        hot_turnover_floor = 150_000_000.0 if vol_profile.instrument_class == "large_altcoin" else 250_000_000.0
        liquid_extreme_confirmed = (
            vol_profile.instrument_class in {"large_altcoin", "altcoin"}
            and quote_volume >= hot_turnover_floor
            and distance is not None
            and distance <= band
            and healthy_heat
            and flow_alignment is not None
            and flow_alignment >= 0.62
            and depth_alignment is not None
            and depth_alignment >= 0.58
            and float(profile["crowding_score"]) < 50.0
            and float(profile["exhaustion_score"]) < 45.0
        )
        liquid_extreme_confirmed = liquid_extreme_confirmed or (
            mover_model_confirmed
            and quote_volume >= (35_000_000.0 if kind == "altcoin" else 50_000_000.0)
            and float(profile["crowding_score"]) < 68.0
            and float(profile["exhaustion_score"]) < 68.0
        )
        if atr_pct_new >= vol_profile.extreme_atr_pct:
            if not liquid_extreme_confirmed:
                hard_warnings.append(
                    f"ATR%={atr_pct_new:.2f} is extreme for {vol_profile.instrument_class}; wait for a fresh base/retest"
                )
            elif mover_model_confirmed:
                soft_warnings.append(
                    f"ATR%={atr_pct_new:.2f} is extreme but matched to 3-day mover retest model; limit-only"
                )
        elif atr_pct_new > vol_profile.hot_atr_pct and (far_from_entry or btc_against_new or not healthy_heat):
            if mover_model_confirmed and not far_from_entry:
                soft_warnings.append(
                    f"ATR%={atr_pct_new:.2f} is hot for {vol_profile.instrument_class}; use planned retest zone only"
                )
            else:
                soft_warnings.append(
                    f"ATR%={atr_pct_new:.2f} is hot for {vol_profile.instrument_class} without enough confirmation; do not chase"
                )
    if volume_ratio_new is not None:
        flow_profile = participation_profile(report.symbol)
        if volume_ratio_new >= flow_profile.extreme_volume_ratio and not healthy_heat:
            hard_warnings.append(f"volume spike={volume_ratio_new:.2f} is blow-off risk")
        elif volume_ratio_new > flow_profile.hot_volume_ratio and (far_from_entry or profile["exhaustion_score"] >= 45.0):
            soft_warnings.append(f"volume spike={volume_ratio_new:.2f} needs retest confirmation")
    if btc_fast_pct_new is not None:
        fast_aligned = True
        if direction == "long":
            fast_aligned = btc_fast_pct_new >= 0
        elif direction == "short":
            fast_aligned = btc_fast_pct_new <= 0
        fast_shock = 5.2 if kind == "altcoin" else 4.6 if kind == "large_altcoin" else 4.5
        if (
            btc_against_new
            and abs(btc_fast_pct_new) >= 2.2
            and not (kind in {"large_altcoin", "altcoin"} and idiosyncratic_alt_confirmation)
        ):
            soft_warnings.append(f"BTC fast move {btc_fast_pct_new:.2f}% is against selected side; needs alt-specific flow")
        elif abs(btc_fast_pct_new) >= fast_shock and not fast_aligned:
            if kind == "core_crypto":
                hard_warnings.append(f"BTC fast move {btc_fast_pct_new:.2f}% rejects selected side")
            else:
                soft_warnings.append(f"BTC fast move {btc_fast_pct_new:.2f}% rejects selected side")
        elif abs(btc_fast_pct_new) >= 5.5 and not healthy_heat:
            hard_warnings.append(f"BTC fast move {btc_fast_pct_new:.2f}% is unstable without confirming flow")
    if btc_trend_pct_new is not None and direction in {"long", "short"}:
        trend_aligned = (direction == "long" and btc_trend_pct_new >= 0) or (direction == "short" and btc_trend_pct_new <= 0)
        conflict_threshold = 2.8 if kind == "altcoin" else 2.2 if kind == "large_altcoin" else 1.6
        force_block_threshold = conflict_threshold * 1.8
        if not trend_aligned and abs(btc_trend_pct_new) >= conflict_threshold:
            if kind in {"large_altcoin", "altcoin"}:
                systemic_shock = abs(btc_trend_pct_new) >= (7.0 if kind == "altcoin" else 6.0)
                highly_correlated = btc_corr_new is None or btc_corr_new >= (0.75 if kind == "altcoin" else 0.65)
                if systemic_shock or (not idiosyncratic_alt_confirmation and (not healthy_heat or highly_correlated)):
                    warning = (
                        f"BTC context {btc_trend_pct_new:.2f}% conflicts; alt needs confirmed relative flow"
                    )
                    if systemic_shock and highly_correlated and not healthy_heat:
                        hard_warnings.append(warning)
                    else:
                        soft_warnings.append(warning)
            elif not healthy_heat or abs(btc_trend_pct_new) >= force_block_threshold:
                hard_warnings.append(f"BTC context {btc_trend_pct_new:.2f}% conflicts with selected side")
    warnings = hard_warnings + soft_warnings
    return bool(hard_warnings), "; ".join(warnings)


def _legacy_derivatives_context(report: SymbolReport) -> dict[str, Any]:
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives", {})
    if not isinstance(public, dict):
        public = {}
    funding = _as_float(public.get("funding_rate"))
    open_interest = _as_float(public.get("open_interest"))
    oi_change = _as_float(public.get("open_interest_change_pct"))
    blocked, warning = derivative_risk(report, report.selected_direction)
    if blocked:
        return {
            "score": 0.0,
            "method": "exchange_public_derivatives",
            "available": True,
            "reason": warning or "funding/OI risk blocked",
        }
    if funding is not None or open_interest is not None or oi_change is not None:
        score = 92.0
        if funding is not None:
            score -= min(22.0, abs(funding) / 0.00035 * 14.0)
        if oi_change is not None:
            score -= min(28.0, abs(oi_change) / 18.0 * 24.0)
        else:
            score -= 8.0
        if open_interest is None:
            score -= 5.0
        time_note = _derivatives_time_note(public)
        return {
            "score": round(_clamp(score), 2),
            "method": "exchange_public_derivatives",
            "available": True,
            "reason": f"public funding/OI readable{time_note}",
        }
    proxy_score, proxy_reason = _proxy_derivatives_context(report)
    return {
        "score": proxy_score,
        "method": "liquidity_volatility_proxy",
        "available": False,
        "reason": proxy_reason,
    }


def _proxy_derivatives_context(report: SymbolReport) -> tuple[float, str]:
    side = selected_side(report)
    volume = float(report.quote_volume_24h or 0.0)
    kind = str(execution_gate_profile(report)["instrument_class"])
    if kind == "altcoin":
        volume_tiers = (150_000_000, 50_000_000, 15_000_000, 5_000_000)
    elif kind == "large_altcoin":
        volume_tiers = (300_000_000, 100_000_000, 30_000_000, 10_000_000)
    else:
        volume_tiers = (500_000_000, 250_000_000, 80_000_000, 20_000_000)
    if volume >= volume_tiers[0]:
        volume_score = 92.0
    elif volume >= volume_tiers[1]:
        volume_score = 84.0
    elif volume >= volume_tiers[2]:
        volume_score = 74.0
    elif volume >= volume_tiers[3]:
        volume_score = 62.0
    else:
        volume_score = 42.0

    spread = _spread_pct(report)
    if spread <= 0.03:
        spread_score = 90.0
    elif spread <= 0.08:
        spread_score = 75.0
    elif spread <= 0.15:
        spread_score = 58.0
    else:
        spread_score = 42.0

    atr_pct = _as_float(side.market_metrics.get("atr_pct"))
    if atr_pct is None:
        atr_score = 70.0
    else:
        vol_profile = volatility_profile(report.symbol)
        if vol_profile.active_low_atr_pct <= atr_pct <= vol_profile.active_high_atr_pct:
            atr_score = 82.0
        elif vol_profile.quiet_atr_pct <= atr_pct <= vol_profile.hot_atr_pct:
            atr_score = 65.0
        else:
            atr_score = 50.0

    volume_ratio = _as_float(side.market_metrics.get("volume_ratio"))
    if volume_ratio is None:
        flow_score = 70.0
    else:
        flow_profile = participation_profile(report.symbol)
        if volume_ratio <= flow_profile.active_high_volume_ratio:
            flow_score = 78.0
        elif volume_ratio <= flow_profile.warm_high_volume_ratio:
            flow_score = 62.0
        else:
            flow_score = 45.0

    score = volume_score * 0.42 + spread_score * 0.20 + atr_score * 0.23 + flow_score * 0.15
    if volume < volume_tiers[3]:
        score = min(score, 49.0)
    reason = (
        f"public funding/OI unreadable; proxy from 24h volume, estimated spread, ATR%, volume ratio "
        f"= {score:.1f}/100"
    )
    return round(_clamp(score), 2), reason


def _derivatives_time_note(public: dict[str, Any]) -> str:
    pieces: list[str] = []
    if public.get("open_interest_time"):
        pieces.append(f"OI as_of {public['open_interest_time']}")
    if public.get("funding_time"):
        pieces.append(f"funding as_of {public['funding_time']}")
    if public.get("fetched_at"):
        pieces.append(f"fetched_at {public['fetched_at']}")
    return "; " + ", ".join(pieces) if pieces else ""


def derivatives_context(report: SymbolReport) -> dict[str, Any]:
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives", {})
    if not isinstance(public, dict):
        public = {}
    external_context = _external_strategy_context(report)
    direction = report.selected_direction
    blocked, warning = derivative_risk(report, direction)
    heat_profile = _derivatives_heat_profile(report, direction)
    public_score = _public_derivatives_score(public, heat_profile)
    context_score = _context_direction_score(external_context, direction)
    has_public = public_score is not None
    has_context = int(external_context.get("evidence_count") or 0) > 0
    if not has_public and not has_context:
        proxy_score, proxy_reason = _proxy_derivatives_context(report)
        return {
            "score": proxy_score,
            "method": "liquidity_volatility_proxy",
            "available": False,
            "reason": proxy_reason,
        }
    if has_public and has_context:
        score = public_score * 0.58 + context_score * 0.42
        method = "external_derivatives_orderflow"
    elif has_context:
        score = context_score
        method = "external_derivatives_orderflow"
    else:
        score = public_score or 0.0
        method = "exchange_public_derivatives"
    bias = str(external_context.get("bias") or "neutral")
    confidence = _as_float(external_context.get("confidence")) or 0.0
    flags = set(external_context.get("risk_flags", [])) if isinstance(external_context.get("risk_flags"), list) else set()
    if direction in {"long", "short"} and bias in {"long", "short"} and bias != direction:
        score -= min(24.0, 8.0 + max(0.0, confidence - 58.0) * 0.45)
    if direction == "long" and "long_crowded" in flags:
        score -= 7.0
    if direction == "short" and "short_crowded" in flags:
        score -= 7.0
    if direction == "long" and "stop_hunt_risk_long" in flags:
        score -= 5.0
    if direction == "short" and "stop_hunt_risk_short" in flags:
        score -= 5.0
    if blocked:
        score = min(score, 38.0)
    time_note = _derivatives_time_note(public) if public else ""
    summary = external_context.get("summary") if has_context else ""
    reason_parts = []
    if has_public:
        reason_parts.append(
            f"public heat={heat_profile['heat_score']:.1f}, trend={heat_profile['trend_confirmation_score']:.1f}, "
            f"crowding={heat_profile['crowding_score']:.1f}, exhaustion={heat_profile['exhaustion_score']:.1f}, "
            f"score {public_score:.1f}{time_note}"
        )
    if has_context:
        reason_parts.append(
            f"external bias={bias}, confidence={confidence:.1f}, "
            f"long={external_context.get('long_score')}, short={external_context.get('short_score')}: {summary}"
        )
    if warning:
        reason_parts.append(warning)
    return {
        "score": round(_clamp(score), 2),
        "method": method,
        "available": True,
        "reason": " | ".join(reason_parts),
    }


def _public_derivatives_score(public: dict[str, Any], heat_profile: dict[str, Any] | None = None) -> float | None:
    funding = _as_float(public.get("funding_rate"))
    open_interest = _as_float(public.get("open_interest"))
    oi_change = _as_float(public.get("open_interest_change_pct"))
    if funding is None and open_interest is None and oi_change is None:
        return None
    if heat_profile is not None:
        score = (
            68.0
            + float(heat_profile.get("heat_score") or 0.0) * 0.16
            + float(heat_profile.get("trend_confirmation_score") or 0.0) * 0.28
            - float(heat_profile.get("crowding_score") or 0.0) * 0.30
            - float(heat_profile.get("exhaustion_score") or 0.0) * 0.34
        )
        if heat_profile.get("state") == "healthy_heat":
            score += 6.0
        if oi_change is None:
            score -= 6.0
        if open_interest is None:
            score -= 4.0
        if heat_profile.get("hard_block"):
            score = min(score, 38.0)
        return round(_clamp(score), 2)
    score = 92.0
    if funding is not None:
        score -= min(22.0, abs(funding) / 0.00035 * 14.0)
    if oi_change is not None:
        score -= min(28.0, abs(oi_change) / 18.0 * 24.0)
    else:
        score -= 8.0
    if open_interest is None:
        score -= 5.0
    return round(_clamp(score), 2)


def _context_direction_score(context: dict[str, Any], direction: str) -> float:
    if not isinstance(context, dict) or int(context.get("evidence_count") or 0) <= 0:
        return 70.0
    if direction == "short":
        score = _as_float(context.get("short_score"))
    elif direction == "long":
        score = _as_float(context.get("long_score"))
    else:
        score = max(_as_float(context.get("long_score")) or 50.0, _as_float(context.get("short_score")) or 50.0)
    return max(0.0, min(100.0, score if score is not None else 50.0))


def _flow_ratio_from_public(public: dict[str, Any]) -> float | None:
    flow = public.get("trade_flow") if isinstance(public.get("trade_flow"), dict) else {}
    ratio = _flow_ratio(flow)
    if ratio is not None:
        return ratio
    taker_ratio = public.get("taker_long_short_ratio")
    if isinstance(taker_ratio, dict):
        value = _as_float(taker_ratio.get("ratio"))
        if value is not None and value >= 0:
            return value / (1.0 + value)
    return None


def _flow_ratio(flow: dict[str, Any]) -> float | None:
    ratio = _as_float(flow.get("taker_buy_ratio"))
    if ratio is not None:
        return max(0.0, min(1.0, ratio))
    buy = _first_float(flow, ("taker_buy_notional", "buy", "buy_volume", "taker_buy_volume"))
    sell = _first_float(flow, ("taker_sell_notional", "sell", "sell_volume", "taker_sell_volume"))
    if buy is None or sell is None or buy + sell <= 0:
        return None
    return max(0.0, min(1.0, buy / (buy + sell)))


def _first_float(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _directional_ratio(ratio: float | None, direction: str) -> float | None:
    if ratio is None:
        return None
    ratio = max(0.0, min(1.0, ratio))
    if direction == "long":
        return ratio
    if direction == "short":
        return 1.0 - ratio
    return None


def _orderbook_alignment(public: dict[str, Any], direction: str) -> float | None:
    orderbook = public.get("orderbook") if isinstance(public.get("orderbook"), dict) else {}
    imbalance = _as_float(orderbook.get("depth_imbalance"))
    if imbalance is None:
        return None
    long_alignment = max(0.0, min(1.0, 0.5 + imbalance / 2.0))
    if direction == "long":
        return long_alignment
    if direction == "short":
        return 1.0 - long_alignment
    return None


def market_behavior_confirmation(report: SymbolReport, direction: str | None = None) -> dict[str, Any]:
    vol_profile = volatility_profile(report.symbol)
    kind = vol_profile.instrument_class
    requires_confirmation = kind in {"large_altcoin", "altcoin"}
    effective_direction = direction if direction in {"long", "short"} else report.selected_direction
    if not requires_confirmation:
        return {
            "required": False,
            "ok": True,
            "score": 100.0,
            "evidence_count": 0,
            "directional_support": 0.0,
            "reason": "core/proxy instruments use market context as risk filter only",
        }
    if effective_direction not in {"long", "short"}:
        return {
            "required": True,
            "ok": False,
            "score": 0.0,
            "evidence_count": 0,
            "directional_support": 0.0,
            "reason": "altcoin execution requires confirmed long/short direction",
        }

    heat = derivatives_heat_profile(report, effective_direction)
    external_context = _external_strategy_context(report)
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives", {}) if isinstance(values.get("exchange_public_derivatives"), dict) else {}
    side = selected_side(report)
    metrics = side.market_metrics if isinstance(getattr(side, "market_metrics", None), dict) else {}
    atr_pct = _as_float(metrics.get("atr_pct")) or 0.0
    volume_ratio = _as_float(metrics.get("volume_ratio")) or 0.0
    quote_volume = _as_float(getattr(report, "quote_volume_24h", None)) or 0.0
    flow_profile = participation_profile(report.symbol)
    high_turnover_floor = 150_000_000.0 if kind == "large_altcoin" else 250_000_000.0
    liquid_hot_alt = quote_volume >= high_turnover_floor and (
        atr_pct >= vol_profile.hot_atr_pct or volume_ratio >= flow_profile.hot_volume_ratio
    )
    low_liquidity_alt = kind == "altcoin" and quote_volume < 50_000_000.0
    evidence_count = 0
    directional_evidence_count = 0
    directional_support = 0.0
    reasons: list[str] = []
    blockers: list[str] = []

    if liquid_hot_alt:
        evidence_count += 1
        if atr_pct > 0.0:
            reasons.append(f"liquid high-vol regime ATR {atr_pct:.2f}%")
        else:
            reasons.append(f"liquid high-vol regime turnover {quote_volume / 1_000_000:.0f}M")

    flow_alignment = _as_float(heat.get("flow_alignment"))
    if flow_alignment is not None:
        evidence_count += 1
        if flow_alignment >= (0.57 if kind == "altcoin" else 0.55):
            directional_evidence_count += 1
            directional_support += 1.2
            reasons.append(f"taker flow aligns {flow_alignment:.2f}")
        elif flow_alignment <= 0.45:
            blockers.append(f"taker flow weak/rejecting {flow_alignment:.2f}")

    depth_alignment = _as_float(heat.get("depth_alignment"))
    if depth_alignment is not None:
        evidence_count += 1
        if depth_alignment >= (0.57 if kind == "altcoin" else 0.55):
            directional_evidence_count += 1
            directional_support += 0.7
            reasons.append(f"orderbook depth aligns {depth_alignment:.2f}")
        elif depth_alignment <= 0.42:
            blockers.append(f"orderbook depth against {depth_alignment:.2f}")

    external_evidence = int(external_context.get("evidence_count") or 0) if isinstance(external_context, dict) else 0
    if external_evidence > 0:
        evidence_count += min(2, max(1, external_evidence // 2))
        context_score = _as_float(heat.get("context_direction_score")) or _context_direction_score(external_context, effective_direction)
        bias = str(external_context.get("bias") or "neutral")
        confidence = _as_float(external_context.get("confidence")) or 0.0
        if context_score >= 64.0 and (bias in {"neutral", effective_direction} or confidence < 62.0):
            directional_evidence_count += 1
            directional_support += 0.8
            reasons.append(f"external context score {context_score:.1f}")
        if bias in {"long", "short"} and bias != effective_direction and confidence >= 66.0:
            blockers.append(f"external bias {bias} conflicts at confidence {confidence:.1f}")

    funding = _as_float(public.get("funding_rate"))
    oi_change = _as_float(public.get("open_interest_change_pct"))
    if funding is not None or oi_change is not None:
        evidence_count += 1
    trend = _as_float(heat.get("trend_confirmation_score")) or 0.0
    if trend >= (44.0 if kind == "altcoin" else 40.0):
        directional_support += 0.8
        reasons.append(f"derivatives trend heat {trend:.1f}")

    crowding = _as_float(heat.get("crowding_score")) or 0.0
    exhaustion = _as_float(heat.get("exhaustion_score")) or 0.0
    crowding_limit = 62.0 if kind == "altcoin" else 66.0
    exhaustion_limit = 62.0 if kind == "altcoin" else 66.0
    if liquid_hot_alt:
        crowding_limit -= 4.0
        exhaustion_limit -= 4.0
    if crowding >= crowding_limit:
        blockers.append(f"crowding {crowding:.1f} too high for alt execution")
    if exhaustion >= exhaustion_limit:
        blockers.append(f"exhaustion {exhaustion:.1f} too high for alt execution")
    if heat.get("hard_block"):
        blockers.append("derivatives hard block")
    funding_oi_confirms = (
        (funding is not None or oi_change is not None)
        and trend >= (44.0 if kind == "altcoin" else 40.0)
        and crowding < crowding_limit * 0.78
        and exhaustion < exhaustion_limit * 0.78
    )
    if funding_oi_confirms:
        directional_evidence_count += 1
        directional_support += 0.4
        reasons.append("funding/OI confirms healthy participation")

    required_evidence = 2 if kind == "large_altcoin" or liquid_hot_alt else 3
    required_directional_evidence = 3 if low_liquidity_alt and not liquid_hot_alt else 2
    support_floor = 1.25 if kind == "large_altcoin" else 1.45
    if liquid_hot_alt:
        support_floor = 1.80
    if not external_derivatives_available(report):
        blockers.append("readable derivatives/orderflow required; proxy context is watch-only for altcoins")
    if directional_evidence_count < required_directional_evidence:
        blockers.append(f"directional behavior evidence {directional_evidence_count} < {required_directional_evidence}")
    if evidence_count < required_evidence:
        blockers.append(f"market behavior evidence {evidence_count} < {required_evidence}")
    if directional_support < support_floor:
        blockers.append(f"directional orderflow support {directional_support:.1f} is not enough")

    score = (
        45.0
        + min(20.0, evidence_count * 5.0)
        + min(8.0, directional_evidence_count * 3.0)
        + min(25.0, directional_support * 10.0)
        + min(10.0, trend * 0.12)
        - min(20.0, crowding * 0.18)
        - min(20.0, exhaustion * 0.18)
    )
    ok = not blockers
    return {
        "required": True,
        "ok": ok,
        "score": round(_clamp(score), 2),
        "evidence_count": evidence_count,
        "directional_evidence_count": directional_evidence_count,
        "required_directional_evidence": required_directional_evidence,
        "directional_support": round(directional_support, 2),
        "reason": "; ".join(reasons or blockers[:3]) or "market behavior confirmed",
        "blockers": blockers,
    }


def quant_diagnostics(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    buckets = getattr(side, "bucket_scores", {}) or {}
    htf = buckets.get("htf_context", _feature_ratio(side, ["liquidity_sweep", "htf_poi"]))
    trigger = buckets.get("ltf_confirmation", _feature_ratio(side, ["mss_bos", "displacement"]))
    entry = buckets.get("entry_location", _feature_ratio(side, ["fvg", "ote"]))
    risk = buckets.get("risk_plan", _feature_ratio(side, ["risk_reward"]))
    market = buckets.get("market_filter", _feature_ratio(side, ["market_quality"]))
    optional = _feature_ratio(side, ["trendline", "amd", "nexus", "paid_data"])
    derivative_blocked, derivative_warning = derivative_risk(report, report.selected_direction)
    market_blocked, market_warning = market_risk(report)
    derivative_context = derivatives_context(report)
    heat_profile = derivatives_heat_profile(report, report.selected_direction)
    behavior = market_behavior_confirmation(report, report.selected_direction)
    api_readiness = configured_api_readiness(report)
    gate = execution_gate_profile(report)
    data_quality = core_data_quality(report, side)
    return {
        "instrument_class": gate["instrument_class"],
        "htf_context": round(float(htf), 1),
        "ltf_trigger": round(float(trigger), 1),
        "entry_quality": round(float(entry), 1),
        "risk_reward_quality": round(float(risk), 1),
        "market_api_quality": round(float(market), 1),
        "core_data_quality": data_quality,
        "min_core_data_quality": gate["min_core_data_quality"],
        "optional_confluence": round(float(optional), 1),
        "external_api_ok": external_derivatives_available(report),
        "derivative_blocked": derivative_blocked,
        "derivative_warning": derivative_warning,
        "market_overheated": market_blocked,
        "market_warning": market_warning,
        "derivatives_context_score": derivative_context["score"],
        "derivatives_context_method": derivative_context["method"],
        "derivatives_context_reason": derivative_context["reason"],
        "derivatives_context_ok": derivative_context["score"] >= gate["min_derivatives_context"],
        "derivatives_heat_state": heat_profile["state"],
        "derivatives_heat_score": heat_profile["heat_score"],
        "derivatives_trend_confirmation_score": heat_profile["trend_confirmation_score"],
        "derivatives_crowding_score": heat_profile["crowding_score"],
        "derivatives_exhaustion_score": heat_profile["exhaustion_score"],
        "derivatives_heat_profile": heat_profile,
        "market_behavior_confirmation": behavior,
        "market_behavior_ok": behavior["ok"],
        "market_behavior_score": behavior["score"],
        "market_behavior_reason": behavior["reason"],
        "configured_api_ready": api_readiness["execution_ready"],
        "configured_api_missing": api_readiness["execution_missing"],
        "configured_api_advisory_missing": api_readiness["advisory_missing"],
        "configured_api_failed": api_readiness["configured_failed"],
        "gate_profile": gate,
        "core_ict_ok": htf >= gate["min_htf_context"]
        and trigger >= gate["min_ltf_trigger"]
        and entry >= gate["min_entry_quality"]
        and risk >= gate["min_risk_quality"],
        "direction_conflict": report.metadata.get("direction_conflict", ""),
    }


def required_min_rr(report: SymbolReport, side: DirectionScore, diag: dict[str, Any] | None = None) -> float:
    diag = diag or quant_diagnostics(report)
    gate = execution_gate_profile(report)
    atr_pct = _as_float(side.market_metrics.get("atr_pct")) or 0.0
    setup_score = float(side.setup_score if side.setup_score is not None else side_score(side))
    exec_score = execution_score(side, side_score(side))
    heat_profile = diag.get("derivatives_heat_profile") or derivatives_heat_profile(report, report.selected_direction)
    if (
        diag["market_overheated"]
        or report.quote_volume_24h < 20_000_000
        or float(heat_profile.get("crowding_score") or 0.0) >= 70.0
        or float(heat_profile.get("exhaustion_score") or 0.0) >= 70.0
    ):
        return float(gate["strict_min_rr"])
    if (
        atr_pct >= 1.0
        and setup_score >= float(gate["limit_min_setup_score"])
        and exec_score >= float(gate["limit_min_execution_score"])
        and diag["ltf_trigger"] >= float(gate["limit_min_ltf_trigger"])
        and diag["entry_quality"] >= float(gate["limit_min_entry_quality"])
        and not diag["market_overheated"]
        and not heat_profile.get("hard_block")
        and float(heat_profile.get("crowding_score") or 0.0) < 60.0
        and report.quote_volume_24h >= 20_000_000
    ):
        return float(gate["scalp_min_rr"])
    return float(gate["min_rr"])


def action_blockers(report: SymbolReport, limit: int | None = 3) -> list[str]:
    side = selected_side(report)
    diag = quant_diagnostics(report)
    gate = execution_gate_profile(report)
    distance = entry_distance_pct(report.price, side.entry_zone)
    proximity = entry_proximity_state(report, side)
    dynamic_band = proximity["dynamic_band_pct"]
    distance_bands = entry_distance_bands(report.symbol, _as_float(side.market_metrics.get("atr_pct")), _spread_pct(report))
    rr = side.rr or 0.0
    required_rr = required_min_rr(report, side, diag)
    score_gap = float(report.metadata.get("score_gap", 0.0) or 0.0)
    score = side_score(side)
    setup_score = float(side.setup_score if side.setup_score is not None else score)
    exec_score = execution_score(side, score)
    blockers: list[str] = []
    if report.selected_direction == "neutral":
        blockers.append(report.metadata.get("direction_conflict") or "多空分差不足，禁止硬選方向")
    if score_gap < gate["min_score_gap"]:
        blockers.append(f"多空分差 {score_gap:.1f} < {gate['min_score_gap']:.0f}")
    if diag["htf_context"] < gate["min_htf_context"]:
        blockers.append(f"HTF bias {diag['htf_context']:.0f} 未達 {gate['min_htf_context']:.0f}")
    if diag["ltf_trigger"] < gate["min_ltf_trigger"]:
        blockers.append(f"LTF MSS/BOS + displacement {diag['ltf_trigger']:.0f} 未達 {gate['min_ltf_trigger']:.0f}")
    if diag["entry_quality"] < gate["min_entry_quality"]:
        blockers.append(f"FVG/OTE/OB 入場品質 {diag['entry_quality']:.0f} 未達 {gate['min_entry_quality']:.0f}")
    if diag["risk_reward_quality"] < gate["min_risk_quality"]:
        blockers.append(f"風控/RR 品質 {diag['risk_reward_quality']:.0f} 未達 {gate['min_risk_quality']:.0f}")
    quality_misses: list[tuple[float, str]] = []
    if score < gate["limit_min_selection_score"]:
        quality_misses.append(
            (
                float(gate["limit_min_selection_score"]) - score,
                f"selection_score {score:.1f} 未達限價門檻 {gate['limit_min_selection_score']:.0f}",
            )
        )
    if setup_score < gate["limit_min_setup_score"]:
        quality_misses.append(
            (
                float(gate["limit_min_setup_score"]) - setup_score,
                f"setup_score {setup_score:.1f} 未達限價門檻 {gate['limit_min_setup_score']:.0f}",
            )
        )
    if exec_score < gate["limit_min_execution_score"]:
        quality_misses.append(
            (
                float(gate["limit_min_execution_score"]) - exec_score,
                f"execution_score {exec_score:.1f} 未達限價門檻 {gate['limit_min_execution_score']:.0f}",
            )
        )
    if quality_misses:
        quality_misses.sort(key=lambda item: item[0], reverse=True)
        blockers.append(f"{quality_misses[0][1]}（分數類門檻只列最弱項，避免同源重複扣分）")
    if not side.entry_zone or side.stop is None or not side.take_profits:
        blockers.append("entry / stop / TP 計畫不完整")
    if rr < required_rr:
        blockers.append(f"RR {rr:.2f}R < 日內門檻 {required_rr:.2f}R")
    if distance is None:
        blockers.append("尚無有效 entry zone，不能執行")
    elif distance > distance_bands["missed"]:
        blockers.append(f"距 entry {distance:.2f}% > 5%，標記錯過/過期")
    elif distance > distance_bands["stale"]:
        blockers.append(f"距 entry {distance:.2f}% > 3%，禁止顯示可以做")
    elif distance > distance_bands["caution"]:
        blockers.append(f"距 entry {distance:.2f}% > 1.2%，不追價")
    elif distance > dynamic_band:
        blockers.append(f"距 entry {distance:.2f}% > 動態 band {dynamic_band:.2f}%，等待回撤")
    if diag["derivatives_context_score"] < gate["min_derivatives_context"]:
        blockers.append(
            f"derivatives context {diag['derivatives_context_score']:.0f} 未達 {gate['min_derivatives_context']:.0f}；"
            f"{diag['derivatives_context_reason']}"
        )
    if not diag.get("configured_api_ready", True):
        missing = ", ".join(diag.get("configured_api_missing", [])) or "configured API"
        blockers.append(f"已設定 API 尚未完整讀取：{missing}")
    if diag["derivative_blocked"]:
        blockers.append(f"funding/OI 過熱：{diag['derivative_warning']}")
    if diag["market_overheated"]:
        blockers.append(f"市場過熱，禁止追價：{diag['market_warning']}")
    if not diag.get("market_behavior_ok", True):
        behavior = diag.get("market_behavior_confirmation", {})
        behavior_blockers = behavior.get("blockers", []) if isinstance(behavior, dict) else []
        blockers.append(
            "市場行為確認不足："
            + ("；".join(str(item) for item in behavior_blockers[:2]) or str(diag.get("market_behavior_reason") or "orderflow not confirmed"))
        )
    if diag["core_data_quality"] < gate["min_core_data_quality"]:
        blockers.append(f"核心資料品質 {diag['core_data_quality']:.0f}% < {float(gate['min_core_data_quality']):.0f}%")
    if hard_direction_conflict(report) and report.selected_direction != "neutral":
        blockers.append(f"方向衝突：{report.metadata['direction_conflict']}")
    output: list[str] = []
    for item in blockers:
        if item and item not in output:
            output.append(item)
    return output if limit is None else output[:limit]


def action_blocker_summary(report: SymbolReport) -> str:
    blockers = action_blockers(report, limit=2)
    if blockers:
        return "；".join(blockers)
    return "等待下一根確認 K 線與 entry zone 回撤"


def invalidation_conditions(report: SymbolReport) -> list[str]:
    side = selected_side(report)
    direction = report.selected_direction
    conditions = [
        "HTF bias 消失或 BTC 市場環境明顯反向",
        "15m/5m 出現反向 MSS/BOS 且伴隨 displacement",
        "FVG/OTE/OB 入場區被完全破壞或價格遠離 entry zone 超過 5%",
        "funding/OI 轉為過熱，或衍生品替代量化分數跌破可控門檻",
    ]
    if side.stop is not None:
        verb = "跌破" if direction == "long" else "突破"
        conditions.insert(0, f"價格{verb} stop {side.stop:g}")
    required_rr = required_min_rr(report, side, quant_diagnostics(report))
    if side.rr is not None and side.rr < required_rr:
        conditions.append(f"RR 降到日內門檻 {required_rr:.2f}R 以下")
    return conditions


def evaluate_execution_gate(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    diag = quant_diagnostics(report)
    gate = execution_gate_profile(report)
    distance = entry_distance_pct(report.price, side.entry_zone)
    proximity = entry_proximity_state(report, side)
    dynamic_band = proximity["dynamic_band_pct"]
    distance_bands = entry_distance_bands(report.symbol, _as_float(side.market_metrics.get("atr_pct")), _spread_pct(report))
    rr = side.rr or 0.0
    score = side_score(side)
    exec_score = execution_score(side, score)
    setup_score = float(side.setup_score if side.setup_score is not None else score)
    required_rr = required_min_rr(report, side, diag)
    score_gap = float(report.metadata.get("score_gap", 0.0) or 0.0)
    blockers = action_blockers(report, limit=None)
    common_checks = {
        "selected_direction": report.selected_direction != "neutral",
        "score_gap": score_gap >= gate["min_score_gap"],
        "htf_context": diag["htf_context"] >= gate["min_htf_context"],
        "risk_quality": diag["risk_reward_quality"] >= gate["min_risk_quality"],
        "rr": rr >= required_rr,
        "entry_distance": distance is not None and distance <= dynamic_band,
        "external_derivatives": diag["external_api_ok"],
        "derivatives_context": diag["derivatives_context_score"] >= gate["min_derivatives_context"],
        "configured_api_ready": diag.get("configured_api_ready", True),
        "derivatives_not_overheated": not diag["derivative_blocked"],
        "market_not_overheated": not diag["market_overheated"],
        "data_completeness": diag["core_data_quality"] >= gate["min_core_data_quality"],
        "complete_trade_plan": bool(side.entry_zone and side.stop is not None and side.take_profits),
        "direction_not_conflicted": not hard_direction_conflict(report),
    }
    behavior = diag.get("market_behavior_confirmation", {})
    if not isinstance(behavior, dict):
        behavior = {}
    behavior_blockers = [str(item) for item in behavior.get("blockers", []) if item]
    severe_behavior_block = any(
        needle in blocker
        for blocker in behavior_blockers
        for needle in (
            "taker flow weak/rejecting",
            "external bias",
            "derivatives hard block",
            "crowding",
            "exhaustion",
            "readable derivatives/orderflow required",
        )
    )
    behavior_evidence_count = int(behavior.get("evidence_count") or 0)
    directional_evidence_count = int(behavior.get("directional_evidence_count") or 0)
    required_directional_evidence = int(
        behavior.get("required_directional_evidence")
        or (2 if gate.get("instrument_class") in {"large_altcoin", "altcoin"} else 0)
    )
    directional_support = _as_float(behavior.get("directional_support")) or 0.0
    limit_entry_distance_ok = distance is not None and distance <= dynamic_band * 2.5
    limit_common_checks = {**common_checks, "entry_distance": limit_entry_distance_ok}
    limit_quality_checks = {
        "selection_score": score >= gate["limit_min_selection_score"],
        "setup_score": setup_score >= gate["limit_min_setup_score"],
        "ltf_trigger": diag["ltf_trigger"] >= gate["limit_min_ltf_trigger"],
        "entry_quality": diag["entry_quality"] >= gate["limit_min_entry_quality"],
        "execution_score": exec_score >= gate["limit_min_execution_score"],
    }
    severe_limit_quality_miss = (
        score < float(gate["limit_min_selection_score"]) - 6.0
        or setup_score < float(gate["limit_min_setup_score"]) - 6.0
        or exec_score < float(gate["limit_min_execution_score"]) - 8.0
        or diag["ltf_trigger"] < float(gate["limit_min_ltf_trigger"]) - 8.0
        or diag["entry_quality"] < float(gate["limit_min_entry_quality"]) - 8.0
    )
    limit_quality_misses = [key for key, value in limit_quality_checks.items() if not value]
    limit_quality_ok = not severe_limit_quality_miss and len(limit_quality_misses) <= 1
    limit_behavior_quality = bool(diag.get("market_behavior_ok", True)) or (
        directional_evidence_count >= required_directional_evidence
        and directional_support >= 1.25
        and not severe_behavior_block
    )
    market_checks = {
        **common_checks,
        "market_behavior": bool(diag.get("market_behavior_ok", True)),
        "market_rr": rr >= max(required_rr, float(gate["min_rr"]), 1.65),
        "ltf_trigger": diag["ltf_trigger"] >= gate["min_ltf_trigger"],
        "entry_quality": diag["entry_quality"] >= gate["min_entry_quality"],
        "execution_score": exec_score >= float(gate["market_min_execution_score"]),
    }
    limit_checks = {
        **limit_common_checks,
        **limit_quality_checks,
        "limit_quality_ok": limit_quality_ok,
        "limit_behavior_quality": limit_behavior_quality,
    }
    market_gate_ready = all(value for key, value in market_checks.items() if key != "external_derivatives")
    tight_market_distance = distance is not None and distance <= min(0.05, dynamic_band * 0.25)
    market_tight_ready = market_gate_ready and tight_market_distance
    market_momentum = market_momentum_execution_profile(
        report, side, diag, gate, distance, dynamic_band, exec_score, rr, required_rr
    )
    market_common_without_entry = all(
        value for key, value in common_checks.items() if key not in {"external_derivatives", "entry_distance"}
    )
    market_momentum_ready = market_common_without_entry and bool(market_momentum["ok"])
    market_ready = market_tight_ready or market_momentum_ready
    soft_limit_keys = set(limit_quality_checks) | {"external_derivatives"}
    normal_limit_ready = all(value for key, value in limit_checks.items() if key not in soft_limit_keys)
    precision_limit = precision_limit_execution_profile(
        report,
        side,
        diag,
        gate,
        distance,
        dynamic_band,
        score,
        setup_score,
        exec_score,
        rr,
        required_rr,
        score_gap,
    )
    precision_limit_ready = bool(precision_limit["ok"])
    limit_ready = normal_limit_ready or precision_limit_ready
    market_as_limit_ready = market_gate_ready and not market_ready
    execution_ready = market_ready or limit_ready or market_as_limit_ready
    pre_market_base_rr_ok = rr >= float(gate["min_rr"])
    pre_market_market_ready = pre_market_base_rr_ok and all(
        value
        for key, value in market_checks.items()
        if key not in {"external_derivatives", "market_not_overheated", "rr", "market_rr"}
    )
    pre_market_limit_ready = pre_market_base_rr_ok and all(
        value
        for key, value in limit_checks.items()
        if key not in (soft_limit_keys | {"market_not_overheated", "rr"})
    )
    market_risk_is_primary = diag["market_overheated"] and (pre_market_market_ready or pre_market_limit_ready)
    checks = {
        **market_checks,
        "instrument_class": gate["instrument_class"],
        "core_data_quality": diag["core_data_quality"],
        "min_core_data_quality": gate["min_core_data_quality"],
        "market_ready": market_ready,
        "market_gate_ready": market_gate_ready,
        "market_tight_ready": market_tight_ready,
        "market_momentum_ready": market_momentum_ready,
        "market_momentum_failed": market_momentum["failed"],
        "market_distance": tight_market_distance or bool(market_momentum["checks"]["distance"]),
        "market_distance_limit": market_momentum["distance_limit"],
        "market_spread_pct": market_momentum["spread_pct"],
        "market_spread_limit": market_momentum["spread_limit"],
        "limit_ready": limit_ready or market_as_limit_ready,
        "normal_limit_ready": normal_limit_ready,
        "precision_limit_ready": precision_limit_ready,
        "precision_limit_failed": precision_limit["failed"],
        "precision_limit_behavior_score": precision_limit["behavior_score"],
        "precision_limit_directional_support": precision_limit["directional_support"],
        "precision_limit_flow_alignment": precision_limit["flow_alignment"],
        "precision_limit_depth_alignment": precision_limit["depth_alignment"],
        "precision_limit_selection_floor": precision_limit["selection_floor"],
        "precision_limit_setup_floor": precision_limit["setup_floor"],
        "precision_limit_execution_floor": precision_limit["execution_floor"],
        "precision_limit_ltf_floor": precision_limit["ltf_floor"],
        "precision_limit_htf_floor": precision_limit["htf_floor"],
        "precision_limit_context_floor": precision_limit["context_floor"],
        "precision_limit_strong_flow": precision_limit["strong_flow_limit"],
        "precision_limit_liquid_flow": precision_limit["liquid_flow_limit"],
        "precision_limit_elite_structure": precision_limit["elite_structure_limit"],
        "limit_entry_distance": limit_entry_distance_ok,
        "limit_quality_ok": limit_quality_ok,
        "limit_quality_misses": limit_quality_misses,
        "limit_behavior_quality": limit_behavior_quality,
        "limit_behavior_evidence_count": behavior_evidence_count,
        "limit_directional_evidence_count": directional_evidence_count,
        "limit_required_directional_evidence": required_directional_evidence,
        "limit_severe_behavior_block": severe_behavior_block,
        "required_rr": round(required_rr, 2),
        "limit_selection_score": limit_checks["selection_score"],
        "limit_setup_score": limit_checks["setup_score"],
        "limit_ltf_trigger": limit_checks["ltf_trigger"],
        "limit_entry_quality": limit_checks["entry_quality"],
        "limit_execution_score": limit_checks["execution_score"],
    }
    failed_gate_reasons = [
        key for key, value in checks.items() if isinstance(value, bool) and not value
    ]
    if execution_ready:
        failed_gate_reasons = []
    primary_failed_reason = failed_gate_reasons[0] if failed_gate_reasons else ""
    if execution_ready:
        code = "market" if market_ready else "limit"
        if market_tight_ready:
            trigger_text = "market-tight gate passed"
        elif market_momentum_ready:
            trigger_text = "market-momentum gate passed; verify live depth before sending"
        elif precision_limit_ready:
            trigger_text = "precision-limit gate passed; place only inside the entry zone and require final trigger before fill"
        else:
            trigger_text = "limit gate passed; final LTF trigger still must be watched before fill"
        return {
            "code": code,
            "label": "可以做",
            "reason": (
                f"execution_score={exec_score:.1f}, setup_score={setup_score:.1f}, {trigger_text}; "
                f"RR {rr:.2f}R >= {required_rr:.2f}R; distance {distance:.2f}% / dynamic band {dynamic_band:.2f}%"
            ),
            "entry_distance_pct": distance,
            "dynamic_entry_band_pct": dynamic_band,
            "entry_proximity_state": proximity["state"],
            "should_execute": True,
            "blockers": [],
            "warnings": visible_warnings(side),
            "invalidation_conditions": invalidation_conditions(report),
            "gate_checks": checks,
            "paid_data_status": paid_data_status(report),
            "required_next_trigger": _required_next_trigger(code, proximity),
            "execution_quality": round(exec_score, 2),
            "failed_gate_reasons": failed_gate_reasons,
            "primary_failed_reason": primary_failed_reason,
        }
    if diag["derivative_blocked"]:
        code, label = "avoid", "禁止"
        reason = f"funding/OI 過熱：{diag['derivative_warning']}"
    elif market_risk_is_primary:
        code, label = "watch", "錯過 / 不追"
        reason = f"市場過熱，禁止追價：{diag['market_warning']}"
    elif distance is not None and distance > distance_bands["stale"]:
        code, label = "watch", "錯過 / 不追"
        reason = action_blocker_summary(report)
    elif report.selected_direction == "neutral":
        code, label = ("watch", "觀察") if score >= 58 else ("avoid", "不能做")
        reason = f"方向未確認：{action_blocker_summary(report)}"
    elif not diag.get("configured_api_ready", True):
        code, label = "watch", "觀察"
        missing = ", ".join(diag.get("configured_api_missing", [])) or "configured API"
        reason = f"已設定 API 尚未完整讀取，不能標記可執行：{missing}"
    elif diag["core_data_quality"] < gate["min_core_data_quality"]:
        code, label = "watch", "觀察"
        reason = f"資料不足，只能觀察：{action_blocker_summary(report)}"
    elif diag["derivatives_context_score"] < gate["min_derivatives_context"]:
        code, label = "watch", "觀察"
        reason = (
            f"衍生品替代量化 {diag['derivatives_context_score']:.1f} "
            f"未達 {gate['min_derivatives_context']:.0f}：{action_blocker_summary(report)}"
        )
    elif distance is not None and distance <= dynamic_band and score >= 62:
        code, label = "watch", "待確認"
        reason = f"到價但確認不足：{action_blocker_summary(report)}"
    elif score >= 62:
        code, label = "watch", "觀察"
        reason = f"值得盯但不能做：{action_blocker_summary(report)}"
    else:
        code, label = "avoid", "不能做"
        reason = f"selection_score={score:.1f} / execution_score={exec_score:.1f} 不足"
    return {
        "code": code,
        "label": label,
        "reason": reason,
        "entry_distance_pct": distance,
        "dynamic_entry_band_pct": dynamic_band,
        "entry_proximity_state": proximity["state"],
        "should_execute": False,
        "blockers": blockers,
        "warnings": visible_warnings(side),
        "invalidation_conditions": invalidation_conditions(report),
        "gate_checks": checks,
        "paid_data_status": paid_data_status(report),
        "required_next_trigger": _required_next_trigger(code, proximity),
        "execution_quality": round(exec_score, 2),
        "failed_gate_reasons": failed_gate_reasons,
        "primary_failed_reason": primary_failed_reason,
    }


def _dedupe_text(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def _entry_origin(side: DirectionScore) -> str:
    origin = str(getattr(side, "entry_origin", "") or "").strip()
    if origin:
        return origin
    if side.entry_zone:
        return "unknown"
    return "fallback"


def _entry_validity(side: DirectionScore) -> str:
    validity = str(getattr(side, "entry_validity", "") or "").strip()
    if validity:
        return validity
    origin = _entry_origin(side)
    if origin in LIMIT_ENTRY_ORIGINS:
        return "valid"
    if origin == "market_price":
        return "market_only"
    return "fallback_only"


def _trade_plan_audit(report: SymbolReport, side: DirectionScore, rr: float, required_rr: float) -> dict[str, Any]:
    origin = _entry_origin(side)
    validity = _entry_validity(side)
    missing: list[str] = []
    if not report.symbol:
        missing.append("symbol")
    if not report.exchange:
        missing.append("exchange")
    if not report.data_time:
        missing.append("data_timestamp")
    if not side.entry_zone:
        missing.append("entry_zone")
    if side.stop is None:
        missing.append("stop")
    if not side.take_profits and side.target is None:
        missing.append("target")
    if side.rr is None:
        missing.append("rr")
    rr_ok = rr >= required_rr
    limit_origin_ok = origin in LIMIT_ENTRY_ORIGINS and validity == "valid"
    structurally_complete = not missing
    return {
        "complete": structurally_complete and rr_ok,
        "structurally_complete": structurally_complete,
        "missing": missing,
        "rr_ok": rr_ok,
        "origin": origin,
        "validity": validity,
        "limit_origin_ok": limit_origin_ok,
        "market_origin_ok": origin in LIMIT_ENTRY_ORIGINS or origin == "market_price",
    }


def _alt_confirmed_limit_profile(
    report: SymbolReport,
    diag: dict[str, Any],
    gate: dict[str, Any],
    behavior: dict[str, Any],
    heat_profile: dict[str, Any],
    distance: float | None,
    dynamic_band: float,
    score: float,
    setup_score: float,
    exec_score: float,
    rr: float,
    required_rr: float,
    severe_behavior_block: bool,
) -> dict[str, Any]:
    """Limit-only route for liquid alts with strong microstructure evidence."""

    kind = str(gate.get("instrument_class") or "")
    is_alt_family = kind in {"large_altcoin", "altcoin"}
    quote_volume = _as_float(getattr(report, "quote_volume_24h", None)) or 0.0
    flow_alignment = _as_float(heat_profile.get("flow_alignment"))
    depth_alignment = _as_float(heat_profile.get("depth_alignment"))
    trend_confirmation = _as_float(heat_profile.get("trend_confirmation_score")) or 0.0
    crowding = _as_float(heat_profile.get("crowding_score")) or 0.0
    exhaustion = _as_float(heat_profile.get("exhaustion_score")) or 0.0
    behavior_score = _as_float(diag.get("market_behavior_score")) or _as_float(behavior.get("score")) or 0.0
    evidence_count = int(behavior.get("evidence_count") or 0)
    directional_evidence = int(behavior.get("directional_evidence_count") or 0)
    spread = _spread_pct(report)

    liquidity_floor = 70_000_000.0 if kind == "altcoin" else 60_000_000.0
    flow_floor = 0.74 if kind == "altcoin" else 0.72
    very_strong_flow_floor = 0.88 if kind == "altcoin" else 0.84
    depth_floor = 0.56 if kind == "altcoin" else 0.54
    depth_neutral_floor = 0.47 if kind == "altcoin" else 0.45

    flow_strong = flow_alignment is not None and flow_alignment >= flow_floor
    flow_very_strong = flow_alignment is not None and flow_alignment >= very_strong_flow_floor
    depth_confirms = depth_alignment is not None and depth_alignment >= depth_floor
    depth_not_hostile = depth_alignment is None or depth_alignment >= depth_neutral_floor
    behavior_proxy_ok = bool(diag.get("market_behavior_ok", True)) or (
        depth_not_hostile
        and (
            (flow_strong and evidence_count >= 2 and behavior_score >= 68.0)
            or (flow_very_strong and evidence_count >= 1)
            or (flow_strong and depth_confirms)
        )
    )

    checks = {
        "alt_family": is_alt_family,
        "selected_direction": report.selected_direction in {"long", "short"},
        "external_derivatives": bool(diag.get("external_api_ok")),
        "configured_api_ready": bool(diag.get("configured_api_ready", True)),
        "derivatives_context": diag["derivatives_context_score"] >= float(gate["min_derivatives_context"]),
        "derivatives_not_overheated": not diag["derivative_blocked"],
        "market_not_overheated": not diag["market_overheated"],
        "no_heat_hard_block": not bool(heat_profile.get("hard_block")),
        "no_severe_behavior_block": not severe_behavior_block,
        "liquidity": quote_volume >= liquidity_floor,
        "spread": spread <= (0.08 if kind == "altcoin" else 0.07),
        "entry_distance": distance is not None and distance <= dynamic_band * 2.2,
        "score": score >= float(gate["limit_min_selection_score"]) - 5.0,
        "setup_score": setup_score >= float(gate["limit_min_setup_score"]),
        "execution_score": exec_score >= float(gate["limit_min_execution_score"]) + 2.0,
        "htf_context": diag["htf_context"] >= float(gate["min_htf_context"]),
        "ltf_trigger": diag["ltf_trigger"] >= float(gate["limit_min_ltf_trigger"]),
        "entry_quality": diag["entry_quality"] >= float(gate["limit_min_entry_quality"]) + 4.0,
        "risk_quality": diag["risk_reward_quality"] >= max(float(gate["min_risk_quality"]), 70.0),
        "rr_edge": rr >= required_rr + 0.15,
        "flow_support": flow_strong or flow_very_strong,
        "depth_not_hostile": depth_not_hostile,
        "behavior_proxy": behavior_proxy_ok,
        "heat_not_crowded": crowding < 58.0 and exhaustion < 52.0,
    }
    ok = all(checks.values())
    failed = [key for key, value in checks.items() if not value]
    reason_parts = []
    if flow_alignment is not None:
        reason_parts.append(f"flow={flow_alignment:.2f}")
    if depth_alignment is not None:
        reason_parts.append(f"depth={depth_alignment:.2f}")
    reason_parts.append(f"behavior_score={behavior_score:.1f}")
    reason_parts.append(f"liquidity={quote_volume / 1_000_000:.0f}M")
    return {
        "ok": ok,
        "checks": checks,
        "failed": failed,
        "reason": ", ".join(reason_parts),
        "flow_alignment": round(flow_alignment, 4) if flow_alignment is not None else None,
        "depth_alignment": round(depth_alignment, 4) if depth_alignment is not None else None,
        "trend_confirmation": round(trend_confirmation, 2),
        "behavior_score": round(behavior_score, 2),
        "evidence_count": evidence_count,
        "directional_evidence_count": directional_evidence,
    }


def _has_execution_orderflow_rejection(behavior_blockers: list[str]) -> bool:
    return any(
        needle in blocker.lower()
        for blocker in behavior_blockers
        for needle in (
            "taker flow weak/rejecting",
            "external bias",
            "derivatives hard block",
            "crowding",
            "exhaustion",
            "readable derivatives/orderflow required",
        )
    )


def _hard_orderflow_rejection(
    behavior_blockers: list[str],
    heat_profile: dict[str, Any],
    instrument: str,
) -> bool:
    """Separate a true veto from noisy altcoin microstructure disagreement."""

    lowered = [blocker.lower() for blocker in behavior_blockers]
    if any(
        needle in blocker
        for blocker in lowered
        for needle in ("external bias", "derivatives hard block", "crowding", "exhaustion")
    ):
        return True

    flow_rejections = [
        blocker
        for blocker in lowered
        if "taker flow weak/rejecting" in blocker
    ]
    if not flow_rejections:
        return False

    # Public taker flow on alts is very jumpy. Treat it as a hard veto only
    # when the opposite flow is extreme and another risk source confirms it.
    flow_alignment = _as_float(heat_profile.get("flow_alignment"))
    depth_alignment = _as_float(heat_profile.get("depth_alignment"))
    crowding = _as_float(heat_profile.get("crowding_score")) or 0.0
    exhaustion = _as_float(heat_profile.get("exhaustion_score")) or 0.0
    extreme_floor = 0.20 if instrument == "altcoin" else 0.22
    depth_floor = 0.42 if instrument == "altcoin" else 0.40
    extreme_opposite_flow = flow_alignment is not None and flow_alignment <= extreme_floor
    depth_confirms_against = depth_alignment is not None and depth_alignment <= depth_floor
    heat_confirms_risk = crowding >= 55.0 or exhaustion >= 55.0
    return bool(extreme_opposite_flow and (depth_confirms_against or heat_confirms_risk))


def _alt_active_momentum_limit_profile(
    report: SymbolReport,
    diag: dict[str, Any],
    gate: dict[str, Any],
    behavior: dict[str, Any],
    heat_profile: dict[str, Any],
    distance: float | None,
    dynamic_band: float,
    score: float,
    setup_score: float,
    exec_score: float,
    rr: float,
    required_rr: float,
    execution_behavior_block: bool,
) -> dict[str, Any]:
    """Limit route for high-beta alts with strong live participation."""

    kind = str(gate.get("instrument_class") or "")
    is_alt_family = kind in {"large_altcoin", "altcoin"}
    quote_volume = _as_float(getattr(report, "quote_volume_24h", None)) or 0.0
    flow_alignment = _as_float(heat_profile.get("flow_alignment"))
    depth_alignment = _as_float(heat_profile.get("depth_alignment"))
    trend_confirmation = _as_float(heat_profile.get("trend_confirmation_score")) or 0.0
    crowding = _as_float(heat_profile.get("crowding_score")) or 0.0
    exhaustion = _as_float(heat_profile.get("exhaustion_score")) or 0.0
    behavior_score = _as_float(diag.get("market_behavior_score")) or _as_float(behavior.get("score")) or 0.0
    context_direction_score = _as_float(heat_profile.get("context_direction_score")) or 0.0
    external_context = _external_strategy_context(report)
    has_external_context = int(external_context.get("evidence_count") or 0) > 0
    spread = _spread_pct(report)

    liquidity_floor = 70_000_000.0 if kind == "altcoin" else 50_000_000.0
    flow_floor = 0.72 if kind == "altcoin" else 0.70
    very_strong_flow_floor = 0.84 if kind == "altcoin" else 0.80
    depth_neutral_floor = 0.45 if kind == "altcoin" else 0.44
    flow_strong = flow_alignment is not None and flow_alignment >= flow_floor
    flow_very_strong = flow_alignment is not None and flow_alignment >= very_strong_flow_floor
    depth_not_hostile = depth_alignment is None or depth_alignment >= depth_neutral_floor
    flow_not_hostile = flow_alignment is None or flow_alignment >= (0.45 if kind == "altcoin" else 0.44)
    behavior_backed_flow_floor = 0.68 if kind == "altcoin" else 0.62
    behavior_backed_score_floor = 80.0
    behavior_backed_flow_support = (
        flow_alignment is not None
        and flow_alignment >= behavior_backed_flow_floor
        and behavior_score >= behavior_backed_score_floor
        and depth_not_hostile
        and not execution_behavior_block
        and crowding < 45.0
        and exhaustion < 45.0
        and quote_volume >= 90_000_000.0
    )
    soft_rejection_floor = 0.38 if kind == "altcoin" else 0.32
    soft_rejection_tolerated = (
        flow_alignment is not None
        and flow_alignment >= soft_rejection_floor
        and depth_not_hostile
        and crowding < 20.0
        and exhaustion < 25.0
    )
    context_structure_support = (
        has_external_context
        and diag["derivatives_context_score"] >= float(gate["min_derivatives_context"]) + (4.0 if kind == "altcoin" else 3.0)
        and context_direction_score >= 65.0
        and flow_not_hostile
        and depth_not_hostile
        and behavior_score >= 58.0
        and not execution_behavior_block
    )
    structure_resilient_support = (
        diag["derivatives_context_score"] >= float(gate["min_derivatives_context"]) - 1.5
        and soft_rejection_tolerated
        and quote_volume >= (100_000_000.0 if kind == "large_altcoin" else 140_000_000.0)
        and diag["htf_context"] >= float(gate["min_htf_context"]) + 8.0
        and diag["ltf_trigger"] >= float(gate["limit_min_ltf_trigger"]) + 10.0
        and diag["entry_quality"] >= float(gate["limit_min_entry_quality"]) + 8.0
        and diag["risk_reward_quality"] >= 80.0
    )
    low_risk_quality_allowed = (flow_very_strong or context_structure_support) and rr >= required_rr + 0.10
    score_floor = (
        60.0
        if flow_very_strong or context_structure_support
        else float(gate["limit_min_selection_score"]) - (4.0 if structure_resilient_support else 8.0)
    )
    setup_floor = (
        60.0
        if flow_very_strong or context_structure_support
        else float(gate["limit_min_setup_score"]) - (0.0 if structure_resilient_support else 4.0)
    )
    execution_floor = (
        50.0
        if flow_very_strong or context_structure_support
        else float(gate["limit_min_execution_score"]) - (0.0 if structure_resilient_support else 3.0)
    )
    risk_floor = 58.0 if low_risk_quality_allowed else max(float(gate["min_risk_quality"]), 68.0)
    behavior_floor = (
        76.0
        if flow_very_strong
        else 58.0
        if context_structure_support
        else behavior_backed_score_floor
        if behavior_backed_flow_support
        else 56.0
        if structure_resilient_support
        else 74.0
    )
    if flow_strong:
        flow_support_mode = "strong"
    elif behavior_backed_flow_support:
        flow_support_mode = "behavior_backed"
    elif context_structure_support:
        flow_support_mode = "context_structure"
    elif structure_resilient_support:
        flow_support_mode = "structure_resilient"
    else:
        flow_support_mode = "none"
    checks = {
        "alt_family": is_alt_family,
        "selected_direction": report.selected_direction in {"long", "short"},
        "external_derivatives": bool(diag.get("external_api_ok")),
        "configured_api_ready": bool(diag.get("configured_api_ready", True)),
        "derivatives_context": diag["derivatives_context_score"] >= float(gate["min_derivatives_context"])
        or structure_resilient_support,
        "derivatives_not_overheated": not diag["derivative_blocked"],
        "market_not_overheated": not diag["market_overheated"],
        "no_heat_hard_block": not bool(heat_profile.get("hard_block")),
        "no_execution_behavior_block": not execution_behavior_block or structure_resilient_support,
        "liquidity": quote_volume >= liquidity_floor,
        "spread": spread <= (0.08 if kind == "altcoin" else 0.07),
        "entry_distance": distance is not None and distance <= dynamic_band * 2.2,
        "score": score >= score_floor,
        "setup_score": setup_score >= setup_floor,
        "execution_score": exec_score >= execution_floor,
        "htf_context": diag["htf_context"] >= float(gate["min_htf_context"]),
        "ltf_trigger": diag["ltf_trigger"] >= float(gate["limit_min_ltf_trigger"]),
        "entry_quality": diag["entry_quality"] >= float(gate["limit_min_entry_quality"]) + 4.0,
        "risk_quality": diag["risk_reward_quality"] >= risk_floor,
        "rr": rr >= required_rr,
        "flow_support": flow_strong
        or behavior_backed_flow_support
        or context_structure_support
        or structure_resilient_support,
        "depth_not_hostile": depth_not_hostile,
        "behavior_proxy": behavior_score >= behavior_floor,
        "heat_not_crowded": crowding < 58.0 and exhaustion < 55.0,
    }
    ok = all(checks.values())
    failed = [key for key, value in checks.items() if not value]
    reason_parts = []
    if flow_alignment is not None:
        reason_parts.append(f"flow={flow_alignment:.2f}")
    if depth_alignment is not None:
        reason_parts.append(f"depth={depth_alignment:.2f}")
    reason_parts.append(f"context={context_direction_score:.1f}")
    reason_parts.append(f"behavior_score={behavior_score:.1f}")
    reason_parts.append(f"trend_heat={trend_confirmation:.1f}")
    reason_parts.append(f"liquidity={quote_volume / 1_000_000:.0f}M")
    return {
        "ok": ok,
        "checks": checks,
        "failed": failed,
        "reason": ", ".join(reason_parts),
        "flow_alignment": round(flow_alignment, 4) if flow_alignment is not None else None,
        "depth_alignment": round(depth_alignment, 4) if depth_alignment is not None else None,
        "behavior_score": round(behavior_score, 2),
        "flow_support_mode": flow_support_mode,
        "score_floor": round(score_floor, 2),
        "setup_floor": round(setup_floor, 2),
        "execution_floor": round(execution_floor, 2),
        "risk_floor": round(risk_floor, 2),
        "behavior_floor": round(behavior_floor, 2),
    }


def _blocker_categories(items: list[str]) -> dict[str, int]:
    categories = {
        "direction": ("direction", "score gap", "two-sided", "side"),
        "plan": ("entry", "stop", "target", "rr", "plan", "fallback"),
        "risk": ("funding", "oi", "crowding", "exhaustion", "risk", "overheat"),
        "data": ("data", "api", "provider", "stale", "kline"),
        "liquidity": ("spread", "depth", "liquidity", "volume"),
        "distance": ("distance", "missed", "far"),
        "quality": ("score", "htf", "ltf", "setup", "execution"),
    }
    output: dict[str, int] = {}
    for item in items:
        text = item.lower()
        matched = "other"
        for category, needles in categories.items():
            if any(needle in text for needle in needles):
                matched = category
                break
        output[matched] = output.get(matched, 0) + 1
    return output


def action_blockers(report: SymbolReport, limit: int | None = 3) -> list[str]:
    gate = evaluate_execution_gate(report)
    blockers = list(gate.get("hard_blockers") or gate.get("blockers") or [])
    if not blockers:
        blockers = list(gate.get("soft_warnings") or [])
    blockers = _dedupe_text([str(item) for item in blockers])
    return blockers if limit is None else blockers[:limit]


def action_blocker_summary(report: SymbolReport) -> str:
    blockers = action_blockers(report, limit=2)
    if blockers:
        return "；".join(blockers)
    return "等待價格回到 entry zone 並重新確認 LTF trigger"


def evaluate_execution_gate(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    diag = quant_diagnostics(report)
    gate = execution_gate_profile(report)
    proximity = entry_proximity_state(report, side)
    dynamic_band = float(proximity["dynamic_band_pct"])
    distance = entry_distance_pct(report.price, side.entry_zone)
    distance_bands = entry_distance_bands(report.symbol, _as_float(side.market_metrics.get("atr_pct")), _spread_pct(report))
    rr = float(side.rr or 0.0)
    score = side_score(side)
    exec_score = execution_score(side, score)
    setup_score = float(side.setup_score if side.setup_score is not None else score)
    required_rr = required_min_rr(report, side, diag)
    score_gap = float(report.metadata.get("score_gap", 0.0) or 0.0)
    plan = _trade_plan_audit(report, side, rr, required_rr)
    behavior = diag.get("market_behavior_confirmation", {})
    if not isinstance(behavior, dict):
        behavior = {}
    behavior_blockers = [str(item) for item in behavior.get("blockers", []) if item]
    spread = _spread_pct(report)
    instrument = str(gate.get("instrument_class") or "")
    is_alt_family = instrument in {"large_altcoin", "altcoin"}
    alt_external_data_ready = (not is_alt_family) or external_derivatives_available(report)
    heat_profile = diag.get("derivatives_heat_profile")
    if not isinstance(heat_profile, dict):
        heat_profile = _derivatives_heat_profile(report, report.selected_direction)
    execution_behavior_block = _has_execution_orderflow_rejection(behavior_blockers)
    hard_behavior_block = _hard_orderflow_rejection(behavior_blockers, heat_profile, instrument)
    flow_seen = _as_float(heat_profile.get("flow_alignment")) is not None
    depth_seen = _as_float(heat_profile.get("depth_alignment")) is not None
    partial_flow_seen = flow_seen or depth_seen
    quote_volume = _as_float(getattr(report, "quote_volume_24h", None)) or 0.0
    entry_anchor_score = _as_float(side.market_metrics.get("entry_anchor_score")) if isinstance(side.market_metrics, dict) else None
    entry_anchor_ok = entry_anchor_score is None or bool(side.market_metrics.get("entry_anchor_ok"))
    mover_profile = str(side.market_metrics.get("mover_profile") or "normal") if isinstance(side.market_metrics, dict) else "normal"
    mover_chase_risk = bool(side.market_metrics.get("mover_chase_risk")) if isinstance(side.market_metrics, dict) else False
    hard_blockers: list[str] = []
    soft_warnings: list[str] = []

    if report.selected_direction == "neutral":
        hard_blockers.append(str(report.metadata.get("direction_conflict") or "direction is neutral / two-sided watch"))
    elif score_gap < float(gate["min_score_gap"]):
        hard_blockers.append(f"score gap {score_gap:.1f} < {float(gate['min_score_gap']):.1f}; two-sided watch")
    if hard_direction_conflict(report):
        hard_blockers.append(str(report.metadata.get("direction_conflict") or "hard direction conflict"))
    if plan["missing"]:
        hard_blockers.append("incomplete trade plan: " + ", ".join(plan["missing"]))
    if plan["origin"] == "fallback":
        hard_blockers.append("entry_origin=fallback; current price fallback cannot be executable")
    if not plan["rr_ok"]:
        hard_blockers.append(f"RR {rr:.2f}R < required {required_rr:.2f}R")
    if diag["core_data_quality"] < float(gate["min_core_data_quality"]):
        hard_blockers.append(f"core data quality {diag['core_data_quality']:.0f}% < {float(gate['min_core_data_quality']):.0f}%")
    if not diag.get("configured_api_ready", True):
        missing = ", ".join(diag.get("configured_api_missing", [])) or "mandatory configured provider"
        hard_blockers.append(f"mandatory API missing: {missing}")
    if is_alt_family and not alt_external_data_ready:
        hard_blockers.append("altcoin execution requires readable exchange/public derivatives or orderflow data")
    if diag["derivative_blocked"]:
        hard_blockers.append(f"derivatives hard risk: {diag['derivative_warning']}")
    if diag["market_overheated"]:
        hard_blockers.append(f"market hard risk: {diag['market_warning']}")
    if hard_behavior_block:
        hard_blockers.append("severe orderflow/derivatives rejection: " + "; ".join(behavior_blockers[:2]))
    if mover_chase_risk:
        hard_blockers.append(f"{mover_profile} same-side chase risk: wait for breakout retest or structural pullback")
    if report.quote_volume_24h < 20_000_000:
        hard_blockers.append(f"24h quote volume {report.quote_volume_24h:.0f} below minimum tradable floor")
    if spread >= (0.18 if instrument == "altcoin" else 0.14 if instrument == "large_altcoin" else 0.10):
        hard_blockers.append(f"spread {spread:.3f}% too wide for execution")

    if not diag.get("external_api_ok", True):
        soft_warnings.append("derivatives/orderflow unavailable; do not upgrade to market")
    if diag["derivatives_context_score"] < float(gate["min_derivatives_context"]):
        soft_warnings.append(
            f"derivatives context {diag['derivatives_context_score']:.1f} < {float(gate['min_derivatives_context']):.0f}; priority reduced"
        )
    if not diag.get("market_behavior_ok", True):
        soft_warnings.append("market behavior not fully confirmed: " + (behavior.get("reason") or "partial orderflow"))
    for missing in diag.get("configured_api_advisory_missing", []) or []:
        soft_warnings.append(f"advisory provider unavailable: {missing}")

    entry_zone_width_pct = 0.0
    if side.entry_zone:
        zone_low, zone_high = min(side.entry_zone), max(side.entry_zone)
        entry_zone_width_pct = (zone_high - zone_low) / max(abs(report.price), 1e-12) * 100.0
    metric_width = _as_float(side.market_metrics.get("entry_zone_width_pct")) if isinstance(side.market_metrics, dict) else None
    if metric_width is not None:
        entry_zone_width_pct = max(entry_zone_width_pct, metric_width)
    limit_distance_budget = dynamic_band * 2.5
    if instrument == "altcoin":
        limit_distance_budget = max(limit_distance_budget, entry_zone_width_pct * 1.35, 0.55)
        limit_distance_budget = min(limit_distance_budget, distance_bands["caution"])
    elif instrument == "large_altcoin":
        limit_distance_budget = max(limit_distance_budget, entry_zone_width_pct * 1.20, 0.42)
        limit_distance_budget = min(limit_distance_budget, distance_bands["caution"])
    limit_distance_ok = distance is not None and distance <= limit_distance_budget
    armed_distance_ok = distance is not None and distance <= distance_bands["stale"]
    missed = distance is not None and distance > distance_bands["stale"]
    behavior_ok = bool(diag.get("market_behavior_ok", True))
    btc_context_soft_conflict = "btc context" in str(diag.get("market_warning") or "").lower()
    strong_limit_override = (
        is_alt_family
        and not behavior_ok
        and alt_external_data_ready
        and partial_flow_seen
        and not execution_behavior_block
        and not btc_context_soft_conflict
        and quote_volume >= (80_000_000.0 if instrument == "altcoin" else 120_000_000.0)
        and distance is not None
        and distance <= dynamic_band * 1.25
        and score >= float(gate["limit_min_selection_score"]) - 3.0
        and setup_score >= float(gate["limit_min_setup_score"]) + 4.0
        and exec_score >= float(gate["limit_min_execution_score"]) + 10.0
        and diag["ltf_trigger"] >= float(gate["limit_min_ltf_trigger"]) + 4.0
    )
    liquid_flow_override = (
        is_alt_family
        and behavior_ok
        and quote_volume >= 80_000_000.0
        and distance is not None
        and distance <= dynamic_band
        and rr >= required_rr + 0.35
        and score >= float(gate["limit_min_selection_score"]) - 6.0
        and setup_score >= float(gate["limit_min_setup_score"])
        and exec_score >= float(gate["limit_min_execution_score"]) + 2.0
        and diag["ltf_trigger"] >= float(gate["limit_min_ltf_trigger"]) + 6.0
    )
    alt_confirmed_limit = _alt_confirmed_limit_profile(
        report,
        diag,
        gate,
        behavior,
        heat_profile,
        distance,
        dynamic_band,
        score,
        setup_score,
        exec_score,
        rr,
        required_rr,
        execution_behavior_block,
    )
    alt_confirmed_limit_ok = bool(alt_confirmed_limit["ok"])
    alt_active_momentum_limit = _alt_active_momentum_limit_profile(
        report,
        diag,
        gate,
        behavior,
        heat_profile,
        distance,
        dynamic_band,
        score,
        setup_score,
        exec_score,
        rr,
        required_rr,
        execution_behavior_block,
    )
    alt_active_momentum_limit_ok = bool(alt_active_momentum_limit["ok"])
    limit_behavior_ok = (
        (not is_alt_family)
        or behavior_ok
        or strong_limit_override
        or liquid_flow_override
        or alt_confirmed_limit_ok
        or alt_active_momentum_limit_ok
    )
    limit_quality = {
        "selection_score": score >= float(gate["limit_min_selection_score"]),
        "setup_score": setup_score >= float(gate["limit_min_setup_score"]),
        "execution_score": exec_score >= float(gate["limit_min_execution_score"]),
        "htf_context": diag["htf_context"] >= float(gate["min_htf_context"]),
        "ltf_trigger": diag["ltf_trigger"] >= float(gate["limit_min_ltf_trigger"]),
        "entry_quality": diag["entry_quality"] >= float(gate["limit_min_entry_quality"]),
        "entry_anchor": entry_anchor_ok,
        "mover_chase_control": not mover_chase_risk,
        "risk_quality": diag["risk_reward_quality"] >= float(gate["min_risk_quality"]),
        "market_behavior_for_limit": limit_behavior_ok,
    }
    normal_limit_quality = dict(limit_quality)
    normal_limit_quality["selection_score"] = score >= float(gate["limit_min_selection_score"])
    if liquid_flow_override or strong_limit_override or alt_confirmed_limit_ok:
        limit_quality["selection_score"] = True
    if alt_active_momentum_limit_ok:
        limit_quality["selection_score"] = True
        limit_quality["setup_score"] = True
        limit_quality["execution_score"] = True
        limit_quality["risk_quality"] = True
    limit_ready = (
        not hard_blockers
        and plan["complete"]
        and bool(plan["limit_origin_ok"])
        and limit_distance_ok
        and all(limit_quality.values())
    )
    normal_limit_ready = (
        not hard_blockers
        and plan["complete"]
        and bool(plan["limit_origin_ok"])
        and limit_distance_ok
        and all(normal_limit_quality.values())
    )
    if alt_active_momentum_limit_ok and not normal_limit_ready:
        limit_route = "active-alt momentum limit"
    elif alt_confirmed_limit_ok and not normal_limit_ready:
        limit_route = "alt-confirmed microstructure limit"
    elif liquid_flow_override:
        limit_route = "liquid-flow limit"
    elif strong_limit_override:
        limit_route = "strong-flow limit"
    else:
        limit_route = "standard limit"
    armed_ready = (
        not hard_blockers
        and plan["complete"]
        and bool(plan["limit_origin_ok"])
        and armed_distance_ok
        and not limit_distance_ok
        and score >= float(gate["limit_min_selection_score"]) - 4.0
        and setup_score >= float(gate["limit_min_setup_score"]) - 4.0
    )
    market_checks = {
        "distance": distance is not None and distance <= dynamic_band,
        "market_behavior": bool(diag.get("market_behavior_ok", True)),
        "ltf_trigger": diag["ltf_trigger"] >= float(gate["min_ltf_trigger"]),
        "entry_quality": diag["entry_quality"] >= float(gate["min_entry_quality"]),
        "execution_score": exec_score >= float(gate["market_min_execution_score"]),
        "rr": rr >= float(gate["strict_min_rr"]),
        "spread": spread <= (0.08 if instrument == "altcoin" else 0.06 if instrument == "large_altcoin" else 0.05),
        "origin": bool(plan["market_origin_ok"]),
    }
    market_ready = limit_ready and all(market_checks.values())

    if market_ready:
        code = "market"
        status = "EXECUTABLE_MARKET"
        should_execute = True
        reason = (
            f"market gate passed: execution={exec_score:.1f}, setup={setup_score:.1f}, "
            f"RR {rr:.2f}R >= {required_rr:.2f}R, distance {distance:.3f}% <= band {dynamic_band:.3f}%"
        )
    elif limit_ready:
        code = "limit"
        status = "EXECUTABLE_LIMIT"
        should_execute = True
        reason = (
            f"limit gate passed ({limit_route}): complete {plan['origin']} plan, selection={score:.1f}, setup={setup_score:.1f}, "
            f"execution={exec_score:.1f}, RR {rr:.2f}R, distance {distance:.3f}% <= {limit_distance_budget:.3f}%"
        )
    elif missed:
        code = "watch"
        status = "MISSED"
        should_execute = False
        reason = f"entry distance {distance:.2f}% is stale/missed; wait for a new setup"
    elif armed_ready:
        code = "watch"
        status = "ARMED_WAIT_ENTRY"
        should_execute = False
        reason = f"setup is tradable but price is not in executable limit band; wait for entry zone trigger"
    elif hard_blockers:
        code = "avoid"
        status = "INVALID" if any("incomplete trade plan" in item or "fallback" in item for item in hard_blockers) else "BLOCKED_RISK"
        should_execute = False
        reason = hard_blockers[0]
    elif score >= 58.0:
        code = "watch"
        status = "WATCH"
        should_execute = False
        failed = [key for key, ok in limit_quality.items() if not ok]
        reason = "watch: " + (", ".join(failed[:3]) if failed else (soft_warnings[0] if soft_warnings else "needs next trigger"))
    else:
        code = "avoid"
        status = "INVALID"
        should_execute = False
        reason = f"selection_score={score:.1f} / execution_score={exec_score:.1f} insufficient"

    checks = {
        "instrument_class": gate["instrument_class"],
        "selected_direction": report.selected_direction in {"long", "short"},
        "score_gap": score_gap >= float(gate["min_score_gap"]),
        "complete_trade_plan": plan["structurally_complete"],
        "entry_origin_valid": plan["limit_origin_ok"],
        "entry_anchor": entry_anchor_ok,
        "entry_anchor_score": round(entry_anchor_score, 2) if entry_anchor_score is not None else None,
        "mover_profile": mover_profile,
        "mover_chase_risk": "yes" if mover_chase_risk else "no",
        "mover_chase_control": not mover_chase_risk,
        "mover_execution_permission": "yes" if isinstance(side.market_metrics, dict) and side.market_metrics.get("mover_execution_permission") else "no",
        "rr": plan["rr_ok"],
        "core_data_quality": diag["core_data_quality"],
        "data_completeness": diag["core_data_quality"] >= float(gate["min_core_data_quality"]),
        "configured_api_ready": diag.get("configured_api_ready", True),
        "external_derivatives": diag.get("external_api_ok", False),
        "alt_external_data_ready": alt_external_data_ready,
        "derivatives_context": diag["derivatives_context_score"] >= float(gate["min_derivatives_context"]),
        "derivatives_not_overheated": not diag["derivative_blocked"],
        "market_not_overheated": not diag["market_overheated"],
        "market_behavior": behavior_ok,
        "limit_behavior_ok": limit_behavior_ok,
        "orderflow_not_hard_rejected": not hard_behavior_block,
        "strong_limit_override": strong_limit_override,
        "liquid_flow_override": liquid_flow_override,
        "alt_confirmed_limit_ready": alt_confirmed_limit_ok,
        "alt_active_momentum_limit_ready": alt_active_momentum_limit_ok,
        "alt_confirmed_limit_liquidity": bool(alt_confirmed_limit["checks"]["liquidity"]),
        "alt_confirmed_limit_behavior_proxy": bool(alt_confirmed_limit["checks"]["behavior_proxy"]),
        "alt_confirmed_limit_rr_edge": bool(alt_confirmed_limit["checks"]["rr_edge"]),
        "alt_confirmed_limit_flow_support": bool(alt_confirmed_limit["checks"]["flow_support"]),
        "alt_confirmed_limit_depth_not_hostile": bool(alt_confirmed_limit["checks"]["depth_not_hostile"]),
        "alt_active_momentum_limit_failed": alt_active_momentum_limit["failed"],
        "alt_active_momentum_limit_reason": alt_active_momentum_limit["reason"],
        "alt_active_momentum_limit_flow_alignment": alt_active_momentum_limit["flow_alignment"],
        "alt_active_momentum_limit_depth_alignment": alt_active_momentum_limit["depth_alignment"],
        "alt_active_momentum_limit_behavior_score": alt_active_momentum_limit["behavior_score"],
        "alt_active_momentum_limit_flow_support_mode": alt_active_momentum_limit["flow_support_mode"],
        "alt_active_momentum_limit_score_floor": alt_active_momentum_limit["score_floor"],
        "alt_active_momentum_limit_setup_floor": alt_active_momentum_limit["setup_floor"],
        "alt_active_momentum_limit_execution_floor": alt_active_momentum_limit["execution_floor"],
        "alt_active_momentum_limit_risk_floor": alt_active_momentum_limit["risk_floor"],
        "alt_active_momentum_limit_behavior_floor": alt_active_momentum_limit["behavior_floor"],
        "entry_zone_width_pct": round(entry_zone_width_pct, 4),
        "dynamic_entry_band_pct": round(dynamic_band, 4),
        "limit_distance_budget_pct": round(limit_distance_budget, 4),
        "entry_distance": bool(limit_distance_ok),
        "limit_entry_distance": bool(limit_distance_ok),
        "market_distance": market_checks["distance"],
        "limit_ready": limit_ready,
        "market_ready": market_ready,
        "armed_ready": armed_ready,
        "precision_limit_ready": limit_ready,
        "normal_limit_ready": normal_limit_ready,
        "market_momentum_ready": market_ready,
        "precision_limit_liquid_flow": limit_ready and bool(diag.get("market_behavior_ok", True)),
        "alt_confirmed_limit_failed": alt_confirmed_limit["failed"],
        "alt_confirmed_limit_reason": alt_confirmed_limit["reason"],
        "alt_confirmed_limit_flow_alignment": alt_confirmed_limit["flow_alignment"],
        "alt_confirmed_limit_depth_alignment": alt_confirmed_limit["depth_alignment"],
        "alt_confirmed_limit_behavior_score": alt_confirmed_limit["behavior_score"],
        "alt_confirmed_limit_evidence_count": alt_confirmed_limit["evidence_count"],
        "alt_confirmed_limit_directional_evidence_count": alt_confirmed_limit["directional_evidence_count"],
        "required_rr": round(required_rr, 2),
        **{f"limit_{key}": value for key, value in limit_quality.items()},
        **{f"market_{key}": value for key, value in market_checks.items()},
    }
    hard_blockers = _dedupe_text(hard_blockers)
    soft_warnings = _dedupe_text(soft_warnings + visible_warnings(side))
    blockers = hard_blockers if hard_blockers else ([] if should_execute else soft_warnings)
    failed_gate_reasons = [key for key, value in checks.items() if isinstance(value, bool) and not value]
    if should_execute:
        failed_gate_reasons = []
    return {
        "code": code,
        "label": ACTION_LABELS.get(code, code),
        "reason": reason,
        "entry_distance_pct": distance,
        "dynamic_entry_band_pct": dynamic_band,
        "entry_proximity_state": proximity["state"],
        "should_execute": should_execute,
        "execution_status": status,
        "execution_status_label": EXECUTION_STATUS_LABELS.get(status, status),
        "gate_version": GATE_VERSION,
        "entry_origin": plan["origin"],
        "entry_validity": plan["validity"],
        "hard_blockers": hard_blockers,
        "soft_warnings": soft_warnings,
        "blocker_categories": _blocker_categories(hard_blockers + soft_warnings),
        "blockers": blockers,
        "warnings": soft_warnings,
        "invalidation_conditions": invalidation_conditions(report),
        "gate_checks": checks,
        "paid_data_status": paid_data_status(report),
        "required_next_trigger": _required_next_trigger(code, proximity),
        "execution_quality": round(exec_score, 2),
        "failed_gate_reasons": failed_gate_reasons,
        "primary_failed_reason": failed_gate_reasons[0] if failed_gate_reasons else "",
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _spread_pct(report: SymbolReport) -> float:
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives", {})
    if not isinstance(public, dict):
        public = {}
    spread = _as_float(public.get("spread_pct"))
    if spread is not None:
        return max(0.0, spread)
    if report.quote_volume_24h >= 250_000_000:
        return 0.02
    if report.quote_volume_24h >= 80_000_000:
        return 0.04
    if report.quote_volume_24h >= 20_000_000:
        return 0.08
    return 0.15


def _required_next_trigger(code: str, proximity: dict[str, Any]) -> str:
    if code in {"market", "limit"}:
        return "已通過 execution gate；下單前確認滑價、部位大小與交易所深度。"
    if proximity.get("state") == "near_entry":
        return "價格已接近 entry，等待 5m/15m micro BOS、displacement 或量能確認。"
    if proximity.get("state") in {"approaching_entry", "far_from_entry"}:
        return "等待價格回到動態 entry band，再重新確認 LTF trigger。"
    if proximity.get("state") == "missed":
        return "已遠離原 entry，不追價；等待新的 FVG/OB/OTE 計畫。"
    return "等待完整 entry / stop / TP 與下一根確認 K 線。"
