from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .instrument_classifier import volatility_profile
from .models import DirectionScore, SymbolReport
from .risk.execution_gate import (
    core_data_quality,
    derivatives_heat_profile,
    entry_distance_pct,
    execution_gate_profile,
    external_derivatives_available,
    market_behavior_confirmation,
    market_risk,
    paid_data_status,
    quant_diagnostics,
    required_min_rr,
    selected_side,
    side_score,
)


PREDICTION_ENGINE_WEIGHTS = {
    "trend_engine": 0.18,
    "market_structure_engine": 0.16,
    "momentum_engine": 0.12,
    "liquidity_engine": 0.12,
    "smart_money_engine": 0.13,
    "derivatives_engine": 0.08,
    "orderflow_engine": 0.07,
    "regime_engine": 0.06,
    "relative_strength_engine": 0.05,
    "risk_context_engine": 0.03,
}


def build_layered_analysis(
    report: SymbolReport,
    regime: dict[str, Any],
    direction_analysis: dict[str, Any],
    gate: dict[str, Any],
    proximity: dict[str, Any],
    opportunity: dict[str, Any],
) -> dict[str, Any]:
    """Build the four-layer trade diagnostic contract for one symbol.

    The prediction layer intentionally avoids entry distance, RR, API readiness,
    and executable checks. Those are handled by trade_plan and execution.
    """

    side = selected_side(report)
    diagnostics = quant_diagnostics(report)
    prediction = _prediction_layer(report, regime, direction_analysis, diagnostics)
    setup = _setup_layer(report, side, regime, prediction, diagnostics, opportunity)
    trade_plan = _trade_plan_layer(report, side, diagnostics, gate, prediction, setup)
    execution = _execution_layer(report, side, diagnostics, gate, trade_plan)
    no_trade = _no_trade_layer(prediction, setup, trade_plan, execution, gate, proximity)
    signal_state = _signal_state(prediction, setup, trade_plan, execution, no_trade)
    data_quality = _data_quality(report, side, diagnostics, trade_plan)
    session_context = _session_context(report)
    return {
        "version": 1,
        "prediction": prediction,
        "setup": setup,
        "trade_plan": trade_plan,
        "execution": execution,
        "data_quality": data_quality,
        "session_context": session_context,
        "signal_state": signal_state,
        "no_trade_type": no_trade["no_trade_type"],
        "primary_blocker": no_trade["primary_blocker"],
        "secondary_blockers": no_trade["secondary_blockers"],
        "next_required_condition": no_trade["next_required_condition"],
        "no_trade": no_trade,
    }


def build_layered_market_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = {
        "no_direction_count": 0,
        "direction_but_no_entry_count": 0,
        "direction_but_bad_rr_count": 0,
        "direction_but_execution_bad_count": 0,
        "direction_conflict_count": 0,
        "hard_block_count": 0,
        "watchlist_count": 0,
        "setup_ready_count": 0,
        "executable_count": 0,
    }
    candidates = [_candidate_payload(row) for row in rows if row.get("layered_analysis")]
    for candidate in candidates:
        no_trade_type = candidate["no_trade_type"]
        signal_state = candidate["signal_state"]
        if no_trade_type == "No Direction":
            diagnostics["no_direction_count"] += 1
        elif no_trade_type == "Direction But No Entry":
            diagnostics["direction_but_no_entry_count"] += 1
        elif no_trade_type == "Direction But Bad RR":
            diagnostics["direction_but_bad_rr_count"] += 1
        elif no_trade_type == "Direction But Execution Bad":
            diagnostics["direction_but_execution_bad_count"] += 1
        elif no_trade_type == "Direction Conflict":
            diagnostics["direction_conflict_count"] += 1
        elif no_trade_type == "Hard Block":
            diagnostics["hard_block_count"] += 1
        if signal_state in {"bias_only", "watchlist"}:
            diagnostics["watchlist_count"] += 1
        elif signal_state == "setup_ready":
            diagnostics["setup_ready_count"] += 1
        elif signal_state in {"limit_executable", "market_executable"}:
            diagnostics["executable_count"] += 1

    long_candidates = [item for item in candidates if item["direction"] == "long"]
    short_candidates = [item for item in candidates if item["direction"] == "short"]
    watchlist = [item for item in candidates if item["signal_state"] in {"bias_only", "watchlist"}]
    setup_ready = [item for item in candidates if item["signal_state"] == "setup_ready"]
    return {
        "no_trade_diagnostics": diagnostics,
        "top_long_candidates": _sort_candidates(long_candidates)[:5],
        "top_short_candidates": _sort_candidates(short_candidates)[:5],
        "top_watchlist": _sort_candidates(watchlist)[:10],
        "top_setup_ready": _sort_candidates(setup_ready)[:10],
    }


def _prediction_layer(
    report: SymbolReport,
    regime: dict[str, Any],
    direction_analysis: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    long_engines = _engine_bundle(report, report.long, "long", regime)
    short_engines = _engine_bundle(report, report.short, "short", regime)
    long_weights = _prediction_engine_weights(report, report.long)
    short_weights = _prediction_engine_weights(report, report.short)
    long_score = _weighted_engine_score(long_engines, long_weights)
    short_score = _weighted_engine_score(short_engines, short_weights)
    edge = abs(long_score - short_score)
    direction = "long" if long_score > short_score else "short"
    fatal_contradiction = _fatal_prediction_contradiction(report, direction_analysis, diagnostics)
    regime_name = str(regime.get("regime") or "")
    strong_liquidity = max(long_engines["liquidity_engine"]["score"], short_engines["liquidity_engine"]["score"]) >= 72.0
    if fatal_contradiction or edge < 7.0 or max(long_score, short_score) < 48.0:
        direction = "neutral"
    elif regime_name in {"squeeze", "low_liquidity"} and edge < 14.0 and not strong_liquidity:
        direction = "neutral"
    selected_engines = long_engines if direction == "long" else short_engines if direction == "short" else {}
    agreement = _engine_agreement(selected_engines, long_engines, short_engines, direction)
    confidence = _prediction_confidence(long_score, short_score, edge, agreement, fatal_contradiction)
    reasons = _prediction_reasons(direction, long_score, short_score, selected_engines, direction_analysis, regime)
    dominant = _dominant_thesis(direction, selected_engines, regime)
    return {
        "prediction_score_long": round(long_score, 2),
        "prediction_score_short": round(short_score, 2),
        "prediction_direction": direction,
        "prediction_edge": round(edge, 2),
        "prediction_confidence": round(confidence, 2),
        "prediction_reason": reasons,
        "dominant_thesis": dominant["text"],
        "dominant_thesis_details": dominant,
        "fatal_contradiction": fatal_contradiction,
        "engines": {
            "long": long_engines,
            "short": short_engines,
        },
        "engine_weights": {
            "long": {key: round(value, 4) for key, value in long_weights.items()},
            "short": {key: round(value, 4) for key, value in short_weights.items()},
        },
    }


def _setup_layer(
    report: SymbolReport,
    side: DirectionScore,
    regime: dict[str, Any],
    prediction: dict[str, Any],
    diagnostics: dict[str, Any],
    opportunity: dict[str, Any],
) -> dict[str, Any]:
    direction = prediction["prediction_direction"]
    setup_quality = float(side.setup_score if side.setup_score is not None else side_score(side))
    feature = lambda name: _feature_pct(side, name)
    strategy_code = str((opportunity.get("strategy_profile") or {}).get("code") or "")
    if direction == "neutral":
        setup_type = "squeeze_wait" if str(regime.get("regime")) == "squeeze" else "no_clear_setup"
    elif strategy_code == "sweep_reversal" or (
        feature("liquidity_sweep") >= 60 and feature("mss_bos") >= 55 and feature("fvg") >= 45
    ):
        setup_type = "ict_bullish_reversal" if direction == "long" else "ict_bearish_reversal"
    elif strategy_code == "trend_continuation":
        setup_type = "trend_continuation"
    elif strategy_code == "pullback_continuation":
        setup_type = "pullback_continuation"
    elif feature("breakout_quality") >= 65 and diagnostics["ltf_trigger"] >= 62:
        setup_type = "breakout_acceptance"
    elif feature("fvg") >= 55 and feature("ote") >= 45:
        setup_type = "ote_fvg_entry"
    elif feature("amd") >= 45:
        setup_type = "amd_long" if direction == "long" else "amd_short"
    else:
        setup_type = "no_clear_setup" if setup_quality < 52 else "watchlist_setup"
    setup_ready = setup_type not in {"no_clear_setup", "squeeze_wait"} and setup_quality >= 58.0
    ict_context = _ict_context(report, side, prediction, diagnostics)
    reasons = [
        f"setup_quality={setup_quality:.1f}",
        f"HTF={diagnostics['htf_context']:.1f}, LTF={diagnostics['ltf_trigger']:.1f}, entry={diagnostics['entry_quality']:.1f}",
    ]
    if strategy_code:
        reasons.append(f"best_strategy_profile={strategy_code}")
    return {
        "setup_type": setup_type,
        "setup_quality": round(setup_quality, 2),
        "setup_ready": setup_ready,
        "setup_reason": reasons,
        "ict_context": ict_context,
    }


def _trade_plan_layer(
    report: SymbolReport,
    side: DirectionScore,
    diagnostics: dict[str, Any],
    gate: dict[str, Any],
    prediction: dict[str, Any],
    setup: dict[str, Any],
) -> dict[str, Any]:
    entry_zone = _zone_to_list(side.entry_zone)
    entry_price = _conservative_entry_price(side.entry_zone, side.direction)
    take_profit = _take_profit(side)
    required_rr = required_min_rr(report, side, diagnostics)
    estimated_rr = _as_float(side.rr)
    cost = _execution_costs(report, side, gate)
    limit_net_rr = _net_rr(estimated_rr, entry_price, side.stop, take_profit, cost["limit_cost_pct"])
    market_net_rr = _net_rr(estimated_rr, entry_price, side.stop, take_profit, cost["market_cost_pct"])
    net_rr = limit_net_rr if limit_net_rr is not None else market_net_rr
    complete = bool(entry_zone and side.stop is not None and take_profit is not None)
    setup_clear = prediction["prediction_direction"] != "neutral" and setup["setup_type"] != "no_clear_setup"
    if not complete:
        status = "incomplete"
    elif net_rr is None or net_rr < required_rr:
        status = "invalid"
    elif not setup_clear:
        status = "invalid"
    else:
        status = "valid"
    reasons = []
    if not complete:
        reasons.append("entry/stop/take_profit plan is incomplete")
    if prediction["prediction_direction"] == "neutral":
        reasons.append("prediction direction is neutral")
    if setup["setup_type"] == "no_clear_setup":
        reasons.append("setup is not clear enough")
    if net_rr is not None and net_rr < required_rr:
        reasons.append(f"net_rr {net_rr:.2f} is below required_rr {required_rr:.2f}")
    if not reasons:
        reasons.append("entry, stop, target, and net RR are available")
    return {
        "direction": prediction["prediction_direction"],
        "setup_type": setup["setup_type"],
        "entry_zone": entry_zone,
        "entry_price": entry_price,
        "entry_price_basis": "worst_fill_inside_limit_zone" if entry_price is not None else None,
        "stop_loss": side.stop,
        "take_profit": take_profit,
        "invalidation_level": side.stop,
        "target_liquidity": take_profit,
        "required_rr": round(required_rr, 2),
        "estimated_rr": round(estimated_rr, 4) if estimated_rr is not None else None,
        "net_rr": round(net_rr, 4) if net_rr is not None else None,
        "limit_net_rr": round(limit_net_rr, 4) if limit_net_rr is not None else None,
        "market_net_rr": round(market_net_rr, 4) if market_net_rr is not None else None,
        "estimated_fee": cost["estimated_fee_pct"],
        "estimated_slippage": cost["estimated_slippage_pct"],
        "plan_status": status,
        "plan_reason": reasons,
        "net_rr_valid": bool(net_rr is not None and net_rr >= required_rr),
    }


def _execution_layer(
    report: SymbolReport,
    side: DirectionScore,
    diagnostics: dict[str, Any],
    gate: dict[str, Any],
    trade_plan: dict[str, Any],
) -> dict[str, Any]:
    checks = gate.get("gate_checks", {}) if isinstance(gate.get("gate_checks"), dict) else {}
    code = str(gate.get("code") or "")
    plan_valid = trade_plan["plan_status"] == "valid"
    limit_executable = bool(plan_valid and gate.get("should_execute") and code == "limit")
    market_executable = bool(plan_valid and gate.get("should_execute") and code == "market")
    spread_pct = _as_float(checks.get("market_spread_pct"))
    spread_limit = _as_float(checks.get("market_spread_limit"))
    behavior = diagnostics.get("market_behavior_confirmation", {})
    failed = list(gate.get("failed_gate_reasons", []))
    blockers = list(gate.get("blockers", []))
    primary = str(gate.get("primary_failed_reason") or (failed[0] if failed else "") or "")
    return {
        "executable": limit_executable or market_executable,
        "order_type": "market" if market_executable else "limit" if limit_executable else "none",
        "execution_score": gate.get("execution_quality", side.execution_score),
        "limit_executable": limit_executable,
        "market_executable": market_executable,
        "execution_blockers": blockers or failed,
        "primary_blocker": primary,
        "slippage_ok": bool(checks.get("market_distance", False) or checks.get("market_momentum_ready", False)),
        "spread_ok": spread_pct is None or spread_limit is None or spread_pct <= spread_limit,
        "depth_ok": bool(diagnostics.get("market_behavior_ok", True) or not behavior),
        "api_ok": bool(diagnostics.get("configured_api_ready", True)),
        "market_distance_ok": bool(checks.get("market_tight_ready", False)),
        "momentum_trigger_strong": bool(checks.get("market_momentum_ready", False)),
        "estimated_slippage": trade_plan["estimated_slippage"],
        "estimated_fee": trade_plan["estimated_fee"],
        "market_net_rr": trade_plan["market_net_rr"],
        "limit_net_rr": trade_plan["limit_net_rr"],
    }


def _no_trade_layer(
    prediction: dict[str, Any],
    setup: dict[str, Any],
    trade_plan: dict[str, Any],
    execution: dict[str, Any],
    gate: dict[str, Any],
    proximity: dict[str, Any],
) -> dict[str, Any]:
    blockers = list(execution.get("execution_blockers", []))
    if execution["executable"]:
        no_trade_type = "Not No Trade"
        primary = ""
    elif _is_hard_block(gate, trade_plan, execution):
        no_trade_type = "Hard Block"
        primary = execution.get("primary_blocker") or "hard_block"
    elif prediction["fatal_contradiction"]:
        no_trade_type = "Direction Conflict"
        primary = "fatal_prediction_contradiction"
    elif prediction["prediction_direction"] == "neutral":
        no_trade_type = "No Direction"
        primary = "prediction_edge_too_low"
    elif trade_plan["plan_status"] == "invalid" and not trade_plan["net_rr_valid"]:
        no_trade_type = "Direction But Bad RR"
        primary = "net_rr_below_required"
    elif trade_plan["plan_status"] == "incomplete" or proximity.get("state") in {"far_from_entry", "missed", "approaching_entry"}:
        no_trade_type = "Direction But No Entry"
        primary = "trade_plan_incomplete" if trade_plan["plan_status"] == "incomplete" else "price_not_in_entry_zone"
    elif setup["setup_ready"]:
        no_trade_type = "Direction But Execution Bad"
        primary = execution.get("primary_blocker") or "execution_gate_not_passed"
    else:
        no_trade_type = "Direction But No Entry"
        primary = "setup_not_ready"
    secondary = [str(item) for item in blockers if item and str(item) != primary][:5]
    next_required = str(gate.get("required_next_trigger") or "")
    if not next_required:
        next_required = _default_next_condition(no_trade_type, setup, trade_plan)
    return {
        "no_trade_type": no_trade_type,
        "primary_blocker": primary,
        "secondary_blockers": secondary,
        "next_required_condition": next_required,
    }


def _signal_state(
    prediction: dict[str, Any],
    setup: dict[str, Any],
    trade_plan: dict[str, Any],
    execution: dict[str, Any],
    no_trade: dict[str, Any],
) -> str:
    if execution["market_executable"]:
        return "market_executable"
    if execution["limit_executable"]:
        return "limit_executable"
    if no_trade["no_trade_type"] in {"Hard Block", "Direction Conflict"}:
        return "blocked"
    if prediction["prediction_direction"] == "neutral":
        return "neutral"
    if no_trade["no_trade_type"] in {"Direction But No Entry", "Direction But Bad RR"}:
        return "watchlist"
    if trade_plan["plan_status"] == "valid":
        return "setup_ready"
    if setup["setup_ready"] or trade_plan["plan_status"] == "invalid":
        return "watchlist"
    return "bias_only"


def _data_quality(
    report: SymbolReport,
    side: DirectionScore,
    diagnostics: dict[str, Any],
    trade_plan: dict[str, Any],
) -> dict[str, Any]:
    gate = execution_gate_profile(report)
    core_score = core_data_quality(report, side)
    missing_core: list[str] = []
    if report.price <= 0:
        missing_core.append("price")
    if not report.data_coverage:
        missing_core.append("OHLCV")
    if trade_plan["direction"] in {"long", "short"} and trade_plan["plan_status"] == "incomplete":
        if not trade_plan["entry_zone"]:
            missing_core.append("entry")
        if trade_plan["stop_loss"] is None:
            missing_core.append("stop")
        if trade_plan["take_profit"] is None:
            missing_core.append("TP")
    missing_optional: list[str] = []
    if not external_derivatives_available(report):
        missing_optional.append("derivatives")
    if not isinstance(report.metadata.get("relative_strength"), dict):
        missing_optional.append("relative_strength")
    core_ok = core_score >= float(gate["min_core_data_quality"]) and not missing_core
    optional_ok = not missing_optional
    if not core_ok:
        impact = "blocked"
    elif not optional_ok:
        impact = "confidence_penalty"
    else:
        impact = "no_impact"
    return {
        "core_ok": core_ok,
        "optional_ok": optional_ok,
        "core_data_quality": core_score,
        "optional_data_quality": diagnostics.get("optional_confluence"),
        "missing_core_fields": missing_core,
        "missing_optional_fields": missing_optional,
        "impact": impact,
    }


def _session_context(report: SymbolReport) -> dict[str, Any]:
    timestamp = report.data_time if isinstance(report.data_time, datetime) else datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    hour = timestamp.astimezone(timezone.utc).hour
    if 7 <= hour < 12:
        session = "london"
    elif 12 <= hour < 16:
        session = "overlap"
    elif 16 <= hour < 21:
        session = "new_york"
    elif 0 <= hour < 7:
        session = "asia"
    else:
        session = "off_session"
    return {
        "session": session,
        "killzone": session in {"london", "overlap", "new_york"},
        "silver_bullet_window": hour in {14, 15, 19, 20},
        "session_high_swept": False,
        "session_low_swept": False,
        "news_risk_window": False,
    }


def _engine_bundle(report: SymbolReport, side: DirectionScore, direction: str, regime: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diagnostics = _side_diagnostics(side)
    heat = derivatives_heat_profile(report, direction)
    behavior = market_behavior_confirmation(report, direction)
    market_blocked, market_warning = market_risk(report)
    derivatives = _derivatives_engine(heat, direction)
    orderflow = _orderflow_engine(behavior, direction)
    metrics = side.market_metrics or {}
    quant_derivatives = _as_float(metrics.get("quant_derivatives_score"))
    if quant_derivatives is not None:
        base = _as_float(derivatives.get("score")) or 50.0
        blended = _clamp(base * 0.62 + quant_derivatives * 0.38)
        derivatives["score"] = round(blended, 2)
        derivatives["quant_derivatives_score"] = round(quant_derivatives, 2)
    no_chase = _as_float(metrics.get("quant_no_chase_score"))
    if no_chase is not None:
        base = _as_float(orderflow.get("score")) or 50.0
        orderflow["score"] = round(_clamp(base * 0.82 + no_chase * 0.18), 2)
        orderflow["no_chase_score"] = round(no_chase, 2)
    return {
        "trend_engine": _trend_engine(report, side, direction, regime, diagnostics),
        "market_structure_engine": _market_structure_engine(side, direction, diagnostics),
        "momentum_engine": _momentum_engine(side, direction, diagnostics),
        "liquidity_engine": _liquidity_engine(side, direction),
        "smart_money_engine": _smart_money_engine(side, direction),
        "derivatives_engine": derivatives,
        "orderflow_engine": orderflow,
        "regime_engine": _regime_engine(regime, direction),
        "relative_strength_engine": _relative_strength_engine(report, direction),
        "risk_context_engine": _risk_context_engine(report, market_blocked, market_warning),
    }


def _trend_engine(
    report: SymbolReport,
    side: DirectionScore,
    direction: str,
    regime: dict[str, Any],
    diagnostics: dict[str, float],
) -> dict[str, Any]:
    score = diagnostics["htf_context"] * 0.65 + _feature_pct(side, "key_level") * 0.15 + _regime_bias_score(regime, direction) * 0.20
    metrics = side.market_metrics or {}
    if metrics.get("btc_against"):
        score -= 8.0
    bias = direction if score >= 58 else "neutral"
    return {
        "trend_bias": bias,
        "trend_strength": round(_clamp(score), 2),
        "score": round(_clamp(score), 2),
        "trend_reason": [
            f"htf_context={diagnostics['htf_context']:.1f}",
            f"regime_bias={_regime_bias_score(regime, direction):.1f}",
        ],
    }


def _market_structure_engine(side: DirectionScore, direction: str, diagnostics: dict[str, float]) -> dict[str, Any]:
    score = (
        _feature_pct(side, "mss_bos") * 0.48
        + _feature_pct(side, "breakout_quality") * 0.20
        + _feature_pct(side, "key_level") * 0.18
        + _feature_pct(side, "liquidity_sweep") * 0.14
    )
    if score >= 72:
        state = "bullish_structure" if direction == "long" else "bearish_structure"
    elif score >= 58:
        state = "transition"
    elif diagnostics["htf_context"] < 42:
        state = "unclear"
    else:
        state = "range"
    return {
        "structure_bias": direction if score >= 58 else "neutral",
        "structure_state": state,
        "score": round(_clamp(score), 2),
        "structure_reason": [
            f"mss_bos={_feature_pct(side, 'mss_bos'):.1f}",
            f"breakout_quality={_feature_pct(side, 'breakout_quality'):.1f}",
        ],
    }


def _momentum_engine(side: DirectionScore, direction: str, diagnostics: dict[str, float]) -> dict[str, Any]:
    metrics = side.market_metrics or {}
    volume_score = 50.0
    volume_ratio = _as_float(metrics.get("volume_ratio"))
    if volume_ratio is not None:
        volume_score = _clamp(35.0 + volume_ratio * 18.0)
    score = (
        _feature_pct(side, "displacement") * 0.34
        + _feature_pct(side, "price_action") * 0.24
        + _feature_pct(side, "breakout_quality") * 0.20
        + volume_score * 0.22
    )
    state = "expanding" if score >= 70 else "fading" if score < 45 else "neutral"
    return {
        "momentum_bias": direction if score >= 55 else "neutral",
        "momentum_strength": round(_clamp(score), 2),
        "momentum_state": state,
        "score": round(_clamp(score), 2),
        "momentum_reason": [
            f"displacement={_feature_pct(side, 'displacement'):.1f}",
            f"volume_score={volume_score:.1f}",
        ],
    }


def _liquidity_engine(side: DirectionScore, direction: str) -> dict[str, Any]:
    sweep = _feature_pct(side, "liquidity_sweep")
    score = sweep * 0.62 + _feature_pct(side, "key_level") * 0.20 + _feature_pct(side, "htf_poi") * 0.18
    if sweep >= 60:
        event = "sell_side_sweep" if direction == "long" else "buy_side_sweep"
    elif score >= 55:
        event = "inducement"
    else:
        event = "none"
    return {
        "liquidity_bias": direction if score >= 55 else "neutral",
        "liquidity_event": event,
        "liquidity_quality": round(_clamp(score), 2),
        "score": round(_clamp(score), 2),
        "liquidity_reason": [f"liquidity_sweep={sweep:.1f}"],
    }


def _smart_money_engine(side: DirectionScore, direction: str) -> dict[str, Any]:
    components = [
        _feature_pct(side, "htf_poi"),
        _feature_pct(side, "liquidity_sweep"),
        _feature_pct(side, "mss_bos"),
        _feature_pct(side, "displacement"),
        _feature_pct(side, "fvg"),
        _feature_pct(side, "ote"),
    ]
    score = sum(components) / len(components)
    if score >= 70 and _feature_pct(side, "liquidity_sweep") >= 55:
        setup = "ict_bullish_reversal" if direction == "long" else "ict_bearish_reversal"
    elif _feature_pct(side, "fvg") >= 55 and _feature_pct(side, "mss_bos") >= 45:
        setup = "fvg_bos"
    elif _feature_pct(side, "ote") >= 45:
        setup = "ote_fvg"
    elif _feature_pct(side, "amd") >= 45:
        setup = "amd_long" if direction == "long" else "amd_short"
    else:
        setup = "none"
    return {
        "smart_money_bias": direction if score >= 55 else "neutral",
        "smart_money_setup": setup,
        "smart_money_quality": round(_clamp(score), 2),
        "score": round(_clamp(score), 2),
        "smart_money_reason": [
            f"fvg={_feature_pct(side, 'fvg'):.1f}",
            f"ote={_feature_pct(side, 'ote'):.1f}",
            f"mss_bos={_feature_pct(side, 'mss_bos'):.1f}",
        ],
    }


def _derivatives_engine(heat: dict[str, Any], direction: str) -> dict[str, Any]:
    trend = _as_float(heat.get("trend_confirmation_score")) or 0.0
    crowding = _as_float(heat.get("crowding_score")) or 0.0
    exhaustion = _as_float(heat.get("exhaustion_score")) or 0.0
    score = _clamp(48.0 + trend * 0.42 - max(0.0, crowding - 55.0) * 0.35 - max(0.0, exhaustion - 55.0) * 0.35)
    state = str(heat.get("state") or "neutral")
    return {
        "derivatives_bias": direction if score >= 55 else "neutral",
        "derivatives_state": state,
        "derivatives_strength": round(score, 2),
        "funding_heat": heat.get("heat_score"),
        "oi_heat": heat.get("heat_score"),
        "score": round(score, 2),
        "derivatives_reason": list(heat.get("reasons", []) or heat.get("warnings", []) or [])[:4],
    }


def _orderflow_engine(behavior: dict[str, Any], direction: str) -> dict[str, Any]:
    score = _as_float(behavior.get("score")) or (64.0 if behavior.get("ok") else 42.0)
    blockers = behavior.get("blockers", []) if isinstance(behavior.get("blockers"), list) else []
    state = "aggressive_buying" if direction == "long" and score >= 65 else "aggressive_selling" if direction == "short" and score >= 65 else "neutral"
    if blockers:
        state = "exhaustion"
    return {
        "orderflow_bias": direction if score >= 55 else "neutral",
        "orderflow_strength": round(_clamp(score), 2),
        "orderflow_state": state,
        "score": round(_clamp(score), 2),
        "orderflow_reason": [str(behavior.get("reason") or "market behavior proxy")],
    }


def _regime_engine(regime: dict[str, Any], direction: str) -> dict[str, Any]:
    name = str(regime.get("regime") or "range")
    mapped = {
        "trend_up": "trend",
        "trend_down": "trend",
        "range": "range",
        "squeeze": "squeeze",
        "high_volatility": "expansion",
        "low_liquidity": "chop",
        "risk_off": "risk_off",
        "alt_rotation": "expansion",
    }.get(name, name or "range")
    score = _regime_bias_score(regime, direction)
    return {
        "regime": mapped,
        "regime_confidence": regime.get("confidence", 0.0),
        "score": round(score, 2),
        "regime_reason": list(regime.get("notes", []))[:3],
    }


def _relative_strength_engine(report: SymbolReport, direction: str) -> dict[str, Any]:
    metrics = report.metadata.get("relative_strength", {})
    if direction == "short":
        score = _as_float(metrics.get("relative_strength_score_short")) or 50.0
    else:
        score = _as_float(metrics.get("relative_strength_score_long")) or 50.0
    return {
        "relative_strength_bias": direction if score >= 55 else "neutral",
        "relative_strength_score": round(_clamp(score), 2),
        "score": round(_clamp(score), 2),
        "relative_strength_reason": [
            f"momentum_24h={metrics.get('momentum_24h')}",
            f"relative_strength_btc={metrics.get('relative_strength_btc')}",
        ],
    }


def _risk_context_engine(report: SymbolReport, market_blocked: bool, market_warning: str) -> dict[str, Any]:
    paid = paid_data_status(report)
    data_score = core_data_quality(report)
    state = "normal"
    if market_blocked or paid.get("blocked") or data_score < 35:
        state = "hard_block"
    elif data_score < 65 or not paid.get("configured_api_ready", True):
        state = "elevated"
    score = 42.0 if state == "hard_block" else 56.0 if state == "elevated" else 68.0
    reasons = []
    if market_warning:
        reasons.append(market_warning)
    if paid.get("warning"):
        reasons.append(str(paid["warning"]))
    reasons.append(f"core_data_quality={data_score:.1f}")
    return {
        "market_risk_state": state,
        "risk_reason": reasons[:4],
        "risk_adjustments": {
            "required_rr_multiplier": 1.25 if state == "elevated" else 1.0,
            "position_size_multiplier": 0.0 if state == "hard_block" else 0.6 if state == "elevated" else 1.0,
            "altcoin_quality_multiplier": 0.85 if state == "elevated" else 1.0,
        },
        "score": score,
    }


def _ict_context(
    report: SymbolReport,
    side: DirectionScore,
    prediction: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    direction = prediction["prediction_direction"]
    entry = _zone_to_list(side.entry_zone)
    premium_discount = "discount" if direction == "long" else "premium" if direction == "short" else "equilibrium"
    fvg_detected = _feature_pct(side, "fvg") >= 45
    mss_detected = _feature_pct(side, "mss_bos") >= 45
    sweep_detected = _feature_pct(side, "liquidity_sweep") >= 45
    return {
        "htf_bias": "bullish" if direction == "long" else "bearish" if direction == "short" else "neutral",
        "htf_poi_type": "demand" if direction == "long" else "supply" if direction == "short" else "none",
        "premium_discount": premium_discount,
        "liquidity_swept": sweep_detected,
        "swept_side": "sell_side" if direction == "long" and sweep_detected else "buy_side" if direction == "short" and sweep_detected else "none",
        "sweep_level": None,
        "mss_detected": mss_detected,
        "mss_direction": "bullish" if direction == "long" and mss_detected else "bearish" if direction == "short" and mss_detected else "none",
        "bos_detected": diagnostics["ltf_trigger"] >= 65.0,
        "bos_direction": "bullish" if direction == "long" and diagnostics["ltf_trigger"] >= 65.0 else "bearish" if direction == "short" and diagnostics["ltf_trigger"] >= 65.0 else "none",
        "displacement_detected": _feature_pct(side, "displacement") >= 45,
        "fvg_detected": fvg_detected,
        "fvg_direction": "bullish" if direction == "long" and fvg_detected else "bearish" if direction == "short" and fvg_detected else "none",
        "fvg_zone": entry,
        "ote_zone": entry if _feature_pct(side, "ote") >= 45 else None,
        "entry_zone": entry,
        "entry_zone_basis": _entry_basis(side),
        "amd_phase": _amd_phase(side),
    }


def _candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    layered = row["layered_analysis"]
    prediction = layered["prediction"]
    trade_plan = layered["trade_plan"]
    direction = prediction["prediction_direction"]
    score = prediction["prediction_score_long"] if direction == "long" else prediction["prediction_score_short"] if direction == "short" else max(
        prediction["prediction_score_long"], prediction["prediction_score_short"]
    )
    blockers = layered.get("secondary_blockers", [])
    main_reason = prediction["prediction_reason"][0] if prediction["prediction_reason"] else prediction["dominant_thesis"]
    return {
        "rank": row.get("rank"),
        "symbol": row["symbol"],
        "direction": direction,
        "prediction_score": score,
        "prediction_confidence": prediction["prediction_confidence"],
        "setup_type": layered["setup"]["setup_type"],
        "signal_state": layered["signal_state"],
        "main_reason": main_reason,
        "why_not_executable": layered.get("primary_blocker") or "",
        "next_trigger": layered.get("next_required_condition") or "",
        "entry_zone": trade_plan.get("entry_zone"),
        "invalidation_level": trade_plan.get("invalidation_level"),
        "estimated_rr": trade_plan.get("estimated_rr"),
        "no_trade_type": layered.get("no_trade_type"),
        "secondary_blockers": blockers[:3],
    }


def _sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("prediction_confidence") or 0.0),
            float(item.get("prediction_score") or 0.0),
            1 if item.get("signal_state") in {"limit_executable", "market_executable"} else 0,
        ),
        reverse=True,
    )


def _prediction_engine_weights(report: SymbolReport, side: DirectionScore) -> dict[str, float]:
    weights = dict(PREDICTION_ENGINE_WEIGHTS)
    instrument = volatility_profile(report.symbol).instrument_class
    metrics = side.market_metrics or {}
    mover_profile = str(metrics.get("mover_profile") or "normal")
    is_alt_family = instrument in {"altcoin", "large_altcoin"}
    is_mover = mover_profile in {"active_mover", "hot_mover", "extreme_mover"}
    if is_alt_family:
        weights["derivatives_engine"] += 0.035
        weights["orderflow_engine"] += 0.025
        weights["liquidity_engine"] += 0.020
        weights["trend_engine"] -= 0.035
        weights["regime_engine"] -= 0.015
    if is_mover:
        weights["market_structure_engine"] += 0.035
        weights["liquidity_engine"] += 0.030
        weights["derivatives_engine"] += 0.035
        weights["orderflow_engine"] += 0.025
        weights["momentum_engine"] -= 0.030
        weights["trend_engine"] -= 0.030
        weights["regime_engine"] -= 0.020
    if bool(metrics.get("mover_chase_risk")):
        weights["momentum_engine"] -= 0.035
        weights["trend_engine"] -= 0.025
        weights["risk_context_engine"] += 0.030
        weights["liquidity_engine"] += 0.015
        weights["orderflow_engine"] += 0.015
    if bool(metrics.get("mover_execution_permission")):
        weights["market_structure_engine"] += 0.020
        weights["liquidity_engine"] += 0.015
        weights["risk_context_engine"] += 0.010
    for key, value in list(weights.items()):
        weights[key] = max(0.01, value)
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def _weighted_engine_score(engines: dict[str, dict[str, Any]], weights: dict[str, float] | None = None) -> float:
    active_weights = weights or PREDICTION_ENGINE_WEIGHTS
    total = 0.0
    for name, weight in active_weights.items():
        total += (_as_float(engines.get(name, {}).get("score")) or 0.0) * weight
    return _clamp(total)


def _engine_agreement(
    selected: dict[str, dict[str, Any]],
    long_engines: dict[str, dict[str, Any]],
    short_engines: dict[str, dict[str, Any]],
    direction: str,
) -> float:
    if direction not in {"long", "short"}:
        return 0.0
    count = sum(1 for engine in selected.values() if (_as_float(engine.get("score")) or 0.0) >= 55.0)
    opposite = short_engines if direction == "long" else long_engines
    opposing_count = sum(1 for engine in opposite.values() if (_as_float(engine.get("score")) or 0.0) >= 62.0)
    return _clamp(count / max(len(selected), 1) * 100.0 - opposing_count * 5.0)


def _prediction_confidence(long_score: float, short_score: float, edge: float, agreement: float, fatal: bool) -> float:
    if fatal:
        return 0.0
    leader = max(long_score, short_score)
    return _clamp(leader * 0.55 + edge * 1.35 + agreement * 0.20)


def _prediction_reasons(
    direction: str,
    long_score: float,
    short_score: float,
    engines: dict[str, dict[str, Any]],
    direction_analysis: dict[str, Any],
    regime: dict[str, Any],
) -> list[str]:
    reasons = [f"prediction scores long={long_score:.1f}, short={short_score:.1f}"]
    if direction == "neutral":
        reasons.append("direction is neutral because edge/confidence is not enough for a directional thesis")
    else:
        leaders = sorted(
            ((name, _as_float(data.get("score")) or 0.0) for name, data in engines.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        reasons.append(f"{direction} supported by " + ", ".join(f"{name}={score:.1f}" for name, score in leaders))
    if direction_analysis.get("conflict_level") in {"mild", "high"}:
        reasons.append(f"conflict_level={direction_analysis.get('conflict_level')}")
    if regime.get("regime"):
        reasons.append(f"market_regime={regime.get('regime')}")
    return reasons


def _dominant_thesis(direction: str, engines: dict[str, dict[str, Any]], regime: dict[str, Any]) -> dict[str, Any]:
    if direction == "neutral":
        thesis_type = "squeeze_wait" if str(regime.get("regime")) == "squeeze" else "no_clear_edge"
        return {
            "type": thesis_type,
            "direction": "neutral",
            "confidence": 0.0,
            "supporting_engines": [],
            "contradicting_engines": [],
            "fatal_contradiction": False,
            "text": thesis_type,
        }
    liquidity = _as_float(engines.get("liquidity_engine", {}).get("score")) or 0.0
    structure = _as_float(engines.get("market_structure_engine", {}).get("score")) or 0.0
    trend = _as_float(engines.get("trend_engine", {}).get("score")) or 0.0
    momentum = _as_float(engines.get("momentum_engine", {}).get("score")) or 0.0
    if liquidity >= 68 and structure >= 58:
        thesis_type = "liquidity_reversal"
    elif trend >= 66 and momentum >= 58:
        thesis_type = "trend_continuation"
    elif structure >= 68 and momentum >= 62:
        thesis_type = "breakout_acceptance"
    elif str(regime.get("regime")) == "range":
        thesis_type = "range_reversion"
    else:
        thesis_type = "pullback_continuation"
    supporting = [
        name for name, data in engines.items() if (_as_float(data.get("score")) or 0.0) >= 58.0
    ]
    contradicting = [
        name for name, data in engines.items() if (_as_float(data.get("score")) or 0.0) < 42.0
    ]
    confidence = sum(_as_float(engines[name].get("score")) or 0.0 for name in supporting) / max(len(supporting), 1)
    return {
        "type": thesis_type,
        "direction": direction,
        "confidence": round(_clamp(confidence), 2),
        "supporting_engines": supporting,
        "contradicting_engines": contradicting,
        "fatal_contradiction": False,
        "text": f"{thesis_type}_{direction}",
    }


def _fatal_prediction_contradiction(
    report: SymbolReport,
    direction_analysis: dict[str, Any],
    diagnostics: dict[str, Any],
) -> bool:
    return bool(
        direction_analysis.get("conflict_level") == "high"
        or (report.metadata.get("direction_conflict") and direction_analysis.get("chosen_direction") == "neutral")
        or diagnostics.get("market_overheated")
        or diagnostics.get("derivative_blocked")
    )


def _is_hard_block(gate: dict[str, Any], trade_plan: dict[str, Any], execution: dict[str, Any]) -> bool:
    failed = set(gate.get("failed_gate_reasons", []))
    if trade_plan["plan_status"] == "incomplete":
        failed = failed - {"complete_trade_plan"}
    hard = {
        "configured_api_ready",
        "derivatives_not_overheated",
        "market_not_overheated",
        "data_completeness",
        "direction_not_conflicted",
    }
    if failed.intersection(hard):
        return True
    return bool(execution.get("primary_blocker") in hard)


def _default_next_condition(no_trade_type: str, setup: dict[str, Any], trade_plan: dict[str, Any]) -> str:
    if no_trade_type == "No Direction":
        return "wait for one side to build a clear prediction edge"
    if no_trade_type == "Direction But Bad RR":
        return "wait for a better entry or wider target so net RR clears required_rr"
    if no_trade_type == "Direction But No Entry":
        return "wait for price to return into the planned entry zone or form a new setup"
    if no_trade_type == "Direction But Execution Bad":
        return "wait for spread, depth, slippage, and trigger quality to pass execution checks"
    if no_trade_type == "Hard Block":
        return "resolve the hard blocker before considering execution"
    return f"monitor {setup.get('setup_type')} with plan_status={trade_plan.get('plan_status')}"


def _execution_costs(report: SymbolReport, side: DirectionScore, gate: dict[str, Any]) -> dict[str, float]:
    checks = gate.get("gate_checks", {}) if isinstance(gate.get("gate_checks"), dict) else {}
    spread_pct = _as_float(checks.get("market_spread_pct"))
    if spread_pct is None:
        spread_pct = _spread_pct(report)
    market_slippage = max(0.01, spread_pct * 0.75)
    limit_slippage = max(0.0, spread_pct * 0.15)
    taker_fee = 0.06
    maker_fee = 0.02
    return {
        "estimated_fee_pct": round(taker_fee * 2.0, 4),
        "estimated_slippage_pct": round(market_slippage, 4),
        "market_cost_pct": round(spread_pct + market_slippage + taker_fee * 2.0, 4),
        "limit_cost_pct": round(limit_slippage + maker_fee * 2.0, 4),
    }


def _net_rr(
    rr: float | None,
    entry: float | None,
    stop: float | None,
    target: float | None,
    cost_pct: float,
) -> float | None:
    if rr is None or entry is None or stop is None or target is None:
        return None
    risk_pct = abs(entry - stop) / max(abs(entry), 1e-12) * 100.0
    if risk_pct <= 0:
        return None
    reward_pct_from_target = abs(target - entry) / max(abs(entry), 1e-12) * 100.0
    reward_pct_from_rr = rr * risk_pct
    reward_pct = min(reward_pct_from_target, reward_pct_from_rr)
    return max(0.0, reward_pct - cost_pct) / max(risk_pct + cost_pct, 1e-12)


def _side_diagnostics(side: DirectionScore) -> dict[str, float]:
    buckets = side.bucket_scores or {}
    return {
        "htf_context": float(buckets.get("htf_context", 0.0)),
        "ltf_confirmation": float(buckets.get("ltf_confirmation", 0.0)),
        "entry_location": float(buckets.get("entry_location", 0.0)),
        "risk_plan": float(buckets.get("risk_plan", 0.0)),
        "market_filter": float(buckets.get("market_filter", 0.0)),
    }


def _regime_bias_score(regime: dict[str, Any], direction: str) -> float:
    name = str(regime.get("regime") or "range")
    confidence = _as_float(regime.get("confidence")) or 50.0
    if name == "trend_up":
        return confidence if direction == "long" else 100.0 - confidence
    if name in {"trend_down", "risk_off"}:
        return confidence if direction == "short" else 100.0 - confidence
    if name == "alt_rotation":
        return 62.0
    if name in {"squeeze", "low_liquidity"}:
        return 48.0
    return 54.0


def _feature_pct(side: DirectionScore, name: str) -> float:
    maximum = float(side.feature_max_scores.get(name, 0.0) or 0.0)
    value = float(side.feature_scores.get(name, 0.0) or 0.0)
    if maximum <= 0:
        bucket = _fallback_bucket_for_feature(name)
        if bucket:
            return float((side.bucket_scores or {}).get(bucket, 0.0) or 0.0)
        return 0.0
    return _clamp(value / maximum * 100.0)


def _fallback_bucket_for_feature(name: str) -> str:
    if name in {"liquidity_sweep", "htf_poi", "key_level"}:
        return "htf_context"
    if name in {"mss_bos", "displacement", "price_action", "breakout_quality"}:
        return "ltf_confirmation"
    if name in {"fvg", "ote"}:
        return "entry_location"
    if name == "risk_reward":
        return "risk_plan"
    if name == "market_quality":
        return "market_filter"
    return ""


def _entry_basis(side: DirectionScore) -> list[str]:
    basis = []
    for name in ("fvg", "ote", "htf_poi", "liquidity_sweep", "mss_bos"):
        if _feature_pct(side, name) >= 45:
            basis.append(name)
    return basis


def _amd_phase(side: DirectionScore) -> str:
    score = _feature_pct(side, "amd")
    if score >= 75:
        return "distribution"
    if score >= 45:
        return "manipulation"
    if score > 0:
        return "accumulation"
    return "unknown"


def _take_profit(side: DirectionScore) -> float | None:
    if side.target is not None:
        return side.target
    if side.take_profits:
        for index in (1, 0):
            if index < len(side.take_profits):
                price = _as_float(side.take_profits[index].get("price"))
                if price is not None:
                    return price
    return None


def _midpoint(zone: tuple[float, float] | None) -> float | None:
    if not zone:
        return None
    return (float(zone[0]) + float(zone[1])) / 2.0


def _conservative_entry_price(zone: tuple[float, float] | None, direction: str) -> float | None:
    if not zone:
        return None
    low, high = float(zone[0]), float(zone[1])
    return max(low, high) if direction == "long" else min(low, high)


def _zone_to_list(zone: tuple[float, float] | None) -> list[float] | None:
    if not zone:
        return None
    return [float(zone[0]), float(zone[1])]


def _spread_pct(report: SymbolReport) -> float:
    paid = report.metadata.get("paid_data", {})
    values = paid.get("values", {}) if isinstance(paid, dict) else {}
    public = values.get("exchange_public_derivatives", {}) if isinstance(values, dict) else {}
    spread = _as_float(public.get("spread_pct")) if isinstance(public, dict) else None
    if spread is not None:
        return max(0.0, spread)
    if report.quote_volume_24h >= 250_000_000:
        return 0.02
    if report.quote_volume_24h >= 80_000_000:
        return 0.04
    if report.quote_volume_24h >= 20_000_000:
        return 0.08
    return 0.15


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
