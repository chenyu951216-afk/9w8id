from __future__ import annotations

import math
from typing import Any

from .direction_analyzer import analyze_direction
from .expected_value import estimate_expected_value
from .instrument_classifier import participation_profile, volatility_profile
from .layered_analysis import build_layered_analysis, build_layered_market_summary
from .models import DirectionScore, SymbolReport
from .quant_scorecard import build_quant_scorecard
from .regime_detector import detect_market_regime, regime_alignment_score
from .relative_strength import attach_relative_strength, relative_strength_score
from .risk.execution_gate import evaluate_execution_gate, selected_side, side_score


LIFECYCLE_ORDER = {
    "EXECUTABLE": 10,
    "MANAGE": 9,
    "ACTIVE": 8,
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
        signal_state = report.metadata.get("signal_state")
        if isinstance(signal_state, dict):
            context["state"] = _state_with_signal_persistence(
                context["state"],
                signal_state,
                context["layered_analysis"],
            )
        report.metadata["market_regime"] = regime
        report.metadata["direction_analysis"] = context["direction_analysis"]
        report.metadata["entry_proximity"] = context["entry_proximity"]
        report.metadata["expected_value"] = context["expected_value"]
        report.metadata["layered_analysis"] = context["layered_analysis"]
        report.metadata["opportunity"] = context
        report.metadata["opportunity_score"] = context["opportunity_score"]
        report.metadata["execution_quality"] = context["execution_quality"]
        report.metadata["candidate_grade"] = context["grade"]
        report.metadata["candidate_status"] = context["state"]
        if isinstance(signal_state, dict):
            signal_state["opportunity_grade"] = context["grade"]
            signal_state["opportunity_status"] = context["state"]
            signal_state["lifecycle_state"] = context["state"]
            signal_state["opportunity_score"] = context["opportunity_score"]
            signal_state["direction_analysis"] = context["direction_analysis"]
            signal_state["next_trigger"] = context["next_trigger"]
            signal_state["trade_thesis"] = context["thesis"]
            signal_state["blockers"] = context["blockers"]
            signal_state["strategy_profile"] = context["strategy_profile"]
            signal_state["risk_notes"] = context["risk_notes"]
            signal_state["failure_conditions"] = context["failure_conditions"]
            signal_state["trade_signal_state"] = context["layered_analysis"]["signal_state"]
            signal_state["layered_trade_plan"] = context["layered_analysis"]["trade_plan"]
        rows.append({"symbol": report.symbol, **context})
    rows.sort(key=lambda item: (item["opportunity_score"], item["execution_quality"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        for report in reports:
            if report.symbol == row["symbol"]:
                report.metadata["opportunity_rank"] = rank
                report.metadata["opportunity"]["rank"] = rank
                break
    layered_summary = build_layered_market_summary(rows)
    execution_diagnostics = _execution_diagnostics(reports, rows)
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
        **layered_summary,
        "execution_diagnostics": execution_diagnostics,
        "no_trade_diagnostics": {
            **layered_summary.get("no_trade_diagnostics", {}),
            **execution_diagnostics,
        },
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
    for _ in range(2):
        context = _build_report_context_once(report, regime)
        prediction = context["layered_analysis"]["prediction"]
        predicted_direction = prediction.get("prediction_direction")
        if predicted_direction not in {"long", "short", "neutral"}:
            return context
        if predicted_direction == report.selected_direction:
            _sync_prediction_scores(report, prediction)
            return context
        old_direction = report.selected_direction
        _apply_prediction_direction(report, str(predicted_direction), prediction)
        report.metadata["layered_direction_override"] = {
            "from": old_direction,
            "to": predicted_direction,
            "reason": "prediction_layer_drives_selected_direction",
        }
    return context


def _build_report_context_once(report: SymbolReport, regime: dict[str, Any]) -> dict[str, Any]:
    direction_analysis = analyze_direction(report, regime)
    chosen_direction = direction_analysis["chosen_direction"]
    if (
        chosen_direction in {"long", "short"}
        and chosen_direction != report.selected_direction
        and direction_analysis.get("conflict_level") != "high"
        and float(direction_analysis.get("direction_edge") or 0.0) >= 6.0
    ):
        override = {
            "from": report.selected_direction,
            "to": chosen_direction,
            "reason": "direction_analysis_side_sync",
        }
        report.metadata["direction_analysis_override"] = override
        report.metadata["layered_direction_override"] = override
        report.selected_direction = str(chosen_direction)
        report.score = round(side_score(selected_side(report)), 2)
        report.metadata["score_gap"] = max(
            float(report.metadata.get("score_gap", 0.0) or 0.0),
            float(direction_analysis.get("direction_edge") or 0.0),
        )
        report.metadata.pop("direction_conflict", None)
    side = selected_side(report)
    effective_direction = chosen_direction if chosen_direction in {"long", "short"} else side.direction
    scorecard = build_quant_scorecard(report)
    report.metadata["quant_scorecard"] = scorecard
    _attach_scorecard_metrics(report, scorecard)
    selected_scorecard = _scorecard_for_direction(scorecard, effective_direction, side.direction)
    proximity = entry_proximity(report, side)
    gate = evaluate_execution_gate(report)
    report.metadata["execution_gate"] = gate
    diagnostics = _diagnostics(report, side)
    setup_quality = float(side.setup_score if side.setup_score is not None else report.score)
    risk_quality = diagnostics["risk_reward_quality"]
    data_quality = float(side.data_completeness)
    relative_strength = relative_strength_score(report, effective_direction)
    regime_alignment = regime_alignment_score(regime, effective_direction, side.market_metrics)
    location_quality = proximity["score"]
    execution_quality = _execution_quality(side, gate, proximity, regime, diagnostics)
    expected_value = estimate_expected_value(report, side, direction_analysis, execution_quality)
    strategy_profile = _strategy_profile(
        report,
        side,
        effective_direction,
        regime,
        direction_analysis,
        proximity,
        diagnostics,
        relative_strength,
        regime_alignment,
        execution_quality,
        expected_value,
        selected_scorecard,
    )
    hard_vetoes, soft_penalties = _risk_penalties(report, side, gate, direction_analysis, proximity, expected_value)
    weights = regime.get("weight_profile", {})
    base_opportunity = (
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
    quant_composite = _as_float(selected_scorecard.get("composite_score"))
    if quant_composite is not None:
        base_opportunity = base_opportunity * 0.78 + quant_composite * 0.22
    opportunity = base_opportunity * 0.86 + float(strategy_profile["fit_score"]) * 0.14
    raw_opportunity = _clamp(opportunity)
    readiness_cap = _opportunity_readiness_cap(
        diagnostics,
        execution_quality,
        proximity,
        hard_vetoes,
        expected_value,
        gate,
    )
    opportunity = min(raw_opportunity, readiness_cap)
    state = _lifecycle_state(gate, opportunity, setup_quality, execution_quality, proximity, hard_vetoes, expected_value)
    blockers = _blockers(gate, hard_vetoes, soft_penalties, proximity, expected_value)
    risk_notes = _risk_notes(report, side, strategy_profile, gate, hard_vetoes, soft_penalties, proximity, expected_value)
    failure_conditions = _failure_conditions(strategy_profile, gate)
    thesis = _thesis(report, side, regime, direction_analysis, proximity, relative_strength, expected_value)
    layered = build_layered_analysis(
        report=report,
        regime=regime,
        direction_analysis=direction_analysis,
        gate=gate,
        proximity=proximity,
        opportunity={
            "strategy_profile": strategy_profile,
            "strategy_fit_score": strategy_profile["fit_score"],
            "strategy_label": strategy_profile["label"],
        },
    )
    state = _layered_lifecycle_state(layered, state)
    grade = _grade(opportunity, setup_quality, execution_quality, direction_analysis, expected_value, state, hard_vetoes)
    context = {
        "state": state,
        "grade": grade,
        "opportunity_score": round(opportunity, 2),
        "raw_opportunity_score": round(raw_opportunity, 2),
        "opportunity_readiness_cap": round(readiness_cap, 2),
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
        "strategy_profile": strategy_profile,
        "strategy_fit_score": strategy_profile["fit_score"],
        "strategy_label": strategy_profile["label"],
        "hard_vetoes": hard_vetoes,
        "soft_penalties": soft_penalties,
        "thesis": thesis,
        "blockers": blockers,
        "risk_notes": risk_notes,
        "failure_conditions": failure_conditions,
        "next_trigger": _next_trigger(state, proximity, direction_analysis, gate),
        "invalidation": gate.get("invalidation_conditions", []),
        "should_execute": bool(gate.get("should_execute")),
        "layered_analysis": layered,
        "quant_scorecard": scorecard,
    }
    return context


def _apply_prediction_direction(report: SymbolReport, direction: str, prediction: dict[str, Any]) -> None:
    report.selected_direction = direction
    long_score = float(prediction.get("prediction_score_long") or side_score(report.long))
    short_score = float(prediction.get("prediction_score_short") or side_score(report.short))
    if direction == "long":
        report.score = round(long_score, 2)
        report.metadata.pop("direction_conflict", None)
    elif direction == "short":
        report.score = round(short_score, 2)
        report.metadata.pop("direction_conflict", None)
    else:
        report.score = round(max(long_score, short_score), 2)
        if not report.metadata.get("direction_conflict"):
            report.metadata["direction_conflict"] = "prediction layer returned neutral; no clear directional edge"
    _sync_prediction_scores(report, prediction)


def _sync_prediction_scores(report: SymbolReport, prediction: dict[str, Any]) -> None:
    long_score = float(prediction.get("prediction_score_long") or side_score(report.long))
    short_score = float(prediction.get("prediction_score_short") or side_score(report.short))
    prediction_edge = float(prediction.get("prediction_edge") or abs(long_score - short_score))
    report.metadata["score_gap"] = round(max(float(report.metadata.get("score_gap", 0.0) or 0.0), prediction_edge), 2)
    report.metadata["prediction_score_long"] = round(long_score, 2)
    report.metadata["prediction_score_short"] = round(short_score, 2)


def _layered_lifecycle_state(layered: dict[str, Any], fallback: str) -> str:
    signal_state = str(layered.get("signal_state") or "")
    no_trade_type = str(layered.get("no_trade_type") or "")
    primary_blocker = str(layered.get("primary_blocker") or "")
    if signal_state in {"limit_executable", "market_executable"}:
        return "EXECUTABLE"
    if signal_state == "blocked":
        return "BLOCKED_GOOD_SETUP" if no_trade_type in {"Hard Block", "Direction Conflict"} else "INVALID"
    if signal_state == "neutral":
        return "INVALID"
    if signal_state == "setup_ready":
        return "ARMED"
    if signal_state == "watchlist":
        if no_trade_type == "Direction But No Entry" and primary_blocker == "price_not_in_entry_zone":
            return "MISSED" if fallback == "MISSED" else "WATCH"
        return "WATCH"
    if signal_state == "bias_only":
        return "SCOUT"
    return fallback


def _state_with_signal_persistence(state: str, signal_state: dict[str, Any], layered: dict[str, Any]) -> str:
    if state in {"EXECUTABLE", "INVALID", "EXPIRED", "MISSED"}:
        return state
    execution_state = str(signal_state.get("execution_state") or "")
    if execution_state not in {"ACTIVE", "MANAGE"}:
        return state
    no_trade_type = str(layered.get("no_trade_type") or "")
    if no_trade_type in {"Hard Block", "Direction Conflict"}:
        return state
    if no_trade_type == "Direction But Bad RR":
        return state
    return execution_state


def _opportunity_readiness_cap(
    diagnostics: dict[str, float],
    execution_quality: float,
    proximity: dict[str, Any],
    hard_vetoes: list[str],
    expected_value: dict[str, Any],
    gate: dict[str, Any],
) -> float:
    cap = 100.0
    gate_execution = _as_float(gate.get("execution_quality"))
    if gate_execution is not None and gate_execution < 63.0:
        cap = min(cap, 70.0 if gate_execution < 40.0 else 80.0)
    if hard_vetoes:
        cap = min(cap, 72.0)
    if diagnostics["ltf_trigger"] < 65.0:
        cap = min(cap, 78.0)
    if diagnostics["entry_quality"] < 65.0:
        cap = min(cap, 78.0)
    if diagnostics["risk_reward_quality"] < 60.0:
        cap = min(cap, 70.0)
    if diagnostics["market_quality"] < 45.0:
        cap = min(cap, 76.0)
    if execution_quality < 40.0:
        cap = min(cap, 70.0)
    elif execution_quality < 55.0:
        cap = min(cap, 80.0)
    state = proximity.get("state")
    if state == "far_from_entry":
        cap = min(cap, 70.0)
    elif state == "missed":
        cap = min(cap, 62.0)
    expected_r = float(expected_value.get("expected_R") or 0.0)
    if expected_r <= 0.0:
        cap = min(cap, 64.0)
    return _clamp(cap)


def _strategy_profile(
    report: SymbolReport,
    side: DirectionScore,
    direction: str,
    regime: dict[str, Any],
    direction_analysis: dict[str, Any],
    proximity: dict[str, Any],
    diagnostics: dict[str, float],
    relative_strength: float,
    regime_alignment: float,
    execution_quality: float,
    expected_value: dict[str, Any],
    scorecard: dict[str, Any],
) -> dict[str, Any]:
    htf = diagnostics["htf_context"]
    ltf = diagnostics["ltf_trigger"]
    entry = diagnostics["entry_quality"]
    risk = diagnostics["risk_reward_quality"]
    market = diagnostics["market_quality"]
    conviction = float(direction_analysis.get("direction_conviction") or 0.0)
    expected_r = float(expected_value.get("expected_R") or 0.0)
    location = _profile_location_score(proximity)
    stability = _profile_market_stability(report, side, regime)
    active_volatility = _profile_active_volatility(report.symbol, side)
    heat_trend = _as_float(diagnostics.get("derivatives_trend_confirmation_score")) or 50.0
    crowding = _as_float(diagnostics.get("derivatives_crowding_score")) or 0.0
    exhaustion = _as_float(diagnostics.get("derivatives_exhaustion_score")) or 0.0
    heat_quality = _clamp(heat_trend - max(0.0, crowding - 55.0) * 0.45 - max(0.0, exhaustion - 55.0) * 0.55)
    derivatives_score = _as_float(scorecard.get("derivatives_score")) or heat_quality
    no_chase_score = _as_float(scorecard.get("no_chase_score")) or 70.0
    quant_composite = _as_float(scorecard.get("composite_score")) or 50.0
    squeeze_fuel = _as_float(scorecard.get("squeeze_fuel_score")) or 0.0
    strategy_hint = str(scorecard.get("strategy_hint") or "")
    mover_profile = str(scorecard.get("mover_profile") or (side.market_metrics or {}).get("mover_profile") or "normal")
    mover_active = mover_profile in {"active_mover", "hot_mover", "extreme_mover"}
    mover_gate = 78.0 if mover_active else 34.0

    sweep = _feature_pct(side, "liquidity_sweep")
    key_level = _feature_pct(side, "key_level")
    price_action = _feature_pct(side, "price_action")
    breakout = _feature_pct(side, "breakout_quality")
    fvg = _feature_pct(side, "fvg")
    ote = _feature_pct(side, "ote")

    profiles = [
        _make_strategy_profile(
            code="hot_mover_structural_retest",
            label="高波動結構回踩",
            fit=(
                htf * 0.13
                + ltf * 0.12
                + entry * 0.18
                + risk * 0.13
                + derivatives_score * 0.16
                + no_chase_score * 0.11
                + active_volatility * 0.07
                + location * 0.04
                + mover_gate * 0.06
                + (4.0 if strategy_hint == "hot_mover_structural_retest" else 0.0)
            ),
            expected_behavior="High-vol alt is only tradable after a structural retest or clean pullback into the planned band; no same-side late chase.",
            reasons=[
                f"mover={mover_profile}, vol_active={active_volatility:.1f}, no_chase={no_chase_score:.1f}",
                f"entry={entry:.1f}, risk={risk:.1f}, derivatives={derivatives_score:.1f}",
                f"quant composite={quant_composite:.1f}, hint={strategy_hint or '-'}",
            ],
            risk_notes=[
                "Use the wider alt entry band as a limit zone, but calculate RR from the least favorable fill inside that zone.",
                "If price leaves the band without retest acceptance, downgrade to watch instead of chasing the current candle.",
            ],
            failure_conditions=[
                "Retest candle fails to hold the entry band or closes back through the structural level.",
                "Derivatives/orderflow flips against the selected side while OI expands.",
            ],
        ),
        _make_strategy_profile(
            code="derivatives_squeeze_reversal",
            label="衍生品擠壓反轉",
            fit=(
                htf * 0.14
                + sweep * 0.15
                + entry * 0.14
                + risk * 0.12
                + derivatives_score * 0.15
                + min(100.0, squeeze_fuel * 9.0 + 42.0) * 0.10
                + no_chase_score * 0.08
                + market * 0.06
                + conviction * 0.06
                + (4.0 if strategy_hint == "derivatives_squeeze_reversal" else 0.0)
            ),
            expected_behavior="Crowded opposite positioning can fuel the trade, but entry still needs sweep/reclaim or structural acceptance.",
            reasons=[
                f"sweep={sweep:.1f}, derivatives={derivatives_score:.1f}, squeeze_fuel={squeeze_fuel:.1f}",
                f"HTF={htf:.1f}, entry={entry:.1f}, RR quality={risk:.1f}",
                f"crowding={crowding:.1f}, exhaustion={exhaustion:.1f}",
            ],
            risk_notes=[
                "Extreme funding/OI is treated as fuel only after price confirms; otherwise it remains a crowding warning.",
                "Stops must sit beyond the structural sweep/OB, not at a tiny minor wick.",
            ],
            failure_conditions=[
                "Sweep fails to reclaim, or new OI expansion follows price against the planned side.",
                "TP1 is no longer at least net positive after fee/slippage and wider stop calculation.",
            ],
        ),
        _make_strategy_profile(
            code="sweep_reversal",
            label="掃流動性反轉",
            fit=(
                htf * 0.24
                + entry * 0.20
                + risk * 0.14
                + sweep * 0.15
                + key_level * 0.08
                + location * 0.08
                + stability * 0.07
                + conviction * 0.04
            ),
            expected_behavior="先掃前高/前低並收回，回補 FVG/OTE/OB 後往下一個流動性目標推進。",
            reasons=[
                f"HTF {htf:.1f} / sweep {sweep:.1f} / entry {entry:.1f}",
                f"位置分 {location:.1f}，市場穩定度 {stability:.1f}",
                f"RR 結構 {risk:.1f}，Expected R {expected_r:.2f}",
            ],
            risk_notes=[
                "這類打法怕二次掃損，必須等價格回到 entry band 或明確收回後再執行。",
                "若 BTC 或大盤方向突然反向，反轉劇本容易變成延續突破的燃料。",
            ],
            failure_conditions=[
                "掃流動性後沒有收回關鍵位，或收盤重新跌回/站回掃蕩方向。",
                "entry zone 被連續收破，且 5m/15m 出現反向 MSS/BOS。",
            ],
        ),
        _make_strategy_profile(
            code="trend_continuation",
            label="趨勢延續突破",
            fit=(
                ltf * 0.22
                + breakout * 0.16
                + price_action * 0.13
                + regime_alignment * 0.14
                + relative_strength * 0.11
                + active_volatility * 0.09
                + heat_quality * 0.08
                + risk * 0.07
            ),
            expected_behavior="突破後不深回，回踩關鍵位或 FVG 仍守住，順著相對強弱與大盤 regime 延續。",
            reasons=[
                f"LTF {ltf:.1f} / breakout {breakout:.1f} / price action {price_action:.1f}",
                f"regime alignment {regime_alignment:.1f}，relative strength {relative_strength:.1f}",
                f"波動活躍度 {active_volatility:.1f}，衍生品 trend heat {heat_quality:.1f}",
            ],
            risk_notes=[
                "延續突破最怕追在放量末端，距 entry 過遠時只保留觀察。",
                "若量能暴衝但衍生品 crowding/exhaustion 升高，容易假突破後回吐。",
            ],
            failure_conditions=[
                "突破 K 收回關鍵位內，或回踩後無法重新站上/跌破突破位。",
                "相對強弱快速消失，並且 BTC/市場 regime 轉向反邊。",
            ],
        ),
        _make_strategy_profile(
            code="pullback_continuation",
            label="回撤承接延續",
            fit=(
                htf * 0.18
                + ltf * 0.14
                + entry * 0.22
                + risk * 0.17
                + fvg * 0.12
                + ote * 0.10
                + location * 0.07
            ),
            expected_behavior="主趨勢已給方向，等待回撤到 FVG/OTE/OB 重疊區承接，再往原方向續走。",
            reasons=[
                f"entry {entry:.1f} / FVG {fvg:.1f} / OTE {ote:.1f}",
                f"HTF {htf:.1f}，LTF {ltf:.1f}，RR {risk:.1f}",
                f"entry proximity {proximity.get('state')}",
            ],
            risk_notes=[
                "這類打法不是越快越好，價格沒回到計畫區就不追。",
                "若回撤過深並破壞 OTE/FVG，原本的承接劇本會失效。",
            ],
            failure_conditions=[
                "回撤跌破/突破 stop 或 OTE 甜蜜區外仍無收回。",
                "回補 FVG 後沒有 displacement，反而出現反向結構突破。",
            ],
        ),
        _make_strategy_profile(
            code="intraday_retest_scalp",
            label="日內回測短打",
            fit=(
                ltf * 0.22
                + entry * 0.16
                + risk * 0.16
                + execution_quality * 0.13
                + location * 0.13
                + active_volatility * 0.08
                + heat_quality * 0.07
                + conviction * 0.05
            ),
            expected_behavior="短線觸發已出現，價格貼近 entry 後用較快的 TP1/TP2 管理，失敗就快速退出。",
            reasons=[
                f"LTF {ltf:.1f}，execution {execution_quality:.1f}，location {location:.1f}",
                f"RR {risk:.1f}，active volatility {active_volatility:.1f}",
                f"direction conviction {conviction:.1f}",
            ],
            risk_notes=[
                "短打容錯低，滑價、手續費與交易所深度會明顯影響 Expected R。",
                "若 5m close 沒有延續，不能把 scalp 硬拿成波段單。",
            ],
            failure_conditions=[
                "觸發後 1-3 根 5m K 沒有往 TP1 推進，或直接回到 entry 下/上方。",
                "RR 降到日內門檻以下，或 spread/滑價突然放大。",
            ],
        ),
    ]
    profiles.sort(key=lambda item: item["fit_score"], reverse=True)
    best = dict(profiles[0])
    best["all_profiles"] = [
        {"code": item["code"], "label": item["label"], "fit_score": item["fit_score"]}
        for item in profiles
    ]
    if best["fit_score"] < 52.0:
        best = {
            **best,
            "code": "watch_no_clear_profile",
            "label": "觀察：打法尚未對齊",
            "expected_behavior": "目前分數可能來自零散共振，尚未形成單一清楚行情劇本。",
            "risk_notes": [
                "不是變嚴格，而是目前看不出價格更像反轉、延續、回撤承接或短打哪一種。",
                "等待新的 sweep、BOS/MSS、FVG 回補或衍生品方向確認後再提高排名。",
            ],
            "failure_conditions": [
                "多空分差持續縮小，或最高分 profile 低於 52。",
                "下一輪掃描仍無 entry / stop / TP 完整計畫。",
            ],
            "reasons": [
                f"最佳 profile 只有 {profiles[0]['label']} {profiles[0]['fit_score']:.1f}",
                f"HTF {htf:.1f} / LTF {ltf:.1f} / entry {entry:.1f} / risk {risk:.1f}",
            ],
            "all_profiles": best["all_profiles"],
        }
    return best


def _make_strategy_profile(
    *,
    code: str,
    label: str,
    fit: float,
    expected_behavior: str,
    reasons: list[str],
    risk_notes: list[str],
    failure_conditions: list[str],
) -> dict[str, Any]:
    fit_score = round(_clamp(fit), 2)
    if fit_score >= 78:
        confidence = "high"
    elif fit_score >= 64:
        confidence = "medium"
    elif fit_score >= 52:
        confidence = "low"
    else:
        confidence = "watch"
    return {
        "code": code,
        "label": label,
        "fit_score": fit_score,
        "confidence": confidence,
        "expected_behavior": expected_behavior,
        "reasons": reasons,
        "risk_notes": risk_notes,
        "failure_conditions": failure_conditions,
    }


def _feature_pct(side: DirectionScore, name: str) -> float:
    maximum = float(side.feature_max_scores.get(name, 0.0) or 0.0)
    if maximum <= 0:
        return 0.0
    return _clamp(float(side.feature_scores.get(name, 0.0) or 0.0) / maximum * 100.0)


def _profile_location_score(proximity: dict[str, Any]) -> float:
    state = str(proximity.get("state") or "")
    if state == "near_entry":
        return 100.0
    if state == "approaching_entry":
        return 76.0
    if state == "far_from_entry":
        return 42.0
    if state == "missed":
        return 18.0
    if state == "no_entry_zone":
        return 8.0
    return float(proximity.get("score") or 50.0)


def _profile_market_stability(report: SymbolReport, side: DirectionScore, regime: dict[str, Any]) -> float:
    metrics = side.market_metrics or {}
    score = 58.0
    atr_pct = _as_float(metrics.get("atr_pct"))
    volume_ratio = _as_float(metrics.get("volume_ratio"))
    if atr_pct is not None:
        vol_profile = volatility_profile(report.symbol)
        if vol_profile.active_low_atr_pct <= atr_pct <= vol_profile.active_high_atr_pct:
            score += 14.0
        elif vol_profile.active_high_atr_pct < atr_pct <= vol_profile.hot_atr_pct:
            score += 6.0
        elif atr_pct >= vol_profile.extreme_atr_pct:
            score -= 24.0
        elif atr_pct > vol_profile.hot_atr_pct:
            score -= 12.0
        else:
            score -= 10.0
    if volume_ratio is not None:
        flow_profile = participation_profile(report.symbol)
        if flow_profile.active_low_volume_ratio <= volume_ratio <= flow_profile.active_high_volume_ratio:
            score += 10.0
        elif volume_ratio > flow_profile.extreme_volume_ratio:
            score -= 18.0
        elif volume_ratio > flow_profile.hot_volume_ratio:
            score -= 10.0
    if metrics.get("btc_against"):
        score -= 16.0
    if metrics.get("btc_overheated"):
        score -= 8.0
    if report.quote_volume_24h >= 100_000_000:
        score += 8.0
    elif report.quote_volume_24h < 20_000_000:
        score -= 14.0
    liquidity = _as_float(regime.get("liquidity_condition"))
    if liquidity is not None:
        score += (liquidity - 50.0) * 0.12
    return _clamp(score)


def _profile_active_volatility(symbol: str, side: DirectionScore) -> float:
    atr_pct = _as_float(side.market_metrics.get("atr_pct"))
    volume_ratio = _as_float(side.market_metrics.get("volume_ratio"))
    score = 48.0
    if atr_pct is not None:
        vol_profile = volatility_profile(symbol)
        trend_low = max(vol_profile.active_low_atr_pct, vol_profile.active_high_atr_pct * 0.42)
        if trend_low <= atr_pct <= vol_profile.active_high_atr_pct:
            score += 24.0
        elif vol_profile.active_high_atr_pct < atr_pct <= vol_profile.hot_atr_pct:
            score += 12.0
        elif atr_pct >= vol_profile.extreme_atr_pct:
            score -= 16.0
        elif atr_pct < vol_profile.quiet_atr_pct:
            score -= 12.0
    if volume_ratio is not None:
        flow_profile = participation_profile(symbol)
        if flow_profile.active_low_volume_ratio <= volume_ratio <= flow_profile.active_high_volume_ratio:
            score += 18.0
        elif flow_profile.active_high_volume_ratio < volume_ratio <= flow_profile.warm_high_volume_ratio:
            score += 6.0
        elif volume_ratio > flow_profile.extreme_volume_ratio:
            score -= 18.0
    return _clamp(score)


def _risk_notes(
    report: SymbolReport,
    side: DirectionScore,
    strategy_profile: dict[str, Any],
    gate: dict[str, Any],
    hard_vetoes: list[str],
    soft_penalties: dict[str, float],
    proximity: dict[str, Any],
    expected_value: dict[str, Any],
) -> list[str]:
    notes: list[str] = list(strategy_profile.get("risk_notes", []))
    for item in gate.get("warnings", [])[:3]:
        notes.append(str(item))
    for item in hard_vetoes[:3]:
        notes.append(str(item))
    if proximity.get("state") not in {"near_entry", "no_entry_zone"}:
        notes.append(f"目前 entry proximity={proximity.get('state')}，距離動態 band {proximity.get('distance_in_bands')} 倍。")
    expected_r = float(expected_value.get("expected_R") or 0.0)
    if expected_r <= 0:
        notes.append(f"Expected R 目前為 {expected_r:.2f}，代表成本/勝率估算尚未給正期望。")
    if side.rr is not None and side.rr < 1.65:
        notes.append(f"RR={side.rr:.2f}R 偏低，需以短打或分批出場處理。")
    for name, value in soft_penalties.items():
        if value:
            notes.append(f"{name} soft penalty {value:.1f}，排名已做降權。")
    if report.metadata.get("direction_conflict"):
        notes.append(f"方向衝突：{report.metadata['direction_conflict']}")
    return _dedupe_text(notes, 7)


def _failure_conditions(strategy_profile: dict[str, Any], gate: dict[str, Any]) -> list[str]:
    conditions: list[str] = list(strategy_profile.get("failure_conditions", []))
    conditions.extend(str(item) for item in gate.get("invalidation_conditions", [])[:5])
    return _dedupe_text(conditions, 8)


def _dedupe_text(values: list[str], limit: int) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in output:
            output.append(text)
        if len(output) >= limit:
            break
    return output


def entry_proximity(report: SymbolReport, side: DirectionScore) -> dict[str, Any]:
    distance = _entry_distance_pct(report.price, side.entry_zone)
    atr_pct = _as_float(side.market_metrics.get("atr_pct")) or 0.0
    spread_pct = _spread_pct(report)
    vol_profile = volatility_profile(report.symbol)
    dynamic_band = max(0.30, atr_pct * vol_profile.entry_band_atr_mult, spread_pct * 3.0)
    adaptive_band = _as_float(side.market_metrics.get("adaptive_entry_band_pct"))
    if adaptive_band is not None:
        dynamic_band = max(dynamic_band, adaptive_band)
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
    paid_status = gate.get("paid_data_status", {})
    if paid_status.get("blocked", False):
        base -= 30.0
    if not paid_status.get("configured_api_ready", True):
        base -= 18.0
    context_score = _as_float(paid_status.get("context_score"))
    if context_score is not None:
        base += (context_score - 70.0) * 0.08
    heat_profile = paid_status.get("heat_profile", {}) if isinstance(paid_status.get("heat_profile"), dict) else {}
    heat_state = str(heat_profile.get("state") or paid_status.get("heat_state") or "")
    trend_heat = _as_float(heat_profile.get("trend_confirmation_score")) or _as_float(paid_status.get("trend_confirmation_score")) or 0.0
    crowding = _as_float(heat_profile.get("crowding_score")) or _as_float(paid_status.get("crowding_score")) or 0.0
    exhaustion = _as_float(heat_profile.get("exhaustion_score")) or _as_float(paid_status.get("exhaustion_score")) or 0.0
    if heat_state == "healthy_heat":
        base += 4.0
    elif trend_heat >= 45.0 and crowding < 60.0:
        base += 2.0
    if crowding >= 60.0:
        base -= min(12.0, (crowding - 55.0) * 0.35)
    if exhaustion >= 55.0:
        base -= min(14.0, (exhaustion - 50.0) * 0.40)
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


def _scorecard_for_direction(scorecard: dict[str, Any], direction: str, fallback_direction: str) -> dict[str, Any]:
    for key in (direction, fallback_direction, str(scorecard.get("selected_card") or "")):
        card = scorecard.get(key)
        if isinstance(card, dict):
            return card
    return {}


def _attach_scorecard_metrics(report: SymbolReport, scorecard: dict[str, Any]) -> None:
    for direction, side in (("long", report.long), ("short", report.short)):
        card = scorecard.get(direction)
        if not isinstance(card, dict):
            continue
        metrics = side.market_metrics
        metrics.setdefault("quant_scorecard_version", scorecard.get("version"))
        metrics["quant_composite_score"] = card.get("composite_score")
        metrics["quant_direction_score"] = card.get("direction_score")
        metrics["quant_direction_adjustment"] = card.get("direction_adjustment")
        metrics["quant_derivatives_score"] = card.get("derivatives_score")
        metrics["quant_entry_precision_score"] = card.get("entry_precision_score")
        metrics["quant_no_chase_score"] = card.get("no_chase_score")
        metrics["quant_crowding_risk"] = card.get("derivatives_crowding_risk")
        metrics["strategy_hint"] = card.get("strategy_hint")


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
    if isinstance(paid, dict):
        heat_profile = paid.get("heat_profile", {}) if isinstance(paid.get("heat_profile"), dict) else {}
        crowding = _as_float(heat_profile.get("crowding_score")) or _as_float(paid.get("crowding_score")) or 0.0
        exhaustion = _as_float(heat_profile.get("exhaustion_score")) or _as_float(paid.get("exhaustion_score")) or 0.0
        if crowding >= 70.0 or exhaustion >= 70.0:
            hard.append(f"Derivatives heat is crowded/exhausted: crowding {crowding:.0f}, exhaustion {exhaustion:.0f}.")
        elif crowding >= 55.0:
            crowding_penalty += min(8.0, (crowding - 50.0) * 0.35)
        elif exhaustion >= 55.0:
            crowding_penalty += min(8.0, (exhaustion - 50.0) * 0.35)
    if isinstance(paid, dict) and paid.get("blocked"):
        hard.append(f"Funding/OI 過熱：{paid.get('warning') or 'derivatives risk blocked'}")
    if isinstance(paid, dict) and not paid.get("configured_api_ready", True):
        missing = ", ".join(paid.get("configured_api_readiness", {}).get("execution_missing", []))
        hard.append(f"已設定 API 尚未完整讀取：{missing or 'configured API'}")
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
    if (
        opportunity >= (80.0 if state == "EXECUTABLE" else 82.0)
        and setup_quality >= 72
        and conviction >= 70
        and expected_r > 0
        and state in {"ARMED", "EXECUTABLE"}
    ):
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
    layered = row.get("layered_analysis", {}) if isinstance(row.get("layered_analysis"), dict) else {}
    prediction = layered.get("prediction", {}) if isinstance(layered.get("prediction"), dict) else {}
    setup = layered.get("setup", {}) if isinstance(layered.get("setup"), dict) else {}
    return {
        "rank": row.get("rank"),
        "symbol": row["symbol"],
        "state": row["state"],
        "grade": row["grade"],
        "signal_state": layered.get("signal_state"),
        "no_trade_type": layered.get("no_trade_type"),
        "prediction_direction": prediction.get("prediction_direction"),
        "prediction_confidence": prediction.get("prediction_confidence"),
        "setup_type": setup.get("setup_type"),
        "opportunity_score": row["opportunity_score"],
        "raw_opportunity_score": row.get("raw_opportunity_score"),
        "opportunity_readiness_cap": row.get("opportunity_readiness_cap"),
        "setup_score": row["setup_score"],
        "execution_quality": row["execution_quality"],
        "direction_conviction": row["direction_conviction"],
        "expected_R": row["expected_R"],
        "strategy_label": row.get("strategy_label"),
        "strategy_fit_score": row.get("strategy_fit_score"),
        "next_trigger": row["next_trigger"],
        "thesis": row["thesis"][:3],
        "risk_notes": row.get("risk_notes", [])[:3],
        "failure_conditions": row.get("failure_conditions", [])[:3],
        "blockers": row["blockers"][:3],
    }


def _execution_diagnostics(reports: list[SymbolReport], rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable_executable = 0
    armed = 0
    watch = 0
    blocked = 0
    raw_market = 0
    raw_limit = 0
    category_counts: dict[str, int] = {}
    near_miss: list[dict[str, Any]] = []
    rows_by_symbol = {row["symbol"]: row for row in rows}
    for report in reports:
        gate = report.metadata.get("execution_gate")
        if not isinstance(gate, dict):
            gate = evaluate_execution_gate(report)
        code = gate.get("code")
        status = str(gate.get("execution_status") or "")
        if code == "market" and gate.get("should_execute"):
            raw_market += 1
        if code == "limit" and gate.get("should_execute"):
            raw_limit += 1
        signal_state = report.metadata.get("signal_state", {})
        if isinstance(signal_state, dict) and signal_state.get("can_execute_now"):
            stable_executable += 1
        if status == "ARMED_WAIT_ENTRY" or report.metadata.get("candidate_status") == "ARMED":
            armed += 1
        elif status == "WATCH" or report.metadata.get("candidate_status") in {"WATCH", "SCOUT"}:
            watch += 1
        elif status in {"BLOCKED_RISK", "INVALID", "MISSED"}:
            blocked += 1
        for category, count in (gate.get("blocker_categories") or {}).items():
            category_counts[str(category)] = category_counts.get(str(category), 0) + int(count or 0)
        if not gate.get("should_execute"):
            side = selected_side(report)
            row = rows_by_symbol.get(report.symbol, {})
            near_miss.append(
                {
                    "symbol": report.symbol,
                    "direction": report.selected_direction,
                    "score": round(float(report.score or 0.0), 2),
                    "setup": round(float(side.setup_score or 0.0), 2),
                    "execution": round(float(side.execution_score or report.score or 0.0), 2),
                    "entry_distance": gate.get("entry_distance_pct"),
                    "rr": side.rr,
                    "primary_gap": gate.get("primary_failed_reason") or (gate.get("blockers") or [""])[0],
                    "execution_status": status,
                    "opportunity_score": row.get("opportunity_score"),
                }
            )
    near_miss.sort(
        key=lambda item: (
            float(item.get("opportunity_score") or 0.0),
            float(item.get("score") or 0.0),
            float(item.get("execution") or 0.0),
        ),
        reverse=True,
    )
    top_categories = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
    return {
        "total_symbols": len(reports),
        "raw_executable_market_count": raw_market,
        "raw_executable_limit_count": raw_limit,
        "stable_executable_count": stable_executable,
        "armed_count": armed,
        "watch_count": watch,
        "blocked_count": blocked,
        "top_blocker_categories": [{"category": key, "count": value} for key, value in top_categories[:8]],
        "near_miss_candidates": near_miss[:20],
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
