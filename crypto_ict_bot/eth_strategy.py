from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .models import DirectionScore, SymbolReport
from .quant_scorecard import build_quant_scorecard


ETH_STRATEGY_VERSION = "eth_dedicated_2026_06_v2"
ETH_STATE_PATH = Path("state/eth_plan_state.json")


def attach_eth_strategy(
    reports: list[SymbolReport],
    config: dict[str, Any],
    state_path: Path = ETH_STATE_PATH,
) -> dict[str, Any]:
    settings = _eth_settings(config)
    if not settings["enabled"]:
        return {"enabled": False, "version": ETH_STRATEGY_VERSION}

    state = _load_state(state_path)
    if settings["defer_active_plan_until_gate_submission"]:
        active = state.get("active_plan") if isinstance(state.get("active_plan"), dict) else {}
        if active and active.get("source") != "gate_submission":
            state["stale_strategy_plan"] = {**active, "cleared_reason": "live Gate mode waits for real submission/position before manage_existing"}
            state.pop("active_plan", None)
    now = datetime.now(timezone.utc)
    analyses: list[dict[str, Any]] = []
    for report in reports:
        if report.symbol.upper() != settings["symbol"]:
            continue
        analysis = build_eth_analysis(report, config, state, now=now)
        report.metadata["eth_analysis"] = analysis
        analyses.append(analysis)
        _update_state_from_analysis(state, analysis, now)

    _save_state(state_path, state)
    primary = analyses[0] if analyses else {}
    return {
        "enabled": True,
        "version": ETH_STRATEGY_VERSION,
        "symbol": settings["symbol"],
        "analysis": primary,
        "active_plan": state.get("active_plan", {}),
        "completed_plan": state.get("completed_plan", {}),
    }


def build_eth_analysis(
    report: SymbolReport,
    config: dict[str, Any],
    state: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    settings = _eth_settings(config)
    now = now or datetime.now(timezone.utc)
    state = state or {}
    scorecard = report.metadata.get("quant_scorecard")
    if not isinstance(scorecard, dict) or not scorecard:
        scorecard = build_quant_scorecard(report)
        report.metadata["quant_scorecard"] = scorecard

    direction_control = _eth_direction_control(report, scorecard, state)
    direction = str(direction_control["direction"])
    side = _side_for_direction(report, direction)
    opposite = report.short if direction == "long" else report.long
    side_card = scorecard.get(direction, {}) if direction in {"long", "short"} else {}
    opposite_card = scorecard.get("short" if direction == "long" else "long", {})
    session = _session_context(now)
    active_status = _active_plan_status(state.get("active_plan"), report, now, settings)
    data_stack = _data_stack(report)
    short_mode = _mode_plan(
        report,
        side,
        opposite,
        side_card,
        opposite_card,
        direction,
        "short_term",
        settings,
        session,
        active_status,
        data_stack,
    )
    swing_mode = _mode_plan(
        report,
        side,
        opposite,
        side_card,
        opposite_card,
        direction,
        "swing",
        settings,
        session,
        active_status,
        data_stack,
    )
    primary_mode = _primary_mode(short_mode, swing_mode)
    trader_mode = _trader_mode(primary_mode)
    lifecycle = _lifecycle_summary(active_status, trader_mode)
    psychology = _market_psychology(report, side_card, direction, session)
    discord_notify = bool(trader_mode.get("notify"))
    if active_status.get("status") in {"stop_hit", "target_hit", "expired", "opposite_signal"}:
        discord_notify = True

    return {
        "version": ETH_STRATEGY_VERSION,
        "symbol": report.symbol,
        "generated_at": now.isoformat(),
        "price": report.price,
        "direction": direction,
        "direction_label": _direction_label(direction),
        "direction_control": direction_control,
        "session": session,
        "primary_mode": trader_mode.get("source_mode", primary_mode.get("mode", "short_term")),
        "primary_plan_state": trader_mode.get("state", "observe"),
        "plan_lifecycle": lifecycle,
        "active_plan_status": active_status,
        "discord_notify": discord_notify,
        "modes": {
            "short_term": short_mode,
            "swing": swing_mode,
        },
        "trader_mode": trader_mode,
        "data_stack": data_stack,
        "market_psychology": psychology,
        "risk_design": {
            "rr_from": "long uses the upper edge of entry zone; short uses the lower edge, so wider limit ranges do not fake RR.",
            "stop_policy": "Stops are volatility- and structure-bounded; widening the stop also widens TP ladder.",
            "no_chase_policy": "Trend continuation is allowed only when breakout/session/flow conditions support it; otherwise wait for retest.",
        },
        "execution_control": {
            "defer_active_plan_until_gate_submission": settings["defer_active_plan_until_gate_submission"],
            "active_plan_owner": "Gate state after live submission/position" if settings["defer_active_plan_until_gate_submission"] else "strategy scanner",
        },
    }


def _eth_settings(config: dict[str, Any]) -> dict[str, Any]:
    eth = config.get("eth", {}) if isinstance(config.get("eth"), dict) else {}
    return {
        "enabled": bool(eth.get("enabled", True)),
        "symbol": str(eth.get("symbol") or "ETHUSDT").upper().replace("/", ""),
        "short_mode_enabled": bool(eth.get("short_mode_enabled", True)),
        "swing_mode_enabled": bool(eth.get("swing_mode_enabled", True)),
        "short_min_score": float(eth.get("short_min_score") or 74.0),
        "swing_min_score": float(eth.get("swing_min_score") or 70.0),
        "active_plan_ttl_minutes": int(eth.get("active_plan_ttl_minutes") or 720),
        "swing_plan_ttl_minutes": int(eth.get("swing_plan_ttl_minutes") or 4320),
        "allow_trend_chase_only_on_breakout": bool(eth.get("allow_trend_chase_only_on_breakout", True)),
        "defer_active_plan_until_gate_submission": _defer_active_plan_until_gate_submission(config),
    }


def _defer_active_plan_until_gate_submission(config: dict[str, Any]) -> bool:
    gate = config.get("gate_trading", {}) if isinstance(config.get("gate_trading"), dict) else {}
    return bool(gate.get("enabled")) and not bool(gate.get("dry_run", True))


def _eth_direction_control(report: SymbolReport, scorecard: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    long_score = _as_float(scorecard.get("long", {}).get("direction_score")) or _side_score(report.long)
    short_score = _as_float(scorecard.get("short", {}).get("direction_score")) or _side_score(report.short)
    edge = abs(long_score - short_score)
    if report.selected_direction in {"long", "short"}:
        raw_direction = report.selected_direction
    elif edge < 6.0:
        raw_direction = "neutral"
    else:
        raw_direction = "long" if long_score > short_score else "short"

    selected = raw_direction
    reason = "raw direction accepted"
    lock = state.get("direction_lock") if isinstance(state.get("direction_lock"), dict) else {}
    locked_direction = str(lock.get("direction") or "")
    opposite_confirm_count = int(lock.get("opposite_confirm_count") or 0)
    switched = False
    if locked_direction in {"long", "short"} and raw_direction != locked_direction:
        locked_score = long_score if locked_direction == "long" else short_score
        raw_score = long_score if raw_direction == "long" else short_score if raw_direction == "short" else 0.0
        if raw_direction not in {"long", "short"} and locked_score >= 55.0:
            selected = locked_direction
            reason = "direction lock held because opposite edge is still neutral"
        elif raw_direction in {"long", "short"} and (edge < 14.0 or raw_score < 78.0 or opposite_confirm_count < 1):
            selected = locked_direction
            reason = "direction lock held; opposite signal needs stronger edge and one more scan"
        else:
            switched = True
            reason = "direction lock switched after strong opposite confirmation"

    return {
        "direction": selected,
        "raw_direction": raw_direction,
        "locked_direction": locked_direction or None,
        "switched": switched,
        "long_direction_score": round(float(long_score), 2),
        "short_direction_score": round(float(short_score), 2),
        "direction_edge": round(float(edge), 2),
        "opposite_confirm_count": opposite_confirm_count,
        "switch_requires": {
            "opposite_edge_at_least": 14.0,
            "opposite_direction_score_at_least": 78.0,
            "consecutive_opposite_scans": 2,
        },
        "reason": reason,
    }


def _select_eth_direction(report: SymbolReport, scorecard: dict[str, Any]) -> str:
    return str(_eth_direction_control(report, scorecard, {})["direction"])


def _side_for_direction(report: SymbolReport, direction: str) -> DirectionScore:
    if direction == "short":
        return report.short
    return report.long


def _mode_plan(
    report: SymbolReport,
    side: DirectionScore,
    opposite: DirectionScore,
    card: dict[str, Any],
    opposite_card: dict[str, Any],
    direction: str,
    mode: str,
    settings: dict[str, Any],
    session: dict[str, Any],
    active_status: dict[str, Any],
    data_stack: dict[str, Any],
) -> dict[str, Any]:
    if direction not in {"long", "short"}:
        return _empty_mode(mode, "wait_direction", "ETH 多空分數沒有拉開，先不建立交易計畫。")
    if mode == "short_term" and not settings["short_mode_enabled"]:
        return _empty_mode(mode, "disabled", "短線模式已停用。")
    if mode == "swing" and not settings["swing_mode_enabled"]:
        return _empty_mode(mode, "disabled", "中長線模式已停用。")

    volatility = _volatility_context(report, side)
    plan = _build_plan(report, side, direction, mode, volatility)
    decision_model = _mode_decision_model(report, side, opposite, card, opposite_card, mode, session, plan, data_stack)
    quality = float(decision_model["quality_score"])
    min_score = settings["short_min_score"] if mode == "short_term" else settings["swing_min_score"]
    no_chase = _as_float(card.get("no_chase_score")) or 50.0
    entry_distance = _entry_distance_pct(report.price, plan["entry_zone"])
    chase = _chase_context(report, side, card, direction, session, entry_distance)
    rr_floor = 1.25 if mode == "short_term" else 1.70
    hard_blockers: list[str] = list(decision_model.get("hard_blockers") or [])
    soft_notes: list[str] = list(decision_model.get("soft_notes") or [])
    quality_ready = quality >= min_score

    if plan["rr"] < rr_floor:
        hard_blockers.append(f"RR {plan['rr']:.2f}R below ETH floor {rr_floor:.2f}R; rebuild the plan before execution.")
    if quality < min_score - 12.0:
        hard_blockers.append(f"{mode} unified score {quality:.1f} is far below execution threshold {min_score:.1f}.")
    elif not quality_ready:
        soft_notes.append(f"{mode} unified score {quality:.1f} is below trade threshold {min_score:.1f}; keep watching, do not force a trade.")
    if mode == "short_term" and chase["is_chasing"] and not chase["continuation_allowed"]:
        hard_blockers.append("No-chase rule: price is extended and this is not a confirmed breakout/retest continuation.")
    if no_chase < 38 and not chase["continuation_allowed"]:
        soft_notes.append("No-chase score is weak; wait for a retest, sweep, or clearer liquidity confirmation.")
    if side.entry_zone is None:
        soft_notes.append("Base model had no complete entry zone; ETH module rebuilt a volatility/structure entry zone.")
    if side.stop is None or not side.take_profits:
        soft_notes.append("Base trade plan was incomplete; ETH module recalculated stop, TP ladder, and RR.")
    if not data_stack.get("has_coinglass"):
        soft_notes.append("CoinGlass paid data was not readable in this scan; it reduces confidence but does not create a hard block by itself.")
    elif not data_stack.get("coinglass_actionable"):
        soft_notes.append("CoinGlass connected, but this scan had limited actionable flow/liquidity fields; rely more on structure and public derivatives.")

    state_name = "observe"
    if active_status.get("status") == "active" and active_status.get("direction") == direction:
        state_name = "manage_existing"
        hard_blockers.append("同方向 ETH 計畫仍在生命週期內，先管理原計畫，不重複開新單。")
    elif active_status.get("status") in {"stop_hit", "target_hit", "expired", "opposite_signal"}:
        state_name = "plan_completed"
    elif hard_blockers:
        state_name = "wait_retest" if chase["is_chasing"] else "observe"
    elif mode == "short_term" and quality_ready and quality >= min_score + 4 and entry_distance <= plan["execution_band_pct"]:
        state_name = "execute_ready"
    elif quality_ready and entry_distance <= plan["execution_band_pct"] * (1.8 if mode == "short_term" else 2.5):
        state_name = "armed_wait_entry"
    else:
        state_name = "wait_retest"

    should_open_new = state_name == "execute_ready"
    notify = state_name in {"execute_ready", "plan_completed"}
    if mode == "swing" and state_name in {"armed_wait_entry", "wait_retest"} and quality >= min_score + 6:
        notify = True

    return {
        "mode": mode,
        "label": "短線執行" if mode == "short_term" else "中長線波段",
        "state": state_name,
        "quality_score": round(quality, 2),
        "min_score": min_score,
        "direction": direction,
        "direction_label": _direction_label(direction),
        "should_open_new": should_open_new,
        "notify": notify,
        "entry_zone": plan["entry_zone"],
        "entry_basis_for_rr": plan["entry_basis_for_rr"],
        "entry_distance_pct": round(entry_distance, 4),
        "execution_band_pct": plan["execution_band_pct"],
        "stop": plan["stop"],
        "take_profits": plan["take_profits"],
        "rr": plan["rr"],
        "plan_id": _plan_id(report.symbol, mode, direction, plan),
        "volatility": volatility,
        "chase_control": chase,
        "component_scores": decision_model["component_scores"],
        "unified_checks": decision_model["checks"],
        "data_policy": decision_model["data_policy"],
        "blocking_policy": {
            "hard_blocks_only_for": [
                "invalid direction",
                "incomplete ETH plan",
                "RR below mode floor",
                "score far below threshold",
                "late chase without breakout/retest confirmation",
                "duplicate active same-direction plan",
            ],
            "score_below_threshold": "watch state, not forced execution",
            "missing_optional_paid_api": "confidence note only",
        },
        "hard_blockers": hard_blockers,
        "soft_notes": soft_notes,
        "decision": _decision_text(state_name, mode, direction, hard_blockers, soft_notes),
    }


def _build_plan(
    report: SymbolReport,
    side: DirectionScore,
    direction: str,
    mode: str,
    volatility: dict[str, Any],
) -> dict[str, Any]:
    price = max(float(report.price), 1e-12)
    atr_pct = float(volatility["atr_pct"])
    if mode == "short_term":
        band_pct = max(0.18, min(0.72, atr_pct * 0.36))
        stop_pct = max(0.42, min(1.45, atr_pct * 1.08))
        tp_multiples = (1.15, 2.15, 3.30)
        execution_band_pct = max(0.16, min(0.60, atr_pct * 0.34))
    else:
        band_pct = max(0.55, min(1.85, atr_pct * 0.95))
        stop_pct = max(1.15, min(4.25, atr_pct * 2.25))
        tp_multiples = (1.45, 2.80, 4.50)
        execution_band_pct = max(0.45, min(1.35, atr_pct * 0.70))

    entry_zone = _widen_entry_zone(price, side.entry_zone, band_pct)
    zone_low, zone_high = entry_zone
    if direction == "long":
        entry_basis = zone_high
        structural_stop = side.stop if side.stop and side.stop < entry_basis else None
        fallback_stop = zone_low - price * stop_pct / 100.0
        stop = _bounded_stop(entry_basis, fallback_stop, structural_stop, "long")
        risk = max(entry_basis - stop, price * 0.001)
        take_profits = [
            {"name": "TP1", "price": entry_basis + risk * tp_multiples[0], "portion_pct": 30, "rr": tp_multiples[0], "note": "先降槓桿風險"},
            {"name": "TP2", "price": entry_basis + risk * tp_multiples[1], "portion_pct": 40, "rr": tp_multiples[1], "note": "主要獲利區"},
            {"name": "TP3", "price": entry_basis + risk * tp_multiples[2], "portion_pct": 30, "rr": tp_multiples[2], "note": "趨勢延伸"},
        ]
        reward = take_profits[1]["price"] - entry_basis
    else:
        entry_basis = zone_low
        structural_stop = side.stop if side.stop and side.stop > entry_basis else None
        fallback_stop = zone_high + price * stop_pct / 100.0
        stop = _bounded_stop(entry_basis, fallback_stop, structural_stop, "short")
        risk = max(stop - entry_basis, price * 0.001)
        take_profits = [
            {"name": "TP1", "price": entry_basis - risk * tp_multiples[0], "portion_pct": 30, "rr": tp_multiples[0], "note": "先降槓桿風險"},
            {"name": "TP2", "price": entry_basis - risk * tp_multiples[1], "portion_pct": 40, "rr": tp_multiples[1], "note": "主要獲利區"},
            {"name": "TP3", "price": entry_basis - risk * tp_multiples[2], "portion_pct": 30, "rr": tp_multiples[2], "note": "趨勢延伸"},
        ]
        reward = entry_basis - take_profits[1]["price"]

    return {
        "entry_zone": (round(zone_low, 8), round(zone_high, 8)),
        "entry_basis_for_rr": round(entry_basis, 8),
        "stop": round(stop, 8),
        "take_profits": [
            {**tp, "price": round(float(tp["price"]), 8), "rr": round(float(tp["rr"]), 2)}
            for tp in take_profits
        ],
        "rr": round(max(0.0, reward / max(risk, 1e-12)), 2),
        "execution_band_pct": round(execution_band_pct, 4),
    }


def _bounded_stop(entry_basis: float, fallback_stop: float, structural_stop: float | None, direction: str) -> float:
    if structural_stop is None:
        return fallback_stop
    fallback_risk = abs(entry_basis - fallback_stop)
    structural_risk = abs(entry_basis - structural_stop)
    if fallback_risk <= 0 or structural_risk <= 0:
        return fallback_stop
    if fallback_risk * 0.65 <= structural_risk <= fallback_risk * 1.35:
        return structural_stop
    return fallback_stop


def _widen_entry_zone(price: float, entry_zone: tuple[float, float] | None, band_pct: float) -> tuple[float, float]:
    minimum_width = price * band_pct / 100.0
    if entry_zone:
        low, high = sorted((float(entry_zone[0]), float(entry_zone[1])))
        center = (low + high) / 2.0
        width = max(high - low, minimum_width)
    else:
        center = price
        width = minimum_width
    return center - width / 2.0, center + width / 2.0


def _mode_decision_model(
    report: SymbolReport,
    side: DirectionScore,
    opposite: DirectionScore,
    card: dict[str, Any],
    opposite_card: dict[str, Any],
    mode: str,
    session: dict[str, Any],
    plan: dict[str, Any],
    data_stack: dict[str, Any],
) -> dict[str, Any]:
    composite = _as_float(card.get("composite_score")) or _side_score(side)
    direction_score = _as_float(card.get("direction_score")) or _side_score(side)
    opposite_direction = _as_float(opposite_card.get("direction_score")) or _side_score(opposite)
    structure = _as_float(card.get("structure_score")) or _bucket(side, "htf_context")
    entry_precision = _as_float(card.get("entry_precision_score")) or _bucket(side, "entry_location")
    risk_plan = _as_float(card.get("risk_plan_score")) or _bucket(side, "risk_plan")
    derivatives = _as_float(card.get("derivatives_score")) or 50.0
    no_chase = _as_float(card.get("no_chase_score")) or 50.0
    session_score = _as_float(session.get("continuation_score")) or 50.0
    rr_target = 1.45 if mode == "short_term" else 2.0
    rr_score = _clamp(plan["rr"] / rr_target * 72.0)
    edge = max(0.0, direction_score - opposite_direction)
    direction_conviction = _clamp(direction_score * 0.70 + min(100.0, edge * 5.0 + 35.0) * 0.30)
    coinglass_score, coinglass_notes = _coinglass_confluence_score(data_stack, card)
    psychology = _clamp(no_chase * 0.45 + derivatives * 0.30 + session_score * 0.25)

    component_scores = {
        "direction_conviction": round(direction_conviction, 2),
        "structure": round(_clamp(structure), 2),
        "entry_timing": round(_clamp(entry_precision * 0.72 + no_chase * 0.28), 2),
        "risk_reward": round(_clamp(risk_plan * 0.48 + rr_score * 0.52), 2),
        "coinglass_orderflow": round(coinglass_score, 2),
        "market_psychology": round(psychology, 2),
        "session": round(_clamp(session_score), 2),
        "raw_composite": round(_clamp(composite), 2),
    }
    if mode == "short_term":
        weights = {
            "direction_conviction": 0.20,
            "structure": 0.16,
            "entry_timing": 0.20,
            "risk_reward": 0.17,
            "coinglass_orderflow": 0.17,
            "market_psychology": 0.06,
            "session": 0.04,
        }
    else:
        weights = {
            "direction_conviction": 0.22,
            "structure": 0.24,
            "entry_timing": 0.10,
            "risk_reward": 0.22,
            "coinglass_orderflow": 0.12,
            "market_psychology": 0.06,
            "session": 0.04,
        }
    quality = _clamp(sum(component_scores[name] * weight for name, weight in weights.items()) / sum(weights.values()))

    hard_blockers: list[str] = []
    soft_notes: list[str] = list(coinglass_notes)
    if direction_conviction < 52.0:
        hard_blockers.append("ETH direction conviction is too weak; long/short edge has not separated enough.")
    if structure < 45.0 and mode == "swing":
        hard_blockers.append("Swing structure is not clean enough for a medium-term ETH plan.")
    elif structure < 45.0:
        soft_notes.append("Structure score is weak; short-term plan needs a cleaner intraday trigger.")
    if entry_precision < 42.0:
        soft_notes.append("Entry timing is not precise yet; wait for the price to return into the planned zone.")
    if derivatives < 42.0 and coinglass_score < 45.0:
        soft_notes.append("Derivatives/orderflow are not aligned enough; keep this as a watch setup.")

    return {
        "quality_score": round(quality, 2),
        "component_scores": component_scores,
        "checks": {
            "direction_ready": direction_conviction >= 52.0,
            "structure_ready": structure >= (45.0 if mode == "short_term" else 50.0),
            "entry_timing_ready": entry_precision >= 42.0,
            "risk_reward_ready": plan["rr"] >= (1.25 if mode == "short_term" else 1.70),
            "coinglass_ready": bool(data_stack.get("has_coinglass")),
            "public_derivatives_ready": bool(data_stack.get("has_public_derivatives")),
            "optional_paid_api_required": False,
        },
        "data_policy": {
            "primary_paid_provider": "CoinGlass",
            "mandatory_paid_provider": "CoinGlass only when configured; missing optional paid APIs never hard-block ETH.",
            "free_baseline": "Bybit/Binance public OI and funding",
            "optional_paid_providers": ["Coinalyze", "Glassnode", "CryptoQuant"],
        },
        "hard_blockers": hard_blockers,
        "soft_notes": soft_notes,
    }


def _coinglass_confluence_score(data_stack: dict[str, Any], card: dict[str, Any]) -> tuple[float, list[str]]:
    derivatives = _as_float(card.get("derivatives_score")) or 50.0
    score = derivatives
    notes: list[str] = []
    if data_stack.get("has_coinglass"):
        score += 8.0
        if data_stack.get("coinglass_actionable"):
            score += 7.0
        else:
            notes.append("CoinGlass is connected, but this scan has limited detailed heatmap/orderflow fields.")
    else:
        score -= 6.0
        notes.append("CoinGlass is the only paid primary data source; this scan did not read it successfully.")
    if data_stack.get("has_public_derivatives"):
        score += 4.0
    else:
        score -= 8.0
        notes.append("Public exchange OI/funding was not available, so derivatives confidence is capped.")
    return _clamp(score), notes


def _mode_quality(
    report: SymbolReport,
    side: DirectionScore,
    card: dict[str, Any],
    opposite_card: dict[str, Any],
    mode: str,
    session: dict[str, Any],
    plan: dict[str, Any],
) -> float:
    composite = _as_float(card.get("composite_score")) or _side_score(side)
    direction_score = _as_float(card.get("direction_score")) or _side_score(side)
    entry_score = _as_float(card.get("entry_precision_score")) or _bucket(side, "entry_location")
    derivatives = _as_float(card.get("derivatives_score")) or 50.0
    no_chase = _as_float(card.get("no_chase_score")) or 50.0
    structure = _as_float(card.get("structure_score")) or _bucket(side, "htf_context")
    risk = _as_float(card.get("risk_plan_score")) or _bucket(side, "risk_plan")
    opposite_direction = _as_float(opposite_card.get("direction_score")) or 50.0
    edge = max(0.0, direction_score - opposite_direction)
    rr_score = min(100.0, plan["rr"] / (1.45 if mode == "short_term" else 2.0) * 70.0)
    session_score = float(session["continuation_score"])
    if mode == "short_term":
        weights = (
            (composite, 0.18),
            (direction_score, 0.18),
            (entry_score, 0.18),
            (derivatives, 0.18),
            (no_chase, 0.12),
            (rr_score, 0.10),
            (session_score, 0.06),
        )
    else:
        weights = (
            (structure, 0.24),
            (direction_score, 0.22),
            (derivatives, 0.16),
            (risk, 0.16),
            (rr_score, 0.14),
            (edge * 4.0 + 45.0, 0.08),
        )
    return _clamp(sum(value * weight for value, weight in weights) / sum(weight for _, weight in weights))


def _volatility_context(report: SymbolReport, side: DirectionScore) -> dict[str, Any]:
    metrics = side.market_metrics if isinstance(side.market_metrics, dict) else {}
    atr_pct = _as_float(metrics.get("atr_pct"))
    if atr_pct is None:
        atr_pct = max(0.45, min(2.6, abs(float(report.change_pct_24h or 0.0)) / 7.5 + 0.35))
    three_day_range = _as_float(metrics.get("three_day_range_pct"))
    if three_day_range is None:
        three_day_range = max(atr_pct * 4.2, abs(float(report.change_pct_24h or 0.0)) * 0.75)
    expansion = _as_float(metrics.get("volatility_expansion_ratio"))
    if expansion is None:
        expansion = 1.0 if atr_pct < 0.9 else min(2.4, atr_pct / 0.8)
    if atr_pct >= 1.55 or expansion >= 1.75:
        regime = "hot"
    elif atr_pct <= 0.62:
        regime = "compressed"
    else:
        regime = "normal"
    return {
        "atr_pct": round(float(atr_pct), 4),
        "three_day_range_pct": round(float(three_day_range), 4),
        "volatility_expansion_ratio": round(float(expansion), 4),
        "regime": regime,
    }


def _chase_context(
    report: SymbolReport,
    side: DirectionScore,
    card: dict[str, Any],
    direction: str,
    session: dict[str, Any],
    entry_distance: float,
) -> dict[str, Any]:
    metrics = side.market_metrics if isinstance(side.market_metrics, dict) else {}
    no_chase = _as_float(card.get("no_chase_score")) or 50.0
    mover_permission = bool(card.get("mover_execution_permission") or metrics.get("mover_execution_permission"))
    strategy_hint = str(card.get("strategy_hint") or "").lower()
    breakout_hint = "breakout" in strategy_hint or "momentum" in strategy_hint
    directional_24h = float(report.change_pct_24h or 0.0)
    direction_chase = (direction == "long" and directional_24h > 2.8) or (direction == "short" and directional_24h < -2.8)
    is_chasing = bool(metrics.get("mover_chase_risk")) or (entry_distance > 0.38 and direction_chase) or no_chase < 42.0
    continuation_allowed = (
        is_chasing
        and (mover_permission or breakout_hint)
        and bool(session.get("trend_continuation_window"))
        and (_as_float(card.get("derivatives_score")) or 50.0) >= 54.0
    )
    return {
        "is_chasing": is_chasing,
        "continuation_allowed": continuation_allowed,
        "mover_permission": mover_permission,
        "breakout_hint": breakout_hint,
        "no_chase_score": round(no_chase, 2),
        "reason": "breakout/session/flow supported" if continuation_allowed else "wait retest unless true breakout confirms",
    }


def _session_context(now: datetime) -> dict[str, Any]:
    hour = now.astimezone(timezone.utc).hour
    if 0 <= hour < 7:
        name = "Asia"
        bias = "range-building / liquidity formation"
        continuation = False
        score = 48.0
    elif 7 <= hour < 12:
        name = "London"
        bias = "breakout or false-break check"
        continuation = True
        score = 68.0
    elif 12 <= hour < 17:
        name = "New York overlap"
        bias = "highest liquidity; continuation allowed only after confirmation"
        continuation = True
        score = 76.0
    elif 17 <= hour < 21:
        name = "Late US"
        bias = "manage position; avoid late chase"
        continuation = False
        score = 55.0
    else:
        name = "Rollover"
        bias = "thin book; reduce automation"
        continuation = False
        score = 42.0
    return {
        "name": name,
        "utc_hour": hour,
        "bias": bias,
        "trend_continuation_window": continuation,
        "continuation_score": score,
    }


def _active_plan_status(active: Any, report: SymbolReport, now: datetime, settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(active, dict) or not active.get("plan_id"):
        return {"status": "none"}
    direction = str(active.get("direction") or "")
    stop = _as_float(active.get("stop"))
    targets = active.get("take_profits") if isinstance(active.get("take_profits"), list) else []
    final_target = _as_float(targets[-1].get("price")) if targets and isinstance(targets[-1], dict) else None
    price = float(report.price)
    created_at = _parse_iso(active.get("created_at"))
    ttl = settings["active_plan_ttl_minutes"]
    if active.get("mode") == "swing":
        ttl = settings["swing_plan_ttl_minutes"]
    if created_at and (now - created_at).total_seconds() > ttl * 60:
        return {**active, "status": "expired", "reason": "計畫超過生命週期。"}
    if direction == "long":
        if stop is not None and price <= stop:
            return {**active, "status": "stop_hit", "reason": "價格觸及 ETH 計畫止損。"}
        if final_target is not None and price >= final_target:
            return {**active, "status": "target_hit", "reason": "價格觸及 ETH 最終目標。"}
    if direction == "short":
        if stop is not None and price >= stop:
            return {**active, "status": "stop_hit", "reason": "價格觸及 ETH 計畫止損。"}
        if final_target is not None and price <= final_target:
            return {**active, "status": "target_hit", "reason": "價格觸及 ETH 最終目標。"}
    if report.selected_direction in {"long", "short"} and report.selected_direction != direction:
        score_gap = abs(_side_score(report.long) - _side_score(report.short))
        if score_gap >= 18.0 and report.score >= 84.0:
            return {**active, "status": "opposite_signal", "reason": "出現高分反向 ETH 訊號，原計畫結束。"}
    return {**active, "status": "active", "reason": "原 ETH 計畫仍有效。"}


def _update_state_from_analysis(state: dict[str, Any], analysis: dict[str, Any], now: datetime) -> None:
    _update_direction_lock(state, analysis, now)
    active_status = analysis.get("active_plan_status") if isinstance(analysis.get("active_plan_status"), dict) else {}
    if active_status.get("status") in {"stop_hit", "target_hit", "expired", "opposite_signal"}:
        state["completed_plan"] = {**active_status, "completed_at": now.isoformat()}
        state.pop("active_plan", None)
    mode = analysis.get("trader_mode")
    if not isinstance(mode, dict):
        mode = analysis.get("modes", {}).get("short_term", {})
    if not isinstance(mode, dict) or not mode.get("should_open_new"):
        return
    plan_record = {
        "plan_id": mode["plan_id"],
        "mode": mode["mode"],
        "direction": mode["direction"],
        "entry_zone": mode["entry_zone"],
        "entry_basis_for_rr": mode["entry_basis_for_rr"],
        "stop": mode["stop"],
        "take_profits": mode["take_profits"],
        "rr": mode["rr"],
        "created_at": now.isoformat(),
        "status": "active",
    }
    execution_control = analysis.get("execution_control") if isinstance(analysis.get("execution_control"), dict) else {}
    if execution_control.get("defer_active_plan_until_gate_submission"):
        state["candidate_plan"] = {**plan_record, "status": "ready_waiting_gate_submission"}
        return
    state["active_plan"] = plan_record


def _update_direction_lock(state: dict[str, Any], analysis: dict[str, Any], now: datetime) -> None:
    control = analysis.get("direction_control") if isinstance(analysis.get("direction_control"), dict) else {}
    direction = str(control.get("direction") or analysis.get("direction") or "")
    raw_direction = str(control.get("raw_direction") or direction)
    lock = state.get("direction_lock") if isinstance(state.get("direction_lock"), dict) else {}
    locked_direction = str(lock.get("direction") or "")
    if locked_direction in {"long", "short"} and raw_direction in {"long", "short"} and raw_direction != locked_direction:
        opposite_count = int(lock.get("opposite_confirm_count") or 0) + 1
    else:
        opposite_count = 0
    if direction not in {"long", "short"}:
        if lock:
            lock["last_seen_at"] = now.isoformat()
            lock["opposite_confirm_count"] = opposite_count
            state["direction_lock"] = lock
        return
    if locked_direction == direction:
        lock.update(
            {
                "direction": direction,
                "last_seen_at": now.isoformat(),
                "confirm_count": int(lock.get("confirm_count") or 0) + 1,
                "opposite_confirm_count": opposite_count,
                "direction_edge": control.get("direction_edge"),
                "long_direction_score": control.get("long_direction_score"),
                "short_direction_score": control.get("short_direction_score"),
            }
        )
        state["direction_lock"] = lock
        return
    state["direction_lock"] = {
        "direction": direction,
        "locked_at": now.isoformat(),
        "last_seen_at": now.isoformat(),
        "confirm_count": 1,
        "opposite_confirm_count": 0,
        "direction_edge": control.get("direction_edge"),
        "long_direction_score": control.get("long_direction_score"),
        "short_direction_score": control.get("short_direction_score"),
        "reason": control.get("reason"),
    }


def _primary_mode(short_mode: dict[str, Any], swing_mode: dict[str, Any]) -> dict[str, Any]:
    order = {
        "execute_ready": 6,
        "plan_completed": 5,
        "manage_existing": 4,
        "armed_wait_entry": 3,
        "wait_retest": 2,
        "observe": 1,
    }
    return max((short_mode, swing_mode), key=lambda item: (order.get(str(item.get("state")), 0), float(item.get("quality_score") or 0.0)))


def _trader_mode(primary_mode: dict[str, Any]) -> dict[str, Any]:
    trader = dict(primary_mode)
    source_mode = str(primary_mode.get("mode") or "short_term")
    trader["source_mode"] = source_mode
    trader["mode"] = "trader"
    trader["label"] = "ETH 交易員模式"
    decision = str(trader.get("decision") or "")
    if decision.startswith("短線 "):
        trader["decision"] = "ETH 交易員 " + decision.removeprefix("短線 ")
    elif decision.startswith("中長線 "):
        trader["decision"] = "ETH 交易員 " + decision.removeprefix("中長線 ")
    trader["scaling_policy"] = {
        "initial_entry": "price must touch the planned entry zone or a confirmed breakout-retest trigger",
        "add_position": "only after +0.8R and a fresh retest/continuation signal; never average down blindly",
        "reduce_position": "take partial profit at TP ladder and tighten stop after R milestones",
        "state_owner": "Gate position state controls live management after an order is submitted",
    }
    return trader


def _lifecycle_summary(active_status: dict[str, Any], primary_mode: dict[str, Any]) -> str:
    status = active_status.get("status")
    if status == "active":
        return "manage_existing_plan"
    if status in {"stop_hit", "target_hit", "expired", "opposite_signal"}:
        return "plan_completed"
    if primary_mode.get("should_open_new"):
        return "new_plan_ready"
    if primary_mode.get("state") == "armed_wait_entry":
        return "plan_armed_waiting_entry"
    return "flat_observe"


def _data_stack(report: SymbolReport) -> dict[str, Any]:
    paid = report.metadata.get("paid_data", {}) if isinstance(report.metadata.get("paid_data"), dict) else {}
    values = paid.get("values", {}) if isinstance(paid.get("values"), dict) else {}
    providers = paid.get("providers", []) if isinstance(paid.get("providers"), list) else []
    public = values.get("exchange_public_derivatives") if isinstance(values.get("exchange_public_derivatives"), dict) else {}
    coinglass_keys = (
        "coinglass_funding",
        "coinglass_taker_buy_sell",
        "coinglass_long_short_ratio",
        "coinglass_liquidation_sum",
        "coinglass_liquidation_heatmap",
        "coinglass_liquidation_map",
        "coinglass_orderbook_heatmap",
    )
    coinglass_actionable = any(key in values for key in coinglass_keys)
    readiness = paid.get("configured_api_readiness", {}) if isinstance(paid.get("configured_api_readiness"), dict) else {}
    return {
        "providers": providers,
        "has_public_derivatives": bool(public),
        "has_coinglass": "coinglass" in values or coinglass_actionable or any("CoinGlass" in str(provider) for provider in providers),
        "coinglass_actionable": coinglass_actionable,
        "has_coinalyze": "coinalyze" in values or any("Coinalyze" in str(provider) for provider in providers),
        "configured_api_readiness": readiness,
        "funding_rate": public.get("funding_rate"),
        "open_interest_change_pct": public.get("open_interest_change_pct"),
        "spread_pct": public.get("spread_pct"),
        "paid_api_policy": {
            "primary_paid": "CoinGlass",
            "not_required": ["Coinalyze", "Glassnode", "CryptoQuant", "Token Metrics"],
            "rule": "Only CoinGlass is treated as the paid primary source. Other paid APIs are optional confluence and cannot block ETH execution when absent.",
        },
        "note": "ETH strategy uses CoinGlass and public exchange derivatives as confluence/risk filters; missing optional paid APIs does not force a block.",
    }


def _market_psychology(
    report: SymbolReport,
    card: dict[str, Any],
    direction: str,
    session: dict[str, Any],
) -> list[str]:
    lines = []
    derivatives_bias = card.get("derivatives_bias") or "neutral"
    crowding = _as_float(card.get("derivatives_crowding_risk")) or 0.0
    squeeze = _as_float(card.get("squeeze_fuel_score")) or 0.0
    lines.append(f"Session {session['name']}: {session['bias']}.")
    if derivatives_bias in {"long", "short"}:
        lines.append(f"Derivatives bias is {derivatives_bias}; selected direction is {direction}.")
    if crowding >= 55.0:
        lines.append("Crowding risk is high; prefer retest or squeeze failure confirmation before entry.")
    if squeeze >= 18.0:
        lines.append("Opposite-side positioning can fuel continuation after breakout confirmation.")
    if abs(float(report.change_pct_24h or 0.0)) >= 3.0:
        lines.append("24h move is extended; trend trade must avoid late chase and use a wider ETH-specific plan.")
    return lines


def _decision_text(state: str, mode: str, direction: str, blockers: list[str], notes: list[str]) -> str:
    label = "ETH 交易員" if mode == "trader" else "短線" if mode == "short_term" else "中長線"
    if state == "execute_ready":
        return f"{label} {direction} 計畫可執行；等待 Gate 交易模組確認乾跑/實單設定。"
    if state == "manage_existing":
        return f"{label} 原計畫仍有效，先管理持倉/掛單，不新增同向計畫。"
    if state == "armed_wait_entry":
        return f"{label} 計畫成立但尚未到理想入場區，等待回測或下一根確認。"
    if state == "wait_retest":
        return f"{label} 方向有機會，但目前不追價，等待回測/突破確認。"
    if state == "plan_completed":
        return f"{label} 原計畫已結束，可重新評估新計畫。"
    if blockers:
        return f"{label} 暫不交易：{blockers[0]}"
    if notes:
        return f"{label} 觀察：{notes[0]}"
    return f"{label} 暫無高品質 ETH 交易計畫。"


def _empty_mode(mode: str, state: str, reason: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "label": "短線執行" if mode == "short_term" else "中長線波段",
        "state": state,
        "quality_score": 0.0,
        "should_open_new": False,
        "notify": False,
        "decision": reason,
        "hard_blockers": [reason],
        "soft_notes": [],
        "component_scores": {},
        "unified_checks": {"optional_paid_api_required": False},
        "data_policy": {
            "primary_paid_provider": "CoinGlass",
            "optional_paid_providers": ["Coinalyze", "Glassnode", "CryptoQuant"],
        },
    }


def _plan_id(symbol: str, mode: str, direction: str, plan: dict[str, Any]) -> str:
    payload = {
        "symbol": symbol,
        "mode": mode,
        "direction": direction,
        "entry_zone": [round(float(x), 2) for x in plan["entry_zone"]],
        "stop": round(float(plan["stop"]), 2),
        "tp2": round(float(plan["take_profits"][1]["price"]), 2),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _entry_distance_pct(price: float, entry_zone: tuple[float, float] | list[float] | None) -> float:
    if not entry_zone or len(entry_zone) < 2:
        return 999.0
    low, high = sorted((float(entry_zone[0]), float(entry_zone[1])))
    if low <= price <= high:
        return 0.0
    return min(abs(price - low), abs(price - high)) / max(abs(price), 1e-12) * 100.0


def _side_score(side: DirectionScore) -> float:
    if side.selection_score is not None:
        return float(side.selection_score)
    if side.calibrated_score is not None:
        return float(side.calibrated_score)
    return float(side.normalized)


def _bucket(side: DirectionScore, key: str) -> float:
    try:
        return float((side.bucket_scores or {}).get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _direction_label(direction: str) -> str:
    return {"long": "看多", "short": "看空", "neutral": "觀望"}.get(direction, direction)


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
