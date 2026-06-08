from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .models import DirectionScore, SymbolReport
from .quant_scorecard import build_quant_scorecard


ETH_STRATEGY_VERSION = "eth_dedicated_2026_06_v1"
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

    direction = _select_eth_direction(report, scorecard)
    side = _side_for_direction(report, direction)
    opposite = report.short if direction == "long" else report.long
    side_card = scorecard.get(direction, {}) if direction in {"long", "short"} else {}
    opposite_card = scorecard.get("short" if direction == "long" else "long", {})
    session = _session_context(now)
    active_status = _active_plan_status(state.get("active_plan"), report, now, settings)
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
    )
    primary_mode = _primary_mode(short_mode, swing_mode)
    lifecycle = _lifecycle_summary(active_status, primary_mode)
    data_stack = _data_stack(report)
    psychology = _market_psychology(report, side_card, direction, session)
    discord_notify = bool(primary_mode.get("notify"))
    if active_status.get("status") in {"stop_hit", "target_hit", "expired", "opposite_signal"}:
        discord_notify = True

    return {
        "version": ETH_STRATEGY_VERSION,
        "symbol": report.symbol,
        "generated_at": now.isoformat(),
        "price": report.price,
        "direction": direction,
        "direction_label": _direction_label(direction),
        "session": session,
        "primary_mode": primary_mode.get("mode", "short_term"),
        "primary_plan_state": primary_mode.get("state", "observe"),
        "plan_lifecycle": lifecycle,
        "active_plan_status": active_status,
        "discord_notify": discord_notify,
        "modes": {
            "short_term": short_mode,
            "swing": swing_mode,
        },
        "data_stack": data_stack,
        "market_psychology": psychology,
        "risk_design": {
            "rr_from": "long uses the upper edge of entry zone; short uses the lower edge, so wider limit ranges do not fake RR.",
            "stop_policy": "Stops are volatility- and structure-bounded; widening the stop also widens TP ladder.",
            "no_chase_policy": "Trend continuation is allowed only when breakout/session/flow conditions support it; otherwise wait for retest.",
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
    }


def _select_eth_direction(report: SymbolReport, scorecard: dict[str, Any]) -> str:
    if report.selected_direction in {"long", "short"}:
        return report.selected_direction
    long_score = _as_float(scorecard.get("long", {}).get("direction_score")) or _side_score(report.long)
    short_score = _as_float(scorecard.get("short", {}).get("direction_score")) or _side_score(report.short)
    edge = abs(long_score - short_score)
    if edge < 6.0:
        return "neutral"
    return "long" if long_score > short_score else "short"


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
) -> dict[str, Any]:
    if direction not in {"long", "short"}:
        return _empty_mode(mode, "wait_direction", "ETH 多空分數沒有拉開，先不建立交易計畫。")
    if mode == "short_term" and not settings["short_mode_enabled"]:
        return _empty_mode(mode, "disabled", "短線模式已停用。")
    if mode == "swing" and not settings["swing_mode_enabled"]:
        return _empty_mode(mode, "disabled", "中長線模式已停用。")

    volatility = _volatility_context(report, side)
    plan = _build_plan(report, side, direction, mode, volatility)
    quality = _mode_quality(report, side, card, opposite_card, mode, session, plan)
    min_score = settings["short_min_score"] if mode == "short_term" else settings["swing_min_score"]
    no_chase = _as_float(card.get("no_chase_score")) or 50.0
    entry_distance = _entry_distance_pct(report.price, plan["entry_zone"])
    chase = _chase_context(report, side, card, direction, session, entry_distance)
    rr_floor = 1.25 if mode == "short_term" else 1.70
    hard_blockers: list[str] = []
    soft_notes: list[str] = []

    if plan["rr"] < rr_floor:
        hard_blockers.append(f"RR {plan['rr']:.2f}R 低於 {rr_floor:.2f}R。")
    if quality < min_score:
        hard_blockers.append(f"{mode} 分數 {quality:.1f} 低於門檻 {min_score:.1f}。")
    if mode == "short_term" and chase["is_chasing"] and not chase["continuation_allowed"]:
        hard_blockers.append("目前屬於追高/追低，不是突破確認或回測入場。")
    if no_chase < 38 and not chase["continuation_allowed"]:
        soft_notes.append("no-chase 分數偏低，需等回測或流動性掃完。")
    if side.entry_zone is None:
        soft_notes.append("原始模型沒有完整結構 entry，ETH 模式用波動/區間重建保守計畫。")
    if side.stop is None or not side.take_profits:
        soft_notes.append("原始模型交易計畫不完整，ETH 模式已重算 stop/TP/RR。")

    state_name = "observe"
    if active_status.get("status") == "active" and active_status.get("direction") == direction:
        state_name = "manage_existing"
        hard_blockers.append("同方向 ETH 計畫仍在生命週期內，先管理原計畫，不重複開新單。")
    elif active_status.get("status") in {"stop_hit", "target_hit", "expired", "opposite_signal"}:
        state_name = "plan_completed"
    elif hard_blockers:
        state_name = "wait_retest" if chase["is_chasing"] else "observe"
    elif mode == "short_term" and quality >= min_score + 4 and entry_distance <= plan["execution_band_pct"]:
        state_name = "execute_ready"
    elif entry_distance <= plan["execution_band_pct"] * (1.8 if mode == "short_term" else 2.5):
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
        if score_gap >= 10.0 and report.score >= 74.0:
            return {**active, "status": "opposite_signal", "reason": "出現高分反向 ETH 訊號，原計畫結束。"}
    return {**active, "status": "active", "reason": "原 ETH 計畫仍有效。"}


def _update_state_from_analysis(state: dict[str, Any], analysis: dict[str, Any], now: datetime) -> None:
    active_status = analysis.get("active_plan_status") if isinstance(analysis.get("active_plan_status"), dict) else {}
    if active_status.get("status") in {"stop_hit", "target_hit", "expired", "opposite_signal"}:
        state["completed_plan"] = {**active_status, "completed_at": now.isoformat()}
        state.pop("active_plan", None)
    mode = analysis.get("modes", {}).get("short_term", {})
    if not isinstance(mode, dict) or not mode.get("should_open_new"):
        return
    state["active_plan"] = {
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
    return {
        "providers": providers,
        "has_public_derivatives": bool(public),
        "has_coinglass": "coinglass" in values or any("CoinGlass" in str(provider) for provider in providers),
        "has_coinalyze": "coinalyze" in values or any("Coinalyze" in str(provider) for provider in providers),
        "funding_rate": public.get("funding_rate"),
        "open_interest_change_pct": public.get("open_interest_change_pct"),
        "spread_pct": public.get("spread_pct"),
        "note": "ETH strategy uses these as confluence and risk filters; missing paid data does not force an automatic trade.",
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
    label = "短線" if mode == "short_term" else "中長線"
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
