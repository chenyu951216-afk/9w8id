from __future__ import annotations

from typing import Any

from .instrument_classifier import trading_standard_profile, volatility_profile
from .models import DirectionScore, SymbolReport


SCORECARD_VERSION = "quant_scorecard_2026_06_v1"


def build_quant_scorecard(report: SymbolReport) -> dict[str, Any]:
    long_card = build_side_scorecard(report, report.long, "long")
    short_card = build_side_scorecard(report, report.short, "short")
    selected = report.selected_direction if report.selected_direction in {"long", "short"} else (
        "long" if float(long_card["composite_score"]) >= float(short_card["composite_score"]) else "short"
    )
    return {
        "version": SCORECARD_VERSION,
        "selected_direction": report.selected_direction,
        "selected_card": selected,
        "long": long_card,
        "short": short_card,
        "edge": round(abs(float(long_card["direction_score"]) - float(short_card["direction_score"])), 2),
    }


def build_side_scorecard(report: SymbolReport, side: DirectionScore, direction: str) -> dict[str, Any]:
    buckets = side.bucket_scores or {}
    metrics = side.market_metrics or {}
    instrument = volatility_profile(report.symbol).instrument_class
    standard = trading_standard_profile(report.symbol)
    external = _external_context(report)
    derivatives = _derivatives_score(report, side, direction, external)
    structure_score = _weighted(
        (
            (_bucket(buckets, "htf_context"), 0.30),
            (_bucket(buckets, "ltf_confirmation"), 0.28),
            (_bucket(buckets, "entry_location"), 0.24),
            (_bucket(buckets, "risk_plan"), 0.18),
        )
    )
    entry_precision = _entry_precision_score(report, side)
    risk_score = _bucket(buckets, "risk_plan")
    execution_base = _as_float(side.execution_score)
    if execution_base is None:
        execution_base = _weighted(
            (
                (structure_score, 0.35),
                (entry_precision, 0.25),
                (risk_score, 0.20),
                (_bucket(buckets, "market_filter"), 0.20),
            )
        )
    no_chase = _no_chase_score(metrics)
    direction_base = _as_float(side.selection_score)
    if direction_base is None:
        direction_base = _as_float(side.calibrated_score)
    if direction_base is None:
        direction_base = side.normalized
    mover_profile = str(metrics.get("mover_profile") or "normal")
    mover_active = mover_profile in {"active_mover", "hot_mover", "extreme_mover"}
    if instrument in {"altcoin", "large_altcoin"}:
        weights = {
            "structure": 0.24,
            "direction": 0.15,
            "entry": 0.18,
            "risk": 0.13,
            "derivatives": 0.18,
            "execution": 0.05,
            "no_chase": 0.07,
        }
    else:
        weights = {
            "structure": 0.27,
            "direction": 0.20,
            "entry": 0.17,
            "risk": 0.14,
            "derivatives": 0.10,
            "execution": 0.07,
            "no_chase": 0.05,
        }
    if mover_active:
        weights["derivatives"] += 0.04
        weights["entry"] += 0.03
        weights["no_chase"] += 0.03
        weights["direction"] = max(0.10, weights["direction"] - 0.04)
        weights["structure"] = max(0.20, weights["structure"] - 0.03)
    total_weight = sum(weights.values()) or 1.0
    composite = (
        structure_score * weights["structure"]
        + direction_base * weights["direction"]
        + entry_precision * weights["entry"]
        + risk_score * weights["risk"]
        + derivatives["score"] * weights["derivatives"]
        + execution_base * weights["execution"]
        + no_chase * weights["no_chase"]
    ) / total_weight
    direction_adjustment = _direction_adjustment(
        direction,
        derivatives,
        metrics,
        entry_precision,
        no_chase,
        standard.min_score_gap,
    )
    strategy_hint = _strategy_hint(instrument, mover_profile, metrics, derivatives, entry_precision, structure_score, risk_score, no_chase)
    return {
        "direction": direction,
        "instrument_class": instrument,
        "structure_score": round(structure_score, 2),
        "direction_score": round(_clamp(direction_base + direction_adjustment), 2),
        "raw_direction_score": round(direction_base, 2),
        "direction_adjustment": round(direction_adjustment, 2),
        "entry_precision_score": round(entry_precision, 2),
        "risk_plan_score": round(risk_score, 2),
        "execution_score": round(execution_base, 2),
        "derivatives_score": round(derivatives["score"], 2),
        "derivatives_bias": derivatives["bias"],
        "derivatives_confidence": derivatives["confidence"],
        "derivatives_crowding_risk": round(derivatives["crowding_risk"], 2),
        "squeeze_fuel_score": round(derivatives["squeeze_fuel_score"], 2),
        "no_chase_score": round(no_chase, 2),
        "composite_score": round(_clamp(composite), 2),
        "strategy_hint": strategy_hint,
        "mover_profile": mover_profile,
        "mover_direction": metrics.get("mover_direction", "neutral"),
        "mover_chase_risk": bool(metrics.get("mover_chase_risk")),
        "mover_execution_permission": bool(metrics.get("mover_execution_permission")),
        "external_context_summary": derivatives["summary"],
        "external_evidence_count": derivatives["evidence_count"],
        "limits": {
            "min_selection_score": standard.limit_min_selection_score,
            "min_execution_score": standard.limit_min_execution_score,
            "min_rr": standard.min_rr,
            "scalp_min_rr": standard.scalp_min_rr,
            "max_entry_distance_pct": standard.max_entry_distance_pct,
        },
        "weights": {key: round(value / total_weight, 4) for key, value in weights.items()},
    }


def _derivatives_score(
    report: SymbolReport,
    side: DirectionScore,
    direction: str,
    external: dict[str, Any],
) -> dict[str, Any]:
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives") if isinstance(values.get("exchange_public_derivatives"), dict) else {}
    context_bias = str(external.get("bias") or "neutral")
    context_confidence = _as_float(external.get("confidence")) or 0.0
    evidence_count = int(external.get("evidence_count") or 0)
    side_key = "long_score" if direction == "long" else "short_score"
    opposite_key = "short_score" if direction == "long" else "long_score"
    side_context = _as_float(external.get(side_key))
    opposite_context = _as_float(external.get(opposite_key))
    score = side_context if side_context is not None else 50.0
    crowding = 0.0
    squeeze_fuel = 0.0
    notes: list[str] = []

    if side_context is not None and opposite_context is not None:
        delta = side_context - opposite_context
        notes.append(f"external context delta {delta:.1f}")
        if context_bias in {"long", "short"} and context_bias != direction and context_confidence >= 68.0:
            crowding += min(28.0, 8.0 + (context_confidence - 68.0) * 0.55)

    funding_values = _funding_values(values, public)
    if funding_values:
        funding = sum(funding_values) / len(funding_values)
        funding_side = "long" if funding > 0 else "short" if funding < 0 else "neutral"
        abs_funding = abs(funding)
        if abs_funding >= 0.00015:
            if funding_side == direction:
                if abs_funding >= 0.0008:
                    crowding += 30.0
                    score -= 10.0
                    notes.append(f"same-side funding crowded {funding:.5f}")
                else:
                    score += 2.0
                    crowding += 5.0
                    notes.append(f"same-side funding participation {funding:.5f}")
            else:
                add = 8.0 if abs_funding >= 0.0008 else 4.0
                squeeze_fuel += add
                score += add
                notes.append(f"opposite-side funding squeeze fuel {funding:.5f}")

    oi_change = _first_float(
        public,
        ("open_interest_change_pct",),
    )
    coinalyze = values.get("coinalyze") if isinstance(values.get("coinalyze"), dict) else {}
    if oi_change is None and isinstance(coinalyze, dict):
        oi_change = _as_float(coinalyze.get("open_interest_change_pct"))
    flow_alignment = _flow_alignment(values, public, direction)
    if oi_change is not None:
        abs_oi = abs(oi_change)
        if oi_change > 0 and flow_alignment is not None and flow_alignment >= 0.56:
            score += min(12.0, 3.0 + abs_oi * 0.30)
            notes.append(f"OI expands with selected flow {oi_change:.1f}%")
        elif oi_change > 0 and flow_alignment is not None and flow_alignment <= 0.44:
            crowding += min(24.0, 5.0 + abs_oi * 0.35)
            score -= min(12.0, 2.0 + abs_oi * 0.20)
            notes.append(f"OI expands against selected flow {oi_change:.1f}%")
        elif abs_oi >= 20.0:
            crowding += min(16.0, abs_oi * 0.22)
            notes.append(f"OI change hot but unconfirmed {oi_change:.1f}%")

    if flow_alignment is not None:
        if flow_alignment >= 0.62:
            score += 7.0
        elif flow_alignment >= 0.56:
            score += 3.0
        elif flow_alignment <= 0.38:
            score -= 9.0
            crowding += 10.0
        elif flow_alignment <= 0.44:
            score -= 4.0

    flags = external.get("risk_flags", []) if isinstance(external.get("risk_flags"), list) else []
    if (direction == "long" and "long_crowded" in flags) or (direction == "short" and "short_crowded" in flags):
        crowding += 24.0
        score -= 8.0
    if (direction == "long" and "stop_hunt_risk_long" in flags) or (direction == "short" and "stop_hunt_risk_short" in flags):
        crowding += 10.0
        score -= 4.0

    metrics = side.market_metrics or {}
    if metrics.get("mover_chase_risk"):
        score -= 12.0
        crowding += 12.0
    elif metrics.get("mover_execution_permission") and str(metrics.get("mover_profile") or "") in {"active_mover", "hot_mover", "extreme_mover"}:
        score += 4.0

    confidence = max(context_confidence, min(92.0, 50.0 + evidence_count * 3.0 + abs(score - 50.0) * 0.7))
    return {
        "score": _clamp(score),
        "bias": context_bias,
        "confidence": round(confidence, 2),
        "crowding_risk": _clamp(crowding),
        "squeeze_fuel_score": _clamp(squeeze_fuel),
        "summary": "; ".join((external.get("notes") or [])[:4]) if isinstance(external.get("notes"), list) else str(external.get("summary") or "; ".join(notes[:4]) or "derivatives context proxy"),
        "evidence_count": evidence_count,
    }


def _entry_precision_score(report: SymbolReport, side: DirectionScore) -> float:
    buckets = side.bucket_scores or {}
    metrics = side.market_metrics or {}
    entry = _bucket(buckets, "entry_location")
    anchor = _as_float(metrics.get("entry_anchor_score"))
    if anchor is None:
        anchor = 45.0 if side.entry_zone else 0.0
    distance = _as_float(side.entry_distance_pct)
    if distance is None:
        distance = _entry_distance_pct(report.price, side.entry_zone)
    band = _as_float(metrics.get("adaptive_entry_band_pct"))
    if band is None:
        band = trading_standard_profile(report.symbol).max_entry_distance_pct
    if distance is None:
        distance_score = 35.0
    elif distance <= band:
        distance_score = 92.0
    elif distance <= band * 2.5:
        distance_score = 72.0
    elif distance <= band * 5.0:
        distance_score = 45.0
    else:
        distance_score = 18.0
    origin = str(getattr(side, "entry_origin", "") or "")
    origin_score = {
        "validated_pullback": 88.0,
        "order_block": 86.0,
        "ote": 72.0,
        "fvg": 64.0,
        "fallback": 18.0,
        "market_price": 24.0,
    }.get(origin, 45.0)
    return _weighted(((entry, 0.32), (anchor, 0.30), (distance_score, 0.23), (origin_score, 0.15)))


def _no_chase_score(metrics: dict[str, Any]) -> float:
    profile = str(metrics.get("mover_profile") or "normal")
    if bool(metrics.get("mover_chase_risk")):
        return 25.0
    if profile in {"hot_mover", "extreme_mover"} and bool(metrics.get("mover_same_side")):
        return 82.0 if bool(metrics.get("mover_execution_permission")) else 38.0
    if profile in {"active_mover", "hot_mover", "extreme_mover"}:
        return 74.0
    return 88.0


def _direction_adjustment(
    direction: str,
    derivatives: dict[str, Any],
    metrics: dict[str, Any],
    entry_precision: float,
    no_chase: float,
    min_gap: float,
) -> float:
    derivative_score = _as_float(derivatives.get("score"))
    adjustment = ((derivative_score if derivative_score is not None else 50.0) - 50.0) * 0.18
    if derivatives.get("bias") in {"long", "short"}:
        bias = str(derivatives["bias"])
        confidence = _as_float(derivatives.get("confidence")) or 0.0
        if bias == direction and confidence >= 64.0:
            adjustment += min(4.0, (confidence - 60.0) * 0.10)
        elif bias != direction and confidence >= 64.0:
            adjustment -= min(6.0, (confidence - 60.0) * 0.14)
    if bool(metrics.get("mover_chase_risk")):
        adjustment -= max(6.0, min_gap * 0.75)
    elif bool(metrics.get("mover_execution_permission")):
        adjustment += 2.0
    if entry_precision < 45.0:
        adjustment -= 5.0
    if no_chase < 40.0:
        adjustment -= 4.0
    return max(-12.0, min(12.0, adjustment))


def _strategy_hint(
    instrument: str,
    mover_profile: str,
    metrics: dict[str, Any],
    derivatives: dict[str, Any],
    entry_precision: float,
    structure_score: float,
    risk_score: float,
    no_chase: float,
) -> str:
    if bool(metrics.get("mover_chase_risk")):
        return "wait_retest_no_chase"
    if mover_profile in {"active_mover", "hot_mover", "extreme_mover"}:
        if bool(metrics.get("mover_execution_permission")) and entry_precision >= 58.0 and risk_score >= 55.0:
            return "hot_mover_structural_retest"
        if derivatives["squeeze_fuel_score"] >= 6.0 and structure_score >= 55.0:
            return "mover_squeeze_reversal"
        return "hot_mover_watch_for_structure"
    if derivatives["squeeze_fuel_score"] >= 6.0 and structure_score >= 58.0:
        return "derivatives_squeeze_reversal"
    if instrument in {"altcoin", "large_altcoin"} and entry_precision >= 65.0:
        return "alt_beta_pullback"
    if structure_score >= 70.0:
        return "precision_ict_setup"
    if no_chase < 55.0:
        return "wait_new_setup"
    return "standard_watch"


def _external_context(report: SymbolReport) -> dict[str, Any]:
    values = _paid_values(report)
    context = values.get("external_strategy_context", {}) if isinstance(values, dict) else {}
    return context if isinstance(context, dict) else {}


def _paid_values(report: SymbolReport) -> dict[str, Any]:
    paid = report.metadata.get("paid_data", {})
    values = paid.get("values", {}) if isinstance(paid, dict) else {}
    return values if isinstance(values, dict) else {}


def _funding_values(values: dict[str, Any], public: dict[str, Any]) -> list[float]:
    output: list[float] = []
    for value in (
        public.get("funding_rate") if isinstance(public, dict) else None,
        values.get("coinglass_funding"),
    ):
        parsed = _as_float(value)
        if parsed is not None:
            output.append(parsed)
    coinalyze = values.get("coinalyze") if isinstance(values.get("coinalyze"), dict) else {}
    if isinstance(coinalyze, dict):
        parsed = _as_float(coinalyze.get("predicted_funding_rate"))
        if parsed is None:
            parsed = _as_float(coinalyze.get("funding_rate"))
        if parsed is not None:
            output.append(parsed)
    return output


def _flow_alignment(values: dict[str, Any], public: dict[str, Any], direction: str) -> float | None:
    for flow in (
        public.get("trade_flow") if isinstance(public, dict) else None,
        values.get("coinglass_taker_buy_sell"),
        (values.get("coinalyze") or {}).get("buy_sell_volume") if isinstance(values.get("coinalyze"), dict) else None,
    ):
        if isinstance(flow, dict):
            ratio = _flow_ratio(flow)
            if ratio is not None:
                return ratio if direction == "long" else 1.0 - ratio
    return None


def _flow_ratio(flow: dict[str, Any]) -> float | None:
    ratio = _as_float(flow.get("taker_buy_ratio"))
    if ratio is not None:
        return max(0.0, min(1.0, ratio))
    buy = _first_float(flow, ("taker_buy_notional", "buy", "buy_volume"))
    sell = _first_float(flow, ("taker_sell_notional", "sell", "sell_volume"))
    if buy is None or sell is None or buy + sell <= 0:
        return None
    return buy / (buy + sell)


def _first_float(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _bucket(buckets: dict[str, float], key: str) -> float:
    return _clamp(float(buckets.get(key, 0.0) or 0.0))


def _weighted(values: tuple[tuple[float, float], ...]) -> float:
    total = sum(weight for _, weight in values) or 1.0
    return _clamp(sum(value * weight for value, weight in values) / total)


def _entry_distance_pct(price: float, entry_zone: tuple[float, float] | None) -> float | None:
    if not entry_zone:
        return None
    low, high = entry_zone
    if low <= price <= high:
        return 0.0
    return min(abs(price - low), abs(price - high)) / max(abs(price), 1e-12) * 100.0


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
