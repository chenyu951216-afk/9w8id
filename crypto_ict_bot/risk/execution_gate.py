from __future__ import annotations

from typing import Any

from ..models import DirectionScore, SymbolReport


EXECUTION_GATE = {
    "min_score_gap": 8.0,
    "min_htf_context": 60.0,
    "min_ltf_trigger": 65.0,
    "min_entry_quality": 65.0,
    "min_risk_quality": 60.0,
    "min_rr": 1.8,
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

ACTION_LABELS = {
    "market": "可以做",
    "limit": "可以做",
    "watch": "觀察",
    "avoid": "不能做",
}

OPTIONAL_SCORE_FEATURES = {"trendline", "amd", "nexus", "paid_data"}


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
    return round(max(EXECUTION_GATE["max_entry_distance_pct"], atr_pct * 0.35, spread_pct * 3.0), 4)


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


def visible_warnings(side: DirectionScore) -> list[str]:
    output: list[str] = []
    for warning in side.warnings:
        if warning and warning not in output:
            output.append(warning)
    return output


def _feature_ratio(side: DirectionScore, names: list[str]) -> float:
    total = sum(side.feature_scores.get(name, 0.0) for name in names)
    max_total = sum(side.feature_max_scores.get(name, 0.0) for name in names)
    if max_total <= 0:
        return 0.0
    return max(0.0, min(100.0, total / max_total * 100.0))


def _paid_values(report: SymbolReport) -> dict[str, Any]:
    paid = report.metadata.get("paid_data", {})
    values = paid.get("values", {}) if isinstance(paid, dict) else {}
    return values if isinstance(values, dict) else {}


def paid_data_status(report: SymbolReport) -> dict[str, Any]:
    paid = report.metadata.get("paid_data", {})
    providers = paid.get("providers", []) if isinstance(paid, dict) else []
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives", {})
    if not isinstance(public, dict):
        public = {}
    blocked, warning = derivative_risk(report, report.selected_direction)
    context = derivatives_context(report)
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
        "configured_failed": list(readiness.get("configured_failed", []))
        if isinstance(readiness.get("configured_failed"), list)
        else [],
        "status": readiness.get("status", {}) if isinstance(readiness.get("status"), dict) else {},
    }


def derivative_risk(report: SymbolReport, direction: str) -> tuple[bool, str]:
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


def market_risk(report: SymbolReport) -> tuple[bool, str]:
    side = selected_side(report)
    metrics = side.market_metrics or {}
    warnings: list[str] = []
    atr_pct = _as_float(metrics.get("atr_pct"))
    volume_ratio = _as_float(metrics.get("volume_ratio"))
    btc_fast_pct = _as_float(metrics.get("btc_fast_pct"))
    if atr_pct is not None and atr_pct > 4.0:
        warnings.append(f"ATR%={atr_pct:.2f} 過熱，禁止追價")
    if volume_ratio is not None and volume_ratio > 5.5:
        warnings.append(f"volume spike={volume_ratio:.2f} 過熱，避免追高/追空")
    if btc_fast_pct is not None and abs(btc_fast_pct) >= 2.2:
        warnings.append(f"BTC 近 4H 快速波動 {btc_fast_pct:.2f}%，alt 禁止追價")
    return bool(warnings), "；".join(warnings)


def derivatives_context(report: SymbolReport) -> dict[str, Any]:
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
    if volume >= 500_000_000:
        volume_score = 92.0
    elif volume >= 250_000_000:
        volume_score = 84.0
    elif volume >= 80_000_000:
        volume_score = 74.0
    elif volume >= 20_000_000:
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
    elif 0.18 <= atr_pct <= 2.4:
        atr_score = 82.0
    elif 0.12 <= atr_pct <= 4.0:
        atr_score = 65.0
    else:
        atr_score = 50.0

    volume_ratio = _as_float(side.market_metrics.get("volume_ratio"))
    if volume_ratio is None:
        flow_score = 70.0
    elif volume_ratio <= 3.5:
        flow_score = 78.0
    elif volume_ratio <= 5.5:
        flow_score = 62.0
    else:
        flow_score = 45.0

    score = volume_score * 0.42 + spread_score * 0.20 + atr_score * 0.23 + flow_score * 0.15
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
    api_readiness = configured_api_readiness(report)
    gate = EXECUTION_GATE
    return {
        "htf_context": round(float(htf), 1),
        "ltf_trigger": round(float(trigger), 1),
        "entry_quality": round(float(entry), 1),
        "risk_reward_quality": round(float(risk), 1),
        "market_api_quality": round(float(market), 1),
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
        "configured_api_ready": api_readiness["execution_ready"],
        "configured_api_missing": api_readiness["execution_missing"],
        "configured_api_failed": api_readiness["configured_failed"],
        "core_ict_ok": htf >= gate["min_htf_context"]
        and trigger >= gate["min_ltf_trigger"]
        and entry >= gate["min_entry_quality"]
        and risk >= gate["min_risk_quality"],
        "direction_conflict": report.metadata.get("direction_conflict", ""),
    }


def action_blockers(report: SymbolReport, limit: int | None = 3) -> list[str]:
    side = selected_side(report)
    diag = quant_diagnostics(report)
    gate = EXECUTION_GATE
    distance = entry_distance_pct(report.price, side.entry_zone)
    proximity = entry_proximity_state(report, side)
    dynamic_band = proximity["dynamic_band_pct"]
    rr = side.rr or 0.0
    score_gap = float(report.metadata.get("score_gap", 0.0) or 0.0)
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
    if not side.entry_zone or side.stop is None or not side.take_profits:
        blockers.append("entry / stop / TP 計畫不完整")
    if rr < gate["min_rr"]:
        blockers.append(f"RR {rr:.2f}R < {gate['min_rr']:.1f}R")
    if distance is None:
        blockers.append("尚無有效 entry zone，不能執行")
    elif distance > 5.0:
        blockers.append(f"距 entry {distance:.2f}% > 5%，標記錯過/過期")
    elif distance > 3.0:
        blockers.append(f"距 entry {distance:.2f}% > 3%，禁止顯示可以做")
    elif distance > 1.2:
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
    if side.data_completeness < gate["min_data_completeness"]:
        blockers.append(f"資料完整度 {side.data_completeness:.0f}% < {gate['min_data_completeness']:.0f}%")
    if report.metadata.get("direction_conflict"):
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
    if side.rr is not None and side.rr < EXECUTION_GATE["min_rr"]:
        conditions.append(f"RR 降到 {EXECUTION_GATE['min_rr']:.1f}R 以下")
    return conditions


def evaluate_execution_gate(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    diag = quant_diagnostics(report)
    gate = EXECUTION_GATE
    distance = entry_distance_pct(report.price, side.entry_zone)
    proximity = entry_proximity_state(report, side)
    dynamic_band = proximity["dynamic_band_pct"]
    rr = side.rr or 0.0
    score = side_score(side)
    exec_score = execution_score(side, score)
    setup_score = float(side.setup_score if side.setup_score is not None else score)
    score_gap = float(report.metadata.get("score_gap", 0.0) or 0.0)
    blockers = action_blockers(report, limit=None)
    common_checks = {
        "selected_direction": report.selected_direction != "neutral",
        "score_gap": score_gap >= gate["min_score_gap"],
        "htf_context": diag["htf_context"] >= gate["min_htf_context"],
        "risk_quality": diag["risk_reward_quality"] >= gate["min_risk_quality"],
        "rr": rr >= gate["min_rr"],
        "entry_distance": distance is not None and distance <= dynamic_band,
        "external_derivatives": diag["external_api_ok"],
        "derivatives_context": diag["derivatives_context_score"] >= gate["min_derivatives_context"],
        "configured_api_ready": diag.get("configured_api_ready", True),
        "derivatives_not_overheated": not diag["derivative_blocked"],
        "market_not_overheated": not diag["market_overheated"],
        "data_completeness": side.data_completeness >= gate["min_data_completeness"],
        "complete_trade_plan": bool(side.entry_zone and side.stop is not None and side.take_profits),
        "direction_not_conflicted": not report.metadata.get("direction_conflict"),
    }
    market_checks = {
        **common_checks,
        "ltf_trigger": diag["ltf_trigger"] >= gate["min_ltf_trigger"],
        "entry_quality": diag["entry_quality"] >= gate["min_entry_quality"],
        "execution_score": exec_score >= 82.0,
    }
    limit_checks = {
        **common_checks,
        "selection_score": score >= gate["limit_min_selection_score"],
        "setup_score": setup_score >= gate["limit_min_setup_score"],
        "ltf_trigger": diag["ltf_trigger"] >= gate["limit_min_ltf_trigger"],
        "entry_quality": diag["entry_quality"] >= gate["limit_min_entry_quality"],
        "execution_score": exec_score >= gate["limit_min_execution_score"],
    }
    market_ready = all(value for key, value in market_checks.items() if key != "external_derivatives")
    limit_ready = all(value for key, value in limit_checks.items() if key != "external_derivatives")
    checks = {
        **market_checks,
        "market_ready": market_ready,
        "limit_ready": limit_ready,
        "limit_selection_score": limit_checks["selection_score"],
        "limit_setup_score": limit_checks["setup_score"],
        "limit_ltf_trigger": limit_checks["ltf_trigger"],
        "limit_entry_quality": limit_checks["entry_quality"],
        "limit_execution_score": limit_checks["execution_score"],
    }
    if market_ready or limit_ready:
        code = "market" if market_ready and distance is not None and distance <= min(0.05, dynamic_band * 0.25) else "limit"
        trigger_text = "market gate passed" if code == "market" else "limit gate passed; final LTF trigger still must be watched before fill"
        return {
            "code": code,
            "label": "可以做",
            "reason": (
                f"execution_score={exec_score:.1f}，HTF/LTF/entry/RR/filter 全部過門檻，"
                f"距 entry {distance:.2f}% / dynamic band {dynamic_band:.2f}%"
            ),
            "reason": (
                f"execution_score={exec_score:.1f}, setup_score={setup_score:.1f}, {trigger_text}; "
                f"distance {distance:.2f}% / dynamic band {dynamic_band:.2f}%"
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
        }
    if diag["derivative_blocked"]:
        code, label = "avoid", "禁止"
        reason = f"funding/OI 過熱：{diag['derivative_warning']}"
    elif diag["market_overheated"]:
        code, label = "watch", "錯過 / 不追"
        reason = f"市場過熱，禁止追價：{diag['market_warning']}"
    elif distance is not None and distance > 3.0:
        code, label = "watch", "錯過 / 不追"
        reason = action_blocker_summary(report)
    elif report.selected_direction == "neutral":
        code, label = ("watch", "觀察") if score >= 58 else ("avoid", "不能做")
        reason = f"方向未確認：{action_blocker_summary(report)}"
    elif not diag.get("configured_api_ready", True):
        code, label = "watch", "觀察"
        missing = ", ".join(diag.get("configured_api_missing", [])) or "configured API"
        reason = f"已設定 API 尚未完整讀取，不能標記可執行：{missing}"
    elif side.data_completeness < gate["min_data_completeness"]:
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
