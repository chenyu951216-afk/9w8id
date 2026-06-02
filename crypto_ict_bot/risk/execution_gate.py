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
    return {
        "derivatives_available": external_derivatives_available(report),
        "providers": providers,
        "funding_rate": _as_float(public.get("funding_rate")),
        "open_interest": _as_float(public.get("open_interest")),
        "open_interest_change_pct": _as_float(public.get("open_interest_change_pct")),
        "blocked": blocked,
        "warning": warning,
    }


def external_derivatives_available(report: SymbolReport) -> bool:
    paid = report.metadata.get("paid_data", {})
    providers = paid.get("providers", []) if isinstance(paid, dict) else []
    values = _paid_values(report)
    if isinstance(values.get("exchange_public_derivatives"), dict):
        return True
    provider_text = " ".join(str(provider) for provider in providers).lower()
    return any(name in provider_text for name in ("exchange", "bybit", "binance", "coinglass", "coinalyze"))


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
    if not diag["external_api_ok"]:
        blockers.append("external derivatives OI/Funding 資料不可用")
    if diag["derivative_blocked"]:
        blockers.append(f"funding/OI 過熱：{diag['derivative_warning']}")
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
        "funding/OI 轉為過熱或資料源失效",
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
    score_gap = float(report.metadata.get("score_gap", 0.0) or 0.0)
    blockers = action_blockers(report, limit=None)
    checks = {
        "selected_direction": report.selected_direction != "neutral",
        "score_gap": score_gap >= gate["min_score_gap"],
        "htf_context": diag["htf_context"] >= gate["min_htf_context"],
        "ltf_trigger": diag["ltf_trigger"] >= gate["min_ltf_trigger"],
        "entry_quality": diag["entry_quality"] >= gate["min_entry_quality"],
        "risk_quality": diag["risk_reward_quality"] >= gate["min_risk_quality"],
        "rr": rr >= gate["min_rr"],
        "entry_distance": distance is not None and distance <= dynamic_band,
        "external_derivatives": diag["external_api_ok"],
        "derivatives_not_overheated": not diag["derivative_blocked"],
        "data_completeness": side.data_completeness >= gate["min_data_completeness"],
        "complete_trade_plan": bool(side.entry_zone and side.stop is not None and side.take_profits),
        "direction_not_conflicted": not report.metadata.get("direction_conflict"),
        "execution_score": exec_score >= 82.0,
    }
    hard_gate_ok = all(checks.values())
    if hard_gate_ok:
        code = "market" if distance is not None and distance <= min(0.05, dynamic_band * 0.25) else "limit"
        return {
            "code": code,
            "label": "可以做",
            "reason": (
                f"execution_score={exec_score:.1f}，HTF/LTF/entry/RR/filter 全部過門檻，"
                f"距 entry {distance:.2f}% / dynamic band {dynamic_band:.2f}%"
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
    elif distance is not None and distance > 3.0:
        code, label = "watch", "錯過 / 不追"
        reason = action_blocker_summary(report)
    elif report.selected_direction == "neutral":
        code, label = ("watch", "觀察") if score >= 58 else ("avoid", "不能做")
        reason = f"方向未確認：{action_blocker_summary(report)}"
    elif not diag["external_api_ok"] or side.data_completeness < gate["min_data_completeness"]:
        code, label = "watch", "觀察"
        reason = f"資料不足，只能觀察：{action_blocker_summary(report)}"
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
