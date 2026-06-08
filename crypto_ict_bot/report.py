from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .instrument_classifier import instrument_class, volatility_profile, volatility_state
from .models import DirectionScore, SymbolReport
from .quant_scorecard import build_quant_scorecard
from .reports.schema import complete_symbol_payload
from .risk.execution_gate import (
    ACTION_LABELS as GATE_ACTION_LABELS,
    action_blocker_summary as gate_action_blocker_summary,
    action_blockers as gate_action_blockers,
    core_data_quality as gate_core_data_quality,
    direction_label as gate_direction_label,
    entry_distance_pct as gate_entry_distance_pct,
    execution_gate_profile as gate_execution_gate_profile,
    evaluate_execution_gate,
    quant_diagnostics as gate_quant_diagnostics,
    selected_side as gate_selected_side,
    side_score as gate_side_score,
    visible_warnings as gate_visible_warnings,
)

OPTIONAL_SCORE_FEATURES = {"trendline", "amd", "nexus", "paid_data"}

FEATURE_LABELS = {
    "liquidity_sweep": "流動性掃蕩",
    "htf_poi": "高週期 POI",
    "mss_bos": "MSS/BOS",
    "displacement": "位移",
    "fvg": "FVG",
    "ote": "OTE",
    "risk_reward": "風報比",
    "market_quality": "市場品質",
    "trendline": "趨勢線",
    "amd": "AMD",
    "nexus": "Nexus",
    "paid_data": "外部資料",
}


def direction_label(direction: str) -> str:
    return gate_direction_label(direction)

    return {"long": "看多", "short": "看空", "neutral": "觀望"}.get(direction, direction)


def candidate_status_label(status: str) -> str:
    lifecycle_labels = {
        "SCOUT": "初步追蹤",
        "WATCH": "高品質觀察",
        "ARMED": "接近入場",
        "EXECUTABLE": "可執行",
        "ACTIVE": "已啟動",
        "MANAGE": "管理中",
        "BLOCKED_GOOD_SETUP": "好 setup / 暫禁",
        "MISSED": "錯過入場",
        "INVALID": "失效",
        "EXPIRED": "過期",
    }
    if status in lifecycle_labels:
        return lifecycle_labels[status]
    clean_labels = {
        "new": "新出現",
        "watching": "觀察中",
        "active": "有效",
        "strengthening": "轉強",
        "weakening": "轉弱",
        "warning": "警告",
        "invalid": "失效",
        "expired": "過期",
        "missed": "錯過",
    }
    if status in clean_labels:
        return clean_labels[status]

    labels = {
        "new": "新出現",
        "watching": "觀察中",
        "active": "有效",
        "strengthening": "增強",
        "weakening": "轉弱",
        "warning": "警告",
        "invalid": "失效",
        "expired": "過期",
        "missed": "錯過",
    }
    return labels.get(status, status or "-")


def score_trend_label(trend: str) -> str:
    clean_labels = {
        "new": "新訊號",
        "stable": "穩定",
        "strengthening": "轉強",
        "weakening": "轉弱",
        "strong_jump": "快速轉強",
        "sharp_drop": "快速轉弱",
    }
    if trend in clean_labels:
        return clean_labels[trend]

    labels = {
        "new": "新訊號",
        "stable": "穩定",
        "strengthening": "增強",
        "weakening": "轉弱",
        "strong_jump": "快速轉強",
        "sharp_drop": "快速轉弱",
    }
    return labels.get(trend, trend or "-")


def fmt_price(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:,.2f}"
    if abs(value) >= 1:
        return f"{value:,.4f}"
    return f"{value:,.8f}".rstrip("0").rstrip(".")


def fmt_volume(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def selected_side(report: SymbolReport) -> DirectionScore:
    return gate_selected_side(report)

    if report.selected_direction == "short":
        return report.short
    if report.selected_direction == "neutral" and _side_score(report.short) > _side_score(report.long):
        return report.short
    return report.long


def _side_score(side: DirectionScore) -> float:
    return gate_side_score(side)

    if side.selection_score is not None:
        return side.selection_score
    if side.calibrated_score is not None:
        return side.calibrated_score
    return side.normalized


def _execution_score(side: DirectionScore, fallback: float) -> float:
    if side.execution_score is not None:
        return side.execution_score
    return fallback


def entry_distance_pct(price: float, entry_zone: tuple[float, float] | None) -> float | None:
    return gate_entry_distance_pct(price, entry_zone)

    if not entry_zone:
        return None
    low, high = entry_zone
    if low <= price <= high:
        return 0.0
    return min(abs(price - low), abs(price - high)) / max(abs(price), 1e-12) * 100.0


def _feature_ratio(side: DirectionScore, names: list[str]) -> float:
    total = sum(side.feature_scores.get(name, 0.0) for name in names)
    max_total = sum(side.feature_max_scores.get(name, 0.0) for name in names)
    if max_total <= 0:
        return 0.0
    return max(0.0, min(100.0, total / max_total * 100.0))


def _paid_values(report: SymbolReport) -> dict[str, Any]:
    return report.metadata.get("paid_data", {}).get("values", {})


def _external_api_ok(report: SymbolReport) -> bool:
    providers = set(report.metadata.get("paid_data", {}).get("providers", []))
    values = _paid_values(report)
    if isinstance(values.get("exchange_public_derivatives"), dict):
        return True
    provider_text = " ".join(str(provider) for provider in providers).lower()
    if any(name in provider_text for name in ("exchange", "bybit", "binance", "coinglass", "coinalyze")):
        return True
    return "交易所公開衍生品" in providers or "CoinGlass" in providers or "Coinalyze" in providers


def _derivative_risk(report: SymbolReport, direction: str) -> tuple[bool, str]:
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives", {})
    funding = public.get("funding_rate")
    oi_change = public.get("open_interest_change_pct")
    try:
        funding = float(funding) if funding is not None else None
    except (TypeError, ValueError):
        funding = None
    try:
        oi_change = float(oi_change) if oi_change is not None else None
    except (TypeError, ValueError):
        oi_change = None

    warnings: list[str] = []
    blocked = False
    if funding is not None:
        if direction == "long" and funding > 0.00035:
            blocked = True
            warnings.append(f"funding={funding:.5f}，多方槓桿過熱")
        if direction == "short" and funding < -0.00035:
            blocked = True
            warnings.append(f"funding={funding:.5f}，空方槓桿過熱")
    if oi_change is not None and abs(oi_change) >= 18:
        blocked = True
        warnings.append(f"OI 近 1h 變動 {oi_change:.2f}%，槓桿流動過劇烈")
    return blocked, "；".join(warnings)


ETH_EXECUTION_STATUS_LABELS = {
    "EXECUTABLE_LIMIT": "ETH 可掛限價",
    "ARMED_WAIT_ENTRY": "等待 ETH 入場區",
    "WATCH": "ETH 觀察",
    "ACTIVE": "ETH 持倉管理",
    "PLAN_COMPLETED": "ETH 計畫結束",
    "INVALID": "ETH 無效",
    "DISABLED": "ETH 模式停用",
}


def _eth_analysis_payload(report: SymbolReport) -> dict[str, Any]:
    analysis = report.metadata.get("eth_analysis")
    return analysis if isinstance(analysis, dict) else {}


def _eth_short_mode(analysis: dict[str, Any]) -> dict[str, Any]:
    modes = analysis.get("modes")
    if not isinstance(modes, dict):
        return {}
    mode = modes.get("short_term")
    return mode if isinstance(mode, dict) else {}


def _eth_status_for_state(state: str) -> str:
    return {
        "execute_ready": "EXECUTABLE_LIMIT",
        "armed_wait_entry": "ARMED_WAIT_ENTRY",
        "wait_retest": "WATCH",
        "observe": "WATCH",
        "manage_existing": "ACTIVE",
        "plan_completed": "PLAN_COMPLETED",
        "wait_direction": "INVALID",
        "disabled": "DISABLED",
    }.get(state, "WATCH")


def _eth_trade_action(report: SymbolReport, analysis: dict[str, Any]) -> dict[str, Any]:
    mode = _eth_short_mode(analysis)
    if not mode:
        return evaluate_execution_gate(report)
    state = str(mode.get("state") or "observe")
    status = _eth_status_for_state(state)
    code = "limit" if state == "execute_ready" else "watch"
    if state in {"wait_direction", "disabled"}:
        code = "avoid"
    hard_blockers = [str(item) for item in (mode.get("hard_blockers") or []) if item]
    soft_notes = [str(item) for item in (mode.get("soft_notes") or []) if item]
    reason_source = mode.get("decision") or (hard_blockers[0] if hard_blockers else "ETH strategy is waiting for a cleaner setup.")
    data_stack = analysis.get("data_stack") if isinstance(analysis.get("data_stack"), dict) else {}
    gate_execution = report.metadata.get("gate_execution") if isinstance(report.metadata.get("gate_execution"), dict) else {}
    reason = _eth_gate_reason(str(reason_source), state, gate_execution)
    return {
        "code": code,
        "label": "限價做" if code == "limit" else "觀察" if code == "watch" else "不能做",
        "reason": reason,
        "entry_distance_pct": mode.get("entry_distance_pct"),
        "should_execute": state == "execute_ready",
        "execution_status": status,
        "execution_status_label": ETH_EXECUTION_STATUS_LABELS.get(status, status),
        "gate_version": analysis.get("version") or "eth_unified",
        "entry_origin": "eth_unified_plan",
        "entry_validity": state,
        "blockers": hard_blockers + soft_notes,
        "hard_blockers": hard_blockers,
        "soft_warnings": soft_notes,
        "warnings": soft_notes,
        "blocker_categories": {"eth_unified_strategy": hard_blockers},
        "gate_checks": mode.get("unified_checks", {}),
        "paid_data_status": {
            "configured_api_readiness": data_stack.get("configured_api_readiness", {}),
            "configured_api_ready": True,
            "policy": (mode.get("data_policy") or {}).get("mandatory_paid_provider"),
        },
    }


def _eth_gate_reason(default_reason: str, state: str, gate_execution: dict[str, Any]) -> str:
    if state != "execute_ready" or not gate_execution:
        return default_reason
    action = str(gate_execution.get("action") or "")
    if action == "submitted":
        intent = gate_execution.get("order_intent") if isinstance(gate_execution.get("order_intent"), dict) else {}
        return f"短線 ETH 計畫已送出 Gate 限價單，entry {intent.get('entry_price')}，等待成交後切換持倉管理。"
    if action == "dry_run":
        return "短線 ETH 計畫可執行，但 Gate 目前是乾跑模式，沒有送出實單。"
    if action == "disabled":
        return "短線 ETH 計畫可執行，但 Gate 自動交易開關目前關閉，沒有送出實單。"
    if action == "blocked":
        blockers = gate_execution.get("live_blockers") if isinstance(gate_execution.get("live_blockers"), list) else []
        blocker_text = "；".join(str(item) for item in blockers[:3]) if blockers else str(gate_execution.get("message") or "")
        return f"短線 ETH 計畫可執行，但 Gate live 安全條件未通過：{blocker_text}"
    if action == "error":
        return f"短線 ETH 計畫可執行，但 Gate 送單失敗：{gate_execution.get('message')}"
    return default_reason


def _eth_quant_diagnostics(report: SymbolReport, analysis: dict[str, Any]) -> dict[str, Any]:
    mode = _eth_short_mode(analysis)
    components = mode.get("component_scores", {}) if isinstance(mode.get("component_scores"), dict) else {}
    checks = mode.get("unified_checks", {}) if isinstance(mode.get("unified_checks"), dict) else {}
    data_stack = analysis.get("data_stack", {}) if isinstance(analysis.get("data_stack"), dict) else {}
    quality = float(mode.get("quality_score") or 0.0)
    min_score = float(mode.get("min_score") or 0.0)
    return {
        "htf_context": round(float(components.get("structure") or 0.0), 1),
        "ltf_trigger": round(float(components.get("direction_conviction") or 0.0), 1),
        "entry_quality": round(float(components.get("entry_timing") or 0.0), 1),
        "risk_reward_quality": round(float(components.get("risk_reward") or 0.0), 1),
        "market_api_quality": round(float(components.get("coinglass_orderflow") or 0.0), 1),
        "optional_confluence": round(float(components.get("market_psychology") or 0.0), 1),
        "external_api_ok": bool(data_stack.get("has_public_derivatives") or data_stack.get("has_coinglass")),
        "derivative_blocked": False,
        "derivative_warning": "",
        "core_ict_ok": quality >= min_score,
        "direction_conflict": "" if checks.get("direction_ready", True) else "ETH direction edge is not separated enough.",
        "configured_api_ready": True,
        "configured_api_missing": [],
        "configured_api_advisory_missing": [] if data_stack.get("has_coinglass") else ["coinglass"],
        "configured_api_failed": [],
        "eth_unified_score": quality,
        "eth_min_score": min_score,
        "eth_component_scores": components,
        "eth_checks": checks,
        "paid_api_policy": (mode.get("data_policy") or data_stack.get("paid_api_policy") or {}),
    }


def quant_diagnostics(report: SymbolReport) -> dict[str, Any]:
    eth_analysis = _eth_analysis_payload(report)
    if eth_analysis:
        return _eth_quant_diagnostics(report, eth_analysis)
    return gate_quant_diagnostics(report)

    side = selected_side(report)
    buckets = getattr(side, "bucket_scores", {}) or {}
    htf = buckets.get("htf_context", _feature_ratio(side, ["liquidity_sweep", "htf_poi"]))
    trigger = buckets.get("ltf_confirmation", _feature_ratio(side, ["mss_bos", "displacement"]))
    entry = buckets.get("entry_location", _feature_ratio(side, ["fvg", "ote"]))
    risk = buckets.get("risk_plan", _feature_ratio(side, ["risk_reward"]))
    market = buckets.get("market_filter", _feature_ratio(side, ["market_quality"]))
    optional = _feature_ratio(side, ["trendline", "amd", "nexus", "paid_data"])
    api_ok = _external_api_ok(report)
    derivative_blocked, derivative_warning = _derivative_risk(report, report.selected_direction)
    core_ok = htf >= 60 and trigger >= 65 and entry >= 65 and risk >= 60
    return {
        "htf_context": round(htf, 1),
        "ltf_trigger": round(trigger, 1),
        "entry_quality": round(entry, 1),
        "risk_reward_quality": round(risk, 1),
        "market_api_quality": round(market, 1),
        "optional_confluence": round(optional, 1),
        "external_api_ok": api_ok,
        "derivative_blocked": derivative_blocked,
        "derivative_warning": derivative_warning,
        "core_ict_ok": core_ok,
        "direction_conflict": report.metadata.get("direction_conflict", ""),
    }


def visible_warnings(side: DirectionScore) -> list[str]:
    return gate_visible_warnings(side)

    hidden_fragments = (
        "未納入分母",
        "未觸發",
        "尚未確認 MSS/BOS",
        "沒有明確大實體",
        "找不到方向一致",
        "最近 HTF 沒有清楚",
    )
    output: list[str] = []
    for warning in side.warnings:
        if any(fragment in warning for fragment in hidden_fragments):
            continue
        if warning not in output:
            output.append(warning)
    return output


def _weakest_core_parts(report: SymbolReport, limit: int = 2) -> list[str]:
    diag = quant_diagnostics(report)
    parts = [
        ("HTF 背景", diag["htf_context"], "等高週期掃流動性或 POI 更清楚"),
        ("LTF 觸發", diag["ltf_trigger"], "等 15m MSS/BOS 與位移確認"),
        ("入場品質", diag["entry_quality"], "等 FVG 回補、OTE 或 OB 重疊"),
        ("風控", diag["risk_reward_quality"], "等 RR 改善或價格靠近入場區"),
        ("市場/API", diag["market_api_quality"], "等成交額/BTC 共振或衍生品資料改善"),
    ]
    weak = [item for item in parts if item[1] < 60]
    weak.sort(key=lambda item: item[1])
    return [f"{name} {value:.0f}，{hint}" for name, value, hint in weak[:limit]]


def action_blockers(report: SymbolReport, limit: int = 3) -> list[str]:
    eth_analysis = _eth_analysis_payload(report)
    if eth_analysis:
        mode = _eth_short_mode(eth_analysis)
        blockers = [
            str(item)
            for item in list(mode.get("hard_blockers") or []) + list(mode.get("soft_notes") or [])
            if item
        ]
        if not blockers and mode.get("decision"):
            blockers = [str(mode["decision"])]
        return blockers[:limit]
    return gate_action_blockers(report, limit=limit)

    side = selected_side(report)
    diag = quant_diagnostics(report)
    blockers: list[str] = []
    distance = entry_distance_pct(report.price, side.entry_zone)
    if report.selected_direction == "neutral":
        blockers.append(report.metadata.get("direction_conflict") or "多空方向未明確")
    if not diag["external_api_ok"]:
        blockers.append("尚未讀到衍生品 OI/Funding，先不給進場")
    if diag["derivative_blocked"]:
        blockers.append(f"槓桿風險過熱：{diag['derivative_warning']}")
    blockers.extend(_weakest_core_parts(report, limit=limit))
    if side.rr is not None and side.rr < 1.5:
        blockers.append(f"RR {side.rr:.2f}R 偏低")
    if distance is not None and distance > 1.2:
        blockers.append(f"現價離入場區 {distance:.2f}%，不追價")
    for note in side.signal_notes:
        if len(blockers) >= limit:
            break
        blockers.append(note)
    for adjustment in getattr(side, "score_adjustments", []):
        if len(blockers) >= limit:
            break
        blockers.append(adjustment)
    deduped: list[str] = []
    for item in blockers:
        if item and item not in deduped:
            deduped.append(item)
    return deduped[:limit]


def action_blocker_summary(report: SymbolReport) -> str:
    eth_analysis = _eth_analysis_payload(report)
    if eth_analysis:
        blockers = action_blockers(report, limit=2)
        if blockers:
            return "；".join(blockers)
        mode = _eth_short_mode(eth_analysis)
        return str(mode.get("decision") or "等待 ETH 回到計畫入場區並確認方向。")
    return gate_action_blocker_summary(report)

    blockers = action_blockers(report, limit=2)
    if blockers:
        return "；".join(blockers)
    return "等待下一根確認 K 線與入場區回補"


def score_model_audit(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    bonus_score = float(getattr(side, "bonus_score", 0.0) or 0.0)
    bonus_max = float(getattr(side, "bonus_max_score", 0.0) or 0.0)
    core_raw = max(0.0, side.score - bonus_score)
    core_max = max(0.0, side.max_score)
    core_normalized = core_raw / core_max * 100.0 if core_max else 0.0
    original_score = side.normalized
    calibrated_score = side.calibrated_score if side.calibrated_score is not None else side.normalized
    optional_features = {
        name: {
            "score": side.feature_scores.get(name, 0.0),
            "max": side.feature_max_scores.get(name, 0.0),
        }
        for name in OPTIONAL_SCORE_FEATURES
        if name in side.feature_scores or name in side.feature_max_scores
    }
    inactive_optional = {
        name: reason
        for name, reason in side.inactive_features.items()
        if name in OPTIONAL_SCORE_FEATURES
    }
    skipped_core = {
        name: reason
        for name, reason in side.skipped_features.items()
        if name not in OPTIONAL_SCORE_FEATURES
    }
    providers = report.metadata.get("paid_data", {}).get("providers", [])
    derivative_context = gate_quant_diagnostics(report)
    return {
        "method": "保留原始 ICT/SMC score；另以 signal_state 追蹤分數趨勢、confirm/miss、候選分級與後續 K 線驗證。",
        "core_raw": round(core_raw, 2),
        "core_available_max": round(core_max, 2),
        "core_score": round(max(0.0, min(100.0, core_normalized)), 2),
        "original_score": round(original_score, 2),
        "calibrated_score": round(calibrated_score, 2),
        "selection_score": round(side.selection_score if side.selection_score is not None else calibrated_score, 2),
        "setup_score": round(side.setup_score if side.setup_score is not None else 0.0, 2),
        "execution_score": round(side.execution_score if side.execution_score is not None else calibrated_score, 2),
        "bucket_scores": getattr(side, "bucket_scores", {}),
        "bucket_weights": getattr(side, "bucket_weights", {}),
        "score_adjustments": getattr(side, "score_adjustments", []),
        "validation_adjustments": getattr(side, "validation_adjustments", []),
        "bonus_score": round(bonus_score, 2),
        "bonus_available_max": round(bonus_max, 2),
        "final_score": report.score,
        "data_completeness": round(side.data_completeness, 2),
        "core_data_quality": gate_core_data_quality(report, side),
        "optional_features": optional_features,
        "skipped_core": skipped_core,
        "inactive_optional_not_penalized": inactive_optional,
        "skipped_optional_not_penalized": inactive_optional,
        "external_providers_used": providers,
        "derivatives_context_score": derivative_context.get("derivatives_context_score"),
        "derivatives_context_method": derivative_context.get("derivatives_context_method"),
        "derivatives_context_reason": derivative_context.get("derivatives_context_reason"),
        "paid_api_rule": "只有成功讀到且形成共振的外部資料才加分；讀不到公開衍生品時不扣 ICT 分數，改用流動性/波動替代量化。",
    }


def raw_trade_action(report: SymbolReport) -> dict[str, Any]:
    eth_analysis = _eth_analysis_payload(report)
    if eth_analysis:
        return _eth_trade_action(report, eth_analysis)
    return evaluate_execution_gate(report)

    side = selected_side(report)
    distance = entry_distance_pct(report.price, side.entry_zone)
    rr = side.rr or 0.0
    completeness = side.data_completeness
    score = report.score
    diag = quant_diagnostics(report)
    execution_score = _execution_score(side, score)
    score_gap = float(report.metadata.get("score_gap", 0.0) or 0.0)

    if report.selected_direction == "neutral":
        label = "觀察" if score >= 58 else "不能做"
        return {
            "code": "watch" if score >= 58 else "avoid",
            "label": label,
            "reason": f"方向未確認：{action_blocker_summary(report)}",
            "entry_distance_pct": distance,
        }
    if diag["derivative_blocked"]:
        return {
            "code": "avoid",
            "label": "禁止",
            "reason": f"Funding/OI 過熱：{diag['derivative_warning']}",
            "entry_distance_pct": distance,
        }
    if distance is not None and distance > 5.0:
        return {
            "code": "watch",
            "label": "錯過 / 不追",
            "reason": f"現價距 entry zone {distance:.2f}% > 5%，setup 已過期，禁止追價",
            "entry_distance_pct": distance,
        }
    if distance is not None and distance > 3.0:
        return {
            "code": "watch",
            "label": "錯過 / 不追",
            "reason": f"現價距 entry zone {distance:.2f}% > 3%，只能等下一個結構",
            "entry_distance_pct": distance,
        }
    if not diag["external_api_ok"]:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": "尚未讀到 OI/Funding，不能把它列為可執行，只保留觀察",
            "entry_distance_pct": distance,
        }
    if completeness < 45:
        return {
            "code": "avoid",
            "label": "不能做",
            "reason": f"資料完整度只有 {completeness:.0f}%，不足以實盤執行",
            "entry_distance_pct": distance,
        }
    if not side.entry_zone or side.stop is None or not side.take_profits:
        return {
            "code": "watch",
            "label": "待確認",
            "reason": "缺少完整 entry / stop / take-profit 計畫",
            "entry_distance_pct": distance,
        }
    if rr < 1.8:
        return {
            "code": "watch" if score >= 62 else "avoid",
            "label": "觀察" if score >= 62 else "不能做",
            "reason": f"RR={rr:.2f}R 低於 1.8R，可盯但不能執行",
            "entry_distance_pct": distance,
        }
    hard_gate_ok = (
        execution_score >= 82
        and diag["htf_context"] >= 60
        and diag["ltf_trigger"] >= 65
        and diag["entry_quality"] >= 65
        and diag["risk_reward_quality"] >= 60
        and distance is not None
        and distance <= 0.3
        and rr >= 1.8
        and score_gap >= 8
        and not report.metadata.get("direction_conflict")
    )
    if hard_gate_ok:
        return {
            "code": "market" if distance <= 0.05 else "limit",
            "label": "可以做",
            "reason": f"execution_score={execution_score:.1f}，HTF/LTF/entry/RR/filter 全部過門檻，距 entry {distance:.2f}%",
            "entry_distance_pct": distance,
        }
    if distance is not None and distance <= 0.3 and score >= 62:
        return {
            "code": "watch",
            "label": "待確認",
            "reason": f"到價但確認不足：execution_score={execution_score:.1f}，{action_blocker_summary(report)}",
            "entry_distance_pct": distance,
        }
    if score >= 62:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": f"值得盯但不能做：execution_score={execution_score:.1f}，{action_blocker_summary(report)}",
            "entry_distance_pct": distance,
        }
    return {
        "code": "avoid",
        "label": "不能做",
        "reason": f"selection_score={score:.1f} / execution_score={execution_score:.1f} 不足",
        "entry_distance_pct": distance,
    }

    if report.selected_direction == "neutral":
        if score >= 58 and completeness >= 45:
            return {
                "code": "watch",
                "label": "觀察",
                "reason": f"方向未明確：{action_blocker_summary(report)}。",
                "entry_distance_pct": distance,
            }
        return {
            "code": "avoid",
            "label": "不能做",
            "reason": "方向不明確且分數不足，沒有交易優勢。",
            "entry_distance_pct": distance,
            }

    if not diag["external_api_ok"]:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": "尚未讀到衍生品 API 資料，先不給進場，只保留觀察。",
            "entry_distance_pct": distance,
        }
    if diag["derivative_blocked"]:
        return {
            "code": "avoid",
            "label": "不能做",
            "reason": f"衍生品 API 顯示槓桿風險過高：{diag['derivative_warning']}",
            "entry_distance_pct": distance,
        }
    if completeness < 45:
        return {
            "code": "avoid",
            "label": "不能做",
            "reason": f"資料完整度只有 {completeness:.0f}%，不足以做短線決策。",
            "entry_distance_pct": distance,
        }
    if not diag["core_ict_ok"] and score >= 72:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": f"分數達標但還差確認：{action_blocker_summary(report)}。",
            "entry_distance_pct": distance,
        }
    if not side.entry_zone or side.stop is None or not side.take_profits:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": "缺少完整入場區、止損或止盈計畫，先不下單。",
            "entry_distance_pct": distance,
        }
    if rr < 1.2:
        return {
            "code": "avoid",
            "label": "不能做",
            "reason": f"風報比只有 {rr:.2f}R，短線不划算。",
            "entry_distance_pct": distance,
        }
    if distance is not None and distance > 5.0:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": f"現價離入場區 {distance:.2f}%，不追價，等回補。",
            "entry_distance_pct": distance,
        }
    if score >= 82 and completeness >= 70 and rr >= 1.8 and distance is not None and distance <= 0.18 and diag["core_ict_ok"]:
        return {
            "code": "market",
            "label": "市價做",
            "reason": "高分、高資料完整度、RR 足夠，且現價已在/貼近入場區。",
            "entry_distance_pct": distance,
        }
    if score >= 72 and completeness >= 60 and rr >= 1.5 and diag["core_ict_ok"]:
        reason = "分數達可交易門檻，等價格回到入場區用限價執行。"
        if distance is not None and distance <= 0.18:
            reason = "分數達可交易門檻，現價接近入場區；保守用限價，不追滑點。"
        return {
            "code": "limit",
            "label": "限價做",
            "reason": reason,
            "entry_distance_pct": distance,
        }
    if score >= 58:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": f"只觀察：{action_blocker_summary(report)}。",
            "entry_distance_pct": distance,
        }
    return {
        "code": "avoid",
        "label": "不能做",
        "reason": "分數低於 58，結構與入場條件不足。",
        "entry_distance_pct": distance,
    }


def trade_action(report: SymbolReport) -> dict[str, Any]:
    raw = raw_trade_action(report)
    if _eth_analysis_payload(report):
        return raw
    signal_state = report.metadata.get("signal_state", {})
    stable = signal_state.get("stable_action")
    if isinstance(stable, dict) and stable.get("code"):
        stable_code = str(stable.get("code"))
        if stable_code in {"market", "limit"} and raw["code"] not in {"market", "limit"}:
            return {
                "code": "watch",
                "label": raw["label"] if raw["label"] in {"待確認", "觀察", "錯過 / 不追", "禁止"} else "觀察",
                "reason": f"穩定器暫停執行：{raw['reason']}",
                "entry_distance_pct": raw.get("entry_distance_pct"),
            }
        if stable_code in {"market", "limit", "watch", "avoid"}:
            reason = stable.get("reason") if isinstance(stable.get("reason"), str) else raw["reason"]
            label = stable.get("label") if isinstance(stable.get("label"), str) else None
            return {
                "code": stable_code,
                "label": label or GATE_ACTION_LABELS.get(stable_code, raw["label"]),
                "reason": reason or raw["reason"],
                "entry_distance_pct": stable.get("entry_distance_pct", raw.get("entry_distance_pct")),
            }
    return raw

    signal_state = report.metadata.get("signal_state", {})
    stable = signal_state.get("stable_action")
    if isinstance(stable, dict) and stable.get("code") and stable.get("label"):
        raw = raw_trade_action(report)
        if stable.get("code") in {"market", "limit"} and raw["code"] not in {"market", "limit"}:
            return {
                "code": "watch",
                "label": "觀察",
                "reason": f"原交易計畫轉弱，暫停執行：{raw['reason']}",
                "entry_distance_pct": raw.get("entry_distance_pct", entry_distance_pct(report.price, selected_side(report).entry_zone)),
            }
        reason = stable.get("reason") or raw["reason"]
        if isinstance(reason, str) and reason.startswith("穩定分數"):
            reason = raw["reason"]
        return {
            "code": stable.get("code"),
            "label": stable.get("label"),
            "reason": reason,
            "entry_distance_pct": stable.get("entry_distance_pct", entry_distance_pct(report.price, selected_side(report).entry_zone)),
        }
    return raw_trade_action(report)


def execution_plan(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    action = trade_action(report)
    eth_analysis = _eth_analysis_payload(report)
    eth_mode = _eth_short_mode(eth_analysis) if eth_analysis else {}
    direction = str(eth_analysis.get("direction_label") or direction_label(report.selected_direction)) if eth_analysis else direction_label(report.selected_direction)
    display_entry_zone = eth_mode.get("entry_zone") or side.entry_zone
    display_stop = eth_mode.get("stop", side.stop)
    display_take_profits = eth_mode.get("take_profits") if isinstance(eth_mode.get("take_profits"), list) else side.take_profits
    display_rr = eth_mode.get("rr", side.rr)
    entry = "-"
    if display_entry_zone:
        entry = f"{fmt_price(display_entry_zone[0])} - {fmt_price(display_entry_zone[1])}"
    stop = fmt_price(display_stop)
    rr = "-" if display_rr is None else f"{float(display_rr):.2f}R"
    tp_text = "；".join(
        f"{tp.get('name', 'TP')} {fmt_price(tp.get('price'))} 出 {float(tp.get('portion_pct') or 0):.0f}%"
        for tp in display_take_profits
    ) or "尚未算出止盈"
    should_execute = action["code"] in {"market", "limit"}
    if should_execute:
        label = "可以做"
        mode = "市價" if action["code"] == "market" else "限價"
        summary = f"{direction}，entry {entry}，stop {stop}，TP {tp_text}，RR {rr}。"
    elif action["label"] == "待確認":
        label = "待確認"
        mode = "不下單"
        reason_text = str(action["reason"]).removeprefix("到價但確認不足：")
        summary = f"{direction} 待確認：{reason_text}"
    elif action["label"] == "錯過 / 不追":
        label = "錯過 / 不追"
        mode = "不追價"
        summary = f"{direction} 已遠離 entry：{action['reason']}"
    elif action["label"] == "禁止":
        label = "禁止"
        mode = "禁止"
        summary = f"禁止執行：{action['reason']}"
    elif action["code"] == "watch":
        label = "觀察"
        mode = "不下單"
        summary = f"{direction} 觀察：{action['reason']}"
    else:
        label = "不能做"
        mode = "禁止"
        summary = f"不能做：{action['reason']}"
    steps = [
        f"方向：{direction}",
        f"執行：{mode}",
        f"selection_score：{report.score:.1f}",
        f"setup_score：{side.setup_score if side.setup_score is not None else 0:.1f}",
        f"execution_score：{side.execution_score if side.execution_score is not None else report.score:.1f}",
        f"入場：{entry}",
        f"止損：{stop}",
        f"止盈：{tp_text}",
        f"風報比：{rr}",
        f"原因：{action['reason']}",
    ]
    blockers = action_blockers(report, limit=3)
    if blockers:
        steps.append("等待條件：" + "；".join(blockers))
    return {
        "label": label,
        "mode": mode,
        "summary": summary,
        "steps": steps,
        "should_execute": should_execute,
    }

    side = selected_side(report)
    action = trade_action(report)
    raw = raw_trade_action(report)
    direction = direction_label(report.selected_direction)
    entry = "-"
    if side.entry_zone:
        entry = f"{fmt_price(side.entry_zone[0])} - {fmt_price(side.entry_zone[1])}"
    stop = fmt_price(side.stop)
    rr = "-" if side.rr is None else f"{side.rr:.2f}R"
    tp_text = "；".join(
        f"{tp.get('name', 'TP')} {fmt_price(tp.get('price'))} 出 {float(tp.get('portion_pct') or 0):.0f}%"
        for tp in side.take_profits
    )
    if not tp_text:
        tp_text = "尚未算出止盈"

    if action["code"] in {"market", "limit"}:
        label = "可以做"
        mode = "市價" if action["code"] == "market" else "限價"
        summary = f"{direction}；entry {entry}；stop {stop}；TP {tp_text}；RR {rr}。"
        should_execute = True
    elif action["label"] == "待確認":
        label = "待確認"
        mode = "不下單"
        summary = f"{direction} 到價但缺確認：{action['reason']}"
        should_execute = False
    elif "錯過" in str(action["label"]):
        label = "錯過 / 不追"
        mode = "不追價"
        summary = f"{direction} 已遠離 entry：{action['reason']}"
        should_execute = False
    elif action["label"] == "禁止":
        label = "禁止"
        mode = "禁止"
        summary = f"禁止執行：{action['reason']}"
        should_execute = False
    elif action["code"] == "watch":
        label = "觀察"
        mode = "不下單"
        summary = f"{direction} 觀察：{action['reason']}"
        should_execute = False
    else:
        label = "不能做"
        mode = "禁止"
        summary = f"不能做：{action['reason']}"
        should_execute = False

    steps = [
        f"方向：{direction}",
        f"執行：{mode}",
        f"selection_score：{report.score:.1f}",
        f"setup_score：{side.setup_score if side.setup_score is not None else 0:.1f}",
        f"execution_score：{side.execution_score if side.execution_score is not None else report.score:.1f}",
        f"入場：{entry}",
        f"止損：{stop}",
        f"止盈：{tp_text}",
        f"風報比：{rr}",
        f"原因：{action['reason']}",
    ]
    blockers = action_blockers(report, limit=3)
    if blockers:
        steps.append("等待條件：" + "；".join(blockers))
    return {
        "label": label,
        "mode": mode,
        "summary": summary,
        "steps": steps,
        "should_execute": should_execute,
    }

    if action["code"] == "market":
        label = "可以做"
        mode = "市價"
        summary = f"{direction}，小倉市價或貼近現價執行；止損 {stop}，止盈：{tp_text}，RR {rr}。"
        should_execute = True
    elif action["code"] == "limit":
        label = "可以做"
        mode = "限價"
        summary = f"{direction}，只掛入場區 {entry}，不到價不追；止損 {stop}，止盈：{tp_text}，RR {rr}。"
        should_execute = True
    elif action["code"] == "watch" and raw["code"] in {"market", "limit"}:
        label = "待確認"
        mode = raw["label"]
        summary = f"原始模型是 {raw['label']}，但穩定器尚未連續確認；先盯 {entry}，確認後再照 {mode} 執行。"
        should_execute = False
    elif action["code"] == "watch":
        label = "觀察"
        mode = "不下單"
        summary = f"{direction} 觀察：{action_blocker_summary(report)}。"
        should_execute = False
    else:
        label = "不能做"
        mode = "禁止"
        summary = f"不能做：{action_blocker_summary(report)}。"
        should_execute = False

    steps = [
        f"方向：{direction}",
        f"執行：{mode}",
        f"入場：{entry}",
        f"止損：{stop}",
        f"止盈：{tp_text}",
        f"風報比：{rr}",
        f"原因：{action['reason']}",
    ]
    blockers = action_blockers(report, limit=3)
    if blockers:
        steps.append("等待條件：" + "；".join(blockers))
    return {
        "label": label,
        "mode": mode,
        "summary": summary,
        "steps": steps,
        "should_execute": should_execute,
    }


def raw_trade_action(report: SymbolReport) -> dict[str, Any]:
    eth_analysis = _eth_analysis_payload(report)
    if eth_analysis:
        return _eth_trade_action(report, eth_analysis)
    return evaluate_execution_gate(report)


def trade_action(report: SymbolReport) -> dict[str, Any]:
    raw = raw_trade_action(report)
    if _eth_analysis_payload(report):
        return raw
    signal_state = report.metadata.get("signal_state", {})
    stable = signal_state.get("stable_action") if isinstance(signal_state, dict) else None
    raw_executable = raw.get("code") in {"market", "limit"} and bool(raw.get("should_execute"))
    if raw_executable:
        return raw
    if isinstance(stable, dict) and stable.get("code") in {"market", "limit", "watch", "avoid"}:
        stable_code = str(stable.get("code"))
        if stable_code in {"market", "limit"}:
            return raw
        return {
            **raw,
            "code": stable_code,
            "label": stable.get("label") or GATE_ACTION_LABELS.get(stable_code, raw.get("label")),
            "reason": stable.get("reason") or raw.get("reason"),
            "entry_distance_pct": stable.get("entry_distance_pct", raw.get("entry_distance_pct")),
        }
    return raw


def execution_plan(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    action = trade_action(report)
    eth_analysis = _eth_analysis_payload(report)
    eth_mode = _eth_short_mode(eth_analysis) if eth_analysis else {}
    direction = str(eth_analysis.get("direction_label") or direction_label(report.selected_direction)) if eth_analysis else direction_label(report.selected_direction)
    display_entry_zone = eth_mode.get("entry_zone") or side.entry_zone
    display_stop = eth_mode.get("stop", side.stop)
    display_take_profits = eth_mode.get("take_profits") if isinstance(eth_mode.get("take_profits"), list) else side.take_profits
    display_rr = eth_mode.get("rr", side.rr)
    entry = "-"
    if display_entry_zone:
        entry = f"{fmt_price(display_entry_zone[0])} - {fmt_price(display_entry_zone[1])}"
    stop = fmt_price(display_stop)
    rr = "-" if display_rr is None else f"{float(display_rr):.2f}R"
    tp_text = "；".join(
        f"{tp.get('name', 'TP')} {fmt_price(tp.get('price'))} 出 {float(tp.get('portion_pct') or 0):.0f}%"
        for tp in display_take_profits
    ) or "尚未算出止盈"
    status = action.get("execution_status") or (
        "EXECUTABLE_MARKET" if action.get("code") == "market"
        else "EXECUTABLE_LIMIT" if action.get("code") == "limit"
        else "WATCH"
    )
    should_execute = action.get("code") in {"market", "limit"} and bool(action.get("should_execute"))
    if status == "EXECUTABLE_MARKET":
        label = "可市價"
        mode = "市價"
        summary = f"{direction}，entry {entry}，stop {stop}，TP {tp_text}，RR {rr}。"
    elif status == "EXECUTABLE_LIMIT":
        label = "可掛限價"
        mode = "限價"
        summary = f"{direction}，只掛 entry zone {entry}，stop {stop}，TP {tp_text}，RR {rr}。"
    elif status == "ARMED_WAIT_ENTRY":
        label = "等待入場"
        mode = "等待觸發"
        summary = f"{direction} setup 成立但尚未到 entry band：{action.get('reason')}"
    elif status == "MISSED":
        label = "已錯過"
        mode = "不追價"
        summary = f"{direction} 已離 entry 過遠：{action.get('reason')}"
    elif status == "BLOCKED_RISK":
        label = "風險阻擋"
        mode = "禁止"
        summary = f"風險硬擋：{action.get('reason')}"
    elif status == "INVALID":
        label = "無效"
        mode = "禁止"
        reason_text = str(action.get("reason") or "")
        if "fallback" in reason_text:
            summary = f"無有效結構入場計畫：{reason_text}"
        elif "direction" in reason_text or "分差" in reason_text or report.selected_direction == "neutral":
            summary = f"方向未確認：{reason_text}"
        elif "incomplete trade plan" in reason_text:
            summary = f"交易計畫不完整：{reason_text}"
        else:
            summary = f"資料或計畫未達可執行標準：{reason_text}"
    else:
        label = "觀察"
        mode = "不下單"
        summary = f"{direction} 觀察：{action.get('reason')}"
    steps = [
        f"direction: {direction}",
        f"execution_status: {status}",
        f"entry_origin: {action.get('entry_origin', getattr(side, 'entry_origin', 'unknown'))}",
        f"selection_score: {report.score:.1f}",
        f"setup_score: {side.setup_score if side.setup_score is not None else 0:.1f}",
        f"execution_score: {side.execution_score if side.execution_score is not None else report.score:.1f}",
        f"entry: {entry}",
        f"stop: {stop}",
        f"take_profit: {tp_text}",
        f"RR: {rr}",
        f"reason: {action.get('reason')}",
    ]
    blockers = list(action.get("hard_blockers") or action.get("blockers") or [])
    warnings = list(action.get("soft_warnings") or [])
    if blockers:
        steps.append("hard_blockers: " + "；".join(str(item) for item in blockers[:4]))
    if warnings:
        steps.append("soft_warnings: " + "；".join(str(item) for item in warnings[:4]))
    return {
        "label": label,
        "mode": mode,
        "summary": summary,
        "steps": steps,
        "should_execute": should_execute,
    }


def _apply_eth_payload_overrides(payload: dict[str, Any], analysis: dict[str, Any], raw_action: dict[str, Any]) -> dict[str, Any]:
    mode = _eth_short_mode(analysis)
    if not mode:
        return payload
    take_profits = mode.get("take_profits") if isinstance(mode.get("take_profits"), list) else []
    blockers = [str(item) for item in (mode.get("hard_blockers") or []) if item]
    notes = [str(item) for item in (mode.get("soft_notes") or []) if item]
    payload["direction"] = analysis.get("direction", payload.get("direction"))
    payload["direction_label"] = analysis.get("direction_label", payload.get("direction_label"))
    payload["selected_direction"] = analysis.get("direction", payload.get("selected_direction"))
    payload["entry_zone"] = mode.get("entry_zone", payload.get("entry_zone"))
    payload["stop"] = mode.get("stop", payload.get("stop"))
    payload["take_profits"] = take_profits or payload.get("take_profits", [])
    payload["target"] = take_profits[1].get("price") if len(take_profits) > 1 and isinstance(take_profits[1], dict) else payload.get("target")
    payload["TP1"] = take_profits[0].get("price") if len(take_profits) > 0 and isinstance(take_profits[0], dict) else payload.get("TP1")
    payload["TP2"] = take_profits[1].get("price") if len(take_profits) > 1 and isinstance(take_profits[1], dict) else payload.get("TP2")
    payload["TP3"] = take_profits[2].get("price") if len(take_profits) > 2 and isinstance(take_profits[2], dict) else payload.get("TP3")
    payload["rr"] = mode.get("rr", payload.get("rr"))
    payload["selection_score"] = mode.get("quality_score", payload.get("selection_score"))
    payload["execution_score"] = mode.get("quality_score", payload.get("execution_score"))
    payload["primary_blocker"] = blockers[0] if blockers else str(mode.get("decision") or "")
    payload["secondary_blockers"] = blockers[1:] + notes[:3]
    payload["next_required_condition"] = str(mode.get("decision") or payload.get("next_required_condition") or "")
    payload["no_trade_type"] = mode.get("state", payload.get("no_trade_type"))
    payload["trade_signal_state"] = mode.get("state", payload.get("trade_signal_state"))
    payload["display_reason"] = [str(mode.get("decision") or "")]
    payload["warning_reason"] = notes
    payload["invalid_reason"] = blockers
    payload["gate_checks"] = raw_action.get("gate_checks", payload.get("gate_checks", {}))
    payload["paid_data_status"] = raw_action.get("paid_data_status", payload.get("paid_data_status", {}))
    return payload


def standout_alerts(reports: list[SymbolReport], limit: int = 3) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for report in reports:
        side = selected_side(report)
        action = raw_trade_action(report)
        diag = quant_diagnostics(report)
        distance = entry_distance_pct(report.price, side.entry_zone)
        score_gap = float(report.metadata.get("score_gap", 0.0) or 0.0)
        execution_score = side.execution_score if side.execution_score is not None else report.score
        rr = side.rr or 0.0
        if (
            action["code"] in {"market", "limit"}
            and execution_score >= 88
            and report.score >= 85
            and score_gap >= 12
            and distance is not None
            and distance <= 0.15
            and rr >= 2.0
            and diag["htf_context"] >= 65
            and diag["ltf_trigger"] >= 70
            and diag["entry_quality"] >= 70
            and not diag["derivative_blocked"]
            and not report.metadata.get("direction_conflict")
        ):
            alerts.append(
                {
                    "symbol": report.symbol,
                    "direction": report.selected_direction,
                    "direction_label": direction_label(report.selected_direction),
                    "selection_score": round(report.score, 2),
                    "execution_score": round(execution_score, 2),
                    "entry_distance_pct": round(distance, 4),
                    "entry_zone": side.entry_zone,
                    "stop": side.stop,
                    "take_profits": side.take_profits,
                    "rr": round(rr, 2),
                    "reason": action["reason"],
                }
            )
    alerts.sort(key=lambda item: (item["execution_score"], item["selection_score"], item["rr"]), reverse=True)
    return alerts[:limit]


def report_to_dict(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    raw_action = raw_trade_action(report)
    action = trade_action(report)
    diagnostics = quant_diagnostics(report)
    model_audit = score_model_audit(report)
    execution = execution_plan(report)
    signal_state = report.metadata.get("signal_state", {})
    opportunity = report.metadata.get("opportunity", {}) if isinstance(report.metadata.get("opportunity"), dict) else {}
    direction_analysis = report.metadata.get("direction_analysis", {}) if isinstance(report.metadata.get("direction_analysis"), dict) else {}
    expected_value = report.metadata.get("expected_value", {}) if isinstance(report.metadata.get("expected_value"), dict) else {}
    entry_proximity = report.metadata.get("entry_proximity", {}) if isinstance(report.metadata.get("entry_proximity"), dict) else {}
    layered_analysis = report.metadata.get("layered_analysis", {}) if isinstance(report.metadata.get("layered_analysis"), dict) else {}
    quant_scorecard = report.metadata.get("quant_scorecard", {}) if isinstance(report.metadata.get("quant_scorecard"), dict) else {}
    eth_analysis = report.metadata.get("eth_analysis", {}) if isinstance(report.metadata.get("eth_analysis"), dict) else {}
    gate_execution = report.metadata.get("gate_execution", {}) if isinstance(report.metadata.get("gate_execution"), dict) else {}
    if not quant_scorecard:
        quant_scorecard = build_quant_scorecard(report)
        report.metadata["quant_scorecard"] = quant_scorecard
    prediction_layer = layered_analysis.get("prediction", {}) if isinstance(layered_analysis.get("prediction"), dict) else {}
    setup_layer = layered_analysis.get("setup", {}) if isinstance(layered_analysis.get("setup"), dict) else {}
    trade_plan_layer = layered_analysis.get("trade_plan", {}) if isinstance(layered_analysis.get("trade_plan"), dict) else {}
    execution_layer = layered_analysis.get("execution", {}) if isinstance(layered_analysis.get("execution"), dict) else {}
    no_trade_layer = layered_analysis.get("no_trade", {}) if isinstance(layered_analysis.get("no_trade"), dict) else {}
    candidate_grade = report.metadata.get("candidate_grade") or signal_state.get("priority_level") or report.grade
    candidate_status = report.metadata.get("candidate_status") or signal_state.get("status") or ""
    score_trend = signal_state.get("score_trend") or report.metadata.get("score_trend") or ""
    vol_profile = volatility_profile(report.symbol)
    standard_profile = gate_execution_gate_profile(report)
    eth_short_mode = eth_analysis.get("modes", {}).get("short_term", {}) if isinstance(eth_analysis.get("modes"), dict) else {}
    eth_swing_mode = eth_analysis.get("modes", {}).get("swing", {}) if isinstance(eth_analysis.get("modes"), dict) else {}
    eth_should_execute = bool(eth_short_mode.get("should_open_new"))
    eth_should_notify = bool(eth_analysis.get("discord_notify"))
    metrics = side.market_metrics if isinstance(side.market_metrics, dict) else {}
    atr_pct = metrics.get("atr_pct")
    mover_context = {
        "mover_profile": metrics.get("mover_profile", "normal"),
        "mover_direction": metrics.get("mover_direction", "neutral"),
        "mover_score": metrics.get("mover_score"),
        "mover_same_side": metrics.get("mover_same_side"),
        "mover_execution_permission": metrics.get("mover_execution_permission"),
        "mover_chase_risk": metrics.get("mover_chase_risk"),
        "adaptive_entry_band_pct": metrics.get("adaptive_entry_band_pct"),
        "three_day_range_pct": metrics.get("three_day_range_pct"),
        "three_day_return_pct": metrics.get("three_day_return_pct"),
        "three_day_avg_range_pct": metrics.get("three_day_avg_range_pct"),
        "three_day_avg_tr_pct": metrics.get("three_day_avg_tr_pct"),
        "volatility_expansion_ratio": metrics.get("volatility_expansion_ratio"),
    }
    payload = {
        "symbol": report.symbol,
        "exchange": report.exchange,
        "instrument_class": instrument_class(report.symbol),
        "volatility_state": volatility_state(report.symbol, float(atr_pct)) if isinstance(atr_pct, (int, float)) else "unknown",
        "volatility_profile": {
            "instrument_class": vol_profile.instrument_class,
            "quiet_atr_pct": vol_profile.quiet_atr_pct,
            "active_low_atr_pct": vol_profile.active_low_atr_pct,
            "active_high_atr_pct": vol_profile.active_high_atr_pct,
            "hot_atr_pct": vol_profile.hot_atr_pct,
            "extreme_atr_pct": vol_profile.extreme_atr_pct,
            "entry_band_atr_mult": vol_profile.entry_band_atr_mult,
        },
        "mover_context": mover_context,
        "mover_profile": mover_context["mover_profile"],
        "mover_chase_risk": mover_context["mover_chase_risk"],
        "adaptive_entry_band_pct": mover_context["adaptive_entry_band_pct"],
        "three_day_range_pct": mover_context["three_day_range_pct"],
        "trading_standard_profile": standard_profile,
        "rank": report.metadata.get("opportunity_rank"),
        "opportunity_rank": report.metadata.get("opportunity_rank"),
        "opportunity_score": report.metadata.get("opportunity_score", opportunity.get("opportunity_score", report.score)),
        "raw_opportunity_score": opportunity.get("raw_opportunity_score"),
        "opportunity_readiness_cap": opportunity.get("opportunity_readiness_cap"),
        "execution_quality": report.metadata.get("execution_quality", opportunity.get("execution_quality")),
        "lifecycle_state": candidate_status,
        "state": candidate_status,
        "trade_thesis": opportunity.get("thesis", []),
        "next_trigger": opportunity.get("next_trigger", ""),
        "opportunity_blockers": opportunity.get("blockers", []),
        "strategy_profile": opportunity.get("strategy_profile", {}),
        "strategy_label": opportunity.get("strategy_label", ""),
        "strategy_fit_score": opportunity.get("strategy_fit_score"),
        "risk_notes": opportunity.get("risk_notes", []),
        "failure_conditions": opportunity.get("failure_conditions", []),
        "hard_vetoes": opportunity.get("hard_vetoes", []),
        "soft_penalties": opportunity.get("soft_penalties", {}),
        "market_regime": report.metadata.get("market_regime", {}),
        "market_regime_alignment": opportunity.get("regime_alignment"),
        "relative_strength": report.metadata.get("relative_strength", {}),
        "relative_strength_score": opportunity.get("relative_strength_score"),
        "layered_analysis": layered_analysis,
        "prediction_layer": prediction_layer,
        "setup_layer": setup_layer,
        "trade_plan": trade_plan_layer,
        "execution": execution_layer,
        "data_quality": layered_analysis.get("data_quality", {}),
        "session_context": layered_analysis.get("session_context", {}),
        "prediction_score_long": prediction_layer.get("prediction_score_long"),
        "prediction_score_short": prediction_layer.get("prediction_score_short"),
        "prediction_direction": prediction_layer.get("prediction_direction"),
        "prediction_edge": prediction_layer.get("prediction_edge"),
        "prediction_confidence": prediction_layer.get("prediction_confidence"),
        "prediction_reason": prediction_layer.get("prediction_reason", []),
        "dominant_thesis": prediction_layer.get("dominant_thesis"),
        "fatal_contradiction": prediction_layer.get("fatal_contradiction"),
        "setup_type": setup_layer.get("setup_type"),
        "trade_signal_state": layered_analysis.get("signal_state"),
        "no_trade_type": no_trade_layer.get("no_trade_type", layered_analysis.get("no_trade_type")),
        "primary_blocker": no_trade_layer.get("primary_blocker", layered_analysis.get("primary_blocker")),
        "secondary_blockers": no_trade_layer.get("secondary_blockers", layered_analysis.get("secondary_blockers", [])),
        "next_required_condition": no_trade_layer.get(
            "next_required_condition", layered_analysis.get("next_required_condition", "")
        ),
        "direction_analysis": direction_analysis,
        "direction_conviction": direction_analysis.get("direction_conviction"),
        "direction_edge": direction_analysis.get("direction_edge"),
        "conflict_level": direction_analysis.get("conflict_level"),
        "expected_value": expected_value,
        "expected_R": expected_value.get("expected_R", opportunity.get("expected_R")),
        "estimated_win_probability": expected_value.get(
            "estimated_win_probability", opportunity.get("estimated_win_probability")
        ),
        "entry_proximity": entry_proximity,
        "distance_to_entry": entry_proximity.get("distance_pct"),
        "dynamic_entry_band_pct": entry_proximity.get("dynamic_band_pct"),
        "direction": report.selected_direction,
        "direction_label": direction_label(report.selected_direction),
        "score": report.score,
        "selection_score": side.selection_score if side.selection_score is not None else report.score,
        "setup_score": side.setup_score,
        "execution_score": side.execution_score,
        "grade": report.grade,
        "candidate_grade": candidate_grade,
        "candidate_status": candidate_status,
        "candidate_status_label": candidate_status_label(candidate_status),
        "score_trend": score_trend,
        "score_trend_label": score_trend_label(score_trend),
        "confirm_count": signal_state.get("confirm_count", 0),
        "miss_count": signal_state.get("miss_count", 0),
        "signal_present": signal_state.get("signal_present"),
        "signal_seen_count": signal_state.get("signal_seen_count", 0),
        "signal_absent_count": signal_state.get("signal_absent_count", 0),
        "executable_confirm_count": signal_state.get("executable_confirm_count", 0),
        "executable_miss_count": signal_state.get("executable_miss_count", 0),
        "execution_state": signal_state.get("execution_state"),
        "can_execute_now": signal_state.get("can_execute_now"),
        "previous_score": signal_state.get("previous_score"),
        "score_change": signal_state.get("score_change", 0),
        "highest_score": signal_state.get("highest_score"),
        "lowest_score": signal_state.get("lowest_score"),
        "future_validation": signal_state.get("future_validation", {}),
        "setup_tags": signal_state.get("setup_tags", []),
        "setup_stats": signal_state.get("setup_stats", report.metadata.get("setup_stats", {})),
        "display_reason": signal_state.get("display_reason", []),
        "warning_reason": signal_state.get("warning_reason", []),
        "invalid_reason": signal_state.get("invalid_reason", []),
        "trade_action": action["code"],
        "trade_action_label": action["label"],
        "trade_action_reason": action["reason"],
        "execution_label": execution["label"],
        "execution_mode": execution["mode"],
        "execution_summary": execution["summary"],
        "execution_steps": execution["steps"],
        "should_execute": execution["should_execute"],
        "raw_trade_action": raw_action["code"],
        "raw_trade_action_label": raw_action["label"],
        "raw_trade_action_reason": raw_action["reason"],
        "raw_should_execute": bool(raw_action.get("should_execute")),
        "stable_trade_action": signal_state.get("stable_action", {}).get("code") if isinstance(signal_state.get("stable_action"), dict) else action["code"],
        "stable_trade_action_label": signal_state.get("stable_action", {}).get("label") if isinstance(signal_state.get("stable_action"), dict) else action["label"],
        "final_trade_action": action["code"],
        "final_should_execute": bool(execution["should_execute"]),
        "gate_version": raw_action.get("gate_version"),
        "execution_status": raw_action.get("execution_status"),
        "execution_status_label": raw_action.get("execution_status_label"),
        "hard_blockers": raw_action.get("hard_blockers", []),
        "soft_warnings": raw_action.get("soft_warnings", []),
        "blocker_categories": raw_action.get("blocker_categories", {}),
        "entry_origin": raw_action.get("entry_origin", getattr(side, "entry_origin", "unknown")),
        "entry_validity": raw_action.get("entry_validity", getattr(side, "entry_validity", "unknown")),
        "entry_distance_pct": action["entry_distance_pct"],
        "quant_diagnostics": diagnostics,
        "quant_scorecard": quant_scorecard,
        "eth_analysis": eth_analysis,
        "gate_execution": gate_execution,
        "eth_primary_mode": eth_analysis.get("primary_mode"),
        "eth_primary_plan_state": eth_analysis.get("primary_plan_state"),
        "eth_plan_lifecycle": eth_analysis.get("plan_lifecycle"),
        "eth_short_mode": eth_short_mode,
        "eth_swing_mode": eth_swing_mode,
        "eth_should_execute": eth_should_execute,
        "eth_should_notify": eth_should_notify,
        "score_model": model_audit,
        "signal_state": signal_state,
        "price": report.price,
        "change_pct_24h": report.change_pct_24h,
        "quote_volume_24h": report.quote_volume_24h,
        "data_time": report.data_time.isoformat(),
        "entry_zone": side.entry_zone,
        "stop": side.stop,
        "target": side.target,
        "take_profits": side.take_profits,
        "rr": side.rr,
        "reasons": side.reasons,
        "warnings": visible_warnings(side),
        "signal_notes": side.signal_notes,
        "feature_scores": side.feature_scores,
        "feature_max_scores": side.feature_max_scores,
        "bucket_scores": getattr(side, "bucket_scores", {}),
        "bucket_weights": getattr(side, "bucket_weights", {}),
        "score_adjustments": getattr(side, "score_adjustments", []),
        "validation_adjustments": getattr(side, "validation_adjustments", []),
        "skipped_features": {name: reason for name, reason in side.skipped_features.items() if name not in OPTIONAL_SCORE_FEATURES},
        "inactive_features": side.inactive_features,
        "raw_score": side.score,
        "core_raw_score": model_audit["core_raw"],
        "core_score": model_audit["core_score"],
        "bonus_score": model_audit["bonus_score"],
        "bonus_available_max": model_audit["bonus_available_max"],
        "available_score_max": side.max_score,
        "data_completeness": side.data_completeness,
        "core_data_quality": diagnostics.get("core_data_quality"),
        "long_score": _side_score(report.long),
        "short_score": _side_score(report.short),
        "long_execution_score": report.long.execution_score,
        "short_execution_score": report.short.execution_score,
        "long_data_completeness": report.long.data_completeness,
        "short_data_completeness": report.short.data_completeness,
        "data_coverage": report.data_coverage,
        "missing_data": report.missing_data,
        "metadata": report.metadata,
    }
    completed = complete_symbol_payload(report, payload, raw_action)
    if eth_analysis:
        completed = _apply_eth_payload_overrides(completed, eth_analysis, raw_action)
    return completed


def print_table(reports: list[SymbolReport], limit: int | None = None) -> str:
    rows = reports[:limit] if limit else reports
    headers = [
        "Rank",
        "Symbol",
        "方向",
        "動作",
        "可執行",
        "分數",
        "Grade",
        "Price",
        "24h%",
        "Vol",
        "Entry",
        "Stop",
        "Target",
        "RR",
        "Top reasons",
    ]
    table_rows: list[list[str]] = []
    for idx, report in enumerate(rows, start=1):
        side = selected_side(report)
        eth_short = {}
        eth_analysis = report.metadata.get("eth_analysis") if isinstance(report.metadata.get("eth_analysis"), dict) else {}
        if isinstance(eth_analysis.get("modes"), dict):
            eth_short = eth_analysis["modes"].get("short_term", {}) if isinstance(eth_analysis["modes"].get("short_term"), dict) else {}
        display_entry_zone = eth_short.get("entry_zone") or side.entry_zone
        display_stop = eth_short.get("stop", side.stop)
        display_take_profits = eth_short.get("take_profits") if isinstance(eth_short.get("take_profits"), list) else side.take_profits
        display_target = display_take_profits[1].get("price") if len(display_take_profits or []) > 1 and isinstance(display_take_profits[1], dict) else side.target
        display_rr = eth_short.get("rr", side.rr)
        entry = "-"
        if display_entry_zone:
            entry = f"{fmt_price(display_entry_zone[0])}-{fmt_price(display_entry_zone[1])}"
        reasons = "; ".join(side.reasons[:2]) if report.selected_direction != "neutral" else "分數不足，保持觀望"
        action = trade_action(report)
        execution = execution_plan(report)
        table_rows.append(
            [
                str(idx),
                report.symbol,
                direction_label(report.selected_direction),
                action["label"],
                execution["label"],
                f"{report.score:.1f}",
                report.grade,
                fmt_price(report.price),
                f"{report.change_pct_24h:+.2f}",
                fmt_volume(report.quote_volume_24h),
                entry,
                fmt_price(display_stop),
                fmt_price(display_target),
                "-" if display_rr is None else f"{float(display_rr):.2f}",
                reasons,
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = [" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))]
    lines.append("-+-".join("-" * width for width in widths))
    for row in table_rows:
        lines.append(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)


def write_json(reports: list[SymbolReport], path: str | Path, meta: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output_meta = dict(meta)
    output_meta["standout_alerts"] = output_meta.get("standout_alerts") or standout_alerts(reports)
    payload = {
        "meta": output_meta,
        "reports": [report_to_dict(report) for report in reports],
    }
    _atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2))
    return target


def write_csv(reports: list[SymbolReport], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "symbol",
                "exchange",
                "direction",
                "trade_action",
                "trade_action_reason",
                "execution_label",
                "execution_mode",
                "execution_summary",
                "opportunity_rank",
                "opportunity_score",
                "lifecycle_state",
                "direction_conviction",
                "expected_R",
                "estimated_win_probability",
                "entry_proximity_state",
                "dynamic_entry_band_pct",
                "next_trigger",
                "candidate_grade",
                "candidate_status",
                "candidate_status_label",
                "score_trend",
                "score_trend_label",
                "confirm_count",
                "miss_count",
                "score",
                "grade",
                "price",
                "change_pct_24h",
                "quote_volume_24h",
                "entry_low",
                "entry_high",
                "stop",
                "target",
                "rr",
                "long_score",
                "short_score",
                "data_completeness",
                "available_score_max",
                "reasons",
                "warnings",
                "data_time",
            ]
        )
        for idx, report in enumerate(reports, start=1):
            side = selected_side(report)
            eth_short = {}
            eth_analysis = report.metadata.get("eth_analysis") if isinstance(report.metadata.get("eth_analysis"), dict) else {}
            if isinstance(eth_analysis.get("modes"), dict):
                eth_short = eth_analysis["modes"].get("short_term", {}) if isinstance(eth_analysis["modes"].get("short_term"), dict) else {}
            display_entry_zone = eth_short.get("entry_zone") or side.entry_zone
            display_stop = eth_short.get("stop", side.stop)
            display_take_profits = eth_short.get("take_profits") if isinstance(eth_short.get("take_profits"), list) else side.take_profits
            display_target = display_take_profits[1].get("price") if len(display_take_profits or []) > 1 and isinstance(display_take_profits[1], dict) else side.target
            display_rr = eth_short.get("rr", side.rr)
            signal_state = report.metadata.get("signal_state", {})
            opportunity = report.metadata.get("opportunity", {}) if isinstance(report.metadata.get("opportunity"), dict) else {}
            direction_analysis = report.metadata.get("direction_analysis", {}) if isinstance(report.metadata.get("direction_analysis"), dict) else {}
            expected_value = report.metadata.get("expected_value", {}) if isinstance(report.metadata.get("expected_value"), dict) else {}
            entry_proximity = report.metadata.get("entry_proximity", {}) if isinstance(report.metadata.get("entry_proximity"), dict) else {}
            entry_low, entry_high = display_entry_zone if display_entry_zone else (None, None)
            writer.writerow(
                [
                    idx,
                    report.symbol,
                    report.exchange,
                    report.selected_direction,
                    trade_action(report)["label"],
                    trade_action(report)["reason"],
                    execution_plan(report)["label"],
                    execution_plan(report)["mode"],
                    execution_plan(report)["summary"],
                    report.metadata.get("opportunity_rank", idx),
                    report.metadata.get("opportunity_score", report.score),
                    report.metadata.get("candidate_status", signal_state.get("status", "")),
                    direction_analysis.get("direction_conviction"),
                    expected_value.get("expected_R"),
                    expected_value.get("estimated_win_probability"),
                    entry_proximity.get("state"),
                    entry_proximity.get("dynamic_band_pct"),
                    opportunity.get("next_trigger", ""),
                    report.metadata.get("candidate_grade", signal_state.get("priority_level", report.grade)),
                    report.metadata.get("candidate_status", signal_state.get("status", "")),
                    candidate_status_label(report.metadata.get("candidate_status", signal_state.get("status", ""))),
                    signal_state.get("score_trend", ""),
                    score_trend_label(signal_state.get("score_trend", "")),
                    signal_state.get("confirm_count", 0),
                    signal_state.get("miss_count", 0),
                    report.score,
                    report.grade,
                    report.price,
                    report.change_pct_24h,
                    report.quote_volume_24h,
                    entry_low,
                    entry_high,
                    display_stop,
                    display_target,
                    display_rr,
                    _side_score(report.long),
                    _side_score(report.short),
                    side.data_completeness,
                    side.max_score,
                    " | ".join(side.reasons),
                    " | ".join(visible_warnings(side)),
                    report.data_time.isoformat(),
                ]
            )
    tmp.replace(target)
    return target


def _atomic_write_text(target: Path, text: str) -> None:
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)


def _feature_bars(feature_scores: dict[str, float], feature_max_scores: dict[str, float]) -> str:
    pieces: list[str] = []
    for name, value in feature_scores.items():
        label = html.escape(FEATURE_LABELS.get(name, name.replace("_", " ")))
        max_value = max(feature_max_scores.get(name, 14.0), 1.0)
        pct = max(0.0, min(100.0, value / max_value * 100.0))
        pieces.append(
            f'<div class="feature"><span>{label}</span><div><i style="width:{pct:.1f}%"></i></div><b>{value:.1f}/{max_value:.0f}</b></div>'
        )
    return "".join(pieces)


def write_html(reports: list[SymbolReport], path: str | Path, meta: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    refresh_minutes = max(1, min(5, int(meta.get("refresh_minutes") or 5)))
    refresh_seconds = refresh_minutes * 60
    rows: list[str] = []
    cards: list[str] = []
    alerts = meta.get("standout_alerts") or standout_alerts(reports)
    alert_html = ""
    if alerts:
        alert_items = "".join(
            f"<li><strong>{html.escape(str(item.get('symbol')))}</strong> {html.escape(str(item.get('direction_label')))} "
            f"execution {float(item.get('execution_score') or 0):.1f} / RR {float(item.get('rr') or 0):.2f} / "
            f"距 entry {float(item.get('entry_distance_pct') or 0):.2f}%：{html.escape(str(item.get('reason') or ''))}</li>"
            for item in alerts
        )
        alert_html = f'<section class="standout"><h2>明顯可做提醒</h2><ul>{alert_items}</ul></section>'
    for idx, report in enumerate(reports, start=1):
        side = selected_side(report)
        eth_short = {}
        eth_analysis = report.metadata.get("eth_analysis") if isinstance(report.metadata.get("eth_analysis"), dict) else {}
        if isinstance(eth_analysis.get("modes"), dict):
            eth_short = eth_analysis["modes"].get("short_term", {}) if isinstance(eth_analysis["modes"].get("short_term"), dict) else {}
        display_entry_zone = eth_short.get("entry_zone") or side.entry_zone
        display_stop = eth_short.get("stop", side.stop)
        display_take_profits = eth_short.get("take_profits") if isinstance(eth_short.get("take_profits"), list) else side.take_profits
        display_target = display_take_profits[1].get("price") if len(display_take_profits or []) > 1 and isinstance(display_take_profits[1], dict) else side.target
        display_rr = eth_short.get("rr", side.rr)
        direction = direction_label(report.selected_direction)
        dir_class = report.selected_direction
        action = trade_action(report)
        execution = execution_plan(report)
        signal_state = report.metadata.get("signal_state", {})
        candidate_grade = signal_state.get("priority_level", report.metadata.get("candidate_grade", report.grade))
        candidate_status = signal_state.get("status", "")
        score_trend = signal_state.get("score_trend", "")
        candidate_status_text = candidate_status_label(candidate_status)
        score_trend_text = score_trend_label(score_trend)
        entry = "-"
        if display_entry_zone:
            entry = f"{fmt_price(display_entry_zone[0])} - {fmt_price(display_entry_zone[1])}"
        top_reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in side.reasons[:4]) or "<li>尚無足夠共振，保持觀望。</li>"
        warnings = "".join(f"<li>{html.escape(warning)}</li>" for warning in visible_warnings(side)[:3])
        if not warnings:
            warnings = "<li>無重大資料缺口。</li>"
        blockers = "".join(f"<li>{html.escape(item)}</li>" for item in action_blockers(report, limit=3))
        action_reason = "" if action["reason"] == execution["summary"] else f"<li>{html.escape(action['reason'])}</li>"
        rows.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td><strong>{html.escape(report.symbol)}</strong></td>
              <td><span class="pill {dir_class}">{direction}</span></td>
              <td><span class="pill {html.escape(action['code'])}">{html.escape(action['label'])}</span></td>
              <td>{html.escape(execution['label'])}</td>
              <td>{html.escape(execution['summary'])}</td>
              <td>{html.escape(str(candidate_grade))}</td>
              <td>{html.escape(candidate_status_text)}</td>
              <td>{html.escape(score_trend_text)}</td>
              <td><span class="score">{report.score:.1f}</span></td>
              <td>{report.grade}</td>
              <td>{fmt_price(report.price)}</td>
              <td>{report.change_pct_24h:+.2f}%</td>
              <td>{fmt_volume(report.quote_volume_24h)}</td>
              <td>{entry}</td>
              <td>{fmt_price(display_stop)}</td>
              <td>{fmt_price(display_target)}</td>
              <td>{"-" if display_rr is None else f"{float(display_rr):.2f}"}</td>
            </tr>
            """
        )
        cards.append(
            f"""
            <article class="card">
              <header>
                <div>
                <span class="rank">#{idx}</span>
                <h2>{html.escape(report.symbol)}</h2>
              </div>
                <span class="pill {html.escape(action['code'])}">{html.escape(str(candidate_grade))} / {html.escape(candidate_status_text if candidate_status else str(execution['label']))}</span>
              </header>
              <div class="scorebar"><i style="width:{report.score:.1f}%"></i></div>
              <dl>
                <div><dt>Score</dt><dd>{report.score:.1f} / 100</dd></div>
                <div><dt>執行</dt><dd>{html.escape(execution['label'])} / {html.escape(execution['mode'])}</dd></div>
                <div><dt>Entry</dt><dd>{entry}</dd></div>
                <div><dt>Stop</dt><dd>{fmt_price(display_stop)}</dd></div>
                <div><dt>Target</dt><dd>{fmt_price(display_target)}</dd></div>
                <div><dt>RR</dt><dd>{"-" if display_rr is None else f"{float(display_rr):.2f}"}</dd></div>
              </dl>
              <section>
                <h3>訊號來源</h3>
                <ul>{top_reasons}</ul>
              </section>
              <section>
                <h3>提醒</h3>
                <ul><li>{html.escape(execution['summary'])}</li>{action_reason}{blockers}{warnings}</ul>
              </section>
              <div class="features">{_feature_bars(side.feature_scores, side.feature_max_scores)}</div>
            </article>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ICT 選幣量化評分報告</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #1c2430;
      --muted: #6b7280;
      --line: #d9dee8;
      --long: #0f9f6e;
      --short: #d94f4f;
      --neutral: #64748b;
      --accent: #2f6fdd;
      --warn: #b7791f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Microsoft JhengHei", "Segoe UI", Arial, sans-serif;
      line-height: 1.5;
    }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    .topline {{ display: flex; justify-content: space-between; align-items: end; gap: 18px; margin-bottom: 22px; }}
    h1 {{ margin: 0; font-size: clamp(24px, 3vw, 36px); letter-spacing: 0; }}
    .meta {{ color: var(--muted); font-size: 14px; text-align: right; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 22px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; }}
    .metric strong {{ font-size: 22px; }}
    .standout {{ background: #fff7ed; border: 2px solid #f59e0b; border-radius: 8px; padding: 14px 16px; margin-bottom: 18px; }}
    .standout h2 {{ margin: 0 0 6px; font-size: 20px; color: #9a3412; }}
    .standout li {{ font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; font-size: 14px; white-space: nowrap; }}
    th {{ color: var(--muted); background: #eef2f7; font-weight: 700; }}
    tr:last-child td {{ border-bottom: 0; }}
    .pill {{ display: inline-flex; align-items: center; min-width: 46px; justify-content: center; padding: 3px 8px; border-radius: 999px; color: #fff; font-size: 13px; }}
    .pill.long {{ background: var(--long); }}
    .pill.short {{ background: var(--short); }}
    .pill.neutral {{ background: var(--neutral); }}
    .pill.market {{ background: var(--long); }}
    .pill.limit {{ background: var(--accent); }}
    .pill.watch {{ background: var(--warn); }}
    .pill.avoid {{ background: var(--neutral); }}
    .score {{ font-weight: 800; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 22px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .card header {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }}
    .card h2 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    .rank {{ color: var(--muted); font-weight: 700; }}
    .scorebar {{ height: 9px; border-radius: 999px; background: #e5e9f0; overflow: hidden; margin: 12px 0 14px; }}
    .scorebar i {{ display: block; height: 100%; background: var(--accent); }}
    dl {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin: 0 0 12px; }}
    dl div {{ border: 1px solid var(--line); border-radius: 8px; padding: 8px; min-width: 0; }}
    dt {{ color: var(--muted); font-size: 12px; }}
    dd {{ margin: 2px 0 0; font-weight: 700; overflow-wrap: anywhere; }}
    section h3 {{ margin: 12px 0 4px; font-size: 15px; }}
    ul {{ margin: 0; padding-left: 20px; }}
    li {{ margin: 2px 0; }}
    .features {{ display: grid; gap: 6px; margin-top: 12px; }}
    .feature {{ display: grid; grid-template-columns: 130px 1fr 38px; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); }}
    .feature div {{ height: 6px; background: #e8edf4; border-radius: 999px; overflow: hidden; }}
    .feature i {{ display: block; height: 100%; background: var(--accent); }}
    .note {{ margin-top: 20px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 980px) {{
      main {{ padding: 18px; }}
      .topline {{ display: block; }}
      .meta {{ text-align: left; margin-top: 8px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .tablewrap {{ overflow-x: auto; }}
      .grid {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
<main>
  <div class="topline">
    <div>
      <h1>ICT 選幣量化評分報告</h1>
      <div class="meta">Exchange: {html.escape(str(meta.get("exchange", "-")))} · Generated: {generated} · 本頁會偵測每 {refresh_minutes} 分鐘新報告後更新</div>
    </div>
    <div class="meta">Data source: public exchange REST API · No simulated candles</div>
  </div>
  <section class="summary">
    <div class="metric"><span>掃描幣種</span><strong>{len(reports)}</strong></div>
    <div class="metric"><span>A/B 候選</span><strong>{sum(1 for r in reports if r.metadata.get("signal_state", {}).get("priority_level", r.grade) in {"A", "B"})}</strong></div>
    <div class="metric"><span>看多</span><strong>{sum(1 for r in reports if r.selected_direction == "long")}</strong></div>
    <div class="metric"><span>看空</span><strong>{sum(1 for r in reports if r.selected_direction == "short")}</strong></div>
  </section>
  {alert_html}
  <div class="tablewrap">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Symbol</th><th>方向</th><th>動作</th><th>可執行</th><th>做單方式</th><th>候選</th><th>狀態</th><th>趨勢</th><th>分數</th><th>Grade</th><th>Price</th><th>24h%</th><th>Vol</th><th>Entry</th><th>Stop</th><th>Target</th><th>RR</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
  <section class="grid">
    {''.join(cards)}
  </section>
  <p class="note">這份報告只做選幣與觀察清單評分，不自動下單，也不是投資建議。重大新聞、鏈上資金流、真實委託簿深度若沒有接付費/授權資料，系統會以保守方式處理。</p>
</main>
<script>
  const currentReportGeneratedAt = {json.dumps(str(meta.get("generated_at") or generated))};
  const checkIntervalMs = Math.min({refresh_seconds}, 60) * 1000;
  async function checkReportUpdate() {{
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {{
      const response = await fetch(`/reports/latest.json?poll=${{Date.now()}}`, {{
        cache: "no-store",
        signal: controller.signal,
      }});
      if (!response.ok) return;
      const payload = await response.json();
      const nextGeneratedAt = payload && payload.meta && payload.meta.generated_at;
      if (nextGeneratedAt && nextGeneratedAt !== currentReportGeneratedAt) {{
        window.location.reload();
      }}
    }} catch (_error) {{
      // 掃描中或網路慢時保留目前報告，避免整頁卡在轉圈。
    }} finally {{
      clearTimeout(timeout);
    }}
  }}
  setInterval(checkReportUpdate, checkIntervalMs);
</script>
</body>
</html>"""
    _atomic_write_text(target, html_doc)
    return target
