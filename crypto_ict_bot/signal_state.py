from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import SymbolReport
from .report import entry_distance_pct, quant_diagnostics, raw_trade_action, selected_side, visible_warnings


STATE_PATH = Path(os.environ.get("SIGNAL_STATE_PATH", "state/signal_state.json"))
VALIDATION_STEPS = (1, 3, 6, 12)
HISTORY_LIMIT = 80

ACTION_LABELS = {
    "market": "市價做",
    "limit": "限價做",
    "watch": "觀察",
    "avoid": "不能做",
}

STATUS_LABELS = {
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

TREND_LABELS = {
    "new": "新訊號",
    "stable": "穩定",
    "strengthening": "增強",
    "weakening": "轉弱",
    "strong_jump": "快速轉強",
    "sharp_drop": "快速轉弱",
}

TAG_LABELS = {
    "liquidity_sweep": "Liquidity Sweep",
    "htf_poi": "HTF POI",
    "mss_bos": "MSS/BOS",
    "displacement": "Displacement",
    "fvg": "FVG",
    "ote": "OTE",
    "trendline": "Trendline Break",
    "amd": "AMD",
    "nexus": "Nexus",
    "risk_reward": "Risk Reward",
    "market_quality": "Market Filter",
    "paid_data": "External Data",
}


def load_signal_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"symbols": {}, "statistics": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"symbols": {}, "statistics": {}}
        data.setdefault("symbols", {})
        data.setdefault("statistics", {})
        return data
    except (OSError, json.JSONDecodeError):
        return {"symbols": {}, "statistics": {}}


def save_signal_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def apply_signal_stability(reports: list[SymbolReport], path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_signal_state(path)
    symbols_state = state.setdefault("symbols", {})
    now = datetime.now(timezone.utc)
    seen: set[str] = set()

    for report in reports:
        key = report.symbol
        seen.add(key)
        previous = symbols_state.get(key, {})
        current = _update_report_state(report, previous, now)
        symbols_state[key] = current

    for symbol, previous in list(symbols_state.items()):
        if symbol in seen:
            continue
        current = dict(previous)
        current["miss_count"] = int(current.get("miss_count", 0) or 0) + 1
        if current["miss_count"] >= 6 and current.get("status") not in {"invalid", "expired", "missed"}:
            current["status"] = "expired"
            current.setdefault("invalid_reason", []).append("連續多輪未回到主榜單，移入歷史觀察。")
            current["priority_level"] = "X"
        symbols_state[symbol] = current

    state["updated_at"] = now.isoformat()
    state["statistics"] = build_statistics(symbols_state)
    save_signal_state(state, path)
    return state


def _update_report_state(report: SymbolReport, previous: dict[str, Any], now: datetime) -> dict[str, Any]:
    side = selected_side(report)
    raw = raw_trade_action(report)
    diag = quant_diagnostics(report)
    direction = report.selected_direction
    previous_direction = previous.get("direction")
    same_direction = previous_direction == direction
    previous_score = _as_float(previous.get("current_score"))
    current_score = float(report.score)
    score_change = 0.0 if previous_score is None else current_score - previous_score

    first_seen_time = previous.get("first_seen_time") if same_direction else None
    if not first_seen_time:
        first_seen_time = now.isoformat()
    first_seen_dt = _parse_time(first_seen_time) or now
    signal_age_minutes = max(0.0, (now - first_seen_dt).total_seconds() / 60.0)

    score_history = list(previous.get("score_history", [])) if same_direction else []
    score_history.append(
        {
            "time": now.isoformat(),
            "score": round(current_score, 2),
            "direction": direction,
            "price": report.price,
        }
    )
    score_history = score_history[-HISTORY_LIMIT:]
    score_trend = _score_trend(score_history, score_change)

    confirm_count = int(previous.get("confirm_count", 0) or 0) if same_direction else 0
    miss_count = int(previous.get("miss_count", 0) or 0) if same_direction else 0

    if current_score >= 80:
        confirm_count += 1
        miss_count = 0
        status = "active" if confirm_count >= 2 else "new"
        if confirm_count >= 2 and score_trend == "strengthening":
            status = "strengthening"
    elif current_score >= 70:
        miss_count = 0
        status = "watching"
    elif current_score >= 60:
        confirm_count = 0
        miss_count += 1
        status = "weakening" if score_trend in {"weakening", "sharp_drop"} else "warning"
    else:
        confirm_count = 0
        miss_count += 1
        status = "invalid" if miss_count >= 2 else "warning"

    distance = entry_distance_pct(report.price, side.entry_zone)
    display_reason = _display_reasons(side)
    warning_reason = _warning_reasons(report, side, diag, score_trend, distance)
    invalid_reason = _invalid_reasons(report, side, diag, score_trend, distance, miss_count, signal_age_minutes)
    if invalid_reason and status not in {"new", "active", "strengthening"}:
        status = "invalid" if current_score < 60 or miss_count >= 2 else "warning"
    if distance is not None and distance > 5.0 and status not in {"invalid", "expired"}:
        status = "missed"
        invalid_reason.append(f"價格離 entry zone {distance:.2f}%，已錯過，不追。")
    if signal_age_minutes > 240 and status in {"watching", "warning", "weakening"}:
        status = "expired"
        invalid_reason.append("訊號超過 240 分鐘仍未延續，移入歷史。")

    market_warning = _has_market_warning(diag, warning_reason)
    liquidity_warning = _has_liquidity_warning(report, warning_reason)
    priority_level = _priority_level(current_score, confirm_count, score_trend, status, market_warning, liquidity_warning, distance)
    htf_bias = _htf_bias(report, diag)
    setup_tags = _setup_tags(side)
    future_validation = _update_future_validation(report, previous, now, direction, current_score)
    failed_reason = _failed_reason(report, side, diag, future_validation, invalid_reason, score_trend, distance)
    if failed_reason:
        future_validation["failed_reason"] = failed_reason

    stable_code = _stable_action_code(priority_level, status, raw)
    stable_reason = _stable_reason(priority_level, status, score_trend, confirm_count, miss_count, raw, warning_reason, invalid_reason)
    trade_plan = _manual_review_plan(report, priority_level, status, score_trend, confirm_count, display_reason, warning_reason, invalid_reason)
    behavior_analysis = _behavior_analysis(report, diag, score_trend, score_change, priority_level, status)

    output = {
        "symbol": report.symbol,
        "direction": direction,
        "current_score": round(current_score, 2),
        "previous_score": round(previous_score, 2) if previous_score is not None else None,
        "score_change": round(score_change, 2),
        "highest_score": round(max([item["score"] for item in score_history] or [current_score]), 2),
        "lowest_score": round(min([item["score"] for item in score_history] or [current_score]), 2),
        "score_history": score_history,
        "score_trend": score_trend,
        "confirm_count": confirm_count,
        "miss_count": miss_count,
        "first_seen_time": first_seen_time,
        "last_seen_time": now.isoformat(),
        "signal_age_minutes": round(signal_age_minutes, 1),
        "status": status,
        "priority_level": priority_level,
        "current_price": report.price,
        "entry_zone": side.entry_zone,
        "distance_to_entry_zone": distance,
        "htf_bias": htf_bias,
        "market_filter_result": "warning" if market_warning else "ok",
        "liquidity_filter_result": "warning" if liquidity_warning else "ok",
        "market_warning": market_warning,
        "liquidity_warning": liquidity_warning,
        "setup_tags": setup_tags,
        "display_reason": display_reason,
        "warning_reason": warning_reason,
        "invalid_reason": invalid_reason,
        "future_validation": future_validation,
    }

    report.metadata["signal_state"] = {
        **output,
        "stable_action": {
            "code": stable_code,
            "label": ACTION_LABELS[stable_code],
            "reason": stable_reason,
            "entry_distance_pct": distance,
        },
        "raw_action_label": raw["label"],
        "raw_action_reason": raw["reason"],
        "stability_reason": stable_reason,
        "behavior_analysis": behavior_analysis,
        "trade_plan": trade_plan,
    }
    report.metadata["candidate_grade"] = priority_level
    report.metadata["candidate_status"] = status
    report.metadata["score_trend"] = score_trend
    return output


def _score_trend(score_history: list[dict[str, Any]], score_change: float) -> str:
    if len(score_history) <= 1:
        return "new"
    if score_change >= 10:
        return "strong_jump"
    if score_change <= -10:
        return "sharp_drop"
    if abs(score_change) <= 5:
        tail = [float(item["score"]) for item in score_history[-3:]]
        if len(tail) >= 3 and tail[-1] > tail[-2] > tail[-3]:
            return "strengthening"
        if len(tail) >= 3 and tail[-1] < tail[-2] < tail[-3]:
            return "weakening"
        return "stable"
    return "strengthening" if score_change > 0 else "weakening"


def _display_reasons(side: Any) -> list[str]:
    reasons = list(side.reasons[:6])
    return reasons or ["目前只有初步結構，等待下一輪確認。"]


def _warning_reasons(report: SymbolReport, side: Any, diag: dict[str, Any], score_trend: str, distance: float | None) -> list[str]:
    warnings = list(visible_warnings(side))
    if diag.get("direction_conflict"):
        warnings.append(str(diag["direction_conflict"]))
    if diag.get("market_api_quality", 100) < 45:
        warnings.append("市場/BTC 共振偏弱，降低優先級。")
    if diag.get("derivative_blocked"):
        warnings.append(f"衍生品風險偏高：{diag.get('derivative_warning')}")
    if not diag.get("external_api_ok"):
        warnings.append("未讀到 OI/Funding，僅保留人工觀察，不把它當作進場確認。")
    if score_trend in {"weakening", "sharp_drop"}:
        warnings.append(f"分數趨勢為 {score_trend}，不要追訊號。")
    if distance is not None and distance > 1.2:
        warnings.append(f"現價離 entry zone {distance:.2f}%，可能已經走掉。")
    if report.quote_volume_24h < 20_000_000:
        warnings.append("24h 成交額低於 2,000 萬 USDT，流動性風險較高。")
    return _dedupe(warnings)


def _invalid_reasons(
    report: SymbolReport,
    side: Any,
    diag: dict[str, Any],
    score_trend: str,
    distance: float | None,
    miss_count: int,
    signal_age_minutes: float,
) -> list[str]:
    reasons: list[str] = []
    if report.score < 60 and miss_count >= 2:
        reasons.append("score 連續低於 60，訊號失效。")
    if score_trend == "sharp_drop":
        reasons.append("分數單輪下跌 10 分以上，原結構可能被破壞。")
    if side.rr is not None and side.rr < 1.2:
        reasons.append(f"RR={side.rr:.2f} 低於最低人工觀察要求。")
    if distance is not None and distance > 5.0:
        reasons.append("價格已遠離有效 entry zone，不追。")
    if diag.get("derivative_blocked"):
        reasons.append("OI/Funding 顯示槓桿風險過熱。")
    if signal_age_minutes > 240:
        reasons.append("訊號等待過久仍未延續。")
    return _dedupe(reasons)


def _priority_level(
    score: float,
    confirm_count: int,
    score_trend: str,
    status: str,
    market_warning: bool,
    liquidity_warning: bool,
    distance: float | None,
) -> str:
    if status in {"invalid", "expired", "missed"} or (score < 60 and confirm_count == 0):
        return "X"
    distance_ok = distance is None or distance <= 1.2
    if score >= 85 and confirm_count >= 2 and score_trend not in {"weakening", "sharp_drop"} and not market_warning and distance_ok:
        level = "A"
    elif score >= 75 and confirm_count >= 1:
        level = "B"
    elif score >= 65:
        level = "C"
    else:
        level = "D"
    if score_trend in {"weakening", "sharp_drop"} or market_warning or liquidity_warning or not distance_ok:
        level = _downgrade(level)
    return level


def _downgrade(level: str) -> str:
    order = ["A", "B", "C", "D", "X"]
    idx = order.index(level)
    return order[min(idx + 1, len(order) - 1)]


def _status_label(value: str) -> str:
    return STATUS_LABELS.get(value, value or "-")


def _trend_label(value: str) -> str:
    return TREND_LABELS.get(value, value or "-")


def _stable_action_code(priority_level: str, status: str, raw: dict[str, Any]) -> str:
    if priority_level == "X" or status in {"invalid", "expired", "missed"}:
        return "avoid"
    if priority_level in {"A", "B"} and raw.get("code") in {"market", "limit"}:
        return raw["code"]
    return "watch"


def _stable_reason(
    priority_level: str,
    status: str,
    score_trend: str,
    confirm_count: int,
    miss_count: int,
    raw: dict[str, Any],
    warnings: list[str],
    invalids: list[str],
) -> str:
    status_text = _status_label(status)
    trend_text = _trend_label(score_trend)
    if priority_level == "A":
        return f"A 級候選：{status_text} / {trend_text}，confirm={confirm_count}，優先人工看圖。"
    if priority_level == "B":
        return f"B 級候選：{status_text} / {trend_text}，confirm={confirm_count}，等待最後確認。"
    if priority_level == "C":
        return f"C 級觀察：結構有雛形，但條件未完整。原始模型：{raw.get('label')}。"
    if priority_level == "D":
        reason = "；".join(warnings[:2]) or f"miss={miss_count}，訊號正在轉弱。"
        return f"D 級低優先：{reason}"
    reason = "；".join(invalids[:2]) or raw.get("reason") or "訊號失效。"
    return f"X 級失效：{reason}"


def _manual_review_plan(
    report: SymbolReport,
    priority_level: str,
    status: str,
    score_trend: str,
    confirm_count: int,
    display: list[str],
    warnings: list[str],
    invalids: list[str],
) -> list[str]:
    side = selected_side(report)
    direction_text = {"long": "看多", "short": "看空", "neutral": "觀望"}.get(report.selected_direction, report.selected_direction)
    plan = [
        f"{priority_level} 級 / {_status_label(status)} / {_trend_label(score_trend)}：{direction_text}候選，這是人工看盤清單，不是自動下單。",
        f"目前 score {report.score:.1f}，confirm={confirm_count}。",
    ]
    if side.entry_zone:
        plan.append(f"人工看圖重點：價格是否仍在 entry zone 附近 {side.entry_zone[0]:g} - {side.entry_zone[1]:g}。")
    plan.extend([f"理由：{item}" for item in display[:3]])
    plan.extend([f"警告：{item}" for item in warnings[:3]])
    plan.extend([f"失效：{item}" for item in invalids[:3]])
    return _dedupe(plan)


def _behavior_analysis(report: SymbolReport, diag: dict[str, Any], score_trend: str, score_change: float, priority_level: str, status: str) -> list[str]:
    return [
        f"原始 score={report.score:.1f}，本輪變化 {score_change:+.1f}，趨勢={_trend_label(score_trend)}。",
        f"候選分級={priority_level}，狀態={_status_label(status)}，用於排序人工看圖優先順序。",
        f"HTF={diag.get('htf_context', 0):.1f}，LTF={diag.get('ltf_trigger', 0):.1f}，入場品質={diag.get('entry_quality', 0):.1f}，風控={diag.get('risk_reward_quality', 0):.1f}。",
    ]


def _update_future_validation(
    report: SymbolReport,
    previous: dict[str, Any],
    now: datetime,
    direction: str,
    current_score: float,
) -> dict[str, Any]:
    previous_validation = previous.get("future_validation") if previous.get("direction") == direction else None
    if isinstance(previous_validation, dict) and previous_validation.get("signal_price"):
        validation = dict(previous_validation)
    else:
        validation = {
            "checked": False,
            "signal_time": now.isoformat(),
            "signal_price": report.price,
            "direction": direction,
            "score_at_signal": round(current_score, 2),
            "price_after_1_candle": None,
            "price_after_3_candles": None,
            "price_after_6_candles": None,
            "price_after_12_candles": None,
            "after_1_candle": None,
            "after_3_candles": None,
            "after_6_candles": None,
            "after_12_candles": None,
            "max_favorable_move": 0.0,
            "max_adverse_move": 0.0,
            "direction_correct": None,
            "failed_reason": "",
        }

    signal_price = float(validation.get("signal_price") or report.price)
    signal_time = _parse_time(validation.get("signal_time")) or now
    elapsed_candles = int(max(0.0, (now - signal_time).total_seconds()) // 300)
    move = _directional_move_pct(direction, signal_price, report.price)
    validation["max_favorable_move"] = round(max(float(validation.get("max_favorable_move") or 0.0), max(move, 0.0)), 4)
    validation["max_adverse_move"] = round(max(float(validation.get("max_adverse_move") or 0.0), max(-move, 0.0)), 4)

    for step in VALIDATION_STEPS:
        key = _validation_key(step)
        price_key = f"price_after_{step}_candles" if step > 1 else "price_after_1_candle"
        if elapsed_candles >= step and validation.get(key) is None:
            result = {
                "price": report.price,
                "move_pct": round(move, 4),
                "direction_correct": move > 0,
                "checked_at": now.isoformat(),
            }
            validation[key] = result
            validation[price_key] = report.price

    for step in reversed(VALIDATION_STEPS):
        result = validation.get(_validation_key(step))
        if isinstance(result, dict):
            validation["checked"] = True
            validation["direction_correct"] = result.get("direction_correct")
            break
    return validation


def _directional_move_pct(direction: str, start: float, current: float) -> float:
    if start <= 0:
        return 0.0
    raw = (current - start) / start * 100.0
    return raw if direction == "long" else -raw


def _validation_key(step: int) -> str:
    return "after_1_candle" if step == 1 else f"after_{step}_candles"


def _failed_reason(
    report: SymbolReport,
    side: Any,
    diag: dict[str, Any],
    validation: dict[str, Any],
    invalids: list[str],
    score_trend: str,
    distance: float | None,
) -> str:
    reasons = list(invalids)
    if validation.get("direction_correct") is False:
        reasons.append("後續 K 線方向與原判斷相反。")
    if diag.get("market_api_quality", 100) < 45:
        reasons.append("BTC/市場共振不足。")
    if diag.get("htf_context", 100) < 45:
        reasons.append("HTF bias 與短線訊號不夠一致。")
    if diag.get("ltf_trigger", 100) < 45:
        reasons.append("掃流動性後沒有足夠 MSS/BOS 確認。")
    if diag.get("entry_quality", 100) < 45:
        reasons.append("FVG/OTE/POI 入場位置不夠乾淨。")
    if score_trend == "sharp_drop":
        reasons.append("訊號分數快速轉弱。")
    if distance is not None and distance > 1.2:
        reasons.append("價格離 entry zone 過遠。")
    if report.quote_volume_24h < 20_000_000:
        reasons.append("幣種流動性偏差。")
    if side.rr is not None and side.rr < 1.5:
        reasons.append("RR 不足。")
    return "；".join(_dedupe(reasons)[:4])


def _setup_tags(side: Any) -> list[str]:
    tags = []
    for key, value in side.feature_scores.items():
        if value > 0:
            tags.append(TAG_LABELS.get(key, key))
    return _dedupe(tags)


def _htf_bias(report: SymbolReport, diag: dict[str, Any]) -> str:
    if report.selected_direction == "neutral":
        return "neutral"
    return f"{report.selected_direction} / HTF {diag.get('htf_context', 0):.1f}"


def _has_market_warning(diag: dict[str, Any], warnings: list[str]) -> bool:
    return bool(diag.get("derivative_blocked")) or diag.get("market_api_quality", 100) < 45 or any("市場" in item or "BTC" in item for item in warnings)


def _has_liquidity_warning(report: SymbolReport, warnings: list[str]) -> bool:
    return report.quote_volume_24h < 20_000_000 or any("流動性" in item or "成交額" in item or "entry zone" in item for item in warnings)


def build_statistics(symbols_state: dict[str, Any]) -> dict[str, Any]:
    records = [item for item in symbols_state.values() if isinstance(item, dict)]
    total = len(records)
    active = [item for item in records if item.get("priority_level") in {"A", "B", "C", "D"}]
    failed = [item for item in records if item.get("priority_level") == "X" or item.get("status") in {"invalid", "expired", "missed"}]
    stats = {
        "total_signals": total,
        "active_signals": len(active),
        "failed_signals": len(failed),
        "long_signals": sum(1 for item in records if item.get("direction") == "long"),
        "short_signals": sum(1 for item in records if item.get("direction") == "short"),
        "grade_counts": _count_by(records, "priority_level"),
        "status_counts": _count_by(records, "status"),
        "accuracy": {},
        "average_mfe": _avg([_as_float(item.get("future_validation", {}).get("max_favorable_move")) for item in records]),
        "average_mae": _avg([_as_float(item.get("future_validation", {}).get("max_adverse_move")) for item in records]),
        "setup_tag_stats": _setup_tag_stats(records),
        "weakness_stats": {
            "sharp_drop_failed": _rate([item for item in records if item.get("score_trend") == "sharp_drop"], failed_status=True),
            "confirm_insufficient_failed": _rate([item for item in records if int(item.get("confirm_count", 0) or 0) < 2], failed_status=True),
            "market_warning_failed": _rate([item for item in records if item.get("market_warning")], failed_status=True),
        },
    }
    for step in VALIDATION_STEPS:
        checked = []
        for item in records:
            validation = item.get("future_validation", {})
            result = validation.get(_validation_key(step))
            if isinstance(result, dict) and result.get("direction_correct") is not None:
                checked.append(bool(result.get("direction_correct")))
        stats["accuracy"][f"after_{step}_candles"] = _bool_rate(checked)
    return stats


def _setup_tag_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in records:
        validation = item.get("future_validation", {})
        correct = validation.get("direction_correct")
        for tag in item.get("setup_tags", []):
            bucket = output.setdefault(tag, {"total": 0, "correct": 0, "failed": 0, "accuracy": None})
            bucket["total"] += 1
            if correct is True:
                bucket["correct"] += 1
            if correct is False or item.get("priority_level") == "X":
                bucket["failed"] += 1
    for bucket in output.values():
        denominator = bucket["correct"] + bucket["failed"]
        bucket["accuracy"] = round(bucket["correct"] / denominator * 100.0, 1) if denominator else None
    return output


def _count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in records:
        value = str(item.get(key) or "-")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _bool_rate(values: list[bool]) -> dict[str, Any]:
    if not values:
        return {"checked": 0, "correct": 0, "rate": None}
    correct = sum(1 for value in values if value)
    return {"checked": len(values), "correct": correct, "rate": round(correct / len(values) * 100.0, 1)}


def _rate(records: list[dict[str, Any]], failed_status: bool = False) -> dict[str, Any]:
    if not records:
        return {"total": 0, "failed": 0, "rate": None}
    if failed_status:
        failed = sum(1 for item in records if item.get("priority_level") == "X" or item.get("status") in {"invalid", "expired", "missed"})
    else:
        failed = 0
    return {"total": len(records), "failed": failed, "rate": round(failed / len(records) * 100.0, 1)}


def _avg(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value and value not in output:
            output.append(value)
    return output
