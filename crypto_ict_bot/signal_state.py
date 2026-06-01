from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import SymbolReport
from .report import entry_distance_pct, quant_diagnostics, raw_trade_action, selected_side


ACTION_LABELS = {
    "market": "市價做",
    "limit": "限價做",
    "watch": "觀察",
    "avoid": "不能做",
}

ACTION_PRIORITY = {
    "avoid": 0,
    "watch": 1,
    "limit": 2,
    "market": 3,
}

STATE_PATH = Path(os.environ.get("SIGNAL_STATE_PATH", "state/signal_state.json"))


def load_signal_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"symbols": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"symbols": {}}
    except (OSError, json.JSONDecodeError):
        return {"symbols": {}}


def save_signal_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_signal_stability(reports: list[SymbolReport], path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_signal_state(path)
    symbols_state = state.setdefault("symbols", {})
    now = datetime.now(timezone.utc).isoformat()
    for report in reports:
        key = report.symbol
        previous = symbols_state.get(key, {})
        current = _update_report_state(report, previous, now)
        symbols_state[key] = current
    state["updated_at"] = now
    save_signal_state(state, path)
    return state


def _update_report_state(report: SymbolReport, previous: dict[str, Any], now: str) -> dict[str, Any]:
    side = selected_side(report)
    raw = raw_trade_action(report)
    diag = quant_diagnostics(report)
    direction = report.selected_direction
    previous_direction = previous.get("direction")
    same_direction = previous_direction == direction
    previous_smoothed = _as_float(previous.get("smoothed_score"))
    alpha = 0.45
    smoothed = report.score if previous_smoothed is None or not same_direction else previous_smoothed * (1 - alpha) + report.score * alpha

    previous_stable = previous.get("stable_action", "avoid")
    if previous_stable not in ACTION_PRIORITY or not same_direction:
        previous_stable = "avoid"

    raw_code = raw["code"]
    hard_invalid, hard_reason = _hard_invalid(report, raw, diag)
    candidate_ok = _candidate_ok(report, raw, diag, smoothed)
    previous_raw = previous.get("raw_action")
    confirm_count = int(previous.get("confirm_count", 0) or 0)
    fail_count = int(previous.get("fail_count", 0) or 0)

    if hard_invalid:
        stable_code = "avoid"
        confirm_count = 0
        fail_count = 0
        stability_reason = hard_reason
    elif raw_code in {"limit", "market"} and candidate_ok:
        confirm_count = confirm_count + 1 if same_direction and previous_raw == raw_code else 1
        fail_count = 0
        stable_code = previous_stable
        required = 2
        if confirm_count >= required:
            stable_code = raw_code
        elif ACTION_PRIORITY[previous_stable] >= ACTION_PRIORITY[raw_code]:
            stable_code = previous_stable
        else:
            stable_code = "watch"
        stability_reason = f"交易候選需連續 {required} 次確認，目前 {confirm_count}/{required}。"
    elif raw_code == "watch" or report.score >= 58:
        confirm_count = 0
        fail_count = fail_count + 1 if same_direction else 1
        if ACTION_PRIORITY[previous_stable] >= ACTION_PRIORITY["limit"] and fail_count < 2:
            stable_code = previous_stable
            stability_reason = f"原本已有交易計畫，但本次轉弱；等待第 {fail_count}/2 次失效確認，暫不立刻撤銷。"
        else:
            stable_code = "watch"
            stability_reason = "訊號尚未完整，只保留觀察。"
    else:
        confirm_count = 0
        fail_count = fail_count + 1 if same_direction else 1
        if ACTION_PRIORITY[previous_stable] >= ACTION_PRIORITY["limit"] and fail_count < 2:
            stable_code = "watch"
            stability_reason = f"分數跌破觀察線，第 {fail_count}/2 次失效；先降為觀察。"
        else:
            stable_code = "avoid"
            stability_reason = "連續失效或分數太低，取消交易計畫。"

    if stable_code == "market" and raw_code != "market":
        stable_code = "limit"

    stable_reason = _stable_reason(report, raw, diag, stability_reason, stable_code, smoothed)
    plan = _trade_plan(report, stable_code, raw, diag, smoothed)
    behavior = _behavior_analysis(report, raw, diag, smoothed)
    state = {
        "direction": direction,
        "raw_action": raw_code,
        "stable_action": stable_code,
        "smoothed_score": round(smoothed, 2),
        "last_score": report.score,
        "confirm_count": confirm_count,
        "fail_count": fail_count,
        "updated_at": now,
    }
    report.metadata["signal_state"] = {
        **state,
        "stable_action": {
            "code": stable_code,
            "label": ACTION_LABELS[stable_code],
            "reason": stable_reason,
            "entry_distance_pct": entry_distance_pct(report.price, side.entry_zone),
        },
        "raw_action_label": raw["label"],
        "raw_action_reason": raw["reason"],
        "stability_reason": stability_reason,
        "behavior_analysis": behavior,
        "trade_plan": plan,
    }
    return state


def _candidate_ok(report: SymbolReport, raw: dict[str, Any], diag: dict[str, Any], smoothed: float) -> bool:
    side = selected_side(report)
    if raw["code"] not in {"limit", "market"}:
        return False
    if smoothed < 72:
        return False
    if side.data_completeness < 60:
        return False
    if not diag.get("external_api_ok"):
        return False
    if not diag.get("core_ict_ok"):
        return False
    if side.rr is None or side.rr < 1.5:
        return False
    return True


def _hard_invalid(report: SymbolReport, raw: dict[str, Any], diag: dict[str, Any]) -> tuple[bool, str]:
    side = selected_side(report)
    distance = entry_distance_pct(report.price, side.entry_zone)
    if report.selected_direction == "neutral":
        return True, "方向轉為觀望，交易計畫失效。"
    if side.rr is not None and side.rr < 1.2:
        return True, f"RR={side.rr:.2f} 低於最低交易要求。"
    if diag.get("derivative_blocked"):
        return True, f"衍生品資料顯示風險過熱：{diag.get('derivative_warning')}"
    if distance is not None and distance > 7.0:
        return True, f"現價離入場區 {distance:.2f}%，已脫離計畫，不追價。"
    if raw["code"] == "avoid" and report.score < 50:
        return True, raw["reason"]
    return False, ""


def _stable_reason(
    report: SymbolReport,
    raw: dict[str, Any],
    diag: dict[str, Any],
    stability_reason: str,
    stable_code: str,
    smoothed: float,
) -> str:
    if stable_code == "market":
        return f"穩定分數 {smoothed:.1f}，連續確認後達市價條件。{stability_reason}"
    if stable_code == "limit":
        return f"穩定分數 {smoothed:.1f}，核心 ICT 與 API 條件達標；用限價等回補。{stability_reason}"
    if stable_code == "watch":
        return f"穩定分數 {smoothed:.1f}，暫不下單。原始判斷：{raw['label']}；{stability_reason}"
    return f"穩定分數 {smoothed:.1f}，不符合交易條件。原始判斷：{raw['reason']}；{stability_reason}"


def _behavior_analysis(report: SymbolReport, raw: dict[str, Any], diag: dict[str, Any], smoothed: float) -> list[str]:
    side = selected_side(report)
    values = [
        f"目前方向：{report.selected_direction}，原始分數 {report.score:.1f}，平滑分數 {smoothed:.1f}。",
        f"HTF 背景 {diag.get('htf_context', 0):.1f}，LTF 觸發 {diag.get('ltf_trigger', 0):.1f}，入場品質 {diag.get('entry_quality', 0):.1f}。",
        f"風控品質 {diag.get('risk_reward_quality', 0):.1f}，市場/API 品質 {diag.get('market_api_quality', 0):.1f}。",
    ]
    if diag.get("external_api_ok"):
        values.append("衍生品/API 資料已參與判斷。")
    else:
        values.append("尚未讀到衍生品/API 資料，進場狀態會被壓到觀察。")
    if side.rr is not None:
        values.append(f"目前主計畫 RR 約 {side.rr:.2f}R。")
    return values


def _trade_plan(report: SymbolReport, stable_code: str, raw: dict[str, Any], diag: dict[str, Any], smoothed: float) -> list[str]:
    side = selected_side(report)
    plan = [
        f"狀態：{ACTION_LABELS[stable_code]}。原始模型：{raw['label']}；平滑分數：{smoothed:.1f}。",
    ]
    if side.entry_zone:
        plan.append(f"只在入場區 {side.entry_zone[0]:g} - {side.entry_zone[1]:g} 附近執行，不追價。")
    if side.stop is not None:
        plan.append(f"止損放在 {side.stop:g}，若被打到代表結構/流動性假設失效。")
    for tp in side.take_profits:
        plan.append(f"{tp['name']} {tp['price']:g}，約 {tp['rr']:.2f}R，建議出 {tp['portion_pct']:.0f}%。{tp['note']}")
    if stable_code == "market":
        plan.append("執行方式：允許小倉市價，但仍以滑點控制為前提。")
    elif stable_code == "limit":
        plan.append("執行方式：掛限價等回補到 FVG/OTE/POI；沒回補就放棄。")
    elif stable_code == "watch":
        plan.append("執行方式：只觀察，等待連續確認、MSS/BOS、FVG/OTE 重疊或 API 風險改善。")
    else:
        plan.append("執行方式：不交易，等待下一輪掃描重新評估。")
    return plan


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
