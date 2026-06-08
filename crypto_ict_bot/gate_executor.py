from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import SymbolReport


GATE_EXECUTOR_VERSION = "gate_eth_executor_2026_06_v1"
GATE_LIVE_CONFIRMATION = "I_UNDERSTAND_GATE_200X_ETH_RISK"
GATE_PROD_BASE_URL = "https://fx-api.gateio.ws/api/v4"
GATE_TESTNET_BASE_URL = "https://fx-api-testnet.gateio.ws/api/v4"
GATE_STATE_PATH = Path("state/gate_eth_trade_state.json")
GATE_ORDER_TEXT_PREFIX = "t-eth-"
GATE_ENTRY_TEXT = "t-eth-entry"
GATE_SL_TEXT = "t-eth-sl"
GATE_BE_TEXT = "t-eth-be"
GATE_TRAIL_TEXT = "t-eth-trail"
GATE_PENDING_ENTRY_TTL_SECONDS = 1800


class GateExecutionError(RuntimeError):
    pass


def attach_gate_execution(reports: list[SymbolReport], config: dict[str, Any]) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    for report in reports:
        if report.symbol.upper() != _eth_symbol(config):
            continue
        result = build_gate_execution(report, config)
        report.metadata["gate_execution"] = result
        outputs.append(result)
    return {
        "version": GATE_EXECUTOR_VERSION,
        "symbol": _eth_symbol(config),
        "results": outputs,
        "primary": outputs[0] if outputs else {},
    }


def build_gate_execution(
    report: SymbolReport,
    config: dict[str, Any],
    state_path: Path = GATE_STATE_PATH,
) -> dict[str, Any]:
    settings = _settings(config)
    trade_state = _load_gate_state(state_path)
    contract_info = fetch_gate_contract_info(config)
    if settings["contract_size_eth"] <= 0 and contract_info.get("contract_size_eth"):
        settings["contract_size_eth"] = float(contract_info["contract_size_eth"])
    analysis = report.metadata.get("eth_analysis") if isinstance(report.metadata.get("eth_analysis"), dict) else {}
    short_mode = analysis.get("modes", {}).get("short_term", {}) if isinstance(analysis.get("modes"), dict) else {}
    intent = _build_order_intent(report, short_mode, settings)
    entry_guard = _entry_guard(settings, trade_state, intent)
    if entry_guard.get("state_changed"):
        _save_gate_state(state_path, trade_state)
    live_blockers = _live_blockers(settings, intent, short_mode) + entry_guard["blockers"]
    base_result = {
        "version": GATE_EXECUTOR_VERSION,
        "enabled": settings["enabled"],
        "dry_run": settings["dry_run"],
        "testnet": settings["testnet"],
        "base_url": settings["base_url"],
        "contract": settings["contract"],
        "leverage": settings["leverage"],
        "margin_usdt": settings["margin_usdt"],
        "max_notional_usdt": settings["max_notional_usdt"],
        "contract_info": contract_info,
        "place_exit_orders": settings["place_exit_orders"],
        "mode_state": short_mode.get("state", "missing_eth_plan"),
        "order_intent": intent,
        "entry_guard": entry_guard,
        "trade_state": _compact_gate_state(trade_state),
        "live_blockers": live_blockers,
        "submitted": False,
        "response": None,
    }
    if not settings["enabled"]:
        return {**base_result, "action": "disabled", "message": "Gate 自動交易停用；只顯示 ETH 下單意圖。"}
    if settings["dry_run"]:
        return {**base_result, "action": "dry_run", "message": "Gate 乾跑模式：不送出實單。"}
    if live_blockers:
        return {**base_result, "action": "blocked", "message": "Gate live 下單被安全條件擋下。"}

    try:
        if settings["set_leverage_before_order"]:
            _set_leverage(settings)
        response = _submit_order(settings, intent["request_body"])
        exit_responses = _submit_exit_orders(settings, intent) if settings["place_exit_orders"] else []
    except Exception as exc:
        return {**base_result, "action": "error", "message": f"Gate submit failed: {type(exc).__name__}: {exc}"}
    _record_pending_entry(trade_state, intent, response)
    _save_gate_state(state_path, trade_state)
    return {
        **base_result,
        "action": "submitted",
        "submitted": True,
        "response": response,
        "exit_order_responses": exit_responses,
        "trade_state": _compact_gate_state(trade_state),
        "message": "Gate order submitted.",
    }


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    gate = config.get("gate_trading", {}) if isinstance(config.get("gate_trading"), dict) else {}
    testnet = bool(gate.get("testnet", True))
    return {
        "enabled": bool(gate.get("enabled", False)),
        "dry_run": bool(gate.get("dry_run", True)),
        "testnet": testnet,
        "base_url": GATE_TESTNET_BASE_URL if testnet else GATE_PROD_BASE_URL,
        "api_key": str(gate.get("api_key") or ""),
        "api_secret": str(gate.get("api_secret") or ""),
        "settle": str(gate.get("settle") or "usdt").lower(),
        "contract": str(gate.get("contract") or "ETH_USDT").upper(),
        "leverage": max(1, min(200, int(gate.get("leverage") or 200))),
        "margin_usdt": max(0.0, float(gate.get("margin_usdt") or 15.0)),
        "max_notional_usdt": max(0.0, float(gate.get("max_notional_usdt") or 3000.0)),
        "contract_size_eth": max(0.0, float(gate.get("contract_size_eth") or 0.0)),
        "set_leverage_before_order": bool(gate.get("set_leverage_before_order", True)),
        "place_exit_orders": bool(gate.get("place_exit_orders", False)),
        "exit_order_expiration_seconds": max(60, min(604800, int(gate.get("exit_order_expiration_seconds") or 86400))),
        "position_management_enabled": bool(gate.get("position_management_enabled", True)),
        "position_poll_seconds": max(3, min(60, int(gate.get("position_poll_seconds") or 10))),
        "break_even_enabled": bool(gate.get("break_even_enabled", True)),
        "break_even_trigger_r": max(0.2, min(5.0, float(gate.get("break_even_trigger_r") or 1.0))),
        "break_even_lock_r": max(0.0, min(1.0, float(gate.get("break_even_lock_r") or 0.05))),
        "trailing_stop_enabled": bool(gate.get("trailing_stop_enabled", True)),
        "trailing_trigger_r": max(0.5, min(10.0, float(gate.get("trailing_trigger_r") or 1.8))),
        "trailing_distance_r": max(0.1, min(5.0, float(gate.get("trailing_distance_r") or 0.75))),
        "cancel_bot_orders_when_flat": bool(gate.get("cancel_bot_orders_when_flat", True)),
        "confirm_live_trading": str(gate.get("confirm_live_trading") or ""),
    }


def _build_order_intent(report: SymbolReport, mode: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    direction = str(mode.get("direction") or report.selected_direction or "neutral")
    entry_zone = mode.get("entry_zone") if isinstance(mode.get("entry_zone"), (list, tuple)) else None
    entry_price = _entry_price(entry_zone, direction, report.price)
    notional = min(settings["margin_usdt"] * settings["leverage"], settings["max_notional_usdt"])
    quantity_eth = notional / max(float(entry_price), 1e-12)
    contract_size = settings["contract_size_eth"]
    signed_size: str | None = None
    if contract_size > 0:
        raw_contracts = max(1, math.floor(quantity_eth / contract_size))
        signed_contracts = raw_contracts if direction == "long" else -raw_contracts if direction == "short" else 0
        signed_size = str(signed_contracts)
    request_body = {
        "contract": settings["contract"],
        "size": int(signed_size) if signed_size not in {None, "0", 0} else signed_size,
        "price": _price_text(entry_price),
        "tif": "gtc",
        "reduce_only": False,
        "text": GATE_ENTRY_TEXT,
    }
    return {
        "ready": direction in {"long", "short"} and mode.get("state") == "execute_ready",
        "direction": direction,
        "order_type": "limit",
        "entry_price": entry_price,
        "entry_zone": entry_zone,
        "notional_usdt": round(notional, 4),
        "quantity_eth": round(quantity_eth, 8),
        "contract_size_eth": contract_size,
        "signed_size": signed_size,
        "request_body": request_body,
        "stop": mode.get("stop"),
        "take_profits": mode.get("take_profits", []),
        "close_size_sign": "-" if direction == "long" else "+" if direction == "short" else "",
        "plan_id": mode.get("plan_id"),
    }


def _entry_price(entry_zone: Any, direction: str, fallback: float) -> float:
    if isinstance(entry_zone, (list, tuple)) and len(entry_zone) >= 2:
        low, high = sorted((float(entry_zone[0]), float(entry_zone[1])))
        if direction == "long":
            return (low + high) / 2.0
        if direction == "short":
            return (low + high) / 2.0
    return float(fallback)


def _live_blockers(settings: dict[str, Any], intent: dict[str, Any], mode: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not intent.get("ready"):
        blockers.append("ETH 短線模式尚未進入 execute_ready。")
    if settings["confirm_live_trading"] != GATE_LIVE_CONFIRMATION:
        blockers.append("confirm_live_trading 未填入 live 風險確認字串。")
    if not settings["api_key"] or not settings["api_secret"]:
        blockers.append("缺少 Gate API key/secret。")
    if intent.get("signed_size") in {None, "0", 0}:
        blockers.append("contract_size_eth 未設定，無法把 15U*200x 換算成 Gate 合約 size。")
    if float(settings["margin_usdt"]) > 15.0:
        blockers.append("margin_usdt 超過 15U 安全上限。")
    if int(settings["leverage"]) > 200:
        blockers.append("leverage 超過 200x 安全上限。")
    if not settings["place_exit_orders"]:
        blockers.append("GATE_PLACE_EXIT_ORDERS must be true so live 200x ETH entries have TP/SL protection.")
    if mode.get("state") == "manage_existing":
        blockers.append("已有同方向計畫，不能重複開新倉。")
    return blockers


def _entry_guard(settings: dict[str, Any], state: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    state_changed = False
    now = time.time()
    pending = state.get("pending_entry_order") if isinstance(state.get("pending_entry_order"), dict) else {}
    if pending:
        submitted_at = _as_float(pending.get("submitted_ts")) or 0.0
        age = now - submitted_at if submitted_at else 0.0
        if submitted_at and age <= GATE_PENDING_ENTRY_TTL_SECONDS:
            blockers.append("已有 Gate ETH entry 掛單等待成交；不重複送新單。")
        else:
            state.pop("pending_entry_order", None)
            state_changed = True
            warnings.append("舊 pending entry 超過生命週期，已解除本地鎖定。")

    if not settings["enabled"] or settings["dry_run"] or not settings["api_key"] or not settings["api_secret"]:
        return {
            "blockers": blockers,
            "warnings": warnings,
            "state_changed": state_changed,
            "checked_position": False,
            "checked_open_orders": False,
        }

    checked_position = False
    checked_open_orders = False
    try:
        position_payload = _position_payload(settings, {"ok": True, "contract": settings["contract"]})
        checked_position = True
        position = position_payload.get("position") if isinstance(position_payload.get("position"), dict) else {}
        if position.get("has_position"):
            blockers.append("Gate 已有 ETH 實際持倉；進入持倉管理，不重複開倉。")
    except Exception as exc:
        warnings.append(f"Gate position pre-check failed: {type(exc).__name__}: {exc}")

    try:
        open_orders = _list_open_orders(settings)
        checked_open_orders = True
        entry_orders = _bot_entry_orders(open_orders)
        if entry_orders:
            blockers.append("Gate 已有 t-eth-entry 開倉掛單；不重複送新單。")
        elif pending and not blockers:
            state.pop("pending_entry_order", None)
            state_changed = True
            warnings.append("Gate 已無 t-eth-entry 開倉掛單，已清掉本地 pending entry。")
    except Exception as exc:
        warnings.append(f"Gate open-order pre-check failed: {type(exc).__name__}: {exc}")
        if pending:
            blockers.append("讀不到 Gate 開倉掛單清單且本地有 pending entry；為避免重複下單，本輪不送新單。")

    return {
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": warnings,
        "state_changed": state_changed,
        "checked_position": checked_position,
        "checked_open_orders": checked_open_orders,
    }


def _record_pending_entry(state: dict[str, Any], intent: dict[str, Any], response: Any) -> None:
    response_id = None
    if isinstance(response, dict):
        response_id = response.get("id") or response.get("order_id")
    state["pending_entry_order"] = {
        "status": "submitted",
        "order_id": response_id,
        "direction": intent.get("direction"),
        "entry_price": intent.get("entry_price"),
        "signed_size": intent.get("signed_size"),
        "plan_id": intent.get("plan_id"),
        "plan": {
            "direction": intent.get("direction"),
            "entry_zone": intent.get("entry_zone"),
            "entry_basis_for_rr": intent.get("entry_price"),
            "stop": intent.get("stop"),
            "take_profits": intent.get("take_profits", []),
            "plan_id": intent.get("plan_id"),
            "source": "gate_pending_entry",
        },
        "submitted_ts": time.time(),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _submit_order(settings: dict[str, Any], body: dict[str, Any]) -> Any:
    path = f"/futures/{settings['settle']}/orders"
    clean_body = {key: value for key, value in body.items() if value is not None}
    return _gate_request(settings, "POST", path, clean_body)


def _submit_exit_orders(settings: dict[str, Any], intent: dict[str, Any]) -> list[dict[str, Any]]:
    signed_size = int(intent.get("signed_size") or 0)
    if signed_size == 0:
        return []
    direction = str(intent.get("direction") or "")
    close_sign = -1 if signed_size > 0 else 1
    total_contracts = abs(signed_size)
    responses: list[dict[str, Any]] = []
    take_profits = [tp for tp in (intent.get("take_profits") or []) if isinstance(tp, dict)]
    contract_slices = _contract_slices(
        total_contracts,
        [max(0.0, min(100.0, _as_float(tp.get("portion_pct")) or 0.0)) for tp in take_profits],
    )
    for idx, (take_profit, close_contracts) in enumerate(zip(take_profits, contract_slices), start=1):
        if not isinstance(take_profit, dict):
            continue
        price = _as_float(take_profit.get("price"))
        if price is None or close_contracts <= 0:
            continue
        body = _price_order_body(
            settings,
            close_size=close_sign * close_contracts,
            trigger_price=price,
            rule=_tp_rule(direction),
            text=f"t-eth-tp{idx}",
        )
        responses.append({"type": f"TP{idx}", "request": body, "response": _submit_price_order(settings, body)})
    stop = _as_float(intent.get("stop"))
    if stop is not None:
        body = _price_order_body(
            settings,
            close_size=close_sign * total_contracts,
            trigger_price=stop,
            rule=_sl_rule(direction),
            text="t-eth-sl",
        )
        responses.append({"type": "SL", "request": body, "response": _submit_price_order(settings, body)})
    return responses


def _price_order_body(
    settings: dict[str, Any],
    close_size: int,
    trigger_price: float,
    rule: int,
    text: str,
) -> dict[str, Any]:
    return {
        "initial": {
            "contract": settings["contract"],
            "size": int(close_size),
            "price": "0",
            "tif": "ioc",
            "reduce_only": True,
            "text": text,
        },
        "trigger": {
            "strategy_type": 0,
            "price_type": 1,
            "price": _price_text(trigger_price),
            "rule": rule,
            "expiration": settings["exit_order_expiration_seconds"],
        },
    }


def _submit_price_order(settings: dict[str, Any], body: dict[str, Any]) -> Any:
    path = f"/futures/{settings['settle']}/price_orders"
    return _gate_request(settings, "POST", path, body)


def _list_price_orders(settings: dict[str, Any]) -> list[dict[str, Any]]:
    path = f"/futures/{settings['settle']}/price_orders"
    payload = _gate_request(settings, "GET", path, body=None, query_params={"status": "open"})
    if isinstance(payload, list):
        return [order for order in payload if isinstance(order, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("orders"), list):
        return [order for order in payload["orders"] if isinstance(order, dict)]
    return []


def _list_open_orders(settings: dict[str, Any]) -> list[dict[str, Any]]:
    path = f"/futures/{settings['settle']}/orders"
    payload = _gate_request(
        settings,
        "GET",
        path,
        body=None,
        query_params={"contract": settings["contract"], "status": "open"},
    )
    if isinstance(payload, list):
        return [order for order in payload if isinstance(order, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("orders"), list):
        return [order for order in payload["orders"] if isinstance(order, dict)]
    return []


def _bot_entry_orders(open_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [order for order in open_orders if str(order.get("text") or "") == GATE_ENTRY_TEXT]


def _cancel_price_order(settings: dict[str, Any], order_id: Any) -> Any:
    path = f"/futures/{settings['settle']}/price_orders/{order_id}"
    return _gate_request(settings, "DELETE", path, body=None)


def _submit_take_profit_orders(
    settings: dict[str, Any],
    direction: str,
    contracts_abs: int,
    take_profits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    close_sign = -1 if direction == "long" else 1
    contract_slices = _contract_slices(
        contracts_abs,
        [max(0.0, min(100.0, _as_float(tp.get("portion_pct")) or 0.0)) for tp in take_profits],
    )
    responses: list[dict[str, Any]] = []
    for idx, (take_profit, close_contracts) in enumerate(zip(take_profits, contract_slices), start=1):
        price = _as_float(take_profit.get("price"))
        if price is None or close_contracts <= 0:
            continue
        body = _price_order_body(
            settings,
            close_size=close_sign * close_contracts,
            trigger_price=price,
            rule=_tp_rule(direction),
            text=f"t-eth-tp{idx}",
        )
        responses.append({"type": f"TP{idx}", "request": body, "response": _submit_price_order(settings, body)})
    return responses


def _submit_stop_order(
    settings: dict[str, Any],
    direction: str,
    contracts_abs: int,
    stop_price: float,
    text: str,
) -> dict[str, Any]:
    close_size = -contracts_abs if direction == "long" else contracts_abs
    body = _price_order_body(
        settings,
        close_size=close_size,
        trigger_price=stop_price,
        rule=_sl_rule(direction),
        text=text,
    )
    return {"type": "SL", "request": body, "response": _submit_price_order(settings, body)}


def _bot_price_orders(raw_orders: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for order in raw_orders:
        initial = order.get("initial") if isinstance(order.get("initial"), dict) else {}
        contract = order.get("contract") or initial.get("contract")
        text = _order_text(order)
        if str(contract or "").upper() == settings["contract"] and text.startswith(GATE_ORDER_TEXT_PREFIX):
            output.append(order)
    return output


def _compact_price_order(order: dict[str, Any]) -> dict[str, Any]:
    initial = order.get("initial") if isinstance(order.get("initial"), dict) else {}
    trigger = order.get("trigger") if isinstance(order.get("trigger"), dict) else {}
    return {
        "id": order.get("id") or order.get("order_id"),
        "text": _order_text(order),
        "contract": order.get("contract") or initial.get("contract"),
        "status": order.get("status"),
        "close_size": _as_float(initial.get("size") if initial else order.get("size")),
        "trigger_price": _as_float(trigger.get("price") if trigger else order.get("trigger_price")),
        "rule": trigger.get("rule") if trigger else order.get("rule"),
    }


def _order_text(order: dict[str, Any]) -> str:
    initial = order.get("initial") if isinstance(order.get("initial"), dict) else {}
    return str(initial.get("text") or order.get("text") or "")


def _stop_orders(bot_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [order for order in bot_orders if order.get("text") in {GATE_SL_TEXT, GATE_BE_TEXT, GATE_TRAIL_TEXT}]


def _tp_orders(bot_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [order for order in bot_orders if str(order.get("text") or "").startswith("t-eth-tp")]


def _current_stop_from_orders(bot_orders: list[dict[str, Any]], direction: str) -> float | None:
    stops = [_as_float(order.get("trigger_price")) for order in _stop_orders(bot_orders)]
    stops = [price for price in stops if price is not None]
    if not stops:
        return None
    return max(stops) if direction == "long" else min(stops)


def _cancel_bot_orders(settings: dict[str, Any], bot_orders: list[dict[str, Any]], live_ready: bool) -> list[dict[str, Any]]:
    actions = []
    for order in bot_orders:
        action = {
            "type": "cancel_bot_price_order",
            "order_id": order.get("id"),
            "text": order.get("text"),
            "mode": "live" if live_ready else "dry_run",
        }
        if live_ready and order.get("id") is not None:
            try:
                action["response"] = _cancel_price_order(settings, order["id"])
            except Exception as exc:
                action["error"] = f"{type(exc).__name__}: {exc}"
        actions.append(action)
    return actions


def _manage_open_position(
    settings: dict[str, Any],
    position: dict[str, Any],
    bot_orders: list[dict[str, Any]],
    eth_analysis: dict[str, Any],
    state: dict[str, Any],
    live_ready: bool,
    now_text: str,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    direction = str(position.get("direction") or "flat")
    entry_price = _as_float(position.get("entry_price"))
    mark_price = _as_float(position.get("mark_price")) or entry_price
    signed_contracts = int(_as_float(position.get("contracts")) or 0)
    contracts_abs = abs(signed_contracts)
    if direction not in {"long", "short"} or entry_price is None or mark_price is None or contracts_abs <= 0:
        actions.append({"type": "position_unmanageable", "message": "Gate position is missing direction, entry, mark, or size."})
        return actions

    plan = _extract_eth_plan(eth_analysis, state, direction, entry_price)
    active = _refresh_active_position_state(state, position, plan, now_text)
    state.pop("pending_entry_order", None)
    initial_stop = _as_float(active.get("initial_stop"))
    if initial_stop is None:
        initial_stop = float(plan["stop"])
        active["initial_stop"] = initial_stop
    risk = abs(entry_price - initial_stop)
    if risk <= 0:
        risk = max(entry_price * 0.0035, 1e-12)
    r_multiple = (mark_price - entry_price) / risk if direction == "long" else (entry_price - mark_price) / risk
    current_stop = _current_stop_from_orders(bot_orders, direction)
    if current_stop is None:
        current_stop = _as_float(active.get("last_stop")) or initial_stop
    active["last_r_multiple"] = round(r_multiple, 4)
    active["last_mark_price"] = mark_price
    active["last_seen_at"] = now_text

    desired_stop = float(current_stop)
    desired_reason = "hold_initial_stop"
    if settings["break_even_enabled"] and r_multiple >= settings["break_even_trigger_r"]:
        be_stop = entry_price + risk * settings["break_even_lock_r"] if direction == "long" else entry_price - risk * settings["break_even_lock_r"]
        if _is_better_stop(be_stop, desired_stop, direction):
            desired_stop = be_stop
            desired_reason = "break_even"
    if settings["trailing_stop_enabled"] and r_multiple >= settings["trailing_trigger_r"]:
        trail_stop = mark_price - risk * settings["trailing_distance_r"] if direction == "long" else mark_price + risk * settings["trailing_distance_r"]
        trail_stop = _clamp_stop_away_from_mark(trail_stop, mark_price, direction)
        if _is_better_stop(trail_stop, desired_stop, direction):
            desired_stop = trail_stop
            desired_reason = "trailing_stop"

    if settings["place_exit_orders"] and not _tp_orders(bot_orders) and plan.get("take_profits"):
        action = {"type": "place_missing_take_profits", "mode": "live" if live_ready else "dry_run"}
        if live_ready:
            try:
                action["responses"] = _submit_take_profit_orders(settings, direction, contracts_abs, plan["take_profits"])
            except Exception as exc:
                action["error"] = f"{type(exc).__name__}: {exc}"
        else:
            action["planned_orders"] = _planned_take_profit_orders(direction, contracts_abs, plan["take_profits"])
        actions.append(action)

    stop_orders = _stop_orders(bot_orders)
    if settings["place_exit_orders"] and not stop_orders:
        action = {
            "type": "place_missing_stop",
            "reason": desired_reason,
            "price": round(desired_stop, 2),
            "mode": "live" if live_ready else "dry_run",
        }
        if live_ready:
            try:
                action["response"] = _submit_stop_order(settings, direction, contracts_abs, desired_stop, _stop_text_for_reason(desired_reason))
            except Exception as exc:
                action["error"] = f"{type(exc).__name__}: {exc}"
        actions.append(action)
        active["last_stop"] = round(desired_stop, 8)
    elif settings["place_exit_orders"] and _is_better_stop(desired_stop, current_stop, direction, min_step=max(entry_price * 0.00005, 0.05)):
        action = {
            "type": "replace_stop",
            "reason": desired_reason,
            "old_stop": round(float(current_stop), 2),
            "new_stop": round(desired_stop, 2),
            "r_multiple": round(r_multiple, 3),
            "mode": "live" if live_ready else "dry_run",
        }
        if live_ready:
            cancel_actions = _cancel_bot_orders(settings, stop_orders, live_ready)
            action["cancelled"] = cancel_actions
            if not any(cancel.get("error") for cancel in cancel_actions):
                try:
                    action["response"] = _submit_stop_order(settings, direction, contracts_abs, desired_stop, _stop_text_for_reason(desired_reason))
                except Exception as exc:
                    action["error"] = f"{type(exc).__name__}: {exc}"
        actions.append(action)
        active["last_stop"] = round(desired_stop, 8)
    else:
        active["last_stop"] = round(float(current_stop), 8)
        actions.append(
            {
                "type": "monitor_position",
                "r_multiple": round(r_multiple, 3),
                "current_stop": round(float(current_stop), 2),
                "next_stop_policy": desired_reason,
                "mode": "live" if live_ready else "dry_run",
            }
        )
    state["active_position"] = active
    return actions


def _extract_eth_plan(
    eth_analysis: dict[str, Any],
    state: dict[str, Any],
    direction: str,
    entry_price: float,
) -> dict[str, Any]:
    analysis = eth_analysis.get("analysis") if isinstance(eth_analysis.get("analysis"), dict) else eth_analysis
    candidates: list[tuple[str, dict[str, Any]]] = []
    active_state = state.get("active_position") if isinstance(state.get("active_position"), dict) else {}
    if active_state:
        candidates.append(("gate_state", active_state))
    pending = state.get("pending_entry_order") if isinstance(state.get("pending_entry_order"), dict) else {}
    pending_plan = pending.get("plan") if isinstance(pending.get("plan"), dict) else {}
    if pending_plan:
        candidates.append(("gate_pending_entry", pending_plan))
    modes = analysis.get("modes") if isinstance(analysis.get("modes"), dict) else {}
    if isinstance(modes.get("short_term"), dict):
        candidates.append(("eth_short_term", modes["short_term"]))
    if isinstance(analysis.get("active_plan_status"), dict):
        candidates.append(("eth_active_plan", analysis["active_plan_status"]))
    for source, raw in candidates:
        plan = _normalise_trade_plan(raw, direction, source)
        if plan:
            return plan
    return _fallback_eth_plan(direction, entry_price)


def _normalise_trade_plan(raw: dict[str, Any], direction: str, source: str) -> dict[str, Any] | None:
    if str(raw.get("direction") or direction) != direction:
        return None
    stop = _as_float(raw.get("stop") or raw.get("initial_stop"))
    if stop is None:
        return None
    take_profits = []
    for item in raw.get("take_profits") or []:
        if not isinstance(item, dict):
            continue
        price = _as_float(item.get("price"))
        if price is None:
            continue
        take_profits.append(
            {
                "name": item.get("name") or f"TP{len(take_profits) + 1}",
                "price": price,
                "portion_pct": _as_float(item.get("portion_pct")) or 0.0,
                "rr": _as_float(item.get("rr")),
            }
        )
    return {
        "direction": direction,
        "entry_zone": raw.get("entry_zone"),
        "entry_basis_for_rr": _as_float(raw.get("entry_basis_for_rr")),
        "stop": stop,
        "take_profits": take_profits,
        "plan_id": raw.get("plan_id"),
        "source": source,
    }


def _fallback_eth_plan(direction: str, entry_price: float) -> dict[str, Any]:
    risk = entry_price * 0.0045
    stop = entry_price - risk if direction == "long" else entry_price + risk
    multipliers = (1.10, 2.00, 3.00)
    portions = (30, 40, 30)
    take_profits = []
    for idx, multiple in enumerate(multipliers, start=1):
        price = entry_price + risk * multiple if direction == "long" else entry_price - risk * multiple
        take_profits.append({"name": f"TP{idx}", "price": price, "portion_pct": portions[idx - 1], "rr": multiple})
    return {
        "direction": direction,
        "entry_zone": None,
        "entry_basis_for_rr": entry_price,
        "stop": stop,
        "take_profits": take_profits,
        "plan_id": "fallback_emergency_eth_protection",
        "source": "fallback_emergency_eth_protection",
    }


def _refresh_active_position_state(
    state: dict[str, Any],
    position: dict[str, Any],
    plan: dict[str, Any],
    now_text: str,
) -> dict[str, Any]:
    direction = str(position.get("direction") or "")
    entry_price = _as_float(position.get("entry_price")) or 0.0
    contracts_abs = int(abs(_as_float(position.get("contracts")) or 0))
    active = state.get("active_position") if isinstance(state.get("active_position"), dict) else {}
    old_entry = _as_float(active.get("entry_price"))
    same_position = (
        active.get("direction") == direction
        and old_entry is not None
        and entry_price > 0
        and abs(old_entry - entry_price) / entry_price <= 0.002
    )
    if same_position:
        active["contracts_abs"] = contracts_abs
        active["plan_source"] = plan.get("source")
        active["plan_id"] = plan.get("plan_id")
        return active
    return {
        "has_position": True,
        "direction": direction,
        "entry_price": entry_price,
        "contracts_abs": contracts_abs,
        "initial_stop": float(plan["stop"]),
        "last_stop": float(plan["stop"]),
        "take_profits": plan.get("take_profits", []),
        "plan_id": plan.get("plan_id"),
        "plan_source": plan.get("source"),
        "opened_seen_at": now_text,
    }


def _planned_take_profit_orders(direction: str, contracts_abs: int, take_profits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    close_sign = -1 if direction == "long" else 1
    slices = _contract_slices(
        contracts_abs,
        [max(0.0, min(100.0, _as_float(tp.get("portion_pct")) or 0.0)) for tp in take_profits],
    )
    planned = []
    for idx, (take_profit, close_contracts) in enumerate(zip(take_profits, slices), start=1):
        planned.append(
            {
                "text": f"t-eth-tp{idx}",
                "close_size": close_sign * close_contracts,
                "price": round(float(take_profit["price"]), 2),
            }
        )
    return planned


def _is_better_stop(new_stop: float, current_stop: float | None, direction: str, min_step: float = 0.0) -> bool:
    if current_stop is None:
        return True
    if direction == "long":
        return new_stop > current_stop + min_step
    return new_stop < current_stop - min_step


def _clamp_stop_away_from_mark(stop: float, mark_price: float, direction: str) -> float:
    if direction == "long":
        return min(stop, mark_price * 0.999)
    return max(stop, mark_price * 1.001)


def _stop_text_for_reason(reason: str) -> str:
    if reason == "break_even":
        return GATE_BE_TEXT
    if reason == "trailing_stop":
        return GATE_TRAIL_TEXT
    return GATE_SL_TEXT


def _live_ready(settings: dict[str, Any]) -> bool:
    return (
        settings["enabled"]
        and not settings["dry_run"]
        and bool(settings["api_key"] and settings["api_secret"])
        and settings["confirm_live_trading"] == GATE_LIVE_CONFIRMATION
    )


def _compact_gate_state(state: dict[str, Any]) -> dict[str, Any]:
    active = state.get("active_position") if isinstance(state.get("active_position"), dict) else {}
    output = {"last_flat_at": state.get("last_flat_at")}
    pending = state.get("pending_entry_order") if isinstance(state.get("pending_entry_order"), dict) else {}
    if pending:
        keep_pending = {
            "status",
            "order_id",
            "direction",
            "entry_price",
            "signed_size",
            "plan_id",
            "submitted_at",
        }
        output["pending_entry_order"] = {key: pending.get(key) for key in keep_pending if key in pending}
    if active:
        keep = {
            "has_position",
            "direction",
            "entry_price",
            "contracts_abs",
            "initial_stop",
            "last_stop",
            "last_r_multiple",
            "last_mark_price",
            "plan_id",
            "plan_source",
            "opened_seen_at",
            "last_seen_at",
        }
        output["active_position"] = {key: active.get(key) for key in keep if key in active}
    return output


def _load_gate_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_gate_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _contract_slices(total_contracts: int, portions: list[float]) -> list[int]:
    if total_contracts <= 0 or not portions:
        return []
    clean = [max(0.0, float(portion)) for portion in portions]
    total_pct = sum(clean) or 100.0
    raw_sizes = [total_contracts * portion / total_pct for portion in clean]
    sizes = [int(math.floor(size)) for size in raw_sizes]
    remainder = total_contracts - sum(sizes)
    order = sorted(range(len(raw_sizes)), key=lambda idx: raw_sizes[idx] - sizes[idx], reverse=True)
    for idx in order[: max(0, remainder)]:
        sizes[idx] += 1
    return sizes


def _tp_rule(direction: str) -> int:
    return 1 if direction == "long" else 2


def _sl_rule(direction: str) -> int:
    return 2 if direction == "long" else 1


def _set_leverage(settings: dict[str, Any]) -> Any:
    path = f"/futures/{settings['settle']}/positions/{settings['contract']}/leverage"
    return _gate_request(settings, "POST", path, body=None, query_params={"leverage": str(settings["leverage"])})


def check_gate_connection(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    contract = fetch_gate_contract_info(config)
    result = {
        "version": GATE_EXECUTOR_VERSION,
        "enabled": settings["enabled"],
        "dry_run": settings["dry_run"],
        "testnet": settings["testnet"],
        "base_url": settings["base_url"],
        "contract": settings["contract"],
        "contract_info": contract,
        "account_ok": False,
        "account": {},
        "errors": [],
        "live_ready": False,
    }
    if not settings["api_key"] or not settings["api_secret"]:
        result["errors"].append("缺少 Gate API key/secret，無法測試帳戶。")
        return result
    try:
        account = _gate_request(settings, "GET", f"/futures/{settings['settle']}/accounts", body=None)
        result["account_ok"] = True
        result["account"] = _compact_account(account)
    except Exception as exc:
        result["errors"].append(f"Gate account check failed: {type(exc).__name__}: {exc}")
    result["live_ready"] = (
        result["account_ok"]
        and bool(contract.get("contract_size_eth") or settings["contract_size_eth"] > 0)
        and settings["confirm_live_trading"] == GATE_LIVE_CONFIRMATION
        and settings["enabled"]
        and not settings["dry_run"]
    )
    return result


def fetch_gate_position(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    contract_info = fetch_gate_contract_info(config)
    if settings["contract_size_eth"] <= 0 and contract_info.get("contract_size_eth"):
        settings["contract_size_eth"] = float(contract_info["contract_size_eth"])
    return _position_payload(settings, contract_info)
    result = {
        "version": GATE_EXECUTOR_VERSION,
        "enabled": settings["enabled"],
        "dry_run": settings["dry_run"],
        "testnet": settings["testnet"],
        "base_url": settings["base_url"],
        "contract": settings["contract"],
        "contract_info": contract_info,
        "api_configured": bool(settings["api_key"] and settings["api_secret"]),
        "position_ok": False,
        "has_position": False,
        "position": {},
        "errors": [],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not settings["api_key"] or not settings["api_secret"]:
        result["errors"].append("缺少 Gate API key/secret，無法讀取持倉。")
        return result
    try:
        position = _gate_request(settings, "GET", f"/futures/{settings['settle']}/positions/{settings['contract']}", body=None)
        result["position_ok"] = True
        result["position"] = _compact_position(position, settings)
        result["has_position"] = bool(result["position"].get("has_position"))
    except Exception as exc:
        result["errors"].append(f"Gate position check failed: {type(exc).__name__}: {exc}")
    return result


def manage_gate_position(
    config: dict[str, Any],
    eth_analysis: dict[str, Any] | None = None,
    state_path: Path = GATE_STATE_PATH,
) -> dict[str, Any]:
    settings = _settings(config)
    contract_info = fetch_gate_contract_info(config)
    if settings["contract_size_eth"] <= 0 and contract_info.get("contract_size_eth"):
        settings["contract_size_eth"] = float(contract_info["contract_size_eth"])
    state = _load_gate_state(state_path)
    position_payload = _position_payload(settings, contract_info)
    now_text = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    live_ready = _live_ready(settings)
    result = {
        "version": GATE_EXECUTOR_VERSION,
        "enabled": settings["enabled"],
        "dry_run": settings["dry_run"],
        "testnet": settings["testnet"],
        "management_enabled": settings["position_management_enabled"],
        "live_ready": live_ready,
        "contract": settings["contract"],
        "poll_seconds": settings["position_poll_seconds"],
        "contract_info": contract_info,
        "position": position_payload,
        "bot_price_orders": [],
        "actions": [],
        "planned_actions": [],
        "errors": list(position_payload.get("errors") or []),
        "state": {},
        "updated_at": now_text,
    }
    if not settings["position_management_enabled"]:
        result["planned_actions"].append({"type": "disabled", "message": "Gate ETH 持倉管理已關閉。"})
        return result
    if not settings["api_key"] or not settings["api_secret"]:
        result["planned_actions"].append({"type": "wait_api", "message": "缺少 Gate API key/secret，暫不能讀倉或管理。"})
        return result

    raw_orders: list[dict[str, Any]] = []
    order_list_ok = True
    try:
        raw_orders = _list_price_orders(settings)
    except Exception as exc:
        order_list_ok = False
        result["errors"].append(f"Gate price order list failed: {type(exc).__name__}: {exc}")
        result["planned_actions"].append(
            {"type": "skip_live_order_changes", "message": "讀不到 Gate 開放條件單，避免重複掛單，本輪不做 live 改單。"}
        )
    bot_orders = [_compact_price_order(order) for order in _bot_price_orders(raw_orders, settings)]
    result["bot_price_orders"] = bot_orders

    compact_position = position_payload.get("position") if isinstance(position_payload.get("position"), dict) else {}
    if not compact_position.get("has_position"):
        state.pop("active_position", None)
        state["last_flat_at"] = now_text
        pending = state.get("pending_entry_order") if isinstance(state.get("pending_entry_order"), dict) else {}
        if pending:
            result["planned_actions"].append(
                {
                    "type": "wait_entry_fill",
                    "message": "已有 Gate ETH entry 掛單，等待成交；成交後自動切換持倉管理。",
                    "plan_id": pending.get("plan_id"),
                    "order_id": pending.get("order_id"),
                }
            )
        if settings["cancel_bot_orders_when_flat"] and bot_orders and not pending:
            result["actions"].extend(_cancel_bot_orders(settings, bot_orders, live_ready))
        result["state"] = _compact_gate_state(state)
        _save_gate_state(state_path, state)
        return result

    result["actions"].extend(
        _manage_open_position(
            settings=settings,
            position=compact_position,
            bot_orders=bot_orders,
            eth_analysis=eth_analysis or {},
            state=state,
            live_ready=live_ready and order_list_ok,
            now_text=now_text,
        )
    )
    result["state"] = _compact_gate_state(state)
    _save_gate_state(state_path, state)
    return result


def _position_payload(settings: dict[str, Any], contract_info: dict[str, Any]) -> dict[str, Any]:
    result = {
        "version": GATE_EXECUTOR_VERSION,
        "enabled": settings["enabled"],
        "dry_run": settings["dry_run"],
        "testnet": settings["testnet"],
        "base_url": settings["base_url"],
        "contract": settings["contract"],
        "contract_info": contract_info,
        "api_configured": bool(settings["api_key"] and settings["api_secret"]),
        "position_ok": False,
        "has_position": False,
        "position": {},
        "errors": [],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not settings["api_key"] or not settings["api_secret"]:
        result["errors"].append("缺少 Gate API key/secret，無法讀取持倉。")
        return result
    try:
        position = _gate_request(settings, "GET", f"/futures/{settings['settle']}/positions/{settings['contract']}", body=None)
        result["position_ok"] = True
        result["position"] = _compact_position(position, settings)
        result["has_position"] = bool(result["position"].get("has_position"))
    except Exception as exc:
        result["errors"].append(f"Gate position check failed: {type(exc).__name__}: {exc}")
    return result


def fetch_gate_contract_info(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    path = f"/futures/{settings['settle']}/contracts/{settings['contract']}"
    try:
        payload = _public_gate_request(settings["base_url"], path)
    except Exception as exc:
        return {
            "ok": False,
            "source": "gate_public_contract",
            "contract": settings["contract"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    contract_size = _first_float(payload, ("quanto_multiplier", "order_size_min", "multiplier"))
    return {
        "ok": True,
        "source": "gate_public_contract",
        "contract": payload.get("name") or settings["contract"],
        "contract_size_eth": contract_size,
        "quanto_multiplier": payload.get("quanto_multiplier"),
        "order_size_min": payload.get("order_size_min"),
        "order_size_max": payload.get("order_size_max"),
        "mark_price": payload.get("mark_price"),
        "index_price": payload.get("index_price"),
        "raw": {
            key: payload.get(key)
            for key in ("name", "type", "quanto_multiplier", "order_size_min", "order_size_max", "leverage_min", "leverage_max", "mark_price", "index_price")
            if key in payload
        },
    }


def _gate_request(
    settings: dict[str, Any],
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    query_params: dict[str, Any] | None = None,
) -> Any:
    body_text = "" if body is None else json.dumps(body or {}, separators=(",", ":"))
    timestamp = str(int(time.time()))
    query = urlencode(query_params or {})
    body_hash = hashlib.sha512(body_text.encode("utf-8")).hexdigest()
    signature_string = "\n".join((method.upper(), f"/api/v4{path}", query, body_hash, timestamp))
    signature = hmac.new(settings["api_secret"].encode("utf-8"), signature_string.encode("utf-8"), hashlib.sha512).hexdigest()
    request = Request(
        settings["base_url"] + path + (f"?{query}" if query else ""),
        data=body_text.encode("utf-8") if body is not None else None,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "KEY": settings["api_key"],
            "Timestamp": timestamp,
            "SIGN": signature,
            "User-Agent": "crypto-ict-eth-gate-executor/1.0",
        },
        method=method.upper(),
    )
    try:
        with urlopen(request, timeout=12) as response:
            text = response.read().decode("utf-8")
            return json.loads(text) if text else {}
    except HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise GateExecutionError(f"HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GateExecutionError(str(exc)) from exc


def _public_gate_request(base_url: str, path: str) -> Any:
    request = Request(
        base_url + path,
        headers={
            "Accept": "application/json",
            "User-Agent": "crypto-ict-eth-gate-executor/1.0",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            text = response.read().decode("utf-8")
            return json.loads(text) if text else {}
    except HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise GateExecutionError(f"HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GateExecutionError(str(exc)) from exc


def _compact_account(account: Any) -> dict[str, Any]:
    if not isinstance(account, dict):
        return {"raw_type": type(account).__name__}
    keep = {
        "total",
        "available",
        "unrealised_pnl",
        "position_margin",
        "order_margin",
        "currency",
        "in_dual_mode",
    }
    return {key: account.get(key) for key in keep if key in account}


def _compact_position(position: Any, settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(position, dict):
        return {"raw_type": type(position).__name__, "has_position": False}
    size = _as_float(position.get("size")) or 0.0
    contract_size = settings.get("contract_size_eth") or 0.0
    quantity_eth = abs(size) * contract_size if contract_size else None
    entry_price = _as_float(position.get("entry_price"))
    mark_price = _as_float(position.get("mark_price"))
    margin = _as_float(position.get("margin"))
    value = _as_float(position.get("value"))
    pnl = _as_float(position.get("unrealised_pnl"))
    if pnl is None:
        pnl = _as_float(position.get("unrealized_pnl"))
    pnl_pct = pnl / margin * 100.0 if pnl is not None and margin and margin > 0 else None
    direction = "long" if size > 0 else "short" if size < 0 else "flat"
    keep = {
        "contract",
        "size",
        "leverage",
        "leverage_max",
        "risk_limit",
        "maintenance_rate",
        "liq_price",
        "margin_mode",
        "mode",
    }
    output = {key: position.get(key) for key in keep if key in position}
    output.update(
        {
            "has_position": abs(size) > 0,
            "direction": direction,
            "contracts": int(size) if float(size).is_integer() else size,
            "contracts_abs": abs(size),
            "contract_size_eth": contract_size,
            "quantity_eth": round(quantity_eth, 8) if quantity_eth is not None else None,
            "entry_price": entry_price,
            "mark_price": mark_price,
            "position_value_usdt": value,
            "margin_usdt": margin,
            "unrealised_pnl": pnl,
            "unrealised_pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else None,
            "liq_price": _as_float(position.get("liq_price")) if position.get("liq_price") not in {None, ""} else None,
        }
    )
    return output


def _first_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        try:
            value = payload.get(key)
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _price_text(value: float) -> str:
    return f"{float(value):.2f}"


def _eth_symbol(config: dict[str, Any]) -> str:
    eth = config.get("eth", {}) if isinstance(config.get("eth"), dict) else {}
    return str(eth.get("symbol") or "ETHUSDT").upper().replace("/", "")
