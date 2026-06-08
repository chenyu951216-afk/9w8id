from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import SymbolReport


GATE_EXECUTOR_VERSION = "gate_eth_executor_2026_06_v1"
GATE_LIVE_CONFIRMATION = "I_UNDERSTAND_GATE_200X_ETH_RISK"
GATE_PROD_BASE_URL = "https://fx-api.gateio.ws/api/v4"
GATE_TESTNET_BASE_URL = "https://fx-api-testnet.gateio.ws/api/v4"


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


def build_gate_execution(report: SymbolReport, config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    contract_info = fetch_gate_contract_info(config)
    if settings["contract_size_eth"] <= 0 and contract_info.get("contract_size_eth"):
        settings["contract_size_eth"] = float(contract_info["contract_size_eth"])
    analysis = report.metadata.get("eth_analysis") if isinstance(report.metadata.get("eth_analysis"), dict) else {}
    short_mode = analysis.get("modes", {}).get("short_term", {}) if isinstance(analysis.get("modes"), dict) else {}
    intent = _build_order_intent(report, short_mode, settings)
    live_blockers = _live_blockers(settings, intent, short_mode)
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
    return {
        **base_result,
        "action": "submitted",
        "submitted": True,
        "response": response,
        "exit_order_responses": exit_responses,
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
        "size": signed_size,
        "price": _price_text(entry_price),
        "tif": "gtc",
        "reduce_only": False,
        "text": "t-eth-dedicated-bot",
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
    if mode.get("state") == "manage_existing":
        blockers.append("已有同方向計畫，不能重複開新倉。")
    return blockers


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
    for idx, take_profit in enumerate(intent.get("take_profits") or [], start=1):
        if not isinstance(take_profit, dict):
            continue
        price = _as_float(take_profit.get("price"))
        if price is None:
            continue
        portion = max(0.0, min(100.0, _as_float(take_profit.get("portion_pct")) or 0.0))
        close_contracts = max(1, math.floor(total_contracts * portion / 100.0))
        close_contracts = min(total_contracts, close_contracts)
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
            "size": str(close_size),
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


def _tp_rule(direction: str) -> int:
    return 1 if direction == "long" else 2


def _sl_rule(direction: str) -> int:
    return 2 if direction == "long" else 1


def _set_leverage(settings: dict[str, Any]) -> Any:
    path = f"/futures/{settings['settle']}/positions/{settings['contract']}/leverage"
    return _gate_request(settings, "POST", path, query_params={"leverage": str(settings["leverage"])}, body={})


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
