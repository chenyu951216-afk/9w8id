from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("config.local.json")
DEFAULT_REFRESH_MINUTES = 5


DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "0.0.0.0",
        "port": 8080,
        "refresh_minutes": 5,
        "open_browser": False,
    },
    "scan": {
        "exchange": "auto",
        "top": 1,
        "symbols": "ETHUSDT",
        "min_volume": 0,
        "min_score": 0,
        "passing_score": 68,
        "workers": 1,
    },
    "eth": {
        "enabled": True,
        "symbol": "ETHUSDT",
        "short_mode_enabled": True,
        "swing_mode_enabled": True,
        "short_min_score": 74,
        "swing_min_score": 70,
        "active_plan_ttl_minutes": 720,
        "swing_plan_ttl_minutes": 4320,
        "session_timezone": "Asia/Taipei",
        "allow_trend_chase_only_on_breakout": True,
    },
    "gate_trading": {
        "enabled": False,
        "dry_run": True,
        "testnet": False,
        "api_key": "",
        "api_secret": "",
        "settle": "usdt",
        "contract": "ETH_USDT",
        "leverage": 200,
        "margin_usdt": 15,
        "max_notional_usdt": 3000,
        "contract_size_eth": 0,
        "set_leverage_before_order": True,
        "place_exit_orders": True,
        "exit_order_expiration_seconds": 86400,
        "position_management_enabled": True,
        "price_poll_seconds": 1,
        "position_poll_seconds": 2,
        "break_even_enabled": True,
        "break_even_trigger_r": 1.0,
        "break_even_lock_r": 0.05,
        "trailing_stop_enabled": True,
        "trailing_trigger_r": 1.8,
        "trailing_distance_r": 0.75,
        "cancel_bot_orders_when_flat": True,
        "confirm_live_trading": "",
    },
    "api_keys": {
        "coingecko": "",
        "cryptopanic": "",
        "coinalyze": "",
        "coinglass": "",
        "glassnode": "",
        "cryptoquant": "",
        "coinmarketcal": "",
        "thetie": "",
        "tokenmetrics": "",
    },
    "paid_data": {
        "enabled": True,
        "preferred_derivatives_exchange": "Bybit",
        "event_lookahead_days": 7,
        "glassnode_interval": "1h",
        "cryptoquant_window": "day",
        "coinglass_workers": 5,
        "coinglass_timeout_seconds": 120,
        "coinglass_heatmap_symbol_limit": 20,
        "coinalyze_history_symbol_limit": 10,
    },
    "notifications": {
        "discord_enabled": True,
        "discord_webhook_url": "",
        "discord_username": "ICT Coin Bot",
    },
}


ENV_KEY_MAP = {
    "coingecko": "COINGECKO_API_KEY",
    "cryptopanic": "CRYPTOPANIC_API_KEY",
    "coinalyze": "COINALYZE_API_KEY",
    "coinglass": "COINGLASS_API_KEY",
    "glassnode": "GLASSNODE_API_KEY",
    "cryptoquant": "CRYPTOQUANT_API_KEY",
    "coinmarketcal": "COINMARKETCAL_API_KEY",
    "thetie": "THETIE_API_KEY",
    "tokenmetrics": "TOKENMETRICS_API_KEY",
}


ENV_SETTING_MAP = {
    ("server", "host"): ("HOST", str),
    ("server", "port"): ("PORT", int),
    ("server", "refresh_minutes"): ("REFRESH_MINUTES", int),
    ("server", "open_browser"): ("OPEN_BROWSER", "bool"),
    ("scan", "exchange"): ("EXCHANGE", str),
    ("scan", "top"): ("SCAN_TOP", int),
    ("scan", "symbols"): ("SYMBOLS", str),
    ("scan", "min_volume"): ("MIN_VOLUME", float),
    ("scan", "min_score"): ("MIN_SCORE", float),
    ("scan", "passing_score"): ("PASSING_SCORE", float),
    ("scan", "workers"): ("WORKERS", int),
    ("eth", "enabled"): ("ETH_MODE_ENABLED", "bool"),
    ("eth", "symbol"): ("ETH_SYMBOL", str),
    ("eth", "short_min_score"): ("ETH_SHORT_MIN_SCORE", float),
    ("eth", "swing_min_score"): ("ETH_SWING_MIN_SCORE", float),
    ("eth", "active_plan_ttl_minutes"): ("ETH_ACTIVE_PLAN_TTL_MINUTES", int),
    ("gate_trading", "enabled"): ("GATE_TRADING_ENABLED", "bool"),
    ("gate_trading", "dry_run"): ("GATE_TRADING_DRY_RUN", "bool"),
    ("gate_trading", "testnet"): ("GATE_TRADING_TESTNET", "bool"),
    ("gate_trading", "api_key"): ("GATE_API_KEY", str),
    ("gate_trading", "api_secret"): ("GATE_API_SECRET", str),
    ("gate_trading", "settle"): ("GATE_SETTLE", str),
    ("gate_trading", "contract"): ("GATE_CONTRACT", str),
    ("gate_trading", "leverage"): ("GATE_LEVERAGE", int),
    ("gate_trading", "margin_usdt"): ("GATE_MARGIN_USDT", float),
    ("gate_trading", "max_notional_usdt"): ("GATE_MAX_NOTIONAL_USDT", float),
    ("gate_trading", "contract_size_eth"): ("GATE_CONTRACT_SIZE_ETH", float),
    ("gate_trading", "set_leverage_before_order"): ("GATE_SET_LEVERAGE_BEFORE_ORDER", "bool"),
    ("gate_trading", "place_exit_orders"): ("GATE_PLACE_EXIT_ORDERS", "bool"),
    ("gate_trading", "exit_order_expiration_seconds"): ("GATE_EXIT_ORDER_EXPIRATION_SECONDS", int),
    ("gate_trading", "position_management_enabled"): ("GATE_POSITION_MANAGEMENT_ENABLED", "bool"),
    ("gate_trading", "price_poll_seconds"): ("GATE_PRICE_POLL_SECONDS", int),
    ("gate_trading", "position_poll_seconds"): ("GATE_POSITION_POLL_SECONDS", int),
    ("gate_trading", "break_even_enabled"): ("GATE_BREAK_EVEN_ENABLED", "bool"),
    ("gate_trading", "break_even_trigger_r"): ("GATE_BREAK_EVEN_TRIGGER_R", float),
    ("gate_trading", "break_even_lock_r"): ("GATE_BREAK_EVEN_LOCK_R", float),
    ("gate_trading", "trailing_stop_enabled"): ("GATE_TRAILING_STOP_ENABLED", "bool"),
    ("gate_trading", "trailing_trigger_r"): ("GATE_TRAILING_TRIGGER_R", float),
    ("gate_trading", "trailing_distance_r"): ("GATE_TRAILING_DISTANCE_R", float),
    ("gate_trading", "cancel_bot_orders_when_flat"): ("GATE_CANCEL_BOT_ORDERS_WHEN_FLAT", "bool"),
    ("gate_trading", "confirm_live_trading"): ("GATE_CONFIRM_LIVE_TRADING", str),
    ("paid_data", "enabled"): ("PAID_DATA_ENABLED", "bool"),
    ("paid_data", "preferred_derivatives_exchange"): ("PREFERRED_DERIVATIVES_EXCHANGE", str),
    ("paid_data", "event_lookahead_days"): ("EVENT_LOOKAHEAD_DAYS", int),
    ("paid_data", "glassnode_interval"): ("GLASSNODE_INTERVAL", str),
    ("paid_data", "cryptoquant_window"): ("CRYPTOQUANT_WINDOW", str),
    ("paid_data", "coinglass_workers"): ("COINGLASS_WORKERS", int),
    ("paid_data", "coinglass_timeout_seconds"): ("COINGLASS_TIMEOUT_SECONDS", int),
    ("paid_data", "coinglass_heatmap_symbol_limit"): ("COINGLASS_HEATMAP_SYMBOL_LIMIT", int),
    ("paid_data", "coinalyze_history_symbol_limit"): ("COINALYZE_HISTORY_SYMBOL_LIMIT", int),
    ("notifications", "discord_enabled"): ("DISCORD_ALERTS_ENABLED", "bool"),
    ("notifications", "discord_webhook_url"): ("DISCORD_WEBHOOK_URL", str),
    ("notifications", "discord_username"): ("DISCORD_USERNAME", str),
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8").strip()
            loaded = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            config = _deep_merge(config, loaded)
    for key, env_name in ENV_KEY_MAP.items():
        env_value = os.environ.get(env_name)
        if env_value:
            config["api_keys"][key] = env_value
    for path_keys, env_spec in ENV_SETTING_MAP.items():
        env_name, caster = env_spec
        env_value = os.environ.get(env_name)
        if env_value is None or env_value == "":
            continue
        section, key = path_keys
        config[section][key] = _cast_env(env_value, caster)
    return normalize_config(config)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    config = _deep_merge(DEFAULT_CONFIG, config)
    server = config["server"]
    scan = config["scan"]
    server["refresh_minutes"] = min(DEFAULT_REFRESH_MINUTES, max(1, int(server.get("refresh_minutes") or DEFAULT_REFRESH_MINUTES)))
    server["port"] = max(1, min(65535, int(server.get("port") or 8765)))
    scan["top"] = max(1, min(300, int(scan.get("top") or 100)))
    scan["min_volume"] = max(0.0, float(scan.get("min_volume") or 0))
    scan["min_score"] = max(0.0, min(100.0, float(scan.get("min_score") or 0)))
    scan["passing_score"] = max(1.0, min(100.0, float(scan.get("passing_score") or 72)))
    scan["workers"] = max(1, min(8, int(scan.get("workers") or 4)))
    if scan.get("exchange") not in {"auto", "bybit", "binance"}:
        scan["exchange"] = "auto"
    eth = config["eth"]
    eth["enabled"] = bool(eth.get("enabled", True))
    eth["symbol"] = str(eth.get("symbol") or "ETHUSDT").upper().replace("/", "")
    eth["short_min_score"] = max(50.0, min(95.0, float(eth.get("short_min_score") or 74)))
    eth["swing_min_score"] = max(50.0, min(95.0, float(eth.get("swing_min_score") or 70)))
    eth["active_plan_ttl_minutes"] = max(30, min(4320, int(eth.get("active_plan_ttl_minutes") or 720)))
    eth["swing_plan_ttl_minutes"] = max(240, min(20160, int(eth.get("swing_plan_ttl_minutes") or 4320)))
    eth["short_mode_enabled"] = bool(eth.get("short_mode_enabled", True))
    eth["swing_mode_enabled"] = bool(eth.get("swing_mode_enabled", True))
    eth["allow_trend_chase_only_on_breakout"] = bool(eth.get("allow_trend_chase_only_on_breakout", True))
    if eth["enabled"]:
        scan["symbols"] = eth["symbol"]
        scan["top"] = 1
        scan["min_volume"] = 0.0
        scan["workers"] = 1
    gate = config["gate_trading"]
    gate["enabled"] = bool(gate.get("enabled", False))
    gate["dry_run"] = bool(gate.get("dry_run", True))
    gate["testnet"] = bool(gate.get("testnet", True))
    gate["api_key"] = str(gate.get("api_key") or "")
    gate["api_secret"] = str(gate.get("api_secret") or "")
    gate["settle"] = str(gate.get("settle") or "usdt").lower()
    gate["contract"] = str(gate.get("contract") or "ETH_USDT").upper()
    gate["leverage"] = max(1, min(200, int(gate.get("leverage") or 200)))
    gate["margin_usdt"] = max(0.0, float(gate.get("margin_usdt") or 15))
    gate["max_notional_usdt"] = max(0.0, float(gate.get("max_notional_usdt") or gate["margin_usdt"] * gate["leverage"]))
    gate["contract_size_eth"] = max(0.0, float(gate.get("contract_size_eth") or 0))
    gate["set_leverage_before_order"] = bool(gate.get("set_leverage_before_order", True))
    gate["place_exit_orders"] = bool(gate.get("place_exit_orders", False))
    gate["exit_order_expiration_seconds"] = max(60, min(604800, int(gate.get("exit_order_expiration_seconds") or 86400)))
    gate["position_management_enabled"] = bool(gate.get("position_management_enabled", True))
    gate["price_poll_seconds"] = max(1, min(15, int(gate.get("price_poll_seconds") or 1)))
    gate["position_poll_seconds"] = max(1, min(30, int(gate.get("position_poll_seconds") or 2)))
    gate["break_even_enabled"] = bool(gate.get("break_even_enabled", True))
    gate["break_even_trigger_r"] = max(0.2, min(5.0, float(gate.get("break_even_trigger_r") or 1.0)))
    gate["break_even_lock_r"] = max(0.0, min(1.0, float(gate.get("break_even_lock_r") or 0.05)))
    gate["trailing_stop_enabled"] = bool(gate.get("trailing_stop_enabled", True))
    gate["trailing_trigger_r"] = max(0.5, min(10.0, float(gate.get("trailing_trigger_r") or 1.8)))
    gate["trailing_distance_r"] = max(0.1, min(5.0, float(gate.get("trailing_distance_r") or 0.75)))
    gate["cancel_bot_orders_when_flat"] = bool(gate.get("cancel_bot_orders_when_flat", True))
    gate["confirm_live_trading"] = str(gate.get("confirm_live_trading") or "")
    notifications = config["notifications"]
    notifications["discord_enabled"] = bool(notifications.get("discord_enabled", True))
    notifications["discord_webhook_url"] = str(notifications.get("discord_webhook_url") or "")
    notifications["discord_username"] = str(notifications.get("discord_username") or "ICT Coin Bot")
    return config


def save_config(update: dict[str, Any], path: Path = CONFIG_PATH) -> dict[str, Any]:
    current = load_config(path)
    clean_update = copy.deepcopy(update)
    api_keys = clean_update.get("api_keys")
    if isinstance(api_keys, dict):
        for key, value in list(api_keys.items()):
            if value == "__keep__":
                api_keys[key] = current.get("api_keys", {}).get(key, "")
    gate_update = clean_update.get("gate_trading")
    if isinstance(gate_update, dict):
        for key in ("api_key", "api_secret", "confirm_live_trading"):
            if gate_update.get(key) == "__keep__":
                gate_update[key] = current.get("gate_trading", {}).get(key, "")
    merged = normalize_config(_deep_merge(current, clean_update))
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["api_keys"] = {
        key: {"configured": bool(value), "masked": mask_secret(value)}
        for key, value in config.get("api_keys", {}).items()
    }
    notifications = result.get("notifications")
    if isinstance(notifications, dict):
        webhook_url = str(notifications.get("discord_webhook_url") or "")
        notifications["discord_webhook_url"] = {
            "configured": bool(webhook_url),
            "masked": mask_secret(webhook_url),
        }
    gate = result.get("gate_trading")
    if isinstance(gate, dict):
        for key in ("api_key", "api_secret"):
            value = str(gate.get(key) or "")
            gate[key] = {"configured": bool(value), "masked": mask_secret(value)}
        confirm = str(gate.get("confirm_live_trading") or "")
        gate["confirm_live_trading"] = {"configured": bool(confirm), "masked": mask_secret(confirm)}
    return result


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def _cast_env(value: str, caster: Any) -> Any:
    if caster == "bool":
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return caster(value)
    except (TypeError, ValueError):
        return value
