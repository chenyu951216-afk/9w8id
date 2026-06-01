from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("config.local.json")


DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "0.0.0.0",
        "port": 8080,
        "refresh_minutes": 5,
        "open_browser": False,
    },
    "scan": {
        "exchange": "auto",
        "top": 70,
        "symbols": "",
        "min_volume": 20_000_000,
        "min_score": 0,
        "passing_score": 72,
        "workers": 4,
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
    ("paid_data", "enabled"): ("PAID_DATA_ENABLED", "bool"),
    ("paid_data", "preferred_derivatives_exchange"): ("PREFERRED_DERIVATIVES_EXCHANGE", str),
    ("paid_data", "event_lookahead_days"): ("EVENT_LOOKAHEAD_DAYS", int),
    ("paid_data", "glassnode_interval"): ("GLASSNODE_INTERVAL", str),
    ("paid_data", "cryptoquant_window"): ("CRYPTOQUANT_WINDOW", str),
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
    server["refresh_minutes"] = max(1, int(server.get("refresh_minutes") or 15))
    server["port"] = max(1, min(65535, int(server.get("port") or 8765)))
    scan["top"] = max(1, min(300, int(scan.get("top") or 40)))
    scan["min_volume"] = max(0.0, float(scan.get("min_volume") or 0))
    scan["min_score"] = max(0.0, min(100.0, float(scan.get("min_score") or 0)))
    scan["passing_score"] = max(1.0, min(100.0, float(scan.get("passing_score") or 72)))
    scan["workers"] = max(1, min(8, int(scan.get("workers") or 4)))
    if scan.get("exchange") not in {"auto", "bybit", "binance"}:
        scan["exchange"] = "auto"
    return config


def save_config(update: dict[str, Any], path: Path = CONFIG_PATH) -> dict[str, Any]:
    current = load_config(path)
    clean_update = copy.deepcopy(update)
    api_keys = clean_update.get("api_keys")
    if isinstance(api_keys, dict):
        for key, value in list(api_keys.items()):
            if value == "__keep__":
                api_keys[key] = current.get("api_keys", {}).get(key, "")
    merged = normalize_config(_deep_merge(current, clean_update))
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["api_keys"] = {
        key: {"configured": bool(value), "masked": mask_secret(value)}
        for key, value in config.get("api_keys", {}).items()
    }
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
