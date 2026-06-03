from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .models import SymbolReport
from .report import fmt_price, report_to_dict


ALERT_STATE_PATH = Path("state/discord_alert_state.json")
WEBHOOK_TIMEOUT_SECONDS = 12
SYMBOL_DIRECTION_COOLDOWN_SECONDS = 30 * 60


class DiscordWebhookError(RuntimeError):
    pass


WebhookPoster = Callable[[str, dict[str, Any], str, bytes, int], None]


def send_discord_executable_reports(
    reports: list[SymbolReport],
    meta: dict[str, Any],
    config: dict[str, Any],
    state_path: Path = ALERT_STATE_PATH,
    post_webhook: WebhookPoster | None = None,
) -> dict[str, Any]:
    settings = _discord_settings(config)
    webhook_url = settings["webhook_url"]
    if not settings["enabled"] or not webhook_url:
        return {"enabled": False, "sent": [], "skipped": [], "errors": []}

    poster = post_webhook or _post_discord_webhook
    state = _load_alert_state(state_path)
    previous_sent = state.get("sent", {}) if isinstance(state.get("sent"), dict) else {}
    previous_cooldowns = state.get("cooldowns", {}) if isinstance(state.get("cooldowns"), dict) else {}
    now_ts = _alert_timestamp(meta)
    next_sent: dict[str, str] = {}
    next_cooldowns = _active_cooldowns(previous_cooldowns, now_ts)
    sent: list[str] = []
    skipped: list[str] = []
    cooldown_skipped: list[str] = []
    errors: list[str] = []
    executable_payloads = _executable_payloads(reports)

    for payload in executable_payloads:
        symbol = str(payload.get("symbol") or "").upper()
        direction = str(payload.get("direction") or "").lower()
        cooldown_key = f"{symbol}:{direction}" if direction else symbol
        signature = _alert_signature(payload)
        if previous_sent.get(symbol) == signature:
            skipped.append(symbol)
            next_sent[symbol] = signature
            continue
        previous_cooldown = next_cooldowns.get(cooldown_key)
        previous_ts = _cooldown_ts(previous_cooldown)
        if previous_ts is not None and now_ts - previous_ts < SYMBOL_DIRECTION_COOLDOWN_SECONDS:
            skipped.append(symbol)
            cooldown_skipped.append(symbol)
            next_sent[symbol] = previous_sent.get(symbol, signature)
            continue
        file_name = f"{_safe_file_stem(symbol)}_trade_report.json"
        attachment = _attachment_payload(payload, meta)
        file_bytes = json.dumps(attachment, ensure_ascii=False, indent=2).encode("utf-8")
        try:
            poster(
                webhook_url,
                _discord_message(payload, meta, file_name, settings["username"]),
                file_name,
                file_bytes,
                WEBHOOK_TIMEOUT_SECONDS,
            )
        except DiscordWebhookError as exc:
            errors.append(f"{symbol}: {exc}")
        except Exception as exc:
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
        else:
            sent.append(symbol)
            next_sent[symbol] = signature
            next_cooldowns[cooldown_key] = {
                "last_sent_ts": now_ts,
                "last_sent_at": _iso_from_ts(now_ts),
                "signature": signature,
            }

    _save_alert_state(
        state_path,
        {
            "sent": next_sent,
            "cooldowns": next_cooldowns,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return {"enabled": True, "sent": sent, "skipped": skipped, "cooldown_skipped": cooldown_skipped, "errors": errors}


def _discord_settings(config: dict[str, Any]) -> dict[str, Any]:
    notifications = config.get("notifications", {})
    if not isinstance(notifications, dict):
        notifications = {}
    webhook_url = str(
        os.environ.get("DC_WEBHOOK_URL")
        or os.environ.get("DISCORD_WEBHOOK_URL")
        or notifications.get("discord_webhook_url")
        or ""
    ).strip()
    enabled_value = os.environ.get("DISCORD_ALERTS_ENABLED")
    if enabled_value is None:
        enabled_value = os.environ.get("DC_ALERTS_ENABLED")
    enabled = _as_bool(enabled_value, bool(notifications.get("discord_enabled", True)))
    username = str(
        os.environ.get("DISCORD_USERNAME")
        or notifications.get("discord_username")
        or "ICT Coin Bot"
    ).strip()
    return {"enabled": enabled, "webhook_url": webhook_url, "username": username}


def _executable_payloads(reports: list[SymbolReport]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for report in reports:
        payload = report_to_dict(report)
        if payload.get("should_execute") is True:
            output.append(payload)
    return output


def _attachment_payload(payload: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "complete_executable_trade_report",
        "generated_at": meta.get("generated_at"),
        "exchange": meta.get("exchange"),
        "scan": {
            "top": meta.get("top"),
            "min_volume": meta.get("min_volume"),
            "symbols": meta.get("symbols", []),
            "timeframe_limits": meta.get("timeframe_limits", {}),
            "refresh_minutes": meta.get("refresh_minutes"),
        },
        "report": payload,
    }


def _discord_message(
    payload: dict[str, Any],
    meta: dict[str, Any],
    file_name: str,
    username: str,
) -> dict[str, Any]:
    symbol = str(payload.get("symbol") or "-")
    direction = str(payload.get("direction_label") or payload.get("direction") or "-")
    action = str(payload.get("trade_action_label") or payload.get("trade_action") or "-")
    execution_mode = str(payload.get("execution_mode") or "-")
    entry = _entry_text(payload.get("entry_zone"))
    stop = fmt_price(_as_float(payload.get("stop")))
    targets = _tp_text(payload)
    rr = payload.get("RR", payload.get("rr"))
    rr_text = "-" if rr is None else f"{float(rr):.2f}R"
    score = _score_text(payload)
    strategy = _strategy_text(payload)
    instrument_standard = _instrument_standard_text(payload)
    risk_notes = _list_text(payload.get("risk_notes") or payload.get("warnings") or [], limit=4)
    failure_conditions = _list_text(
        payload.get("failure_conditions") or payload.get("invalidation_conditions") or [],
        limit=4,
    )
    summary = str(payload.get("execution_summary") or payload.get("trade_action_reason") or "")
    if len(summary) > 900:
        summary = summary[:897] + "..."
    return {
        "username": username,
        "content": f"Executable trade signal: **{symbol}**. Risk and failure conditions are included; complete report attached as `{file_name}`.",
        "allowed_mentions": {"parse": []},
        "embeds": [
            {
                "title": f"{symbol} executable trade signal",
                "description": summary or "Execution gate passed.",
                "color": 0x2ECC71 if str(payload.get("direction")) == "long" else 0xE74C3C,
                "timestamp": str(meta.get("generated_at") or payload.get("data_time") or ""),
                "fields": [
                    {"name": "Direction", "value": direction, "inline": True},
                    {"name": "Action", "value": f"{action} / {execution_mode}", "inline": True},
                    {"name": "Score", "value": score, "inline": True},
                    {"name": "Strategy", "value": strategy, "inline": True},
                    {"name": "Instrument Standard", "value": instrument_standard, "inline": False},
                    {"name": "Entry", "value": entry, "inline": True},
                    {"name": "Stop", "value": stop, "inline": True},
                    {"name": "RR", "value": rr_text, "inline": True},
                    {"name": "Targets", "value": targets, "inline": False},
                    {"name": "Risk", "value": risk_notes, "inline": False},
                    {"name": "Failure Conditions", "value": failure_conditions, "inline": False},
                ],
            }
        ],
    }


def _score_text(payload: dict[str, Any]) -> str:
    selection = payload.get("selection_score", payload.get("score"))
    execution = payload.get("execution_score")
    selection_text = "-" if selection is None else f"{float(selection):.1f}"
    execution_text = "-" if execution is None else f"{float(execution):.1f}"
    return f"selection {selection_text} / execution {execution_text}"


def _strategy_text(payload: dict[str, Any]) -> str:
    label = str(payload.get("strategy_label") or "")
    fit = payload.get("strategy_fit_score")
    if not label:
        profile = payload.get("strategy_profile", {})
        if isinstance(profile, dict):
            label = str(profile.get("label") or "")
            fit = fit if fit is not None else profile.get("fit_score")
    fit_text = "-" if fit is None else f"{float(fit):.1f}"
    return f"{label or '-'} / fit {fit_text}"


def _instrument_standard_text(payload: dict[str, Any]) -> str:
    instrument = str(payload.get("instrument_class") or "-")
    profile = payload.get("volatility_profile", {})
    if not isinstance(profile, dict):
        return instrument
    low = profile.get("active_low_atr_pct")
    high = profile.get("active_high_atr_pct")
    hot = profile.get("hot_atr_pct")
    extreme = profile.get("extreme_atr_pct")
    state = payload.get("volatility_state") or "unknown"
    if None in {low, high, hot, extreme}:
        return f"{instrument} / volatility {state}"
    return (
        f"{instrument} / volatility {state}; "
        f"active ATR {float(low):.2f}-{float(high):.2f}%, hot>{float(hot):.2f}%, extreme>={float(extreme):.2f}%"
    )


def _list_text(values: Any, limit: int = 4) -> str:
    if not isinstance(values, list):
        values = [values] if values else []
    clean: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in clean:
            clean.append(text)
        if len(clean) >= limit:
            break
    if not clean:
        clean = ["下單前確認滑價、倉位大小、交易所深度與重大新聞風險。"]
    output = "\n".join(f"- {item}" for item in clean)
    return output[:1000]


def _entry_text(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{fmt_price(_as_float(value[0]))} - {fmt_price(_as_float(value[1]))}"
    return "-"


def _tp_text(payload: dict[str, Any]) -> str:
    targets: list[str] = []
    for key in ("TP1", "TP2", "TP3"):
        item = payload.get(key)
        if isinstance(item, dict):
            name = str(item.get("name") or key)
            price = fmt_price(_as_float(item.get("price")))
            portion = item.get("portion_pct")
            if portion is None:
                targets.append(f"{name} {price}")
            else:
                targets.append(f"{name} {price} ({float(portion):.0f}%)")
    return " | ".join(targets) or "-"


def _alert_signature(payload: dict[str, Any]) -> str:
    signature = {
        "symbol": payload.get("symbol"),
        "direction": payload.get("direction"),
        "trade_action": payload.get("trade_action"),
        "entry_zone": payload.get("entry_zone"),
        "stop": payload.get("stop"),
        "TP1": payload.get("TP1"),
        "TP2": payload.get("TP2"),
        "TP3": payload.get("TP3"),
        "rr": payload.get("rr", payload.get("RR")),
    }
    return json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _alert_timestamp(meta: dict[str, Any]) -> float:
    generated_at = meta.get("generated_at")
    if isinstance(generated_at, str) and generated_at.strip():
        value = generated_at.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
    return time.time()


def _iso_from_ts(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _cooldown_ts(value: Any) -> float | None:
    if isinstance(value, dict):
        raw = value.get("last_sent_ts")
    else:
        raw = value
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _active_cooldowns(cooldowns: dict[str, Any], now_ts: float) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in cooldowns.items():
        previous_ts = _cooldown_ts(value)
        if previous_ts is None:
            continue
        if now_ts - previous_ts <= SYMBOL_DIRECTION_COOLDOWN_SECONDS:
            output[key] = value
    return output


def _post_discord_webhook(
    webhook_url: str,
    payload: dict[str, Any],
    file_name: str,
    file_bytes: bytes,
    timeout_seconds: int,
) -> None:
    boundary = uuid.uuid4().hex
    body = _multipart_body(boundary, payload, file_name, file_bytes)
    request = urllib.request.Request(
        webhook_url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "User-Agent": "crypto-ict-bot/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status >= 300:
                raise DiscordWebhookError(f"HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        detail = exc.read(300).decode("utf-8", errors="replace")
        raise DiscordWebhookError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DiscordWebhookError(str(exc.reason)) from exc


def _multipart_body(boundary: str, payload: dict[str, Any], file_name: str, file_bytes: bytes) -> bytes:
    parts: list[bytes] = []
    parts.append(f"--{boundary}\r\n".encode("ascii"))
    parts.append(b'Content-Disposition: form-data; name="payload_json"\r\n')
    parts.append(b"Content-Type: application/json; charset=utf-8\r\n\r\n")
    parts.append(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode("ascii"))
    parts.append(f'Content-Disposition: form-data; name="files[0]"; filename="{file_name}"\r\n'.encode("ascii"))
    parts.append(b"Content-Type: application/json; charset=utf-8\r\n\r\n")
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(parts)


def _load_alert_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_alert_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _safe_file_stem(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean or "symbol"


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
