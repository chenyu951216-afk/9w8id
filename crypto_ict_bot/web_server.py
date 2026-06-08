from __future__ import annotations

import json
import mimetypes
import base64
import os
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

from .backtest_evaluator import calibration_metrics
from .cli import TIMEFRAME_LIMITS, _resolve_tickers
from .config import load_config, redacted_config, save_config
from .eth_strategy import attach_eth_strategy
from .exchanges import DataUnavailable
from .gate_executor import attach_gate_execution, check_gate_connection
from .notifications import send_discord_executable_reports
from .opportunity_ranker import enrich_opportunity_context, opportunity_sort_key
from .paid_data import check_provider_connections, enrich_reports, provider_statuses
from .report import report_to_dict, selected_side, standout_alerts, visible_warnings, write_csv, write_html, write_json
from .scoring import score_symbol
from .signal_state import apply_signal_stability
from .signal_logger import log_signal_snapshot


STATIC_DIR = Path(__file__).with_name("static")


class DashboardState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.config = load_config()
        self.reports = []
        self.meta: dict[str, Any] = {}
        self.errors: list[str] = []
        self.running = False
        self.last_started: str | None = None
        self.last_completed: str | None = None
        self.next_refresh_ts = time.time()
        self.scan_count = 0
        self.last_refresh_reason = "尚未刷新"
        self.state_version = 0
        self.cached_state_payload: dict[str, Any] = {}
        self.cached_state_version = -1
        self.scan_generation = 0
        self.thread: threading.Thread | None = None
        self.provider_status = provider_statuses(self.config)

    def refresh_config(self) -> None:
        with self.lock:
            self.config = load_config()
            self.provider_status = provider_statuses(self.config)
            self.next_refresh_ts = time.time() + int(self.config["server"]["refresh_minutes"]) * 60
            self.cached_state_payload = {}
            self.cached_state_version = -1

    def snapshot(self, since: int | None = None) -> dict[str, Any]:
        self.start_due_scan()
        with self.lock:
            if self.cached_state_version != self.state_version or not self.cached_state_payload:
                self._rebuild_cached_state_locked()
            if since is not None and since == self.state_version and self.cached_state_version == self.state_version:
                heartbeat = self._heartbeat_locked()
                heartbeat["not_modified"] = True
                return heartbeat
            payload = dict(self.cached_state_payload)
            payload.update(self._runtime_payload_locked())
            payload["not_modified"] = False
            return payload

    def heartbeat(self) -> dict[str, Any]:
        self.start_due_scan()
        with self.lock:
            return self._heartbeat_locked()

    def _runtime_payload_locked(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "last_started": self.last_started,
            "last_completed": self.last_completed,
            "next_refresh_ts": self.next_refresh_ts,
            "scan_count": self.scan_count,
            "state_version": self.state_version,
            "last_refresh_reason": self.last_refresh_reason,
            "server_time": time.time(),
        }

    def _heartbeat_locked(self) -> dict[str, Any]:
        passing_score = float(self.config["scan"]["passing_score"])
        provider_gap_count = sum(1 for provider in self.provider_status if _provider_has_gap(provider))
        compact_stats = _compact_signal_statistics(self.meta.get("signal_statistics", {}))
        payload = {
            **self._runtime_payload_locked(),
            "report_count": len(self.reports),
            "passing_score": passing_score,
            "passed_count": sum(1 for report in self.reports if report.score >= passing_score),
            "exchange": self.meta.get("exchange", "-"),
            "provider_gap_count": provider_gap_count,
            "signal_statistics": compact_stats,
            "standout_alerts": self.meta.get("standout_alerts", []),
            "errors": self.errors[-3:],
        }
        return payload

    def _rebuild_cached_state_locked(self) -> None:
        passing_score = float(self.config["scan"]["passing_score"])
        compact_stats = _compact_signal_statistics(self.meta.get("signal_statistics", {}))
        compact_meta = _compact_dashboard_meta(self.meta, compact_stats)
        payload = {
            "config": redacted_config(self.config),
            "exchange": self.meta.get("exchange", "-"),
            "meta": compact_meta,
            "signal_statistics": compact_stats,
            "standout_alerts": self.meta.get("standout_alerts", []),
            "errors": list(self.errors),
            "providers": list(self.provider_status),
            "passing_score": passing_score,
        }
        reports = []
        for idx, report in enumerate(list(self.reports), start=1):
            try:
                item = report_to_dict(report)
                item["rank"] = idx
                item["passed"] = report.score >= passing_score
                item["selected_reasons"] = selected_side(report).reasons
                item["selected_warnings"] = visible_warnings(selected_side(report))
                reports.append(_compact_dashboard_item(item, self.state_version))
            except Exception as exc:
                payload["errors"].append(f"{getattr(report, 'symbol', 'unknown')}: dashboard payload failed: {exc}")
        payload["reports"] = reports
        self.cached_state_payload = payload
        self.cached_state_version = self.state_version

    def start_scan(self, reason: str = "手動刷新") -> bool:
        with self.lock:
            if self.running:
                return False
            self.scan_generation += 1
            generation = self.scan_generation
            self.running = True
            self.last_started = datetime.now(timezone.utc).isoformat()
            self.last_refresh_reason = reason
            self.errors = []
            self.thread = threading.Thread(target=self._scan_worker, args=(reason, generation), daemon=True)
            self.thread.start()
            return True

    def start_due_scan(self) -> bool:
        with self.lock:
            if self.running and self.last_started:
                started = datetime.fromisoformat(self.last_started)
                age = (datetime.now(timezone.utc) - started).total_seconds()
                if age > self._scan_timeout_seconds():
                    self.scan_generation += 1
                    self.running = False
                    self.next_refresh_ts = time.time() + self._scan_cooldown_seconds()
                    self.errors.append(f"上一輪掃描超過 {self._scan_timeout_seconds():.0f}s，已釋放排程並保留舊榜單。")
            due = time.time() >= self.next_refresh_ts
            running = self.running
            minutes = int(self.config["server"]["refresh_minutes"])
        if due and not running:
            return self.start_scan(f"{minutes} 分鐘自動刷新")
        return False

    def _scan_timeout_seconds(self) -> float:
        minutes = int(self.config["server"].get("refresh_minutes", 5) or 5)
        return max(100.0, min(285.0, minutes * 60.0 - 10.0))

    def _scan_cooldown_seconds(self) -> float:
        minutes = int(self.config["server"].get("refresh_minutes", 5) or 5)
        return max(45.0, min(120.0, minutes * 60.0))

    def _scan_worker(self, reason: str, generation: int) -> None:
        config = load_config()
        reports = []
        meta: dict[str, Any] = {"reason": reason}
        errors: list[str] = []
        provider_status = provider_statuses(config)
        try:
            args = SimpleNamespace(**config["scan"])
            args.min_score = 0.0
            client, tickers, source_errors = _resolve_tickers(args)
            meta.update(
                {
                    "exchange": client.name,
                    "symbols": [ticker.symbol for ticker in tickers],
                    "timeframe_limits": TIMEFRAME_LIMITS,
                    "source_errors": source_errors,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            errors.extend(source_errors)
            btc_1h = None
            try:
                btc_1h = client.klines("BTCUSDT", "1h", 500)
            except DataUnavailable as exc:
                errors.append(f"BTC context unavailable: {exc}")

            workers = max(1, min(int(config["scan"].get("workers", 4)), 8))
            scan_timeout = max(45.0, min(180.0, int(config["server"]["refresh_minutes"]) * 60.0 - 80.0))
            scan_deadline = time.monotonic() + scan_timeout
            executor = ThreadPoolExecutor(max_workers=workers)
            future_map = {executor.submit(_fetch_symbol, client, ticker, btc_1h, scan_deadline): ticker.symbol for ticker in tickers}
            try:
                for future in as_completed(future_map, timeout=scan_timeout):
                    report, error = future.result()
                    if report is not None:
                        reports.append(report)
                    if error:
                        errors.append(error)
            except FuturesTimeoutError:
                pending = [symbol for future, symbol in future_map.items() if not future.done()]
                errors.append(f"交易所 K 線掃描超過 {scan_timeout:.0f}s，已先用完成的 {len(reports)} 個幣種更新；未完成 {len(pending)} 個。")
                for future in future_map:
                    if not future.done():
                        future.cancel()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            if reports:
                reports.sort(key=_report_sort_key, reverse=True)
                with self.lock:
                    active_generation = generation == self.scan_generation
                if active_generation:
                    # Do not publish raw scorer reports here. They have not yet
                    # passed paid-data enrichment, signal stability, or the
                    # layered opportunity pipeline, so the dashboard would
                    # temporarily lose status/trend/strategy/entry fields during
                    # every refresh. Keep the last complete board visible until
                    # the new board is fully enriched below.
                    paid_meta = enrich_reports(reports, config)
                    provider_status = paid_meta.get("providers", provider_status)
                    errors.extend(paid_meta.get("errors", []))
                    opportunity_meta = enrich_opportunity_context(reports)
                    signal_state = apply_signal_stability(reports)
                    opportunity_meta = enrich_opportunity_context(reports)
                    eth_meta = attach_eth_strategy(reports, config)
                    gate_meta = attach_gate_execution(reports, config)
                    reports.sort(key=_report_sort_key, reverse=True)
                    meta["paid_data"] = paid_meta
                    meta["signal_statistics"] = signal_state.get("statistics", {})
                    meta.update(opportunity_meta)
                    meta["eth_analysis"] = eth_meta
                    meta["gate_trading"] = gate_meta
                    meta["calibration_metrics"] = calibration_metrics(signal_state)
                    meta["standout_alerts"] = standout_alerts(reports)
                    meta["scan_errors"] = errors
                    meta["passing_score"] = config["scan"]["passing_score"]
                    meta["refresh_minutes"] = config["server"]["refresh_minutes"]
                    with self.lock:
                        active_generation = generation == self.scan_generation
                    if active_generation:
                        discord_alerts = send_discord_executable_reports(reports, meta, config)
                        meta["discord_alerts"] = discord_alerts
                        errors.extend(discord_alerts.get("errors", []))
                        meta["scan_errors"] = errors
                        log_signal_snapshot(reports, meta)
                        write_json(reports, "reports/latest.json", meta)
                        write_csv(reports, "reports/latest.csv")
                        write_html(reports, "reports/latest.html", meta)
                    else:
                        errors.append("本輪掃描在補資料期間逾時釋放，結果不覆蓋目前榜單。")
                else:
                    errors.append("本輪掃描已逾時釋放，結果不覆蓋目前榜單。")
            else:
                errors.append("本輪沒有任何幣種完成掃描，保留上一輪榜單。")
        except Exception as exc:
            errors.append(str(exc))
        finally:
            with self.lock:
                if generation == self.scan_generation:
                    self.config = config
                    if reports:
                        self.reports = reports
                    self.meta = meta
                    self.errors = errors
                    self.running = False
                    self.last_completed = datetime.now(timezone.utc).isoformat()
                    if reports:
                        self.scan_count += 1
                    if reports or errors:
                        self.state_version += 1
                    self.next_refresh_ts = time.time() + int(config["server"]["refresh_minutes"]) * 60
                    self.provider_status = provider_status
                    self._rebuild_cached_state_locked()


def _fetch_symbol(client: Any, ticker: Any, btc_1h: list | None, deadline: float | None = None) -> tuple[Any | None, str | None]:
    candles_by_tf: dict[str, list] = {}
    fetch_errors: list[str] = []
    for timeframe, limit in TIMEFRAME_LIMITS.items():
        if deadline is not None and time.monotonic() >= deadline:
            return None, f"{ticker.symbol}: 本輪掃描已達時間上限，跳過剩餘 K 線"
        try:
            candles_by_tf[timeframe] = client.klines(ticker.symbol, timeframe, limit)
        except DataUnavailable as exc:
            candles_by_tf[timeframe] = []
            fetch_errors.append(f"{ticker.symbol} {timeframe}: {exc}")
    if not any(candles_by_tf.values()):
        return None, f"{ticker.symbol}: 無法取得任何真實 K 線資料"
    report = score_symbol(client.name, ticker, candles_by_tf, btc_1h=btc_1h)
    if fetch_errors:
        report.metadata["fetch_errors"] = fetch_errors
        report.long.warnings.extend(fetch_errors)
        report.short.warnings.extend(fetch_errors)
    return report, None


def _compact_dashboard_item(item: dict[str, Any], state_version: int | None = None) -> dict[str, Any]:
    """Keep the live dashboard payload small enough for frequent refreshes."""
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    paid_data = metadata.get("paid_data") if isinstance(metadata.get("paid_data"), dict) else {}
    compact_metadata: dict[str, Any] = {}
    if paid_data:
        compact_metadata["paid_data"] = _compact_paid_data(paid_data)
    if metadata.get("direction_conflict"):
        compact_metadata["direction_conflict"] = metadata.get("direction_conflict")
    if metadata.get("candidate_grade"):
        compact_metadata["candidate_grade"] = metadata.get("candidate_grade")
    if metadata.get("candidate_status"):
        compact_metadata["candidate_status"] = metadata.get("candidate_status")
    item["metadata"] = compact_metadata

    signal_state = item.get("signal_state") if isinstance(item.get("signal_state"), dict) else {}
    if signal_state:
        compact_signal = _compact_signal_state(signal_state)
        item["signal_state"] = compact_signal
        item.pop("future_validation", None)
    item.pop("signal_lifecycle", None)
    item.pop("setup_stats", None)
    item.pop("paid_data_status", None)
    item.pop("display_reason", None)
    item.pop("warning_reason", None)
    item.pop("invalid_reason", None)
    if item.get("selected_reasons"):
        item.pop("reasons", None)

    if state_version is not None:
        symbol = str(item.get("symbol") or "")
        action = str(item.get("trade_action") or "")
        execution_status = str(item.get("execution_status") or "")
        item["state_version"] = state_version
        item["row_cache_key"] = f"{symbol}:{state_version}:{action}:{execution_status}"

    model = item.get("score_model") if isinstance(item.get("score_model"), dict) else {}
    if model:
        item["score_model"] = _compact_score_model(model)
        item.pop("score_adjustments", None)
        item.pop("validation_adjustments", None)
    return item


def _compact_dashboard_meta(meta: dict[str, Any], compact_stats: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "exchange",
        "generated_at",
        "source_errors",
        "scan_errors",
        "paid_data_status",
        "standout_alerts",
        "no_trade_diagnostics",
        "top_long_candidates",
        "top_short_candidates",
        "top_watchlist",
        "top_setup_ready",
        "eth_analysis",
        "gate_trading",
    }
    output = {key: meta.get(key) for key in keep if key in meta}
    if compact_stats:
        output["signal_statistics"] = compact_stats
    return output


def _compact_signal_statistics(stats: Any) -> dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    keep = {
        "total_signals",
        "active_signals",
        "failed_signals",
        "long_signals",
        "short_signals",
        "grade_counts",
        "status_counts",
        "accuracy",
        "average_mfe",
        "average_mae",
    }
    return {key: stats.get(key) for key in keep if key in stats}


def _compact_paid_data(paid_data: dict[str, Any]) -> dict[str, Any]:
    values = paid_data.get("values") if isinstance(paid_data.get("values"), dict) else {}
    public = values.get("exchange_public_derivatives") if isinstance(values.get("exchange_public_derivatives"), dict) else {}
    compact_values: dict[str, Any] = {}
    if public:
        keep = {
            "source",
            "source_exchange",
            "funding_rate",
            "funding_time",
            "open_interest",
            "open_interest_time",
            "open_interest_previous_time",
            "open_interest_change_pct",
            "fetched_at",
            "spread_pct",
            "orderbook",
            "trade_flow",
            "taker_long_short_ratio",
            "global_long_short_ratio",
            "top_long_short_position_ratio",
            "partial_errors",
        }
        compact_values["exchange_public_derivatives"] = {key: public.get(key) for key in keep if key in public}
    for key in (
        "external_strategy_context",
        "coinalyze",
        "coinglass_taker_buy_sell",
        "coinglass_long_short_ratio",
        "coinglass_liquidation_sum",
        "coinglass_liquidation_heatmap",
        "coinglass_liquidation_map",
        "coinglass_orderbook_heatmap",
        "glassnode_exchange_netflow_usd",
        "cryptoquant_exchange_netflow",
    ):
        if key in values:
            compact_values[key] = _compact_paid_value(values[key])
    output = {
        "providers": paid_data.get("providers", []),
        "values": compact_values,
    }
    readiness = paid_data.get("configured_api_readiness")
    if isinstance(readiness, dict):
        output["configured_api_readiness"] = readiness
    errors = paid_data.get("errors")
    if isinstance(errors, list) and errors:
        output["errors"] = errors[:5]
    return output


def _compact_paid_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_compact_paid_value(item) for item in value[-8:]]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, list):
                compact[key] = [_compact_paid_value(entry) for entry in item[-8:]]
            elif isinstance(item, dict):
                compact[key] = _compact_paid_value(item)
            else:
                compact[key] = item
        return compact
    return value


def _compact_signal_state(signal_state: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "symbol",
        "direction",
        "current_score",
        "previous_score",
        "score_change",
        "highest_score",
        "lowest_score",
        "score_trend",
        "confirm_count",
        "miss_count",
        "signal_present",
        "signal_seen_count",
        "signal_absent_count",
        "executable_confirm_count",
        "executable_miss_count",
        "execution_state",
        "can_execute_now",
        "first_seen_time",
        "last_seen_time",
        "signal_age_minutes",
        "status",
        "lifecycle_state",
        "priority_level",
        "opportunity_score",
        "current_price",
        "entry_zone",
        "distance_to_entry_zone",
        "htf_bias",
        "market_filter_result",
        "liquidity_filter_result",
        "market_warning",
        "liquidity_warning",
        "setup_tags",
        "display_reason",
        "warning_reason",
        "invalid_reason",
        "stable_action",
        "raw_action_label",
        "raw_action_reason",
        "stability_reason",
        "behavior_analysis",
        "trade_plan",
        "layered_trade_plan",
        "trade_signal_state",
        "trade_thesis",
        "next_trigger",
        "blockers",
        "direction_analysis",
    }
    compact = {key: signal_state.get(key) for key in keep if key in signal_state}
    history = signal_state.get("score_history")
    if isinstance(history, list):
        compact["score_history"] = history[-6:]
    validation = signal_state.get("future_validation")
    if isinstance(validation, dict):
        compact["future_validation"] = _compact_future_validation(validation)
    return compact


def _compact_future_validation(validation: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "checked",
        "signal_time",
        "signal_price",
        "direction",
        "score",
        "entry_zone",
        "stop",
        "entry_distance_pct",
        "funding",
        "oi_change",
        "btc_trend",
        "atr_pct",
        "volume_ratio",
        "mfe",
        "mae",
        "max_favorable_move",
        "max_adverse_move",
        "entry_observed",
        "tp1_hit_first",
        "sl_hit_first",
        "first_touch",
        "direction_correct",
        "failed_reason",
        "after_1_candle",
        "after_3_candles",
        "after_6_candles",
        "after_12_candles",
    }
    return {key: validation.get(key) for key in keep if key in validation}


def _compact_score_model(model: dict[str, Any]) -> dict[str, Any]:
    compact = dict(model)
    for key in ("score_adjustments", "validation_adjustments"):
        value = compact.get(key)
        if isinstance(value, list) and len(value) > 8:
            compact[key] = value[:8] + [f"另有 {len(value) - 8} 則校準訊息，請查看 latest.json"]
    return compact


def _report_sort_key(report: Any) -> tuple[int, float, float, float]:
    return opportunity_sort_key(report)


def scheduler_loop(state: DashboardState) -> None:
    while True:
        time.sleep(2)
        state.start_due_scan()


def _provider_has_gap(provider: dict[str, Any]) -> bool:
    configured = bool(provider.get("configured"))
    enabled = bool(provider.get("enabled"))
    readable = provider.get("readable")
    success_count = int(provider.get("success_count") or 0)
    failure_count = int(provider.get("failure_count") or 0)
    if not configured and not enabled:
        return False
    if enabled and readable is False:
        return True
    return success_count > 0 and failure_count > 0


class DashboardHandler(BaseHTTPRequestHandler):
    state: DashboardState

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if not self._authorized():
            return
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self._serve_static("index.html")
        if path == "/api/state":
            since = None
            raw_since = parse_qs(parsed.query).get("since", [None])[0]
            if raw_since not in (None, ""):
                try:
                    since = int(raw_since)
                except (TypeError, ValueError):
                    since = None
            if since is not None:
                self.state.start_due_scan()
                with self.state.lock:
                    if since == self.state.state_version:
                        heartbeat = self.state._heartbeat_locked()
                        heartbeat["not_modified"] = True
                        return self._json(heartbeat)
            return self._json(self.state.snapshot(since=since))
        if path == "/api/heartbeat":
            return self._json(self.state.heartbeat())
        if path == "/api/providers":
            return self._json({"providers": provider_statuses(load_config())})
        if path.startswith("/static/"):
            return self._serve_static(path.removeprefix("/static/"))
        if path == "/reports/latest.json":
            return self._serve_file(Path("reports/latest.json"))
        if path == "/reports/latest.csv":
            return self._serve_file(Path("reports/latest.csv"))
        if path == "/reports/latest.html":
            return self._serve_file(Path("reports/latest.html"))
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path == "/api/scan":
            started = self.state.start_scan("手動刷新")
            return self._json({"started": started, "message": "已開始刷新" if started else "目前正在刷新"})
        if self.path == "/api/settings":
            payload = self._read_json()
            config = save_config(payload)
            self.state.refresh_config()
            return self._json({"ok": True, "config": redacted_config(config)})
        if self.path == "/api/check-apis":
            config = load_config()
            statuses = check_provider_connections(config)
            with self.state.lock:
                self.state.provider_status = statuses
            return self._json({"providers": statuses})
        if self.path == "/api/check-gate":
            config = load_config()
            return self._json({"gate": check_gate_connection(config)})
        self.send_error(404, "Not found")

    def _authorized(self) -> bool:
        password = os.environ.get("APP_PASSWORD", "")
        if not password:
            return True
        username = os.environ.get("APP_USERNAME", "admin")
        auth = self.headers.get("Authorization", "")
        expected_raw = f"{username}:{password}".encode("utf-8")
        expected = "Basic " + base64.b64encode(expected_raw).decode("ascii")
        if auth == expected:
            return True
        raw = "需要登入".encode("utf-8")
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="ICT Coin Bot"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        return False

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _serve_static(self, name: str) -> None:
        safe = Path(name).name if "/" not in name and "\\" not in name else Path(name)
        path = STATIC_DIR / safe
        self._serve_file(path)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, "Not found")
            return
        raw = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix in {".html", ".css", ".js", ".json", ".csv"}:
            mime = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".csv": "text/csv; charset=utf-8",
            }[path.suffix]
        self.send_response(200)
        self.send_header("Content-Type", mime)
        if str(path).startswith("reports") or path.suffix in {".html", ".css", ".js", ".json", ".csv"}:
            self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_server(host: str | None = None, port: int | None = None, open_browser: bool | None = None) -> tuple[str, int]:
    state = DashboardState()
    host = host or state.config["server"]["host"]
    port = int(port or state.config["server"]["port"])
    if open_browser is None:
        open_browser = bool(state.config["server"].get("open_browser", True))
    DashboardHandler.state = state

    server = None
    selected_port = port
    for candidate in range(port, min(port + 20, 65535)):
        try:
            server = ThreadingHTTPServer((host, candidate), DashboardHandler)
            selected_port = candidate
            break
        except OSError:
            continue
    if server is None:
        raise OSError("找不到可用的本機連接埠")

    threading.Thread(target=scheduler_loop, args=(state,), daemon=True).start()
    state.start_scan("啟動時首次刷新")
    url = f"http://{host}:{selected_port}"
    print(f"ICT 選幣 UI 已啟動: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    server.serve_forever()
    return host, selected_port
