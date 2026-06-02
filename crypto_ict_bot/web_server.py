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

from .backtest_evaluator import calibration_metrics
from .cli import TIMEFRAME_LIMITS, _resolve_tickers
from .config import load_config, redacted_config, save_config
from .exchanges import DataUnavailable
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
        self.scan_generation = 0
        self.thread: threading.Thread | None = None
        self.provider_status = provider_statuses(self.config)

    def refresh_config(self) -> None:
        with self.lock:
            self.config = load_config()
            self.provider_status = provider_statuses(self.config)
            self.next_refresh_ts = time.time() + int(self.config["server"]["refresh_minutes"]) * 60

    def snapshot(self) -> dict[str, Any]:
        self.start_due_scan()
        with self.lock:
            passing_score = float(self.config["scan"]["passing_score"])
            report_source = list(self.reports)
            payload = {
                "config": redacted_config(self.config),
                "running": self.running,
                "last_started": self.last_started,
                "last_completed": self.last_completed,
                "next_refresh_ts": self.next_refresh_ts,
                "scan_count": self.scan_count,
                "state_version": self.state_version,
                "last_refresh_reason": self.last_refresh_reason,
                "server_time": time.time(),
                "meta": dict(self.meta),
                "signal_statistics": self.meta.get("signal_statistics", {}),
                "standout_alerts": self.meta.get("standout_alerts", []),
                "errors": list(self.errors),
                "providers": list(self.provider_status),
                "passing_score": passing_score,
            }
        reports = []
        for idx, report in enumerate(report_source, start=1):
            item = report_to_dict(report)
            item["rank"] = idx
            item["passed"] = report.score >= passing_score
            item["selected_reasons"] = selected_side(report).reasons
            item["selected_warnings"] = visible_warnings(selected_side(report))
            item = _compact_dashboard_item(item)
            reports.append(item)
        payload["reports"] = reports
        return payload

    def heartbeat(self) -> dict[str, Any]:
        self.start_due_scan()
        with self.lock:
            passing_score = float(self.config["scan"]["passing_score"])
            provider_gap_count = sum(1 for provider in self.provider_status if _provider_has_gap(provider))
            return {
                "running": self.running,
                "last_started": self.last_started,
                "last_completed": self.last_completed,
                "next_refresh_ts": self.next_refresh_ts,
                "scan_count": self.scan_count,
                "state_version": self.state_version,
                "last_refresh_reason": self.last_refresh_reason,
                "server_time": time.time(),
                "report_count": len(self.reports),
                "passing_score": passing_score,
                "passed_count": sum(1 for report in self.reports if report.score >= passing_score),
                "exchange": self.meta.get("exchange", "-"),
                "provider_gap_count": provider_gap_count,
                "signal_statistics": self.meta.get("signal_statistics", {}),
                "standout_alerts": self.meta.get("standout_alerts", []),
                "errors": self.errors[-3:],
            }

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
                    partial_meta = dict(meta)
                    partial_meta["scan_errors"] = errors
                    partial_meta["passing_score"] = config["scan"]["passing_score"]
                    partial_meta["refresh_minutes"] = config["server"]["refresh_minutes"]
                    with self.lock:
                        if generation == self.scan_generation:
                            self.reports = list(reports)
                            self.meta = partial_meta
                            self.errors = list(errors)
                            self.provider_status = provider_status
                            self.state_version += 1
                    paid_meta = enrich_reports(reports, config)
                    provider_status = paid_meta.get("providers", provider_status)
                    errors.extend(paid_meta.get("errors", []))
                    signal_state = apply_signal_stability(reports)
                    opportunity_meta = enrich_opportunity_context(reports)
                    reports.sort(key=_report_sort_key, reverse=True)
                    meta["paid_data"] = paid_meta
                    meta["signal_statistics"] = signal_state.get("statistics", {})
                    meta.update(opportunity_meta)
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
                        self.state_version += 1
                    self.next_refresh_ts = time.time() + int(config["server"]["refresh_minutes"]) * 60
                    self.provider_status = provider_status


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


def _compact_dashboard_item(item: dict[str, Any]) -> dict[str, Any]:
    """Keep the live dashboard payload small enough for frequent refreshes."""
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    paid_data = metadata.get("paid_data") if isinstance(metadata.get("paid_data"), dict) else {}
    compact_metadata: dict[str, Any] = {}
    if paid_data:
        compact_metadata["paid_data"] = paid_data
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
        item["signal_lifecycle"] = compact_signal

    model = item.get("score_model") if isinstance(item.get("score_model"), dict) else {}
    if model:
        item["score_model"] = _compact_score_model(model)
    return item


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
    text = str(provider.get("state") or "")
    return any(fragment in text for fragment in ("未設定", "讀不到", "失敗", "部分"))


class DashboardHandler(BaseHTTPRequestHandler):
    state: DashboardState

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if not self._authorized():
            return
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._serve_static("index.html")
        if path == "/api/state":
            return self._json(self.state.snapshot())
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
