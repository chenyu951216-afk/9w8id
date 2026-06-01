from __future__ import annotations

import json
import mimetypes
import base64
import os
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .cli import TIMEFRAME_LIMITS, _resolve_tickers
from .config import load_config, redacted_config, save_config
from .exchanges import DataUnavailable
from .paid_data import check_provider_connections, enrich_reports, provider_statuses
from .report import report_to_dict, selected_side, visible_warnings, write_csv, write_html, write_json
from .scoring import score_symbol
from .signal_state import apply_signal_stability


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
            reports = []
            for idx, report in enumerate(self.reports, start=1):
                item = report_to_dict(report)
                item["rank"] = idx
                item["passed"] = report.score >= passing_score
                item["selected_reasons"] = selected_side(report).reasons
                item["selected_warnings"] = visible_warnings(selected_side(report))
                reports.append(item)
            return {
                "config": redacted_config(self.config),
                "running": self.running,
                "last_started": self.last_started,
                "last_completed": self.last_completed,
                "next_refresh_ts": self.next_refresh_ts,
                "scan_count": self.scan_count,
                "last_refresh_reason": self.last_refresh_reason,
                "server_time": time.time(),
                "reports": reports,
                "meta": self.meta,
                "errors": self.errors,
                "providers": self.provider_status,
                "passing_score": passing_score,
            }

    def start_scan(self, reason: str = "手動刷新") -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.last_started = datetime.now(timezone.utc).isoformat()
            self.last_refresh_reason = reason
            self.errors = []
            self.thread = threading.Thread(target=self._scan_worker, args=(reason,), daemon=True)
            self.thread.start()
            return True

    def start_due_scan(self) -> bool:
        with self.lock:
            due = time.time() >= self.next_refresh_ts
            running = self.running
            minutes = int(self.config["server"]["refresh_minutes"])
        if due and not running:
            return self.start_scan(f"{minutes} 分鐘自動刷新")
        return False

    def _scan_worker(self, reason: str) -> None:
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
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {executor.submit(_fetch_symbol, client, ticker, btc_1h): ticker.symbol for ticker in tickers}
                for future in as_completed(future_map):
                    report, error = future.result()
                    if report is not None:
                        reports.append(report)
                    if error:
                        errors.append(error)

            reports.sort(key=lambda item: (item.score, item.quote_volume_24h), reverse=True)
            paid_meta = enrich_reports(reports, config)
            provider_status = paid_meta.get("providers", provider_status)
            errors.extend(paid_meta.get("errors", []))
            apply_signal_stability(reports)
            reports.sort(key=lambda item: (item.score, item.quote_volume_24h), reverse=True)
            meta["paid_data"] = paid_meta
            meta["scan_errors"] = errors
            meta["passing_score"] = config["scan"]["passing_score"]
            write_json(reports, "reports/latest.json", meta)
            write_csv(reports, "reports/latest.csv")
            write_html(reports, "reports/latest.html", meta)
        except Exception as exc:
            errors.append(str(exc))
        finally:
            with self.lock:
                self.config = config
                self.reports = reports
                self.meta = meta
                self.errors = errors
                self.running = False
                self.last_completed = datetime.now(timezone.utc).isoformat()
                if reports:
                    self.scan_count += 1
                self.next_refresh_ts = time.time() + int(config["server"]["refresh_minutes"]) * 60
                self.provider_status = provider_status


def _fetch_symbol(client: Any, ticker: Any, btc_1h: list | None) -> tuple[Any | None, str | None]:
    candles_by_tf: dict[str, list] = {}
    fetch_errors: list[str] = []
    for timeframe, limit in TIMEFRAME_LIMITS.items():
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


def scheduler_loop(state: DashboardState) -> None:
    while True:
        time.sleep(2)
        state.start_due_scan()


class DashboardHandler(BaseHTTPRequestHandler):
    state: DashboardState

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if not self._authorized():
            return
        if self.path == "/" or self.path.startswith("/?"):
            return self._serve_static("index.html")
        if self.path == "/api/state":
            return self._json(self.state.snapshot())
        if self.path == "/api/providers":
            return self._json({"providers": provider_statuses(load_config())})
        if self.path.startswith("/static/"):
            return self._serve_static(self.path.removeprefix("/static/"))
        if self.path == "/reports/latest.json":
            return self._serve_file(Path("reports/latest.json"))
        if self.path == "/reports/latest.csv":
            return self._serve_file(Path("reports/latest.csv"))
        if self.path == "/reports/latest.html":
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
