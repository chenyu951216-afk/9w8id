from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .exchanges import DataUnavailable, Ticker, create_auto_client, create_client
from .config import load_config
from .notifications import send_discord_executable_reports
from .paid_data import enrich_reports
from .report import print_table, standout_alerts, write_csv, write_html, write_json
from .scoring import score_symbol
from .signal_state import apply_signal_stability


TIMEFRAME_LIMITS = {
    "4h": 360,
    "1h": 500,
    "15m": 500,
    "5m": 500,
}


RULE_SUMMARY = """
圖片規則 -> 量化模組
1. HTF POI + MSS + IDM + FVG + OTE:
   4H/1H 掃流動性、折價/溢價、HTF FVG/OB，15m MSS/BOS、位移、FVG 回補、OTE 0.62-0.79。
2. 趨勢線破位:
   4H 找 2-3 觸點且橫跨約一週以上的趨勢線，破位且離線不遠加分。
3. ICT 核心:
   偵測 FVG、訂單塊、breaker/反向缺口重疊、實體缺口與未填補狀態。
4. FVG + BOS + 溢價/折價:
   低週期掃流動性後 BOS，FVG 位於折價/溢價與 RR>=2 加分。
5. 流動性與聰明資金:
   掃前高/前低後收回、單向大實體位移、留下 FVG 視為主力參與線索。
6. AMD:
   盤整吸籌 -> 掃 range 高/低操縱 -> 位移派發，符合階段越完整分數越高。
7. ICT 2022:
   高週期定方向、低週期找入場，MSS + FVG + RR 作為核心共振。
8. 供需高概率:
   有效掃蕩、失衡、結構突破、誘導/流動性目標一起評估。
9. Nexus Blast:
   5m 紐約 10-11 / 14-15 ET，倫敦高低點被掃後 BOS + FVG 加分。
"""


def _parse_symbols(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _resolve_auto_with_symbols(symbols: list[str], min_quote_volume: float) -> tuple[object, list[Ticker], list[str]]:
    errors: list[str] = []
    for exchange in ("bybit", "binance"):
        client = create_client(exchange)
        try:
            tickers = client.top_symbols(limit=1200, min_quote_volume=0.0)
            by_symbol = {ticker.symbol: ticker for ticker in tickers}
            missing = [symbol for symbol in symbols if symbol not in by_symbol]
            if missing:
                errors.append(f"{client.name}: missing requested symbols {', '.join(missing)}")
                continue
            selected = [by_symbol[symbol] for symbol in symbols if by_symbol[symbol].quote_volume >= min_quote_volume]
            low_volume = [symbol for symbol in symbols if by_symbol[symbol].quote_volume < min_quote_volume]
            if low_volume:
                errors.append(f"{client.name}: below min volume {', '.join(low_volume)}")
            return client, selected, errors
        except DataUnavailable as exc:
            errors.append(f"{client.name}: {exc}")
    raise DataUnavailable("Could not resolve requested symbols on public exchanges: " + " | ".join(errors))


def _resolve_tickers(args: argparse.Namespace) -> tuple[object, list[Ticker], list[str]]:
    symbols = _parse_symbols(args.symbols)
    if args.exchange == "auto":
        if symbols:
            return _resolve_auto_with_symbols(symbols, args.min_volume)
        return create_auto_client(args.top, args.min_volume)

    client = create_client(args.exchange)
    if symbols:
        tickers = client.top_symbols(limit=1200, min_quote_volume=0.0)
        by_symbol = {ticker.symbol: ticker for ticker in tickers}
        missing = [symbol for symbol in symbols if symbol not in by_symbol]
        if missing:
            raise DataUnavailable(f"{client.name} does not expose requested symbols: {', '.join(missing)}")
        selected = [by_symbol[symbol] for symbol in symbols if by_symbol[symbol].quote_volume >= args.min_volume]
        return client, selected, []
    return client, client.top_symbols(limit=args.top, min_quote_volume=args.min_volume), []


def _fetch_one(client: object, ticker: Ticker, btc_1h: list | None) -> tuple[object | None, str | None]:
    candles_by_tf: dict[str, list] = {}
    fetch_errors: list[str] = []
    for timeframe, limit in TIMEFRAME_LIMITS.items():
        try:
            candles_by_tf[timeframe] = client.klines(ticker.symbol, timeframe, limit)
        except DataUnavailable as exc:
            candles_by_tf[timeframe] = []
            fetch_errors.append(f"{timeframe}: {exc}")
    if not any(candles_by_tf.values()):
        return None, f"{ticker.symbol}: no candle data returned from {client.name}"
    report = score_symbol(client.name, ticker, candles_by_tf, btc_1h=btc_1h)
    if fetch_errors:
        report.metadata["fetch_errors"] = fetch_errors
        report.long.warnings.extend(fetch_errors)
        report.short.warnings.extend(fetch_errors)
    return report, None


def scan(args: argparse.Namespace) -> int:
    try:
        client, tickers, source_errors = _resolve_tickers(args)
    except (DataUnavailable, ValueError) as exc:
        print(f"資料來源失敗: {exc}", file=sys.stderr)
        return 2

    if not tickers:
        print("沒有符合條件的幣種。請降低 --min-volume 或指定 --symbols。", file=sys.stderr)
        return 2

    print(f"Using {client.name}; scanning {len(tickers)} symbols with real exchange candles...", file=sys.stderr)
    for item in source_errors:
        print(f"Source note: {item}", file=sys.stderr)

    btc_1h = None
    try:
        btc_1h = client.klines("BTCUSDT", "1h", 500)
    except DataUnavailable as exc:
        print(f"BTC context unavailable: {exc}", file=sys.stderr)

    reports = []
    errors: list[str] = []
    workers = max(1, min(args.workers, 8))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_fetch_one, client, ticker, btc_1h): ticker.symbol for ticker in tickers}
        for future in as_completed(future_map):
            report, error = future.result()
            if report is not None:
                reports.append(report)
            if error is not None:
                errors.append(error)

    config = load_config()
    config["scan"].update(
        {
            "exchange": args.exchange,
            "top": args.top,
            "symbols": args.symbols or "",
            "min_volume": args.min_volume,
            "min_score": args.min_score,
            "workers": args.workers,
        }
    )
    paid_meta = enrich_reports(reports, config)
    errors.extend(paid_meta.get("errors", []))
    signal_state = apply_signal_stability(reports)
    reports.sort(key=_report_sort_key, reverse=True)
    if args.min_score > 0:
        reports = [report for report in reports if report.score >= args.min_score]

    meta = {
        "exchange": client.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top": args.top,
        "min_volume": args.min_volume,
        "symbols": [ticker.symbol for ticker in tickers],
        "timeframe_limits": TIMEFRAME_LIMITS,
        "source_errors": source_errors,
        "scan_errors": errors,
        "paid_data": paid_meta,
        "signal_statistics": signal_state.get("statistics", {}),
        "standout_alerts": standout_alerts(reports),
        "refresh_minutes": config["server"]["refresh_minutes"],
    }
    discord_alerts = send_discord_executable_reports(reports, meta, config)
    meta["discord_alerts"] = discord_alerts
    errors.extend(discord_alerts.get("errors", []))
    meta["scan_errors"] = errors

    print(print_table(reports, limit=args.print_limit))
    if errors:
        print("\n以下幣種沒有完整 K 線資料，已略過或保守處理:", file=sys.stderr)
        for error in errors[:20]:
            print(f"- {error}", file=sys.stderr)

    if args.json:
        path = write_json(reports, args.json, meta)
        print(f"JSON report: {path.resolve()}", file=sys.stderr)
    if args.csv:
        path = write_csv(reports, args.csv)
        print(f"CSV report: {path.resolve()}", file=sys.stderr)
    if args.html:
        path = write_html(reports, args.html, meta)
        print(f"HTML report: {path.resolve()}", file=sys.stderr)
    return 0


def _report_sort_key(report: object) -> tuple[int, float, float]:
    metadata = getattr(report, "metadata", {})
    priority = metadata.get("signal_state", {}).get("priority_level", metadata.get("candidate_grade", ""))
    priority_rank = {"A": 5, "B": 4, "C": 3, "D": 2, "X": 1}.get(str(priority), 0)
    return priority_rank, float(getattr(report, "score", 0.0) or 0.0), float(getattr(report, "quote_volume_24h", 0.0) or 0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto-ict-bot",
        description="Real-data ICT/SMC crypto coin selection scorer.",
    )
    subparsers = parser.add_subparsers(dest="command")

    scan_parser = subparsers.add_parser("scan", help="scan symbols using public exchange data")
    scan_parser.add_argument("--exchange", choices=["auto", "bybit", "binance"], default="auto")
    scan_parser.add_argument("--top", type=int, default=70, help="top symbols by 24h quote volume")
    scan_parser.add_argument("--symbols", help="comma separated symbols, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    scan_parser.add_argument("--min-volume", type=float, default=20_000_000, help="minimum 24h quote volume in USDT")
    scan_parser.add_argument("--min-score", type=float, default=0.0, help="only keep reports above this score")
    scan_parser.add_argument("--workers", type=int, default=4)
    scan_parser.add_argument("--print-limit", type=int, default=20)
    scan_parser.add_argument("--json", default="reports/latest.json")
    scan_parser.add_argument("--csv", default="reports/latest.csv")
    scan_parser.add_argument("--html", default="reports/latest.html")
    scan_parser.set_defaults(func=scan)

    rules_parser = subparsers.add_parser("rules", help="print the image-rule mapping used by the scorer")
    rules_parser.set_defaults(func=lambda _args: print(RULE_SUMMARY.strip()) or 0)

    ui_parser = subparsers.add_parser("ui", help="start the Chinese web dashboard")
    ui_parser.add_argument("--host", default=None)
    ui_parser.add_argument("--port", type=int, default=None)
    ui_parser.add_argument("--no-browser", action="store_true")
    ui_parser.set_defaults(func=start_ui)
    return parser


def start_ui(args: argparse.Namespace) -> int:
    from .web_server import run_server

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["scan"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
