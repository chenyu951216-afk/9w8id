from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import Candle, utc_from_ms


class DataUnavailable(RuntimeError):
    """Raised when a real exchange endpoint cannot provide requested data."""


STABLE_BASES = {
    "USDC",
    "BUSD",
    "TUSD",
    "USDP",
    "FDUSD",
    "DAI",
    "EUR",
    "TRY",
    "BRL",
    "GBP",
    "JPY",
    "AUD",
}


INTERVALS_BYBIT = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "1d": "D",
}


def _json_get(base_url: str, path: str, params: dict[str, Any] | None = None, timeout: int = 6) -> Any:
    query = f"?{urlencode(params or {})}" if params else ""
    url = f"{base_url}{path}{query}"
    request = Request(
        url,
        headers={
            "User-Agent": "crypto-ict-selection-bot/0.1",
            "Accept": "application/json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 1:
                time.sleep(0.35 * (attempt + 1))
    raise DataUnavailable(f"Cannot fetch real data from {url}: {last_error}")


@dataclass(frozen=True)
class Ticker:
    symbol: str
    last_price: float
    quote_volume: float
    change_pct: float


class ExchangeClient:
    name = "base"

    def top_symbols(self, limit: int, min_quote_volume: float) -> list[Ticker]:
        raise NotImplementedError

    def klines(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        raise NotImplementedError


class BinanceFuturesClient(ExchangeClient):
    name = "binance-futures"
    base_url = "https://fapi.binance.com"

    def _tradable_symbols(self) -> set[str]:
        payload = _json_get(self.base_url, "/fapi/v1/exchangeInfo")
        symbols = set()
        for item in payload.get("symbols", []):
            if item.get("status") != "TRADING":
                continue
            if item.get("contractType") != "PERPETUAL":
                continue
            if item.get("quoteAsset") != "USDT":
                continue
            base = item.get("baseAsset", "")
            if base in STABLE_BASES:
                continue
            symbols.add(item.get("symbol", ""))
        if not symbols:
            raise DataUnavailable("Binance Futures exchangeInfo returned no tradable USDT perpetual symbols.")
        return symbols

    def top_symbols(self, limit: int, min_quote_volume: float) -> list[Ticker]:
        tradable = self._tradable_symbols()
        payload = _json_get(self.base_url, "/fapi/v1/ticker/24hr")
        tickers: list[Ticker] = []
        for item in payload:
            symbol = item.get("symbol", "")
            if symbol not in tradable:
                continue
            quote_volume = float(item.get("quoteVolume") or 0.0)
            if quote_volume < min_quote_volume:
                continue
            tickers.append(
                Ticker(
                    symbol=symbol,
                    last_price=float(item.get("lastPrice") or 0.0),
                    quote_volume=quote_volume,
                    change_pct=float(item.get("priceChangePercent") or 0.0),
                )
            )
        tickers.sort(key=lambda t: t.quote_volume, reverse=True)
        return tickers[:limit]

    def klines(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        payload = _json_get(
            self.base_url,
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        candles: list[Candle] = []
        for row in payload:
            candles.append(
                Candle(
                    open_time=utc_from_ms(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    close_time=utc_from_ms(row[6]),
                )
            )
        if not candles:
            raise DataUnavailable(f"Binance Futures returned no candles for {symbol} {interval}.")
        return candles


class BybitLinearClient(ExchangeClient):
    name = "bybit-linear"
    base_url = "https://api.bybit.com"

    def _tradable_symbols(self) -> set[str]:
        symbols: set[str] = set()
        cursor = ""
        while True:
            params = {"category": "linear", "limit": "1000"}
            if cursor:
                params["cursor"] = cursor
            payload = _json_get(self.base_url, "/v5/market/instruments-info", params)
            if str(payload.get("retCode")) != "0":
                raise DataUnavailable(f"Bybit instruments-info error: {payload.get('retMsg')}")
            result = payload.get("result", {})
            for item in result.get("list", []):
                if item.get("status") != "Trading":
                    continue
                if item.get("quoteCoin") != "USDT":
                    continue
                base = item.get("baseCoin", "")
                if base in STABLE_BASES:
                    continue
                symbols.add(item.get("symbol", ""))
            cursor = result.get("nextPageCursor") or ""
            if not cursor:
                break
        if not symbols:
            raise DataUnavailable("Bybit instruments-info returned no tradable USDT linear symbols.")
        return symbols

    def top_symbols(self, limit: int, min_quote_volume: float) -> list[Ticker]:
        tradable = self._tradable_symbols()
        payload = _json_get(self.base_url, "/v5/market/tickers", {"category": "linear"})
        if str(payload.get("retCode")) != "0":
            raise DataUnavailable(f"Bybit tickers error: {payload.get('retMsg')}")
        tickers: list[Ticker] = []
        for item in payload.get("result", {}).get("list", []):
            symbol = item.get("symbol", "")
            if symbol not in tradable:
                continue
            quote_volume = float(item.get("turnover24h") or 0.0)
            if quote_volume < min_quote_volume:
                continue
            tickers.append(
                Ticker(
                    symbol=symbol,
                    last_price=float(item.get("lastPrice") or 0.0),
                    quote_volume=quote_volume,
                    change_pct=float(item.get("price24hPcnt") or 0.0) * 100.0,
                )
            )
        tickers.sort(key=lambda t: t.quote_volume, reverse=True)
        return tickers[:limit]

    def klines(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        bybit_interval = INTERVALS_BYBIT.get(interval)
        if bybit_interval is None:
            raise DataUnavailable(f"Bybit does not support interval {interval}.")
        payload = _json_get(
            self.base_url,
            "/v5/market/kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "limit": str(limit),
            },
        )
        if str(payload.get("retCode")) != "0":
            raise DataUnavailable(f"Bybit kline error for {symbol} {interval}: {payload.get('retMsg')}")
        rows = payload.get("result", {}).get("list", [])
        candles = [
            Candle(
                open_time=utc_from_ms(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in rows
        ]
        candles.sort(key=lambda candle: candle.open_time)
        if not candles:
            raise DataUnavailable(f"Bybit returned no candles for {symbol} {interval}.")
        return candles


def create_client(exchange: str) -> ExchangeClient:
    normalized = exchange.lower().strip()
    if normalized in {"binance", "binance-futures", "futures"}:
        return BinanceFuturesClient()
    if normalized in {"bybit", "bybit-linear", "linear"}:
        return BybitLinearClient()
    raise ValueError(f"Unknown exchange: {exchange}")


def create_auto_client(limit: int, min_quote_volume: float) -> tuple[ExchangeClient, list[Ticker], list[str]]:
    errors: list[str] = []
    for client in (BybitLinearClient(), BinanceFuturesClient()):
        try:
            tickers = client.top_symbols(limit=limit, min_quote_volume=min_quote_volume)
            if tickers:
                return client, tickers, errors
            errors.append(f"{client.name}: no symbols passed min_quote_volume={min_quote_volume}")
        except DataUnavailable as exc:
            errors.append(f"{client.name}: {exc}")
    raise DataUnavailable("All configured public exchange data sources failed: " + " | ".join(errors))
