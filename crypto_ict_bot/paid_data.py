from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import DirectionScore, SymbolReport


class PaidDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    name: str
    category: str
    purpose: str
    apply_url: str
    docs_url: str
    key_hint: str
    data_latency: str = "依方案與端點而定"
    realtime_status: str = "非完全即時"
    health_note: str = ""


@dataclass
class ProviderStatus:
    id: str
    name: str
    category: str
    purpose: str
    apply_url: str
    docs_url: str
    configured: bool
    enabled: bool
    state: str
    message: str
    key_hint: str
    data_latency: str
    realtime_status: str
    health_note: str
    readable: bool | None = None
    success_count: int = 0
    failure_count: int = 0
    last_error: str = ""


@dataclass
class ExternalSymbolMetrics:
    symbol: str
    bonus_long: float = 0.0
    bonus_short: float = 0.0
    reasons_long: list[str] = field(default_factory=list)
    reasons_short: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)
    providers: list[str] = field(default_factory=list)
    provider_status: dict[str, dict[str, Any]] = field(default_factory=dict)


PROVIDERS = [
    ProviderDefinition(
        id="exchange_public",
        name="Bybit / Binance 公開衍生品資料",
        category="免費內建 / OI + Funding",
        purpose="不需要 key，直接用交易所公開 API 補未平倉量與 funding，作為所有掃描的底層資料。",
        apply_url="https://bybit-exchange.github.io/docs/v5/market/open-interest",
        docs_url="https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest",
        key_hint="不需要 API key",
        data_latency="交易所公開端點，通常秒級到分鐘級",
        realtime_status="近即時",
        health_note="每次掃描都會讀 funding 與 OI；若交易所限流會顯示部分可讀。",
    ),
    ProviderDefinition(
        id="coinglass",
        name="CoinGlass",
        category="最佳衍生品 / 清算熱力圖",
        purpose="OI、funding、爆倉、清算熱力圖、訂單簿熱力圖。",
        apply_url="https://www.coinglass.com/CryptoApi",
        docs_url="https://docs.coinglass.com/reference/getting-started-with-your-api",
        key_hint="CG-API-KEY",
        data_latency="依方案，頁面標示 Updates ≤ 1 min",
        realtime_status="近即時 / 付費",
        health_note="清算、熱力圖、L2/L3 與 ETF 等高價值資料依方案開通。",
    ),
    ProviderDefinition(
        id="coinalyze",
        name="Coinalyze",
        category="免費衍生品 / CVD",
        purpose="OI、funding、long/short、buy/sell、爆倉歷史；可補 ICT 位移背後的槓桿與主動買賣資料。",
        apply_url="https://coinalyze.net",
        docs_url="https://api.coinalyze.net/v1/doc/",
        key_hint="api_key",
        data_latency="API 文件標示 intraday 保留約 1500-2000 筆，40 calls/min",
        realtime_status="近即時 / 免費",
        health_note="適合補 OI、funding、long-short 與爆倉歷史。",
    ),
    ProviderDefinition(
        id="coingecko",
        name="CoinGecko",
        category="多功能市場資料 / 免費 Demo",
        purpose="一個 API 補價格、市值、交易所、衍生品、鏈上 DEX、新聞與基本資料；免費 Demo 可先用。",
        apply_url="https://www.coingecko.com/en/api",
        docs_url="https://docs.coingecko.com/reference/setting-up-your-api-key",
        key_hint="x-cg-demo-api-key",
        data_latency="Demo/付費方案不同，通常分鐘級",
        realtime_status="非交易所即時",
        health_note="用於交叉檢查市值、交易量、基本資料與鏈上 DEX 背景。",
    ),
    ProviderDefinition(
        id="defillama",
        name="DeFiLlama",
        category="免費 DeFi / 穩定幣 / TVL",
        purpose="免費公開 API，補 DeFi TVL、穩定幣供給、DEX 量、費用與鏈上宏觀背景。",
        apply_url="https://defillama.com/docs/api",
        docs_url="https://defillama.com/docs/api",
        key_hint="不需要 API key",
        data_latency="多數宏觀資料為分鐘到日級",
        realtime_status="非即時",
        health_note="TVL、穩定幣、DEX 量等宏觀背景，不直接當入場訊號。",
    ),
    ProviderDefinition(
        id="cryptopanic",
        name="CryptoPanic",
        category="新聞事件 / 免費 + 付費",
        purpose="加密新聞、情緒、重要事件；免費 token 可先用，PLUS 可升級更多欄位。",
        apply_url="https://cryptopanic.com/developers/api/",
        docs_url="https://cryptopanic.com/developers/api/",
        key_hint="auth_token",
        data_latency="新聞快取，官方建議不必每 30 秒內重複請求",
        realtime_status="近即時新聞",
        health_note="用於標記事件風險，不直接產生入場。",
    ),
    ProviderDefinition(
        id="glassnode",
        name="Glassnode",
        category="鏈上資金流",
        purpose="交易所淨流入/淨流出、ETF/衍生品與鏈上週期資料。",
        apply_url="https://studio.glassnode.com/pricing",
        docs_url="https://docs.glassnode.com/basic-api/api",
        key_hint="X-Api-Key 或 api_key",
        data_latency="鏈上資料通常分鐘到小時級，依指標而定",
        realtime_status="非完全即時",
        health_note="交易所淨流、鏈上週期、ETF/資金流屬高品質背景資料。",
    ),
    ProviderDefinition(
        id="cryptoquant",
        name="CryptoQuant",
        category="鏈上資金流",
        purpose="交易所 reserve/netflow、礦工、穩定幣與專業鏈上指標。",
        apply_url="https://cryptoquant.com/en/pricing",
        docs_url="https://cryptoquant.com/en/docs",
        key_hint="Authorization: Bearer",
        data_latency="鏈上/交易所指標通常分鐘到小時級",
        realtime_status="非完全即時",
        health_note="交易所 reserve/netflow、礦工、穩定幣資料用於風險共振。",
    ),
    ProviderDefinition(
        id="coinmarketcal",
        name="CoinMarketCal",
        category="事件日曆",
        purpose="上幣、解鎖、升級、監管、投票等可能造成誘導或劇烈波動的事件。",
        apply_url="https://coinmarketcal.com/en/api",
        docs_url="https://www.postman.com/coinmarketcalapi/coinmarketcal-api/documentation/61mcdog/coinmarketcal-api",
        key_hint="x-api-key",
        data_latency="事件日曆，日級/事件級",
        realtime_status="非即時",
        health_note="用於提前標示上幣、解鎖、升級等事件風險。",
    ),
    ProviderDefinition(
        id="thetie",
        name="The Tie",
        category="最佳新聞 / 情緒 / 解鎖",
        purpose="新聞、情緒、token unlock、開發活動與鏈上基本面；偏機構級。",
        apply_url="https://www.thetie.io/solutions/api",
        docs_url="https://docs.thetie.io/reference",
        key_hint="API key",
        data_latency="新聞/情緒依方案與來源而定",
        realtime_status="近即時 / 付費",
        health_note="機構級新聞、情緒、解鎖與開發活動資料。",
    ),
    ProviderDefinition(
        id="tokenmetrics",
        name="Token Metrics",
        category="AI 評級 / 情緒 / 訊號",
        purpose="交易評級、投資評級、市場情緒與 AI 指標；作為外部共振而非取代本策略。",
        apply_url="https://www.tokenmetrics.com/api",
        docs_url="https://developers.tokenmetrics.com/docs/getting-started",
        key_hint="x-api-key",
        data_latency="評級與訊號依方案更新",
        realtime_status="非交易所即時",
        health_note="Trader Grade、Investor Grade、Bull/Bear signal 作為外部 AI 共振。",
    ),
]

NO_KEY_PROVIDERS = {"exchange_public", "defillama"}


def _request_json(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 6,
) -> Any:
    query = f"?{urlencode(params or {})}" if params else ""
    request = Request(
        f"{base_url}{path}{query}",
        headers={
            "User-Agent": "crypto-ict-selection-bot/0.2",
            "Accept": "application/json",
            **(headers or {}),
        },
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 1:
                time.sleep(0.25 * (attempt + 1))
    raise PaidDataError(str(last_error)) from last_error


def _base_symbol(symbol: str) -> str:
    return symbol.upper().replace("USDT", "").replace("USD", "")


def _has_derivative_values(values: dict[str, Any]) -> bool:
    return any(
        _as_float(values.get(key)) is not None
        for key in ("funding_rate", "open_interest", "open_interest_change_pct")
    )


def _bybit_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise PaidDataError("Bybit returned a non-object payload")
    if str(payload.get("retCode")) not in {"0", ""}:
        raise PaidDataError(str(payload.get("retMsg") or payload.get("retCode") or "Bybit API error"))
    rows = payload.get("result", {}).get("list", [])
    return rows if isinstance(rows, list) else []


class ExchangePublicDerivativesClient:
    bybit_base_url = "https://api.bybit.com"
    binance_base_url = "https://fapi.binance.com"

    def symbol_metrics(self, symbol: str, exchange: str) -> dict[str, Any]:
        preferred = "binance" if "binance" in exchange.lower() else "bybit"
        exchanges = [preferred, "bybit" if preferred == "binance" else "binance"]
        errors: list[str] = []
        for name in exchanges:
            try:
                values = self._binance(symbol) if name == "binance" else self._bybit(symbol)
            except PaidDataError as exc:
                errors.append(f"{name}: {exc}")
                continue
            if _has_derivative_values(values):
                values["source_exchange"] = name
                if errors:
                    values["fallback_errors"] = errors
                return values
            errors.append(f"{name}: no usable funding/OI values")
        raise PaidDataError(" | ".join(errors) or "no public derivatives values")

    def _bybit(self, symbol: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        errors: list[str] = []
        try:
            funding = _request_json(
                self.bybit_base_url,
                "/v5/market/funding/history",
                {"category": "linear", "symbol": symbol, "limit": 1},
            )
            rows = _bybit_rows(funding)
            if rows:
                result["funding_rate"] = _as_float(rows[0].get("fundingRate"))
        except PaidDataError as exc:
            errors.append(f"funding: {exc}")
        try:
            oi = _request_json(
                self.bybit_base_url,
                "/v5/market/open-interest",
                {"category": "linear", "symbol": symbol, "intervalTime": "1h", "limit": 2},
            )
            oi_rows = _bybit_rows(oi)
            values = [_as_float(row.get("openInterest")) for row in oi_rows]
            values = [value for value in values if value is not None]
            if values:
                result["open_interest"] = values[0]
                if len(values) >= 2 and values[1]:
                    result["open_interest_change_pct"] = (values[0] - values[1]) / abs(values[1]) * 100.0
        except PaidDataError as exc:
            errors.append(f"open_interest: {exc}")
        if errors:
            result["partial_errors"] = errors
        if not _has_derivative_values(result):
            raise PaidDataError("; ".join(errors) or "Bybit returned no usable values")
        return result

    def _binance(self, symbol: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        errors: list[str] = []
        try:
            premium = _request_json(self.binance_base_url, "/fapi/v1/premiumIndex", {"symbol": symbol})
            if isinstance(premium, dict):
                result["funding_rate"] = _as_float(premium.get("lastFundingRate"))
        except PaidDataError as exc:
            errors.append(f"funding: {exc}")
        try:
            oi = _request_json(self.binance_base_url, "/fapi/v1/openInterest", {"symbol": symbol})
            if isinstance(oi, dict):
                result["open_interest"] = _as_float(oi.get("openInterest"))
        except PaidDataError as exc:
            errors.append(f"open_interest: {exc}")
        try:
            history = _request_json(
                self.binance_base_url,
                "/futures/data/openInterestHist",
                {"symbol": symbol, "period": "1h", "limit": 2},
            )
            values = [_as_float(row.get("sumOpenInterest")) for row in history if isinstance(row, dict)]
            values = [value for value in values if value is not None]
            if len(values) >= 2 and values[-2]:
                result["open_interest_change_pct"] = (values[-1] - values[-2]) / abs(values[-2]) * 100.0
        except PaidDataError as exc:
            errors.append(f"open_interest_history: {exc}")
        if errors:
            result["partial_errors"] = errors
        if not _has_derivative_values(result):
            raise PaidDataError("; ".join(errors) or "Binance returned no usable values")
        return result


class CoinGeckoClient:
    base_url = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {"x-cg-demo-api-key": self.api_key} if self.api_key else {}

    def check(self) -> None:
        _request_json(self.base_url, "/ping", headers=self.headers)

    def simple_prices(self, symbols: list[str]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        bases = [_base_symbol(symbol).lower() for symbol in symbols]
        for start in range(0, len(bases), 45):
            chunk = bases[start : start + 45]
            payload = _request_json(
                self.base_url,
                "/simple/price",
                {
                    "vs_currencies": "usd",
                    "symbols": ",".join(chunk),
                    "include_market_cap": "true",
                    "include_24hr_vol": "true",
                    "include_24hr_change": "true",
                },
                self.headers,
            )
            if isinstance(payload, dict):
                output.update(payload)
        return output


class DefiLlamaClient:
    base_url = "https://api.llama.fi"

    def check(self) -> None:
        _request_json(self.base_url, "/protocols")

    def macro_snapshot(self) -> dict[str, Any]:
        protocols = _request_json(self.base_url, "/protocols")
        total_tvl = 0.0
        if isinstance(protocols, list):
            for row in protocols:
                total_tvl += _as_float(row.get("tvl")) or 0.0
        return {"protocol_count": len(protocols) if isinstance(protocols, list) else None, "total_tvl": total_tvl}


def _extract_data(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("data", "result", "items"):
            if key in payload:
                return payload[key]
    return payload


def _last_number(value: Any, preferred_keys: tuple[str, ...] = ("c", "close", "value", "v")) -> float | None:
    if isinstance(value, dict):
        for key in preferred_keys:
            if key in value:
                found = _last_number(value[key], preferred_keys)
                if found is not None:
                    return found
        for item in reversed(list(value.values())):
            found = _last_number(item, preferred_keys)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in reversed(value):
            found = _last_number(item, preferred_keys)
            if found is not None:
                return found
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


class CoinglassClient:
    base_url = "https://open-api-v4.coinglass.com"

    def __init__(self, api_key: str, exchange: str = "Bybit") -> None:
        self.api_key = api_key
        self.exchange = exchange

    @property
    def headers(self) -> dict[str, str]:
        return {"CG-API-KEY": self.api_key}

    def check(self) -> None:
        _request_json(self.base_url, "/api/futures/supported-coins", headers=self.headers)

    def symbol_metrics(self, symbol: str) -> dict[str, Any]:
        coin = _base_symbol(symbol)
        result: dict[str, Any] = {}
        try:
            result["open_interest"] = _request_json(
                self.base_url,
                "/api/futures/open-interest/aggregated-history",
                {"symbol": coin, "interval": "1h", "limit": 24, "unit": "usd"},
                self.headers,
            )
        except PaidDataError as exc:
            result["open_interest_error"] = str(exc)
        try:
            result["funding"] = _request_json(
                self.base_url,
                "/api/futures/funding-rate/vol-weight-history",
                {"symbol": coin, "interval": "1h", "limit": 24},
                self.headers,
            )
        except PaidDataError as exc:
            result["funding_error"] = str(exc)
        try:
            result["liquidation"] = _request_json(
                self.base_url,
                "/api/futures/liquidation/aggregated-history",
                {"exchange_list": "Binance,OKX,Bybit", "symbol": coin, "interval": "1h", "limit": 24},
                self.headers,
            )
        except PaidDataError as exc:
            result["liquidation_error"] = str(exc)
        return result


class CoinalyzeClient:
    base_url = "https://api.coinalyze.net/v1"

    def __init__(self, api_key: str, preferred_exchange: str = "Bybit") -> None:
        self.api_key = api_key
        self.preferred_exchange = preferred_exchange.lower()
        self._market_cache: dict[str, str] | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {"api_key": self.api_key}

    def check(self) -> None:
        _request_json(self.base_url, "/exchanges", headers=self.headers)

    def _market_map(self) -> dict[str, str]:
        if self._market_cache is not None:
            return self._market_cache
        markets = _request_json(self.base_url, "/future-markets", headers=self.headers)
        mapping: dict[str, str] = {}
        fallback: dict[str, str] = {}
        for market in markets or []:
            if not market.get("is_perpetual"):
                continue
            if market.get("quote_asset") not in {"USDT", "USD"}:
                continue
            exchange = str(market.get("exchange", "")).lower()
            exchange_symbol = str(market.get("symbol_on_exchange") or "").upper()
            api_symbol = str(market.get("symbol") or "")
            if not exchange_symbol or not api_symbol:
                continue
            fallback.setdefault(exchange_symbol, api_symbol)
            if self.preferred_exchange in exchange:
                mapping[exchange_symbol] = api_symbol
        fallback.update(mapping)
        self._market_cache = fallback
        return fallback

    def metrics(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        market_map = self._market_map()
        resolved = {symbol: market_map.get(symbol) for symbol in symbols}
        resolved = {symbol: api_symbol for symbol, api_symbol in resolved.items() if api_symbol}
        output = {symbol: {"coinalyze_symbol": api_symbol} for symbol, api_symbol in resolved.items()}
        api_symbols = list(resolved.values())
        for start in range(0, len(api_symbols), 20):
            chunk = api_symbols[start : start + 20]
            params = {"symbols": ",".join(chunk)}
            for endpoint, key in (
                ("/open-interest", "open_interest"),
                ("/funding-rate", "funding_rate"),
                ("/predicted-funding-rate", "predicted_funding_rate"),
            ):
                try:
                    rows = _request_json(self.base_url, endpoint, params, self.headers)
                except PaidDataError as exc:
                    for symbol, api_symbol in resolved.items():
                        if api_symbol in chunk:
                            output.setdefault(symbol, {})[f"{key}_error"] = str(exc)
                    continue
                by_api_symbol = {row.get("symbol"): row for row in rows or []}
                for symbol, api_symbol in resolved.items():
                    if api_symbol in chunk and api_symbol in by_api_symbol:
                        output.setdefault(symbol, {})[key] = by_api_symbol[api_symbol].get("value")
        return output


class GlassnodeClient:
    base_url = "https://api.glassnode.com"

    def __init__(self, api_key: str, interval: str = "1h") -> None:
        self.api_key = api_key
        self.interval = interval

    def check(self) -> None:
        _request_json(self.base_url, "/v1/metrics/assets", headers={"X-Api-Key": self.api_key})

    def exchange_netflow(self, symbol: str) -> float | None:
        asset = _base_symbol(symbol).lower()
        since = int((datetime.now(timezone.utc) - timedelta(days=2)).timestamp())
        payload = _request_json(
            self.base_url,
            "/v1/metrics/transactions/transfers_volume_exchanges_net",
            {"a": asset, "s": since, "i": self.interval, "c": "USD"},
            {"X-Api-Key": self.api_key},
        )
        return _last_number(payload)


class CryptoQuantClient:
    base_url = "https://api.cryptoquant.com"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def check(self) -> None:
        _request_json(self.base_url, "/v1/btc/exchange-flows/netflow", {"window": "day"}, self.headers)

    def exchange_netflow(self, symbol: str, window: str = "day") -> float | None:
        asset = _base_symbol(symbol).lower()
        payload = _request_json(self.base_url, f"/v1/{asset}/exchange-flows/netflow", {"window": window}, self.headers)
        return _last_number(payload)


class CoinMarketCalClient:
    base_url = "https://developers.coinmarketcal.com"

    def __init__(self, api_key: str, lookahead_days: int = 7) -> None:
        self.api_key = api_key
        self.lookahead_days = lookahead_days

    @property
    def headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    def check(self) -> None:
        _request_json(self.base_url, "/v1/categories", headers=self.headers)

    def upcoming_events(self, symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
        start = datetime.now(timezone.utc).date()
        end = start + timedelta(days=self.lookahead_days)
        payload = _request_json(
            self.base_url,
            "/v1/events",
            {
                "page": 1,
                "max": 75,
                "dateRangeStart": start.isoformat(),
                "dateRangeEnd": end.isoformat(),
                "sortBy": "catalyst_events",
                "showVotes": "true",
                "showViews": "true",
            },
            self.headers,
        )
        rows = _extract_data(payload) or []
        output = {symbol: [] for symbol in symbols}
        wanted = {_base_symbol(symbol): symbol for symbol in symbols}
        for row in rows if isinstance(rows, list) else []:
            coins = row.get("coins") or row.get("coin") or []
            if isinstance(coins, dict):
                coins = [coins]
            matched: set[str] = set()
            for coin in coins:
                symbol = str(coin.get("symbol") or coin.get("code") or "").upper()
                if symbol in wanted:
                    matched.add(wanted[symbol])
            for symbol in matched:
                output.setdefault(symbol, []).append(row)
        return output


class CryptoPanicClient:
    base_url = "https://cryptopanic.com/api/free/v1"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def check(self) -> None:
        _request_json(self.base_url, "/posts/", {"auth_token": self.api_key, "public": "true"})

    def posts(self, symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
        output = {symbol: [] for symbol in symbols}
        wanted = {_base_symbol(symbol): symbol for symbol in symbols}
        bases = list(wanted.keys())
        for start in range(0, len(bases), 50):
            chunk = bases[start : start + 50]
            payload = _request_json(
                self.base_url,
                "/posts/",
                {
                    "auth_token": self.api_key,
                    "public": "true",
                    "currencies": ",".join(chunk),
                    "filter": "important",
                },
            )
            rows = payload.get("results", []) if isinstance(payload, dict) else []
            for row in rows:
                currencies = row.get("currencies") or []
                for currency in currencies:
                    code = str(currency.get("code") or "").upper()
                    symbol = wanted.get(code)
                    if symbol:
                        output.setdefault(symbol, []).append(row)
        return output


class TheTieClient:
    base_url = "https://terminal-api.thetie.io"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        value = self.api_key if self.api_key.lower().startswith("bearer ") else f"Bearer {self.api_key}"
        return {"Authorization": value}

    def check(self) -> None:
        _request_json(self.base_url, "/v1/news_sentiment", {"limit": 1}, self.headers)

    def news_sentiment(self, symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
        payload = _request_json(self.base_url, "/v1/news_sentiment", {"limit": 200}, self.headers)
        rows = _extract_data(payload) or []
        output = {symbol: [] for symbol in symbols}
        wanted = {_base_symbol(symbol): symbol for symbol in symbols}
        for row in rows if isinstance(rows, list) else []:
            text = json.dumps(row, ensure_ascii=False).upper()
            for base, full_symbol in wanted.items():
                if base in text:
                    output.setdefault(full_symbol, []).append(row)
        return output


class TokenMetricsClient:
    base_url = "https://api.tokenmetrics.com"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    def check(self) -> None:
        _request_json(self.base_url, "/v2/tokens", {"limit": 1}, self.headers)

    def tm_grade(self, symbol: str) -> dict[str, Any] | None:
        payload = _request_json(self.base_url, "/v2/tm-grade", {"symbol": _base_symbol(symbol)}, self.headers)
        data = _extract_data(payload)
        if isinstance(data, list):
            return data[0] if data else None
        if isinstance(data, dict):
            return data
        return None


def provider_statuses(config: dict[str, Any], live_check: bool = False) -> list[dict[str, Any]]:
    keys = config.get("api_keys", {})
    statuses: list[ProviderStatus] = []
    for provider in PROVIDERS:
        key = keys.get(provider.id, "")
        configured = provider.id in NO_KEY_PROVIDERS or bool(key)
        status = ProviderStatus(
            id=provider.id,
            name=provider.name,
            category=provider.category,
            purpose=provider.purpose,
            apply_url=provider.apply_url,
            docs_url=provider.docs_url,
            configured=configured,
            enabled=bool(config.get("paid_data", {}).get("enabled", True)) and configured,
            state="免費內建" if provider.id in NO_KEY_PROVIDERS else ("未設定" if not configured else "已設定"),
            message="不需要 API key，掃描時自動使用" if provider.id in NO_KEY_PROVIDERS else ("尚未輸入 API key" if not configured else "已保存 key，掃描時會嘗試使用"),
            key_hint=provider.key_hint,
            data_latency=provider.data_latency,
            realtime_status=provider.realtime_status,
            health_note=provider.health_note,
            readable=None if configured else False,
        )
        statuses.append(status)
    return [status.__dict__ for status in statuses]


def _status_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {status["id"]: status for status in provider_statuses(config)}


def _mark_provider(
    statuses: dict[str, dict[str, Any]],
    provider_id: str,
    state: str,
    message: str,
    readable: bool | None,
    success_count: int = 0,
    failure_count: int = 0,
    last_error: str = "",
) -> None:
    if provider_id not in statuses:
        return
    statuses[provider_id].update(
        {
            "state": state,
            "message": message,
            "readable": readable,
            "success_count": success_count,
            "failure_count": failure_count,
            "last_error": last_error,
        }
    )


def enrich_reports(reports: list[SymbolReport], config: dict[str, Any]) -> dict[str, Any]:
    if not config.get("paid_data", {}).get("enabled", True):
        return {"enabled": False, "providers": provider_statuses(config), "errors": []}

    metrics = {report.symbol: ExternalSymbolMetrics(report.symbol) for report in reports}
    errors: list[str] = []
    keys = config.get("api_keys", {})
    paid_cfg = config.get("paid_data", {})
    symbols = [report.symbol for report in reports]
    statuses = _status_map(config)

    public_derivatives = ExchangePublicDerivativesClient()
    public_success = 0
    public_failure = 0
    public_error = ""
    public_reports = reports
    configured_public_timeout = paid_cfg.get("public_derivatives_timeout_seconds")
    dynamic_public_timeout = max(30.0, min(120.0, len(public_reports) * 1.8))
    public_timeout = max(20.0, min(120.0, float(configured_public_timeout or dynamic_public_timeout)))
    public_workers = max(1, min(8, int(paid_cfg.get("public_derivatives_workers", 6))))
    executor = ThreadPoolExecutor(max_workers=public_workers)
    future_map = {
        executor.submit(public_derivatives.symbol_metrics, report.symbol, report.exchange): report
        for report in public_reports
    }
    timed_out = False
    try:
        completed = as_completed(future_map, timeout=public_timeout)
        for future in completed:
            report = future_map[future]
            try:
                values = future.result()
            except PaidDataError as exc:
                public_failure += 1
                public_error = public_error or str(exc)
                metrics[report.symbol].provider_status["exchange_public"] = {
                    "state": "failed",
                    "error": str(exc),
                }
                metrics[report.symbol].warnings.append(f"交易所公開衍生品資料暫時不可用：{exc}")
                continue
            public_success += 1
            item = metrics[report.symbol]
            item.providers.append(f"exchange_public:{values.get('source_exchange', 'unknown')}")
            item.values["exchange_public_derivatives"] = values
            item.provider_status["exchange_public"] = {
                "state": "read",
                "source": values.get("source_exchange", "unknown"),
            }
            funding = _as_float(values.get("funding_rate"))
            if funding is not None:
                _score_funding(item, funding, "交易所公開資料")
            oi_change = _as_float(values.get("open_interest_change_pct"))
            if oi_change is not None and abs(oi_change) >= 8:
                item.warnings.append(f"公開 OI 近 1h 變動 {oi_change:.2f}%，槓桿流入/流出較劇烈")
    except FuturesTimeoutError:
        timed_out = True
        public_error = public_error or f"公開衍生品補資料超過 {public_timeout:.0f}s，已保留已讀到的資料"
    finally:
        for future, report in future_map.items():
            if not future.done():
                future.cancel()
                public_failure += 1
        executor.shutdown(wait=False, cancel_futures=True)
    if public_success:
        public_state = "可讀取" if not public_failure else "部分可讀"
        message = f"本次讀到 {public_success} 個幣種，失敗 {public_failure} 個。"
        if timed_out:
            message += " 部分幣種因時間上限跳過，避免刷新卡死。"
        _mark_provider(
            statuses,
            "exchange_public",
            public_state,
            message,
            True,
            public_success,
            public_failure,
            public_error,
        )
    else:
        _mark_provider(statuses, "exchange_public", "讀不到", "交易所公開衍生品資料本次沒有讀到。", False, 0, public_failure, public_error)

    if keys.get("coingecko"):
        try:
            coingecko = CoinGeckoClient(keys["coingecko"])
            cg_prices = coingecko.simple_prices(symbols)
            cg_success = 0
            for symbol in symbols:
                base = _base_symbol(symbol).lower()
                if base not in cg_prices:
                    metrics[symbol].provider_status["coingecko"] = {"state": "no_symbol_data"}
                    continue
                cg_success += 1
                item = metrics[symbol]
                item.providers.append("CoinGecko")
                item.values["coingecko"] = cg_prices[base]
                item.provider_status["coingecko"] = {"state": "read"}
            _mark_provider(statuses, "coingecko", "可讀取" if cg_success else "可連線但無匹配資料", f"本次匹配 {cg_success} 個幣種。", bool(cg_success), cg_success, len(symbols) - cg_success)
        except PaidDataError as exc:
            errors.append(f"CoinGecko: {exc}")
            for symbol in symbols:
                metrics[symbol].provider_status["coingecko"] = {"state": "failed", "error": str(exc)}
            _mark_provider(statuses, "coingecko", "讀不到", "CoinGecko 本次讀取失敗。", False, 0, 1, str(exc))

    try:
        llama = DefiLlamaClient()
        macro = llama.macro_snapshot()
        if macro.get("total_tvl"):
            for item in metrics.values():
                item.providers.append("DeFiLlama")
                item.values["defillama_macro"] = macro
            _mark_provider(statuses, "defillama", "可讀取", "已讀取 DeFiLlama 宏觀 TVL/協議資料。", True, 1, 0)
        else:
            _mark_provider(statuses, "defillama", "可連線但無資料", "DeFiLlama 回應成功，但本次沒有可用宏觀值。", False, 0, 1)
    except PaidDataError as exc:
        errors.append(f"DeFiLlama: {exc}")
        _mark_provider(statuses, "defillama", "讀不到", "DeFiLlama 本次讀取失敗。", False, 0, 1, str(exc))

    if keys.get("coinalyze"):
        try:
            coinalyze = CoinalyzeClient(keys["coinalyze"], paid_cfg.get("preferred_derivatives_exchange", "Bybit"))
            coinalyze_data = coinalyze.metrics(symbols)
            coinalyze_success = 0
            for symbol in symbols:
                metrics[symbol].provider_status["coinalyze"] = {"state": "no_symbol_data"}
            for symbol, values in coinalyze_data.items():
                coinalyze_success += 1
                item = metrics[symbol]
                item.providers.append("Coinalyze")
                item.values["coinalyze"] = values
                item.provider_status["coinalyze"] = {"state": "read"}
                funding = _as_float(values.get("funding_rate"))
                predicted = _as_float(values.get("predicted_funding_rate"))
                funding_value = predicted if predicted is not None else funding
                if funding_value is not None:
                    _score_funding(item, funding_value, "Coinalyze")
            _mark_provider(statuses, "coinalyze", "可讀取" if coinalyze_success else "可連線但無匹配資料", f"本次匹配 {coinalyze_success} 個幣種。", bool(coinalyze_success), coinalyze_success, len(symbols) - coinalyze_success)
        except PaidDataError as exc:
            errors.append(f"Coinalyze: {exc}")
            for symbol in symbols:
                metrics[symbol].provider_status["coinalyze"] = {"state": "failed", "error": str(exc)}
            _mark_provider(statuses, "coinalyze", "讀不到", "Coinalyze 本次讀取失敗。", False, 0, 1, str(exc))

    if keys.get("coinglass"):
        coinglass = CoinglassClient(keys["coinglass"], paid_cfg.get("preferred_derivatives_exchange", "Bybit"))
        coinglass_success = 0
        coinglass_failure = 0
        coinglass_error = ""
        for symbol in symbols:
            try:
                values = coinglass.symbol_metrics(symbol)
                coinglass_success += 1
                item = metrics[symbol]
                item.providers.append("CoinGlass")
                item.values["coinglass"] = _compact_payload(values)
                item.provider_status["coinglass"] = {"state": "read"}
                funding = _last_number(values.get("funding"))
                if funding is not None:
                    _score_funding(item, funding, "CoinGlass")
                liquidation = values.get("liquidation")
                long_liq = _sum_named(liquidation, ("long", "longLiquidation", "long_liquidation"))
                short_liq = _sum_named(liquidation, ("short", "shortLiquidation", "short_liquidation"))
                if long_liq or short_liq:
                    item.values["coinglass_liquidation_sum"] = {"long": long_liq, "short": short_liq}
                    _score_liquidation(item, long_liq, short_liq, "CoinGlass")
            except PaidDataError as exc:
                coinglass_failure += 1
                coinglass_error = coinglass_error or str(exc)
                metrics[symbol].provider_status["coinglass"] = {"state": "failed", "error": str(exc)}
                errors.append(f"CoinGlass {symbol}: {exc}")
        if coinglass_success:
            state = "可讀取" if not coinglass_failure else "部分可讀"
            _mark_provider(statuses, "coinglass", state, f"本次讀到 {coinglass_success} 個幣種，失敗 {coinglass_failure} 個。", True, coinglass_success, coinglass_failure, coinglass_error)
        else:
            _mark_provider(statuses, "coinglass", "讀不到", "CoinGlass 本次沒有讀到資料。", False, 0, coinglass_failure, coinglass_error)

    if keys.get("glassnode"):
        glassnode = GlassnodeClient(keys["glassnode"], paid_cfg.get("glassnode_interval", "1h"))
        glassnode_success = 0
        glassnode_failure = 0
        glassnode_error = ""
        for symbol in symbols:
            try:
                netflow = glassnode.exchange_netflow(symbol)
                if netflow is None:
                    metrics[symbol].provider_status["glassnode"] = {"state": "no_symbol_data"}
                    continue
                glassnode_success += 1
                item = metrics[symbol]
                item.providers.append("Glassnode")
                item.values["glassnode_exchange_netflow_usd"] = netflow
                item.provider_status["glassnode"] = {"state": "read"}
                _score_netflow(item, netflow, "Glassnode")
            except PaidDataError as exc:
                glassnode_failure += 1
                glassnode_error = glassnode_error or str(exc)
                metrics[symbol].provider_status["glassnode"] = {"state": "failed", "error": str(exc)}
                errors.append(f"Glassnode {symbol}: {exc}")
        if glassnode_success:
            state = "可讀取" if not glassnode_failure else "部分可讀"
            _mark_provider(statuses, "glassnode", state, f"本次讀到 {glassnode_success} 個幣種，失敗 {glassnode_failure} 個。", True, glassnode_success, glassnode_failure, glassnode_error)
        else:
            _mark_provider(statuses, "glassnode", "讀不到", "Glassnode 本次沒有讀到可用資料。", False, 0, glassnode_failure, glassnode_error)

    if keys.get("cryptoquant"):
        cryptoquant = CryptoQuantClient(keys["cryptoquant"])
        cq_success = 0
        cq_failure = 0
        cq_error = ""
        for symbol in symbols:
            try:
                netflow = cryptoquant.exchange_netflow(symbol, paid_cfg.get("cryptoquant_window", "day"))
                if netflow is None:
                    metrics[symbol].provider_status["cryptoquant"] = {"state": "no_symbol_data"}
                    continue
                cq_success += 1
                item = metrics[symbol]
                item.providers.append("CryptoQuant")
                item.values["cryptoquant_exchange_netflow"] = netflow
                item.provider_status["cryptoquant"] = {"state": "read"}
                _score_netflow(item, netflow, "CryptoQuant")
            except PaidDataError as exc:
                cq_failure += 1
                cq_error = cq_error or str(exc)
                metrics[symbol].provider_status["cryptoquant"] = {"state": "failed", "error": str(exc)}
                errors.append(f"CryptoQuant {symbol}: {exc}")
        if cq_success:
            state = "可讀取" if not cq_failure else "部分可讀"
            _mark_provider(statuses, "cryptoquant", state, f"本次讀到 {cq_success} 個幣種，失敗 {cq_failure} 個。", True, cq_success, cq_failure, cq_error)
        else:
            _mark_provider(statuses, "cryptoquant", "讀不到", "CryptoQuant 本次沒有讀到可用資料。", False, 0, cq_failure, cq_error)

    if keys.get("coinmarketcal"):
        try:
            calendar = CoinMarketCalClient(keys["coinmarketcal"], int(paid_cfg.get("event_lookahead_days", 7)))
            events = calendar.upcoming_events(symbols)
            event_count = 0
            for symbol, rows in events.items():
                if not rows:
                    metrics[symbol].provider_status["coinmarketcal"] = {"state": "read_no_hit"}
                    continue
                event_count += len(rows)
                item = metrics[symbol]
                item.providers.append("CoinMarketCal")
                item.values["coinmarketcal_events"] = [_event_summary(row) for row in rows[:5]]
                item.provider_status["coinmarketcal"] = {"state": "read", "count": len(rows)}
                item.warnings.append(f"未來 {paid_cfg.get('event_lookahead_days', 7)} 天有 {len(rows)} 個事件，需避開新聞誘導與跳空")
            _mark_provider(statuses, "coinmarketcal", "可讀取", f"事件日曆可讀，本次命中 {event_count} 個事件。", True, 1, 0)
        except PaidDataError as exc:
            errors.append(f"CoinMarketCal: {exc}")
            for symbol in symbols:
                metrics[symbol].provider_status["coinmarketcal"] = {"state": "failed", "error": str(exc)}
            _mark_provider(statuses, "coinmarketcal", "讀不到", "CoinMarketCal 本次讀取失敗。", False, 0, 1, str(exc))

    if keys.get("cryptopanic"):
        try:
            cryptopanic = CryptoPanicClient(keys["cryptopanic"])
            posts = cryptopanic.posts(symbols)
            post_count = 0
            for symbol, rows in posts.items():
                if not rows:
                    metrics[symbol].provider_status["cryptopanic"] = {"state": "read_no_hit"}
                    continue
                post_count += len(rows)
                item = metrics[symbol]
                item.providers.append("CryptoPanic")
                item.values["cryptopanic_posts"] = [_news_summary(row) for row in rows[:5]]
                item.provider_status["cryptopanic"] = {"state": "read", "count": len(rows)}
                item.warnings.append(f"CryptoPanic 有 {len(rows)} 則重要新聞，避免在事件波動中盲目追價")
            _mark_provider(statuses, "cryptopanic", "可讀取", f"新聞資料可讀，本次命中 {post_count} 則重要新聞。", True, 1, 0)
        except PaidDataError as exc:
            errors.append(f"CryptoPanic: {exc}")
            for symbol in symbols:
                metrics[symbol].provider_status["cryptopanic"] = {"state": "failed", "error": str(exc)}
            _mark_provider(statuses, "cryptopanic", "讀不到", "CryptoPanic 本次讀取失敗。", False, 0, 1, str(exc))

    if keys.get("thetie"):
        try:
            thetie = TheTieClient(keys["thetie"])
            news = thetie.news_sentiment(symbols)
            thetie_count = 0
            for symbol, rows in news.items():
                if not rows:
                    metrics[symbol].provider_status["thetie"] = {"state": "read_no_hit"}
                    continue
                thetie_count += len(rows)
                item = metrics[symbol]
                item.providers.append("The Tie")
                item.values["thetie_news_count"] = len(rows)
                item.values["thetie_latest_news"] = _compact_payload(rows, max_items=3)
                item.provider_status["thetie"] = {"state": "read", "count": len(rows)}
                item.warnings.append(f"The Tie 偵測到 {len(rows)} 則相關新聞/情緒資料，進場前需檢查是否為事件驅動")
            _mark_provider(statuses, "thetie", "可讀取", f"The Tie 可讀，本次命中 {thetie_count} 筆新聞/情緒資料。", True, 1, 0)
        except PaidDataError as exc:
            errors.append(f"The Tie: {exc}")
            for symbol in symbols:
                metrics[symbol].provider_status["thetie"] = {"state": "failed", "error": str(exc)}
            _mark_provider(statuses, "thetie", "讀不到", "The Tie 本次讀取失敗。", False, 0, 1, str(exc))

    if keys.get("tokenmetrics"):
        tokenmetrics = TokenMetricsClient(keys["tokenmetrics"])
        tm_success = 0
        tm_failure = 0
        tm_error = ""
        for symbol in symbols:
            try:
                grade = tokenmetrics.tm_grade(symbol)
                if not grade:
                    metrics[symbol].provider_status["tokenmetrics"] = {"state": "read_no_hit"}
                    continue
                tm_success += 1
                item = metrics[symbol]
                item.providers.append("Token Metrics")
                item.values["tokenmetrics_tm_grade"] = grade
                item.provider_status["tokenmetrics"] = {"state": "read"}
                _score_tokenmetrics(item, grade)
            except PaidDataError as exc:
                tm_failure += 1
                tm_error = tm_error or str(exc)
                metrics[symbol].provider_status["tokenmetrics"] = {"state": "failed", "error": str(exc)}
                errors.append(f"Token Metrics {symbol}: {exc}")
        if tm_success:
            state = "可讀取" if not tm_failure else "部分可讀"
            _mark_provider(statuses, "tokenmetrics", state, f"本次讀到 {tm_success} 個幣種，失敗 {tm_failure} 個。", True, tm_success, tm_failure, tm_error)
        else:
            _mark_provider(statuses, "tokenmetrics", "讀不到", "Token Metrics 本次沒有讀到可用資料。", False, 0, tm_failure, tm_error)

    for report in reports:
        item = metrics[report.symbol]
        _apply_external(report.long, item, "long")
        _apply_external(report.short, item, "short")
        if item.warnings:
            report.long.warnings.extend(item.warnings)
            report.short.warnings.extend(item.warnings)
        report.metadata["paid_data"] = {
            "providers": sorted(set(item.providers)),
            "values": item.values,
            "bonus_long": item.bonus_long,
            "bonus_short": item.bonus_short,
            "warnings": item.warnings,
            "provider_status": item.provider_status,
            "configured_api_readiness": _configured_api_readiness(item, keys),
        }
        from .scoring import finalize_report_scores

        finalize_report_scores(report)
        continue
        selected = report.long.normalized if report.long.normalized >= report.short.normalized else report.short.normalized
        report.score = round(selected, 2)
        if max(report.long.normalized, report.short.normalized) < 52:
            report.selected_direction = "neutral"
        elif report.short.normalized > report.long.normalized:
            report.selected_direction = "short"
        else:
            report.selected_direction = "long"

    return {"enabled": True, "providers": list(statuses.values()), "errors": errors}


def check_provider_connections(config: dict[str, Any]) -> list[dict[str, Any]]:
    statuses = provider_statuses(config)
    keys = config.get("api_keys", {})
    paid_cfg = config.get("paid_data", {})
    checks = {
        "exchange_public": lambda: ExchangePublicDerivativesClient().symbol_metrics("BTCUSDT", "bybit-linear"),
        "coingecko": lambda: CoinGeckoClient(keys.get("coingecko", "")).check(),
        "defillama": lambda: DefiLlamaClient().check(),
        "cryptopanic": lambda: CryptoPanicClient(keys["cryptopanic"]).check(),
        "coinglass": lambda: CoinglassClient(keys["coinglass"]).check(),
        "coinalyze": lambda: CoinalyzeClient(keys["coinalyze"], paid_cfg.get("preferred_derivatives_exchange", "Bybit")).check(),
        "glassnode": lambda: GlassnodeClient(keys["glassnode"]).check(),
        "cryptoquant": lambda: CryptoQuantClient(keys["cryptoquant"]).check(),
        "coinmarketcal": lambda: CoinMarketCalClient(keys["coinmarketcal"]).check(),
        "thetie": lambda: TheTieClient(keys["thetie"]).check(),
        "tokenmetrics": lambda: TokenMetricsClient(keys["tokenmetrics"]).check(),
    }
    by_id = {status["id"]: status for status in statuses}
    for provider_id, check in checks.items():
        if provider_id not in NO_KEY_PROVIDERS and not keys.get(provider_id):
            continue
        started = time.time()
        try:
            check()
            by_id[provider_id]["state"] = "連線成功"
            by_id[provider_id]["message"] = f"API key 可用，耗時 {time.time() - started:.1f}s"
            by_id[provider_id]["readable"] = True
            by_id[provider_id]["success_count"] = 1
            by_id[provider_id]["failure_count"] = 0
            by_id[provider_id]["last_error"] = ""
        except Exception as exc:
            by_id[provider_id]["state"] = "連線失敗"
            by_id[provider_id]["message"] = str(exc)
            by_id[provider_id]["readable"] = False
            by_id[provider_id]["success_count"] = 0
            by_id[provider_id]["failure_count"] = 1
            by_id[provider_id]["last_error"] = str(exc)
    return statuses


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _configured_api_readiness(item: ExternalSymbolMetrics, keys: dict[str, Any]) -> dict[str, Any]:
    configured = [provider_id for provider_id, value in keys.items() if value]
    execution_required = ["exchange_public"] + [
        provider_id for provider_id in ("coinglass", "coinalyze") if keys.get(provider_id)
    ]
    status: dict[str, dict[str, Any]] = {}
    for provider_id in ["exchange_public", *configured]:
        raw = item.provider_status.get(provider_id)
        if isinstance(raw, dict):
            status[provider_id] = raw
        else:
            status[provider_id] = {"state": "pending"}

    execution_missing = [
        provider_id
        for provider_id in execution_required
        if status.get(provider_id, {}).get("state") not in {"read", "read_no_hit"}
    ]
    configured_failed = [
        provider_id
        for provider_id in configured
        if status.get(provider_id, {}).get("state") in {"failed", "pending"}
    ]
    return {
        "configured": configured,
        "execution_required": execution_required,
        "execution_ready": not execution_missing,
        "all_configured_reached": not configured_failed,
        "execution_missing": execution_missing,
        "configured_failed": configured_failed,
        "status": status,
    }


def _score_funding(item: ExternalSymbolMetrics, funding: float, source: str) -> None:
    item.values[f"{source.lower()}_funding"] = funding
    if funding <= -0.00015:
        item.bonus_long += 1.5
        item.reasons_long.append(f"{source} funding 偏負，空方擁擠，對看多反轉加分")
        item.warnings.append(f"{source} funding={funding:.5f}，可能有軋空/快速反抽風險")
    elif funding >= 0.00015:
        item.bonus_short += 1.5
        item.reasons_short.append(f"{source} funding 偏正，多方擁擠，對看空反轉加分")
        item.warnings.append(f"{source} funding={funding:.5f}，多方槓桿偏熱")


def _score_liquidation(item: ExternalSymbolMetrics, long_liq: float, short_liq: float, source: str) -> None:
    total = long_liq + short_liq
    if total <= 0:
        return
    long_share = long_liq / total
    short_share = short_liq / total
    if long_share >= 0.65:
        item.bonus_long += 1.0
        item.reasons_long.append(f"{source} 近期多單爆倉占比高，掃低後反彈機率提高")
    if short_share >= 0.65:
        item.bonus_short += 1.0
        item.reasons_short.append(f"{source} 近期空單爆倉占比高，掃高後回落需警惕")


def _score_netflow(item: ExternalSymbolMetrics, netflow: float, source: str) -> None:
    if netflow < 0:
        item.bonus_long += 1.2
        item.reasons_long.append(f"{source} 交易所淨流出，現貨籌碼壓力下降")
    elif netflow > 0:
        item.bonus_short += 1.2
        item.reasons_short.append(f"{source} 交易所淨流入，潛在賣壓上升")


def _score_tokenmetrics(item: ExternalSymbolMetrics, grade: dict[str, Any]) -> None:
    text = json.dumps(grade, ensure_ascii=False).lower()
    numeric_grade = None
    for key in ("TM_GRADE", "tm_grade", "trader_grade", "TRADER_GRADE"):
        if key in grade:
            numeric_grade = _as_float(grade.get(key))
            break
    if numeric_grade is not None:
        item.values["tokenmetrics_grade_value"] = numeric_grade
        if numeric_grade >= 75:
            item.bonus_long += 1.2
            item.reasons_long.append(f"Token Metrics TM Grade={numeric_grade:.1f}，外部 AI 評級偏強")
        elif numeric_grade <= 35:
            item.bonus_short += 1.2
            item.reasons_short.append(f"Token Metrics TM Grade={numeric_grade:.1f}，外部 AI 評級偏弱")
    if "strong buy" in text or "\"buy\"" in text:
        item.bonus_long += 0.8
        item.reasons_long.append("Token Metrics 訊號偏買入")
    if "strong sell" in text or "\"sell\"" in text:
        item.bonus_short += 0.8
        item.reasons_short.append("Token Metrics 訊號偏賣出")


def _apply_external(score: DirectionScore, item: ExternalSymbolMetrics, direction: str) -> None:
    bonus = item.bonus_long if direction == "long" else item.bonus_short
    reasons = item.reasons_long if direction == "long" else item.reasons_short
    if bonus <= 0:
        return
    bonus = min(3.0, bonus)
    if "paid_data" not in score.feature_max_scores:
        score.bonus_max_score += 3.0
        score.feature_max_scores["paid_data"] = 3.0
    score.score += bonus
    score.bonus_score += bonus
    score.feature_scores["paid_data"] = round(score.feature_scores.get("paid_data", 0.0) + bonus, 2)
    for reason in reasons:
        score.reasons.append(f"+{bonus:.1f}/3 外部資料輔助共振：{reason}")
    return
    for reason in reasons:
        score.reasons.append(f"+{bonus:.1f}/8 外部資料共振加分：{reason}")


def _sum_named(value: Any, names: tuple[str, ...]) -> float:
    total = 0.0
    if isinstance(value, dict):
        for key, item in value.items():
            if any(name.lower() in str(key).lower() for name in names):
                number = _last_number(item)
                if number is not None:
                    total += number
            else:
                total += _sum_named(item, names)
    elif isinstance(value, list):
        for item in value:
            total += _sum_named(item, names)
    return total


def _compact_payload(payload: Any, max_items: int = 5) -> Any:
    data = _extract_data(payload)
    if isinstance(data, list):
        return data[-max_items:]
    if isinstance(data, dict):
        compact: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, list):
                compact[key] = value[-max_items:]
            else:
                compact[key] = value
        return compact
    return data


def _event_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": row.get("title") or row.get("name"),
        "date": row.get("date_event") or row.get("date"),
        "category": row.get("category") or row.get("categories"),
        "confidence": row.get("confidence"),
        "source": row.get("source") or row.get("proof"),
    }


def _news_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": row.get("title"),
        "published_at": row.get("published_at") or row.get("created_at"),
        "url": row.get("url"),
        "domain": row.get("domain"),
        "votes": row.get("votes"),
    }
