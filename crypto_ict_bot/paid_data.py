from __future__ import annotations

import json
import time
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


PROVIDERS = [
    ProviderDefinition(
        id="exchange_public",
        name="Bybit / Binance 公開衍生品資料",
        category="免費內建 / OI + Funding",
        purpose="不需要 key，直接用交易所公開 API 補未平倉量與 funding，作為所有掃描的底層資料。",
        apply_url="https://bybit-exchange.github.io/docs/v5/market/open-interest",
        docs_url="https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest",
        key_hint="不需要 API key",
    ),
    ProviderDefinition(
        id="coinglass",
        name="CoinGlass",
        category="最佳衍生品 / 清算熱力圖",
        purpose="OI、funding、爆倉、清算熱力圖、訂單簿熱力圖。",
        apply_url="https://www.coinglass.com/api",
        docs_url="https://docs.coinglass.com/reference/getting-started-with-your-api",
        key_hint="CG-API-KEY",
    ),
    ProviderDefinition(
        id="coinalyze",
        name="Coinalyze",
        category="免費衍生品 / CVD",
        purpose="OI、funding、long/short、buy/sell、爆倉歷史；可補 ICT 位移背後的槓桿與主動買賣資料。",
        apply_url="https://coinalyze.net",
        docs_url="https://api.coinalyze.net/v1/doc/",
        key_hint="api_key",
    ),
    ProviderDefinition(
        id="coingecko",
        name="CoinGecko",
        category="多功能市場資料 / 免費 Demo",
        purpose="一個 API 補價格、市值、交易所、衍生品、鏈上 DEX、新聞與基本資料；免費 Demo 可先用。",
        apply_url="https://www.coingecko.com/en/api",
        docs_url="https://docs.coingecko.com/reference/setting-up-your-api-key",
        key_hint="x-cg-demo-api-key",
    ),
    ProviderDefinition(
        id="defillama",
        name="DeFiLlama",
        category="免費 DeFi / 穩定幣 / TVL",
        purpose="免費公開 API，補 DeFi TVL、穩定幣供給、DEX 量、費用與鏈上宏觀背景。",
        apply_url="https://defillama.com/docs/api",
        docs_url="https://defillama.com/docs/api",
        key_hint="不需要 API key",
    ),
    ProviderDefinition(
        id="cryptopanic",
        name="CryptoPanic",
        category="新聞事件 / 免費 + 付費",
        purpose="加密新聞、情緒、重要事件；免費 token 可先用，PLUS 可升級更多欄位。",
        apply_url="https://cryptopanic.com/developers/api/",
        docs_url="https://cryptopanic.com/developers/api/",
        key_hint="auth_token",
    ),
    ProviderDefinition(
        id="glassnode",
        name="Glassnode",
        category="鏈上資金流",
        purpose="交易所淨流入/淨流出、ETF/衍生品與鏈上週期資料。",
        apply_url="https://studio.glassnode.com/pricing",
        docs_url="https://docs.glassnode.com/basic-api/api",
        key_hint="X-Api-Key 或 api_key",
    ),
    ProviderDefinition(
        id="cryptoquant",
        name="CryptoQuant",
        category="鏈上資金流",
        purpose="交易所 reserve/netflow、礦工、穩定幣與專業鏈上指標。",
        apply_url="https://cryptoquant.com/en/pricing",
        docs_url="https://cryptoquant.com/en/docs",
        key_hint="Authorization: Bearer",
    ),
    ProviderDefinition(
        id="coinmarketcal",
        name="CoinMarketCal",
        category="事件日曆",
        purpose="上幣、解鎖、升級、監管、投票等可能造成誘導或劇烈波動的事件。",
        apply_url="https://coinmarketcal.com/en/api",
        docs_url="https://www.postman.com/coinmarketcalapi/coinmarketcal-api/documentation/61mcdog/coinmarketcal-api",
        key_hint="x-api-key",
    ),
    ProviderDefinition(
        id="thetie",
        name="The Tie",
        category="最佳新聞 / 情緒 / 解鎖",
        purpose="新聞、情緒、token unlock、開發活動與鏈上基本面；偏機構級。",
        apply_url="https://www.thetie.io/solutions/api",
        docs_url="https://docs.thetie.io/reference",
        key_hint="API key",
    ),
    ProviderDefinition(
        id="tokenmetrics",
        name="Token Metrics",
        category="AI 評級 / 情緒 / 訊號",
        purpose="交易評級、投資評級、市場情緒與 AI 指標；作為外部共振而非取代本策略。",
        apply_url="https://www.tokenmetrics.com/api",
        docs_url="https://developers.tokenmetrics.com/docs/getting-started",
        key_hint="x-api-key",
    ),
]

NO_KEY_PROVIDERS = {"exchange_public", "defillama"}


def _request_json(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 12,
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
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PaidDataError(str(exc)) from exc


def _base_symbol(symbol: str) -> str:
    return symbol.upper().replace("USDT", "").replace("USD", "")


class ExchangePublicDerivativesClient:
    bybit_base_url = "https://api.bybit.com"
    binance_base_url = "https://fapi.binance.com"

    def symbol_metrics(self, symbol: str, exchange: str) -> dict[str, Any]:
        if "binance" in exchange.lower():
            return self._binance(symbol)
        return self._bybit(symbol)

    def _bybit(self, symbol: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        funding = _request_json(
            self.bybit_base_url,
            "/v5/market/funding/history",
            {"category": "linear", "symbol": symbol, "limit": 1},
        )
        rows = funding.get("result", {}).get("list", []) if isinstance(funding, dict) else []
        if rows:
            result["funding_rate"] = _as_float(rows[0].get("fundingRate"))
        oi = _request_json(
            self.bybit_base_url,
            "/v5/market/open-interest",
            {"category": "linear", "symbol": symbol, "intervalTime": "1h", "limit": 2},
        )
        oi_rows = oi.get("result", {}).get("list", []) if isinstance(oi, dict) else []
        values = [_as_float(row.get("openInterest")) for row in oi_rows]
        values = [value for value in values if value is not None]
        if values:
            result["open_interest"] = values[0]
            if len(values) >= 2 and values[1]:
                result["open_interest_change_pct"] = (values[0] - values[1]) / abs(values[1]) * 100.0
        return result

    def _binance(self, symbol: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        premium = _request_json(self.binance_base_url, "/fapi/v1/premiumIndex", {"symbol": symbol})
        if isinstance(premium, dict):
            result["funding_rate"] = _as_float(premium.get("lastFundingRate"))
        oi = _request_json(self.binance_base_url, "/fapi/v1/openInterest", {"symbol": symbol})
        if isinstance(oi, dict):
            result["open_interest"] = _as_float(oi.get("openInterest"))
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
        )
        statuses.append(status)
    return [status.__dict__ for status in statuses]


def enrich_reports(reports: list[SymbolReport], config: dict[str, Any]) -> dict[str, Any]:
    if not config.get("paid_data", {}).get("enabled", True):
        return {"enabled": False, "providers": provider_statuses(config), "errors": []}

    metrics = {report.symbol: ExternalSymbolMetrics(report.symbol) for report in reports}
    errors: list[str] = []
    keys = config.get("api_keys", {})
    paid_cfg = config.get("paid_data", {})
    symbols = [report.symbol for report in reports]

    public_derivatives = ExchangePublicDerivativesClient()
    for report in reports[: min(len(reports), 70)]:
        try:
            values = public_derivatives.symbol_metrics(report.symbol, report.exchange)
            item = metrics[report.symbol]
            item.providers.append("交易所公開衍生品")
            item.values["exchange_public_derivatives"] = values
            funding = _as_float(values.get("funding_rate"))
            if funding is not None:
                _score_funding(item, funding, "交易所公開資料")
            oi_change = _as_float(values.get("open_interest_change_pct"))
            if oi_change is not None and abs(oi_change) >= 8:
                item.warnings.append(f"公開 OI 近 1h 變動 {oi_change:.2f}%，槓桿流入/流出較劇烈")
        except PaidDataError as exc:
            metrics[report.symbol].warnings.append(f"交易所公開衍生品資料暫時不可用：{exc}")

    if keys.get("coingecko"):
        try:
            coingecko = CoinGeckoClient(keys["coingecko"])
            cg_prices = coingecko.simple_prices(symbols)
            for symbol in symbols:
                base = _base_symbol(symbol).lower()
                if base not in cg_prices:
                    continue
                item = metrics[symbol]
                item.providers.append("CoinGecko")
                item.values["coingecko"] = cg_prices[base]
        except PaidDataError as exc:
            errors.append(f"CoinGecko: {exc}")

    try:
        llama = DefiLlamaClient()
        macro = llama.macro_snapshot()
        if macro.get("total_tvl"):
            for item in metrics.values():
                item.providers.append("DeFiLlama")
                item.values["defillama_macro"] = macro
    except PaidDataError as exc:
        errors.append(f"DeFiLlama: {exc}")

    if keys.get("coinalyze"):
        try:
            coinalyze = CoinalyzeClient(keys["coinalyze"], paid_cfg.get("preferred_derivatives_exchange", "Bybit"))
            coinalyze_data = coinalyze.metrics(symbols)
            for symbol, values in coinalyze_data.items():
                item = metrics[symbol]
                item.providers.append("Coinalyze")
                item.values["coinalyze"] = values
                funding = _as_float(values.get("funding_rate"))
                predicted = _as_float(values.get("predicted_funding_rate"))
                funding_value = predicted if predicted is not None else funding
                if funding_value is not None:
                    _score_funding(item, funding_value, "Coinalyze")
        except PaidDataError as exc:
            errors.append(f"Coinalyze: {exc}")

    if keys.get("coinglass"):
        coinglass = CoinglassClient(keys["coinglass"], paid_cfg.get("preferred_derivatives_exchange", "Bybit"))
        for symbol in symbols[: min(len(symbols), 30)]:
            try:
                values = coinglass.symbol_metrics(symbol)
                item = metrics[symbol]
                item.providers.append("CoinGlass")
                item.values["coinglass"] = _compact_payload(values)
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
                errors.append(f"CoinGlass {symbol}: {exc}")

    if keys.get("glassnode"):
        glassnode = GlassnodeClient(keys["glassnode"], paid_cfg.get("glassnode_interval", "1h"))
        for symbol in symbols[: min(len(symbols), 20)]:
            try:
                netflow = glassnode.exchange_netflow(symbol)
                if netflow is None:
                    continue
                item = metrics[symbol]
                item.providers.append("Glassnode")
                item.values["glassnode_exchange_netflow_usd"] = netflow
                _score_netflow(item, netflow, "Glassnode")
            except PaidDataError as exc:
                errors.append(f"Glassnode {symbol}: {exc}")

    if keys.get("cryptoquant"):
        cryptoquant = CryptoQuantClient(keys["cryptoquant"])
        for symbol in symbols[: min(len(symbols), 20)]:
            try:
                netflow = cryptoquant.exchange_netflow(symbol, paid_cfg.get("cryptoquant_window", "day"))
                if netflow is None:
                    continue
                item = metrics[symbol]
                item.providers.append("CryptoQuant")
                item.values["cryptoquant_exchange_netflow"] = netflow
                _score_netflow(item, netflow, "CryptoQuant")
            except PaidDataError as exc:
                errors.append(f"CryptoQuant {symbol}: {exc}")

    if keys.get("coinmarketcal"):
        try:
            calendar = CoinMarketCalClient(keys["coinmarketcal"], int(paid_cfg.get("event_lookahead_days", 7)))
            events = calendar.upcoming_events(symbols)
            for symbol, rows in events.items():
                if not rows:
                    continue
                item = metrics[symbol]
                item.providers.append("CoinMarketCal")
                item.values["coinmarketcal_events"] = [_event_summary(row) for row in rows[:5]]
                item.warnings.append(f"未來 {paid_cfg.get('event_lookahead_days', 7)} 天有 {len(rows)} 個事件，需避開新聞誘導與跳空")
        except PaidDataError as exc:
            errors.append(f"CoinMarketCal: {exc}")

    if keys.get("cryptopanic"):
        try:
            cryptopanic = CryptoPanicClient(keys["cryptopanic"])
            posts = cryptopanic.posts(symbols)
            for symbol, rows in posts.items():
                if not rows:
                    continue
                item = metrics[symbol]
                item.providers.append("CryptoPanic")
                item.values["cryptopanic_posts"] = [_news_summary(row) for row in rows[:5]]
                item.warnings.append(f"CryptoPanic 有 {len(rows)} 則重要新聞，避免在事件波動中盲目追價")
        except PaidDataError as exc:
            errors.append(f"CryptoPanic: {exc}")

    if keys.get("thetie"):
        try:
            thetie = TheTieClient(keys["thetie"])
            news = thetie.news_sentiment(symbols)
            for symbol, rows in news.items():
                if not rows:
                    continue
                item = metrics[symbol]
                item.providers.append("The Tie")
                item.values["thetie_news_count"] = len(rows)
                item.values["thetie_latest_news"] = _compact_payload(rows, max_items=3)
                item.warnings.append(f"The Tie 偵測到 {len(rows)} 則相關新聞/情緒資料，進場前需檢查是否為事件驅動")
        except PaidDataError as exc:
            errors.append(f"The Tie: {exc}")

    if keys.get("tokenmetrics"):
        tokenmetrics = TokenMetricsClient(keys["tokenmetrics"])
        for symbol in symbols[: min(len(symbols), 30)]:
            try:
                grade = tokenmetrics.tm_grade(symbol)
                if not grade:
                    continue
                item = metrics[symbol]
                item.providers.append("Token Metrics")
                item.values["tokenmetrics_tm_grade"] = grade
                _score_tokenmetrics(item, grade)
            except PaidDataError as exc:
                errors.append(f"Token Metrics {symbol}: {exc}")

    for report in reports:
        item = metrics[report.symbol]
        _apply_external(report.long, item, "long")
        _apply_external(report.short, item, "short")
        if item.warnings:
            report.long.warnings.extend(item.warnings)
            report.short.warnings.extend(item.warnings)
        if item.providers or item.values:
            report.metadata["paid_data"] = {
                "providers": sorted(set(item.providers)),
                "values": item.values,
                "bonus_long": item.bonus_long,
                "bonus_short": item.bonus_short,
                "warnings": item.warnings,
            }
        selected = report.long.normalized if report.long.normalized >= report.short.normalized else report.short.normalized
        report.score = round(selected, 2)
        if max(report.long.normalized, report.short.normalized) < 52:
            report.selected_direction = "neutral"
        elif report.short.normalized > report.long.normalized:
            report.selected_direction = "short"
        else:
            report.selected_direction = "long"

    return {"enabled": True, "providers": provider_statuses(config), "errors": errors}


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
        except Exception as exc:
            by_id[provider_id]["state"] = "連線失敗"
            by_id[provider_id]["message"] = str(exc)
    return statuses


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
    bonus = min(8.0, bonus)
    score.score += bonus
    score.feature_scores["paid_data"] = round(score.feature_scores.get("paid_data", 0.0) + bonus, 2)
    for reason in reasons:
        score.reasons.append(f"+{bonus:.1f}/8 付費資料共振：{reason}")


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
