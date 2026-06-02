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


def _iso_from_ms(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


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


def _level_pairs(rows: Any) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    if not isinstance(rows, list):
        return output
    for row in rows:
        price: Any = None
        size: Any = None
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            price, size = row[0], row[1]
        elif isinstance(row, dict):
            price = row.get("price") or row.get("p")
            size = row.get("size") or row.get("qty") or row.get("quantity") or row.get("q")
        price_f = _as_float(price)
        size_f = _as_float(size)
        if price_f is None or size_f is None or price_f <= 0 or size_f <= 0:
            continue
        output.append((price_f, size_f))
    return output


def _orderbook_metrics(bids_raw: Any, asks_raw: Any) -> dict[str, Any]:
    bids = sorted(_level_pairs(bids_raw), key=lambda item: item[0], reverse=True)
    asks = sorted(_level_pairs(asks_raw), key=lambda item: item[0])
    if not bids or not asks:
        return {}
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    bid_notional = sum(price * size for price, size in bids)
    ask_notional = sum(price * size for price, size in asks)
    total = bid_notional + ask_notional
    if total <= 0 or mid <= 0:
        return {}
    bid_wall = _nearest_significant_wall(bids, mid, "bid")
    ask_wall = _nearest_significant_wall(asks, mid, "ask")
    imbalance = (bid_notional - ask_notional) / total
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_pct": (best_ask - best_bid) / mid * 100.0,
        "bid_notional": bid_notional,
        "ask_notional": ask_notional,
        "depth_imbalance": imbalance,
        "bid_wall_price": bid_wall.get("price"),
        "bid_wall_notional": bid_wall.get("notional"),
        "bid_wall_distance_pct": _distance_pct(mid, bid_wall.get("price")),
        "ask_wall_price": ask_wall.get("price"),
        "ask_wall_notional": ask_wall.get("notional"),
        "ask_wall_distance_pct": _distance_pct(mid, ask_wall.get("price")),
    }


def _nearest_significant_wall(levels: list[tuple[float, float]], mid: float, side: str) -> dict[str, float | None]:
    if not levels or mid <= 0:
        return {"price": None, "notional": None}
    notionals = [(price, price * size) for price, size in levels]
    max_notional = max(notional for _, notional in notionals)
    avg_notional = sum(notional for _, notional in notionals) / max(len(notionals), 1)
    threshold = max(avg_notional * 1.8, max_notional * 0.32)
    candidates = [(price, notional) for price, notional in notionals if notional >= threshold]
    if side == "bid":
        candidates = [(price, notional) for price, notional in candidates if price <= mid]
    else:
        candidates = [(price, notional) for price, notional in candidates if price >= mid]
    if not candidates:
        price, notional = max(notionals, key=lambda item: item[1])
        return {"price": price, "notional": notional}
    price, notional = min(candidates, key=lambda item: abs(item[0] - mid))
    return {"price": price, "notional": notional}


def _distance_pct(base: float, level: Any) -> float | None:
    level_f = _as_float(level)
    if level_f is None or base <= 0:
        return None
    return abs(level_f - base) / base * 100.0


def _trade_flow_metrics(buy_notional: float, sell_notional: float, trade_count: int) -> dict[str, Any]:
    total = buy_notional + sell_notional
    if total <= 0:
        return {}
    return {
        "taker_buy_notional": buy_notional,
        "taker_sell_notional": sell_notional,
        "taker_buy_ratio": buy_notional / total,
        "taker_delta_notional": buy_notional - sell_notional,
        "trade_count": trade_count,
    }


def _bybit_trade_flow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buy_notional = 0.0
    sell_notional = 0.0
    trade_count = 0
    for row in rows:
        price = _as_float(row.get("price"))
        size = _as_float(row.get("size"))
        if price is None or size is None:
            continue
        notional = price * size
        side = str(row.get("side") or "").lower()
        if side == "buy":
            buy_notional += notional
        elif side == "sell":
            sell_notional += notional
        trade_count += 1
    return _trade_flow_metrics(buy_notional, sell_notional, trade_count)


def _binance_trade_flow(rows: Any) -> dict[str, Any]:
    buy_notional = 0.0
    sell_notional = 0.0
    trade_count = 0
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        price = _as_float(row.get("p"))
        qty = _as_float(row.get("q"))
        if price is None or qty is None:
            continue
        notional = price * qty
        if bool(row.get("m")):
            sell_notional += notional
        else:
            buy_notional += notional
        trade_count += 1
    return _trade_flow_metrics(buy_notional, sell_notional, trade_count)


def _latest_ratio(rows: Any) -> dict[str, Any]:
    row: Any = None
    if isinstance(rows, list) and rows:
        row = rows[-1]
    elif isinstance(rows, dict):
        data = _extract_data(rows)
        if isinstance(data, list) and data:
            row = data[-1]
        elif isinstance(data, dict):
            row = data
    if not isinstance(row, dict):
        return {}
    ratio = (
        _as_float(row.get("longShortRatio"))
        or _as_float(row.get("long_short_ratio"))
        or _as_float(row.get("buySellRatio"))
        or _as_float(row.get("buy_sell_ratio"))
        or _as_float(row.get("ratio"))
        or _as_float(row.get("r"))
    )
    long_share = _as_float(row.get("longAccount")) or _as_float(row.get("long_account")) or _as_float(row.get("l"))
    short_share = _as_float(row.get("shortAccount")) or _as_float(row.get("short_account")) or _as_float(row.get("s"))
    return {
        "ratio": ratio,
        "long_share": long_share,
        "short_share": short_share,
        "timestamp": row.get("timestamp") or row.get("time") or row.get("t"),
    }


def _history_change_pct(history: Any) -> float | None:
    rows = history if isinstance(history, list) else []
    values = [_as_float(row.get("c") if isinstance(row, dict) else None) for row in rows]
    values = [value for value in values if value is not None]
    if len(values) < 2 or values[0] == 0:
        return None
    return (values[-1] - values[0]) / abs(values[0]) * 100.0


def _history_sum(history: Any, key: str) -> float:
    total = 0.0
    for row in history if isinstance(history, list) else []:
        if isinstance(row, dict):
            total += _as_float(row.get(key)) or 0.0
    return total


def _compact_history_summary(history: Any) -> dict[str, Any]:
    rows = history if isinstance(history, list) else []
    return {
        "points": len(rows),
        "first_time": rows[0].get("t") if rows and isinstance(rows[0], dict) else None,
        "last_time": rows[-1].get("t") if rows and isinstance(rows[-1], dict) else None,
        "change_pct": _history_change_pct(rows),
    }


def _collect_price_value_pairs(payload: Any, current_price: float, max_depth: int = 5) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []

    def walk(value: Any, depth: int = 0) -> None:
        if depth > max_depth:
            return
        if isinstance(value, dict):
            price = None
            weight = None
            for key in ("price", "p", "level", "priceLevel"):
                if key in value:
                    price = _as_float(value.get(key))
                    break
            for key in ("value", "v", "amount", "size", "volume", "liq", "liquidation", "long", "short"):
                if key in value:
                    found = _as_float(value.get(key))
                    if found is not None:
                        weight = (weight or 0.0) + abs(found)
            if price is not None and weight is not None and current_price > 0:
                if 0.45 * current_price <= price <= 1.85 * current_price:
                    pairs.append((price, weight))
            for item in value.values():
                walk(item, depth + 1)
        elif isinstance(value, list):
            if len(value) >= 2 and all(isinstance(item, (int, float, str)) for item in value[:3]):
                nums = [_as_float(item) for item in value[:4]]
                nums = [num for num in nums if num is not None]
                for i, number in enumerate(nums):
                    if current_price > 0 and 0.45 * current_price <= number <= 1.85 * current_price:
                        weight_candidates = [abs(candidate) for j, candidate in enumerate(nums) if j != i and abs(candidate) > 0]
                        if weight_candidates:
                            pairs.append((number, max(weight_candidates)))
                            break
            for item in value:
                walk(item, depth + 1)

    walk(_extract_data(payload))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[:20]


def _heatmap_level_summary(payload: Any, current_price: float) -> dict[str, Any]:
    pairs = _collect_price_value_pairs(payload, current_price)
    above = [(price, value) for price, value in pairs if price > current_price]
    below = [(price, value) for price, value in pairs if price < current_price]
    nearest_above = min(above, key=lambda item: item[0] - current_price) if above else None
    nearest_below = min(below, key=lambda item: current_price - item[0]) if below else None
    return {
        "level_count": len(pairs),
        "nearest_above_price": nearest_above[0] if nearest_above else None,
        "nearest_above_weight": nearest_above[1] if nearest_above else None,
        "nearest_above_distance_pct": _distance_pct(current_price, nearest_above[0]) if nearest_above else None,
        "nearest_below_price": nearest_below[0] if nearest_below else None,
        "nearest_below_weight": nearest_below[1] if nearest_below else None,
        "nearest_below_distance_pct": _distance_pct(current_price, nearest_below[0]) if nearest_below else None,
        "top_levels": [{"price": price, "weight": weight} for price, weight in pairs[:5]],
    }


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
        result: dict[str, Any] = {"fetched_at": datetime.now(timezone.utc).isoformat()}
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
                result["funding_time"] = _iso_from_ms(rows[0].get("fundingRateTimestamp"))
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
                result["open_interest_time"] = _iso_from_ms(oi_rows[0].get("timestamp")) if oi_rows else None
                if len(values) >= 2 and values[1]:
                    result["open_interest_change_pct"] = (values[0] - values[1]) / abs(values[1]) * 100.0
                    result["open_interest_previous_time"] = (
                        _iso_from_ms(oi_rows[1].get("timestamp")) if len(oi_rows) >= 2 else None
                    )
        except PaidDataError as exc:
            errors.append(f"open_interest: {exc}")
        try:
            orderbook = _request_json(
                self.bybit_base_url,
                "/v5/market/orderbook",
                {"category": "linear", "symbol": symbol, "limit": 50},
                timeout=4,
            )
            result_data = orderbook.get("result", {}) if isinstance(orderbook, dict) else {}
            metrics = _orderbook_metrics(result_data.get("b"), result_data.get("a"))
            if metrics:
                result["orderbook"] = metrics
                result["spread_pct"] = metrics.get("spread_pct")
        except PaidDataError as exc:
            errors.append(f"orderbook: {exc}")
        try:
            trades = _request_json(
                self.bybit_base_url,
                "/v5/market/recent-trade",
                {"category": "linear", "symbol": symbol, "limit": 100},
                timeout=4,
            )
            flow = _bybit_trade_flow(_bybit_rows(trades))
            if flow:
                result["trade_flow"] = flow
        except PaidDataError as exc:
            errors.append(f"recent_trades: {exc}")
        if errors:
            result["partial_errors"] = errors
        if not _has_derivative_values(result):
            raise PaidDataError("; ".join(errors) or "Bybit returned no usable values")
        return result

    def _binance(self, symbol: str) -> dict[str, Any]:
        result: dict[str, Any] = {"fetched_at": datetime.now(timezone.utc).isoformat()}
        errors: list[str] = []
        try:
            premium = _request_json(self.binance_base_url, "/fapi/v1/premiumIndex", {"symbol": symbol})
            if isinstance(premium, dict):
                result["funding_rate"] = _as_float(premium.get("lastFundingRate"))
                result["funding_time"] = _iso_from_ms(premium.get("time"))
                result["next_funding_time"] = _iso_from_ms(premium.get("nextFundingTime"))
        except PaidDataError as exc:
            errors.append(f"funding: {exc}")
        try:
            oi = _request_json(self.binance_base_url, "/fapi/v1/openInterest", {"symbol": symbol})
            if isinstance(oi, dict):
                result["open_interest"] = _as_float(oi.get("openInterest"))
                result["open_interest_time"] = _iso_from_ms(oi.get("time"))
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
                result["open_interest_previous_time"] = _iso_from_ms(history[-2].get("timestamp"))
                result["open_interest_time"] = _iso_from_ms(history[-1].get("timestamp")) or result.get("open_interest_time")
        except PaidDataError as exc:
            errors.append(f"open_interest_history: {exc}")
        try:
            depth = _request_json(self.binance_base_url, "/fapi/v1/depth", {"symbol": symbol, "limit": 100}, timeout=4)
            if isinstance(depth, dict):
                metrics = _orderbook_metrics(depth.get("bids"), depth.get("asks"))
                if metrics:
                    result["orderbook"] = metrics
                    result["spread_pct"] = metrics.get("spread_pct")
        except PaidDataError as exc:
            errors.append(f"orderbook: {exc}")
        try:
            trades = _request_json(self.binance_base_url, "/fapi/v1/aggTrades", {"symbol": symbol, "limit": 100}, timeout=4)
            flow = _binance_trade_flow(trades)
            if flow:
                result["trade_flow"] = flow
        except PaidDataError as exc:
            errors.append(f"agg_trades: {exc}")
        for endpoint, key in (
            ("/futures/data/takerlongshortRatio", "taker_long_short_ratio"),
            ("/futures/data/globalLongShortAccountRatio", "global_long_short_ratio"),
            ("/futures/data/topLongShortPositionRatio", "top_long_short_position_ratio"),
        ):
            try:
                rows = _request_json(
                    self.binance_base_url,
                    endpoint,
                    {"symbol": symbol, "period": "5m", "limit": 12},
                    timeout=4,
                )
                ratio = _latest_ratio(rows)
                if ratio:
                    result[key] = ratio
            except PaidDataError as exc:
                errors.append(f"{key}: {exc}")
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

    def symbol_metrics(self, symbol: str, include_heatmaps: bool = False) -> dict[str, Any]:
        coin = _base_symbol(symbol)
        pair_symbol = symbol.upper()
        exchange = self.exchange or "Binance"
        result: dict[str, Any] = {}
        try:
            result["open_interest"] = _request_json(
                self.base_url,
                "/api/futures/open-interest/aggregated-history",
                {"symbol": coin, "interval": "1h", "limit": 24, "unit": "usd"},
                self.headers,
                timeout=5,
            )
        except PaidDataError as exc:
            result["open_interest_error"] = str(exc)
        try:
            result["funding"] = _request_json(
                self.base_url,
                "/api/futures/funding-rate/vol-weight-history",
                {"symbol": coin, "interval": "1h", "limit": 24},
                self.headers,
                timeout=5,
            )
        except PaidDataError as exc:
            result["funding_error"] = str(exc)
        try:
            result["liquidation"] = _request_json(
                self.base_url,
                "/api/futures/liquidation/aggregated-history",
                {"exchange_list": "Binance,OKX,Bybit", "symbol": coin, "interval": "1h", "limit": 24},
                self.headers,
                timeout=5,
            )
        except PaidDataError as exc:
            result["liquidation_error"] = str(exc)
        try:
            result["taker_buy_sell"] = _request_json(
                self.base_url,
                "/api/futures/aggregated-taker-buy-sell-volume/history",
                {
                    "exchange_list": "Binance,OKX,Bybit",
                    "symbol": coin,
                    "interval": "1h",
                    "limit": 24,
                    "unit": "usd",
                },
                self.headers,
                timeout=5,
            )
        except PaidDataError as exc:
            result["taker_buy_sell_error"] = str(exc)
        try:
            result["long_short_ratio"] = _request_json(
                self.base_url,
                "/api/futures/global-long-short-account-ratio/history",
                {"exchange": exchange, "symbol": pair_symbol, "interval": "1h", "limit": 24},
                self.headers,
                timeout=5,
            )
        except PaidDataError as exc:
            result["long_short_ratio_error"] = str(exc)
        if include_heatmaps:
            try:
                result["liquidation_heatmap"] = _request_json(
                    self.base_url,
                    "/api/futures/liquidation/aggregated-heatmap/model1",
                    {"symbol": coin, "range": "24h"},
                    self.headers,
                    timeout=5,
                )
            except PaidDataError as exc:
                result["liquidation_heatmap_error"] = str(exc)
            try:
                result["liquidation_map"] = _request_json(
                    self.base_url,
                    "/api/futures/liquidation/aggregated-map",
                    {"symbol": coin, "range": "1d"},
                    self.headers,
                    timeout=5,
                )
            except PaidDataError as exc:
                result["liquidation_map_error"] = str(exc)
            try:
                result["orderbook_heatmap"] = _request_json(
                    self.base_url,
                    "/api/spot/orderbook/history",
                    {"exchange": exchange, "symbol": pair_symbol, "interval": "5m", "limit": 12},
                    self.headers,
                    timeout=5,
                )
            except PaidDataError as exc:
                result["orderbook_heatmap_error"] = str(exc)
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

    def metrics(self, symbols: list[str], history_symbol_limit: int = 0) -> dict[str, dict[str, Any]]:
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
        history_api_symbols = api_symbols[: max(0, int(history_symbol_limit or 0))]
        if history_api_symbols:
            now = int(datetime.now(timezone.utc).timestamp())
            since = now - 6 * 60 * 60
            for start in range(0, len(history_api_symbols), 20):
                chunk = history_api_symbols[start : start + 20]
                base_params = {
                    "symbols": ",".join(chunk),
                    "interval": "15min",
                    "from": since,
                    "to": now,
                }
                for endpoint, key in (
                    ("/open-interest-history", "open_interest_history"),
                    ("/liquidation-history", "liquidation_history"),
                    ("/long-short-ratio-history", "long_short_ratio_history"),
                    ("/ohlcv-history", "ohlcv_history"),
                ):
                    params = dict(base_params)
                    if key in {"open_interest_history", "liquidation_history"}:
                        params["convert_to_usd"] = "true"
                    try:
                        rows = _request_json(self.base_url, endpoint, params, self.headers)
                    except PaidDataError as exc:
                        for symbol, api_symbol in resolved.items():
                            if api_symbol in chunk:
                                output.setdefault(symbol, {})[f"{key}_error"] = str(exc)
                        continue
                    by_api_symbol = {row.get("symbol"): row for row in rows or [] if isinstance(row, dict)}
                    for symbol, api_symbol in resolved.items():
                        if api_symbol not in chunk or api_symbol not in by_api_symbol:
                            continue
                        history = by_api_symbol[api_symbol].get("history") or []
                        if not isinstance(history, list):
                            continue
                        compact_history = history[-24:]
                        item = output.setdefault(symbol, {})
                        item[key] = compact_history
                        item[f"{key}_summary"] = _compact_history_summary(compact_history)
                        if key == "open_interest_history":
                            change_pct = _history_change_pct(compact_history)
                            if change_pct is not None:
                                item["open_interest_change_pct"] = change_pct
                        elif key == "liquidation_history":
                            item["liquidation_sum"] = {
                                "long": _history_sum(compact_history, "l"),
                                "short": _history_sum(compact_history, "s"),
                            }
                        elif key == "long_short_ratio_history":
                            ratio = _latest_ratio(compact_history)
                            if ratio:
                                item["long_short_ratio"] = ratio
                        elif key == "ohlcv_history":
                            buy = _history_sum(compact_history, "bv")
                            total = _history_sum(compact_history, "v")
                            sell = max(0.0, total - buy)
                            flow = _trade_flow_metrics(buy, sell, len(compact_history))
                            if flow:
                                item["buy_sell_volume"] = flow
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
    report_by_symbol = {report.symbol: report for report in reports}
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
            history_limit = int(paid_cfg.get("coinalyze_history_symbol_limit", 10) or 0)
            coinalyze_data = coinalyze.metrics(symbols, history_limit)
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
        heatmap_limit = max(0, int(paid_cfg.get("coinglass_heatmap_symbol_limit", 20) or 0))
        heatmap_symbols = set(symbols[:heatmap_limit])
        coinglass_workers = max(1, min(8, int(paid_cfg.get("coinglass_workers", 5) or 5)))
        configured_timeout = paid_cfg.get("coinglass_timeout_seconds")
        coinglass_timeout = max(30.0, min(240.0, float(configured_timeout or max(60, len(symbols) * 2.4))))
        executor = ThreadPoolExecutor(max_workers=coinglass_workers)
        future_map = {
            executor.submit(coinglass.symbol_metrics, symbol, symbol in heatmap_symbols): symbol
            for symbol in symbols
        }
        try:
            completed = as_completed(future_map, timeout=coinglass_timeout)
            for future in completed:
                symbol = future_map[future]
                try:
                    values = future.result()
                except PaidDataError as exc:
                    coinglass_failure += 1
                    coinglass_error = coinglass_error or str(exc)
                    metrics[symbol].provider_status["coinglass"] = {"state": "failed", "error": str(exc)}
                    errors.append(f"CoinGlass {symbol}: {exc}")
                    continue
                if not _payload_has_data(values):
                    coinglass_failure += 1
                    error_text = _payload_error_summary(values) or "CoinGlass returned only empty/error payloads"
                    coinglass_error = coinglass_error or error_text
                    metrics[symbol].provider_status["coinglass"] = {"state": "failed", "error": error_text}
                    continue
                coinglass_success += 1
                item = metrics[symbol]
                _ingest_coinglass_values(item, values, report_by_symbol.get(symbol))
        except FuturesTimeoutError:
            coinglass_error = coinglass_error or f"CoinGlass timeout after {coinglass_timeout:.0f}s"
        finally:
            for future, symbol in future_map.items():
                if not future.done():
                    future.cancel()
                    coinglass_failure += 1
                    metrics[symbol].provider_status["coinglass"] = {
                        "state": "failed",
                        "error": "CoinGlass request cancelled by scan timeout",
                    }
            executor.shutdown(wait=False, cancel_futures=True)
        for symbol in []:
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
        _score_external_strategy_context(item, report)
        _adjust_trade_plan_with_external_context(report, item)
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


def _score_external_strategy_context(item: ExternalSymbolMetrics, report: SymbolReport) -> None:
    context = _build_external_strategy_context(item, report)
    if not context or int(context.get("evidence_count") or 0) <= 0:
        return
    item.values["external_strategy_context"] = context
    long_score = _as_float(context.get("long_score")) or 50.0
    short_score = _as_float(context.get("short_score")) or 50.0
    delta = long_score - short_score
    summary = str(context.get("summary") or "")
    if delta >= 6.0:
        bonus = min(2.8, 0.8 + (delta - 6.0) / 8.0)
        item.bonus_long += bonus
        item.reasons_long.append(f"external orderflow/liquidation context favors long: {summary}")
    elif delta <= -6.0:
        bonus = min(2.8, 0.8 + (abs(delta) - 6.0) / 8.0)
        item.bonus_short += bonus
        item.reasons_short.append(f"external orderflow/liquidation context favors short: {summary}")
    for flag in context.get("risk_flags", []) if isinstance(context.get("risk_flags"), list) else []:
        if flag == "long_crowded":
            item.warnings.append("External data shows crowded longs; avoid late long chase before a sweep/retest.")
        elif flag == "short_crowded":
            item.warnings.append("External data shows crowded shorts; avoid late short chase before a sweep/retest.")
        elif flag == "stop_hunt_risk_long":
            item.warnings.append("Liquidity/liquidation magnet is close below; long stop should not sit at an obvious minor low.")
        elif flag == "stop_hunt_risk_short":
            item.warnings.append("Liquidity/liquidation magnet is close above; short stop should not sit at an obvious minor high.")


def _build_external_strategy_context(item: ExternalSymbolMetrics, report: SymbolReport) -> dict[str, Any]:
    values = item.values
    long_score = 50.0
    short_score = 50.0
    notes: list[str] = []
    risk_flags: list[str] = []
    levels: dict[str, Any] = {}
    evidence_count = 0

    def add(direction: str, points: float, note: str) -> None:
        nonlocal long_score, short_score, evidence_count
        if direction == "long":
            long_score += points
        elif direction == "short":
            short_score += points
        evidence_count += 1
        notes.append(note)

    def flag(name: str) -> None:
        if name not in risk_flags:
            risk_flags.append(name)

    public = values.get("exchange_public_derivatives")
    if isinstance(public, dict):
        _apply_funding_to_context(public.get("funding_rate"), "public funding", add, flag)
        oi_change = _as_float(public.get("open_interest_change_pct"))
        orderbook = public.get("orderbook") if isinstance(public.get("orderbook"), dict) else {}
        if orderbook:
            imbalance = _as_float(orderbook.get("depth_imbalance"))
            if imbalance is not None:
                if imbalance >= 0.18:
                    add("long", min(5.0, 2.0 + imbalance * 8.0), f"orderbook bid depth imbalance {imbalance:.2f}")
                elif imbalance <= -0.18:
                    add("short", min(5.0, 2.0 + abs(imbalance) * 8.0), f"orderbook ask depth imbalance {imbalance:.2f}")
            _merge_level(levels, "support_wall", orderbook.get("bid_wall_price"), orderbook.get("bid_wall_distance_pct"), "exchange orderbook")
            _merge_level(levels, "resistance_wall", orderbook.get("ask_wall_price"), orderbook.get("ask_wall_distance_pct"), "exchange orderbook")
        flow = public.get("trade_flow") if isinstance(public.get("trade_flow"), dict) else {}
        if flow:
            _apply_flow_to_context(flow, "public taker flow", add)
        taker_ratio = public.get("taker_long_short_ratio")
        if isinstance(taker_ratio, dict):
            ratio_value = _as_float(taker_ratio.get("ratio"))
            if ratio_value is not None and ratio_value >= 0:
                _apply_flow_to_context(
                    {"taker_buy_ratio": ratio_value / (1.0 + ratio_value)},
                    "Binance taker buy/sell ratio",
                    add,
                )
        for key, label in (
            ("global_long_short_ratio", "Binance global long/short"),
            ("top_long_short_position_ratio", "Binance top-trader positioning"),
        ):
            ratio = public.get(key)
            if isinstance(ratio, dict):
                _apply_ratio_to_context(ratio, label, add, flag)
        if oi_change is not None and abs(oi_change) >= 8.0:
            flow_ratio = _flow_ratio(flow) or 0.5
            if oi_change > 0 and flow_ratio >= 0.56 and "long_crowded" not in risk_flags:
                add("long", min(3.0, oi_change / 8.0), f"OI +{oi_change:.1f}% with buy-side flow")
            elif oi_change > 0 and flow_ratio <= 0.44 and "short_crowded" not in risk_flags:
                add("short", min(3.0, oi_change / 8.0), f"OI +{oi_change:.1f}% with sell-side flow")
            elif abs(oi_change) >= 18.0:
                notes.append(f"OI change {oi_change:.1f}% is hot; reduce late-chase confidence")
                flag("derivatives_hot")

    coinalyze = values.get("coinalyze")
    if isinstance(coinalyze, dict):
        funding = _as_float(coinalyze.get("predicted_funding_rate"))
        if funding is None:
            funding = _as_float(coinalyze.get("funding_rate"))
        _apply_funding_to_context(funding, "Coinalyze funding", add, flag)
        if isinstance(coinalyze.get("long_short_ratio"), dict):
            _apply_ratio_to_context(coinalyze["long_short_ratio"], "Coinalyze long/short", add, flag)
        if isinstance(coinalyze.get("buy_sell_volume"), dict):
            _apply_flow_to_context(coinalyze["buy_sell_volume"], "Coinalyze buy/sell volume", add)
        oi_change = _as_float(coinalyze.get("open_interest_change_pct"))
        if oi_change is not None and abs(oi_change) >= 10:
            notes.append(f"Coinalyze OI 6h change {oi_change:.1f}%")
            evidence_count += 1
        liq_sum = coinalyze.get("liquidation_sum")
        if isinstance(liq_sum, dict):
            _apply_liquidation_to_context(liq_sum, "Coinalyze liquidation", add)

    cg_funding = _as_float(values.get("coinglass_funding"))
    _apply_funding_to_context(cg_funding, "CoinGlass funding", add, flag)
    if isinstance(values.get("coinglass_taker_buy_sell"), dict):
        _apply_flow_to_context(values["coinglass_taker_buy_sell"], "CoinGlass taker buy/sell", add)
    if isinstance(values.get("coinglass_long_short_ratio"), dict):
        _apply_ratio_to_context(values["coinglass_long_short_ratio"], "CoinGlass long/short", add, flag)
    if isinstance(values.get("coinglass_liquidation_sum"), dict):
        _apply_liquidation_to_context(values["coinglass_liquidation_sum"], "CoinGlass liquidation", add)
    for key, label in (
        ("coinglass_liquidation_heatmap", "CoinGlass liquidation heatmap"),
        ("coinglass_liquidation_map", "CoinGlass liquidation map"),
        ("coinglass_orderbook_heatmap", "CoinGlass orderbook heatmap"),
    ):
        summary = values.get(key)
        if isinstance(summary, dict):
            _apply_heatmap_to_context(summary, label, report.price, levels, add, flag)

    long_score = max(0.0, min(100.0, long_score))
    short_score = max(0.0, min(100.0, short_score))
    delta = long_score - short_score
    if delta >= 6:
        bias = "long"
    elif delta <= -6:
        bias = "short"
    else:
        bias = "neutral"
    confidence = max(0.0, min(92.0, 50.0 + abs(delta) * 2.5 + min(10, evidence_count)))
    if "liquidation_below" in levels:
        below_dist = _as_float(levels["liquidation_below"].get("distance_pct"))
        if below_dist is not None and below_dist <= 0.9:
            flag("stop_hunt_risk_long")
    if "liquidation_above" in levels:
        above_dist = _as_float(levels["liquidation_above"].get("distance_pct"))
        if above_dist is not None and above_dist <= 0.9:
            flag("stop_hunt_risk_short")
    top_notes = notes[:6]
    return {
        "bias": bias,
        "confidence": round(confidence, 2),
        "long_score": round(long_score, 2),
        "short_score": round(short_score, 2),
        "delta": round(delta, 2),
        "evidence_count": evidence_count,
        "notes": top_notes,
        "risk_flags": risk_flags,
        "levels": levels,
        "summary": "; ".join(top_notes) if top_notes else "external data readable",
    }


def _apply_funding_to_context(funding_value: Any, source: str, add: Any, flag: Any) -> None:
    funding = _as_float(funding_value)
    if funding is None:
        return
    if funding >= 0.00035:
        add("short", min(5.0, 2.0 + funding / 0.00035), f"{source} positive/crowded {funding:.5f}")
        flag("long_crowded")
    elif funding >= 0.00015:
        add("short", 1.4, f"{source} mildly positive {funding:.5f}")
    elif funding <= -0.00035:
        add("long", min(5.0, 2.0 + abs(funding) / 0.00035), f"{source} negative/crowded {funding:.5f}")
        flag("short_crowded")
    elif funding <= -0.00015:
        add("long", 1.4, f"{source} mildly negative {funding:.5f}")


def _apply_flow_to_context(flow: dict[str, Any], source: str, add: Any) -> None:
    ratio = _flow_ratio(flow)
    if ratio is None:
        return
    if ratio >= 0.62:
        add("long", min(5.0, 2.0 + (ratio - 0.62) * 18.0), f"{source} buy ratio {ratio:.2f}")
    elif ratio >= 0.57:
        add("long", 1.6, f"{source} buy ratio {ratio:.2f}")
    elif ratio <= 0.38:
        add("short", min(5.0, 2.0 + (0.38 - ratio) * 18.0), f"{source} sell ratio {ratio:.2f}")
    elif ratio <= 0.43:
        add("short", 1.6, f"{source} sell ratio {ratio:.2f}")


def _apply_ratio_to_context(ratio_payload: dict[str, Any], source: str, add: Any, flag: Any) -> None:
    ratio = _as_float(ratio_payload.get("ratio"))
    if ratio is None:
        return
    if ratio >= 1.8:
        add("short", min(4.0, 1.6 + (ratio - 1.8)), f"{source} long crowding ratio {ratio:.2f}")
        flag("long_crowded")
    elif ratio >= 1.35:
        add("short", 1.0, f"{source} longs leaning {ratio:.2f}")
    elif ratio <= 0.55:
        add("long", min(4.0, 1.6 + (0.55 - ratio) * 2.0), f"{source} short crowding ratio {ratio:.2f}")
        flag("short_crowded")
    elif ratio <= 0.75:
        add("long", 1.0, f"{source} shorts leaning {ratio:.2f}")


def _apply_liquidation_to_context(liq_sum: dict[str, Any], source: str, add: Any) -> None:
    long_liq = _as_float(liq_sum.get("long")) or 0.0
    short_liq = _as_float(liq_sum.get("short")) or 0.0
    total = long_liq + short_liq
    if total <= 0:
        return
    long_share = long_liq / total
    short_share = short_liq / total
    if long_share >= 0.65:
        add("long", min(3.0, 1.2 + (long_share - 0.65) * 5.0), f"{source} long flush share {long_share:.2f}")
    elif short_share >= 0.65:
        add("short", min(3.0, 1.2 + (short_share - 0.65) * 5.0), f"{source} short flush share {short_share:.2f}")


def _apply_heatmap_to_context(
    summary: dict[str, Any],
    source: str,
    price: float,
    levels: dict[str, Any],
    add: Any,
    flag: Any,
) -> None:
    above = _as_float(summary.get("nearest_above_price"))
    above_dist = _as_float(summary.get("nearest_above_distance_pct"))
    below = _as_float(summary.get("nearest_below_price"))
    below_dist = _as_float(summary.get("nearest_below_distance_pct"))
    if above is not None:
        _merge_level(levels, "liquidation_above", above, above_dist, source)
    if below is not None:
        _merge_level(levels, "liquidation_below", below, below_dist, source)
    if above is not None and above_dist is not None and 0.15 <= above_dist <= 1.8:
        add("long", max(0.8, 2.4 - above_dist * 0.55), f"{source} liquidity magnet above {above_dist:.2f}%")
    if below is not None and below_dist is not None and 0.15 <= below_dist <= 1.8:
        add("short", max(0.8, 2.4 - below_dist * 0.55), f"{source} liquidity magnet below {below_dist:.2f}%")
    if above_dist is not None and above_dist <= 0.9:
        flag("stop_hunt_risk_short")
    if below_dist is not None and below_dist <= 0.9:
        flag("stop_hunt_risk_long")


def _flow_ratio(flow: dict[str, Any]) -> float | None:
    ratio = _as_float(flow.get("taker_buy_ratio"))
    if ratio is not None:
        return ratio
    buy = _first_float(flow, ("taker_buy_notional", "buy", "buy_volume"))
    sell = _first_float(flow, ("taker_sell_notional", "sell", "sell_volume"))
    if buy is None or sell is None or buy + sell <= 0:
        return None
    return buy / (buy + sell)


def _first_float(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _merge_level(levels: dict[str, Any], name: str, price: Any, distance_pct: Any, source: str) -> None:
    price_f = _as_float(price)
    if price_f is None or price_f <= 0:
        return
    distance_f = _as_float(distance_pct)
    current = levels.get(name)
    if isinstance(current, dict):
        current_distance = _as_float(current.get("distance_pct"))
        if current_distance is not None and distance_f is not None and current_distance <= distance_f:
            return
    levels[name] = {"price": price_f, "distance_pct": distance_f, "source": source}


def _adjust_trade_plan_with_external_context(report: SymbolReport, item: ExternalSymbolMetrics) -> None:
    context = item.values.get("external_strategy_context")
    if not isinstance(context, dict):
        return
    levels = context.get("levels")
    if not isinstance(levels, dict) or not levels:
        return
    for side in (report.long, report.short):
        _adjust_side_trade_plan(side, levels)


def _adjust_side_trade_plan(side: DirectionScore, levels: dict[str, Any]) -> None:
    if not side.entry_zone or side.stop is None or side.target is None:
        return
    entry = (side.entry_zone[0] + side.entry_zone[1]) / 2.0
    if entry <= 0:
        return
    direction = side.direction
    old_stop = float(side.stop)
    old_target = float(side.target)
    old_risk = _side_risk(direction, entry, old_stop)
    if old_risk <= 0:
        return
    atr_pct = _as_float(side.market_metrics.get("atr_pct")) or 0.0
    buffer = entry * max(0.0008, min(0.0035, atr_pct / 100.0 * 0.14 if atr_pct else 0.0012))
    max_risk = max(old_risk * 1.25, entry * max(0.0075, min(0.035, (atr_pct / 100.0) * 1.05 if atr_pct else 0.012)))
    new_stop = old_stop
    stop_level = _best_external_stop_level(direction, entry, levels)
    if stop_level is not None:
        candidate = stop_level - buffer if direction == "long" else stop_level + buffer
        candidate_risk = _side_risk(direction, entry, candidate)
        if candidate_risk > old_risk * 1.03 and candidate_risk <= max_risk:
            new_stop = candidate
    target_level = _best_external_target_level(direction, entry, new_stop, old_target, levels)
    new_target = target_level if target_level is not None else old_target
    if new_stop == old_stop and new_target == old_target:
        return
    side.stop = new_stop
    side.target = new_target
    side.take_profits = _external_take_profits(direction, entry, new_stop, new_target)
    side.rr = _side_reward(direction, entry, new_target) / max(_side_risk(direction, entry, new_stop), 1e-12)
    side.signal_notes.append("External liquidity/orderflow levels adjusted SL/TP within intraday risk caps.")


def _best_external_stop_level(direction: str, entry: float, levels: dict[str, Any]) -> float | None:
    names = ("support_wall", "liquidation_below") if direction == "long" else ("resistance_wall", "liquidation_above")
    candidates: list[tuple[float, float]] = []
    for name in names:
        raw = levels.get(name)
        if not isinstance(raw, dict):
            continue
        price = _as_float(raw.get("price"))
        distance = _as_float(raw.get("distance_pct"))
        if price is None or distance is None:
            continue
        if direction == "long" and price < entry and distance <= 2.8:
            candidates.append((price, distance))
        elif direction == "short" and price > entry and distance <= 2.8:
            candidates.append((price, distance))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])[0]


def _best_external_target_level(direction: str, entry: float, stop: float, current_target: float, levels: dict[str, Any]) -> float | None:
    risk = _side_risk(direction, entry, stop)
    if risk <= 0:
        return None
    names = ("resistance_wall", "liquidation_above") if direction == "long" else ("support_wall", "liquidation_below")
    candidates: list[tuple[float, float]] = []
    for name in names:
        raw = levels.get(name)
        if not isinstance(raw, dict):
            continue
        price = _as_float(raw.get("price"))
        if price is None:
            continue
        front_run = price - risk * 0.08 if direction == "long" else price + risk * 0.08
        rr = _side_reward(direction, entry, front_run) / risk
        if 1.25 <= rr <= 3.6:
            candidates.append((front_run, rr))
    if not candidates:
        return None
    current_rr = _side_reward(direction, entry, current_target) / risk
    candidates.sort(key=lambda item: (abs(item[1] - max(1.55, min(2.4, current_rr))), item[1]))
    return candidates[0][0]


def _external_take_profits(direction: str, entry: float, stop: float, target: float) -> list[dict[str, float | str]]:
    risk = max(_side_risk(direction, entry, stop), abs(entry) * 0.001)
    target_rr = max(1.25, _side_reward(direction, entry, target) / risk)
    tp1_rr = max(0.85, min(1.25, target_rr * 0.55))
    tp3_rr = min(3.8, max(target_rr + 0.55, 2.15))
    return [
        {
            "name": "TP1",
            "price": _rr_level(direction, entry, risk, tp1_rr),
            "rr": tp1_rr,
            "portion_pct": 30.0,
            "note": "first external/liquidity reaction area",
        },
        {
            "name": "TP2",
            "price": target,
            "rr": target_rr,
            "portion_pct": 45.0,
            "note": "main target front-runs external wall/liquidation magnet",
        },
        {
            "name": "TP3",
            "price": _rr_level(direction, entry, risk, tp3_rr),
            "rr": tp3_rr,
            "portion_pct": 25.0,
            "note": "extension only if flow keeps confirming",
        },
    ]


def _side_risk(direction: str, entry: float, stop: float) -> float:
    return max(entry - stop, 0.0) if direction == "long" else max(stop - entry, 0.0)


def _side_reward(direction: str, entry: float, target: float) -> float:
    return max(target - entry, 0.0) if direction == "long" else max(entry - target, 0.0)


def _rr_level(direction: str, entry: float, risk: float, rr: float) -> float:
    return entry + risk * rr if direction == "long" else entry - risk * rr


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


def _payload_has_data(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).endswith("_error"):
                continue
            if value not in (None, {}, []):
                return True
        return False
    return payload not in (None, {}, [])


def _payload_error_summary(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    errors = [str(value) for key, value in payload.items() if str(key).endswith("_error") and value]
    return " | ".join(errors[:3])


def _sum_exact_fields(value: Any, names: tuple[str, ...]) -> float:
    wanted = {_normalize_field_name(name) for name in names}
    total = 0.0
    if isinstance(value, dict):
        for key, item in value.items():
            key_name = _normalize_field_name(str(key))
            if key_name in wanted:
                number = _last_number(item)
                if number is not None:
                    total += abs(number)
            else:
                total += _sum_exact_fields(item, names)
    elif isinstance(value, list):
        for item in value:
            total += _sum_exact_fields(item, names)
    return total


def _normalize_field_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _ingest_coinglass_values(
    item: ExternalSymbolMetrics,
    values: dict[str, Any],
    report: SymbolReport | None,
) -> None:
    item.providers.append("CoinGlass")
    raw_summary = {
        key: value
        for key, value in values.items()
        if key not in {"liquidation_heatmap", "liquidation_map", "orderbook_heatmap"}
    }
    item.values["coinglass"] = _compact_payload(raw_summary)
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
    buy = _sum_exact_fields(values.get("taker_buy_sell"), ("buy", "buyvolume", "buyvol", "takerbuy", "takerbuyvolume"))
    sell = _sum_exact_fields(values.get("taker_buy_sell"), ("sell", "sellvolume", "sellvol", "takersell", "takersellvolume"))
    flow = _trade_flow_metrics(buy, sell, 0)
    if flow:
        item.values["coinglass_taker_buy_sell"] = flow
    ratio = _latest_ratio(values.get("long_short_ratio"))
    if ratio:
        item.values["coinglass_long_short_ratio"] = ratio
    if report is None:
        return
    for raw_key, out_key in (
        ("liquidation_heatmap", "coinglass_liquidation_heatmap"),
        ("liquidation_map", "coinglass_liquidation_map"),
        ("orderbook_heatmap", "coinglass_orderbook_heatmap"),
    ):
        if raw_key not in values:
            continue
        summary = _heatmap_level_summary(values.get(raw_key), report.price)
        if summary.get("level_count"):
            item.values[out_key] = summary


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
