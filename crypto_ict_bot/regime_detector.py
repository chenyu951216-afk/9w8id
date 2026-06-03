from __future__ import annotations

from statistics import median
from typing import Any

from .models import SymbolReport


def detect_market_regime(reports: list[SymbolReport]) -> dict[str, Any]:
    if not reports:
        return _empty_regime()

    changes = [float(report.change_pct_24h or 0.0) for report in reports]
    volumes = [float(report.quote_volume_24h or 0.0) for report in reports]
    positive = sum(1 for value in changes if value > 0)
    advance_ratio = positive / max(len(changes), 1)
    median_change = median(changes)
    btc_change = _symbol_change(reports, "BTCUSDT")
    eth_change = _symbol_change(reports, "ETHUSDT")
    top_liquidity = sorted(volumes, reverse=True)[: max(3, min(12, len(volumes)))]
    median_volume = median(volumes)
    top_volume_median = median(top_liquidity) if top_liquidity else median_volume
    atr_values = _atr_values(reports)
    median_atr = median(atr_values) if atr_values else None

    volatility = _score_volatility(median_atr)
    liquidity = _score_liquidity(median_volume, top_volume_median)
    risk_appetite = _score_risk_appetite(median_change, advance_ratio, btc_change, eth_change)
    btc_alignment = _score_btc_alignment(btc_change, median_change, advance_ratio)
    regime, confidence, notes = _classify_regime(
        median_change=median_change,
        advance_ratio=advance_ratio,
        btc_change=btc_change,
        eth_change=eth_change,
        volatility=volatility,
        liquidity=liquidity,
        risk_appetite=risk_appetite,
    )

    return {
        "regime": regime,
        "confidence": round(confidence, 1),
        "btc_alignment": round(btc_alignment, 1),
        "volatility_percentile": round(volatility, 1),
        "liquidity_condition": round(liquidity, 1),
        "risk_appetite": round(risk_appetite, 1),
        "advance_decline_ratio": round(advance_ratio, 4),
        "median_change_pct_24h": round(median_change, 4),
        "btc_change_pct_24h": btc_change,
        "eth_change_pct_24h": eth_change,
        "median_atr_pct": round(median_atr, 4) if median_atr is not None else None,
        "notes": notes,
        "weight_profile": _weight_profile(regime, volatility, liquidity),
    }


def regime_alignment_score(regime: dict[str, Any], direction: str, side_metrics: dict[str, Any]) -> float:
    base = 55.0
    name = str(regime.get("regime") or "range")
    btc_trend = _as_float(side_metrics.get("btc_trend_pct"))
    btc_against = bool(side_metrics.get("btc_against"))
    if name == "trend_up":
        base = 72.0 if direction == "long" else 42.0
    elif name == "trend_down":
        base = 72.0 if direction == "short" else 42.0
    elif name == "alt_rotation":
        base = 66.0
    elif name == "risk_off":
        base = 62.0 if direction == "short" else 38.0
    elif name == "high_volatility":
        base = 52.0
    elif name == "low_liquidity":
        base = 45.0
    elif name == "squeeze":
        base = 58.0
    if btc_trend is not None:
        aligned = (direction == "long" and btc_trend >= 0) or (direction == "short" and btc_trend <= 0)
        base += 8.0 if aligned else -8.0
    if btc_against:
        base -= 14.0
    base += (float(regime.get("risk_appetite") or 50.0) - 50.0) * 0.18
    base += (float(regime.get("liquidity_condition") or 50.0) - 50.0) * 0.08
    return round(_clamp(base), 2)


def _classify_regime(
    *,
    median_change: float,
    advance_ratio: float,
    btc_change: float | None,
    eth_change: float | None,
    volatility: float,
    liquidity: float,
    risk_appetite: float,
) -> tuple[str, float, list[str]]:
    notes: list[str] = []
    btc = btc_change if btc_change is not None else median_change
    eth = eth_change if eth_change is not None else median_change
    if liquidity < 38:
        notes.append("全市場流動性偏低，假突破與滑價風險提高。")
        return "low_liquidity", 72.0, notes
    if volatility >= 78:
        notes.append("ATR/波動偏高，entry 距離與滑價需要更嚴格管理。")
        if risk_appetite < 42:
            notes.append("波動偏高且風險偏好弱，優先保守觀察。")
            return "risk_off", 76.0, notes
        return "high_volatility", 72.0, notes
    if btc <= -1.2 and advance_ratio < 0.42:
        notes.append("BTC 與多數幣同步走弱，偏 risk-off。")
        return "risk_off", 78.0, notes
    if btc >= 1.0 and advance_ratio >= 0.58:
        notes.append("BTC 與廣泛市場同步走強，偏多頭趨勢環境。")
        return "trend_up", 76.0, notes
    if btc <= -1.0 and advance_ratio <= 0.45:
        notes.append("BTC 趨勢偏下，空方 setup 權重提高。")
        return "trend_down", 74.0, notes
    if advance_ratio >= 0.56 and median_change > 0 and abs(btc) < 1.2:
        notes.append("市場中位數轉強且 BTC 未吸血，偏 alt rotation。")
        return "alt_rotation", 70.0, notes
    if volatility <= 35 and abs(median_change) < 0.5:
        notes.append("波動收斂，等待突破方向，不宜追價。")
        return "squeeze", 66.0, notes
    notes.append("市場偏震盪，排名更重視 HTF POI、流動性掃描與入場位置。")
    if eth > btc + 0.8 and advance_ratio >= 0.52:
        notes.append("ETH 相對 BTC 偏強，山寨 beta 有回溫跡象。")
    return "range", 62.0, notes


def _weight_profile(regime: str, volatility: float, liquidity: float) -> dict[str, float]:
    weights = {
        "regime_alignment": 0.18,
        "direction_conviction": 0.22,
        "setup_quality": 0.22,
        "location_quality": 0.14,
        "risk_reward_quality": 0.12,
        "relative_strength": 0.07,
        "data_quality": 0.05,
    }
    if regime in {"trend_up", "trend_down", "alt_rotation"}:
        weights["direction_conviction"] += 0.03
        weights["relative_strength"] += 0.03
        weights["setup_quality"] -= 0.03
        weights["location_quality"] -= 0.03
    elif regime in {"range", "squeeze"}:
        weights["setup_quality"] += 0.02
        weights["location_quality"] += 0.03
        weights["relative_strength"] -= 0.02
        weights["regime_alignment"] -= 0.03
    if volatility >= 70:
        weights["location_quality"] += 0.03
        weights["risk_reward_quality"] += 0.02
        weights["relative_strength"] -= 0.02
        weights["direction_conviction"] -= 0.03
    if liquidity < 45:
        weights["data_quality"] += 0.03
        weights["location_quality"] += 0.02
        weights["relative_strength"] -= 0.02
        weights["regime_alignment"] -= 0.03
    total = sum(weights.values())
    return {key: round(value / total, 4) for key, value in weights.items()}


def _score_volatility(median_atr: float | None) -> float:
    if median_atr is None:
        return 50.0
    return _clamp((median_atr - 0.15) / 3.0 * 100.0)


def _score_liquidity(median_volume: float, top_volume_median: float) -> float:
    base = min(100.0, median_volume / 60_000_000 * 72.0)
    top_bonus = min(28.0, top_volume_median / 350_000_000 * 28.0)
    return _clamp(base + top_bonus)


def _score_risk_appetite(median_change: float, advance_ratio: float, btc_change: float | None, eth_change: float | None) -> float:
    score = 50.0 + median_change * 5.0 + (advance_ratio - 0.5) * 65.0
    if btc_change is not None:
        score += btc_change * 2.2
    if eth_change is not None and btc_change is not None and eth_change > btc_change:
        score += min(8.0, (eth_change - btc_change) * 2.0)
    return _clamp(score)


def _score_btc_alignment(btc_change: float | None, median_change: float, advance_ratio: float) -> float:
    if btc_change is None:
        return 50.0
    same_sign = (btc_change >= 0 and median_change >= 0) or (btc_change <= 0 and median_change <= 0)
    score = 50.0 + (18.0 if same_sign else -18.0) + (advance_ratio - 0.5) * 35.0
    score -= min(18.0, abs(btc_change - median_change) * 2.5)
    return _clamp(score)


def _symbol_change(reports: list[SymbolReport], symbol: str) -> float | None:
    for report in reports:
        if report.symbol.upper() == symbol:
            return float(report.change_pct_24h or 0.0)
    return None


def _atr_values(reports: list[SymbolReport]) -> list[float]:
    values: list[float] = []
    for report in reports:
        for side in (report.long, report.short):
            atr_pct = side.market_metrics.get("atr_pct")
            if isinstance(atr_pct, (int, float)):
                values.append(float(atr_pct))
    return values


def _empty_regime() -> dict[str, Any]:
    return {
        "regime": "unknown",
        "confidence": 0.0,
        "btc_alignment": 0.0,
        "volatility_percentile": 0.0,
        "liquidity_condition": 0.0,
        "risk_appetite": 0.0,
        "notes": ["尚無足夠資料判斷市場 regime。"],
        "weight_profile": _weight_profile("range", 50.0, 50.0),
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
