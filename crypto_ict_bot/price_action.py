from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Candle
from .technicals import atr, swing_points


@dataclass(frozen=True)
class PriceActionContext:
    key_level_score: float
    price_action_score: float
    breakout_score: float
    notes: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


def analyze_price_action(
    direction: str,
    price: float,
    candles_4h: list[Candle],
    candles_1h: list[Candle],
    candles_15m: list[Candle],
    candles_5m: list[Candle],
) -> PriceActionContext:
    htf_candles = candles_4h or candles_1h
    ltf_candles = candles_15m or candles_5m or candles_1h
    key_score, key_notes, key_warnings, key_metrics = _key_level_context(direction, price, htf_candles, ltf_candles)
    candle_score, candle_notes, candle_warnings, candle_metrics = _candle_confirmation(direction, ltf_candles)
    breakout_score, breakout_notes, breakout_warnings, breakout_metrics = _breakout_quality(direction, ltf_candles)
    metrics = {
        **key_metrics,
        **candle_metrics,
        **breakout_metrics,
    }
    notes = [*key_notes, *candle_notes, *breakout_notes]
    warnings = [*key_warnings, *candle_warnings, *breakout_warnings]
    return PriceActionContext(
        key_level_score=round(_clamp(key_score), 2),
        price_action_score=round(_clamp(candle_score), 2),
        breakout_score=round(_clamp(breakout_score), 2),
        notes=notes,
        warnings=warnings,
        metrics=metrics,
    )


def _key_level_context(
    direction: str,
    price: float,
    htf_candles: list[Candle],
    ltf_candles: list[Candle],
) -> tuple[float, list[str], list[str], dict[str, Any]]:
    if len(htf_candles) < 30:
        return 0.0, [], ["HTF 歷史不足，無法可靠定位支撐/壓力"], {}
    sample = htf_candles[-160:]
    local_atr = atr(sample, 14) or price * 0.004
    swings = swing_points(sample, left=2, right=2)
    highs = [s.price for s in swings if s.kind == "high"]
    lows = [s.price for s in swings if s.kind == "low"]
    resistance = min((level for level in highs if level > price), default=None)
    support = max((level for level in lows if level < price), default=None)
    ema_fast = _ema([c.close for c in sample], 21)
    ema_slow = _ema([c.close for c in sample], 55)
    trend = _trend_from_swings(swings)
    notes: list[str] = []
    warnings: list[str] = []
    score = 35.0

    if direction == "long":
        if support is not None:
            support_atr = abs(price - support) / max(local_atr, 1e-12)
            if support_atr <= 0.8:
                score += 22.0
                notes.append(f"現價貼近 HTF 支撐 {support:g}，距離 {support_atr:.2f} ATR")
            elif support_atr <= 1.8:
                score += 12.0
                notes.append(f"現價接近 HTF 支撐 {support:g}")
        if resistance is not None:
            room_atr = abs(resistance - price) / max(local_atr, 1e-12)
            if room_atr >= 1.4:
                score += 18.0
                notes.append(f"上方到壓力 {resistance:g} 仍有 {room_atr:.2f} ATR 空間")
            else:
                score -= 18.0
                warnings.append(f"現價過近上方壓力 {resistance:g}，只有 {room_atr:.2f} ATR 空間")
        if ema_fast is not None and ema_slow is not None:
            if price >= ema_fast >= ema_slow:
                score += 12.0
                notes.append("價格站上 21/55 EMA，多方結構順")
            elif price < ema_fast < ema_slow:
                score -= 14.0
                warnings.append("價格低於 21/55 EMA，多單屬逆勢")
        if trend == "up":
            score += 13.0
            notes.append("HTF swing 結構偏多")
        elif trend == "down":
            score -= 15.0
            warnings.append("HTF swing 結構偏空，做多需等待明確反轉")
    else:
        if resistance is not None:
            resistance_atr = abs(resistance - price) / max(local_atr, 1e-12)
            if resistance_atr <= 0.8:
                score += 22.0
                notes.append(f"現價貼近 HTF 壓力 {resistance:g}，距離 {resistance_atr:.2f} ATR")
            elif resistance_atr <= 1.8:
                score += 12.0
                notes.append(f"現價接近 HTF 壓力 {resistance:g}")
        if support is not None:
            room_atr = abs(price - support) / max(local_atr, 1e-12)
            if room_atr >= 1.4:
                score += 18.0
                notes.append(f"下方到支撐 {support:g} 仍有 {room_atr:.2f} ATR 空間")
            else:
                score -= 18.0
                warnings.append(f"現價過近下方支撐 {support:g}，只有 {room_atr:.2f} ATR 空間")
        if ema_fast is not None and ema_slow is not None:
            if price <= ema_fast <= ema_slow:
                score += 12.0
                notes.append("價格跌破 21/55 EMA，空方結構順")
            elif price > ema_fast > ema_slow:
                score -= 14.0
                warnings.append("價格高於 21/55 EMA，空單屬逆勢")
        if trend == "down":
            score += 13.0
            notes.append("HTF swing 結構偏空")
        elif trend == "up":
            score -= 15.0
            warnings.append("HTF swing 結構偏多，做空需等待明確反轉")

    metrics = {
        "nearest_support": support,
        "nearest_resistance": resistance,
        "key_level_atr": round(local_atr, 8),
        "htf_swing_trend": trend,
        "ema21": ema_fast,
        "ema55": ema_slow,
    }
    return score, notes, warnings, metrics


def _candle_confirmation(direction: str, candles: list[Candle]) -> tuple[float, list[str], list[str], dict[str, Any]]:
    if len(candles) < 6:
        return 0.0, [], ["短線 K 線不足，無法判斷收盤確認"], {}
    latest = candles[-1]
    previous = candles[-2]
    local_atr = atr(candles[-80:], 14) or latest.close * 0.004
    body_atr = latest.body / max(local_atr, 1e-12)
    volume_ratio = _volume_ratio(candles)
    close_location = (latest.close - latest.low) / max(latest.high - latest.low, 1e-12)
    if direction == "short":
        close_location = (latest.high - latest.close) / max(latest.high - latest.low, 1e-12)
    notes: list[str] = []
    warnings: list[str] = []
    score = 25.0

    directional = latest.close > latest.open if direction == "long" else latest.close < latest.open
    if directional and close_location >= 0.64:
        score += 22.0
        notes.append(f"最近 K 線順向收盤在實體有利端，close_location={close_location:.2f}")
    elif close_location < 0.45:
        score -= 14.0
        warnings.append(f"最近 K 線收盤位置不佳，close_location={close_location:.2f}")

    if _engulfing(direction, latest, previous):
        score += 18.0
        notes.append("出現順向 engulfing / 反包確認")
    if _rejection_wick(direction, latest):
        score += 16.0
        notes.append("出現順向拒絕影線")
    if body_atr >= 0.75 and directional:
        score += 12.0
        notes.append(f"順向 K 線 body/ATR={body_atr:.2f}")
    if volume_ratio >= 1.15 and directional:
        score += 12.0
        notes.append(f"成交量較基準放大 {volume_ratio:.2f} 倍")
    elif volume_ratio < 0.75:
        score -= 10.0
        warnings.append(f"成交量低於基準，volume_ratio={volume_ratio:.2f}")

    return score, notes, warnings, {
        "candle_body_atr": round(body_atr, 4),
        "candle_close_location": round(close_location, 4),
        "candle_volume_ratio": round(volume_ratio, 4),
    }


def _breakout_quality(direction: str, candles: list[Candle]) -> tuple[float, list[str], list[str], dict[str, Any]]:
    if len(candles) < 35:
        return 0.0, [], ["短線歷史不足，無法判斷突破真假"], {}
    latest = candles[-1]
    local_atr = atr(candles[-90:], 14) or latest.close * 0.004
    prior = candles[:-1]
    swings = swing_points(prior[-120:], left=2, right=2)
    volume_ratio = _volume_ratio(candles)
    score = 30.0
    notes: list[str] = []
    warnings: list[str] = []
    breakout_level: float | None = None
    false_breakout = False
    close_break = False

    if direction == "long":
        highs = [s.price for s in swings if s.kind == "high" and s.price < latest.close + local_atr * 2]
        breakout_level = max(highs) if highs else max(c.high for c in prior[-24:])
        close_break = latest.close > breakout_level
        wick_only = latest.high > breakout_level and latest.close <= breakout_level
        false_breakout = wick_only and latest.close < latest.open
    else:
        lows = [s.price for s in swings if s.kind == "low" and s.price > latest.close - local_atr * 2]
        breakout_level = min(lows) if lows else min(c.low for c in prior[-24:])
        close_break = latest.close < breakout_level
        wick_only = latest.low < breakout_level and latest.close >= breakout_level
        false_breakout = wick_only and latest.close > latest.open

    distance_atr = abs(latest.close - breakout_level) / max(local_atr, 1e-12) if breakout_level else 0.0
    if close_break:
        score += 24.0
        notes.append(f"收盤突破短線關鍵位 {breakout_level:g}，距離 {distance_atr:.2f} ATR")
        if volume_ratio >= 1.15:
            score += 18.0
            notes.append(f"突破伴隨量能確認 {volume_ratio:.2f} 倍")
        else:
            score -= 10.0
            warnings.append(f"突破量能不足，volume_ratio={volume_ratio:.2f}")
        if 0.08 <= distance_atr <= 1.3:
            score += 10.0
        elif distance_atr > 1.8:
            score -= 18.0
            warnings.append(f"突破後已遠離關鍵位 {distance_atr:.2f} ATR，追價風險高")
    elif false_breakout:
        score -= 28.0
        warnings.append(f"只刺破關鍵位 {breakout_level:g} 但未收過，疑似假突破")
    else:
        notes.append("尚未完成收盤突破，偏等待回測或下一根確認")

    return score, notes, warnings, {
        "breakout_level": breakout_level,
        "breakout_distance_atr": round(distance_atr, 4),
        "breakout_close_confirmed": close_break,
        "false_breakout_risk": false_breakout,
    }


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    value = sum(values[:period]) / period
    for item in values[period:]:
        value = item * alpha + value * (1.0 - alpha)
    return value


def _trend_from_swings(swings: list[Any]) -> str:
    highs = [s.price for s in swings if s.kind == "high"][-3:]
    lows = [s.price for s in swings if s.kind == "low"][-3:]
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            return "up"
        if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            return "down"
    return "range"


def _volume_ratio(candles: list[Candle]) -> float:
    if len(candles) < 25:
        return 1.0
    recent = sum(c.volume for c in candles[-3:]) / 3.0
    base = sum(c.volume for c in candles[-24:-3]) / max(len(candles[-24:-3]), 1)
    return recent / max(base, 1e-12)


def _engulfing(direction: str, latest: Candle, previous: Candle) -> bool:
    if direction == "long":
        return latest.close > latest.open and previous.close < previous.open and latest.close >= previous.open and latest.open <= previous.close
    return latest.close < latest.open and previous.close > previous.open and latest.close <= previous.open and latest.open >= previous.close


def _rejection_wick(direction: str, candle: Candle) -> bool:
    body = max(candle.body, candle.close * 0.0005)
    upper = candle.high - max(candle.open, candle.close)
    lower = min(candle.open, candle.close) - candle.low
    if direction == "long":
        return lower >= body * 1.15 and upper <= body * 1.25
    return upper >= body * 1.15 and lower <= body * 1.25


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
