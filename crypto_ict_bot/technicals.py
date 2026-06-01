from __future__ import annotations

import math
from datetime import datetime
from statistics import median
from zoneinfo import ZoneInfo

from .models import Candle, Displacement, FVG, StructureBreak, Sweep, Swing, Zone


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct_distance(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return abs(a - b) / abs(b) * 100.0


def true_ranges(candles: list[Candle]) -> list[float]:
    values: list[float] = []
    for idx, candle in enumerate(candles):
        if idx == 0:
            values.append(candle.high - candle.low)
            continue
        previous = candles[idx - 1].close
        values.append(max(candle.high - candle.low, abs(candle.high - previous), abs(candle.low - previous)))
    return values


def atr(candles: list[Candle], period: int = 14, end: int | None = None) -> float:
    if not candles:
        return 0.0
    end_index = len(candles) if end is None else max(1, min(len(candles), end + 1))
    ranges = true_ranges(candles[:end_index])
    if not ranges:
        return 0.0
    sample = ranges[-period:]
    return sum(sample) / len(sample)


def median_range(candles: list[Candle], period: int = 30) -> float:
    sample = [c.range for c in candles[-period:] if c.range > 0]
    return median(sample) if sample else 0.0


def swing_points(candles: list[Candle], left: int = 2, right: int = 2) -> list[Swing]:
    swings: list[Swing] = []
    if len(candles) < left + right + 1:
        return swings
    for idx in range(left, len(candles) - right):
        candle = candles[idx]
        left_slice = candles[idx - left : idx]
        right_slice = candles[idx + 1 : idx + right + 1]
        if all(candle.high > other.high for other in left_slice + right_slice):
            swings.append(Swing("high", idx, candle.open_time, candle.high))
        if all(candle.low < other.low for other in left_slice + right_slice):
            swings.append(Swing("low", idx, candle.open_time, candle.low))
    return swings


def last_swing(swings: list[Swing], kind: str, before_index: int | None = None) -> Swing | None:
    for swing in reversed(swings):
        if swing.kind != kind:
            continue
        if before_index is not None and swing.index >= before_index:
            continue
        return swing
    return None


def detect_fvgs(candles: list[Candle], max_age: int = 160, min_size_atr: float = 0.05) -> list[FVG]:
    if len(candles) < 3:
        return []
    start = max(2, len(candles) - max_age)
    results: list[FVG] = []
    latest_atr = atr(candles, 14) or median_range(candles, 30) or candles[-1].close * 0.001
    for idx in range(start, len(candles)):
        first = candles[idx - 2]
        third = candles[idx]
        if first.high < third.low:
            lower, upper = first.high, third.low
            if upper - lower >= latest_atr * min_size_atr:
                tapped = False
                filled = False
                for later in candles[idx + 1 :]:
                    if later.low <= lower:
                        filled = True
                        tapped = True
                        break
                    if later.low <= upper:
                        tapped = True
                results.append(
                    FVG(
                        direction="long",
                        index=idx,
                        start_time=third.open_time,
                        lower=lower,
                        upper=upper,
                        size_pct=(upper - lower) / max(third.close, 1e-12) * 100.0,
                        tapped=tapped,
                        filled=filled,
                    )
                )
        if first.low > third.high:
            lower, upper = third.high, first.low
            if upper - lower >= latest_atr * min_size_atr:
                tapped = False
                filled = False
                for later in candles[idx + 1 :]:
                    if later.high >= upper:
                        filled = True
                        tapped = True
                        break
                    if later.high >= lower:
                        tapped = True
                results.append(
                    FVG(
                        direction="short",
                        index=idx,
                        start_time=third.open_time,
                        lower=lower,
                        upper=upper,
                        size_pct=(upper - lower) / max(third.close, 1e-12) * 100.0,
                        tapped=tapped,
                        filled=filled,
                    )
                )
    return results


def recent_relevant_fvg(candles: list[Candle], direction: str, max_age: int = 90) -> FVG | None:
    price = candles[-1].close
    candidates = [gap for gap in detect_fvgs(candles, max_age=max_age) if gap.direction == direction and not gap.filled]
    if not candidates:
        return None
    if direction == "long":
        near = [gap for gap in candidates if gap.lower <= price <= gap.upper * 1.01 or price >= gap.lower]
    else:
        near = [gap for gap in candidates if gap.lower * 0.99 <= price <= gap.upper or price <= gap.upper]
    return (near or candidates)[-1]


def detect_liquidity_sweep(candles: list[Candle], direction: str, lookback: int = 70) -> Sweep | None:
    if len(candles) < 20:
        return None
    swings = swing_points(candles, left=2, right=2)
    latest_atr = atr(candles, 14) or median_range(candles, 30) or candles[-1].close * 0.001
    start = max(8, len(candles) - lookback)
    best: Sweep | None = None
    for idx in range(start, len(candles)):
        candle = candles[idx]
        if direction == "long":
            reference = last_swing(swings, "low", before_index=idx)
            if not reference:
                continue
            swept = candle.low < reference.price and candle.close > reference.price
            if swept:
                strength = clamp((reference.price - candle.low) / max(latest_atr, 1e-12), 0.05, 3.0)
                best = Sweep(direction, idx, candle.open_time, reference.price, candle.low, strength)
        else:
            reference = last_swing(swings, "high", before_index=idx)
            if not reference:
                continue
            swept = candle.high > reference.price and candle.close < reference.price
            if swept:
                strength = clamp((candle.high - reference.price) / max(latest_atr, 1e-12), 0.05, 3.0)
                best = Sweep(direction, idx, candle.open_time, reference.price, candle.high, strength)
    return best


def detect_structure_break(
    candles: list[Candle],
    direction: str,
    after_index: int | None = None,
    lookback: int = 90,
) -> StructureBreak | None:
    if len(candles) < 20:
        return None
    swings = swing_points(candles, left=2, right=2)
    start = max(6, len(candles) - lookback)
    if after_index is not None:
        start = max(start, after_index + 1)
    best: StructureBreak | None = None
    for idx in range(start, len(candles)):
        if direction == "long":
            reference = last_swing(swings, "high", before_index=idx)
            if reference and candles[idx].close > reference.price:
                kind = "MSS" if after_index is not None and idx > after_index else "BOS"
                best = StructureBreak(direction, idx, candles[idx].open_time, reference.price, candles[idx].close, kind)
        else:
            reference = last_swing(swings, "low", before_index=idx)
            if reference and candles[idx].close < reference.price:
                kind = "MSS" if after_index is not None and idx > after_index else "BOS"
                best = StructureBreak(direction, idx, candles[idx].open_time, reference.price, candles[idx].close, kind)
    return best


def detect_displacement(
    candles: list[Candle],
    direction: str,
    after_index: int | None = None,
    lookback: int = 45,
    body_atr_threshold: float = 1.15,
) -> Displacement | None:
    if len(candles) < 18:
        return None
    start = max(14, len(candles) - lookback)
    if after_index is not None:
        start = max(start, after_index)
    best: Displacement | None = None
    gaps = detect_fvgs(candles, max_age=lookback + 5)
    for idx in range(start, len(candles)):
        candle = candles[idx]
        if direction == "long" and candle.close <= candle.open:
            continue
        if direction == "short" and candle.close >= candle.open:
            continue
        local_atr = atr(candles, 14, idx) or median_range(candles[: idx + 1], 30) or candle.close * 0.001
        body_atr = candle.body / max(local_atr, 1e-12)
        if body_atr < body_atr_threshold:
            continue
        close_location = (candle.close - candle.low) / max(candle.high - candle.low, 1e-12)
        if direction == "short":
            close_location = (candle.high - candle.close) / max(candle.high - candle.low, 1e-12)
        if close_location < 0.6:
            continue
        has_fvg = any(gap.direction == direction and abs(gap.index - idx) <= 2 for gap in gaps)
        best = Displacement(direction, idx, candle.open_time, body_atr, close_location, has_fvg)
    return best


def price_position_in_range(candles: list[Candle], lookback: int = 120) -> tuple[float, float, float]:
    sample = candles[-lookback:] if len(candles) > lookback else candles
    low = min(c.low for c in sample)
    high = max(c.high for c in sample)
    if high <= low:
        return 0.5, low, high
    position = (candles[-1].close - low) / (high - low)
    return clamp(position, 0.0, 1.0), low, high


def ote_zone(candles: list[Candle], direction: str, lookback: int = 90) -> tuple[float, float, float] | None:
    sample = candles[-lookback:] if len(candles) > lookback else candles
    if len(sample) < 8:
        return None
    low = min(c.low for c in sample)
    high = max(c.high for c in sample)
    if high <= low:
        return None
    price = candles[-1].close
    if direction == "long":
        zone_low = high - (high - low) * 0.79
        zone_high = high - (high - low) * 0.62
        retracement = (high - price) / (high - low)
    else:
        zone_low = low + (high - low) * 0.62
        zone_high = low + (high - low) * 0.79
        retracement = (price - low) / (high - low)
    return zone_low, zone_high, retracement


def order_block(candles: list[Candle], direction: str, before_index: int | None = None, lookback: int = 50) -> Zone | None:
    if len(candles) < 5:
        return None
    end = len(candles) - 1 if before_index is None else min(before_index, len(candles) - 1)
    start = max(0, end - lookback)
    for idx in range(end, start, -1):
        candle = candles[idx]
        if direction == "long" and candle.close < candle.open:
            return Zone(direction, "bullish_order_block", candle.low, candle.high, idx, candle.open_time)
        if direction == "short" and candle.close > candle.open:
            return Zone(direction, "bearish_order_block", candle.low, candle.high, idx, candle.open_time)
    return None


def zone_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    return max(a_low, b_low) <= min(a_high, b_high)


def trendline_breakout(candles: list[Candle], direction: str) -> dict[str, float | int | bool | str]:
    if len(candles) < 60:
        return {"hit": False, "touches": 0, "risk": "na", "distance_atr": math.inf}
    swings = swing_points(candles, left=2, right=2)
    kind = "high" if direction == "long" else "low"
    relevant = [s for s in swings if s.kind == kind][-8:]
    if len(relevant) < 2:
        return {"hit": False, "touches": 0, "risk": "na", "distance_atr": math.inf}

    best: tuple[Swing, Swing] | None = None
    for first_idx in range(len(relevant) - 1):
        for second_idx in range(first_idx + 1, len(relevant)):
            a, b = relevant[first_idx], relevant[second_idx]
            days = (b.time - a.time).total_seconds() / 86400
            if days < 6.5:
                continue
            slope = (b.price - a.price) / max(b.index - a.index, 1)
            if direction == "long" and slope >= 0:
                continue
            if direction == "short" and slope <= 0:
                continue
            best = (a, b)
    if not best:
        return {"hit": False, "touches": 0, "risk": "na", "distance_atr": math.inf}

    a, b = best
    slope = (b.price - a.price) / max(b.index - a.index, 1)

    def line_value(index: int) -> float:
        return a.price + slope * (index - a.index)

    latest_idx = len(candles) - 1
    previous_idx = latest_idx - 1
    latest_line = line_value(latest_idx)
    previous_line = line_value(previous_idx)
    previous_close = candles[previous_idx].close
    latest_close = candles[latest_idx].close
    if direction == "long":
        hit = previous_close <= previous_line and latest_close > latest_line
        distance = latest_close - latest_line
    else:
        hit = previous_close >= previous_line and latest_close < latest_line
        distance = latest_line - latest_close
    latest_atr = atr(candles, 14) or median_range(candles, 30) or latest_close * 0.001
    distance_atr = distance / max(latest_atr, 1e-12)
    touches = 0
    for swing in relevant:
        projected = line_value(swing.index)
        if abs(swing.price - projected) <= latest_atr * 0.45:
            touches += 1
    risk = "low" if distance_atr <= 0.8 else "high"
    return {
        "hit": bool(hit),
        "touches": touches,
        "risk": risk,
        "distance_atr": round(distance_atr, 3),
    }


def amd_signal(candles: list[Candle], direction: str) -> dict[str, float | bool | str | int]:
    if len(candles) < 90:
        return {"hit": False, "phase": "insufficient", "score": 0.0}
    latest_atr = atr(candles, 14) or median_range(candles, 30) or candles[-1].close * 0.001
    range_window = candles[-80:-25]
    range_high = max(c.high for c in range_window)
    range_low = min(c.low for c in range_window)
    width = range_high - range_low
    median_width = median_range(range_window, 30) or latest_atr
    accumulation = width <= median_width * 8.0
    if not accumulation:
        return {"hit": False, "phase": "no_accumulation", "score": 0.0}
    manipulation_window = candles[-25:-5]
    distribution_window = candles[-10:]
    swept = False
    if direction == "long":
        swept = any(c.low < range_low and c.close > range_low for c in manipulation_window)
        distributed = any(c.close > (range_low + width * 0.65) and c.body > latest_atr * 0.8 for c in distribution_window)
    else:
        swept = any(c.high > range_high and c.close < range_high for c in manipulation_window)
        distributed = any(c.close < (range_low + width * 0.35) and c.body > latest_atr * 0.8 for c in distribution_window)
    if swept and distributed:
        return {"hit": True, "phase": "manipulation_to_distribution", "score": 1.0}
    if swept:
        return {"hit": False, "phase": "manipulation_seen_waiting_distribution", "score": 0.45}
    return {"hit": False, "phase": "accumulation_only", "score": 0.2}


def _ny_time(dt: datetime) -> datetime:
    return dt.astimezone(ZoneInfo("America/New_York"))


def _in_silver_bullet(dt: datetime) -> bool:
    local = _ny_time(dt)
    minutes = local.hour * 60 + local.minute
    return 10 * 60 <= minutes < 11 * 60 or 14 * 60 <= minutes < 15 * 60


def nexus_signal(candles: list[Candle], direction: str) -> dict[str, float | bool | str | int]:
    if len(candles) < 360:
        return {"hit": False, "reason": "insufficient_5m_history", "score": 0.0}
    latest_local_day = _ny_time(candles[-1].open_time).date()
    sessions: dict[object, list[Candle]] = {}
    for candle in candles:
        local = _ny_time(candle.open_time)
        if 2 <= local.hour < 5:
            sessions.setdefault(local.date(), []).append(candle)
    london = sessions.get(latest_local_day)
    if not london:
        previous_days = sorted(sessions)
        london = sessions.get(previous_days[-1]) if previous_days else None
    if not london:
        return {"hit": False, "reason": "no_london_session", "score": 0.0}
    london_high = max(c.high for c in london)
    london_low = min(c.low for c in london)
    window_indices = [idx for idx, candle in enumerate(candles[-180:], start=len(candles) - 180) if _in_silver_bullet(candle.open_time)]
    if not window_indices:
        return {"hit": False, "reason": "outside_silver_bullet", "score": 0.0}
    best_score = 0.0
    best_reason = "no_sweep"
    for idx in window_indices:
        candle = candles[idx]
        if direction == "long" and candle.low < london_low and candle.close > london_low:
            after = detect_structure_break(candles[: min(len(candles), idx + 24)], "long", after_index=idx, lookback=30)
            fvg = recent_relevant_fvg(candles[: min(len(candles), idx + 24)], "long", max_age=35)
            if after and fvg:
                return {"hit": True, "reason": "london_low_sweep_bos_fvg", "score": 1.0, "index": idx}
            best_score = max(best_score, 0.45)
            best_reason = "london_low_sweep_waiting_bos_fvg"
        if direction == "short" and candle.high > london_high and candle.close < london_high:
            after = detect_structure_break(candles[: min(len(candles), idx + 24)], "short", after_index=idx, lookback=30)
            fvg = recent_relevant_fvg(candles[: min(len(candles), idx + 24)], "short", max_age=35)
            if after and fvg:
                return {"hit": True, "reason": "london_high_sweep_bos_fvg", "score": 1.0, "index": idx}
            best_score = max(best_score, 0.45)
            best_reason = "london_high_sweep_waiting_bos_fvg"
    return {"hit": False, "reason": best_reason, "score": best_score}


def nearest_liquidity_targets(candles: list[Candle], current_price: float) -> tuple[float | None, float | None]:
    swings = swing_points(candles, left=2, right=2)
    highs = [s.price for s in swings if s.kind == "high" and s.price > current_price]
    lows = [s.price for s in swings if s.kind == "low" and s.price < current_price]
    buy_side = min(highs) if highs else None
    sell_side = max(lows) if lows else None
    return buy_side, sell_side


def returns(candles: list[Candle], limit: int = 80) -> list[float]:
    sample = candles[-(limit + 1) :]
    values: list[float] = []
    for previous, current in zip(sample, sample[1:]):
        if previous.close:
            values.append((current.close - previous.close) / previous.close)
    return values


def correlation(a: list[float], b: list[float]) -> float:
    length = min(len(a), len(b))
    if length < 10:
        return 0.0
    ax = a[-length:]
    bx = b[-length:]
    avg_a = sum(ax) / length
    avg_b = sum(bx) / length
    cov = sum((x - avg_a) * (y - avg_b) for x, y in zip(ax, bx))
    var_a = sum((x - avg_a) ** 2 for x in ax)
    var_b = sum((y - avg_b) ** 2 for y in bx)
    if var_a <= 0 or var_b <= 0:
        return 0.0
    return cov / math.sqrt(var_a * var_b)

