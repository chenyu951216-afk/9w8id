from __future__ import annotations

from datetime import datetime, timezone

from .exchanges import Ticker
from .models import Candle, DirectionScore, SymbolReport
from .technicals import (
    amd_signal,
    atr,
    correlation,
    detect_displacement,
    detect_fvgs,
    detect_liquidity_sweep,
    detect_structure_break,
    nearest_liquidity_targets,
    nexus_signal,
    order_block,
    ote_zone,
    price_position_in_range,
    recent_relevant_fvg,
    returns,
    swing_points,
    trendline_breakout,
    zone_overlap,
)


WEIGHTS = {
    "liquidity_sweep": 14.0,
    "htf_poi": 10.0,
    "mss_bos": 12.0,
    "displacement": 10.0,
    "fvg": 12.0,
    "ote": 8.0,
    "trendline": 8.0,
    "amd": 8.0,
    "nexus": 6.0,
    "risk_reward": 8.0,
    "market_quality": 4.0,
}


def _add(score: DirectionScore, feature: str, points: float, reason: str, weight_override: float | None = None) -> None:
    weight = weight_override if weight_override is not None else WEIGHTS[feature]
    if feature not in score.feature_max_scores:
        score.max_score += weight
        score.feature_max_scores[feature] = round(weight, 2)
    value = max(0.0, min(weight, points))
    score.score += value
    score.feature_scores[feature] = round(value, 2)
    if value > 0:
        score.reasons.append(f"+{value:.1f}/{weight:.0f} {reason}")


def _skip(score: DirectionScore, feature: str, reason: str) -> None:
    score.skipped_features[feature] = reason
    _warn(score, f"{feature} 未納入分母：{reason}")


def _warn(score: DirectionScore, message: str) -> None:
    if message not in score.warnings:
        score.warnings.append(message)


def _tf(candles_by_tf: dict[str, list[Candle]], key: str) -> list[Candle]:
    return candles_by_tf.get(key, [])


def _price(candles_by_tf: dict[str, list[Candle]], ticker: Ticker) -> float:
    for key in ("5m", "15m", "1h", "4h"):
        candles = _tf(candles_by_tf, key)
        if candles:
            return candles[-1].close
    return ticker.last_price


def _price_near_zone(price: float, low: float, high: float, tolerance_pct: float = 0.35) -> bool:
    tolerance = price * tolerance_pct / 100.0
    return low - tolerance <= price <= high + tolerance


def _zone_distance_pct(price: float, low: float, high: float) -> float:
    if low <= price <= high:
        return 0.0
    return min(abs(price - low), abs(price - high)) / max(price, 1e-12) * 100.0


def _risk_setup(
    direction: str,
    candles_15m: list[Candle],
    candles_1h: list[Candle],
    price: float,
    fvg_zone: tuple[float, float] | None,
    ote: tuple[float, float, float] | None,
) -> tuple[tuple[float, float], float, float, float, list[dict[str, float | str]]]:
    active_zone = fvg_zone or ((ote[0], ote[1]) if ote else (price, price))
    zone_low, zone_high = active_zone
    entry = (zone_low + zone_high) / 2.0
    candles = candles_15m or candles_1h
    if not candles:
        buffer = price * 0.004
        if direction == "long":
            stop = entry - buffer
            target = entry + buffer * 2
        else:
            stop = entry + buffer
            target = entry - buffer * 2
        take_profits = _build_take_profits(direction, entry, stop, target, None)
        return (zone_low, zone_high), stop, target, 2.0, take_profits

    local_atr = atr(candles, 14) or price * 0.004
    buy_side, sell_side = nearest_liquidity_targets(candles, price)
    swings = swing_points(candles, left=2, right=2)
    if direction == "long":
        lows = [s.price for s in swings if s.kind == "low" and s.price < min(zone_low, price)]
        base_stop = max(lows) if lows else min(c.low for c in candles[-20:])
        stop = min(base_stop, zone_low) - local_atr * 0.2
        risk = max(entry - stop, price * 0.001)
        liquidity_target = buy_side if buy_side and buy_side > entry else None
        target = liquidity_target if liquidity_target and liquidity_target >= entry + risk * 1.2 else entry + risk * 2.0
        reward = max(target - entry, 0.0)
    else:
        highs = [s.price for s in swings if s.kind == "high" and s.price > max(zone_high, price)]
        base_stop = min(highs) if highs else max(c.high for c in candles[-20:])
        stop = max(base_stop, zone_high) + local_atr * 0.2
        risk = max(stop - entry, price * 0.001)
        liquidity_target = sell_side if sell_side and sell_side < entry else None
        target = liquidity_target if liquidity_target and liquidity_target <= entry - risk * 1.2 else entry - risk * 2.0
        reward = max(entry - target, 0.0)
    rr = reward / max(risk, 1e-12)
    take_profits = _build_take_profits(direction, entry, stop, target, liquidity_target)
    return (zone_low, zone_high), stop, target, rr, take_profits


def _build_take_profits(
    direction: str,
    entry: float,
    stop: float,
    target: float,
    liquidity_target: float | None,
) -> list[dict[str, float | str]]:
    if direction == "long":
        risk = max(entry - stop, abs(entry) * 0.001)
        tp1 = entry + risk
        tp2 = liquidity_target if liquidity_target and liquidity_target > tp1 else entry + risk * 2.0
        tp3 = max(target, entry + risk * 3.0)
        return [
            {"name": "TP1", "price": tp1, "rr": 1.0, "portion_pct": 30.0, "note": "到 1R 可先減倉，保護本金"},
            {"name": "TP2", "price": tp2, "rr": (tp2 - entry) / risk, "portion_pct": 40.0, "note": "主目標，優先看最近買方流動性"},
            {"name": "TP3", "price": tp3, "rr": (tp3 - entry) / risk, "portion_pct": 30.0, "note": "延伸目標，趨勢強才保留"},
        ]
    risk = max(stop - entry, abs(entry) * 0.001)
    tp1 = entry - risk
    tp2 = liquidity_target if liquidity_target and liquidity_target < tp1 else entry - risk * 2.0
    tp3 = min(target, entry - risk * 3.0)
    return [
        {"name": "TP1", "price": tp1, "rr": 1.0, "portion_pct": 30.0, "note": "到 1R 可先減倉，保護本金"},
        {"name": "TP2", "price": tp2, "rr": (entry - tp2) / risk, "portion_pct": 40.0, "note": "主目標，優先看最近賣方流動性"},
        {"name": "TP3", "price": tp3, "rr": (entry - tp3) / risk, "portion_pct": 30.0, "note": "延伸目標，趨勢強才保留"},
    ]


def _evaluate_direction(
    direction: str,
    ticker: Ticker,
    candles_by_tf: dict[str, list[Candle]],
    btc_1h: list[Candle] | None,
) -> DirectionScore:
    score = DirectionScore(direction=direction, reference_max_score=sum(WEIGHTS.values()))
    candles_4h = _tf(candles_by_tf, "4h")
    candles_1h = _tf(candles_by_tf, "1h")
    candles_15m = _tf(candles_by_tf, "15m")
    candles_5m = _tf(candles_by_tf, "5m")
    price = _price(candles_by_tf, ticker)

    if not candles_4h:
        _warn(score, "缺少 4H K 線，HTF POI 與趨勢線分數會偏保守")
    if not candles_1h:
        _warn(score, "缺少 1H K 線，流動性掃蕩分數會偏保守")
    if not candles_15m:
        _warn(score, "缺少 15m K 線，MSS/BOS 與 FVG 分數會偏保守")
    if not candles_5m:
        _warn(score, "缺少 5m K 線，Nexus/Silver Bullet 分數會偏保守")

    sweep_1h = detect_liquidity_sweep(candles_1h, direction, lookback=90) if candles_1h else None
    sweep_4h = detect_liquidity_sweep(candles_4h, direction, lookback=70) if candles_4h else None
    sweep = sweep_1h or sweep_4h
    if candles_1h or candles_4h:
        if sweep:
            source = "1H" if sweep_1h else "4H"
            points = WEIGHTS["liquidity_sweep"] * min(1.0, 0.5 + sweep.strength / 2.2)
            if sweep_1h and sweep_4h:
                points = WEIGHTS["liquidity_sweep"]
                source = "1H + 4H"
            _add(score, "liquidity_sweep", points, f"{source} 出現有效流動性掃蕩並收回關鍵位")
        else:
            _add(score, "liquidity_sweep", 0.0, "")
            _warn(score, "最近 HTF 沒有清楚的掃高/掃低後收回訊號")
    else:
        _skip(score, "liquidity_sweep", "缺少 1H/4H K 線")

    poi_points = 0.0
    poi_notes: list[str] = []
    if candles_4h:
        position, low, high = price_position_in_range(candles_4h, lookback=120)
        if direction == "long":
            if position <= 0.5:
                poi_points += 4.0 if position <= 0.35 else 2.8
                poi_notes.append(f"4H 位於折價區 position={position:.2f}")
        else:
            if position >= 0.5:
                poi_points += 4.0 if position >= 0.65 else 2.8
                poi_notes.append(f"4H 位於溢價區 position={position:.2f}")
        htf_fvg = recent_relevant_fvg(candles_4h, direction, max_age=80)
        if htf_fvg and _price_near_zone(price, htf_fvg.lower, htf_fvg.upper, tolerance_pct=0.75):
            poi_points += 3.0
            poi_notes.append("接近 4H FVG/失衡區")
        ob = order_block(candles_4h, direction, before_index=(sweep_4h.index if sweep_4h else None), lookback=70)
        if ob and _price_near_zone(price, ob.lower, ob.upper, tolerance_pct=0.75):
            poi_points += 3.0
            poi_notes.append("接近 4H 訂單塊 POI")
        _add(score, "htf_poi", poi_points, "、".join(poi_notes) if poi_notes else "")
    else:
        _skip(score, "htf_poi", "缺少 4H K 線，無法判斷 HTF POI / 折價溢價")

    ltf_sweep = detect_liquidity_sweep(candles_15m, direction, lookback=100) if candles_15m else None
    after_index = ltf_sweep.index if ltf_sweep else None
    structure = detect_structure_break(candles_15m, direction, after_index=after_index, lookback=100) if candles_15m else None
    if candles_15m:
        if structure:
            base = 7.0 if structure.kind == "BOS" else 9.0
            if ltf_sweep:
                base += 2.0
            if sweep:
                base += 1.0
            base = min(WEIGHTS["mss_bos"], base)
            _add(score, "mss_bos", base, f"15m {structure.kind} 收盤突破 {structure.level:g}")
        else:
            _add(score, "mss_bos", 0.0, "")
            _warn(score, "15m 尚未確認 MSS/BOS")
    else:
        _skip(score, "mss_bos", "缺少 15m K 線")

    displacement = detect_displacement(
        candles_15m,
        direction,
        after_index=(structure.index if structure else after_index),
        lookback=55,
    ) if candles_15m else None
    if candles_15m:
        if displacement:
            points = 6.0 + (2.0 if displacement.has_fvg else 0.0)
            if structure and displacement.index >= structure.index:
                points += 1.2
            if ltf_sweep and displacement.index >= ltf_sweep.index:
                points += 0.8
            points = min(WEIGHTS["displacement"], points * min(1.2, displacement.body_atr / 1.15))
            fvg_text = "並留下 FVG" if displacement.has_fvg else "但 FVG 不明顯"
            _add(score, "displacement", points, f"15m 位移 K body/ATR={displacement.body_atr:.2f} {fvg_text}")
        else:
            _add(score, "displacement", 0.0, "")
            _warn(score, "沒有明確大實體單向位移")
    else:
        _skip(score, "displacement", "缺少 15m K 線")

    ote = ote_zone(candles_15m or candles_1h, direction, lookback=100) if (candles_15m or candles_1h) else None
    fvg_15m = recent_relevant_fvg(candles_15m, direction, max_age=110) if candles_15m else None
    fvg_5m = recent_relevant_fvg(candles_5m, direction, max_age=160) if candles_5m else None
    selected_fvg = fvg_15m or fvg_5m
    fvg_zone: tuple[float, float] | None = None
    if candles_15m or candles_5m:
        if selected_fvg:
            fvg_zone = (selected_fvg.lower, selected_fvg.upper)
            points = 6.0
            if selected_fvg.tapped and not selected_fvg.filled:
                points += 2.5
            if _price_near_zone(price, selected_fvg.lower, selected_fvg.upper, tolerance_pct=0.35):
                points += 1.0
            overlap = False
            if candles_15m:
                ob = order_block(candles_15m, direction, before_index=selected_fvg.index, lookback=45)
                overlap = bool(ob and zone_overlap(selected_fvg.lower, selected_fvg.upper, ob.lower, ob.upper))
            if overlap:
                points += 1.2
            if structure and abs(selected_fvg.index - structure.index) <= 4:
                points += 1.0
            if ote and zone_overlap(selected_fvg.lower, selected_fvg.upper, ote[0], ote[1]):
                points += 1.3
            points = min(WEIGHTS["fvg"], points)
            note = "已回補測試但未完全填補" if selected_fvg.tapped and not selected_fvg.filled else "尚未完全填補"
            if overlap:
                note += "，且與 OB 重疊"
            if ote and zone_overlap(selected_fvg.lower, selected_fvg.upper, ote[0], ote[1]):
                note += "，且與 OTE 重疊"
            _add(score, "fvg", points, f"{selected_fvg.start_time.isoformat()} {direction} FVG {note}")
        else:
            _add(score, "fvg", 0.0, "")
            _warn(score, "近期找不到方向一致且未完全填補的 FVG")
    else:
        _skip(score, "fvg", "缺少 15m/5m K 線")

    if ote:
        zone_low, zone_high, retracement = ote
        overlap_note = ""
        overlap_bonus = 0.0
        if fvg_zone and zone_overlap(fvg_zone[0], fvg_zone[1], zone_low, zone_high):
            overlap_note = "，且與 FVG 入場區重疊"
            overlap_bonus = 1.5
        if 0.62 <= retracement <= 0.79:
            _add(score, "ote", min(8.0, 7.0 + overlap_bonus), f"價格位於 OTE 0.62-0.79 回撤區 ({retracement:.2f}){overlap_note}")
        elif 0.50 <= retracement <= 0.86:
            _add(score, "ote", min(8.0, 4.0 + overlap_bonus), f"價格接近 OTE 區但未進核心帶 ({retracement:.2f}){overlap_note}")
        else:
            _add(score, "ote", 0.0, "")
    else:
        _skip(score, "ote", "缺少 15m/1H K 線，無法計算 OTE")

    if candles_4h and len(candles_4h) >= 60:
        trendline = trendline_breakout(candles_4h, direction)
        if trendline.get("hit"):
            touches = int(trendline.get("touches") or 0)
            points = 5.5 + min(2.0, max(0, touches - 2) * 1.0)
            if trendline.get("risk") == "low":
                points += 0.5
            _add(score, "trendline", points, f"4H 趨勢線破位，觸碰 {touches} 次，風險距離={trendline.get('risk')}")
        elif int(trendline.get("touches") or 0) >= 2:
            _add(score, "trendline", 2.5, f"4H 有 {trendline.get('touches')} 觸點趨勢線，但尚未破位")
        else:
            _add(score, "trendline", 0.0, "")
    else:
        _skip(score, "trendline", "缺少足夠 4H K 線，無法判斷 2-3 觸點趨勢線破位")

    amd_candles = candles_15m or candles_5m
    if amd_candles and len(amd_candles) >= 90:
        amd = amd_signal(amd_candles, direction)
        amd_points = float(amd.get("score") or 0.0) * WEIGHTS["amd"]
        _add(score, "amd", amd_points, f"AMD phase={amd.get('phase')}" if amd_points else "")
    else:
        _skip(score, "amd", "缺少足夠 15m/5m 歷史，無法判斷吸籌-操縱-派發")

    if candles_5m and len(candles_5m) >= 360:
        nexus = nexus_signal(candles_5m, direction)
        nexus_points = float(nexus.get("score") or 0.0) * WEIGHTS["nexus"]
        _add(score, "nexus", nexus_points, f"Nexus/Silver Bullet: {nexus.get('reason')}" if nexus_points else "")
    else:
        _skip(score, "nexus", "缺少足夠 5m 歷史，無法判斷倫敦高低點與 Silver Bullet")

    if candles_15m or candles_1h:
        entry_zone, stop, target, rr, take_profits = _risk_setup(direction, candles_15m, candles_1h, price, fvg_zone, ote)
        score.entry_zone = entry_zone
        score.stop = stop
        score.target = target
        score.take_profits = take_profits
        score.rr = rr
        zone_dist = _zone_distance_pct(price, entry_zone[0], entry_zone[1])
        chase_penalty = 1.5 if zone_dist > 1.2 else 0.0
        if rr >= 2.0:
            _add(score, "risk_reward", max(0.0, 8.0 - chase_penalty), f"以最近流動性目標估算 RR={rr:.2f}")
        elif rr >= 1.5:
            _add(score, "risk_reward", max(0.0, 5.0 - chase_penalty), f"RR={rr:.2f}，可觀察但不是最漂亮")
        elif rr >= 1.1:
            _add(score, "risk_reward", max(0.0, 2.0 - chase_penalty), f"RR={rr:.2f} 偏低")
        else:
            _add(score, "risk_reward", 0.0, "")
            _warn(score, f"RR={rr:.2f} 不足，入場區到目標/止損不划算")
        if zone_dist > 1.2:
            _warn(score, f"現價離入場區約 {zone_dist:.2f}%，依圖片規則不追價，等回補/回測再看")
    else:
        _skip(score, "risk_reward", "缺少 15m/1H K 線，無法估算入場區、止損與目標")

    quality_points = 0.0
    quality_notes: list[str] = []
    if ticker.quote_volume >= 100_000_000:
        quality_points += 2.0
        quality_notes.append("24h 成交額 > 1 億 USDT")
    elif ticker.quote_volume >= 20_000_000:
        quality_points += 1.2
        quality_notes.append("24h 成交額 > 2,000 萬 USDT")
    else:
        _warn(score, "24h 成交額偏低，滑點與假突破風險較高")

    quality_max = 2.0
    if btc_1h and candles_1h and ticker.symbol != "BTCUSDT":
        quality_max = 4.0
        corr = correlation(returns(candles_1h, 80), returns(btc_1h, 80))
        btc_trend = btc_1h[-1].close - btc_1h[-24].close if len(btc_1h) >= 24 else 0.0
        aligned = (direction == "long" and btc_trend >= 0) or (direction == "short" and btc_trend <= 0)
        if corr >= 0.45 and aligned:
            quality_points += 2.0
            quality_notes.append(f"與 BTC 相關性 {corr:.2f} 且方向一致")
        elif corr >= 0.45:
            quality_points += 0.8
            quality_notes.append(f"與 BTC 相關性 {corr:.2f}，但 BTC 方向不完全同向")
    elif ticker.symbol == "BTCUSDT":
        quality_points += 1.0
        quality_notes.append("BTC 本身作為市場基準")
    else:
        _warn(score, "BTC 1H 相關性資料缺失，市場品質只用成交額評估")

    _add(score, "market_quality", quality_points, "、".join(quality_notes) if quality_notes else "", weight_override=quality_max)
    return score


def score_symbol(
    exchange_name: str,
    ticker: Ticker,
    candles_by_tf: dict[str, list[Candle]],
    btc_1h: list[Candle] | None = None,
) -> SymbolReport:
    missing = [tf for tf in ("4h", "1h", "15m", "5m") if not candles_by_tf.get(tf)]
    data_time = datetime.now(timezone.utc)
    for key in ("5m", "15m", "1h", "4h"):
        candles = candles_by_tf.get(key)
        if candles:
            data_time = candles[-1].open_time
            break
    price = _price(candles_by_tf, ticker)

    long = _evaluate_direction("long", ticker, candles_by_tf, btc_1h)
    short = _evaluate_direction("short", ticker, candles_by_tf, btc_1h)
    if long.normalized > short.normalized:
        direction = "long"
        selected = long.normalized
    elif short.normalized > long.normalized:
        direction = "short"
        selected = short.normalized
    else:
        direction = "neutral"
        selected = long.normalized

    if max(long.normalized, short.normalized) < 52:
        direction = "neutral"

    return SymbolReport(
        symbol=ticker.symbol,
        exchange=exchange_name,
        price=price,
        quote_volume_24h=ticker.quote_volume,
        change_pct_24h=ticker.change_pct,
        data_time=data_time,
        selected_direction=direction,
        score=round(selected, 2),
        long=long,
        short=short,
        data_coverage={tf: len(candles_by_tf.get(tf, [])) for tf in ("4h", "1h", "15m", "5m")},
        missing_data=missing,
    )
