from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..exchanges import Ticker
from ..instrument_classifier import participation_profile, volatility_profile
from ..models import Candle, DirectionScore, SymbolReport
from ..price_action import analyze_price_action
from ..technicals import (
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
    "liquidity_sweep": 16.0,
    "htf_poi": 14.0,
    "key_level": 14.0,
    "mss_bos": 16.0,
    "displacement": 12.0,
    "price_action": 12.0,
    "breakout_quality": 10.0,
    "fvg": 14.0,
    "ote": 10.0,
    "risk_reward": 12.0,
    "market_quality": 6.0,
}

OPTIONAL_BONUS_WEIGHTS = {
    "trendline": 2.0,
    "amd": 2.0,
    "nexus": 2.0,
}

MAX_CALIBRATED_BONUS = 4.0
MIN_DIRECTION_GAP = 8.0
CONFLICT_DIRECTION_GAP = 12.0
VALIDATION_STATE_PATH = Path("state/signal_state.json")
VALIDATION_REPORT_PATH = Path("reports/latest.json")

BUCKET_WEIGHTS = {
    "htf_context": 26.0,
    "ltf_confirmation": 24.0,
    "entry_location": 20.0,
    "risk_plan": 18.0,
    "market_filter": 12.0,
}

BUCKET_FEATURES = {
    "htf_context": ["liquidity_sweep", "htf_poi", "key_level"],
    "ltf_confirmation": ["mss_bos", "displacement", "price_action", "breakout_quality"],
    "entry_location": ["fvg", "ote"],
    "risk_plan": ["risk_reward"],
    "market_filter": ["market_quality"],
}

CORRELATED_FEATURE_CAPS = (
    ({"htf_poi", "key_level"}, 20.0),
    ({"price_action", "breakout_quality"}, 16.0),
    ({"fvg", "ote"}, 21.0),
)

SETUP_TAG_LABELS = {
    "liquidity_sweep": "Liquidity Sweep",
    "htf_poi": "HTF POI",
    "key_level": "Key Level",
    "mss_bos": "MSS/BOS",
    "displacement": "Displacement",
    "price_action": "Price Action",
    "breakout_quality": "Breakout Quality",
    "fvg": "FVG",
    "ote": "OTE",
    "trendline": "Trendline Break",
    "amd": "AMD",
    "nexus": "Nexus",
    "risk_reward": "Risk Reward",
    "market_quality": "Market Filter",
    "paid_data": "External Data",
}

BROAD_VALIDATION_TAGS = set(SETUP_TAG_LABELS.values()) | {
    "key_level",
    "price_action",
    "breakout_quality",
}

NEXUS_REASON_LABELS = {
    "outside_silver_bullet": "目前不在紐約 10-11 / 14-15 Silver Bullet 窗口",
    "insufficient_5m_history": "5m 歷史不足",
    "no_london_session": "近期未解析到倫敦盤區間",
    "no_sweep": "窗口內尚未掃倫敦高/低點",
    "london_low_sweep_waiting_bos_fvg": "已掃倫敦低點，等待 BOS + FVG",
    "london_high_sweep_waiting_bos_fvg": "已掃倫敦高點，等待 BOS + FVG",
    "london_low_sweep_bos_fvg": "倫敦低點掃蕩 + BOS + FVG 成立",
    "london_high_sweep_bos_fvg": "倫敦高點掃蕩 + BOS + FVG 成立",
}

AMD_PHASE_LABELS = {
    "insufficient": "歷史不足",
    "no_accumulation": "未形成吸籌盤整",
    "accumulation_only": "只有吸籌，尚未操縱/派發",
    "manipulation_seen_waiting_distribution": "已操縱，等待派發位移",
    "manipulation_to_distribution": "操縱後派發成立",
}


def _label(mapping: dict[str, str], value: object) -> str:
    text = str(value or "")
    return mapping.get(text, text or "未觸發")


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


def _bonus(score: DirectionScore, feature: str, points: float, reason: str, weight_override: float | None = None) -> None:
    weight = weight_override if weight_override is not None else OPTIONAL_BONUS_WEIGHTS[feature]
    if feature not in score.feature_max_scores:
        score.bonus_max_score += weight
        score.feature_max_scores[feature] = round(weight, 2)
    value = max(0.0, min(weight, points))
    score.score += value
    score.bonus_score += value
    score.feature_scores[feature] = round(score.feature_scores.get(feature, 0.0) + value, 2)
    if value > 0:
        score.reasons.append(f"+{value:.1f}/{weight:.0f} 共振加分：{reason}")


def _skip(score: DirectionScore, feature: str, reason: str) -> None:
    score.skipped_features[feature] = reason
    _warn(score, f"{feature} 未納入分母：{reason}")


def _inactive(score: DirectionScore, feature: str, reason: str) -> None:
    score.inactive_features[feature] = reason


def _warn(score: DirectionScore, message: str) -> None:
    if message not in score.warnings:
        score.warnings.append(message)


def _note(score: DirectionScore, message: str) -> None:
    if message not in score.signal_notes:
        score.signal_notes.append(message)


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


def _bucket_ratio(score: DirectionScore, names: list[str]) -> tuple[float | None, float]:
    available = sum(score.feature_max_scores.get(name, 0.0) for name in names)
    if available <= 0:
        return None, 0.0
    value = sum(score.feature_scores.get(name, 0.0) for name in names)
    value, available = _apply_correlated_feature_caps(score, names, value, available)
    return max(0.0, min(100.0, value / available * 100.0)), available


def _apply_correlated_feature_caps(
    score: DirectionScore,
    names: list[str],
    value: float,
    available: float,
) -> tuple[float, float]:
    name_set = set(names)
    for group, cap in CORRELATED_FEATURE_CAPS:
        if not group.issubset(name_set):
            continue
        group_available = sum(score.feature_max_scores.get(name, 0.0) for name in group)
        if group_available <= cap:
            continue
        group_value = sum(score.feature_scores.get(name, 0.0) for name in group)
        value -= max(0.0, group_value - cap)
        available -= group_available - cap
    return max(0.0, value), max(1e-9, available)


def _effective_score(score: DirectionScore) -> float:
    if score.selection_score is not None:
        return score.selection_score
    if score.calibrated_score is not None:
        return score.calibrated_score
    return score.normalized


def _setup_tags_from_score(score: DirectionScore) -> list[str]:
    tags = []
    for name, value in score.feature_scores.items():
        if value > 0:
            tags.append(SETUP_TAG_LABELS.get(name, name))
    output: list[str] = []
    for tag in tags:
        if tag not in output:
            output.append(tag)
    return output


def _setup_is_complete(score: DirectionScore) -> bool:
    buckets = score.bucket_scores or {}
    return (
        buckets.get("htf_context", 0.0) >= 60.0
        and buckets.get("ltf_confirmation", 0.0) >= 65.0
        and buckets.get("entry_location", 0.0) >= 65.0
        and buckets.get("risk_plan", 0.0) >= 60.0
    )


def _intersect_zones(*zones: tuple[float, float] | None) -> tuple[float, float] | None:
    clean = [zone for zone in zones if zone is not None]
    if not clean:
        return None
    low = max(zone[0] for zone in clean)
    high = min(zone[1] for zone in clean)
    if low <= high:
        return low, high
    return None


def _fvg_mid_entry_zone(direction: str, lower: float, upper: float) -> tuple[float, float]:
    midpoint = (lower + upper) / 2.0
    if direction == "long":
        return midpoint, upper
    return lower, midpoint


def _atr_pct(price: float, local_atr: float) -> float:
    return local_atr / max(abs(price), 1e-12) * 100.0


def _intraday_stop_atr_mult(atr_pct: float) -> float:
    if atr_pct >= 4.0:
        return 0.22
    if atr_pct >= 2.4:
        return 0.26
    if atr_pct >= 1.2:
        return 0.30
    return 0.34


def _intraday_min_stop_risk_atr_mult(atr_pct: float) -> float:
    if atr_pct >= 4.0:
        return 0.55
    if atr_pct >= 2.4:
        return 0.60
    if atr_pct >= 1.2:
        return 0.65
    return 0.55


def _intraday_max_stop_risk_atr_mult(atr_pct: float) -> float:
    if atr_pct >= 4.0:
        return 1.15
    if atr_pct >= 2.4:
        return 1.35
    if atr_pct >= 1.2:
        return 1.55
    return 1.35


def _intraday_max_stop_pct(atr_pct: float) -> float:
    if atr_pct >= 4.0:
        return 3.0
    if atr_pct >= 2.4:
        return 2.4
    if atr_pct >= 1.2:
        return 1.8
    return 1.2


def _intraday_default_target_rr(atr_pct: float) -> float:
    if atr_pct >= 2.4:
        return 2.2
    return 2.0


def _intraday_min_liquidity_rr(atr_pct: float) -> float:
    if atr_pct >= 2.4:
        return 1.45
    return 1.55


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _rr_to_level(direction: str, entry: float, risk: float, rr: float) -> float:
    if direction == "long":
        return entry + risk * rr
    return entry - risk * rr


def _level_rr(direction: str, entry: float, risk: float, level: float) -> float:
    if risk <= 0:
        return 0.0
    if direction == "long":
        return (level - entry) / risk
    return (entry - level) / risk


def _front_run_target(direction: str, level: float, local_atr: float) -> float:
    buffer = max(local_atr * 0.08, abs(level) * 0.0002)
    if direction == "long":
        return level - buffer
    return level + buffer


def _dedupe_levels(levels: list[float], local_atr: float, reverse: bool = False) -> list[float]:
    ordered = sorted((level for level in levels if level > 0), reverse=reverse)
    if not ordered:
        return []
    tolerance = max(local_atr * 0.18, abs(ordered[0]) * 0.0005)
    output: list[float] = []
    for level in ordered:
        if output and abs(level - output[-1]) <= tolerance:
            continue
        output.append(level)
    return output


def _equal_liquidity_extreme(
    swings: list[Any],
    kind: str,
    entry: float,
    local_atr: float,
    direction: str,
) -> float | None:
    tolerance = max(local_atr * 0.25, abs(entry) * 0.0006)
    if kind == "low":
        prices = [s.price for s in swings if s.kind == "low" and s.price < entry]
    else:
        prices = [s.price for s in swings if s.kind == "high" and s.price > entry]
    for price in prices:
        cluster = [candidate for candidate in prices if abs(candidate - price) <= tolerance]
        if len(cluster) >= 2:
            return min(cluster) if direction == "long" else max(cluster)
    return None


def _bounded_structural_stop(
    direction: str,
    candles: list[Candle],
    swings: list[Any],
    entry: float,
    zone_low: float,
    zone_high: float,
    local_atr: float,
    atr_pct: float,
) -> float:
    stop_buffer = local_atr * _intraday_stop_atr_mult(atr_pct)
    min_risk = max(abs(entry) * 0.0015, local_atr * _intraday_min_stop_risk_atr_mult(atr_pct))
    max_risk = min(local_atr * _intraday_max_stop_risk_atr_mult(atr_pct), abs(entry) * _intraday_max_stop_pct(atr_pct) / 100.0)
    max_risk = max(min_risk * 1.05, max_risk)
    recent_start = max(0, len(candles) - 96)
    recent_swings = [s for s in swings if s.index >= recent_start]
    recent = candles[-20:] if candles else []
    sweep = detect_liquidity_sweep(candles, direction, lookback=90) if candles else None
    level_candidates: list[float] = []
    if direction == "long":
        level_candidates.extend(s.price for s in recent_swings if s.kind == "low" and s.price < entry)
        if recent:
            level_candidates.append(min(c.low for c in recent))
        equal_low = _equal_liquidity_extreme(recent_swings, "low", entry, local_atr, direction)
        if equal_low is not None:
            level_candidates.append(equal_low)
        if sweep is not None:
            level_candidates.append(sweep.extreme)
        stops = [(min(level, zone_low) - stop_buffer, entry - (min(level, zone_low) - stop_buffer)) for level in level_candidates if level < entry]
        fallback_near = entry - min_risk
        fallback_far = entry - max_risk
    else:
        level_candidates.extend(s.price for s in recent_swings if s.kind == "high" and s.price > entry)
        if recent:
            level_candidates.append(max(c.high for c in recent))
        equal_high = _equal_liquidity_extreme(recent_swings, "high", entry, local_atr, direction)
        if equal_high is not None:
            level_candidates.append(equal_high)
        if sweep is not None:
            level_candidates.append(sweep.extreme)
        stops = [(max(level, zone_high) + stop_buffer, (max(level, zone_high) + stop_buffer) - entry) for level in level_candidates if level > entry]
        fallback_near = entry + min_risk
        fallback_far = entry + max_risk
    stops = [(stop, risk) for stop, risk in stops if risk > 0]
    stops.sort(key=lambda item: item[1])
    for stop, risk in stops:
        if min_risk <= risk <= max_risk:
            return stop
    if stops:
        if stops[0][1] > max_risk:
            if stops[0][1] <= max_risk * 1.12:
                return stops[0][0]
            return fallback_far
        return fallback_near
    return fallback_near


def _profit_reference_levels(
    direction: str,
    candles: list[Candle],
    swings: list[Any],
    entry: float,
    liquidity_target: float | None,
    local_atr: float,
) -> list[float]:
    recent_start = max(0, len(candles) - 120)
    recent_swings = [s for s in swings if s.index >= recent_start]
    levels: list[float] = []
    if direction == "long":
        levels.extend(s.price for s in recent_swings if s.kind == "high" and s.price > entry)
        levels.extend(c.high for c in candles[-24:] if c.high > entry)
        if liquidity_target and liquidity_target > entry:
            levels.append(liquidity_target)
        return _dedupe_levels(levels, local_atr)
    levels.extend(s.price for s in recent_swings if s.kind == "low" and s.price < entry)
    levels.extend(c.low for c in candles[-24:] if c.low < entry)
    if liquidity_target and liquidity_target < entry:
        levels.append(liquidity_target)
    return _dedupe_levels(levels, local_atr, reverse=True)


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=8)
def _validation_stats_cached(state_mtime: float, report_mtime: float) -> dict[str, Any]:
    state = _load_json_file(VALIDATION_STATE_PATH)
    stats = state.get("statistics") if isinstance(state.get("statistics"), dict) else {}
    if stats:
        return stats
    latest = _load_json_file(VALIDATION_REPORT_PATH)
    meta = latest.get("meta") if isinstance(latest.get("meta"), dict) else {}
    latest_stats = meta.get("signal_statistics") if isinstance(meta.get("signal_statistics"), dict) else {}
    return latest_stats if isinstance(latest_stats, dict) else {}


def _load_validation_stats() -> dict[str, Any]:
    return _validation_stats_cached(_file_mtime(VALIDATION_STATE_PATH), _file_mtime(VALIDATION_REPORT_PATH))


def _validation_adjustment_from_stats(score: DirectionScore, stats: dict[str, Any]) -> tuple[float, float, float, list[str]]:
    tags = score.setup_tags or _setup_tags_from_score(score)
    tag_stats = stats.get("setup_tag_stats", {}) if isinstance(stats, dict) else {}
    combo_stats = stats.get("setup_combo_stats", {}) if isinstance(stats, dict) else {}
    penalty = 0.0
    selection_cap = 100.0
    execution_cap = 100.0
    reasons: list[str] = []
    raw_penalty = 0.0
    combo_penalty, combo_selection_cap, combo_execution_cap, combo_reasons = _validation_adjustment_from_combos(tags, combo_stats)
    penalty += combo_penalty
    raw_penalty += combo_penalty
    selection_cap = min(selection_cap, combo_selection_cap)
    execution_cap = min(execution_cap, combo_execution_cap)
    reasons.extend(combo_reasons)
    for tag in tags:
        item = tag_stats.get(tag)
        if not isinstance(item, dict):
            continue
        total = int(item.get("total") or 0)
        accuracy = item.get("accuracy")
        avg_mfe = item.get("average_mfe")
        avg_mae = item.get("average_mae")
        sl_first_rate = item.get("sl_first_rate")
        high_score_reversal_rate = item.get("high_score_reversal_rate")
        if total >= 8 and accuracy is not None and float(accuracy) < 45.0:
            if _is_broad_validation_tag(tag):
                if total < 20:
                    continue
                cut = min(2.0, (45.0 - float(accuracy)) * 0.07)
                selection_cap = min(selection_cap, 88.0)
                execution_cap = min(execution_cap, 78.0)
                reasons.append(f"{tag} 是常見結構 tag，近期勝率 {float(accuracy):.1f}% 僅輕量校準 {cut:.1f} 分")
            else:
                cut = min(8.0, (45.0 - float(accuracy)) * 0.28)
                selection_cap = min(selection_cap, 82.0)
                execution_cap = min(execution_cap, 72.0)
                reasons.append(f"{tag} 最近樣本勝率 {float(accuracy):.1f}% 低於 45%，校準降權 {cut:.1f} 分")
            penalty += cut
            raw_penalty += cut
        if total >= 8 and avg_mfe is not None and avg_mae is not None and float(avg_mae) > float(avg_mfe):
            cut = 0.8 if _is_broad_validation_tag(tag) else 3.0
            penalty += cut
            raw_penalty += cut
            execution_cap = min(execution_cap, 76.0 if _is_broad_validation_tag(tag) else 70.0)
            reasons.append(f"{tag} MAE 大於 MFE，校準 execution_score {cut:.1f} 分")
        if total >= 8 and sl_first_rate is not None and float(sl_first_rate) >= 50.0:
            if _is_broad_validation_tag(tag):
                if float(sl_first_rate) >= 75.0:
                    execution_cap = min(execution_cap, 78.0)
                    reasons.append(f"{tag} 的 SL-first 統計偏高，但屬基礎 tag，只作風險提醒不直接禁止")
            else:
                execution_cap = min(execution_cap, 64.0)
                reasons.append(f"{tag} 經常先碰 SL，禁止直接執行")
        if total >= 8 and high_score_reversal_rate is not None and float(high_score_reversal_rate) >= 50.0:
            selection_cap = min(selection_cap, 84.0 if _is_broad_validation_tag(tag) else 78.0)
            execution_cap = min(execution_cap, 72.0 if _is_broad_validation_tag(tag) else 64.0)
            reasons.append(f"{tag} 高分後 3-6 根 K 常反向，降低候選等級")
    if raw_penalty > 14.0:
        penalty = min(penalty, 10.0)
        reasons.append("相關 setup tag 同時出現，歷史校準扣分封頂 10 分，避免重複懲罰同一結構")
    return penalty, selection_cap, execution_cap, reasons


def _validation_adjustment_from_combos(tags: list[str], combo_stats: dict[str, Any]) -> tuple[float, float, float, list[str]]:
    if not isinstance(combo_stats, dict):
        return 0.0, 100.0, 100.0, []
    tag_set = set(tags)
    combo_key = _validation_combo_key(tag_set)
    if not combo_key:
        return 0.0, 100.0, 100.0, []
    item = combo_stats.get(combo_key)
    if not isinstance(item, dict):
        return 0.0, 100.0, 100.0, []
    total = int(item.get("total") or 0)
    if total < 8:
        return 0.0, 100.0, 100.0, []
    accuracy = _as_float(item.get("accuracy"))
    avg_mfe = _as_float(item.get("average_mfe"))
    avg_mae = _as_float(item.get("average_mae"))
    penalty = 0.0
    selection_cap = 100.0
    execution_cap = 100.0
    reasons: list[str] = []
    if accuracy is not None and accuracy < 45.0:
        cut = min(5.0, (45.0 - accuracy) * 0.16)
        penalty += cut
        selection_cap = min(selection_cap, 84.0)
        execution_cap = min(execution_cap, 72.0)
        reasons.append(f"{combo_key} 組合近期勝率 {accuracy:.1f}% 低於 45%，校準降權 {cut:.1f} 分")
    if avg_mfe is not None and avg_mae is not None and avg_mae > avg_mfe:
        penalty += 2.0
        execution_cap = min(execution_cap, 70.0)
        reasons.append(f"{combo_key} 組合 MAE 大於 MFE，降低 execution_score")
    return penalty, selection_cap, execution_cap, reasons


def _validation_combo_key(tag_set: set[str]) -> str | None:
    if {"Liquidity Sweep", "MSS/BOS", "FVG", "OTE"}.issubset(tag_set):
        return "Liquidity Sweep + MSS/BOS + FVG + OTE"
    if {"Liquidity Sweep", "MSS/BOS", "FVG"}.issubset(tag_set):
        return "Liquidity Sweep + MSS/BOS + FVG"
    if "Liquidity Sweep" in tag_set and "FVG" in tag_set and "OTE" not in tag_set:
        return "Sweep + FVG 但無 OTE"
    return None


def _is_broad_validation_tag(tag: str) -> bool:
    return tag in BROAD_VALIDATION_TAGS


def _calibrate_score(score: DirectionScore, ticker: Ticker, price: float) -> None:
    bucket_scores: dict[str, float] = {}
    available_weight = 0.0
    weighted = 0.0
    for bucket, features in BUCKET_FEATURES.items():
        ratio, feature_available = _bucket_ratio(score, features)
        if ratio is None:
            continue
        weight = BUCKET_WEIGHTS[bucket]
        bucket_scores[bucket] = round(ratio, 2)
        available_weight += weight
        weighted += ratio / 100.0 * weight

    if available_weight <= 0:
        score.calibrated_score = 0.0
        score.calibrated_available_max = 0.0
        score.selection_score = 0.0
        score.setup_score = 0.0
        score.execution_score = 0.0
        return

    base = weighted / available_weight * 100.0
    score.calibrated_available_max = min(100.0, available_weight)
    score.bucket_scores = bucket_scores
    score.bucket_weights = {key: BUCKET_WEIGHTS[key] for key in bucket_scores}

    htf = bucket_scores.get("htf_context", 0.0)
    trigger = bucket_scores.get("ltf_confirmation", 0.0)
    entry = bucket_scores.get("entry_location", 0.0)
    risk = bucket_scores.get("risk_plan", 0.0)
    market = bucket_scores.get("market_filter", 0.0)
    selection_cap = 100.0
    execution_cap = 100.0
    execution_penalty = 0.0
    adjustments: list[str] = []

    setup_weighted = htf * 0.32 + trigger * 0.27 + entry * 0.25 + risk * 0.16
    score.setup_score = round(max(0.0, min(100.0, setup_weighted)), 2)

    if htf < 45:
        selection_cap = min(selection_cap, 70.0)
        execution_cap = min(execution_cap, 70.0)
        adjustments.append("HTF 沒有明確掃蕩/POI，最高分限制 70")
    elif htf < 60:
        selection_cap = min(selection_cap, 78.0)
        execution_cap = min(execution_cap, 74.0)
        adjustments.append("HTF bias 不夠乾淨，降低選幣與執行上限")
    if trigger < 45:
        selection_cap = min(selection_cap, 68.0)
        execution_cap = min(execution_cap, 68.0)
        adjustments.append("沒有有效 LTF MSS/BOS，最高分限制 68")
    elif trigger < 65:
        execution_cap = min(execution_cap, 76.0)
        adjustments.append("LTF trigger 不夠完整，execution_score 降級")
    if not score.entry_zone:
        selection_cap = min(selection_cap, 68.0)
        execution_cap = min(execution_cap, 65.0)
        adjustments.append("沒有有效 entry zone，禁止高分執行")
    elif entry < 45:
        selection_cap = min(selection_cap, 70.0)
        execution_cap = min(execution_cap, 68.0)
        adjustments.append("FVG/OTE/OB 入場位置不足，最高分限制 70")
    elif entry < 65:
        execution_cap = min(execution_cap, 76.0)
        adjustments.append("entry quality 尚未達執行門檻")
    if risk < 60:
        selection_cap = min(selection_cap, 74.0)
        execution_cap = min(execution_cap, 74.0)
        adjustments.append("RR 或風控結構不足，最高分限制 74")
    if market < 45:
        selection_cap = min(selection_cap, 80.0)
        execution_cap = min(execution_cap, 72.0)
        adjustments.append("BTC/流動性/波動品質不足，execution_score 降級")

    bonus_ratio = 0.0
    if score.bonus_max_score > 0:
        bonus_ratio = max(0.0, min(1.0, score.bonus_score / score.bonus_max_score))
    bonus = 0.0
    if base >= 58 and htf >= 60 and trigger >= 65 and entry >= 65:
        bonus = min(MAX_CALIBRATED_BONUS, bonus_ratio * MAX_CALIBRATED_BONUS)
    elif bonus_ratio > 0:
        adjustments.append("輔助共振只列入 reasons；核心 bucket 不足時不補分")

    if score.entry_zone:
        distance = _zone_distance_pct(price, score.entry_zone[0], score.entry_zone[1])
        score.entry_distance_pct = round(distance, 4)
        if distance > 5.0:
            selection_cap = min(selection_cap, 70.0)
            execution_cap = min(execution_cap, 35.0)
            adjustments.append(f"現價距 entry zone {distance:.2f}%，標記 missed / expired，不追")
        elif distance > 3.0:
            selection_cap = min(selection_cap, 76.0)
            execution_cap = min(execution_cap, 55.0)
            adjustments.append(f"現價距 entry zone {distance:.2f}%，禁止顯示可以做")
        elif distance > 1.2:
            penalty = min(18.0, 8.0 + (distance - 1.2) * 5.0)
            execution_penalty += penalty
            execution_cap = min(execution_cap, 70.0)
            adjustments.append(f"現價距 entry zone {distance:.2f}%，execution_score 扣 {penalty:.1f} 分")
        elif distance > 0.3:
            penalty = min(6.0, (distance - 0.3) * 6.0)
            execution_penalty += penalty
            adjustments.append(f"現價尚未貼近 entry zone（{distance:.2f}%），等待回補")
    else:
        score.entry_distance_pct = None

    if ticker.quote_volume < 20_000_000:
        execution_penalty += 5.0
        selection_cap = min(selection_cap, 78.0)
        execution_cap = min(execution_cap, 68.0)
        adjustments.append("24h 成交額低於 2,000 萬 USDT，降低實盤執行分")

    atr_pct = _as_float(score.market_metrics.get("atr_pct"))
    volume_ratio = _as_float(score.market_metrics.get("volume_ratio"))
    if atr_pct is not None:
        vol_profile = volatility_profile(ticker.symbol)
        if atr_pct < vol_profile.quiet_atr_pct:
            execution_cap = min(execution_cap, 68.0)
            execution_penalty += 4.0
            adjustments.append(f"ATR%={atr_pct:.2f} 過低，可能缺乏推進")
        elif atr_pct >= vol_profile.extreme_atr_pct:
            execution_cap = min(execution_cap, 62.0)
            execution_penalty += 8.0
            adjustments.append(f"ATR%={atr_pct:.2f} 極端波動，需要重新築底/回測後才可執行")
        elif atr_pct > vol_profile.hot_atr_pct:
            execution_cap = min(execution_cap, 72.0)
            execution_penalty += 4.0
            adjustments.append(f"ATR%={atr_pct:.2f} 偏熱，只能貼近 entry 並等回測確認")
        elif atr_pct > vol_profile.active_high_atr_pct:
            execution_cap = min(execution_cap, 84.0)
            execution_penalty += 1.0
            adjustments.append(f"ATR%={atr_pct:.2f} 屬主動趨勢波動，不視為否決但降低追價容忍")
        elif atr_pct >= vol_profile.active_low_atr_pct and vol_profile.instrument_class in {"altcoin", "large_altcoin"}:
            adjustments.append(f"ATR%={atr_pct:.2f} 符合 {vol_profile.instrument_class} crypto-beta 波動帶，不當作黃金/白銀式過熱")
    if volume_ratio is not None:
        flow_profile = participation_profile(ticker.symbol)
        if volume_ratio >= flow_profile.extreme_volume_ratio:
            execution_cap = min(execution_cap, 66.0)
            execution_penalty += 6.0
            adjustments.append(f"volume spike {volume_ratio:.2f} 倍可能是 blow-off，需等待回測或 orderflow 確認")
        elif volume_ratio > flow_profile.hot_volume_ratio:
            execution_cap = min(execution_cap, 74.0)
            execution_penalty += 3.0
            adjustments.append(f"volume spike {volume_ratio:.2f} 倍偏熱，避免追高/追空")
    if score.market_metrics.get("btc_against"):
        selection_cap = min(selection_cap, 78.0)
        execution_cap = min(execution_cap, 68.0)
        execution_penalty += 6.0
        adjustments.append("BTC 1H 方向明顯反向，alt 交易降級")
    if score.market_metrics.get("btc_overheated"):
        if score.market_metrics.get("btc_against"):
            selection_cap = min(selection_cap, 72.0)
            execution_cap = min(execution_cap, 60.0)
            execution_penalty += 8.0
            adjustments.append("BTC 反向急拉/急跌，禁止追小幣")
        else:
            execution_cap = min(execution_cap, 76.0)
            execution_penalty += 3.0
            adjustments.append("BTC 快速波動，同向單只降級並要求回撤確認")

    score.setup_tags = _setup_tags_from_score(score)
    validation_penalty, validation_selection_cap, validation_execution_cap, validation_reasons = _validation_adjustment_from_stats(score, _load_validation_stats())
    if validation_penalty:
        execution_penalty += validation_penalty
        selection_cap = min(selection_cap, validation_selection_cap)
        execution_cap = min(execution_cap, validation_execution_cap)
    score.validation_adjustments = validation_reasons
    adjustments.extend(validation_reasons)

    selection = min(selection_cap, base + bonus)
    execution = min(execution_cap, selection, score.setup_score) - execution_penalty
    score.selection_score = round(max(0.0, min(100.0, selection)), 2)
    score.calibrated_score = score.selection_score
    score.execution_score = round(max(0.0, min(100.0, execution)), 2)
    score.score_adjustments = adjustments
    return

    cap = 100.0
    adjustments: list[str] = []

    if htf < 35:
        cap = min(cap, 70.0)
        adjustments.append("HTF 背景不足，最高分限制 70")
    if trigger < 45:
        cap = min(cap, 68.0)
        adjustments.append("MSS/BOS 與位移不足，最高分限制 68")
    if entry < 45:
        cap = min(cap, 70.0)
        adjustments.append("FVG/OTE 入場位置不足，最高分限制 70")
    if risk < 50:
        cap = min(cap, 74.0)
        adjustments.append("RR 或止損結構不足，最高分限制 74")
    if market < 35:
        cap = min(cap, 76.0)
        adjustments.append("成交量、BTC 共振或波動品質不足，最高分限制 76")

    bonus_ratio = 0.0
    if score.bonus_max_score > 0:
        bonus_ratio = max(0.0, min(1.0, score.bonus_score / score.bonus_max_score))
    bonus = 0.0
    if base >= 58 and min(htf, trigger, entry) >= 40:
        bonus = min(4.0, bonus_ratio * 4.0)
    elif bonus_ratio > 0:
        adjustments.append("額外劇本只當共振，不用來補足缺少的核心條件")

    distance_penalty = 0.0
    if score.entry_zone:
        distance = _zone_distance_pct(price, score.entry_zone[0], score.entry_zone[1])
        if distance > 3.0:
            cap = min(cap, 65.0)
            adjustments.append(f"現價離入場區 {distance:.2f}%，視為追價風險")
        elif distance > 1.2:
            distance_penalty = min(6.0, (distance - 1.2) * 2.5)
            adjustments.append(f"現價離入場區 {distance:.2f}%，扣 {distance_penalty:.1f} 分")

    volume_penalty = 0.0
    if ticker.quote_volume < 20_000_000:
        volume_penalty = 4.0
        adjustments.append("24h 成交額低於短線篩選門檻，扣 4 分")

    final = min(cap, base + bonus - distance_penalty - volume_penalty)
    score.calibrated_score = round(max(0.0, min(100.0, final)), 2)
    score.score_adjustments = adjustments


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
    atr_pct = _atr_pct(price, local_atr)
    default_target_rr = _intraday_default_target_rr(atr_pct)
    min_liquidity_rr = _intraday_min_liquidity_rr(atr_pct)
    buy_side, sell_side = nearest_liquidity_targets(candles, price)
    swings = swing_points(candles, left=2, right=2)
    if direction == "long":
        stop = _bounded_structural_stop(direction, candles, swings, entry, zone_low, zone_high, local_atr, atr_pct)
        risk = max(entry - stop, price * 0.001)
        liquidity_target = buy_side if buy_side and buy_side > entry else None
        fallback_target = liquidity_target if liquidity_target and liquidity_target >= entry + risk * min_liquidity_rr else entry + risk * default_target_rr
    else:
        stop = _bounded_structural_stop(direction, candles, swings, entry, zone_low, zone_high, local_atr, atr_pct)
        risk = max(stop - entry, price * 0.001)
        liquidity_target = sell_side if sell_side and sell_side < entry else None
        fallback_target = liquidity_target if liquidity_target and liquidity_target <= entry - risk * min_liquidity_rr else entry - risk * default_target_rr
    target_levels = _profit_reference_levels(direction, candles, swings, entry, liquidity_target, local_atr)
    take_profits = _build_take_profits(direction, entry, stop, fallback_target, liquidity_target, target_levels, local_atr)
    target = float(take_profits[1]["price"]) if len(take_profits) > 1 else fallback_target
    reward = max(target - entry, 0.0) if direction == "long" else max(entry - target, 0.0)
    rr = reward / max(risk, 1e-12)
    return (zone_low, zone_high), stop, target, rr, take_profits


def _build_take_profits(
    direction: str,
    entry: float,
    stop: float,
    target: float,
    liquidity_target: float | None,
    target_levels: list[float] | None = None,
    local_atr: float = 0.0,
) -> list[dict[str, float | str]]:
    if direction == "long":
        risk = max(entry - stop, abs(entry) * 0.001)
        tp1 = _select_take_profit(direction, entry, risk, target_levels or [], 0.85, 1.35, 1.0, local_atr)
        tp2 = _select_take_profit(direction, entry, risk, target_levels or [], 1.45, 2.35, _level_rr(direction, entry, risk, target), local_atr)
        if tp2 <= tp1:
            tp2 = _rr_to_level(direction, entry, risk, max(1.45, _level_rr(direction, entry, risk, target)))
        tp3 = _select_take_profit(direction, entry, risk, target_levels or [], max(2.1, _level_rr(direction, entry, risk, tp2) + 0.45), 3.4, max(2.4, _level_rr(direction, entry, risk, tp2) + 0.8), local_atr)
        if tp3 <= tp2:
            tp3 = _rr_to_level(direction, entry, risk, _level_rr(direction, entry, risk, tp2) + 0.7)
        return [
            {"name": "TP1", "price": tp1, "rr": (tp1 - entry) / risk, "portion_pct": 30.0, "note": "第一阻力/內部流動性前先減倉，降低回吐壓力"},
            {"name": "TP2", "price": tp2, "rr": (tp2 - entry) / risk, "portion_pct": 45.0, "note": "主目標，對應主要買方流動性或合理 R 區間"},
            {"name": "TP3", "price": tp3, "rr": (tp3 - entry) / risk, "portion_pct": 25.0, "note": "延伸目標，只在放量延續且 BTC 未反向時保留"},
        ]
    risk = max(stop - entry, abs(entry) * 0.001)
    tp1 = _select_take_profit(direction, entry, risk, target_levels or [], 0.85, 1.35, 1.0, local_atr)
    tp2 = _select_take_profit(direction, entry, risk, target_levels or [], 1.45, 2.35, _level_rr(direction, entry, risk, target), local_atr)
    if tp2 >= tp1:
        tp2 = _rr_to_level(direction, entry, risk, max(1.45, _level_rr(direction, entry, risk, target)))
    tp3 = _select_take_profit(direction, entry, risk, target_levels or [], max(2.1, _level_rr(direction, entry, risk, tp2) + 0.45), 3.4, max(2.4, _level_rr(direction, entry, risk, tp2) + 0.8), local_atr)
    if tp3 >= tp2:
        tp3 = _rr_to_level(direction, entry, risk, _level_rr(direction, entry, risk, tp2) + 0.7)
    return [
        {"name": "TP1", "price": tp1, "rr": (entry - tp1) / risk, "portion_pct": 30.0, "note": "第一支撐/內部流動性前先減倉，降低回吐壓力"},
        {"name": "TP2", "price": tp2, "rr": (entry - tp2) / risk, "portion_pct": 45.0, "note": "主目標，對應主要賣方流動性或合理 R 區間"},
        {"name": "TP3", "price": tp3, "rr": (entry - tp3) / risk, "portion_pct": 25.0, "note": "延伸目標，只在放量延續且 BTC 未反向時保留"},
    ]


def _select_take_profit(
    direction: str,
    entry: float,
    risk: float,
    levels: list[float],
    min_rr: float,
    max_rr: float,
    fallback_rr: float,
    local_atr: float,
) -> float:
    fallback_rr = max(min_rr, min(max_rr, fallback_rr if fallback_rr > 0 else min_rr))
    for level in levels:
        price = _front_run_target(direction, level, local_atr)
        rr = _level_rr(direction, entry, risk, price)
        if min_rr <= rr <= max_rr:
            return price
    return _rr_to_level(direction, entry, risk, fallback_rr)


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
            _note(score, "HTF 尚未出現清楚掃高/掃低後收回")
    else:
        _skip(score, "liquidity_sweep", "缺少 1H/4H K 線")

    poi_points = 0.0
    poi_notes: list[str] = []
    if candles_4h:
        position, low, high = price_position_in_range(candles_4h, lookback=120)
        if direction == "long":
            if position <= 0.5:
                poi_points += 5.0 if position <= 0.35 else 3.5
                poi_notes.append(f"4H 位於折價區 position={position:.2f}")
        else:
            if position >= 0.5:
                poi_points += 5.0 if position >= 0.65 else 3.5
                poi_notes.append(f"4H 位於溢價區 position={position:.2f}")
        htf_fvg = recent_relevant_fvg(candles_4h, direction, max_age=80)
        if htf_fvg and _price_near_zone(price, htf_fvg.lower, htf_fvg.upper, tolerance_pct=0.75):
            poi_points += 4.5
            poi_notes.append("接近 4H FVG/失衡區")
        ob = order_block(candles_4h, direction, before_index=(sweep_4h.index if sweep_4h else None), lookback=70)
        if ob and _price_near_zone(price, ob.lower, ob.upper, tolerance_pct=0.75):
            poi_points += 4.5
            poi_notes.append("接近 4H 訂單塊 POI")
        _add(score, "htf_poi", poi_points, "、".join(poi_notes) if poi_notes else "")
    else:
        _skip(score, "htf_poi", "缺少 4H K 線，無法判斷 HTF POI / 折價溢價")

    ltf_sweep = detect_liquidity_sweep(candles_15m, direction, lookback=100) if candles_15m else None
    after_index = ltf_sweep.index if ltf_sweep else None
    structure = detect_structure_break(candles_15m, direction, after_index=after_index, lookback=100) if candles_15m else None
    if candles_15m:
        if structure:
            base = 10.0 if structure.kind == "BOS" else 12.0
            if ltf_sweep:
                base += 2.5
            if sweep:
                base += 1.5
            base = min(WEIGHTS["mss_bos"], base)
            _add(score, "mss_bos", base, f"15m {structure.kind} 收盤突破 {structure.level:g}")
        else:
            _add(score, "mss_bos", 0.0, "")
            _note(score, "15m 尚未確認 MSS/BOS")
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
            points = 7.0 + (2.5 if displacement.has_fvg else 0.0)
            if structure and displacement.index >= structure.index:
                points += 1.4
            if ltf_sweep and displacement.index >= ltf_sweep.index:
                points += 1.1
            points = min(WEIGHTS["displacement"], points * min(1.2, displacement.body_atr / 1.15))
            fvg_text = "並留下 FVG" if displacement.has_fvg else "但 FVG 不明顯"
            _add(score, "displacement", points, f"15m 位移 K body/ATR={displacement.body_atr:.2f} {fvg_text}")
        else:
            _add(score, "displacement", 0.0, "")
            _note(score, "尚未看到明確大實體單向位移")
    else:
        _skip(score, "displacement", "缺少 15m K 線")

    ote = ote_zone(candles_15m or candles_1h, direction, lookback=100) if (candles_15m or candles_1h) else None
    fvg_anchor_index = structure.index if structure else after_index
    fvg_15m = recent_relevant_fvg(candles_15m, direction, max_age=110, anchor_index=fvg_anchor_index) if candles_15m else None
    fvg_5m = recent_relevant_fvg(candles_5m, direction, max_age=160) if candles_5m else None
    selected_fvg = fvg_15m or fvg_5m
    fvg_zone: tuple[float, float] | None = None
    if candles_15m or candles_5m:
        if selected_fvg:
            fvg_zone = (selected_fvg.lower, selected_fvg.upper)
            points = 7.0
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
            _note(score, "近期找不到方向一致且未完全填補的 FVG")
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
            _add(score, "ote", min(WEIGHTS["ote"], 8.5 + overlap_bonus), f"價格位於 OTE 0.62-0.79 回撤區 ({retracement:.2f}){overlap_note}")
        elif 0.50 <= retracement <= 0.86:
            _add(score, "ote", min(WEIGHTS["ote"], 5.0 + overlap_bonus), f"價格接近 OTE 區但未進核心帶 ({retracement:.2f}){overlap_note}")
        else:
            _add(score, "ote", 0.0, "")
    else:
        _skip(score, "ote", "缺少 15m/1H K 線，無法計算 OTE")

    if candles_4h and len(candles_4h) >= 60:
        trendline = trendline_breakout(candles_4h, direction)
        if trendline.get("hit"):
            touches = int(trendline.get("touches") or 0)
            points = 2.8 + min(0.8, max(0, touches - 2) * 0.4)
            if trendline.get("risk") == "low":
                points += 0.4
            _bonus(score, "trendline", points, f"4H 趨勢線破位，觸碰 {touches} 次，風險距離={trendline.get('risk')}")
        elif int(trendline.get("touches") or 0) >= 2:
            _bonus(score, "trendline", 1.0, f"4H 有 {trendline.get('touches')} 觸點趨勢線，但尚未破位")
        else:
            _inactive(score, "trendline", "未形成有效 2-3 觸點趨勢線破位")
    else:
        _inactive(score, "trendline", "4H 歷史不足，暫不評估趨勢線劇本")

    amd_candles = candles_15m or candles_5m
    if amd_candles and len(amd_candles) >= 90:
        amd = amd_signal(amd_candles, direction)
        amd_points = float(amd.get("score") or 0.0) * OPTIONAL_BONUS_WEIGHTS["amd"]
        amd_phase = _label(AMD_PHASE_LABELS, amd.get("phase"))
        if amd_points > 0:
            _bonus(score, "amd", amd_points, f"AMD：{amd_phase}")
        else:
            _inactive(score, "amd", f"AMD 未進入可交易階段：{amd_phase}")
    else:
        _inactive(score, "amd", "15m/5m 歷史不足，暫不評估 AMD 劇本")

    if candles_5m and len(candles_5m) >= 360:
        nexus = nexus_signal(candles_5m, direction)
        nexus_points = float(nexus.get("score") or 0.0) * OPTIONAL_BONUS_WEIGHTS["nexus"]
        nexus_reason = _label(NEXUS_REASON_LABELS, nexus.get("reason"))
        if nexus_points > 0:
            _bonus(score, "nexus", nexus_points, f"Nexus/Silver Bullet：{nexus_reason}")
        else:
            _inactive(score, "nexus", f"Nexus/Silver Bullet 未觸發：{nexus_reason}")
    else:
        _inactive(score, "nexus", "5m 歷史不足，暫不評估 Nexus/Silver Bullet 劇本")

    if candles_15m or candles_1h:
        entry_zone, stop, target, rr, take_profits = _risk_setup(direction, candles_15m, candles_1h, price, fvg_zone, ote)
        score.entry_zone = entry_zone
        score.stop = stop
        score.target = target
        score.take_profits = take_profits
        score.rr = rr
        zone_dist = _zone_distance_pct(price, entry_zone[0], entry_zone[1])
        chase_penalty = 2.0 if zone_dist > 1.2 else 0.0
        if rr >= 2.0:
            _add(score, "risk_reward", max(0.0, 12.0 - chase_penalty), f"以最近流動性目標估算 RR={rr:.2f}")
        elif rr >= 1.5:
            _add(score, "risk_reward", max(0.0, 8.0 - chase_penalty), f"RR={rr:.2f}，可觀察但不是最漂亮")
        elif rr >= 1.1:
            _add(score, "risk_reward", max(0.0, 3.0 - chase_penalty), f"RR={rr:.2f} 偏低")
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
        quality_points += 3.0
        quality_notes.append("24h 成交額 > 1 億 USDT")
    elif ticker.quote_volume >= 20_000_000:
        quality_points += 2.0
        quality_notes.append("24h 成交額 > 2,000 萬 USDT")
    else:
        _warn(score, "24h 成交額偏低，滑點與假突破風險較高")

    quality_max = 3.0
    if btc_1h and candles_1h and ticker.symbol != "BTCUSDT":
        quality_max = WEIGHTS["market_quality"]
        corr = correlation(returns(candles_1h, 80), returns(btc_1h, 80))
        btc_trend = btc_1h[-1].close - btc_1h[-24].close if len(btc_1h) >= 24 else 0.0
        aligned = (direction == "long" and btc_trend >= 0) or (direction == "short" and btc_trend <= 0)
        if corr >= 0.45 and aligned:
            quality_points += 3.0
            quality_notes.append(f"與 BTC 相關性 {corr:.2f} 且方向一致")
        elif corr >= 0.45:
            quality_points += 1.2
            quality_notes.append(f"與 BTC 相關性 {corr:.2f}，但 BTC 方向不完全同向")
    elif ticker.symbol == "BTCUSDT":
        quality_max = WEIGHTS["market_quality"]
        quality_points += 3.0
        quality_notes.append("BTC 本身作為市場基準")
    else:
        _warn(score, "BTC 1H 相關性資料缺失，市場品質只用成交額評估")

    market_candles = candles_15m or candles_1h
    if market_candles and len(market_candles) >= 40:
        local_atr = atr(market_candles, 14) or 0.0
        atr_pct = local_atr / max(price, 1e-12) * 100.0
        vol_profile = volatility_profile(ticker.symbol)
        if vol_profile.active_low_atr_pct <= atr_pct <= vol_profile.active_high_atr_pct:
            quality_notes.append(f"短線波動率 {atr_pct:.2f}% 位於可交易區")
        elif atr_pct > vol_profile.hot_atr_pct:
            _warn(score, f"短線波動率 {atr_pct:.2f}% 過熱，容易掃損")
        elif atr_pct > vol_profile.active_high_atr_pct:
            quality_notes.append(f"短線波動率 {atr_pct:.2f}% 偏活躍，需貼近 entry")
        elif 0 < atr_pct < vol_profile.quiet_atr_pct:
            _warn(score, f"短線波動率 {atr_pct:.2f}% 偏低，可能缺乏推進")

        recent_volume = sum(c.volume for c in market_candles[-12:])
        base_sample = market_candles[-60:-12]
        base_volume = sum(c.volume for c in base_sample) / max(len(base_sample), 1) * 12 if base_sample else 0.0
        if base_volume > 0:
            volume_ratio = recent_volume / base_volume
            flow_profile = participation_profile(ticker.symbol)
            if flow_profile.active_low_volume_ratio <= volume_ratio <= flow_profile.active_high_volume_ratio:
                quality_notes.append(f"近期量能擴張 {volume_ratio:.2f} 倍")
            elif flow_profile.active_high_volume_ratio < volume_ratio <= flow_profile.warm_high_volume_ratio:
                quality_points += 0.4
                quality_notes.append(f"volume ratio={volume_ratio:.2f} hot but tradable")
            elif volume_ratio > flow_profile.hot_volume_ratio:
                _warn(score, f"近期量能暴衝 {volume_ratio:.2f} 倍，注意追高/追空風險")
    else:
        _warn(score, "短線波動率與量能資料不足，市場品質保守評估")

    quality_points = min(quality_points, quality_max)
    _add(score, "market_quality", quality_points, "、".join(quality_notes) if quality_notes else "", weight_override=quality_max)
    return score


def _evaluate_direction_v2(
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
    price_action = analyze_price_action(direction, price, candles_4h, candles_1h, candles_15m, candles_5m)
    score.market_metrics.update(price_action.metrics)
    for warning in price_action.warnings:
        _warn(score, warning)
    _add(
        score,
        "key_level",
        price_action.key_level_score / 100.0 * WEIGHTS["key_level"],
        "；".join(price_action.notes[:3]) if price_action.notes else "關鍵位背景不足，等待更清楚支撐/壓力",
    )
    _add(
        score,
        "price_action",
        price_action.price_action_score / 100.0 * WEIGHTS["price_action"],
        "；".join(price_action.notes[3:6]) if len(price_action.notes) > 3 else "短線 K 線尚未給出強確認",
    )
    _add(
        score,
        "breakout_quality",
        price_action.breakout_score / 100.0 * WEIGHTS["breakout_quality"],
        "；".join(price_action.notes[6:9]) if len(price_action.notes) > 6 else "突破品質尚待收盤與量能確認",
    )

    sweep_1h = detect_liquidity_sweep(candles_1h, direction, lookback=90) if candles_1h else None
    sweep_4h = detect_liquidity_sweep(candles_4h, direction, lookback=70) if candles_4h else None
    htf_sweep = sweep_1h or sweep_4h
    if candles_1h or candles_4h:
        if htf_sweep:
            source = "1H" if sweep_1h else "4H"
            points = WEIGHTS["liquidity_sweep"] * min(1.0, 0.55 + htf_sweep.strength / 2.4)
            if sweep_1h and sweep_4h:
                points = WEIGHTS["liquidity_sweep"]
                source = "1H + 4H"
            _add(score, "liquidity_sweep", points, f"{source} 有效流動性掃蕩後收回，符合 HTF bias")
        else:
            _add(score, "liquidity_sweep", 0.0, "")
            _note(score, "HTF 尚未出現明確掃高/掃低後收回")
    else:
        _skip(score, "liquidity_sweep", "缺少 1H/4H K 線")

    poi_points = 0.0
    poi_notes: list[str] = []
    if candles_4h:
        position, _, _ = price_position_in_range(candles_4h, lookback=120)
        if direction == "long" and position <= 0.45:
            poi_points += 5.0 if position <= 0.35 else 3.5
            poi_notes.append(f"4H 位於折價區 position={position:.2f}")
        if direction == "short" and position >= 0.55:
            poi_points += 5.0 if position >= 0.65 else 3.5
            poi_notes.append(f"4H 位於溢價區 position={position:.2f}")
        htf_fvg = recent_relevant_fvg(
            candles_4h,
            direction,
            max_age=80,
            anchor_index=sweep_4h.index if sweep_4h else None,
            max_distance_atr=1.5,
        )
        if htf_fvg and _price_near_zone(price, htf_fvg.lower, htf_fvg.upper, tolerance_pct=0.75):
            poi_points += 4.0
            poi_notes.append("接近未完全填補的 4H FVG")
        htf_ob = order_block(candles_4h, direction, before_index=(sweep_4h.index if sweep_4h else None), lookback=70)
        if htf_ob and _price_near_zone(price, htf_ob.lower, htf_ob.upper, tolerance_pct=0.75):
            poi_points += 4.0
            poi_notes.append("接近 4H order block / POI")
        _add(score, "htf_poi", min(WEIGHTS["htf_poi"], poi_points), "、".join(poi_notes) if poi_notes else "")
    else:
        _skip(score, "htf_poi", "缺少 4H K 線，無法判斷 HTF POI")

    ltf_sweep = detect_liquidity_sweep(candles_15m, direction, lookback=100) if candles_15m else None
    after_index = ltf_sweep.index if ltf_sweep else None
    structure = detect_structure_break(candles_15m, direction, after_index=after_index, lookback=100) if candles_15m else None
    if candles_15m:
        if structure:
            points = 12.0 if structure.kind == "MSS" else 10.0
            if ltf_sweep:
                points += 2.0
            if htf_sweep:
                points += 2.0
            _add(score, "mss_bos", min(WEIGHTS["mss_bos"], points), f"15m {structure.kind} 確認，突破 level={structure.level:g}")
        else:
            _add(score, "mss_bos", 0.0, "")
            _note(score, "15m 尚未確認 MSS/BOS")
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
            points = 7.0 + (2.5 if displacement.has_fvg else 0.0)
            if structure and displacement.index >= structure.index:
                points += 1.5
            if ltf_sweep and displacement.index >= ltf_sweep.index:
                points += 1.0
            points *= min(1.2, displacement.body_atr / 1.15)
            _add(score, "displacement", min(WEIGHTS["displacement"], points), f"15m displacement body/ATR={displacement.body_atr:.2f}")
        else:
            _add(score, "displacement", 0.0, "")
            _note(score, "尚未看到明確大實體位移")
    else:
        _skip(score, "displacement", "缺少 15m K 線")

    ote = ote_zone(candles_15m or candles_1h, direction, lookback=100) if (candles_15m or candles_1h) else None
    fvg_anchor = structure.index if structure else after_index
    fvg_15m = recent_relevant_fvg(candles_15m, direction, max_age=110, anchor_index=fvg_anchor) if candles_15m else None
    fvg_5m = recent_relevant_fvg(candles_5m, direction, max_age=160, max_distance_atr=1.2) if candles_5m else None
    selected_fvg = fvg_15m or fvg_5m
    fvg_zone: tuple[float, float] | None = None
    ob_zone: tuple[float, float] | None = None
    if selected_fvg:
        raw_fvg_zone = (selected_fvg.lower, selected_fvg.upper)
        local_entry_candles = candles_15m if fvg_15m else candles_5m
        local_atr = atr(local_entry_candles, 14) if local_entry_candles else 0.0
        distance_atr = _zone_distance_pct(price, selected_fvg.lower, selected_fvg.upper) / 100.0 * price / max(local_atr, price * 0.001)
        if candles_15m:
            ob = order_block(candles_15m, direction, before_index=selected_fvg.index, lookback=45)
            if ob and zone_overlap(selected_fvg.lower, selected_fvg.upper, ob.lower, ob.upper):
                ob_zone = (ob.lower, ob.upper)
        ote_overlap_zone = (ote[0], ote[1]) if ote and zone_overlap(selected_fvg.lower, selected_fvg.upper, ote[0], ote[1]) else None
        triple_zone = _intersect_zones(raw_fvg_zone, ote_overlap_zone, ob_zone)
        double_zone = _intersect_zones(raw_fvg_zone, ote_overlap_zone) or _intersect_zones(raw_fvg_zone, ob_zone)
        if triple_zone:
            fvg_zone = triple_zone
            points = 13.0
            note = "FVG ∩ OTE ∩ OB 三重重疊，entry zone 品質高"
        elif double_zone:
            fvg_zone = double_zone
            points = 10.5
            note = "FVG 與 OTE/OB 重疊，entry zone 可觀察"
        else:
            fvg_zone = _fvg_mid_entry_zone(direction, selected_fvg.lower, selected_fvg.upper)
            points = 6.5
            note = "未形成 OTE/OB 重疊，只採 FVG midpoint 到 50% 半區"
        if selected_fvg.tapped and not selected_fvg.filled:
            points += 1.0
        if structure and abs(selected_fvg.index - structure.index) <= 4:
            points += 1.0
        if distance_atr > 1.0:
            points = min(points, 6.0)
            note += f"，但距離 FVG 約 {distance_atr:.2f} ATR，只當保守入場"
        _add(score, "fvg", min(WEIGHTS["fvg"], points), f"{selected_fvg.start_time.isoformat()} {direction} FVG：{note}")
    elif candles_15m or candles_5m:
        _add(score, "fvg", 0.0, "")
        _note(score, "近期沒有未填補且靠近 sweep/MSS/BOS 的有效 FVG")
    else:
        _skip(score, "fvg", "缺少 15m/5m K 線")

    if ote:
        zone_low, zone_high, retracement = ote
        overlap = bool(fvg_zone and zone_overlap(fvg_zone[0], fvg_zone[1], zone_low, zone_high))
        if 0.62 <= retracement <= 0.79:
            points = 8.0 + (2.0 if overlap else 0.0)
            _add(score, "ote", min(WEIGHTS["ote"], points), f"有效 impulse leg OTE 0.62-0.79，retracement={retracement:.2f}")
        elif 0.50 <= retracement <= 0.86:
            points = 4.5 + (1.0 if overlap else 0.0)
            _add(score, "ote", min(WEIGHTS["ote"], points), f"接近 OTE 但未在核心甜蜜區，retracement={retracement:.2f}")
        else:
            _add(score, "ote", 0.0, "")
            _note(score, f"有效 impulse leg 已有，但現價 retracement={retracement:.2f} 不在 OTE")
    else:
        _add(score, "ote", 0.0, "")
        _note(score, "沒有 sweep → MSS/BOS → displacement 的有效 OTE leg")

    if candles_4h and len(candles_4h) >= 60:
        trendline = trendline_breakout(candles_4h, direction)
        if trendline.get("hit"):
            touches = int(trendline.get("touches") or 0)
            points = min(OPTIONAL_BONUS_WEIGHTS["trendline"], 1.2 + min(0.8, max(0, touches - 2) * 0.4))
            _bonus(score, "trendline", points, f"4H trendline break，touches={touches}")
        elif int(trendline.get("touches") or 0) >= 2:
            _bonus(score, "trendline", 0.5, f"4H 有 {trendline.get('touches')} 個 trendline touches，尚未破位")
        else:
            _inactive(score, "trendline", "沒有有效 trendline 共振")
    else:
        _inactive(score, "trendline", "4H 歷史不足，暫不評估 trendline")

    amd_candles = candles_15m or candles_5m
    if amd_candles and len(amd_candles) >= 90:
        amd = amd_signal(amd_candles, direction)
        amd_points = float(amd.get("score") or 0.0) * OPTIONAL_BONUS_WEIGHTS["amd"]
        if amd_points > 0:
            _bonus(score, "amd", amd_points, f"AMD phase={amd.get('phase')}")
        else:
            _inactive(score, "amd", f"AMD 尚未形成可交易階段：{amd.get('phase')}")
    else:
        _inactive(score, "amd", "15m/5m 歷史不足，暫不評估 AMD")

    if candles_5m and len(candles_5m) >= 360:
        nexus = nexus_signal(candles_5m, direction)
        nexus_points = float(nexus.get("score") or 0.0) * OPTIONAL_BONUS_WEIGHTS["nexus"]
        if nexus_points > 0:
            _bonus(score, "nexus", nexus_points, f"Nexus/Silver Bullet：{nexus.get('reason')}")
        else:
            _inactive(score, "nexus", f"Nexus 未觸發：{nexus.get('reason')}")
    else:
        _inactive(score, "nexus", "5m 歷史不足，暫不評估 Nexus")

    if candles_15m or candles_1h:
        entry_source = fvg_zone or ((ote[0], ote[1]) if ote else None)
        entry_zone, stop, target, rr, take_profits = _risk_setup(direction, candles_15m, candles_1h, price, entry_source, ote)
        score.entry_zone = entry_zone
        score.stop = stop
        score.target = target
        score.take_profits = take_profits
        score.rr = rr
        zone_dist = _zone_distance_pct(price, entry_zone[0], entry_zone[1])
        risk_candles = candles_15m or candles_1h
        risk_atr = atr(risk_candles, 14) if risk_candles else None
        risk_atr_pct = _atr_pct(price, risk_atr or price * 0.004)
        if rr >= 2.0:
            points = 12.0
            reason = f"RR={rr:.2f}，符合實盤風報比"
        elif rr >= 1.65:
            points = 10.0
            reason = f"RR={rr:.2f}，達日內短線標準"
        elif rr >= 1.45 and risk_atr_pct >= 1.0:
            points = 8.0
            reason = f"RR={rr:.2f}，高波動日內 scalp 可接受，但需嚴格分批出場"
        elif rr >= 1.25:
            points = 4.0
            reason = f"RR={rr:.2f} 偏低，只能列觀察"
        else:
            points = 0.0
            reason = ""
            _warn(score, f"RR={rr:.2f} 不足，風報比不適合執行")
        if zone_dist > 1.2:
            points = max(0.0, points - 3.0)
            _warn(score, f"現價距 entry zone {zone_dist:.2f}%，不追價，等回補")
        _add(score, "risk_reward", points, reason)
    else:
        _skip(score, "risk_reward", "缺少 15m/1H K 線，無法估算 entry/SL/TP")

    quality_points = 0.0
    quality_notes: list[str] = []
    quality_max = WEIGHTS["market_quality"]
    if ticker.quote_volume >= 100_000_000:
        quality_points += 2.0
        quality_notes.append("24h 成交額 > 1 億 USDT")
    elif ticker.quote_volume >= 20_000_000:
        quality_points += 1.2
        quality_notes.append("24h 成交額 > 2,000 萬 USDT")
    else:
        _warn(score, "24h 成交額偏低，降低 execution_score")

    if btc_1h and len(btc_1h) >= 24 and candles_1h and ticker.symbol != "BTCUSDT":
        corr = correlation(returns(candles_1h, 80), returns(btc_1h, 80))
        btc_trend_pct = (btc_1h[-1].close - btc_1h[-24].close) / max(btc_1h[-24].close, 1e-12) * 100.0
        btc_fast_pct = (btc_1h[-1].close - btc_1h[-4].close) / max(btc_1h[-4].close, 1e-12) * 100.0 if len(btc_1h) >= 4 else 0.0
        aligned = (direction == "long" and btc_trend_pct >= 0) or (direction == "short" and btc_trend_pct <= 0)
        score.market_metrics["btc_corr"] = round(corr, 4)
        score.market_metrics["btc_trend_pct"] = round(btc_trend_pct, 4)
        score.market_metrics["btc_fast_pct"] = round(btc_fast_pct, 4)
        score.market_metrics["btc_against"] = bool(corr >= 0.45 and not aligned and abs(btc_trend_pct) >= 0.8)
        score.market_metrics["btc_overheated"] = bool(abs(btc_fast_pct) >= 2.2)
        if corr >= 0.45 and aligned:
            quality_points += 2.0
            quality_notes.append(f"BTC correlation={corr:.2f} 且方向一致")
        elif corr >= 0.45:
            quality_points += 0.3
            _warn(score, f"BTC 1H trend={btc_trend_pct:.2f}% 與 {direction} 反向，alt 交易降級")
        if abs(btc_fast_pct) >= 2.2:
            _warn(score, f"BTC 近 4H 快速波動 {btc_fast_pct:.2f}%，避免追小幣")
    elif ticker.symbol == "BTCUSDT":
        quality_points += 2.0
        quality_notes.append("BTC 本身作為市場基準")
    else:
        _warn(score, "缺少 BTC 1H context，市場 filter 保守處理")

    market_candles = candles_15m or candles_1h
    if market_candles and len(market_candles) >= 40:
        local_atr = atr(market_candles, 14) or 0.0
        atr_pct = local_atr / max(price, 1e-12) * 100.0
        score.market_metrics["atr_pct"] = round(atr_pct, 4)
        vol_profile = volatility_profile(ticker.symbol)
        if vol_profile.active_low_atr_pct <= atr_pct <= vol_profile.active_high_atr_pct:
            quality_points += 1.0
            quality_notes.append(f"ATR%={atr_pct:.2f} 在可交易區")
        elif atr_pct > vol_profile.hot_atr_pct:
            _warn(score, f"ATR%={atr_pct:.2f} 過熱，禁止追價")
        elif atr_pct > vol_profile.active_high_atr_pct:
            quality_points += 0.4
            quality_notes.append(f"ATR%={atr_pct:.2f} 偏活躍，僅降低追價容忍")
        elif atr_pct > 0:
            _warn(score, f"ATR%={atr_pct:.2f} 偏低，可能缺乏推進")
        recent_volume = sum(c.volume for c in market_candles[-12:])
        base_sample = market_candles[-60:-12]
        base_volume = sum(c.volume for c in base_sample) / max(len(base_sample), 1) * 12 if base_sample else 0.0
        if base_volume > 0:
            volume_ratio = recent_volume / base_volume
            score.market_metrics["volume_ratio"] = round(volume_ratio, 4)
            flow_profile = participation_profile(ticker.symbol)
            if flow_profile.active_low_volume_ratio <= volume_ratio <= flow_profile.active_high_volume_ratio:
                quality_points += 1.0
                quality_notes.append(f"volume ratio={volume_ratio:.2f}")
            elif volume_ratio > flow_profile.hot_volume_ratio:
                _warn(score, f"volume spike={volume_ratio:.2f} 過熱，避免追高/追空")
    else:
        _warn(score, "短線 ATR/volume 資料不足，市場 filter 保守處理")

    _add(score, "market_quality", min(quality_max, quality_points), "、".join(quality_notes) if quality_notes else "", weight_override=quality_max)
    _calibrate_score(score, ticker, price)
    return score


def _select_direction(long: DirectionScore, short: DirectionScore) -> tuple[str, float, float, str]:
    clean_long_score = _effective_score(long)
    clean_short_score = _effective_score(short)
    clean_gap = abs(clean_long_score - clean_short_score)
    clean_selected = max(clean_long_score, clean_short_score)
    if clean_selected < 52:
        return "neutral", clean_selected, clean_gap, "多空分數都低於 52，維持觀察"
    if clean_gap < MIN_DIRECTION_GAP:
        return "neutral", clean_selected, clean_gap, f"多空分差只有 {clean_gap:.1f} 分，低於 {MIN_DIRECTION_GAP:.0f}，禁止硬選方向"
    if clean_gap < CONFLICT_DIRECTION_GAP and _setup_is_complete(long) and _setup_is_complete(short):
        return "neutral", clean_selected, clean_gap, f"多空分差 {clean_gap:.1f} 分且兩邊 setup 都完整，標記 direction_conflict"
    if clean_long_score > clean_short_score:
        return "long", clean_long_score, clean_gap, ""
    if clean_short_score > clean_long_score:
        return "short", clean_short_score, clean_gap, ""
    return "neutral", clean_selected, clean_gap, "多空同分，維持 neutral"

    long_score = _effective_score(long)
    short_score = _effective_score(short)
    score_gap = abs(long_score - short_score)
    selected = max(long_score, short_score)
    conflict = ""
    if selected < 52:
        return "neutral", selected, score_gap, "多空分數皆不足 52，維持觀察"
    if score_gap < MIN_DIRECTION_GAP:
        return "neutral", selected, score_gap, f"多空分差只有 {score_gap:.1f} 分，小於 {MIN_DIRECTION_GAP:.0f}，禁止硬選方向"
    if score_gap < CONFLICT_DIRECTION_GAP and _setup_is_complete(long) and _setup_is_complete(short):
        conflict = f"多空分差 {score_gap:.1f} 分且兩邊 setup 都完整，標記 direction_conflict"
        return "neutral", selected, score_gap, conflict
    if long_score > short_score:
        return "long", long_score, score_gap, conflict
    if short_score > long_score:
        return "short", short_score, score_gap, conflict
    return "neutral", selected, score_gap, "多空同分，維持 neutral"


def finalize_report_scores(report: SymbolReport) -> None:
    ticker = Ticker(report.symbol, report.price, report.quote_volume_24h, report.change_pct_24h)
    _calibrate_score(report.long, ticker, report.price)
    _calibrate_score(report.short, ticker, report.price)
    direction, selected, score_gap, conflict = _select_direction(report.long, report.short)
    report.selected_direction = direction
    report.score = round(selected, 2)
    report.metadata["score_gap"] = round(score_gap, 2)
    report.metadata["long_selection_score"] = _effective_score(report.long)
    report.metadata["short_selection_score"] = _effective_score(report.short)
    if conflict:
        report.metadata["direction_conflict"] = conflict
    elif "direction_conflict" in report.metadata:
        report.metadata.pop("direction_conflict", None)


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

    long = _evaluate_direction_v2("long", ticker, candles_by_tf, btc_1h)
    short = _evaluate_direction_v2("short", ticker, candles_by_tf, btc_1h)
    long_score = long.normalized
    short_score = short.normalized
    score_gap = abs(long_score - short_score)
    if long_score > short_score:
        direction = "long"
        selected = long_score
    elif short_score > long_score:
        direction = "short"
        selected = short_score
    else:
        direction = "neutral"
        selected = long_score

    if max(long_score, short_score) < 52:
        direction = "neutral"
    elif score_gap < 4.0 and max(long_score, short_score) < 82:
        direction = "neutral"

    report = SymbolReport(
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
    if direction == "neutral" and score_gap < 4.0:
        report.metadata["direction_conflict"] = f"多空分差只有 {score_gap:.1f} 分，避免方向打架硬做。"
    finalize_report_scores(report)
    return report
