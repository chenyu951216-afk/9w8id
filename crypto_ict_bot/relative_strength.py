from __future__ import annotations

from statistics import median
from typing import Any

from .models import SymbolReport


def attach_relative_strength(reports: list[SymbolReport]) -> None:
    if not reports:
        return
    changes = [float(report.change_pct_24h or 0.0) for report in reports]
    volumes = [float(report.quote_volume_24h or 0.0) for report in reports]
    median_change = median(changes)
    for report in reports:
        change = float(report.change_pct_24h or 0.0)
        volume = float(report.quote_volume_24h or 0.0)
        change_pctile = _percentile_rank(changes, change)
        volume_pctile = _percentile_rank(volumes, volume)
        atr_pct = _side_atr(report)
        vol_adjusted = 50.0 + (change - median_change) / max(atr_pct or 1.0, 0.25) * 18.0
        long_rs = change_pctile * 0.52 + volume_pctile * 0.18 + _clamp(vol_adjusted) * 0.30
        short_rs = (100.0 - change_pctile) * 0.52 + volume_pctile * 0.18 + _clamp(100.0 - vol_adjusted) * 0.30
        metrics = {
            "relative_strength_btc": round(change - _btc_change(reports), 4),
            "relative_strength_score_long": round(_clamp(long_rs), 2),
            "relative_strength_score_short": round(_clamp(short_rs), 2),
            "relative_volume": round(volume_pctile, 2),
            "momentum_24h": round(change, 4),
            "volatility_adjusted_momentum": round(_clamp(vol_adjusted), 2),
            "market_relative_return": round(change - median_change, 4),
        }
        report.metadata["relative_strength"] = metrics


def relative_strength_score(report: SymbolReport, direction: str) -> float:
    metrics = report.metadata.get("relative_strength", {})
    if direction == "short":
        return float(metrics.get("relative_strength_score_short", 50.0) or 50.0)
    return float(metrics.get("relative_strength_score_long", 50.0) or 50.0)


def _percentile_rank(values: list[float], value: float) -> float:
    if not values:
        return 50.0
    below = sum(1 for item in values if item < value)
    equal = sum(1 for item in values if item == value)
    return (below + equal * 0.5) / len(values) * 100.0


def _side_atr(report: SymbolReport) -> float | None:
    values = []
    for side in (report.long, report.short):
        value = side.market_metrics.get("atr_pct")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return median(values) if values else None


def _btc_change(reports: list[SymbolReport]) -> float:
    for report in reports:
        if report.symbol.upper() == "BTCUSDT":
            return float(report.change_pct_24h or 0.0)
    return median(float(report.change_pct_24h or 0.0) for report in reports)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
