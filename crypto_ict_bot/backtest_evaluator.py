from __future__ import annotations

from typing import Any


def calibration_metrics(signal_state: dict[str, Any]) -> dict[str, Any]:
    stats = signal_state.get("statistics", {}) if isinstance(signal_state, dict) else {}
    if not isinstance(stats, dict):
        stats = {}
    return {
        "mode": "online_forward_validation",
        "lookahead_bias_control": "uses only stored future_validation after subsequent scans; no future candles are read during scoring.",
        "fee_model": "heuristic fee_cost_R=0.025 in expected_value.py",
        "slippage_model": "volume/ATR heuristic in expected_value.py",
        "available_statistics": {
            "total_signals": stats.get("total_signals", 0),
            "active_signals": stats.get("active_signals", 0),
            "failed_signals": stats.get("failed_signals", 0),
            "grade_counts": stats.get("grade_counts", {}),
            "setup_tag_stats": stats.get("setup_tag_stats", {}),
        },
        "top_k_metrics": {
            "precision_at_5": None,
            "precision_at_10": None,
            "top_5_average_R": None,
            "top_10_average_R": None,
            "rank_correlation_future_R": None,
            "note": "Needs resolved signal_logger outcomes before these become calibrated metrics.",
        },
    }
