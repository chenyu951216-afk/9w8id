from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import SymbolReport


SIGNAL_LOG_PATH = Path("state/signal_log.jsonl")


def log_signal_snapshot(reports: list[SymbolReport], meta: dict[str, Any], path: Path = SIGNAL_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scan_id = str(meta.get("generated_at") or datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as handle:
        for report in reports:
            opportunity = report.metadata.get("opportunity", {})
            side = _selected_side(report)
            row = {
                "signal_id": f"{scan_id}:{report.symbol}",
                "scan_id": scan_id,
                "timestamp": meta.get("generated_at"),
                "symbol": report.symbol,
                "direction": opportunity.get("direction_analysis", {}).get("chosen_direction", report.selected_direction),
                "features": {
                    "bucket_scores": getattr(side, "bucket_scores", {}),
                    "relative_strength": report.metadata.get("relative_strength", {}),
                    "market_regime": report.metadata.get("market_regime", {}).get("regime"),
                },
                "scores": {
                    "opportunity_score": opportunity.get("opportunity_score"),
                    "setup_score": opportunity.get("setup_score"),
                    "execution_quality": opportunity.get("execution_quality"),
                    "expected_R": opportunity.get("expected_R"),
                },
                "entry": side.entry_zone,
                "stop": side.stop,
                "tp": side.take_profits,
                "rr": side.rr,
                "market_regime": report.metadata.get("market_regime", {}),
                "outcome": {
                    "whether_entry_touched": None,
                    "whether_tp_before_sl": None,
                    "mfe": None,
                    "mae": None,
                    "time_to_entry": None,
                    "time_to_resolution": None,
                    "final_R": None,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _selected_side(report: SymbolReport) -> Any:
    if report.selected_direction == "short":
        return report.short
    if report.selected_direction == "neutral" and _side_score(report.short) > _side_score(report.long):
        return report.short
    return report.long


def _side_score(side: Any) -> float:
    if side.selection_score is not None:
        return float(side.selection_score)
    if side.calibrated_score is not None:
        return float(side.calibrated_score)
    return float(side.normalized)
