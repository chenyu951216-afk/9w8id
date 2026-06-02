from __future__ import annotations

from typing import Any

from ..models import SymbolReport
from ..risk.execution_gate import (
    evaluate_execution_gate,
    invalidation_conditions,
    paid_data_status,
    selected_side,
    side_score,
)


REQUIRED_SYMBOL_FIELDS = [
    "symbol",
    "exchange",
    "price",
    "volume",
    "data_time",
    "selected_direction",
    "long_score",
    "short_score",
    "score_gap",
    "selection_score",
    "setup_score",
    "execution_score",
    "bucket_scores",
    "trade_action",
    "entry_zone",
    "stop",
    "TP1",
    "TP2",
    "TP3",
    "rr",
    "blockers",
    "warnings",
    "invalidation_conditions",
    "paid_data_status",
    "signal_lifecycle",
]


def complete_symbol_payload(
    report: SymbolReport,
    payload: dict[str, Any],
    gate_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    side = selected_side(report)
    gate = gate_result or evaluate_execution_gate(report)
    take_profits = list(side.take_profits or [])
    payload["selected_direction"] = report.selected_direction
    payload["volume"] = report.quote_volume_24h
    payload["long_score"] = side_score(report.long)
    payload["short_score"] = side_score(report.short)
    payload["score_gap"] = float(report.metadata.get("score_gap", 0.0) or 0.0)
    payload["selection_score"] = side.selection_score if side.selection_score is not None else side_score(side)
    payload["setup_score"] = side.setup_score
    payload["execution_score"] = side.execution_score
    payload["bucket_scores"] = getattr(side, "bucket_scores", {}) or {}
    payload["blockers"] = list(gate.get("blockers", []))
    payload["warnings"] = list(dict.fromkeys(list(payload.get("warnings", [])) + list(gate.get("warnings", []))))
    payload["invalidation_conditions"] = gate.get("invalidation_conditions") or invalidation_conditions(report)
    payload["paid_data_status"] = gate.get("paid_data_status") or paid_data_status(report)
    payload["signal_lifecycle"] = payload.get("signal_state", {})
    payload["TP1"] = _tp_at(take_profits, 0)
    payload["TP2"] = _tp_at(take_profits, 1)
    payload["TP3"] = _tp_at(take_profits, 2)
    payload["rr"] = side.rr
    payload["gate_checks"] = gate.get("gate_checks", {})
    payload["missing_contract_fields"] = [
        name
        for name in REQUIRED_SYMBOL_FIELDS
        if name not in payload or payload.get(name) is None and name not in {"setup_score", "execution_score", "stop", "TP1", "TP2", "TP3", "rr"}
    ]
    return payload


def _tp_at(take_profits: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    if index >= len(take_profits):
        return None
    return take_profits[index]
