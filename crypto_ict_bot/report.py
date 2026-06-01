from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import DirectionScore, SymbolReport

OPTIONAL_SCORE_FEATURES = {"trendline", "amd", "nexus", "paid_data"}

FEATURE_LABELS = {
    "liquidity_sweep": "流動性掃蕩",
    "htf_poi": "高週期 POI",
    "mss_bos": "MSS/BOS",
    "displacement": "位移",
    "fvg": "FVG",
    "ote": "OTE",
    "risk_reward": "風報比",
    "market_quality": "市場品質",
    "trendline": "趨勢線",
    "amd": "AMD",
    "nexus": "Nexus",
    "paid_data": "外部資料",
}


def direction_label(direction: str) -> str:
    return {"long": "看多", "short": "看空", "neutral": "觀望"}.get(direction, direction)


def fmt_price(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:,.2f}"
    if abs(value) >= 1:
        return f"{value:,.4f}"
    return f"{value:,.8f}".rstrip("0").rstrip(".")


def fmt_volume(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def selected_side(report: SymbolReport) -> DirectionScore:
    if report.selected_direction == "short":
        return report.short
    if report.selected_direction == "neutral" and report.short.normalized > report.long.normalized:
        return report.short
    return report.long


def entry_distance_pct(price: float, entry_zone: tuple[float, float] | None) -> float | None:
    if not entry_zone:
        return None
    low, high = entry_zone
    if low <= price <= high:
        return 0.0
    return min(abs(price - low), abs(price - high)) / max(abs(price), 1e-12) * 100.0


def _feature_ratio(side: DirectionScore, names: list[str]) -> float:
    total = sum(side.feature_scores.get(name, 0.0) for name in names)
    max_total = sum(side.feature_max_scores.get(name, 0.0) for name in names)
    if max_total <= 0:
        return 0.0
    return max(0.0, min(100.0, total / max_total * 100.0))


def _paid_values(report: SymbolReport) -> dict[str, Any]:
    return report.metadata.get("paid_data", {}).get("values", {})


def _external_api_ok(report: SymbolReport) -> bool:
    providers = set(report.metadata.get("paid_data", {}).get("providers", []))
    return "交易所公開衍生品" in providers or "CoinGlass" in providers or "Coinalyze" in providers


def _derivative_risk(report: SymbolReport, direction: str) -> tuple[bool, str]:
    values = _paid_values(report)
    public = values.get("exchange_public_derivatives", {})
    funding = public.get("funding_rate")
    oi_change = public.get("open_interest_change_pct")
    try:
        funding = float(funding) if funding is not None else None
    except (TypeError, ValueError):
        funding = None
    try:
        oi_change = float(oi_change) if oi_change is not None else None
    except (TypeError, ValueError):
        oi_change = None

    warnings: list[str] = []
    blocked = False
    if funding is not None:
        if direction == "long" and funding > 0.00035:
            blocked = True
            warnings.append(f"funding={funding:.5f}，多方槓桿過熱")
        if direction == "short" and funding < -0.00035:
            blocked = True
            warnings.append(f"funding={funding:.5f}，空方槓桿過熱")
    if oi_change is not None and abs(oi_change) >= 18:
        blocked = True
        warnings.append(f"OI 近 1h 變動 {oi_change:.2f}%，槓桿流動過劇烈")
    return blocked, "；".join(warnings)


def quant_diagnostics(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    buckets = getattr(side, "bucket_scores", {}) or {}
    htf = buckets.get("htf_context", _feature_ratio(side, ["liquidity_sweep", "htf_poi"]))
    trigger = buckets.get("ltf_confirmation", _feature_ratio(side, ["mss_bos", "displacement"]))
    entry = buckets.get("entry_location", _feature_ratio(side, ["fvg", "ote"]))
    risk = buckets.get("risk_plan", _feature_ratio(side, ["risk_reward"]))
    market = buckets.get("market_filter", _feature_ratio(side, ["market_quality"]))
    optional = _feature_ratio(side, ["trendline", "amd", "nexus", "paid_data"])
    api_ok = _external_api_ok(report)
    derivative_blocked, derivative_warning = _derivative_risk(report, report.selected_direction)
    core_ok = htf >= 45 and trigger >= 50 and entry >= 45 and risk >= 50
    return {
        "htf_context": round(htf, 1),
        "ltf_trigger": round(trigger, 1),
        "entry_quality": round(entry, 1),
        "risk_reward_quality": round(risk, 1),
        "market_api_quality": round(market, 1),
        "optional_confluence": round(optional, 1),
        "external_api_ok": api_ok,
        "derivative_blocked": derivative_blocked,
        "derivative_warning": derivative_warning,
        "core_ict_ok": core_ok,
        "direction_conflict": report.metadata.get("direction_conflict", ""),
    }


def visible_warnings(side: DirectionScore) -> list[str]:
    hidden_fragments = (
        "未納入分母",
        "未觸發",
        "尚未確認 MSS/BOS",
        "沒有明確大實體",
        "找不到方向一致",
        "最近 HTF 沒有清楚",
    )
    output: list[str] = []
    for warning in side.warnings:
        if any(fragment in warning for fragment in hidden_fragments):
            continue
        if warning not in output:
            output.append(warning)
    return output


def _weakest_core_parts(report: SymbolReport, limit: int = 2) -> list[str]:
    diag = quant_diagnostics(report)
    parts = [
        ("HTF 背景", diag["htf_context"], "等高週期掃流動性或 POI 更清楚"),
        ("LTF 觸發", diag["ltf_trigger"], "等 15m MSS/BOS 與位移確認"),
        ("入場品質", diag["entry_quality"], "等 FVG 回補、OTE 或 OB 重疊"),
        ("風控", diag["risk_reward_quality"], "等 RR 改善或價格靠近入場區"),
        ("市場/API", diag["market_api_quality"], "等成交額/BTC 共振或衍生品資料改善"),
    ]
    weak = [item for item in parts if item[1] < 60]
    weak.sort(key=lambda item: item[1])
    return [f"{name} {value:.0f}，{hint}" for name, value, hint in weak[:limit]]


def action_blockers(report: SymbolReport, limit: int = 3) -> list[str]:
    side = selected_side(report)
    diag = quant_diagnostics(report)
    blockers: list[str] = []
    distance = entry_distance_pct(report.price, side.entry_zone)
    if report.selected_direction == "neutral":
        blockers.append(report.metadata.get("direction_conflict") or "多空方向未明確")
    if not diag["external_api_ok"]:
        blockers.append("尚未讀到衍生品 OI/Funding，先不給進場")
    if diag["derivative_blocked"]:
        blockers.append(f"槓桿風險過熱：{diag['derivative_warning']}")
    blockers.extend(_weakest_core_parts(report, limit=limit))
    if side.rr is not None and side.rr < 1.5:
        blockers.append(f"RR {side.rr:.2f}R 偏低")
    if distance is not None and distance > 1.2:
        blockers.append(f"現價離入場區 {distance:.2f}%，不追價")
    for note in side.signal_notes:
        if len(blockers) >= limit:
            break
        blockers.append(note)
    for adjustment in getattr(side, "score_adjustments", []):
        if len(blockers) >= limit:
            break
        blockers.append(adjustment)
    deduped: list[str] = []
    for item in blockers:
        if item and item not in deduped:
            deduped.append(item)
    return deduped[:limit]


def action_blocker_summary(report: SymbolReport) -> str:
    blockers = action_blockers(report, limit=2)
    if blockers:
        return "；".join(blockers)
    return "等待下一根確認 K 線與入場區回補"


def score_model_audit(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    bonus_score = float(getattr(side, "bonus_score", 0.0) or 0.0)
    bonus_max = float(getattr(side, "bonus_max_score", 0.0) or 0.0)
    core_raw = max(0.0, side.score - bonus_score)
    core_max = max(0.0, side.max_score)
    core_normalized = core_raw / core_max * 100.0 if core_max else 0.0
    calibrated = side.normalized
    optional_features = {
        name: {
            "score": side.feature_scores.get(name, 0.0),
            "max": side.feature_max_scores.get(name, 0.0),
        }
        for name in OPTIONAL_SCORE_FEATURES
        if name in side.feature_scores or name in side.feature_max_scores
    }
    inactive_optional = {
        name: reason
        for name, reason in side.inactive_features.items()
        if name in OPTIONAL_SCORE_FEATURES
    }
    skipped_core = {
        name: reason
        for name, reason in side.skipped_features.items()
        if name not in OPTIONAL_SCORE_FEATURES
    }
    providers = report.metadata.get("paid_data", {}).get("providers", [])
    return {
        "method": "分層桶狀量化：HTF、LTF、入場、風控、行情品質各自封頂；同類訊號不重複灌分，未申請或未觸發的資料不當 0 分扣分。",
        "core_raw": round(core_raw, 2),
        "core_available_max": round(core_max, 2),
        "core_score": round(max(0.0, min(100.0, core_normalized)), 2),
        "calibrated_score": round(calibrated, 2),
        "bucket_scores": getattr(side, "bucket_scores", {}),
        "bucket_weights": getattr(side, "bucket_weights", {}),
        "score_adjustments": getattr(side, "score_adjustments", []),
        "bonus_score": round(bonus_score, 2),
        "bonus_available_max": round(bonus_max, 2),
        "final_score": report.score,
        "data_completeness": round(side.data_completeness, 2),
        "optional_features": optional_features,
        "skipped_core": skipped_core,
        "inactive_optional_not_penalized": inactive_optional,
        "skipped_optional_not_penalized": inactive_optional,
        "external_providers_used": providers,
        "paid_api_rule": "只有成功讀到且形成共振的外部資料才加分；未設定的付費 API 不納入分母。",
    }


def raw_trade_action(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    distance = entry_distance_pct(report.price, side.entry_zone)
    rr = side.rr or 0.0
    completeness = side.data_completeness
    score = report.score
    diag = quant_diagnostics(report)

    if report.selected_direction == "neutral":
        if score >= 58 and completeness >= 45:
            return {
                "code": "watch",
                "label": "觀察",
                "reason": f"方向未明確：{action_blocker_summary(report)}。",
                "entry_distance_pct": distance,
            }
        return {
            "code": "avoid",
            "label": "不能做",
            "reason": "方向不明確且分數不足，沒有交易優勢。",
            "entry_distance_pct": distance,
            }

    if not diag["external_api_ok"]:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": "尚未讀到衍生品 API 資料，先不給進場，只保留觀察。",
            "entry_distance_pct": distance,
        }
    if diag["derivative_blocked"]:
        return {
            "code": "avoid",
            "label": "不能做",
            "reason": f"衍生品 API 顯示槓桿風險過高：{diag['derivative_warning']}",
            "entry_distance_pct": distance,
        }
    if completeness < 45:
        return {
            "code": "avoid",
            "label": "不能做",
            "reason": f"資料完整度只有 {completeness:.0f}%，不足以做短線決策。",
            "entry_distance_pct": distance,
        }
    if not diag["core_ict_ok"] and score >= 72:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": f"分數達標但還差確認：{action_blocker_summary(report)}。",
            "entry_distance_pct": distance,
        }
    if not side.entry_zone or side.stop is None or not side.take_profits:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": "缺少完整入場區、止損或止盈計畫，先不下單。",
            "entry_distance_pct": distance,
        }
    if rr < 1.2:
        return {
            "code": "avoid",
            "label": "不能做",
            "reason": f"風報比只有 {rr:.2f}R，短線不划算。",
            "entry_distance_pct": distance,
        }
    if distance is not None and distance > 5.0:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": f"現價離入場區 {distance:.2f}%，不追價，等回補。",
            "entry_distance_pct": distance,
        }
    if score >= 82 and completeness >= 70 and rr >= 1.8 and distance is not None and distance <= 0.18 and diag["core_ict_ok"]:
        return {
            "code": "market",
            "label": "市價做",
            "reason": "高分、高資料完整度、RR 足夠，且現價已在/貼近入場區。",
            "entry_distance_pct": distance,
        }
    if score >= 72 and completeness >= 60 and rr >= 1.5 and diag["core_ict_ok"]:
        reason = "分數達可交易門檻，等價格回到入場區用限價執行。"
        if distance is not None and distance <= 0.18:
            reason = "分數達可交易門檻，現價接近入場區；保守用限價，不追滑點。"
        return {
            "code": "limit",
            "label": "限價做",
            "reason": reason,
            "entry_distance_pct": distance,
        }
    if score >= 58:
        return {
            "code": "watch",
            "label": "觀察",
            "reason": f"只觀察：{action_blocker_summary(report)}。",
            "entry_distance_pct": distance,
        }
    return {
        "code": "avoid",
        "label": "不能做",
        "reason": "分數低於 58，結構與入場條件不足。",
        "entry_distance_pct": distance,
    }


def trade_action(report: SymbolReport) -> dict[str, Any]:
    signal_state = report.metadata.get("signal_state", {})
    stable = signal_state.get("stable_action")
    if isinstance(stable, dict) and stable.get("code") and stable.get("label"):
        raw = raw_trade_action(report)
        if stable.get("code") in {"market", "limit"} and raw["code"] not in {"market", "limit"}:
            return {
                "code": "watch",
                "label": "觀察",
                "reason": f"原交易計畫轉弱，暫停執行：{raw['reason']}",
                "entry_distance_pct": raw.get("entry_distance_pct", entry_distance_pct(report.price, selected_side(report).entry_zone)),
            }
        reason = stable.get("reason") or raw["reason"]
        if isinstance(reason, str) and reason.startswith("穩定分數"):
            reason = raw["reason"]
        return {
            "code": stable.get("code"),
            "label": stable.get("label"),
            "reason": reason,
            "entry_distance_pct": stable.get("entry_distance_pct", entry_distance_pct(report.price, selected_side(report).entry_zone)),
        }
    return raw_trade_action(report)


def execution_plan(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    action = trade_action(report)
    raw = raw_trade_action(report)
    direction = direction_label(report.selected_direction)
    entry = "-"
    if side.entry_zone:
        entry = f"{fmt_price(side.entry_zone[0])} - {fmt_price(side.entry_zone[1])}"
    stop = fmt_price(side.stop)
    rr = "-" if side.rr is None else f"{side.rr:.2f}R"
    tp_text = "；".join(
        f"{tp.get('name', 'TP')} {fmt_price(tp.get('price'))} 出 {float(tp.get('portion_pct') or 0):.0f}%"
        for tp in side.take_profits
    )
    if not tp_text:
        tp_text = "尚未算出止盈"

    if action["code"] == "market":
        label = "可以做"
        mode = "市價"
        summary = f"{direction}，小倉市價或貼近現價執行；止損 {stop}，止盈：{tp_text}，RR {rr}。"
        should_execute = True
    elif action["code"] == "limit":
        label = "可以做"
        mode = "限價"
        summary = f"{direction}，只掛入場區 {entry}，不到價不追；止損 {stop}，止盈：{tp_text}，RR {rr}。"
        should_execute = True
    elif action["code"] == "watch" and raw["code"] in {"market", "limit"}:
        label = "待確認"
        mode = raw["label"]
        summary = f"原始模型是 {raw['label']}，但穩定器尚未連續確認；先盯 {entry}，確認後再照 {mode} 執行。"
        should_execute = False
    elif action["code"] == "watch":
        label = "觀察"
        mode = "不下單"
        summary = f"{direction} 觀察：{action_blocker_summary(report)}。"
        should_execute = False
    else:
        label = "不能做"
        mode = "禁止"
        summary = f"不能做：{action_blocker_summary(report)}。"
        should_execute = False

    steps = [
        f"方向：{direction}",
        f"執行：{mode}",
        f"入場：{entry}",
        f"止損：{stop}",
        f"止盈：{tp_text}",
        f"風報比：{rr}",
        f"原因：{action['reason']}",
    ]
    blockers = action_blockers(report, limit=3)
    if blockers:
        steps.append("等待條件：" + "；".join(blockers))
    return {
        "label": label,
        "mode": mode,
        "summary": summary,
        "steps": steps,
        "should_execute": should_execute,
    }


def report_to_dict(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    raw_action = raw_trade_action(report)
    action = trade_action(report)
    diagnostics = quant_diagnostics(report)
    model_audit = score_model_audit(report)
    execution = execution_plan(report)
    return {
        "symbol": report.symbol,
        "exchange": report.exchange,
        "direction": report.selected_direction,
        "direction_label": direction_label(report.selected_direction),
        "score": report.score,
        "grade": report.grade,
        "trade_action": action["code"],
        "trade_action_label": action["label"],
        "trade_action_reason": action["reason"],
        "execution_label": execution["label"],
        "execution_mode": execution["mode"],
        "execution_summary": execution["summary"],
        "execution_steps": execution["steps"],
        "should_execute": execution["should_execute"],
        "raw_trade_action": raw_action["code"],
        "raw_trade_action_label": raw_action["label"],
        "raw_trade_action_reason": raw_action["reason"],
        "entry_distance_pct": action["entry_distance_pct"],
        "quant_diagnostics": diagnostics,
        "score_model": model_audit,
        "signal_state": report.metadata.get("signal_state", {}),
        "price": report.price,
        "change_pct_24h": report.change_pct_24h,
        "quote_volume_24h": report.quote_volume_24h,
        "data_time": report.data_time.isoformat(),
        "entry_zone": side.entry_zone,
        "stop": side.stop,
        "target": side.target,
        "take_profits": side.take_profits,
        "rr": side.rr,
        "reasons": side.reasons,
        "warnings": visible_warnings(side),
        "signal_notes": side.signal_notes,
        "feature_scores": side.feature_scores,
        "feature_max_scores": side.feature_max_scores,
        "bucket_scores": getattr(side, "bucket_scores", {}),
        "bucket_weights": getattr(side, "bucket_weights", {}),
        "score_adjustments": getattr(side, "score_adjustments", []),
        "skipped_features": {name: reason for name, reason in side.skipped_features.items() if name not in OPTIONAL_SCORE_FEATURES},
        "inactive_features": side.inactive_features,
        "raw_score": side.score,
        "core_raw_score": model_audit["core_raw"],
        "core_score": model_audit["core_score"],
        "bonus_score": model_audit["bonus_score"],
        "bonus_available_max": model_audit["bonus_available_max"],
        "available_score_max": side.max_score,
        "data_completeness": side.data_completeness,
        "long_score": report.long.normalized,
        "short_score": report.short.normalized,
        "long_data_completeness": report.long.data_completeness,
        "short_data_completeness": report.short.data_completeness,
        "data_coverage": report.data_coverage,
        "missing_data": report.missing_data,
        "metadata": report.metadata,
    }


def print_table(reports: list[SymbolReport], limit: int | None = None) -> str:
    rows = reports[:limit] if limit else reports
    headers = [
        "Rank",
        "Symbol",
        "方向",
        "動作",
        "可執行",
        "分數",
        "Grade",
        "Price",
        "24h%",
        "Vol",
        "Entry",
        "Stop",
        "Target",
        "RR",
        "Top reasons",
    ]
    table_rows: list[list[str]] = []
    for idx, report in enumerate(rows, start=1):
        side = selected_side(report)
        entry = "-"
        if side.entry_zone:
            entry = f"{fmt_price(side.entry_zone[0])}-{fmt_price(side.entry_zone[1])}"
        reasons = "; ".join(side.reasons[:2]) if report.selected_direction != "neutral" else "分數不足，保持觀望"
        action = trade_action(report)
        execution = execution_plan(report)
        table_rows.append(
            [
                str(idx),
                report.symbol,
                direction_label(report.selected_direction),
                action["label"],
                execution["label"],
                f"{report.score:.1f}",
                report.grade,
                fmt_price(report.price),
                f"{report.change_pct_24h:+.2f}",
                fmt_volume(report.quote_volume_24h),
                entry,
                fmt_price(side.stop),
                fmt_price(side.target),
                "-" if side.rr is None else f"{side.rr:.2f}",
                reasons,
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = [" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))]
    lines.append("-+-".join("-" * width for width in widths))
    for row in table_rows:
        lines.append(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)


def write_json(reports: list[SymbolReport], path: str | Path, meta: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "reports": [report_to_dict(report) for report in reports],
    }
    _atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2))
    return target


def write_csv(reports: list[SymbolReport], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "symbol",
                "exchange",
                "direction",
                "trade_action",
                "trade_action_reason",
                "execution_label",
                "execution_mode",
                "execution_summary",
                "score",
                "grade",
                "price",
                "change_pct_24h",
                "quote_volume_24h",
                "entry_low",
                "entry_high",
                "stop",
                "target",
                "rr",
                "long_score",
                "short_score",
                "data_completeness",
                "available_score_max",
                "reasons",
                "warnings",
                "data_time",
            ]
        )
        for idx, report in enumerate(reports, start=1):
            side = selected_side(report)
            entry_low, entry_high = side.entry_zone if side.entry_zone else (None, None)
            writer.writerow(
                [
                    idx,
                    report.symbol,
                    report.exchange,
                    report.selected_direction,
                    trade_action(report)["label"],
                    trade_action(report)["reason"],
                    execution_plan(report)["label"],
                    execution_plan(report)["mode"],
                    execution_plan(report)["summary"],
                    report.score,
                    report.grade,
                    report.price,
                    report.change_pct_24h,
                    report.quote_volume_24h,
                    entry_low,
                    entry_high,
                    side.stop,
                    side.target,
                    side.rr,
                    report.long.normalized,
                    report.short.normalized,
                    side.data_completeness,
                    side.max_score,
                    " | ".join(side.reasons),
                    " | ".join(visible_warnings(side)),
                    report.data_time.isoformat(),
                ]
            )
    tmp.replace(target)
    return target


def _atomic_write_text(target: Path, text: str) -> None:
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)


def _feature_bars(feature_scores: dict[str, float], feature_max_scores: dict[str, float]) -> str:
    pieces: list[str] = []
    for name, value in feature_scores.items():
        label = html.escape(FEATURE_LABELS.get(name, name.replace("_", " ")))
        max_value = max(feature_max_scores.get(name, 14.0), 1.0)
        pct = max(0.0, min(100.0, value / max_value * 100.0))
        pieces.append(
            f'<div class="feature"><span>{label}</span><div><i style="width:{pct:.1f}%"></i></div><b>{value:.1f}/{max_value:.0f}</b></div>'
        )
    return "".join(pieces)


def write_html(reports: list[SymbolReport], path: str | Path, meta: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    refresh_minutes = max(1, min(5, int(meta.get("refresh_minutes") or 5)))
    refresh_seconds = refresh_minutes * 60
    rows: list[str] = []
    cards: list[str] = []
    for idx, report in enumerate(reports, start=1):
        side = selected_side(report)
        direction = direction_label(report.selected_direction)
        dir_class = report.selected_direction
        action = trade_action(report)
        execution = execution_plan(report)
        entry = "-"
        if side.entry_zone:
            entry = f"{fmt_price(side.entry_zone[0])} - {fmt_price(side.entry_zone[1])}"
        top_reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in side.reasons[:4]) or "<li>尚無足夠共振，保持觀望。</li>"
        warnings = "".join(f"<li>{html.escape(warning)}</li>" for warning in visible_warnings(side)[:3])
        if not warnings:
            warnings = "<li>無重大資料缺口。</li>"
        blockers = "".join(f"<li>{html.escape(item)}</li>" for item in action_blockers(report, limit=3))
        action_reason = "" if action["reason"] == execution["summary"] else f"<li>{html.escape(action['reason'])}</li>"
        rows.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td><strong>{html.escape(report.symbol)}</strong></td>
              <td><span class="pill {dir_class}">{direction}</span></td>
              <td><span class="pill {html.escape(action['code'])}">{html.escape(action['label'])}</span></td>
              <td>{html.escape(execution['label'])}</td>
              <td>{html.escape(execution['summary'])}</td>
              <td><span class="score">{report.score:.1f}</span></td>
              <td>{report.grade}</td>
              <td>{fmt_price(report.price)}</td>
              <td>{report.change_pct_24h:+.2f}%</td>
              <td>{fmt_volume(report.quote_volume_24h)}</td>
              <td>{entry}</td>
              <td>{fmt_price(side.stop)}</td>
              <td>{fmt_price(side.target)}</td>
              <td>{"-" if side.rr is None else f"{side.rr:.2f}"}</td>
            </tr>
            """
        )
        cards.append(
            f"""
            <article class="card">
              <header>
                <div>
                <span class="rank">#{idx}</span>
                <h2>{html.escape(report.symbol)}</h2>
              </div>
                <span class="pill {html.escape(action['code'])}">{html.escape(execution['label'])}</span>
              </header>
              <div class="scorebar"><i style="width:{report.score:.1f}%"></i></div>
              <dl>
                <div><dt>Score</dt><dd>{report.score:.1f} / 100</dd></div>
                <div><dt>執行</dt><dd>{html.escape(execution['label'])} / {html.escape(execution['mode'])}</dd></div>
                <div><dt>Entry</dt><dd>{entry}</dd></div>
                <div><dt>Stop</dt><dd>{fmt_price(side.stop)}</dd></div>
                <div><dt>Target</dt><dd>{fmt_price(side.target)}</dd></div>
                <div><dt>RR</dt><dd>{"-" if side.rr is None else f"{side.rr:.2f}"}</dd></div>
              </dl>
              <section>
                <h3>訊號來源</h3>
                <ul>{top_reasons}</ul>
              </section>
              <section>
                <h3>提醒</h3>
                <ul><li>{html.escape(execution['summary'])}</li>{action_reason}{blockers}{warnings}</ul>
              </section>
              <div class="features">{_feature_bars(side.feature_scores, side.feature_max_scores)}</div>
            </article>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ICT 選幣量化評分報告</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #1c2430;
      --muted: #6b7280;
      --line: #d9dee8;
      --long: #0f9f6e;
      --short: #d94f4f;
      --neutral: #64748b;
      --accent: #2f6fdd;
      --warn: #b7791f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Microsoft JhengHei", "Segoe UI", Arial, sans-serif;
      line-height: 1.5;
    }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    .topline {{ display: flex; justify-content: space-between; align-items: end; gap: 18px; margin-bottom: 22px; }}
    h1 {{ margin: 0; font-size: clamp(24px, 3vw, 36px); letter-spacing: 0; }}
    .meta {{ color: var(--muted); font-size: 14px; text-align: right; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 22px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; }}
    .metric strong {{ font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; font-size: 14px; white-space: nowrap; }}
    th {{ color: var(--muted); background: #eef2f7; font-weight: 700; }}
    tr:last-child td {{ border-bottom: 0; }}
    .pill {{ display: inline-flex; align-items: center; min-width: 46px; justify-content: center; padding: 3px 8px; border-radius: 999px; color: #fff; font-size: 13px; }}
    .pill.long {{ background: var(--long); }}
    .pill.short {{ background: var(--short); }}
    .pill.neutral {{ background: var(--neutral); }}
    .pill.market {{ background: var(--long); }}
    .pill.limit {{ background: var(--accent); }}
    .pill.watch {{ background: var(--warn); }}
    .pill.avoid {{ background: var(--neutral); }}
    .score {{ font-weight: 800; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 22px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .card header {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }}
    .card h2 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    .rank {{ color: var(--muted); font-weight: 700; }}
    .scorebar {{ height: 9px; border-radius: 999px; background: #e5e9f0; overflow: hidden; margin: 12px 0 14px; }}
    .scorebar i {{ display: block; height: 100%; background: var(--accent); }}
    dl {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin: 0 0 12px; }}
    dl div {{ border: 1px solid var(--line); border-radius: 8px; padding: 8px; min-width: 0; }}
    dt {{ color: var(--muted); font-size: 12px; }}
    dd {{ margin: 2px 0 0; font-weight: 700; overflow-wrap: anywhere; }}
    section h3 {{ margin: 12px 0 4px; font-size: 15px; }}
    ul {{ margin: 0; padding-left: 20px; }}
    li {{ margin: 2px 0; }}
    .features {{ display: grid; gap: 6px; margin-top: 12px; }}
    .feature {{ display: grid; grid-template-columns: 130px 1fr 38px; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); }}
    .feature div {{ height: 6px; background: #e8edf4; border-radius: 999px; overflow: hidden; }}
    .feature i {{ display: block; height: 100%; background: var(--accent); }}
    .note {{ margin-top: 20px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 980px) {{
      main {{ padding: 18px; }}
      .topline {{ display: block; }}
      .meta {{ text-align: left; margin-top: 8px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .tablewrap {{ overflow-x: auto; }}
      .grid {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
<main>
  <div class="topline">
    <div>
      <h1>ICT 選幣量化評分報告</h1>
      <div class="meta">Exchange: {html.escape(str(meta.get("exchange", "-")))} · Generated: {generated} · 本頁會偵測每 {refresh_minutes} 分鐘新報告後更新</div>
    </div>
    <div class="meta">Data source: public exchange REST API · No simulated candles</div>
  </div>
  <section class="summary">
    <div class="metric"><span>掃描幣種</span><strong>{len(reports)}</strong></div>
    <div class="metric"><span>A/B 候選</span><strong>{sum(1 for r in reports if r.grade in {"A", "B"})}</strong></div>
    <div class="metric"><span>看多</span><strong>{sum(1 for r in reports if r.selected_direction == "long")}</strong></div>
    <div class="metric"><span>看空</span><strong>{sum(1 for r in reports if r.selected_direction == "short")}</strong></div>
  </section>
  <div class="tablewrap">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Symbol</th><th>方向</th><th>動作</th><th>可執行</th><th>做單方式</th><th>分數</th><th>Grade</th><th>Price</th><th>24h%</th><th>Vol</th><th>Entry</th><th>Stop</th><th>Target</th><th>RR</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
  <section class="grid">
    {''.join(cards)}
  </section>
  <p class="note">這份報告只做選幣與觀察清單評分，不自動下單，也不是投資建議。重大新聞、鏈上資金流、真實委託簿深度若沒有接付費/授權資料，系統會以保守方式處理。</p>
</main>
<script>
  const currentReportGeneratedAt = {json.dumps(str(meta.get("generated_at") or generated))};
  const checkIntervalMs = Math.min({refresh_seconds}, 60) * 1000;
  async function checkReportUpdate() {{
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {{
      const response = await fetch(`/reports/latest.json?poll=${{Date.now()}}`, {{
        cache: "no-store",
        signal: controller.signal,
      }});
      if (!response.ok) return;
      const payload = await response.json();
      const nextGeneratedAt = payload && payload.meta && payload.meta.generated_at;
      if (nextGeneratedAt && nextGeneratedAt !== currentReportGeneratedAt) {{
        window.location.reload();
      }}
    }} catch (_error) {{
      // 掃描中或網路慢時保留目前報告，避免整頁卡在轉圈。
    }} finally {{
      clearTimeout(timeout);
    }}
  }}
  setInterval(checkReportUpdate, checkIntervalMs);
</script>
</body>
</html>"""
    _atomic_write_text(target, html_doc)
    return target
