from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import DirectionScore, SymbolReport


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


def report_to_dict(report: SymbolReport) -> dict[str, Any]:
    side = selected_side(report)
    return {
        "symbol": report.symbol,
        "exchange": report.exchange,
        "direction": report.selected_direction,
        "direction_label": direction_label(report.selected_direction),
        "score": report.score,
        "grade": report.grade,
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
        "warnings": side.warnings,
        "feature_scores": side.feature_scores,
        "feature_max_scores": side.feature_max_scores,
        "skipped_features": side.skipped_features,
        "raw_score": side.score,
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
        table_rows.append(
            [
                str(idx),
                report.symbol,
                direction_label(report.selected_direction),
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
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_csv(reports: list[SymbolReport], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "symbol",
                "exchange",
                "direction",
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
                    " | ".join(side.warnings),
                    report.data_time.isoformat(),
                ]
            )
    return target


def _feature_bars(feature_scores: dict[str, float]) -> str:
    pieces: list[str] = []
    for name, value in feature_scores.items():
        label = html.escape(name.replace("_", " "))
        pct = max(0.0, min(100.0, value / 14.0 * 100.0))
        pieces.append(
            f'<div class="feature"><span>{label}</span><div><i style="width:{pct:.1f}%"></i></div><b>{value:.1f}</b></div>'
        )
    return "".join(pieces)


def write_html(reports: list[SymbolReport], path: str | Path, meta: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    rows: list[str] = []
    cards: list[str] = []
    for idx, report in enumerate(reports, start=1):
        side = selected_side(report)
        direction = direction_label(report.selected_direction)
        dir_class = report.selected_direction
        entry = "-"
        if side.entry_zone:
            entry = f"{fmt_price(side.entry_zone[0])} - {fmt_price(side.entry_zone[1])}"
        top_reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in side.reasons[:5]) or "<li>尚無足夠共振，保持觀望。</li>"
        warnings = "".join(f"<li>{html.escape(warning)}</li>" for warning in side.warnings[:4])
        if not warnings:
            warnings = "<li>無重大資料缺口。</li>"
        rows.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td><strong>{html.escape(report.symbol)}</strong></td>
              <td><span class="pill {dir_class}">{direction}</span></td>
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
                <span class="pill {dir_class}">{direction}</span>
              </header>
              <div class="scorebar"><i style="width:{report.score:.1f}%"></i></div>
              <dl>
                <div><dt>Score</dt><dd>{report.score:.1f} / 100</dd></div>
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
                <ul>{warnings}</ul>
              </section>
              <div class="features">{_feature_bars(side.feature_scores)}</div>
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
      <div class="meta">Exchange: {html.escape(str(meta.get("exchange", "-")))} · Generated: {generated}</div>
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
          <th>#</th><th>Symbol</th><th>方向</th><th>分數</th><th>Grade</th><th>Price</th><th>24h%</th><th>Vol</th><th>Entry</th><th>Stop</th><th>Target</th><th>RR</th>
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
</body>
</html>"""
    target.write_text(html_doc, encoding="utf-8")
    return target
