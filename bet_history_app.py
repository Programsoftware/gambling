from __future__ import annotations

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
BACKTEST_DIR = ROOT / "data" / "nba" / "backtests" / "transformer"
MODEL_DIR = ROOT / "data" / "nba" / "models"
RAW_BETS_CSV = BACKTEST_DIR / "nba_transformer_backtest_bets.csv"
CLEAN_BETS_CSV = BACKTEST_DIR / "nba_transformer_bet_history_clean.csv"
SCHEDULE_CSV = ROOT / "data" / "nba" / "latest" / "team_schedule_games.csv"
WALK_FORWARD_REPORT = BACKTEST_DIR / "nba_walk_forward_validation_report_latest.json"
DEFAULT_CSV = RAW_BETS_CSV


def optional_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def float_value(value: object, default: float = 0.0) -> float:
    parsed = optional_float(value)
    return parsed if parsed is not None else default


def rounded_value(value: object, digits: int = 2) -> float | str:
    parsed = optional_float(value)
    return "" if parsed is None else round(parsed, digits)


def pct_value(value: object) -> float | str:
    parsed = optional_float(value)
    return "" if parsed is None else round(parsed * 100, 2)


def bool_value(value: object) -> bool | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n"}:
        return False
    try:
        return bool(int(float(text)))
    except (TypeError, ValueError):
        return None


def score_text(away_score: object, home_score: object) -> str:
    away = optional_float(away_score)
    home = optional_float(home_score)
    if away is None or home is None:
        return ""
    return f"{int(away)}-{int(home)}"


def season_type_text(value: object) -> str:
    parsed = optional_float(value)
    if parsed == 2:
        return "Regular"
    if parsed == 3:
        return "Playoffs"
    return str(value or "")


def reason_label(value: object) -> str:
    text = str(value or "").strip()
    return text.replace("_", " ").title() if text else ""


def load_scores() -> dict[str, tuple[str, str]]:
    if not SCHEDULE_CSV.exists():
        return {}
    with SCHEDULE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        scores: dict[str, tuple[str, str]] = {}
        for row in csv.DictReader(handle):
            event_id = str(row.get("event_id") or "")
            if event_id and event_id not in scores:
                scores[event_id] = (str(row.get("away_score") or ""), str(row.get("home_score") or ""))
    return scores


def moneyline_text(value: object) -> str:
    try:
        odds = int(float(str(value)))
    except (TypeError, ValueError):
        return ""
    return f"+{odds}" if odds > 0 else str(odds)


def moneyline_number(value: object) -> int | str:
    parsed = optional_float(value)
    return "" if parsed is None else int(parsed)


def no_vig_bet_side_prob(row: dict[str, object], side: str) -> float | None:
    market_prob = optional_float(row.get("market_prob"))
    if market_prob is not None:
        return market_prob
    home_market = optional_float(row.get("market_home_no_vig_prob"))
    if home_market is None:
        return None
    if side == "home":
        return home_market
    if side == "away":
        return 1.0 - home_market
    return None


def normalize_raw_row(row: dict[str, object], index: int, scores: dict[str, tuple[str, str]]) -> dict[str, object]:
    event_id = str(row.get("event_id") or "")
    away_team = str(row.get("away_team") or "")
    home_team = str(row.get("home_team") or "")
    side = str(row.get("bet_side") or "skip").lower()
    candidate_side = str(row.get("candidate_side") or (side if side in {"home", "away"} else "")).lower()
    candidate_team = str(row.get("candidate_team") or "")
    if not candidate_team:
        candidate_team = home_team if candidate_side == "home" else away_team if candidate_side == "away" else ""
    placed = side in {"home", "away"}
    won = bool_value(row.get("won"))
    home_won = bool_value(row.get("target_home_win"))
    actual_winner = home_team if home_won else away_team if home_won is False else ""
    away_score, home_score = scores.get(event_id, ("", ""))
    market_prob = no_vig_bet_side_prob(row, candidate_side)
    result = "WIN" if placed and won else "LOSS" if placed else "SKIP"

    return {
        "bet_number": index,
        "event_id": event_id,
        "game_date": str(row.get("game_date") or ""),
        "season_type": row.get("season_type"),
        "season_type_text": season_type_text(row.get("season_type")),
        "matchup": f"{away_team} at {home_team}",
        "final_score": score_text(away_score, home_score),
        "bet_side": side,
        "candidate_side": candidate_side,
        "candidate_team": candidate_team,
        "bet_team": str(row.get("bet_team") or ""),
        "display_team": str(row.get("bet_team") or candidate_team),
        "actual_winner": actual_winner,
        "result": result,
        "skip_reason": str(row.get("skip_reason") or ""),
        "skip_reason_label": reason_label(row.get("skip_reason")),
        "model_prob_pct": pct_value(row.get("model_prob")),
        "market_no_vig_prob_pct": "" if market_prob is None else round(market_prob * 100, 2),
        "break_even_prob_pct": pct_value(row.get("break_even_prob")),
        "model_edge_vs_break_even_pct": pct_value(row.get("model_edge_vs_break_even")),
        "model_edge_vs_market_pct": pct_value(row.get("model_edge_vs_market")),
        "american_odds": moneyline_number(row.get("offered_american_odds")),
        "decimal_odds": rounded_value(row.get("decimal_odds"), 3),
        "raw_kelly_fraction_pct": pct_value(row.get("raw_kelly_fraction")),
        "stake_fraction_pct": pct_value(row.get("stake_fraction")),
        "stake": float_value(row.get("stake")),
        "profit": float_value(row.get("profit")),
        "bankroll_after": float_value(row.get("bankroll_after")),
        "placed": placed,
    }


def normalize_clean_row(row: dict[str, object], index: int) -> dict[str, object]:
    side = str(row.get("bet_side") or "").lower()
    result = str(row.get("result") or "").upper()
    return {
        **row,
        "bet_number": index,
        "season_type_text": season_type_text(row.get("season_type")),
        "candidate_side": str(row.get("candidate_side") or side),
        "candidate_team": str(row.get("candidate_team") or row.get("bet_team") or ""),
        "display_team": str(row.get("bet_team") or row.get("candidate_team") or ""),
        "skip_reason": str(row.get("skip_reason") or ""),
        "skip_reason_label": reason_label(row.get("skip_reason")),
        "model_prob_pct": rounded_value(row.get("model_prob_pct")),
        "market_no_vig_prob_pct": rounded_value(row.get("market_no_vig_prob_pct")),
        "break_even_prob_pct": rounded_value(row.get("break_even_prob_pct")),
        "model_edge_vs_break_even_pct": rounded_value(row.get("model_edge_vs_break_even_pct")),
        "model_edge_vs_market_pct": rounded_value(row.get("model_edge_vs_market_pct")),
        "decimal_odds": rounded_value(row.get("decimal_odds"), 3),
        "raw_kelly_fraction_pct": rounded_value(row.get("raw_kelly_fraction_pct")),
        "stake_fraction_pct": rounded_value(row.get("stake_fraction_pct")),
        "stake": float_value(row.get("stake")),
        "profit": float_value(row.get("profit")),
        "bankroll_after": float_value(row.get("bankroll_after")),
        "american_odds": moneyline_number(row.get("american_odds")),
        "placed": result != "SKIP",
    }


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    if {"target_home_win", "away_team", "home_team", "model_prob"}.issubset(fieldnames):
        scores = load_scores()
        normalized = [normalize_raw_row(row, index, scores) for index, row in enumerate(rows, start=1)]
    else:
        normalized = [normalize_clean_row(row, index) for index, row in enumerate(rows, start=1)]

    for row in normalized:
        row["american_odds_text"] = moneyline_text(row.get("american_odds"))
    return normalized


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    placed = [row for row in rows if str(row.get("result", "")).upper() in {"WIN", "LOSS"}]
    skipped = [row for row in rows if str(row.get("result", "")).upper() == "SKIP"]
    wins = sum(1 for row in placed if str(row.get("result", "")).upper() == "WIN")
    losses = sum(1 for row in placed if str(row.get("result", "")).upper() == "LOSS")
    total_staked = round(sum(float_value(row.get("stake")) for row in placed), 2)
    profit = round(sum(float_value(row.get("profit")) for row in rows), 2)
    ending_bankroll = round(float_value(rows[-1].get("bankroll_after")) if rows else 0.0, 2)
    starting_bankroll = round(float_value(rows[0].get("bankroll_after")) - float_value(rows[0].get("profit")), 2) if rows else 0.0
    roi_staked = round((profit / total_staked) * 100, 2) if total_staked else 0.0
    win_rate = round((wins / len(placed)) * 100, 2) if placed else 0.0
    model_rows = [row for row in placed if optional_float(row.get("model_prob_pct")) is not None]
    edge_rows = [row for row in placed if optional_float(row.get("model_edge_vs_break_even_pct")) is not None]
    avg_model_prob = round(sum(float_value(row.get("model_prob_pct")) for row in model_rows) / len(model_rows), 2) if model_rows else 0.0
    avg_edge = round(sum(float_value(row.get("model_edge_vs_break_even_pct")) for row in edge_rows) / len(edge_rows), 2) if edge_rows else 0.0
    best = max(placed, key=lambda row: float_value(row.get("profit")), default={})
    worst = min(placed, key=lambda row: float_value(row.get("profit")), default={})
    skip_counts: dict[str, int] = {}
    for row in skipped:
        reason = str(row.get("skip_reason") or "unknown")
        skip_counts[reason] = skip_counts.get(reason, 0) + 1
    return {
        "evaluated_games": len(rows),
        "bets": len(placed),
        "skips": len(skipped),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": ending_bankroll,
        "total_staked": total_staked,
        "profit": profit,
        "roi_staked_pct": roi_staked,
        "avg_model_prob_pct": avg_model_prob,
        "avg_edge_vs_break_even_pct": avg_edge,
        "skip_reason_counts": skip_counts,
        "best_bet": {
            "team": best.get("display_team") or best.get("bet_team", ""),
            "profit": round(float_value(best.get("profit")), 2),
            "date": best.get("game_date", ""),
        },
        "worst_bet": {
            "team": worst.get("display_team") or worst.get("bet_team", ""),
            "profit": round(float_value(worst.get("profit")), 2),
            "date": worst.get("game_date", ""),
        },
    }


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def run_id_for_path(path: Path) -> str:
    stem = path.stem
    prefix = "nba_transformer_backtest_bets"
    if stem == prefix:
        return "latest"
    if stem.startswith(prefix + "_"):
        return stem[len(prefix) + 1 :]
    return stem


def summary_path_for_bets(path: Path) -> Path:
    suffix = run_id_for_path(path)
    if suffix == "latest":
        return BACKTEST_DIR / "nba_transformer_backtest_summary.json"
    return BACKTEST_DIR / f"nba_transformer_backtest_summary_{suffix}.json"


def load_json_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def label_for_bet_run(run_id: str, meta: dict[str, object]) -> str:
    label = meta.get("label")
    if isinstance(label, str) and label:
        return label
    if run_id == "latest":
        return "Latest saved backtest"
    if run_id == "profit500":
        return "Profit $500 full run"
    if run_id.startswith("profit500_") and len(run_id) == len("profit500_202606"):
        month = run_id.removeprefix("profit500_")
        return f"{month[:4]}-{month[4:]} standalone month"
    return run_id.replace("_", " ").title()


def bet_run_paths() -> dict[str, Path]:
    paths = sorted(BACKTEST_DIR.glob("nba_transformer_backtest_bets*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
    return {run_id_for_path(path): path for path in paths}


def discover_bet_runs(default_path: Path | None = None) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    default_id = run_id_for_path(default_path) if default_path else ""
    for run_id, path in bet_run_paths().items():
        rows = load_rows(path)
        summary = summarize(rows)
        meta = load_json_file(summary_path_for_bets(path))
        dates = [str(row.get("game_date") or "") for row in rows if row.get("game_date")]
        settings = meta.get("settings") if isinstance(meta.get("settings"), dict) else meta.get("kelly", {})
        runs.append(
            {
                "id": run_id,
                "label": label_for_bet_run(run_id, meta),
                "path": relative_path(path),
                "summary_path": relative_path(summary_path_for_bets(path)) if summary_path_for_bets(path).exists() else "",
                "is_default": run_id == default_id,
                "created_at": meta.get("created_at") or path.stat().st_mtime,
                "date_start": min(dates) if dates else "",
                "date_end": max(dates) if dates else "",
                "settings": settings,
                "summary": summary,
            }
        )
    runs.sort(key=lambda run: (not bool(run.get("is_default")), str(run.get("label"))))
    return runs


def resolve_bet_run_csv(run_id: str | None, default_path: Path) -> Path:
    paths = bet_run_paths()
    if run_id and run_id in paths:
        return paths[run_id]
    return default_path


def model_diagnostics(data: dict[str, object]) -> dict[str, object]:
    diagnostics = data.get("training_diagnostics") or {}
    if isinstance(diagnostics, list):
        diagnostics = diagnostics[0] if diagnostics and isinstance(diagnostics[0], dict) else {}
    return diagnostics if isinstance(diagnostics, dict) else {}


def discover_model_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for path in sorted(MODEL_DIR.glob("nba_transformer_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.name == "nba_transformer_latest.json":
            continue
        data = load_json_file(path)
        args = data.get("training_args") if isinstance(data.get("training_args"), dict) else {}
        diagnostics = model_diagnostics(data)
        runs.append(
            {
                "id": path.stem,
                "file": relative_path(path),
                "created_at": data.get("created_at") or "",
                "feature_count": len(data.get("feature_columns", [])) if isinstance(data.get("feature_columns"), list) else 0,
                "feature_mode": args.get("feature_mode", ""),
                "model_type": args.get("model_type", data.get("model_type", "transformer")),
                "epochs": args.get("epochs", ""),
                "batch_size": args.get("batch_size", ""),
                "train_rows": diagnostics.get("train_rows", ""),
                "validation_rows": diagnostics.get("validation_rows", ""),
                "best_epoch": diagnostics.get("best_epoch", ""),
                "best_validation_loss": rounded_value(diagnostics.get("best_validation_loss"), 4),
                "validation_auc": rounded_value(diagnostics.get("validation_auc"), 4),
                "validation_accuracy": rounded_value(diagnostics.get("validation_accuracy_0_50"), 4),
            }
        )
    return runs


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NBA Backtest Bet History</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --surface: #ffffff;
      --surface-2: #eef3f8;
      --line: #d7dee8;
      --line-strong: #b8c4d2;
      --text: #17202c;
      --muted: #66758a;
      --green: #168255;
      --green-soft: #e6f5ee;
      --red: #c43d4b;
      --red-soft: #fae9ec;
      --blue: #2367a8;
      --blue-soft: #e8f1fb;
      --amber: #9f6408;
      --amber-soft: #fff2d8;
      --shadow: 0 14px 32px rgba(24, 38, 54, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.4;
    }

    button, input, select { font: inherit; }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(245, 247, 250, 0.94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }

    .topbar-inner {
      max-width: 1500px;
      margin: 0 auto;
      padding: 16px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 750;
      letter-spacing: 0;
    }

    .subtle {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }

    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .button {
      height: 36px;
      border: 1px solid var(--line-strong);
      border-radius: 7px;
      background: var(--surface);
      color: var(--text);
      padding: 0 12px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      text-decoration: none;
      box-shadow: 0 1px 1px rgba(24, 38, 54, 0.04);
    }

    .button:hover { border-color: #8fa1b5; }

    .button.primary {
      background: var(--text);
      color: #fff;
      border-color: var(--text);
    }

    .summary-band {
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }

    .summary {
      max-width: 1500px;
      margin: 0 auto;
      padding: 14px 20px;
      display: grid;
      grid-template-columns: repeat(8, minmax(112px, 1fr));
      gap: 10px;
    }

    .metric {
      min-height: 72px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 11px;
      background: #fff;
      overflow: hidden;
    }

    .metric-label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
      white-space: nowrap;
    }

    .metric-value {
      margin-top: 6px;
      font-size: 20px;
      line-height: 1.1;
      font-weight: 760;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .metric-note {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .positive { color: var(--green); }
    .negative { color: var(--red); }
    .neutral { color: var(--blue); }

    .main {
      max-width: 1500px;
      width: 100%;
      margin: 0 auto;
      padding: 16px 20px 28px;
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      gap: 16px;
    }

    .sidebar {
      align-self: start;
      position: sticky;
      top: 86px;
      display: grid;
      gap: 12px;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .panel-header {
      padding: 12px 13px;
      border-bottom: 1px solid var(--line);
      font-weight: 720;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }

    .panel-body {
      padding: 13px;
    }

    .filters {
      display: grid;
      gap: 12px;
    }

    .field {
      display: grid;
      gap: 6px;
    }

    label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    input, select {
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line-strong);
      border-radius: 7px;
      background: #fff;
      color: var(--text);
      padding: 7px 9px;
      outline: none;
    }

    input:focus, select:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px var(--blue-soft);
    }

    .range {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .chart-wrap {
      height: 174px;
      padding: 10px 12px 12px;
    }

    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    .table-panel {
      min-width: 0;
      overflow: hidden;
    }

    .table-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }

    .table-count {
      color: var(--muted);
      white-space: nowrap;
    }

    .table-scroll {
      overflow: auto;
      max-height: calc(100vh - 255px);
      background: #fff;
    }

    table {
      width: 100%;
      min-width: 1500px;
      border-collapse: separate;
      border-spacing: 0;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }

    th {
      position: sticky;
      top: 0;
      z-index: 5;
      background: #f9fbfd;
      color: #435166;
      font-size: 12px;
      font-weight: 760;
      cursor: pointer;
      user-select: none;
      box-shadow: inset 0 -1px 0 var(--line);
    }

    th:hover { color: var(--text); }

    tbody tr:hover { background: #fbfdff; }

    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .muted { color: var(--muted); }
    .team-cell { max-width: 210px; overflow: hidden; text-overflow: ellipsis; }
    .matchup-cell { max-width: 300px; overflow: hidden; text-overflow: ellipsis; }

    .pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 48px;
      height: 24px;
      border-radius: 999px;
      padding: 0 8px;
      font-size: 12px;
      font-weight: 760;
      border: 1px solid transparent;
    }

    .pill.win {
      color: var(--green);
      background: var(--green-soft);
      border-color: #b8e2cd;
    }

    .pill.loss {
      color: var(--red);
      background: var(--red-soft);
      border-color: #efc3ca;
    }

    .pill.skip {
      color: var(--amber);
      background: var(--amber-soft);
      border-color: #edcf91;
    }

    .side {
      color: var(--blue);
      background: var(--blue-soft);
      border-color: #bed7f0;
      text-transform: uppercase;
    }

    .side.skip {
      color: var(--amber);
      background: var(--amber-soft);
      border-color: #edcf91;
    }

    .prob-cell {
      min-width: 118px;
    }

    .bar {
      height: 5px;
      margin-top: 5px;
      border-radius: 999px;
      background: var(--surface-2);
      overflow: hidden;
    }

    .bar > span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: var(--blue);
    }

    .empty {
      padding: 34px 18px;
      color: var(--muted);
      text-align: center;
    }

    .model-list {
      display: grid;
      gap: 8px;
      max-height: 320px;
      overflow: auto;
    }

    .model-row {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px;
      background: #fbfdff;
    }

    .model-row strong {
      display: block;
      font-size: 12px;
      overflow-wrap: anywhere;
    }

    .model-row span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-top: 3px;
    }

    @media (max-width: 1100px) {
      .summary { grid-template-columns: repeat(4, minmax(120px, 1fr)); }
      .main { grid-template-columns: 1fr; }
      .sidebar { position: static; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .table-scroll { max-height: none; }
    }

    @media (max-width: 720px) {
      .topbar-inner { align-items: flex-start; flex-direction: column; }
      .actions { justify-content: flex-start; width: 100%; }
      .button { flex: 1 1 auto; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .main { padding: 12px; }
      .sidebar { grid-template-columns: 1fr; }
      .table-toolbar { align-items: flex-start; flex-direction: column; }
      .table-count { white-space: normal; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="topbar-inner">
        <div>
          <h1>NBA Backtest Bet History</h1>
          <div class="subtle" id="sourceText">Loading bet history...</div>
        </div>
        <div class="actions">
          <button class="button" id="refreshBtn" type="button">Refresh</button>
          <a class="button primary" href="/api/export.csv" id="exportLink">Download CSV</a>
        </div>
      </div>
    </header>

    <section class="summary-band">
      <div class="summary" id="summary"></div>
    </section>

    <main class="main">
      <aside class="sidebar">
        <section class="panel">
          <div class="panel-header">Filters</div>
          <div class="panel-body filters">
            <div class="field">
              <label for="runSelect">Backtest run</label>
              <select id="runSelect"></select>
            </div>
            <div class="field">
              <label for="searchInput">Search</label>
              <input id="searchInput" type="search" placeholder="Team, matchup, date">
            </div>
            <div class="range">
              <div class="field">
                <label for="resultFilter">Result</label>
                <select id="resultFilter">
                  <option value="all">All</option>
                  <option value="WIN">Wins</option>
                  <option value="LOSS">Losses</option>
                  <option value="SKIP">Skips</option>
                </select>
              </div>
              <div class="field">
                <label for="sideFilter">Side</label>
                <select id="sideFilter">
                  <option value="all">All</option>
                  <option value="home">Home</option>
                  <option value="away">Away</option>
                  <option value="skip">Skip</option>
                </select>
              </div>
            </div>
            <div class="range">
              <div class="field">
                <label for="skipFilter">Skip reason</label>
                <select id="skipFilter">
                  <option value="all">All</option>
                  <option value="market_disagreement">Market disagreement</option>
                  <option value="no_positive_kelly">No positive Kelly</option>
                  <option value="below_min_edge">Below min edge</option>
                </select>
              </div>
              <div class="field">
                <label for="minEdge">Min edge %</label>
                <input id="minEdge" type="number" step="0.1" placeholder="0">
              </div>
            </div>
            <div class="range">
              <div class="field">
                <label for="minStake">Min stake</label>
                <input id="minStake" type="number" step="1" placeholder="0">
              </div>
              <div class="field">
                <label for="gameTypeFilter">Game type</label>
                <select id="gameTypeFilter">
                  <option value="all">All</option>
                  <option value="Regular">Regular</option>
                  <option value="Playoffs">Playoffs</option>
                </select>
              </div>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-header">Bankroll</div>
          <div class="chart-wrap">
            <canvas id="bankrollChart" width="520" height="260"></canvas>
          </div>
        </section>

        <section class="panel">
          <div class="panel-header">Validation Fold</div>
          <div class="panel-body">
            <div class="model-list" id="validationReport"></div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-header">Model Runs</div>
          <div class="panel-body">
            <div class="model-list" id="modelRunList"></div>
          </div>
        </section>
      </aside>

      <section class="panel table-panel">
        <div class="table-toolbar">
          <strong>Bet Ledger</strong>
          <div class="table-count" id="tableCount"></div>
        </div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr id="tableHead"></tr>
            </thead>
            <tbody id="tableBody"></tbody>
          </table>
        </div>
      </section>
    </main>
  </div>

  <script>
    const columns = [
      { key: "bet_number", label: "#", type: "number", cls: "num" },
      { key: "game_date", label: "Date" },
      { key: "season_type_text", label: "Type" },
      { key: "matchup", label: "Matchup", cls: "matchup-cell" },
      { key: "final_score", label: "Score" },
      { key: "bet_side", label: "Decision", render: row => `<span class="pill side ${row.bet_side === "skip" ? "skip" : ""}">${escapeHtml(row.bet_side)}</span>` },
      { key: "display_team", label: "Candidate", cls: "team-cell" },
      { key: "actual_winner", label: "Winner", cls: "team-cell" },
      { key: "result", label: "Result", render: row => `<span class="pill ${resultClass(row.result)}">${escapeHtml(row.result)}</span>` },
      { key: "skip_reason_label", sortKey: "skip_reason", label: "Skip Reason", cls: "team-cell" },
      { key: "model_prob_pct", label: "Model %", type: "number", cls: "num prob-cell", render: row => probCell(row.model_prob_pct) },
      { key: "market_no_vig_prob_pct", label: "Market %", type: "number", cls: "num", render: row => percentOrDash(row.market_no_vig_prob_pct) },
      { key: "break_even_prob_pct", label: "Break Even %", type: "number", cls: "num", render: row => percentOrDash(row.break_even_prob_pct) },
      { key: "model_edge_vs_break_even_pct", label: "Edge %", type: "number", cls: "num", render: row => percentOrDash(row.model_edge_vs_break_even_pct) },
      { key: "american_odds_text", sortKey: "american_odds", label: "Odds", type: "number", cls: "num", render: row => textOrDash(row.american_odds_text) },
      { key: "stake", label: "Stake", type: "number", cls: "num", render: row => money(row.stake) },
      { key: "profit", label: "Profit", type: "number", cls: "num", render: row => `<span class="${profitClass(row.profit)}">${money(row.profit)}</span>` },
      { key: "bankroll_after", label: "Bankroll", type: "number", cls: "num", render: row => money(row.bankroll_after) }
    ];

    let allRows = [];
    let filteredRows = [];
    let summary = {};
    let betRuns = [];
    let modelRuns = [];
    let validationReport = {};
    let sort = { key: "bet_number", direction: "asc", type: "number" };

    const els = {
      sourceText: document.getElementById("sourceText"),
      summary: document.getElementById("summary"),
      tableHead: document.getElementById("tableHead"),
      tableBody: document.getElementById("tableBody"),
      tableCount: document.getElementById("tableCount"),
      runSelect: document.getElementById("runSelect"),
      searchInput: document.getElementById("searchInput"),
      resultFilter: document.getElementById("resultFilter"),
      sideFilter: document.getElementById("sideFilter"),
      skipFilter: document.getElementById("skipFilter"),
      gameTypeFilter: document.getElementById("gameTypeFilter"),
      minEdge: document.getElementById("minEdge"),
      minStake: document.getElementById("minStake"),
      refreshBtn: document.getElementById("refreshBtn"),
      exportLink: document.getElementById("exportLink"),
      validationReport: document.getElementById("validationReport"),
      modelRunList: document.getElementById("modelRunList"),
      chart: document.getElementById("bankrollChart")
    };

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[c]));
    }

    function numberValue(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n : 0;
    }

    function hasNumber(value) {
      if (value === null || value === undefined || value === "") return false;
      return Number.isFinite(Number(value));
    }

    function textOrDash(value) {
      const text = String(value ?? "");
      return text ? escapeHtml(text) : `<span class="muted">-</span>`;
    }

    function money(value) {
      const n = numberValue(value);
      const sign = n < 0 ? "-" : "";
      return `${sign}$${Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }

    function percent(value) {
      return `${numberValue(value).toFixed(2)}%`;
    }

    function percentOrDash(value) {
      return hasNumber(value) ? percent(value) : `<span class="muted">-</span>`;
    }

    function resultClass(result) {
      if (result === "WIN") return "win";
      if (result === "SKIP") return "skip";
      return "loss";
    }

    function profitClass(value) {
      const n = numberValue(value);
      if (n > 0) return "positive";
      if (n < 0) return "negative";
      return "muted";
    }

    function probCell(value) {
      if (!hasNumber(value)) return `<span class="muted">-</span>`;
      const v = numberValue(value);
      const width = Math.max(0, Math.min(100, v));
      return `<div>${percent(v)}<div class="bar"><span style="width:${width}%"></span></div></div>`;
    }

    function metric(label, value, note = "", tone = "") {
      return `
        <div class="metric">
          <div class="metric-label">${escapeHtml(label)}</div>
          <div class="metric-value ${tone}">${value}</div>
          <div class="metric-note">${escapeHtml(note)}</div>
        </div>`;
    }

    function selectedRunId() {
      return els.runSelect.value || "";
    }

    function renderRunOptions(defaultRunId = "") {
      const current = selectedRunId() || defaultRunId;
      els.runSelect.innerHTML = betRuns.map(run => {
        const profit = run.summary ? money(run.summary.profit) : "";
        const dates = run.date_start && run.date_end ? `${run.date_start} to ${run.date_end}` : "";
        const label = `${run.label} ${profit ? `(${profit})` : ""}`;
        return `<option value="${escapeHtml(run.id)}" title="${escapeHtml(dates)}">${escapeHtml(label)}</option>`;
      }).join("");
      if (betRuns.some(run => run.id === current)) {
        els.runSelect.value = current;
      } else if (defaultRunId) {
        els.runSelect.value = defaultRunId;
      }
      updateExportLink();
    }

    function renderModelRuns() {
      if (!modelRuns.length) {
        els.modelRunList.innerHTML = `<div class="empty">No saved transformer checkpoints found.</div>`;
        return;
      }
      els.modelRunList.innerHTML = modelRuns.slice(0, 12).map(run => {
        const stats = [
          `${run.feature_count || 0} features`,
          run.feature_mode ? `mode ${run.feature_mode}` : "",
          run.train_rows ? `${run.train_rows} train` : "",
          run.validation_rows ? `${run.validation_rows} val` : "",
          run.best_validation_loss ? `loss ${run.best_validation_loss}` : "",
          run.validation_auc ? `AUC ${run.validation_auc}` : ""
        ].filter(Boolean).join(" / ");
        return `<div class="model-row">
          <strong>${escapeHtml(run.id)}</strong>
          <span>${escapeHtml(run.created_at || run.file)}</span>
          <span>${escapeHtml(stats)}</span>
        </div>`;
      }).join("");
    }

    function renderValidationReport() {
      if (!validationReport || !validationReport.fold) {
        els.validationReport.innerHTML = `<div class="empty">No validation report found.</div>`;
        return;
      }
      const fold = validationReport.fold || {};
      const metrics = validationReport.model_metrics || {};
      const audit = validationReport.feature_audit || {};
      const range = fold.target_date_range || {};
      const stats = [
        `${fold.target_rows || 0} games`,
        range.start && range.end ? `${range.start} to ${range.end}` : "",
        hasNumber(metrics.brier) ? `Brier ${numberValue(metrics.brier).toFixed(4)}` : "",
        hasNumber(metrics.auc) ? `AUC ${numberValue(metrics.auc).toFixed(4)}` : "",
        hasNumber(metrics.accuracy_0_50) ? `Acc ${(numberValue(metrics.accuracy_0_50) * 100).toFixed(1)}%` : "",
        `${audit.selected_market_like_count || 0} market inputs`
      ].filter(Boolean).join(" / ");
      els.validationReport.innerHTML = `<div class="model-row">
        <strong>${escapeHtml(validationReport.label || "Walk-forward validation")}</strong>
        <span>${escapeHtml(stats)}</span>
        <span>${escapeHtml(`Train cutoff ${fold.train_cutoff_date || ""}`)}</span>
      </div>`;
    }

    function updateExportLink() {
      els.exportLink.href = `/api/export.csv?run=${encodeURIComponent(selectedRunId())}`;
    }

    async function loadRuns() {
      const response = await fetch("/api/runs", { cache: "no-store" });
      if (!response.ok) throw new Error(`runs HTTP ${response.status}`);
      const data = await response.json();
      betRuns = data.bet_runs || [];
      modelRuns = data.model_runs || [];
      validationReport = data.walk_forward_validation || {};
      renderRunOptions(data.default_run_id || "");
      renderValidationReport();
      renderModelRuns();
    }

    function renderSummary() {
      const profitTone = summary.profit >= 0 ? "positive" : "negative";
      const evaluated = summary.evaluated_games ?? allRows.length;
      const skipNote = Object.entries(summary.skip_reason_counts || {})
        .map(([reason, count]) => `${reason.replace(/_/g, " ")} ${count}`)
        .join(" / ");
      els.summary.innerHTML = [
        metric("Profit", money(summary.profit), `${percent(summary.roi_staked_pct)} ROI on staked`, profitTone),
        metric("Bankroll", money(summary.ending_bankroll), `Start ${money(summary.starting_bankroll)}`),
        metric("Placed Bets", `${summary.bets ?? 0} / ${evaluated}`, `${summary.wins ?? 0} wins / ${summary.losses ?? 0} losses`),
        metric("Skipped", String(summary.skips ?? 0), skipNote || "No skipped decisions", "neutral"),
        metric("Win Rate", percent(summary.win_rate_pct), "Actual bets only"),
        metric("Total Staked", money(summary.total_staked), "Across placed bets"),
        metric("Avg Model", percent(summary.avg_model_prob_pct), "Bet-side probability", "neutral"),
        metric("Avg Edge", percent(summary.avg_edge_vs_break_even_pct), "Vs break-even", "neutral")
      ].join("");
    }

    function renderHeader() {
      els.tableHead.innerHTML = columns.map(col => {
        const active = sort.key === (col.sortKey || col.key);
        const marker = active ? (sort.direction === "asc" ? " ^" : " v") : "";
        return `<th data-key="${col.sortKey || col.key}" data-type="${col.type || "text"}">${escapeHtml(col.label)}${marker}</th>`;
      }).join("");
      els.tableHead.querySelectorAll("th").forEach(th => {
        th.addEventListener("click", () => {
          const key = th.dataset.key;
          const type = th.dataset.type;
          if (sort.key === key) {
            sort.direction = sort.direction === "asc" ? "desc" : "asc";
          } else {
            sort = { key, direction: type === "number" ? "desc" : "asc", type };
          }
          render();
        });
      });
    }

    function applyFilters() {
      const query = els.searchInput.value.trim().toLowerCase();
      const result = els.resultFilter.value;
      const side = els.sideFilter.value;
      const skipReason = els.skipFilter.value;
      const gameType = els.gameTypeFilter.value;
      const minEdge = els.minEdge.value === "" ? null : Number(els.minEdge.value);
      const minStake = els.minStake.value === "" ? null : Number(els.minStake.value);

      filteredRows = allRows.filter(row => {
        if (result !== "all" && row.result !== result) return false;
        if (side !== "all" && row.bet_side !== side) return false;
        if (skipReason !== "all" && row.skip_reason !== skipReason) return false;
        if (gameType !== "all" && row.season_type_text !== gameType) return false;
        if (minEdge !== null && numberValue(row.model_edge_vs_break_even_pct) < minEdge) return false;
        if (minStake !== null && numberValue(row.stake) < minStake) return false;
        if (!query) return true;
        return [
          row.game_date,
          row.season_type_text,
          row.matchup,
          row.display_team,
          row.candidate_team,
          row.bet_team,
          row.actual_winner,
          row.result,
          row.skip_reason_label,
          row.skip_reason,
          row.american_odds_text
        ].join(" ").toLowerCase().includes(query);
      });
    }

    function sortRows(rows) {
      const dir = sort.direction === "asc" ? 1 : -1;
      return [...rows].sort((a, b) => {
        const av = sort.type === "number" ? numberValue(a[sort.key]) : String(a[sort.key] ?? "").toLowerCase();
        const bv = sort.type === "number" ? numberValue(b[sort.key]) : String(b[sort.key] ?? "").toLowerCase();
        if (av < bv) return -1 * dir;
        if (av > bv) return 1 * dir;
        return numberValue(a.bet_number) - numberValue(b.bet_number);
      });
    }

    function renderTable() {
      const rows = sortRows(filteredRows);
      els.tableCount.textContent = `${rows.length.toLocaleString()} shown / ${allRows.length.toLocaleString()} evaluated games`;
      if (!rows.length) {
        els.tableBody.innerHTML = `<tr><td class="empty" colspan="${columns.length}">No games match the current filters.</td></tr>`;
        return;
      }
      els.tableBody.innerHTML = rows.map(row => {
        const cells = columns.map(col => {
          const value = col.render ? col.render(row) : escapeHtml(row[col.key]);
          const cls = col.cls ? ` class="${col.cls}"` : "";
          const title = col.cls && col.cls.includes("cell") ? ` title="${escapeHtml(row[col.key])}"` : "";
          return `<td${cls}${title}>${value}</td>`;
        }).join("");
        return `<tr>${cells}</tr>`;
      }).join("");
    }

    function drawChart() {
      const canvas = els.chart;
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, Math.round(rect.width * dpr));
      canvas.height = Math.max(180, Math.round(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      const values = filteredRows.length ? filteredRows.map(r => numberValue(r.bankroll_after)) : allRows.map(r => numberValue(r.bankroll_after));
      if (!values.length) return;
      const start = summary.starting_bankroll || values[0];
      const series = [start, ...values];
      const min = Math.min(...series);
      const max = Math.max(...series);
      const pad = 18;
      const span = Math.max(1, max - min);

      ctx.strokeStyle = "#d7dee8";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, h - pad);
      ctx.lineTo(w - pad, h - pad);
      ctx.stroke();

      const zeroY = h - pad - ((start - min) / span) * (h - pad * 2);
      ctx.strokeStyle = "#b8c4d2";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pad, zeroY);
      ctx.lineTo(w - pad, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = series[series.length - 1] >= start ? "#168255" : "#c43d4b";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      series.forEach((value, i) => {
        const x = pad + (i / Math.max(1, series.length - 1)) * (w - pad * 2);
        const y = h - pad - ((value - min) / span) * (h - pad * 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = "#66758a";
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText(money(max), pad, 12);
      ctx.fillText(money(min), pad, h - 4);
    }

    function render() {
      applyFilters();
      renderHeader();
      renderTable();
      drawChart();
    }

    async function loadData() {
      els.sourceText.textContent = "Loading bet history...";
      updateExportLink();
      const response = await fetch(`/api/bets?run=${encodeURIComponent(selectedRunId())}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      allRows = data.rows || [];
      summary = data.summary || {};
      const run = data.run || {};
      const range = run.date_start && run.date_end ? `${run.date_start} to ${run.date_end}` : "selected dates";
      els.sourceText.textContent = `${run.label || "Backtest"}: ${(summary.evaluated_games ?? allRows.length).toLocaleString()} games, ${(summary.bets ?? 0).toLocaleString()} placed, ${(summary.skips ?? 0).toLocaleString()} skipped / ${range} / ${data.path || "CSV"}`;
      renderSummary();
      render();
    }

    [els.searchInput, els.resultFilter, els.sideFilter, els.skipFilter, els.gameTypeFilter, els.minEdge, els.minStake].forEach(el => {
      el.addEventListener("input", render);
      el.addEventListener("change", render);
    });
    els.runSelect.addEventListener("change", loadData);
    els.refreshBtn.addEventListener("click", async () => {
      await loadRuns();
      await loadData();
    });
    window.addEventListener("resize", drawChart);

    (async function init() {
      await loadRuns();
      await loadData();
    })().catch(error => {
      els.sourceText.textContent = `Could not load bet history: ${error.message}`;
      els.tableBody.innerHTML = `<tr><td class="empty" colspan="${columns.length}">Could not load bet history.</td></tr>`;
    });
  </script>
</body>
</html>
"""


class BetHistoryHandler(BaseHTTPRequestHandler):
    csv_path: Path = DEFAULT_CSV

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path in {"/", "/index.html"}:
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/runs":
            default_run_id = run_id_for_path(self.csv_path)
            payload = {
                "default_run_id": default_run_id,
                "bet_runs": discover_bet_runs(self.csv_path),
                "model_runs": discover_model_runs(),
                "walk_forward_validation": load_json_file(WALK_FORWARD_REPORT),
            }
            self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            return

        if parsed.path == "/api/bets":
            run_id = qs.get("run", [None])[0]
            csv_path = resolve_bet_run_csv(run_id, self.csv_path)
            run = next((item for item in discover_bet_runs(self.csv_path) if item["id"] == run_id_for_path(csv_path)), {})
            rows = load_rows(csv_path)
            payload = {
                "path": relative_path(csv_path),
                "run": run,
                "summary": summarize(rows),
                "rows": rows,
            }
            self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            return

        if parsed.path == "/api/export.csv":
            run_id = qs.get("run", [None])[0]
            csv_path = resolve_bet_run_csv(run_id, self.csv_path)
            if not csv_path.exists():
                self.send_bytes(b"missing bet history csv\n", "text/plain; charset=utf-8", status=404)
                return
            body = csv_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{csv_path.name}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_bytes(b"not found\n", "text/plain; charset=utf-8", status=404)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    args = parser.parse_args()

    BetHistoryHandler.csv_path = Path(args.csv).resolve()
    server = ThreadingHTTPServer((args.host, args.port), BetHistoryHandler)
    print(f"NBA bet history app running at http://{args.host}:{args.port}", flush=True)
    print(f"Reading {BetHistoryHandler.csv_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
