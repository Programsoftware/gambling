from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "data" / "nba" / "models"
PREDICTION_DIR = ROOT / "data" / "nba" / "predictions"
BACKTEST_DIR = ROOT / "data" / "nba" / "backtests" / "transformer"

CHECKPOINT_JSON = MODEL_DIR / "nba_transformer_latest.json"
FEATURE_TABLE = MODEL_DIR / "nba_historical_feature_table.csv"
CURRENT_PREDICTION = PREDICTION_DIR / "nba_current_prediction.csv"
BACKTEST_PREDICTIONS = BACKTEST_DIR / "nba_transformer_backtest_predictions.csv"

IDENTITY_COLUMNS = [
    "event_id",
    "game_date",
    "away_team",
    "home_team",
    "season_type",
    "round_type",
    "is_playoff",
    "is_finals",
]

SOURCE_PATHS = {
    "current": CURRENT_PREDICTION,
    "backtest": BACKTEST_PREDICTIONS,
    "historical": FEATURE_TABLE,
}

SOURCE_LABELS = {
    "current": "Current prediction",
    "backtest": "Latest backtest rows",
    "historical": "Historical feature table",
}


_data_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_checkpoint_cache: tuple[float, dict] | None = None


def numeric(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def clean_value(value: object) -> object:
    parsed = numeric(value)
    if parsed is not None:
        return round(parsed, 6)
    if value is None or pd.isna(value):
        return None
    return str(value)


def load_checkpoint() -> dict:
    global _checkpoint_cache
    mtime = CHECKPOINT_JSON.stat().st_mtime
    if _checkpoint_cache is None or _checkpoint_cache[0] != mtime:
        _checkpoint_cache = (mtime, json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8")))
    return _checkpoint_cache[1]


def load_source(source: str) -> pd.DataFrame:
    path = SOURCE_PATHS[source]
    mtime = path.stat().st_mtime
    cached = _data_cache.get(source)
    if cached and cached[0] == mtime:
        return cached[1]
    df = pd.read_csv(path)
    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].astype(str)
    if "game_date" in df.columns:
        df["game_date"] = df["game_date"].astype(str)
    df = df.sort_values([col for col in ["game_date", "event_id"] if col in df.columns]).reset_index(drop=True)
    _data_cache[source] = (mtime, df)
    return df


def split_and_stat(feature: str) -> tuple[str | None, str]:
    if not feature.startswith("edge_"):
        return None, feature
    rest = feature[len("edge_") :]
    for split in [
        "strength",
        "series",
        "schedule",
        "style",
        "players",
        "regular",
        "playoffs",
        "last20",
        "last10",
        "last5",
        "last3",
        "h2h",
        "all",
    ]:
        prefix = f"{split}_"
        if rest.startswith(prefix):
            return split, rest[len(prefix) :]
    return None, rest


def group_for_feature(feature: str) -> str:
    if not feature.startswith("edge_"):
        return "Context"
    split, _ = split_and_stat(feature)
    return {
        "all": "Long-history baseline",
        "regular": "Current regular season",
        "playoffs": "Current playoffs",
        "last3": "Recent form",
        "last5": "Recent form",
        "last10": "Recent form",
        "last20": "Recent form",
        "h2h": "Current-season H2H",
        "players": "Player rotation",
        "strength": "Team strength",
        "series": "Series context",
        "schedule": "Schedule/travel",
        "style": "Style matchup",
    }.get(str(split), "Other edges")


def stat_type(stat: str) -> str:
    text = stat.lower()
    if text in {"season_type", "neutral_site", "is_playoff", "is_finals"}:
        return "Context"
    if "series" in text or "clinching" in text or "elimination" in text:
        return "Series"
    if "travel" in text or "rest" in text:
        return "Schedule"
    if "elo" in text or "power" in text or "seed" in text or "rank" in text or "strength" in text:
        return "Strength"
    if "market" in text or "spread" in text or "implied" in text:
        return "Market"
    if "win_pct" in text or text == "games":
        return "Record"
    if "points" in text or "margin" in text or "rating" in text:
        return "Scoring"
    if "field_goal" in text or "efg" in text or "three" in text or "free_throw" in text or "ft_rate" in text:
        return "Shooting"
    if "rebound" in text or "oreb" in text:
        return "Rebounding"
    if "assist" in text or "turnover" in text or "usage" in text or "possessions" in text:
        return "Possession"
    if "steal" in text or "block" in text or "foul" in text:
        return "Defense"
    if "player" in text or "scorer" in text or "minutes" in text or "per36" in text or "plus_minus" in text:
        return "Players"
    return "Other"


def nice_label(feature: str) -> str:
    if feature == "season_type":
        return "Season type"
    if feature == "neutral_site":
        return "Neutral site"
    if feature == "is_playoff":
        return "Playoff flag"
    if feature == "is_finals":
        return "Finals flag"
    split, stat = split_and_stat(feature)
    label = stat.replace("avg_", "average ").replace("_pct", " percentage").replace("_", " ")
    if split:
        return f"{split.replace('last', 'last ')}: {label}"
    return label


def feature_note(feature: str) -> str:
    if not feature.startswith("edge_"):
        return "Fed directly as a context flag."
    split, stat = split_and_stat(feature)
    split_notes = {
        "all": "Long-history baseline across prior games.",
        "regular": "Current NBA regular season only.",
        "playoffs": "Current NBA playoffs only.",
        "h2h": "Current-season head-to-head games only.",
        "players": "Current-season player-game logs before this game.",
        "strength": "Pregame team-strength snapshot: Elo, power rating, seed/rank, schedule strength, quality wins, and market-derived strength where available.",
        "series": "Current playoff series state before this game.",
        "schedule": "Pregame schedule context such as rest and home/away travel flip.",
        "style": "Style matchup features comparing team statistical profiles.",
        "last3": "Last 3 games in the current season.",
        "last5": "Last 5 games in the current season.",
        "last10": "Last 10 games in the current season.",
        "last20": "Last 20 games in the current season.",
    }
    base = split_notes.get(str(split), "Home minus away edge.")
    return f"{base} Model feature is home value minus away value for {stat.replace('_', ' ')}."


def row_for_event(source: str, event_id: str | None) -> pd.Series:
    df = load_source(source)
    if df.empty:
        raise ValueError(f"No rows in {source}")
    if event_id:
        hit = df[df["event_id"].astype(str) == str(event_id)]
        if not hit.empty:
            return hit.iloc[0]
    return df.iloc[-1] if source != "current" else df.iloc[0]


def game_options(source: str) -> list[dict[str, object]]:
    df = load_source(source)
    options = []
    for _, row in df.iterrows():
        away = row.get("away_team", "")
        home = row.get("home_team", "")
        label = f"{row.get('game_date', '')} | {away} at {home}"
        if "model_home_win_prob" in row and pd.notna(row.get("model_home_win_prob")):
            label += f" | model home {float(row.get('model_home_win_prob')) * 100:.1f}%"
        options.append(
            {
                "event_id": str(row.get("event_id", "")),
                "label": label,
                "game_date": str(row.get("game_date", "")),
                "away_team": str(away),
                "home_team": str(home),
            }
        )
    return options


def build_feature_payload(source: str, event_id: str | None) -> dict[str, object]:
    checkpoint = load_checkpoint()
    feature_columns: list[str] = checkpoint["feature_columns"]
    scaler = checkpoint.get("scaler", {})
    means = scaler.get("mean", [])
    stds = scaler.get("std", [])
    medians = scaler.get("median", [])
    row = row_for_event(source, event_id)

    features = []
    for index, feature in enumerate(feature_columns):
        raw = numeric(row.get(feature))
        mean = numeric(means[index]) if index < len(means) else None
        std = numeric(stds[index]) if index < len(stds) else None
        median = numeric(medians[index]) if index < len(medians) else None
        imputed = raw is None
        raw_for_scale = median if imputed else raw
        scaled = None
        if raw_for_scale is not None and mean is not None and std not in (None, 0):
            scaled = (raw_for_scale - mean) / std

        split, stat = split_and_stat(feature)
        home_col = f"home_{split}_{stat}" if split else None
        away_col = f"away_{split}_{stat}" if split else None
        home_value = clean_value(row.get(home_col)) if home_col and home_col in row.index else None
        away_value = clean_value(row.get(away_col)) if away_col and away_col in row.index else None

        features.append(
            {
                "index": index + 1,
                "feature": feature,
                "label": nice_label(feature),
                "group": group_for_feature(feature),
                "stat_type": stat_type(stat),
                "split": split or "context",
                "raw_value": clean_value(raw),
                "scaled_value": round(scaled, 6) if scaled is not None else None,
                "abs_scaled": round(abs(scaled), 6) if scaled is not None else None,
                "mean": round(mean, 6) if mean is not None else None,
                "std": round(std, 6) if std is not None else None,
                "median": round(median, 6) if median is not None else None,
                "imputed": imputed,
                "home_col": home_col,
                "away_col": away_col,
                "home_value": home_value,
                "away_value": away_value,
                "note": feature_note(feature),
            }
        )

    row_meta = {col: clean_value(row.get(col)) for col in IDENTITY_COLUMNS if col in row.index}
    for col in [
        "model_home_win_prob",
        "model_away_win_prob",
        "market_home_no_vig_prob",
        "market_away_no_vig_prob",
        "home_odds",
        "away_odds",
        "market_details",
        "target_home_win",
    ]:
        if col in row.index:
            row_meta[col] = clean_value(row.get(col))

    groups: dict[str, int] = {}
    stat_types: dict[str, int] = {}
    for feature in features:
        groups[str(feature["group"])] = groups.get(str(feature["group"]), 0) + 1
        stat_types[str(feature["stat_type"])] = stat_types.get(str(feature["stat_type"]), 0) + 1

    return {
        "source": source,
        "source_label": SOURCE_LABELS[source],
        "checkpoint_created_at": checkpoint.get("created_at"),
        "training_args": checkpoint.get("training_args", {}),
        "feature_count": len(feature_columns),
        "row_meta": row_meta,
        "groups": groups,
        "stat_types": stat_types,
        "features": features,
        "scaler_note": "Scaled values use the latest saved checkpoint scaler. This is the exact transformer input for the current prediction row.",
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NBA Model Feature Inspector</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --surface: #ffffff;
      --soft: #eef3f7;
      --line: #d7dee7;
      --line-strong: #aebccb;
      --text: #17202c;
      --muted: #66758a;
      --blue: #2367a8;
      --green: #168255;
      --red: #c43d4b;
      --amber: #9f6408;
      --shadow: 0 14px 32px rgba(24, 38, 54, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }

    button, input, select { font: inherit; }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 30;
      background: rgba(244, 246, 248, 0.94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }

    .topbar-inner {
      max-width: 1560px;
      margin: 0 auto;
      padding: 15px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }

    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 760;
      letter-spacing: 0;
    }

    .subtle {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
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
      text-decoration: none;
    }

    .button.primary {
      background: var(--text);
      color: #fff;
      border-color: var(--text);
    }

    .app {
      max-width: 1560px;
      margin: 0 auto;
      padding: 16px 20px 28px;
      display: grid;
      grid-template-columns: 310px minmax(0, 1fr);
      gap: 16px;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      padding: 12px 13px;
      border-bottom: 1px solid var(--line);
      font-weight: 740;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
    }

    .panel-body {
      padding: 13px;
    }

    .sidebar {
      align-self: start;
      position: sticky;
      top: 82px;
      display: grid;
      gap: 12px;
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
      font-weight: 680;
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
      box-shadow: 0 0 0 3px #e8f1fb;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }

    .metric {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      min-height: 76px;
    }

    .metric-label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
    }

    .metric-value {
      margin-top: 7px;
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
      margin-top: 4px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .main {
      min-width: 0;
    }

    .game-strip {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 16px;
    }

    .kv {
      display: grid;
      grid-template-columns: 140px minmax(0, 1fr);
      gap: 6px 10px;
      font-size: 13px;
    }

    .kv .k { color: var(--muted); }
    .kv .v { font-weight: 650; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    .group-list {
      display: grid;
      gap: 7px;
    }

    .group-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfe;
      color: var(--muted);
      font-size: 12px;
    }

    .group-row strong { color: var(--text); }

    .table-toolbar {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    .table-count { color: var(--muted); white-space: nowrap; }

    .table-scroll {
      max-height: calc(100vh - 290px);
      overflow: auto;
      background: #fff;
    }

    table {
      width: 100%;
      min-width: 1450px;
      border-collapse: separate;
      border-spacing: 0;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
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
      box-shadow: inset 0 -1px 0 var(--line);
      white-space: nowrap;
    }

    td {
      font-variant-numeric: tabular-nums;
    }

    .num { text-align: right; white-space: nowrap; }
    .feature-name { max-width: 230px; }
    .feature-name strong { display: block; font-size: 13px; }
    .feature-name span { display: block; color: var(--muted); font-size: 12px; margin-top: 3px; overflow-wrap: anywhere; }
    .note { color: var(--muted); min-width: 260px; max-width: 360px; line-height: 1.35; }

    .pill {
      display: inline-flex;
      align-items: center;
      height: 23px;
      border-radius: 999px;
      padding: 0 8px;
      font-size: 12px;
      font-weight: 720;
      border: 1px solid #cdd8e4;
      background: var(--soft);
      color: #435166;
      white-space: nowrap;
    }

    .positive { color: var(--green); }
    .negative { color: var(--red); }
    .warn { color: var(--amber); }

    .bar {
      height: 5px;
      margin-top: 5px;
      border-radius: 999px;
      background: var(--soft);
      overflow: hidden;
      min-width: 90px;
    }

    .bar span {
      display: block;
      height: 100%;
      background: var(--blue);
      border-radius: inherit;
    }

    .small {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }

    .empty {
      padding: 30px;
      text-align: center;
      color: var(--muted);
    }

    @media (max-width: 1180px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { position: static; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .game-strip { grid-template-columns: 1fr; }
      .table-scroll { max-height: none; }
    }

    @media (max-width: 720px) {
      .topbar-inner { align-items: flex-start; flex-direction: column; }
      .app { padding: 12px; }
      .sidebar { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div>
        <h1>NBA Model Feature Inspector</h1>
        <div class="subtle" id="subtitle">Loading model features...</div>
      </div>
      <button class="button primary" id="refreshBtn" type="button">Refresh</button>
    </div>
  </header>

  <div class="app">
    <aside class="sidebar">
      <section class="panel">
        <div class="panel-header">Game</div>
        <div class="panel-body filters">
          <div class="field">
            <label for="sourceSelect">Source</label>
            <select id="sourceSelect">
              <option value="current">Current prediction</option>
              <option value="backtest">Latest backtest rows</option>
              <option value="historical">Historical feature table</option>
            </select>
          </div>
          <div class="field">
            <label for="gameSearch">Search games</label>
            <input id="gameSearch" type="search" placeholder="Team, date, event id">
          </div>
          <div class="field">
            <label for="gameSelect">Selected game</label>
            <select id="gameSelect"></select>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">Feature Filters</div>
        <div class="panel-body filters">
          <div class="field">
            <label for="featureSearch">Search features</label>
            <input id="featureSearch" type="search" placeholder="win pct, player, last10">
          </div>
          <div class="field">
            <label for="groupFilter">Group</label>
            <select id="groupFilter"><option value="all">All groups</option></select>
          </div>
          <div class="field">
            <label for="typeFilter">Type</label>
            <select id="typeFilter"><option value="all">All types</option></select>
          </div>
          <div class="field">
            <label for="sortSelect">Sort</label>
            <select id="sortSelect">
              <option value="index">Model order</option>
              <option value="abs_scaled">Most unusual scaled value</option>
              <option value="raw_abs">Largest raw edge magnitude</option>
              <option value="group">Group</option>
            </select>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">Groups</div>
        <div class="panel-body">
          <div class="group-list" id="groupList"></div>
        </div>
      </section>
    </aside>

    <main class="main">
      <section class="metric-grid" id="metrics"></section>

      <section class="game-strip">
        <div class="panel">
          <div class="panel-header">Selected Row</div>
          <div class="panel-body kv" id="gameMeta"></div>
        </div>
        <div class="panel">
          <div class="panel-header">How To Read This</div>
          <div class="panel-body small">
            The model is running in edge mode. It feeds context flags plus feature values like home minus away. The home and away columns shown here are explanatory; the transformer receives the raw feature value after standardization. Positive edges favor the home team, negative edges favor the away team.
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="table-toolbar">
          <strong>Every Feature Used By The Model</strong>
          <div class="table-count" id="tableCount"></div>
        </div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Feature</th>
                <th>Group</th>
                <th>Type</th>
                <th>Model Raw Input</th>
                <th>Standardized Input</th>
                <th>Home Value</th>
                <th>Away Value</th>
                <th>Scaler Mean</th>
                <th>Scaler Std</th>
                <th>Meaning</th>
              </tr>
            </thead>
            <tbody id="featureBody"></tbody>
          </table>
        </div>
      </section>
    </main>
  </div>

  <script>
    let games = [];
    let payload = null;
    let filteredFeatures = [];

    const els = {
      subtitle: document.getElementById("subtitle"),
      sourceSelect: document.getElementById("sourceSelect"),
      gameSearch: document.getElementById("gameSearch"),
      gameSelect: document.getElementById("gameSelect"),
      featureSearch: document.getElementById("featureSearch"),
      groupFilter: document.getElementById("groupFilter"),
      typeFilter: document.getElementById("typeFilter"),
      sortSelect: document.getElementById("sortSelect"),
      groupList: document.getElementById("groupList"),
      metrics: document.getElementById("metrics"),
      gameMeta: document.getElementById("gameMeta"),
      featureBody: document.getElementById("featureBody"),
      tableCount: document.getElementById("tableCount"),
      refreshBtn: document.getElementById("refreshBtn")
    };

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[c]));
    }

    function numberValue(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }

    function fmt(value, digits = 3) {
      const n = numberValue(value);
      if (n === null) return value === null || value === undefined || value === "" ? "-" : escapeHtml(value);
      if (Math.abs(n) >= 100) return n.toLocaleString(undefined, { maximumFractionDigits: 1 });
      if (Math.abs(n) >= 10) return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
      return n.toLocaleString(undefined, { maximumFractionDigits: digits });
    }

    function pct(value) {
      const n = numberValue(value);
      return n === null ? "-" : `${(n * 100).toFixed(1)}%`;
    }

    function signedClass(value) {
      const n = numberValue(value);
      if (n === null || Math.abs(n) < 1e-9) return "";
      return n > 0 ? "positive" : "negative";
    }

    function metric(label, value, note = "") {
      return `<div class="metric"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${value}</div><div class="metric-note">${escapeHtml(note)}</div></div>`;
    }

    async function loadGames() {
      const source = els.sourceSelect.value;
      const res = await fetch(`/api/games?source=${encodeURIComponent(source)}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`games ${res.status}`);
      games = (await res.json()).games || [];
      renderGameOptions();
    }

    function renderGameOptions() {
      const query = els.gameSearch.value.trim().toLowerCase();
      const shown = games.filter(g => !query || `${g.event_id} ${g.label}`.toLowerCase().includes(query));
      const current = els.gameSelect.value;
      els.gameSelect.innerHTML = shown.map(g => `<option value="${escapeHtml(g.event_id)}">${escapeHtml(g.label)}</option>`).join("");
      if (shown.some(g => g.event_id === current)) {
        els.gameSelect.value = current;
      }
    }

    async function loadFeatures() {
      const source = els.sourceSelect.value;
      const eventId = els.gameSelect.value || "";
      const res = await fetch(`/api/features?source=${encodeURIComponent(source)}&event_id=${encodeURIComponent(eventId)}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`features ${res.status}`);
      payload = await res.json();
      renderAll();
    }

    function renderMetrics() {
      const meta = payload.row_meta || {};
      const args = payload.training_args || {};
      const modelHome = meta.model_home_win_prob !== undefined ? pct(meta.model_home_win_prob) : "-";
      const modelAway = meta.model_away_win_prob !== undefined ? pct(meta.model_away_win_prob) : "-";
      const marketHome = meta.market_home_no_vig_prob !== undefined ? pct(meta.market_home_no_vig_prob) : "-";
      els.metrics.innerHTML = [
        metric("Features", String(payload.feature_count), args.feature_mode ? `mode ${args.feature_mode}` : "latest checkpoint"),
        metric("Home Model", modelHome, meta.home_team || ""),
        metric("Away Model", modelAway, meta.away_team || ""),
        metric("Market Home", marketHome, meta.market_details || ""),
        metric("Checkpoint", payload.checkpoint_created_at || "-", "latest saved weights")
      ].join("");
    }

    function renderMeta() {
      const meta = payload.row_meta || {};
      const rows = [
        ["Event", meta.event_id],
        ["Date", meta.game_date],
        ["Matchup", `${meta.away_team || ""} at ${meta.home_team || ""}`],
        ["Season type", meta.season_type],
        ["Round", meta.round_type],
        ["Playoff", meta.is_playoff],
        ["Finals", meta.is_finals],
        ["Home odds", meta.home_odds],
        ["Away odds", meta.away_odds]
      ];
      els.gameMeta.innerHTML = rows.map(([k, v]) => `<div class="k">${escapeHtml(k)}</div><div class="v" title="${escapeHtml(v)}">${fmt(v)}</div>`).join("");
    }

    function renderControls() {
      const groups = Object.keys(payload.groups || {}).sort();
      const types = Object.keys(payload.stat_types || {}).sort();
      const currentGroup = els.groupFilter.value;
      const currentType = els.typeFilter.value;
      els.groupFilter.innerHTML = `<option value="all">All groups</option>` + groups.map(g => `<option value="${escapeHtml(g)}">${escapeHtml(g)}</option>`).join("");
      els.typeFilter.innerHTML = `<option value="all">All types</option>` + types.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join("");
      if (groups.includes(currentGroup)) els.groupFilter.value = currentGroup;
      if (types.includes(currentType)) els.typeFilter.value = currentType;
      els.groupList.innerHTML = groups.map(g => `<div class="group-row"><span>${escapeHtml(g)}</span><strong>${payload.groups[g]}</strong></div>`).join("");
    }

    function applyFeatureFilters() {
      const q = els.featureSearch.value.trim().toLowerCase();
      const group = els.groupFilter.value;
      const type = els.typeFilter.value;
      filteredFeatures = (payload.features || []).filter(f => {
        if (group !== "all" && f.group !== group) return false;
        if (type !== "all" && f.stat_type !== type) return false;
        if (!q) return true;
        return `${f.feature} ${f.label} ${f.group} ${f.stat_type} ${f.note}`.toLowerCase().includes(q);
      });

      const sort = els.sortSelect.value;
      filteredFeatures.sort((a, b) => {
        if (sort === "abs_scaled") return (Number(b.abs_scaled) || 0) - (Number(a.abs_scaled) || 0);
        if (sort === "raw_abs") return Math.abs(Number(b.raw_value) || 0) - Math.abs(Number(a.raw_value) || 0);
        if (sort === "group") return `${a.group} ${a.index}`.localeCompare(`${b.group} ${b.index}`);
        return Number(a.index) - Number(b.index);
      });
    }

    function scaledCell(value) {
      const n = numberValue(value);
      if (n === null) return "-";
      const width = Math.min(100, Math.abs(n) / 3 * 100);
      const cls = Math.abs(n) >= 2 ? "warn" : signedClass(n);
      return `<div class="${cls}">${fmt(n, 2)}</div><div class="bar"><span style="width:${width}%"></span></div>`;
    }

    function renderTable() {
      applyFeatureFilters();
      els.tableCount.textContent = `${filteredFeatures.length.toLocaleString()} shown / ${payload.feature_count.toLocaleString()} used`;
      if (!filteredFeatures.length) {
        els.featureBody.innerHTML = `<tr><td class="empty" colspan="11">No features match the current filters.</td></tr>`;
        return;
      }
      els.featureBody.innerHTML = filteredFeatures.map(f => {
        const rawCls = signedClass(f.raw_value);
        return `<tr>
          <td class="num">${f.index}</td>
          <td class="feature-name"><strong>${escapeHtml(f.label)}</strong><span>${escapeHtml(f.feature)}</span></td>
          <td><span class="pill">${escapeHtml(f.group)}</span></td>
          <td><span class="pill">${escapeHtml(f.stat_type)}</span></td>
          <td class="num ${rawCls}">${fmt(f.raw_value)}</td>
          <td class="num">${scaledCell(f.scaled_value)}</td>
          <td class="num">${fmt(f.home_value)}</td>
          <td class="num">${fmt(f.away_value)}</td>
          <td class="num">${fmt(f.mean)}</td>
          <td class="num">${fmt(f.std)}</td>
          <td class="note">${escapeHtml(f.note)}${f.imputed ? " Value was missing and imputed by scaler median." : ""}</td>
        </tr>`;
      }).join("");
    }

    function renderAll() {
      els.subtitle.textContent = `${payload.source_label}: ${payload.row_meta?.away_team || ""} at ${payload.row_meta?.home_team || ""}`;
      renderMetrics();
      renderMeta();
      renderControls();
      renderTable();
    }

    async function reloadEverything() {
      await loadGames();
      await loadFeatures();
    }

    els.sourceSelect.addEventListener("change", async () => {
      await loadGames();
      await loadFeatures();
    });
    els.gameSearch.addEventListener("input", renderGameOptions);
    els.gameSelect.addEventListener("change", loadFeatures);
    [els.featureSearch, els.groupFilter, els.typeFilter, els.sortSelect].forEach(el => {
      el.addEventListener("input", renderTable);
      el.addEventListener("change", renderTable);
    });
    els.refreshBtn.addEventListener("click", reloadEverything);

    reloadEverything().catch(err => {
      els.subtitle.textContent = `Could not load feature inspector: ${err.message}`;
      els.featureBody.innerHTML = `<tr><td class="empty" colspan="11">Could not load features.</td></tr>`;
    });
  </script>
</body>
</html>
"""


class FeatureInspectorHandler(BaseHTTPRequestHandler):
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
        try:
            if parsed.path in {"/", "/index.html"}:
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/games":
                source = qs.get("source", ["current"])[0]
                if source not in SOURCE_PATHS:
                    raise ValueError(f"Unknown source: {source}")
                payload = {"source": source, "label": SOURCE_LABELS[source], "games": game_options(source)}
                self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
                return
            if parsed.path == "/api/features":
                source = qs.get("source", ["current"])[0]
                event_id = qs.get("event_id", [None])[0]
                if source not in SOURCE_PATHS:
                    raise ValueError(f"Unknown source: {source}")
                payload = build_feature_payload(source, event_id)
                self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
                return
            self.send_bytes(b"not found\n", "text/plain; charset=utf-8", status=404)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_bytes(body, "application/json; charset=utf-8", status=500)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), FeatureInspectorHandler)
    print(f"NBA feature inspector running at http://{args.host}:{args.port}", flush=True)
    print(f"Reading checkpoint {CHECKPOINT_JSON}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
