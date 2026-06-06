from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kelly_moneyline_backtest import (
    FEATURE_COLUMNS,
    DEFAULT_SEASONS,
    american_to_decimal,
    american_to_implied,
    build_feature_table,
    fetch_nhl_schedule,
    fetch_statmuse_moneylines,
    kelly_fraction,
    make_model,
    no_vig_home_prob,
    parse_seasons,
    walk_forward_predictions,
)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
BACKTEST_DIR = DATA_ROOT / "backtests"
LATEST_DIR = DATA_ROOT / "latest"
PREDICTION_DIR = DATA_ROOT / "predictions"


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def moneyline_consensus(odds: pd.DataFrame, away_team: str, home_team: str) -> dict[str, Any]:
    moneyline = odds[odds.get("market").astype(str).str.lower().eq("moneyline")].copy()
    moneyline["american_odds"] = pd.to_numeric(moneyline["american_odds"], errors="coerce")
    moneyline = moneyline.dropna(subset=["american_odds", "team"])
    moneyline = moneyline[moneyline["american_odds"].abs().between(50, 1000)]

    out: dict[str, Any] = {}
    for side, team in [("away", away_team), ("home", home_team)]:
        values = moneyline[moneyline["team"].astype(str).str.upper().eq(team)]["american_odds"].astype(float)
        if values.empty:
            raise RuntimeError(f"No moneyline odds found for {team}.")
        out[f"{side}_odds"] = int(round(float(values.median())))
        out[f"{side}_odds_values"] = [int(round(float(value))) for value in values.tolist()]
        out[f"{side}_odds_count"] = int(len(values))

    out["home_decimal_odds"] = american_to_decimal(int(out["home_odds"]))
    out["away_decimal_odds"] = american_to_decimal(int(out["away_odds"]))
    out["market_home_raw_prob"] = american_to_implied(int(out["home_odds"]))
    out["market_away_raw_prob"] = american_to_implied(int(out["away_odds"]))
    out["market_home_no_vig_prob"] = no_vig_home_prob(int(out["home_odds"]), int(out["away_odds"]))
    return out


def normalize_loaded_odds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["home_odds", "away_odds", "home_score", "away_score", "home_win", "season"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["game_date", "home_team", "away_team", "home_odds", "away_odds", "home_win"])
    out["home_odds"] = out["home_odds"].astype(int)
    out["away_odds"] = out["away_odds"].astype(int)
    out["home_win"] = out["home_win"].astype(int)
    out["market_home_raw_prob"] = out["home_odds"].apply(american_to_implied)
    out["market_away_raw_prob"] = out["away_odds"].apply(american_to_implied)
    out["market_home_no_vig_prob"] = out.apply(
        lambda row: no_vig_home_prob(int(row["home_odds"]), int(row["away_odds"])),
        axis=1,
    )
    out["home_decimal_odds"] = out["home_odds"].apply(american_to_decimal)
    out["away_decimal_odds"] = out["away_odds"].apply(american_to_decimal)
    return out.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)


def load_historical_odds(seasons: list[int], refresh: bool) -> pd.DataFrame:
    if refresh:
        return fetch_statmuse_moneylines(seasons=seasons, refresh=True)

    expanded = BACKTEST_DIR / "historical_moneyline_games_expanded.csv"
    standard = BACKTEST_DIR / "historical_moneyline_games.csv"
    path = expanded if expanded.exists() else standard
    if path.exists():
        return normalize_loaded_odds(pd.read_csv(path))

    return fetch_statmuse_moneylines(seasons=seasons, refresh=False)


def previous_processed_runs() -> list[Path]:
    processed_root = DATA_ROOT / "processed"
    if not processed_root.exists():
        return []
    return sorted([path for path in processed_root.iterdir() if path.is_dir()])


def append_recent_snapshot_games(
    odds: pd.DataFrame,
    schedule: pd.DataFrame,
    target_date: str,
    teams: set[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    existing = {
        (str(row.game_date), str(row.home_team), str(row.away_team))
        for row in odds[["game_date", "home_team", "away_team"]].itertuples(index=False)
    }
    schedule_by_id = schedule.set_index("game_id", drop=False)

    for run_dir in previous_processed_runs():
        current_path = run_dir / "current_game.csv"
        odds_path = run_dir / "odds_market_snapshot.csv"
        if not current_path.exists() or not odds_path.exists():
            continue
        try:
            current = pd.read_csv(current_path).iloc[0].to_dict()
        except Exception:
            continue

        game_date = str(current.get("game_date") or "")
        away_team = str(current.get("away_team") or "").upper()
        home_team = str(current.get("home_team") or "").upper()
        game_id = safe_float(current.get("game_id"))
        if not game_date or game_date >= target_date or {away_team, home_team} != teams or game_id is None:
            continue
        key = (game_date, home_team, away_team)
        if key in existing or int(game_id) not in schedule_by_id.index:
            continue

        actual = schedule_by_id.loc[int(game_id)]
        odds_snapshot = pd.read_csv(odds_path)
        prices = moneyline_consensus(odds_snapshot, away_team=away_team, home_team=home_team)
        row = {
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": prices["home_odds"],
            "away_odds": prices["away_odds"],
            "home_score": int(actual["home_score"]),
            "away_score": int(actual["away_score"]),
            "home_win": int(actual["home_win"]),
            "season": int(actual["season"]),
            "source_query": f"processed_snapshot:{run_dir.name}",
            "source": "processed_snapshot",
            "market_home_raw_prob": prices["market_home_raw_prob"],
            "market_away_raw_prob": prices["market_away_raw_prob"],
            "market_home_no_vig_prob": prices["market_home_no_vig_prob"],
            "home_decimal_odds": prices["home_decimal_odds"],
            "away_decimal_odds": prices["away_decimal_odds"],
        }
        rows.append(row)
        existing.add(key)

    if not rows:
        return odds, []
    combined = normalize_loaded_odds(pd.concat([odds, pd.DataFrame(rows)], ignore_index=True))
    return combined, rows


def binary_metrics(df: pd.DataFrame, prob_col: str = "model_home_win_prob") -> dict[str, Any]:
    clean = df.dropna(subset=["home_win", prob_col]).copy()
    if clean.empty:
        return {}
    y = clean["home_win"].astype(float).to_numpy()
    p = clean[prob_col].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    return {
        "rows": int(len(clean)),
        "home_win_rate": float(y.mean()),
        "avg_probability": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy_0_50": float(((p >= 0.5) == y).mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-date", required=True)
    parser.add_argument("--teams", default="VGK,CAR")
    parser.add_argument("--seasons", default=",".join(str(season) for season in DEFAULT_SEASONS))
    parser.add_argument("--min-train", type=int, default=80)
    parser.add_argument("--backtest-games", type=int, default=250)
    parser.add_argument("--refresh-odds", action="store_true")
    parser.add_argument("--refresh-schedule", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    teams = {item.strip().upper() for item in args.teams.split(",") if item.strip()}
    seasons = parse_seasons(args.seasons)

    current_path = LATEST_DIR / "current_game.csv"
    odds_path = LATEST_DIR / "odds_market_snapshot.csv"
    if not current_path.exists() or not odds_path.exists():
        raise RuntimeError("Run collect_nhl_data.py first so data/latest has current_game and odds_market_snapshot.")

    current = pd.read_csv(current_path).iloc[0].to_dict()
    if str(current.get("game_date")) != args.game_date:
        raise RuntimeError(f"Latest current_game is {current.get('game_date')}, expected {args.game_date}.")

    away_team = str(current.get("away_team") or "").upper()
    home_team = str(current.get("home_team") or "").upper()
    if {away_team, home_team} != teams:
        raise RuntimeError(f"Latest current_game teams are {away_team},{home_team}, expected {sorted(teams)}.")

    schedule = fetch_nhl_schedule(seasons=seasons, refresh=args.refresh_schedule)
    odds = load_historical_odds(seasons=seasons, refresh=args.refresh_odds)
    odds, appended_rows = append_recent_snapshot_games(odds, schedule, args.game_date, teams)
    odds.to_csv(BACKTEST_DIR / "historical_moneyline_games_latest_for_prediction.csv", index=False)

    features = build_feature_table(odds, schedule)
    features.to_csv(BACKTEST_DIR / "moneyline_feature_table_latest_for_prediction.csv", index=False)
    train = features[(features["game_date"] < args.game_date) & features["home_win"].notna()].copy()
    if len(train) < args.min_train:
        raise RuntimeError(f"Only {len(train)} training rows; need at least {args.min_train}.")

    y_train = train["home_win"].astype(int)
    model = make_model(len(train))
    model.fit(train[FEATURE_COLUMNS], y_train)

    odds_snapshot = pd.read_csv(odds_path)
    current_prices = moneyline_consensus(odds_snapshot, away_team=away_team, home_team=home_team)
    current_row = {
        "game_date": args.game_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_odds": current_prices["home_odds"],
        "away_odds": current_prices["away_odds"],
        "home_score": np.nan,
        "away_score": np.nan,
        "home_win": np.nan,
        "season": int(current.get("season") or seasons[-1]),
        "source_query": "data/latest/odds_market_snapshot.csv",
        "source": "current_market_snapshot",
        "market_home_raw_prob": current_prices["market_home_raw_prob"],
        "market_away_raw_prob": current_prices["market_away_raw_prob"],
        "market_home_no_vig_prob": current_prices["market_home_no_vig_prob"],
        "home_decimal_odds": current_prices["home_decimal_odds"],
        "away_decimal_odds": current_prices["away_decimal_odds"],
    }
    current_features = build_feature_table(pd.DataFrame([current_row]), schedule)
    home_prob = float(model.predict_proba(current_features[FEATURE_COLUMNS])[0, 1])
    home_prob = min(max(home_prob, 0.02), 0.98)
    away_prob = 1.0 - home_prob

    home_kelly = kelly_fraction(home_prob, float(current_features.iloc[0]["home_decimal_odds"]))
    away_kelly = kelly_fraction(away_prob, float(current_features.iloc[0]["away_decimal_odds"]))
    predicted_winner = home_team if home_prob >= away_prob else away_team
    if home_kelly <= 0 and away_kelly <= 0:
        bet_side = "none"
        bet_team = None
        bet_edge = 0.0
    elif home_kelly >= away_kelly:
        bet_side = "home"
        bet_team = home_team
        bet_edge = home_prob - american_to_implied(int(current_prices["home_odds"]))
    else:
        bet_side = "away"
        bet_team = away_team
        bet_edge = away_prob - american_to_implied(int(current_prices["away_odds"]))

    backtest_predictions = pd.DataFrame()
    backtest_metrics: dict[str, Any] = {}
    if args.backtest_games > 0:
        backtest_predictions = walk_forward_predictions(features, args.backtest_games, args.min_train)
        backtest_predictions.to_csv(BACKTEST_DIR / "nhl_moneyline_walk_forward_predictions_latest.csv", index=False)
        backtest_metrics = binary_metrics(backtest_predictions)

    prediction = {
        **current_row,
        **{key: value for key, value in current_features.iloc[0].to_dict().items() if key in FEATURE_COLUMNS},
        "model_home_win_prob": home_prob,
        "model_away_win_prob": away_prob,
        "predicted_winner": predicted_winner,
        "recommended_bet_side": bet_side,
        "recommended_bet_team": bet_team,
        "recommended_bet_edge": bet_edge,
        "home_kelly_fraction": home_kelly,
        "away_kelly_fraction": away_kelly,
    }
    prediction_df = pd.DataFrame([prediction])
    prediction_df.to_csv(PREDICTION_DIR / "nhl_tonight_prediction.csv", index=False)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game_date": args.game_date,
        "game_id": current.get("game_id"),
        "start_time_utc": current.get("start_time_utc"),
        "away_team": away_team,
        "home_team": home_team,
        "model_type": model.named_steps["model"].__class__.__name__,
        "training_rows": int(len(train)),
        "historical_odds_rows": int(len(odds)),
        "completed_schedule_rows": int(len(schedule)),
        "recent_snapshot_rows_appended": appended_rows,
        "current_market": {
            "home_odds": int(current_prices["home_odds"]),
            "away_odds": int(current_prices["away_odds"]),
            "home_no_vig_probability": float(current_prices["market_home_no_vig_prob"]),
            "away_no_vig_probability": float(1 - current_prices["market_home_no_vig_prob"]),
            "home_odds_values": current_prices["home_odds_values"],
            "away_odds_values": current_prices["away_odds_values"],
        },
        "prediction": {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": predicted_winner,
            "recommended_bet_side": bet_side,
            "recommended_bet_team": bet_team,
            "recommended_bet_edge": bet_edge,
            "home_kelly_fraction": home_kelly,
            "away_kelly_fraction": away_kelly,
        },
        "backtest_metrics": backtest_metrics,
        "artifacts": {
            "prediction_csv": str(PREDICTION_DIR / "nhl_tonight_prediction.csv"),
            "summary_json": str(PREDICTION_DIR / "nhl_tonight_prediction_summary.json"),
            "training_feature_table": str(BACKTEST_DIR / "moneyline_feature_table_latest_for_prediction.csv"),
            "historical_odds": str(BACKTEST_DIR / "historical_moneyline_games_latest_for_prediction.csv"),
            "backtest_predictions": str(BACKTEST_DIR / "nhl_moneyline_walk_forward_predictions_latest.csv"),
        },
    }
    (PREDICTION_DIR / "nhl_tonight_prediction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
