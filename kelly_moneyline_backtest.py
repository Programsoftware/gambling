from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "backtests"
RAW_DIR = OUT_DIR / "raw_statmuse"
DEFAULT_SEASONS = [20222023, 20232024, 20242025, 20252026]
NHL_WEB = "https://api-web.nhle.com/v1"
COMPLETED_STATES = {"OFF", "FINAL"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}

TEAM_NAMES = [
    "Anaheim Ducks",
    "Arizona Coyotes",
    "Boston Bruins",
    "Buffalo Sabres",
    "Calgary Flames",
    "Carolina Hurricanes",
    "Chicago Blackhawks",
    "Colorado Avalanche",
    "Columbus Blue Jackets",
    "Dallas Stars",
    "Detroit Red Wings",
    "Edmonton Oilers",
    "Florida Panthers",
    "Los Angeles Kings",
    "Minnesota Wild",
    "Montreal Canadiens",
    "Nashville Predators",
    "New Jersey Devils",
    "New York Islanders",
    "New York Rangers",
    "Ottawa Senators",
    "Philadelphia Flyers",
    "Pittsburgh Penguins",
    "San Jose Sharks",
    "Seattle Kraken",
    "St. Louis Blues",
    "Tampa Bay Lightning",
    "Toronto Maple Leafs",
    "Utah Mammoth",
    "Vancouver Canucks",
    "Vegas Golden Knights",
    "Washington Capitals",
    "Winnipeg Jets",
]

TEAM_NAME_TO_ABBREV = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}

TEAM_ABBREVS = [
    "ANA",
    "ARI",
    "BOS",
    "BUF",
    "CGY",
    "CAR",
    "CHI",
    "COL",
    "CBJ",
    "DAL",
    "DET",
    "EDM",
    "FLA",
    "LAK",
    "MIN",
    "MTL",
    "NSH",
    "NJD",
    "NYI",
    "NYR",
    "OTT",
    "PHI",
    "PIT",
    "SJS",
    "SEA",
    "STL",
    "TBL",
    "TOR",
    "UTA",
    "VAN",
    "VGK",
    "WSH",
    "WPG",
]

FEATURE_COLUMNS = [
    "market_home_no_vig_prob",
    "market_home_raw_prob",
    "market_away_raw_prob",
    "home_is_favorite",
    "home_games_played",
    "away_games_played",
    "home_win_pct",
    "away_win_pct",
    "home_last5_win_pct",
    "away_last5_win_pct",
    "home_gf_per_game",
    "away_gf_per_game",
    "home_ga_per_game",
    "away_ga_per_game",
    "home_goal_diff_per_game",
    "away_goal_diff_per_game",
    "home_last5_gf_per_game",
    "away_last5_gf_per_game",
    "home_last5_ga_per_game",
    "away_last5_ga_per_game",
    "home_rest_days",
    "away_rest_days",
    "home_form_edge",
    "home_goal_diff_edge",
    "home_rest_edge",
    "home_market_edge_feature",
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def parse_seasons(value: str | None) -> list[int]:
    if not value:
        return DEFAULT_SEASONS.copy()

    seasons: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(str(start)[:2] + end_text) if len(end_text) == 2 else int(end_text)
            seasons.append(start * 10000 + end)
        else:
            seasons.append(int(item))

    return sorted(set(seasons))


def season_label(season: int) -> str:
    start = int(str(season)[:4])
    end = int(str(season)[4:])
    return f"{start}-{str(end)[-2:]}"


def date_in_season(date_text: str, season: int) -> bool:
    game_date = pd.to_datetime(date_text, errors="coerce")
    if pd.isna(game_date):
        return False

    start = int(str(season)[:4])
    end = int(str(season)[4:])
    return pd.Timestamp(start, 9, 1) <= game_date <= pd.Timestamp(end, 8, 31)


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def no_vig_home_prob(home_odds: int, away_odds: int) -> float:
    home = american_to_implied(home_odds)
    away = american_to_implied(away_odds)
    return home / (home + away)


def parse_american(text: str) -> int | None:
    cleaned = text.replace("−", "-").replace("–", "-").strip()
    if cleaned.lower() in {"even", "ev", "pk"}:
        return 100
    match = re.search(r"[+-]?\d+", cleaned)
    if not match:
        return None
    value = int(match.group())
    if value == 0:
        return None
    return value


def parse_date(text: str) -> str | None:
    text = text.strip()
    for fmt in ["%m/%d/%Y", "%b %d, %Y", "%a, %b %d, %Y"]:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def fetch_text(url: str, path: Path, refresh: bool = False) -> str:
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8", errors="replace")
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8", errors="replace")
    time.sleep(0.12)
    return response.text


def parse_statmuse_rows(
    html: str,
    query: str,
    season: int,
    expected_team: str | None = None,
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return []

    rows: list[dict[str, Any]] = []
    for tr in tables[0].find_all("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all("td")]
        if len(cells) < 22:
            continue

        game_date = parse_date(cells[3])
        side = cells[4].strip()
        side_odds = parse_american(cells[5])
        relation = cells[6].strip()
        opponent = cells[7].strip()
        opponent_odds = parse_american(cells[8])
        result = cells[9].strip()
        result_match = re.match(r"([WL])\s+(\d+)-(\d+)", result)

        if not all([game_date, side, opponent, side_odds, opponent_odds, result_match]):
            continue
        if expected_team and side != expected_team:
            continue
        if not date_in_season(game_date, season):
            continue
        if relation not in {"@", "vs"}:
            continue

        side_goals = int(result_match.group(2))
        opponent_goals = int(result_match.group(3))

        if relation == "@":
            away_team = side
            home_team = opponent
            away_odds = side_odds
            home_odds = opponent_odds
            away_score = side_goals
            home_score = opponent_goals
        else:
            home_team = side
            away_team = opponent
            home_odds = side_odds
            away_odds = opponent_odds
            home_score = side_goals
            away_score = opponent_goals

        rows.append(
            {
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": int(home_odds),
                "away_odds": int(away_odds),
                "home_score": home_score,
                "away_score": away_score,
                "home_win": int(home_score > away_score),
                "season": season,
                "source_query": query,
                "source": "statmuse",
            }
        )

    return rows


def fetch_statmuse_moneylines(
    seasons: list[int] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    seasons = seasons or DEFAULT_SEASONS.copy()
    rows: list[dict[str, Any]] = []
    query_templates = [
        "{team} highest moneyline {season_label}",
        "{team} lowest moneyline {season_label}",
    ]

    for season in seasons:
        label = season_label(season)
        for team in TEAM_NAMES:
            expected_team = TEAM_NAME_TO_ABBREV.get(team)
            for template in query_templates:
                query = template.format(team=team, season_label=label)
                url = "https://www.statmuse.com/nhl/ask/" + requests.utils.quote(query)
                path = RAW_DIR / str(season) / f"{safe_name(query)}.html"
                try:
                    html = fetch_text(url, path, refresh=refresh)
                    rows.extend(parse_statmuse_rows(html, query, season, expected_team=expected_team))
                except Exception as exc:
                    print(f"Could not fetch StatMuse query '{query}': {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["game_date", "home_team", "away_team"])
    df["market_home_raw_prob"] = df["home_odds"].apply(american_to_implied)
    df["market_away_raw_prob"] = df["away_odds"].apply(american_to_implied)
    df["market_home_no_vig_prob"] = df.apply(
        lambda row: no_vig_home_prob(int(row["home_odds"]), int(row["away_odds"])),
        axis=1,
    )
    df["home_decimal_odds"] = df["home_odds"].apply(american_to_decimal)
    df["away_decimal_odds"] = df["away_odds"].apply(american_to_decimal)
    df = df.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)
    return df


def fetch_nhl_schedule(
    seasons: list[int] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    seasons = seasons or DEFAULT_SEASONS.copy()
    season_key = "_".join(str(season) for season in seasons)
    schedule_path = OUT_DIR / f"nhl_completed_games_{season_key}.csv"
    if schedule_path.exists() and not refresh:
        return pd.read_csv(schedule_path)

    games_by_id: dict[int, dict[str, Any]] = {}
    for season in seasons:
        for team in TEAM_ABBREVS:
            url = f"{NHL_WEB}/club-schedule-season/{team}/{season}"
            try:
                response = requests.get(url, headers=HEADERS, timeout=30)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                print(f"Could not fetch NHL schedule for {team} {season}: {exc}")
                continue
            time.sleep(0.04)

            for game in data.get("games", []):
                game_id = game.get("id")
                away = game.get("awayTeam", {})
                home = game.get("homeTeam", {})
                away_score = away.get("score")
                home_score = home.get("score")
                game_date = game.get("gameDate")
                if game_id is None or game_date is None:
                    continue
                if game.get("gameState") not in COMPLETED_STATES:
                    continue
                if away_score is None or home_score is None:
                    continue

                games_by_id[int(game_id)] = {
                    "game_id": int(game_id),
                    "season": season,
                    "game_date": game_date,
                    "game_type": game.get("gameType"),
                    "away_team": away.get("abbrev"),
                    "home_team": home.get("abbrev"),
                    "away_score": int(away_score),
                    "home_score": int(home_score),
                    "home_win": int(int(home_score) > int(away_score)),
                }

    df = pd.DataFrame(games_by_id.values())
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    df.to_csv(schedule_path, index=False)
    return df


def team_history_before(schedule: pd.DataFrame, team: str, game_date: str) -> pd.DataFrame:
    mask = (schedule["game_date"] < game_date) & (
        (schedule["home_team"] == team) | (schedule["away_team"] == team)
    )
    history = schedule[mask].copy()
    if history.empty:
        return history

    history["team_score"] = np.where(history["home_team"] == team, history["home_score"], history["away_score"])
    history["opp_score"] = np.where(history["home_team"] == team, history["away_score"], history["home_score"])
    history["team_win"] = (history["team_score"] > history["opp_score"]).astype(int)
    return history.sort_values("game_date")


def pregame_team_features(schedule: pd.DataFrame, team: str, game_date: str, prefix: str) -> dict[str, float]:
    history = team_history_before(schedule, team, game_date)
    defaults = {
        f"{prefix}_games_played": 0.0,
        f"{prefix}_win_pct": 0.5,
        f"{prefix}_last5_win_pct": 0.5,
        f"{prefix}_gf_per_game": 3.0,
        f"{prefix}_ga_per_game": 3.0,
        f"{prefix}_goal_diff_per_game": 0.0,
        f"{prefix}_last5_gf_per_game": 3.0,
        f"{prefix}_last5_ga_per_game": 3.0,
        f"{prefix}_rest_days": 3.0,
    }
    if history.empty:
        return defaults

    last5 = history.tail(5)
    game_dt = pd.to_datetime(game_date)
    last_game_dt = pd.to_datetime(history["game_date"].iloc[-1])
    rest_days = max(float((game_dt - last_game_dt).days), 0.0)

    return {
        f"{prefix}_games_played": float(len(history)),
        f"{prefix}_win_pct": float(history["team_win"].mean()),
        f"{prefix}_last5_win_pct": float(last5["team_win"].mean()),
        f"{prefix}_gf_per_game": float(history["team_score"].mean()),
        f"{prefix}_ga_per_game": float(history["opp_score"].mean()),
        f"{prefix}_goal_diff_per_game": float((history["team_score"] - history["opp_score"]).mean()),
        f"{prefix}_last5_gf_per_game": float(last5["team_score"].mean()),
        f"{prefix}_last5_ga_per_game": float(last5["opp_score"].mean()),
        f"{prefix}_rest_days": rest_days,
    }


def build_feature_table(odds: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, game in odds.iterrows():
        row = game.to_dict()
        row.update(pregame_team_features(schedule, game["home_team"], game["game_date"], "home"))
        row.update(pregame_team_features(schedule, game["away_team"], game["game_date"], "away"))
        row["home_is_favorite"] = int(game["home_odds"] < game["away_odds"])
        row["home_form_edge"] = row["home_last5_win_pct"] - row["away_last5_win_pct"]
        row["home_goal_diff_edge"] = row["home_goal_diff_per_game"] - row["away_goal_diff_per_game"]
        row["home_rest_edge"] = row["home_rest_days"] - row["away_rest_days"]
        row["home_market_edge_feature"] = row["market_home_no_vig_prob"] - 0.5
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)


def make_model(train_rows: int) -> Pipeline:
    # Logistic regression is steadier early; gradient boosting takes over once the sample is less tiny.
    if train_rows < 180:
        estimator = LogisticRegression(C=0.7, max_iter=2000)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    estimator = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.045,
        max_leaf_nodes=7,
        min_samples_leaf=14,
        l2_regularization=0.15,
        random_state=42,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def walk_forward_predictions(features: pd.DataFrame, requested_games: int, min_train: int) -> pd.DataFrame:
    df = features.dropna(subset=["home_win", "home_odds", "away_odds"]).copy()
    df = df.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)

    candidate_indices: list[int] = []
    for idx, row in df.iterrows():
        train_mask = df["game_date"] < row["game_date"]
        if int(train_mask.sum()) >= min_train:
            candidate_indices.append(idx)

    if not candidate_indices:
        raise RuntimeError("Not enough historical rows to run a walk-forward backtest.")

    test_indices = candidate_indices[-requested_games:]
    predictions: list[dict[str, Any]] = []

    for idx in test_indices:
        row = df.loc[idx]
        train = df[df["game_date"] < row["game_date"]]
        y_train = train["home_win"].astype(int)
        if y_train.nunique() < 2:
            continue

        model = make_model(len(train))
        model.fit(train[FEATURE_COLUMNS], y_train)
        model_prob = float(model.predict_proba(pd.DataFrame([row[FEATURE_COLUMNS]]))[0, 1])
        model_prob = min(max(model_prob, 0.02), 0.98)

        out = row.to_dict()
        out["train_rows"] = int(len(train))
        out["model_home_win_prob"] = model_prob
        out["model_away_win_prob"] = 1 - model_prob
        out["market_home_no_vig_prob"] = float(row["market_home_no_vig_prob"])
        out["market_away_no_vig_prob"] = 1 - float(row["market_home_no_vig_prob"])
        predictions.append(out)

    return pd.DataFrame(predictions)


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    q = 1 - probability
    return max((b * probability - q) / b, 0.0)


def simulate_kelly(
    predictions: pd.DataFrame,
    starting_bankroll: float,
    kelly_multiplier: float,
    max_bet_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bankroll = float(starting_bankroll)
    bets: list[dict[str, Any]] = []

    for _, game in predictions.iterrows():
        home_kelly = kelly_fraction(float(game["model_home_win_prob"]), float(game["home_decimal_odds"]))
        away_kelly = kelly_fraction(float(game["model_away_win_prob"]), float(game["away_decimal_odds"]))

        if home_kelly <= 0 and away_kelly <= 0:
            side = "skip"
            raw_fraction = 0.0
            stake_fraction = 0.0
            stake = 0.0
            profit = 0.0
            won = None
            decimal_odds = None
            model_prob = None
            offered_odds = None
        elif home_kelly >= away_kelly:
            side = "home"
            raw_fraction = home_kelly
            stake_fraction = min(home_kelly * kelly_multiplier, max_bet_fraction)
            stake = bankroll * stake_fraction
            won = int(game["home_win"]) == 1
            decimal_odds = float(game["home_decimal_odds"])
            offered_odds = int(game["home_odds"])
            model_prob = float(game["model_home_win_prob"])
            profit = stake * (decimal_odds - 1) if won else -stake
        else:
            side = "away"
            raw_fraction = away_kelly
            stake_fraction = min(away_kelly * kelly_multiplier, max_bet_fraction)
            stake = bankroll * stake_fraction
            won = int(game["home_win"]) == 0
            decimal_odds = float(game["away_decimal_odds"])
            offered_odds = int(game["away_odds"])
            model_prob = float(game["model_away_win_prob"])
            profit = stake * (decimal_odds - 1) if won else -stake

        bankroll += profit
        bets.append(
            {
                "game_date": game["game_date"],
                "away_team": game["away_team"],
                "home_team": game["home_team"],
                "away_score": game["away_score"],
                "home_score": game["home_score"],
                "bet_side": side,
                "bet_team": game["home_team"] if side == "home" else game["away_team"] if side == "away" else None,
                "offered_american_odds": offered_odds,
                "decimal_odds": decimal_odds,
                "model_prob": model_prob,
                "market_home_no_vig_prob": game["market_home_no_vig_prob"],
                "raw_kelly_fraction": raw_fraction,
                "stake_fraction": stake_fraction,
                "stake": stake,
                "won": won,
                "profit": profit,
                "bankroll_after": bankroll,
            }
        )

    bets_df = pd.DataFrame(bets)
    placed = bets_df[bets_df["bet_side"] != "skip"].copy()
    total_staked = float(placed["stake"].sum()) if not placed.empty else 0.0
    total_profit = float(bets_df["profit"].sum()) if not bets_df.empty else 0.0
    summary = {
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": bankroll,
        "profit": total_profit,
        "roi_on_starting_bankroll": total_profit / starting_bankroll,
        "bets_placed": int(len(placed)),
        "games_evaluated": int(len(bets_df)),
        "win_rate": float(placed["won"].mean()) if not placed.empty else 0.0,
        "total_staked": total_staked,
        "roi_on_staked": total_profit / total_staked if total_staked > 0 else 0.0,
        "kelly_multiplier": kelly_multiplier,
        "max_bet_fraction": max_bet_fraction,
        "max_drawdown": max_drawdown(bets_df["bankroll_after"].tolist(), starting_bankroll),
    }
    return bets_df, summary


def max_drawdown(values: list[float], starting_bankroll: float) -> float:
    peak = starting_bankroll
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, (value - peak) / peak)
    return abs(worst)


def bootstrap_paths(
    bets: pd.DataFrame,
    starting_bankroll: float,
    trials: int,
    games: int,
    seed: int = 42,
) -> dict[str, Any]:
    placed = bets[bets["bet_side"] != "skip"].copy()
    if placed.empty:
        return {}

    rng = np.random.default_rng(seed)
    final_bankrolls: list[float] = []
    for _ in range(trials):
        bankroll = starting_bankroll
        sample = placed.iloc[rng.integers(0, len(placed), size=games)]
        for _, bet in sample.iterrows():
            stake = bankroll * float(bet["stake_fraction"])
            if bool(bet["won"]):
                bankroll += stake * (float(bet["decimal_odds"]) - 1)
            else:
                bankroll -= stake
        final_bankrolls.append(bankroll)

    values = np.array(final_bankrolls)
    return {
        "trials": trials,
        "sampled_games_per_trial": games,
        "mean_final_bankroll": float(values.mean()),
        "median_final_bankroll": float(np.median(values)),
        "p05_final_bankroll": float(np.percentile(values, 5)),
        "p95_final_bankroll": float(np.percentile(values, 95)),
        "probability_profit": float((values > starting_bankroll).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--bankroll", type=float, default=1000)
    parser.add_argument("--min-train", type=int, default=80)
    parser.add_argument("--kelly-multiplier", type=float, default=1.0)
    parser.add_argument("--max-bet-fraction", type=float, default=0.10)
    parser.add_argument(
        "--seasons",
        default=",".join(str(season) for season in DEFAULT_SEASONS),
        help="Comma-separated NHL season ids, e.g. 20222023,20232024,20242025,20252026.",
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    seasons = parse_seasons(args.seasons)
    odds = fetch_statmuse_moneylines(seasons=seasons, refresh=args.refresh)
    if odds.empty:
        raise RuntimeError("No historical odds rows were parsed.")
    odds.to_csv(OUT_DIR / "historical_moneyline_games.csv", index=False)

    schedule = fetch_nhl_schedule(seasons=seasons, refresh=args.refresh)
    features = build_feature_table(odds, schedule)
    features.to_csv(OUT_DIR / "moneyline_feature_table.csv", index=False)

    predictions = walk_forward_predictions(features, args.games, args.min_train)
    predictions.to_csv(OUT_DIR / "moneyline_walk_forward_predictions.csv", index=False)

    bets, summary = simulate_kelly(
        predictions,
        starting_bankroll=args.bankroll,
        kelly_multiplier=args.kelly_multiplier,
        max_bet_fraction=args.max_bet_fraction,
    )
    bets.to_csv(OUT_DIR / "kelly_backtest_bets.csv", index=False)

    summary["historical_odds_rows"] = int(len(odds))
    summary["feature_rows"] = int(len(features))
    summary["prediction_rows"] = int(len(predictions))
    summary["seasons"] = seasons
    summary["bootstrap"] = bootstrap_paths(
        bets,
        starting_bankroll=args.bankroll,
        trials=1000,
        games=min(args.games, max(int(summary["bets_placed"]), 1)),
    )
    summary["created_at"] = datetime.now().isoformat(timespec="seconds")
    summary["notes"] = [
        "Odds are parsed from public StatMuse team moneyline tables.",
        "Rolling features are calculated only from NHL games before each target game date.",
        "Walk-forward training excludes games on the same date as the target game.",
        "Kelly is applied to the model's best positive-EV side and capped by max_bet_fraction.",
    ]

    (OUT_DIR / "kelly_backtest_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
