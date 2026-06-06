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

TEAM_SITE_INFO = {
    "ANA": (33.8079, -117.8765, -8.0),
    "ARI": (33.4255, -111.9400, -7.0),
    "BOS": (42.3662, -71.0621, -5.0),
    "BUF": (42.8750, -78.8764, -5.0),
    "CAR": (35.8033, -78.7218, -5.0),
    "CBJ": (39.9693, -83.0060, -5.0),
    "CGY": (51.0374, -114.0519, -7.0),
    "CHI": (41.8807, -87.6742, -6.0),
    "COL": (39.7487, -105.0077, -7.0),
    "DAL": (32.7905, -96.8103, -6.0),
    "DET": (42.3410, -83.0550, -5.0),
    "EDM": (53.5461, -113.4978, -7.0),
    "EIS": (52.5059, 13.4432, 1.0),
    "FLA": (26.1585, -80.3256, -5.0),
    "LAK": (34.0430, -118.2673, -8.0),
    "MIN": (44.9447, -93.1011, -6.0),
    "MTL": (45.4960, -73.5693, -5.0),
    "MUN": (48.2188, 11.6247, 1.0),
    "NJD": (40.7336, -74.1711, -5.0),
    "NSH": (36.1592, -86.7785, -6.0),
    "NYI": (40.7118, -73.7278, -5.0),
    "NYR": (40.7505, -73.9934, -5.0),
    "OTT": (45.2968, -75.9272, -5.0),
    "PHI": (39.9012, -75.1719, -5.0),
    "PIT": (40.4395, -79.9892, -5.0),
    "SCB": (46.9580, 7.4640, 1.0),
    "SEA": (47.6221, -122.3541, -8.0),
    "SJS": (37.3328, -121.9012, -8.0),
    "STL": (38.6268, -90.2026, -6.0),
    "TBL": (27.9427, -82.4518, -5.0),
    "TOR": (43.6435, -79.3791, -5.0),
    "UTA": (40.7683, -111.9011, -7.0),
    "VAN": (49.2778, -123.1088, -8.0),
    "VGK": (36.1028, -115.1783, -8.0),
    "WPG": (49.8927, -97.1437, -6.0),
    "WSH": (38.8981, -77.0209, -5.0),
}

TEAM_FEATURE_NAMES = [
    "games_played",
    "win_pct",
    "last5_win_pct",
    "last10_win_pct",
    "last20_win_pct",
    "gf_per_game",
    "ga_per_game",
    "goal_diff_per_game",
    "last5_gf_per_game",
    "last5_ga_per_game",
    "last5_goal_diff_per_game",
    "last10_gf_per_game",
    "last10_ga_per_game",
    "last10_goal_diff_per_game",
    "last20_goal_diff_per_game",
    "season_games_played",
    "season_win_pct",
    "season_goal_diff_per_game",
    "season_last10_win_pct",
    "season_last10_goal_diff_per_game",
    "type_games_played",
    "type_win_pct",
    "type_goal_diff_per_game",
    "home_split_win_pct",
    "home_split_goal_diff_per_game",
    "away_split_win_pct",
    "away_split_goal_diff_per_game",
    "current_venue_win_pct",
    "current_venue_goal_diff_per_game",
    "rest_days",
    "back_to_back",
    "rested_3plus",
    "games_last7",
    "games_last14",
    "three_in_four",
    "win_streak",
    "loss_streak",
    "goal_diff_std_last10",
    "recent_scoring_trend",
    "recent_defense_trend",
    "prev_game_was_home",
    "same_site_as_last_game",
    "travel_km_since_last_game",
    "timezone_shift_since_last_game",
    "timezone_shift_abs_since_last_game",
    "travel_km_last7",
    "travel_km_last14",
    "timezone_shift_abs_last7",
    "timezone_shift_abs_last14",
    "road_trip_length",
    "home_stand_length",
    "current_site_timezone_offset",
]

EDGE_FEATURE_NAMES = [
    "home_games_edge",
    "home_win_pct_edge",
    "home_last5_win_pct_edge",
    "home_last10_win_pct_edge",
    "home_last20_win_pct_edge",
    "home_gf_edge",
    "home_ga_edge",
    "home_goal_diff_edge",
    "home_last5_goal_diff_edge",
    "home_last10_goal_diff_edge",
    "home_last20_goal_diff_edge",
    "home_season_win_pct_edge",
    "home_season_goal_diff_edge",
    "home_type_win_pct_edge",
    "home_type_goal_diff_edge",
    "home_current_venue_win_pct_edge",
    "home_current_venue_goal_diff_edge",
    "home_rest_edge",
    "home_workload_last7_edge",
    "home_streak_edge",
    "home_elo_edge",
    "home_travel_since_last_edge",
    "home_travel_last7_edge",
    "home_travel_last14_edge",
    "home_timezone_shift_edge",
    "home_road_trip_edge",
    "home_home_stand_edge",
    "home_same_site_edge",
    "home_market_edge_feature",
]

H2H_FEATURE_NAMES = [
    "h2h_games",
    "h2h_home_win_pct",
    "h2h_home_goal_diff_per_game",
    "h2h_home_last5_win_pct",
    "h2h_home_last5_goal_diff_per_game",
    "h2h_season_games",
    "h2h_season_home_win_pct",
    "h2h_season_home_goal_diff_per_game",
    "h2h_type_games",
    "h2h_type_home_win_pct",
    "h2h_type_home_goal_diff_per_game",
    "h2h_home_series_wins",
    "h2h_away_series_wins",
    "h2h_series_game_number",
    "h2h_home_won_previous_meeting",
    "h2h_home_goal_diff_previous_meeting",
]

FEATURE_COLUMNS = [
    "market_home_no_vig_prob",
    "market_home_raw_prob",
    "market_away_raw_prob",
    "home_is_favorite",
    "is_playoff",
    "home_elo_rating",
    "away_elo_rating",
    "home_elo_win_prob",
] + [f"{side}_{name}" for side in ("home", "away") for name in TEAM_FEATURE_NAMES] + H2H_FEATURE_NAMES + EDGE_FEATURE_NAMES


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
    history["is_home"] = (history["home_team"] == team).astype(int)
    return history.sort_values(["game_date", "game_id"])


def mean_or_default(values: pd.Series, default: float) -> float:
    if values.empty:
        return default
    out = float(values.mean())
    return out if np.isfinite(out) else default


def std_or_default(values: pd.Series, default: float) -> float:
    if len(values) < 2:
        return default
    out = float(values.std(ddof=0))
    return out if np.isfinite(out) else default


def streak_features(history: pd.DataFrame) -> tuple[float, float]:
    if history.empty:
        return 0.0, 0.0
    wins = history["team_win"].astype(int).tolist()
    last = wins[-1]
    length = 0
    for value in reversed(wins):
        if value != last:
            break
        length += 1
    return (float(length), 0.0) if last == 1 else (0.0, float(length))


def window_goal_diff(history: pd.DataFrame, games: int) -> pd.Series:
    if history.empty:
        return pd.Series(dtype=float)
    frame = history.tail(games)
    return frame["team_score"].astype(float) - frame["opp_score"].astype(float)


def split_stats(history: pd.DataFrame, is_home: bool) -> tuple[float, float]:
    split = history[history["is_home"] == int(is_home)]
    if split.empty:
        return 0.5, 0.0
    goal_diff = split["team_score"].astype(float) - split["opp_score"].astype(float)
    return mean_or_default(split["team_win"].astype(float), 0.5), mean_or_default(goal_diff, 0.0)


def count_games_since(history: pd.DataFrame, game_date: str, days: int) -> float:
    if history.empty:
        return 0.0
    dates = pd.to_datetime(history["game_date"], errors="coerce")
    target = pd.to_datetime(game_date)
    return float(((dates >= target - pd.Timedelta(days=days)) & (dates < target)).sum())


def team_site_info(team: str) -> tuple[float, float, float]:
    return TEAM_SITE_INFO.get(str(team), (39.8283, -98.5795, -6.0))


def site_distance_km(from_site: str | None, to_site: str | None) -> float:
    if not from_site or not to_site or from_site == to_site:
        return 0.0
    lat1, lon1, _ = team_site_info(from_site)
    lat2, lon2, _ = team_site_info(to_site)
    phi1, phi2 = np.radians([lat1, lat2])
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    return float(6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def site_timezone_offset(site: str | None) -> float:
    return float(team_site_info(str(site))[2]) if site else -6.0


def travel_window_load(history: pd.DataFrame, game_date: str, days: int, current_site_team: str) -> tuple[float, float]:
    if history.empty:
        return 0.0, 0.0
    target = pd.to_datetime(game_date)
    dates = pd.to_datetime(history["game_date"], errors="coerce")
    frame = history[(dates >= target - pd.Timedelta(days=days)) & (dates < target)].sort_values(["game_date", "game_id"])
    sites = [str(site) for site in frame["home_team"].tolist()] + [current_site_team]
    if len(sites) < 2:
        return 0.0, 0.0
    distance = 0.0
    timezone_shift = 0.0
    for previous_site, next_site in zip(sites, sites[1:], strict=False):
        distance += site_distance_km(previous_site, next_site)
        timezone_shift += abs(site_timezone_offset(next_site) - site_timezone_offset(previous_site))
    return float(distance), float(timezone_shift)


def venue_streak_lengths(history: pd.DataFrame, current_is_home: bool) -> tuple[float, float]:
    home_stand = 1.0 if current_is_home else 0.0
    road_trip = 0.0 if current_is_home else 1.0
    expected = int(current_is_home)
    for value in reversed(history["is_home"].astype(int).tolist()):
        if value != expected:
            break
        if current_is_home:
            home_stand += 1.0
        else:
            road_trip += 1.0
    return road_trip, home_stand


def pregame_team_features(
    schedule: pd.DataFrame,
    team: str,
    game_date: str,
    prefix: str,
    season: int | None = None,
    game_type: int | None = None,
    current_is_home: bool = False,
    current_site_team: str | None = None,
) -> dict[str, float]:
    history = team_history_before(schedule, team, game_date)
    current_site_team = current_site_team or team
    defaults = {
        f"{prefix}_{name}": 0.0 for name in TEAM_FEATURE_NAMES
    }
    for name in TEAM_FEATURE_NAMES:
        if "win_pct" in name:
            defaults[f"{prefix}_{name}"] = 0.5
        elif name.endswith("gf_per_game") or name.endswith("ga_per_game"):
            defaults[f"{prefix}_{name}"] = 3.0
    defaults[f"{prefix}_rest_days"] = 3.0
    defaults[f"{prefix}_current_venue_win_pct"] = 0.5
    defaults[f"{prefix}_goal_diff_std_last10"] = 1.5
    defaults[f"{prefix}_current_site_timezone_offset"] = site_timezone_offset(current_site_team)
    if history.empty:
        return defaults

    last5 = history.tail(5)
    last10 = history.tail(10)
    last20 = history.tail(20)
    season_history = history[history["season"].astype(str) == str(season)] if season is not None and "season" in history else pd.DataFrame()
    type_history = history[history["game_type"].astype(str) == str(game_type)] if game_type is not None and "game_type" in history else pd.DataFrame()
    game_dt = pd.to_datetime(game_date)
    last_game_dt = pd.to_datetime(history["game_date"].iloc[-1])
    rest_days = max(float((game_dt - last_game_dt).days), 0.0)
    all_goal_diff = history["team_score"].astype(float) - history["opp_score"].astype(float)
    last5_goal_diff = window_goal_diff(history, 5)
    last10_goal_diff = window_goal_diff(history, 10)
    last20_goal_diff = window_goal_diff(history, 20)
    season_last10 = season_history.tail(10)
    season_goal_diff = season_history["team_score"].astype(float) - season_history["opp_score"].astype(float) if not season_history.empty else pd.Series(dtype=float)
    season_last10_goal_diff = window_goal_diff(season_history, 10)
    type_goal_diff = type_history["team_score"].astype(float) - type_history["opp_score"].astype(float) if not type_history.empty else pd.Series(dtype=float)
    home_split_win, home_split_gd = split_stats(history, True)
    away_split_win, away_split_gd = split_stats(history, False)
    venue_win, venue_gd = split_stats(history, current_is_home)
    win_streak, loss_streak = streak_features(history)
    last5_gf = mean_or_default(last5["team_score"].astype(float), 3.0)
    last5_ga = mean_or_default(last5["opp_score"].astype(float), 3.0)
    season_gf = mean_or_default(season_history["team_score"].astype(float), mean_or_default(history["team_score"].astype(float), 3.0)) if not season_history.empty else mean_or_default(history["team_score"].astype(float), 3.0)
    season_ga = mean_or_default(season_history["opp_score"].astype(float), mean_or_default(history["opp_score"].astype(float), 3.0)) if not season_history.empty else mean_or_default(history["opp_score"].astype(float), 3.0)
    previous_site_team = str(history["home_team"].iloc[-1])
    previous_was_home = float(int(history["is_home"].iloc[-1]))
    timezone_shift = site_timezone_offset(current_site_team) - site_timezone_offset(previous_site_team)
    travel_last7, timezone_last7 = travel_window_load(history, game_date, 7, current_site_team)
    travel_last14, timezone_last14 = travel_window_load(history, game_date, 14, current_site_team)
    road_trip_length, home_stand_length = venue_streak_lengths(history, current_is_home)

    return {
        f"{prefix}_games_played": float(len(history)),
        f"{prefix}_win_pct": mean_or_default(history["team_win"].astype(float), 0.5),
        f"{prefix}_last5_win_pct": mean_or_default(last5["team_win"].astype(float), 0.5),
        f"{prefix}_last10_win_pct": mean_or_default(last10["team_win"].astype(float), 0.5),
        f"{prefix}_last20_win_pct": mean_or_default(last20["team_win"].astype(float), 0.5),
        f"{prefix}_gf_per_game": mean_or_default(history["team_score"].astype(float), 3.0),
        f"{prefix}_ga_per_game": mean_or_default(history["opp_score"].astype(float), 3.0),
        f"{prefix}_goal_diff_per_game": mean_or_default(all_goal_diff, 0.0),
        f"{prefix}_last5_gf_per_game": last5_gf,
        f"{prefix}_last5_ga_per_game": last5_ga,
        f"{prefix}_last5_goal_diff_per_game": mean_or_default(last5_goal_diff, 0.0),
        f"{prefix}_last10_gf_per_game": mean_or_default(last10["team_score"].astype(float), 3.0),
        f"{prefix}_last10_ga_per_game": mean_or_default(last10["opp_score"].astype(float), 3.0),
        f"{prefix}_last10_goal_diff_per_game": mean_or_default(last10_goal_diff, 0.0),
        f"{prefix}_last20_goal_diff_per_game": mean_or_default(last20_goal_diff, 0.0),
        f"{prefix}_season_games_played": float(len(season_history)) if not season_history.empty else 0.0,
        f"{prefix}_season_win_pct": mean_or_default(season_history["team_win"].astype(float), 0.5) if not season_history.empty else 0.5,
        f"{prefix}_season_goal_diff_per_game": mean_or_default(season_goal_diff, 0.0),
        f"{prefix}_season_last10_win_pct": mean_or_default(season_last10["team_win"].astype(float), 0.5) if not season_last10.empty else 0.5,
        f"{prefix}_season_last10_goal_diff_per_game": mean_or_default(season_last10_goal_diff, 0.0),
        f"{prefix}_type_games_played": float(len(type_history)) if not type_history.empty else 0.0,
        f"{prefix}_type_win_pct": mean_or_default(type_history["team_win"].astype(float), 0.5) if not type_history.empty else 0.5,
        f"{prefix}_type_goal_diff_per_game": mean_or_default(type_goal_diff, 0.0),
        f"{prefix}_home_split_win_pct": home_split_win,
        f"{prefix}_home_split_goal_diff_per_game": home_split_gd,
        f"{prefix}_away_split_win_pct": away_split_win,
        f"{prefix}_away_split_goal_diff_per_game": away_split_gd,
        f"{prefix}_current_venue_win_pct": venue_win,
        f"{prefix}_current_venue_goal_diff_per_game": venue_gd,
        f"{prefix}_rest_days": rest_days,
        f"{prefix}_back_to_back": float(rest_days <= 1.0),
        f"{prefix}_rested_3plus": float(rest_days >= 3.0),
        f"{prefix}_games_last7": count_games_since(history, game_date, 7),
        f"{prefix}_games_last14": count_games_since(history, game_date, 14),
        f"{prefix}_three_in_four": float(count_games_since(history, game_date, 4) >= 3),
        f"{prefix}_win_streak": win_streak,
        f"{prefix}_loss_streak": loss_streak,
        f"{prefix}_goal_diff_std_last10": std_or_default(last10_goal_diff, 1.5),
        f"{prefix}_recent_scoring_trend": last5_gf - season_gf,
        f"{prefix}_recent_defense_trend": season_ga - last5_ga,
        f"{prefix}_prev_game_was_home": previous_was_home,
        f"{prefix}_same_site_as_last_game": float(previous_site_team == current_site_team),
        f"{prefix}_travel_km_since_last_game": site_distance_km(previous_site_team, current_site_team),
        f"{prefix}_timezone_shift_since_last_game": timezone_shift,
        f"{prefix}_timezone_shift_abs_since_last_game": abs(timezone_shift),
        f"{prefix}_travel_km_last7": travel_last7,
        f"{prefix}_travel_km_last14": travel_last14,
        f"{prefix}_timezone_shift_abs_last7": timezone_last7,
        f"{prefix}_timezone_shift_abs_last14": timezone_last14,
        f"{prefix}_road_trip_length": road_trip_length,
        f"{prefix}_home_stand_length": home_stand_length,
        f"{prefix}_current_site_timezone_offset": site_timezone_offset(current_site_team),
    }


def h2h_features(
    schedule: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: str,
    season: int | None = None,
    game_type: int | None = None,
) -> dict[str, float]:
    mask = (
        (schedule["game_date"] < game_date)
        & (
            ((schedule["home_team"] == home_team) & (schedule["away_team"] == away_team))
            | ((schedule["home_team"] == away_team) & (schedule["away_team"] == home_team))
        )
    )
    history = schedule[mask].sort_values("game_date").copy()
    defaults = {
        "h2h_games": 0.0,
        "h2h_home_win_pct": 0.5,
        "h2h_home_goal_diff_per_game": 0.0,
        "h2h_home_last5_win_pct": 0.5,
        "h2h_home_last5_goal_diff_per_game": 0.0,
        "h2h_season_games": 0.0,
        "h2h_season_home_win_pct": 0.5,
        "h2h_season_home_goal_diff_per_game": 0.0,
        "h2h_type_games": 0.0,
        "h2h_type_home_win_pct": 0.5,
        "h2h_type_home_goal_diff_per_game": 0.0,
        "h2h_home_series_wins": 0.0,
        "h2h_away_series_wins": 0.0,
        "h2h_series_game_number": 1.0,
        "h2h_home_won_previous_meeting": 0.5,
        "h2h_home_goal_diff_previous_meeting": 0.0,
    }
    if history.empty:
        return defaults

    def summarize_frame(frame: pd.DataFrame) -> tuple[float, float, float, pd.Series, pd.Series]:
        if frame.empty:
            return 0.0, 0.5, 0.0, pd.Series(dtype=float), pd.Series(dtype=float)
        home_score = np.where(frame["home_team"] == home_team, frame["home_score"], frame["away_score"])
        away_score = np.where(frame["home_team"] == home_team, frame["away_score"], frame["home_score"])
        home_wins = pd.Series((home_score > away_score).astype(int), dtype=float)
        goal_diff = pd.Series(home_score.astype(float) - away_score.astype(float))
        return float(len(frame)), mean_or_default(home_wins, 0.5), mean_or_default(goal_diff, 0.0), home_wins, goal_diff

    games, win_pct, gd_per_game, home_wins, goal_diff = summarize_frame(history)
    season_history = history[history["season"].astype(str) == str(season)] if season is not None and "season" in history else pd.DataFrame()
    type_history = season_history[season_history["game_type"].astype(str) == str(game_type)] if game_type is not None and not season_history.empty and "game_type" in season_history else pd.DataFrame()
    season_games, season_win_pct, season_gd, _, _ = summarize_frame(season_history)
    type_games, type_win_pct, type_gd, type_home_wins, _ = summarize_frame(type_history)
    previous_home_win = float(home_wins.iloc[-1]) if not home_wins.empty else 0.5
    previous_goal_diff = float(goal_diff.iloc[-1]) if not goal_diff.empty else 0.0
    return {
        "h2h_games": games,
        "h2h_home_win_pct": win_pct,
        "h2h_home_goal_diff_per_game": gd_per_game,
        "h2h_home_last5_win_pct": mean_or_default(home_wins.tail(5).astype(float), 0.5),
        "h2h_home_last5_goal_diff_per_game": mean_or_default(goal_diff.tail(5), 0.0),
        "h2h_season_games": season_games,
        "h2h_season_home_win_pct": season_win_pct,
        "h2h_season_home_goal_diff_per_game": season_gd,
        "h2h_type_games": type_games,
        "h2h_type_home_win_pct": type_win_pct,
        "h2h_type_home_goal_diff_per_game": type_gd,
        "h2h_home_series_wins": float(type_home_wins.sum()) if not type_home_wins.empty else 0.0,
        "h2h_away_series_wins": float((1.0 - type_home_wins).sum()) if not type_home_wins.empty else 0.0,
        "h2h_series_game_number": type_games + 1.0,
        "h2h_home_won_previous_meeting": previous_home_win,
        "h2h_home_goal_diff_previous_meeting": previous_goal_diff,
    }


def build_elo_lookup(schedule: pd.DataFrame) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    lookup: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    ratings = {team: 1500.0 for team in TEAM_ABBREVS}
    home_advantage = 35.0
    for date, day_games in schedule.sort_values(["game_date", "game_id"]).groupby("game_date", sort=True):
        updates: dict[str, float] = {}
        for _, game in day_games.iterrows():
            home = str(game["home_team"])
            away = str(game["away_team"])
            home_rating = ratings.get(home, 1500.0)
            away_rating = ratings.get(away, 1500.0)
            expected_home = 1.0 / (1.0 + 10.0 ** (-((home_rating + home_advantage) - away_rating) / 400.0))
            lookup[(str(date), home, away)] = (home_rating, away_rating, expected_home)
            result = float(int(game["home_win"]))
            margin = abs(float(game["home_score"]) - float(game["away_score"]))
            margin_multiplier = min(1.75, max(0.75, np.log1p(margin)))
            game_type = int(game.get("game_type", 2) or 2)
            k = (24.0 if game_type == 3 else 18.0) * margin_multiplier
            delta = k * (result - expected_home)
            updates[home] = updates.get(home, 0.0) + delta
            updates[away] = updates.get(away, 0.0) - delta
        for team, delta in updates.items():
            ratings[team] = ratings.get(team, 1500.0) + delta
    return lookup


def elo_before_date(schedule: pd.DataFrame, game_date: str, home_team: str, away_team: str) -> tuple[float, float, float]:
    subset = schedule[schedule["game_date"] < game_date]
    lookup = build_elo_lookup(subset)
    if subset.empty:
        return 1500.0, 1500.0, 0.5
    ratings = {team: 1500.0 for team in TEAM_ABBREVS}
    for _, game in subset.sort_values(["game_date", "game_id"]).iterrows():
        key = (str(game["game_date"]), str(game["home_team"]), str(game["away_team"]))
        home_rating, away_rating, expected_home = lookup.get(key, (ratings.get(str(game["home_team"]), 1500.0), ratings.get(str(game["away_team"]), 1500.0), 0.5))
        result = float(int(game["home_win"]))
        margin = abs(float(game["home_score"]) - float(game["away_score"]))
        game_type = int(game.get("game_type", 2) or 2)
        k = (24.0 if game_type == 3 else 18.0) * min(1.75, max(0.75, np.log1p(margin)))
        delta = k * (result - expected_home)
        ratings[str(game["home_team"])] = home_rating + delta
        ratings[str(game["away_team"])] = away_rating - delta
    home_rating = ratings.get(home_team, 1500.0)
    away_rating = ratings.get(away_team, 1500.0)
    expected_home = 1.0 / (1.0 + 10.0 ** (-((home_rating + 35.0) - away_rating) / 400.0))
    return home_rating, away_rating, expected_home


def game_context(schedule: pd.DataFrame, game: pd.Series) -> tuple[int | None, int | None]:
    game_date = str(game["game_date"])
    home_team = str(game["home_team"])
    away_team = str(game["away_team"])
    match = schedule[
        (schedule["game_date"] == game_date)
        & (schedule["home_team"] == home_team)
        & (schedule["away_team"] == away_team)
    ]
    if match.empty:
        season = int(game["season"]) if "season" in game and pd.notna(game["season"]) else None
        game_type = int(game["game_type"]) if "game_type" in game and pd.notna(game["game_type"]) else None
        return season, game_type
    row = match.iloc[0]
    return int(row["season"]) if pd.notna(row.get("season")) else None, int(row["game_type"]) if pd.notna(row.get("game_type")) else None


def build_feature_table(odds: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    elo_lookup = build_elo_lookup(schedule)
    for _, game in odds.iterrows():
        row = game.to_dict()
        season, game_type = game_context(schedule, game)
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        game_date = str(game["game_date"])
        row["season"] = season if season is not None else row.get("season")
        row["game_type"] = game_type
        row["is_playoff"] = float(game_type == 3)
        row.update(
            pregame_team_features(
                schedule,
                home_team,
                game_date,
                "home",
                season=season,
                game_type=game_type,
                current_is_home=True,
                current_site_team=home_team,
            )
        )
        row.update(
            pregame_team_features(
                schedule,
                away_team,
                game_date,
                "away",
                season=season,
                game_type=game_type,
                current_is_home=False,
                current_site_team=home_team,
            )
        )
        elo_values = elo_lookup.get((game_date, home_team, away_team))
        if elo_values is None:
            elo_values = elo_before_date(schedule, game_date, home_team, away_team)
        home_elo, away_elo, home_elo_prob = elo_values
        row["home_elo_rating"] = home_elo
        row["away_elo_rating"] = away_elo
        row["home_elo_win_prob"] = home_elo_prob
        row.update(h2h_features(schedule, home_team, away_team, game_date, season=season, game_type=game_type))
        row["home_is_favorite"] = int(game["home_odds"] < game["away_odds"])
        row["home_form_edge"] = row["home_last5_win_pct"] - row["away_last5_win_pct"]
        row["home_games_edge"] = row["home_games_played"] - row["away_games_played"]
        row["home_win_pct_edge"] = row["home_win_pct"] - row["away_win_pct"]
        row["home_last5_win_pct_edge"] = row["home_last5_win_pct"] - row["away_last5_win_pct"]
        row["home_last10_win_pct_edge"] = row["home_last10_win_pct"] - row["away_last10_win_pct"]
        row["home_last20_win_pct_edge"] = row["home_last20_win_pct"] - row["away_last20_win_pct"]
        row["home_gf_edge"] = row["home_gf_per_game"] - row["away_gf_per_game"]
        row["home_ga_edge"] = row["away_ga_per_game"] - row["home_ga_per_game"]
        row["home_goal_diff_edge"] = row["home_goal_diff_per_game"] - row["away_goal_diff_per_game"]
        row["home_last5_goal_diff_edge"] = row["home_last5_goal_diff_per_game"] - row["away_last5_goal_diff_per_game"]
        row["home_last10_goal_diff_edge"] = row["home_last10_goal_diff_per_game"] - row["away_last10_goal_diff_per_game"]
        row["home_last20_goal_diff_edge"] = row["home_last20_goal_diff_per_game"] - row["away_last20_goal_diff_per_game"]
        row["home_season_win_pct_edge"] = row["home_season_win_pct"] - row["away_season_win_pct"]
        row["home_season_goal_diff_edge"] = row["home_season_goal_diff_per_game"] - row["away_season_goal_diff_per_game"]
        row["home_type_win_pct_edge"] = row["home_type_win_pct"] - row["away_type_win_pct"]
        row["home_type_goal_diff_edge"] = row["home_type_goal_diff_per_game"] - row["away_type_goal_diff_per_game"]
        row["home_current_venue_win_pct_edge"] = row["home_current_venue_win_pct"] - row["away_current_venue_win_pct"]
        row["home_current_venue_goal_diff_edge"] = row["home_current_venue_goal_diff_per_game"] - row["away_current_venue_goal_diff_per_game"]
        row["home_rest_edge"] = row["home_rest_days"] - row["away_rest_days"]
        row["home_workload_last7_edge"] = row["away_games_last7"] - row["home_games_last7"]
        row["home_streak_edge"] = (row["home_win_streak"] - row["home_loss_streak"]) - (row["away_win_streak"] - row["away_loss_streak"])
        row["home_elo_edge"] = home_elo - away_elo
        row["home_travel_since_last_edge"] = row["away_travel_km_since_last_game"] - row["home_travel_km_since_last_game"]
        row["home_travel_last7_edge"] = row["away_travel_km_last7"] - row["home_travel_km_last7"]
        row["home_travel_last14_edge"] = row["away_travel_km_last14"] - row["home_travel_km_last14"]
        row["home_timezone_shift_edge"] = row["away_timezone_shift_abs_since_last_game"] - row["home_timezone_shift_abs_since_last_game"]
        row["home_road_trip_edge"] = row["away_road_trip_length"] - row["home_road_trip_length"]
        row["home_home_stand_edge"] = row["home_home_stand_length"] - row["away_home_stand_length"]
        row["home_same_site_edge"] = row["home_same_site_as_last_game"] - row["away_same_site_as_last_game"]
        row["home_market_edge_feature"] = row["market_home_no_vig_prob"] - 0.5
        rows.append(row)

    df = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
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
    min_edge: float = 0.0,
    max_market_disagreement: float | None = None,
    market_blend: float = 0.0,
    min_model_prob: float = 0.0,
    side_filter: str = "all",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bankroll = float(starting_bankroll)
    bets: list[dict[str, Any]] = []

    for _, game in predictions.iterrows():
        model_home_prob = float(game["model_home_win_prob"])
        market_home_prob = float(game["market_home_no_vig_prob"])
        home_prob = market_blend * market_home_prob + (1.0 - market_blend) * model_home_prob
        away_prob = 1.0 - home_prob
        home_kelly = kelly_fraction(home_prob, float(game["home_decimal_odds"]))
        away_kelly = kelly_fraction(away_prob, float(game["away_decimal_odds"]))
        home_break_even = american_to_implied(int(game["home_odds"]))
        away_break_even = american_to_implied(int(game["away_odds"]))
        home_edge = home_prob - home_break_even
        away_edge = away_prob - away_break_even
        home_disagreement = abs(home_prob - market_home_prob)
        away_disagreement = abs(away_prob - (1.0 - market_home_prob))

        if home_kelly <= 0 and away_kelly <= 0:
            side = "skip"
            skip_reason = "no_positive_kelly"
            raw_fraction = 0.0
            stake_fraction = 0.0
            stake = 0.0
            profit = 0.0
            won = None
            decimal_odds = None
            model_prob = None
            break_even_prob = None
            edge = None
            market_prob = None
            offered_odds = None
        elif home_kelly >= away_kelly:
            side = "home"
            skip_reason = ""
            raw_fraction = home_kelly
            stake_fraction = min(home_kelly * kelly_multiplier, max_bet_fraction)
            stake = bankroll * stake_fraction
            won = int(game["home_win"]) == 1
            decimal_odds = float(game["home_decimal_odds"])
            offered_odds = int(game["home_odds"])
            model_prob = home_prob
            break_even_prob = home_break_even
            edge = home_edge
            market_prob = market_home_prob
            disagreement = home_disagreement
            profit = stake * (decimal_odds - 1) if won else -stake
        else:
            side = "away"
            skip_reason = ""
            raw_fraction = away_kelly
            stake_fraction = min(away_kelly * kelly_multiplier, max_bet_fraction)
            stake = bankroll * stake_fraction
            won = int(game["home_win"]) == 0
            decimal_odds = float(game["away_decimal_odds"])
            offered_odds = int(game["away_odds"])
            model_prob = away_prob
            break_even_prob = away_break_even
            edge = away_edge
            market_prob = 1.0 - market_home_prob
            disagreement = away_disagreement
            profit = stake * (decimal_odds - 1) if won else -stake

        if side != "skip":
            if side_filter in {"home", "away"} and side != side_filter:
                skip_reason = f"side_filter_{side_filter}"
            elif edge is not None and edge < min_edge:
                skip_reason = "below_min_edge"
            elif model_prob is not None and model_prob < min_model_prob:
                skip_reason = "below_min_model_prob"
            elif max_market_disagreement is not None and disagreement > max_market_disagreement:
                skip_reason = "market_disagreement"
            if skip_reason:
                side = "skip"
                raw_fraction = 0.0
                stake_fraction = 0.0
                stake = 0.0
                profit = 0.0
                won = None

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
                "market_prob": market_prob,
                "break_even_prob": break_even_prob,
                "model_edge_vs_break_even": edge,
                "model_edge_vs_market": None if model_prob is None or market_prob is None else model_prob - market_prob,
                "market_home_no_vig_prob": game["market_home_no_vig_prob"],
                "raw_kelly_fraction": raw_fraction,
                "stake_fraction": stake_fraction,
                "stake": stake,
                "won": won,
                "profit": profit,
                "bankroll_after": bankroll,
                "skip_reason": skip_reason,
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
        "min_edge": min_edge,
        "max_market_disagreement": max_market_disagreement,
        "market_blend": market_blend,
        "min_model_prob": min_model_prob,
        "side_filter": side_filter,
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
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--max-market-disagreement", type=float, default=-1.0)
    parser.add_argument("--market-blend", type=float, default=0.0)
    parser.add_argument("--min-model-prob", type=float, default=0.0)
    parser.add_argument("--side-filter", choices=["all", "home", "away"], default="all")
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
        min_edge=args.min_edge,
        max_market_disagreement=None if args.max_market_disagreement < 0 else args.max_market_disagreement,
        market_blend=args.market_blend,
        min_model_prob=args.min_model_prob,
        side_filter=args.side_filter,
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
