from __future__ import annotations

import json
import math
import re
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from difflib import get_close_matches
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


SEASON = 20252026
SEASON_START_YEAR = 2025
GAME_DATE = "2026-06-02"
GAME_DATE_COMPACT = GAME_DATE.replace("-", "")
GAME_TYPE_REGULAR = 2
GAME_TYPE_PLAYOFFS = 3
GAME_TYPES = [GAME_TYPE_REGULAR, GAME_TYPE_PLAYOFFS]
TEAMS = ["VGK", "CAR"]
COMPLETED_GAME_STATES = {"FINAL", "OFF"}
INJURY_COLUMNS = [
    "team",
    "athlete_id",
    "player",
    "status",
    "type",
    "short_comment",
    "long_comment",
    "date",
]
TEAM_SLUGS = {
    "CAR": "carolina-hurricanes",
    "VGK": "vegas-golden-knights",
}
ESPN_TEAM_IDS = {
    "CAR": "7",
    "VGK": "37",
}

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RAW_DIR = DATA_ROOT / "raw" / RUN_ID
PROCESSED_DIR = DATA_ROOT / "processed" / RUN_ID
LATEST_DIR = DATA_ROOT / "latest"

NHL_WEB = "https://api-web.nhle.com/v1"
NHL_STATS = "https://api.nhle.com/stats/rest/en"
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

sources: list[dict[str, Any]] = []
artifacts: list[dict[str, Any]] = []


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, LATEST_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_")


def clean_name(name: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.replace("`", "'").replace("’", "'")
    return re.sub(r"\s+", " ", normalized).strip()


def default_json(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=default_json),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def write_df(name: str, df: pd.DataFrame) -> Path:
    run_path = PROCESSED_DIR / f"{name}.csv"
    latest_path = LATEST_DIR / f"{name}.csv"
    df.to_csv(run_path, index=False)
    df.to_csv(latest_path, index=False)
    artifacts.append(
        {
            "name": name,
            "run_path": str(run_path.relative_to(ROOT)),
            "latest_path": str(latest_path.relative_to(ROOT)),
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
    )
    return run_path


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [df for df in frames if df is not None and not df.empty]
    if not usable:
        return pd.DataFrame()
    return pd.concat(usable, ignore_index=True)


def request_text(name: str, url: str, raw_filename: str, sleep_s: float = 0.0) -> str | None:
    if sleep_s:
        time.sleep(sleep_s)

    raw_path = RAW_DIR / raw_filename
    record = {
        "name": name,
        "url": url,
        "raw_path": str(raw_path.relative_to(ROOT)),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        response = SESSION.get(url, timeout=35)
        record["status_code"] = response.status_code
        record["content_type"] = response.headers.get("content-type")
        record["bytes"] = len(response.content)
        response.raise_for_status()
        text = response.text
        write_text(raw_path, text)
        record["ok"] = True
        return text
    except Exception as exc:
        record["ok"] = False
        record["error"] = str(exc)
        return None
    finally:
        sources.append(record)


def request_json(name: str, url: str, raw_filename: str, sleep_s: float = 0.0) -> Any | None:
    text = request_text(name, url, raw_filename, sleep_s=sleep_s)
    if text is None:
        return None

    try:
        data = json.loads(text)
        write_json(RAW_DIR / raw_filename, data)
        return data
    except Exception as exc:
        sources[-1]["ok"] = False
        sources[-1]["error"] = f"JSON parse failed: {exc}"
        return None


def request_csv(name: str, url: str, raw_filename: str, sleep_s: float = 0.0) -> pd.DataFrame:
    text = request_text(name, url, raw_filename, sleep_s=sleep_s)
    if text is None:
        return pd.DataFrame()

    try:
        return pd.read_csv(StringIO(text))
    except Exception as exc:
        sources[-1]["ok"] = False
        sources[-1]["error"] = f"CSV parse failed: {exc}"
        return pd.DataFrame()


def ref_url(ref: str | None) -> str | None:
    if not ref:
        return None
    return ref.replace("http://", "https://")


def request_ref(name: str, ref: str | None, raw_filename: str) -> Any | None:
    url = ref_url(ref)
    if not url:
        return None
    return request_json(name, url, raw_filename, sleep_s=0.05)


def nested_default(value: dict[str, Any] | None, key: str = "default") -> Any:
    if isinstance(value, dict):
        return value.get(key) or value.get("default")
    return value


def parse_price(value: Any) -> dict[str, Any]:
    text = str(value).strip()
    out: dict[str, Any] = {
        "raw_price": text,
        "price_type": None,
        "american_odds": None,
        "decimal_odds": None,
        "implied_probability": None,
    }

    if not text:
        return out

    try:
        number = float(text.replace("+", ""))
    except ValueError:
        return out

    if text.startswith("+") or text.startswith("-"):
        american = int(number)
        out["price_type"] = "american"
        out["american_odds"] = american
        if american > 0:
            out["decimal_odds"] = 1 + american / 100
            out["implied_probability"] = 100 / (american + 100)
        else:
            out["decimal_odds"] = 1 + 100 / abs(american)
            out["implied_probability"] = abs(american) / (abs(american) + 100)
        return out

    if number > 1:
        out["price_type"] = "decimal"
        out["decimal_odds"] = number
        out["implied_probability"] = 1 / number
        if number >= 2:
            out["american_odds"] = round((number - 1) * 100)
        else:
            out["american_odds"] = round(-100 / (number - 1))

    return out


def matchup_game_from_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    games: list[dict[str, Any]] = []
    for week in schedule.get("gameWeek", []):
        games.extend(week.get("games", []))

    for game in games:
        away = game.get("awayTeam", {}).get("abbrev")
        home = game.get("homeTeam", {}).get("abbrev")
        if {away, home} == set(TEAMS):
            return game

    raise RuntimeError(f"No {TEAMS} matchup found in NHL schedule for {GAME_DATE}.")


def normalize_current_game(game: dict[str, Any]) -> pd.DataFrame:
    row = {
        "game_id": game.get("id"),
        "season": game.get("season"),
        "game_type": game.get("gameType"),
        "game_date": game.get("gameDate") or GAME_DATE,
        "start_time_utc": game.get("startTimeUTC"),
        "game_state": game.get("gameState"),
        "venue": nested_default(game.get("venue")),
        "venue_timezone": game.get("venueTimezone"),
        "away_team": game.get("awayTeam", {}).get("abbrev"),
        "home_team": game.get("homeTeam", {}).get("abbrev"),
        "away_team_name": nested_default(game.get("awayTeam", {}).get("commonName")),
        "home_team_name": nested_default(game.get("homeTeam", {}).get("commonName")),
    }
    return pd.DataFrame([row])


def normalize_nhl_schedule_odds(game: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for side_key, home_away in [("awayTeam", "away"), ("homeTeam", "home")]:
        team = game.get(side_key, {})
        for item in team.get("odds", []) or []:
            price = parse_price(item.get("value"))
            rows.append(
                {
                    "source": "nhl_schedule",
                    "provider_id": item.get("providerId"),
                    "provider_name": None,
                    "market": "moneyline",
                    "team": team.get("abbrev"),
                    "home_away": home_away,
                    **price,
                }
            )
    return pd.DataFrame(rows)


def normalize_espn_odds(odds_data: dict[str, Any] | None, game: dict[str, Any]) -> pd.DataFrame:
    if not odds_data:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    away_abbrev = game.get("awayTeam", {}).get("abbrev")
    home_abbrev = game.get("homeTeam", {}).get("abbrev")

    for item in odds_data.get("items", []):
        provider = item.get("provider", {}) or {}
        base = {
            "source": "espn_core_odds",
            "provider_id": provider.get("id"),
            "provider_name": provider.get("name"),
            "details": item.get("details"),
            "total": item.get("overUnder"),
            "over_odds": item.get("overOdds"),
            "under_odds": item.get("underOdds"),
            "spread": item.get("spread"),
        }
        for side, abbrev in [("away", away_abbrev), ("home", home_abbrev)]:
            side_odds = item.get(f"{side}TeamOdds", {}) or {}
            moneyline = side_odds.get("moneyLine")
            spread_odds = side_odds.get("spreadOdds")
            rows.append(
                {
                    **base,
                    "market": "moneyline",
                    "team": abbrev,
                    "home_away": side,
                    "favorite": side_odds.get("favorite"),
                    "underdog": side_odds.get("underdog"),
                    **parse_price(moneyline),
                }
            )
            rows.append(
                {
                    **base,
                    "market": "puckline_spread_odds",
                    "team": abbrev,
                    "home_away": side,
                    "favorite": side_odds.get("favorite"),
                    "underdog": side_odds.get("underdog"),
                    **parse_price(spread_odds),
                }
            )

    return pd.DataFrame(rows)


def roster_rows(team: str, roster: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not roster:
        return []

    rows: list[dict[str, Any]] = []
    for group, player_type in [
        ("forwards", "skater"),
        ("defensemen", "skater"),
        ("goalies", "goalie"),
    ]:
        for player in roster.get(group, []) or []:
            name = f"{nested_default(player.get('firstName'))} {nested_default(player.get('lastName'))}"
            rows.append(
                {
                    "team": team,
                    "player_id": player.get("id"),
                    "player": clean_name(name),
                    "position": player.get("positionCode"),
                    "player_type": player_type,
                    "sweater_number": player.get("sweaterNumber"),
                    "shoots_catches": player.get("shootsCatches"),
                    "height_inches": player.get("heightInInches"),
                    "weight_pounds": player.get("weightInPounds"),
                    "birth_date": player.get("birthDate"),
                    "birth_country": player.get("birthCountry"),
                    "headshot": player.get("headshot"),
                }
            )
    return rows


def match_player(name: str, roster: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = clean_name(name).lower()
    for player in roster:
        if clean_name(player["player"]).lower() == target:
            return player

    cleaned = [clean_name(player["player"]) for player in roster]
    hit = get_close_matches(clean_name(name), cleaned, n=1, cutoff=0.78)
    if not hit:
        return None

    return next(player for player in roster if clean_name(player["player"]) == hit[0])


def parse_dailyfaceoff_lines(team: str, html: str | None, roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    candidate_names = re.findall(r"\b[A-Z][A-Za-z.'-]+ [A-Z][A-Za-z.'-]+\b", text)

    seen: set[Any] = set()
    matched: list[dict[str, Any]] = []
    for candidate in candidate_names:
        player = match_player(candidate, roster)
        if not player:
            continue
        if player["player_id"] in seen:
            continue
        seen.add(player["player_id"])
        matched.append(player)

    forwards = [p for p in matched if p.get("position") not in ["D", "G"]]
    defense = [p for p in matched if p.get("position") == "D"]
    goalies = [p for p in matched if p.get("position") == "G"]

    rows: list[dict[str, Any]] = []
    for idx, player in enumerate(forwards[:12]):
        rows.append(
            {
                "team": team,
                "unit_type": "forward",
                "unit": f"F{idx // 3 + 1}",
                "slot": idx % 3 + 1,
                "player_id": player["player_id"],
                "player": player["player"],
                "position": player["position"],
                "source": "dailyfaceoff_line_combinations",
            }
        )

    for idx, player in enumerate(defense[:6]):
        rows.append(
            {
                "team": team,
                "unit_type": "defense",
                "unit": f"D{idx // 2 + 1}",
                "slot": idx % 2 + 1,
                "player_id": player["player_id"],
                "player": player["player"],
                "position": player["position"],
                "source": "dailyfaceoff_line_combinations",
            }
        )

    for idx, player in enumerate(goalies[:3]):
        rows.append(
            {
                "team": team,
                "unit_type": "goalie",
                "unit": "G",
                "slot": idx + 1,
                "player_id": player["player_id"],
                "player": player["player"],
                "position": player["position"],
                "source": "dailyfaceoff_line_combinations",
            }
        )

    return rows


def parse_starting_goalie_mentions(html: str | None, rosters: pd.DataFrame) -> pd.DataFrame:
    if not html or rosters.empty:
        return pd.DataFrame()

    text = BeautifulSoup(html, "html.parser").get_text(" ")
    rows: list[dict[str, Any]] = []
    goalies = rosters[rosters["position"] == "G"].copy()

    for _, goalie in goalies.iterrows():
        name = str(goalie["player"])
        idx = text.lower().find(name.lower())
        if idx < 0:
            continue
        snippet = re.sub(r"\s+", " ", text[max(0, idx - 140) : idx + 260]).strip()
        rows.append(
            {
                "team": goalie["team"],
                "player_id": goalie["player_id"],
                "player": name,
                "source": "dailyfaceoff_starting_goalies",
                "first_mention_index": idx,
                "context": snippet,
            }
        )

    return pd.DataFrame(rows).sort_values(["team", "first_mention_index"])


def flatten_stats_report(data: dict[str, Any] | None, report: str, game_type: int) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data.get("data", []))
    if df.empty:
        return df
    df.insert(0, "game_type", game_type)
    df.insert(0, "source_report", report)

    team_cols = [c for c in ["teamAbbrevs", "teamAbbrev", "team", "teamCode"] if c in df.columns]
    if team_cols:
        col = team_cols[0]
        mask = df[col].astype(str).str.upper().apply(lambda value: any(t in value.split(",") or t in value for t in TEAMS))
        df = df[mask].copy()
    return df


def flatten_club_stats(team: str, game_type: int, data: dict[str, Any] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not data:
        return pd.DataFrame(), pd.DataFrame()

    skaters = pd.DataFrame(data.get("skaters", []))
    goalies = pd.DataFrame(data.get("goalies", []))
    for df in [skaters, goalies]:
        if not df.empty:
            df.insert(0, "game_type", game_type)
            df.insert(0, "team", team)
    return skaters, goalies


def normalize_moneypuck(entity: str, season_type: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out.insert(0, "season_type", season_type)
    out.insert(0, "entity", entity)
    if "team" in out.columns:
        out = out[out["team"].astype(str).str.upper().isin(TEAMS)].copy()
    return out


def normalize_schedule(team: str, schedule: dict[str, Any] | None) -> pd.DataFrame:
    if not schedule:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for game in schedule.get("games", []):
        away = game.get("awayTeam", {})
        home = game.get("homeTeam", {})
        away_abbrev = away.get("abbrev")
        home_abbrev = home.get("abbrev")
        is_home = home_abbrev == team
        team_block = home if is_home else away
        opp_block = away if is_home else home
        gf = team_block.get("score")
        ga = opp_block.get("score")
        rows.append(
            {
                "team": team,
                "game_id": game.get("id"),
                "season": game.get("season"),
                "game_type": game.get("gameType"),
                "game_date": game.get("gameDate"),
                "start_time_utc": game.get("startTimeUTC"),
                "game_state": game.get("gameState"),
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "opponent": opp_block.get("abbrev"),
                "is_home": is_home,
                "venue": nested_default(game.get("venue")),
                "goals_for": gf,
                "goals_against": ga,
                "result": None if gf is None or ga is None else ("W" if gf > ga else "L"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["game_date_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["team", "game_date_dt", "game_id"])
    df["days_since_previous"] = df.groupby("team")["game_date_dt"].diff().dt.days
    return df.drop(columns=["game_date_dt"])


def head_to_head_from_schedules(schedule_rows: pd.DataFrame) -> pd.DataFrame:
    if schedule_rows.empty:
        return pd.DataFrame()
    h2h = schedule_rows[
        schedule_rows["home_team"].isin(TEAMS)
        & schedule_rows["away_team"].isin(TEAMS)
    ].copy()
    return h2h.drop_duplicates(subset=["game_id"]).sort_values("game_date")


def flatten_player_logs(raw_logs: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for team, players in raw_logs.items():
        for player_id, payload in players.items():
            player = payload["player"]
            for game_type, game_logs in payload["logs_by_game_type"].items():
                for game in game_logs:
                    row = {
                        "team": team,
                        "player_id": player_id,
                        "player": player["player"],
                        "position": player["position"],
                        "game_type": int(game_type),
                    }
                    row.update(game)
                    rows.append(row)
    return pd.DataFrame(rows)


def toi_to_minutes(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value)
    if ":" not in text:
        try:
            return float(text) / 60
        except ValueError:
            return None
    minutes, seconds = text.split(":", 1)
    try:
        return float(minutes) + float(seconds) / 60
    except ValueError:
        return None


def summarize_player_logs(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame()

    df = logs.copy()
    df["gameDate_sort"] = pd.to_datetime(df.get("gameDate"), errors="coerce")
    for col in ["goals", "assists", "points", "shots", "hits", "blockedShots"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    toi_col = "toi" if "toi" in df.columns else "timeOnIce" if "timeOnIce" in df.columns else None
    if toi_col:
        df["toi_minutes"] = df[toi_col].apply(toi_to_minutes)
    else:
        df["toi_minutes"] = math.nan

    rows: list[dict[str, Any]] = []
    group_cols = ["team", "player_id", "player", "position", "game_type"]
    for keys, group in df.sort_values("gameDate_sort", ascending=False).groupby(group_cols):
        last5 = group.head(5)
        last10 = group.head(10)
        base = dict(zip(group_cols, keys, strict=False))
        base.update(
            {
                "games": len(group),
                "goals_per_game": group["goals"].mean(),
                "assists_per_game": group["assists"].mean(),
                "points_per_game": group["points"].mean(),
                "shots_per_game": group["shots"].mean(),
                "hits_per_game": group["hits"].mean(),
                "blocks_per_game": group["blockedShots"].mean(),
                "toi_minutes_per_game": group["toi_minutes"].mean(),
                "last5_goals_per_game": last5["goals"].mean(),
                "last5_points_per_game": last5["points"].mean(),
                "last5_shots_per_game": last5["shots"].mean(),
                "last10_goals_per_game": last10["goals"].mean(),
                "last10_points_per_game": last10["points"].mean(),
                "last10_shots_per_game": last10["shots"].mean(),
            }
        )
        rows.append(base)

    return pd.DataFrame(rows)


def normalize_injuries(team: str, injury_index: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not injury_index:
        return []

    rows: list[dict[str, Any]] = []
    raw_items: list[dict[str, Any]] = []
    for item in injury_index.get("items", []):
        injury = request_ref(
            f"espn_injury_{team}",
            item.get("$ref"),
            f"espn_injury_{team}_{safe_name(item.get('$ref', 'item'))}.json",
        )
        if not injury:
            continue

        athlete = request_ref(
            f"espn_injury_athlete_{team}",
            (injury.get("athlete") or {}).get("$ref"),
            f"espn_injury_athlete_{team}_{safe_name((injury.get('athlete') or {}).get('$ref', 'athlete'))}.json",
        )
        raw_items.append({"injury": injury, "athlete": athlete})

        rows.append(
            {
                "team": team,
                "athlete_id": (athlete or {}).get("id"),
                "player": (athlete or {}).get("displayName") or (athlete or {}).get("fullName"),
                "status": nested_default(injury.get("status")),
                "type": nested_default(injury.get("type")),
                "short_comment": injury.get("shortComment"),
                "long_comment": injury.get("longComment"),
                "date": injury.get("date"),
            }
        )

    write_json(RAW_DIR / f"espn_injuries_resolved_{team}.json", raw_items)
    return rows


def flatten_shiftcharts(game_id: int, data: dict[str, Any] | None) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data.get("data", []))
    if df.empty:
        return df
    df.insert(0, "game_id_source", game_id)
    return df


def flatten_play_by_play(game_id: int, data: dict[str, Any] | None) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for play in data.get("plays", []) or []:
        team_abbrev = None
        details = play.get("details") or {}
        if isinstance(details.get("eventOwnerTeamId"), int):
            owner = details.get("eventOwnerTeamId")
            if owner == (data.get("awayTeam") or {}).get("id"):
                team_abbrev = (data.get("awayTeam") or {}).get("abbrev")
            elif owner == (data.get("homeTeam") or {}).get("id"):
                team_abbrev = (data.get("homeTeam") or {}).get("abbrev")

        rows.append(
            {
                "game_id": game_id,
                "play_id": play.get("eventId"),
                "period": (play.get("periodDescriptor") or {}).get("number"),
                "period_type": (play.get("periodDescriptor") or {}).get("periodType"),
                "time_in_period": play.get("timeInPeriod"),
                "time_remaining": play.get("timeRemaining"),
                "event_type": play.get("typeDescKey"),
                "team": team_abbrev,
                "x_coord": details.get("xCoord"),
                "y_coord": details.get("yCoord"),
                "shot_type": details.get("shotType"),
                "zone": details.get("zoneCode"),
                "details": json.dumps(details, default=default_json, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def playoff_and_h2h_game_ids(schedule_rows: pd.DataFrame, current_game_id: int) -> list[int]:
    if schedule_rows.empty:
        return []

    past = schedule_rows[
        (schedule_rows["game_state"].isin(COMPLETED_GAME_STATES))
        & (schedule_rows["game_id"].notna())
        & (schedule_rows["game_id"] != current_game_id)
    ].copy()
    playoff_ids = past[past["game_type"] == GAME_TYPE_PLAYOFFS]["game_id"].astype(int).tolist()
    h2h = head_to_head_from_schedules(schedule_rows)
    h2h_ids = h2h[
        h2h["game_state"].isin(COMPLETED_GAME_STATES)
        & (h2h["game_id"] != current_game_id)
    ]["game_id"].astype(int).tolist()
    return sorted(set(playoff_ids + h2h_ids))


def fetch_news_sources() -> pd.DataFrame:
    urls = {
        "nhl_game1_preview": "https://www.nhl.com/news/topic/playoffs/vegas-golden-knights-carolina-hurricanes-stanley-cup-final-game-1-preview-june-2-2026",
        "nhl_series_preview": "https://www.nhl.com/news/vegas-golden-knights-carolina-hurricanes-2026-stanley-cup-final-series-preview",
        "nhl_final_schedule": "https://www.nhl.com/news/topic/playoffs/2026-stanley-cup-final-schedule-television-results",
        "hurricanes_preview": "https://www.nhl.com/hurricanes/news/preview-scf-game-1-vs-vegas",
        "golden_knights_morning_skate": "https://www.nhl.com/goldenknights/news/morning-skate-report-june-2-2026",
        "ap_game1_preview": "https://apnews.com/article/81a093f7f73f3ce434854caf5693cc48",
        "covers_odds_preview": "https://www.covers.com/nhl/stanley-cup/golden-knights-vs-hurricanes-prediction-picks-best-bets-sgp-tuesday-6-2-2026",
    }
    rows: list[dict[str, Any]] = []
    for name, url in urls.items():
        html = request_text(name, url, f"news_{name}.html", sleep_s=0.1)
        if not html:
            rows.append({"name": name, "url": url, "title": None, "text_chars": 0})
            continue
        soup = BeautifulSoup(html, "html.parser")
        rows.append(
            {
                "name": name,
                "url": url,
                "title": (soup.title.string.strip() if soup.title and soup.title.string else None),
                "text_chars": len(soup.get_text(" ")),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    print(f"Collecting NHL model data into data/raw/{RUN_ID} and data/processed/{RUN_ID}")

    schedule = request_json(
        "nhl_schedule_game_date",
        f"{NHL_WEB}/schedule/{GAME_DATE}",
        f"nhl_schedule_{GAME_DATE}.json",
    )
    if not schedule:
        raise RuntimeError("Could not fetch NHL schedule.")

    game = matchup_game_from_schedule(schedule)
    current_game_id = int(game["id"])
    write_df("current_game", normalize_current_game(game))

    odds_frames = [normalize_nhl_schedule_odds(game)]

    for endpoint in ["landing", "boxscore", "right-rail", "play-by-play"]:
        request_json(
            f"nhl_gamecenter_{endpoint}",
            f"{NHL_WEB}/gamecenter/{current_game_id}/{endpoint}",
            f"nhl_gamecenter_{current_game_id}_{endpoint}.json",
            sleep_s=0.05,
        )

    espn_scoreboard = request_json(
        "espn_scoreboard",
        f"{ESPN_SITE}/scoreboard?dates={GAME_DATE_COMPACT}",
        f"espn_scoreboard_{GAME_DATE_COMPACT}.json",
    )
    espn_event_id = None
    espn_competition_id = None
    if espn_scoreboard:
        for event in espn_scoreboard.get("events", []):
            comp = (event.get("competitions") or [{}])[0]
            comp_teams = {c.get("team", {}).get("abbreviation") for c in comp.get("competitors", [])}
            if comp_teams == set(TEAMS):
                espn_event_id = event.get("id")
                espn_competition_id = comp.get("id")
                break

    if espn_event_id and espn_competition_id:
        espn_odds = request_json(
            "espn_event_odds",
            f"{ESPN_CORE}/events/{espn_event_id}/competitions/{espn_competition_id}/odds?lang=en&region=us",
            f"espn_event_{espn_event_id}_odds.json",
        )
        odds_frames.append(normalize_espn_odds(espn_odds, game))

    odds_df = concat_or_empty(odds_frames)
    write_df("odds_market_snapshot", odds_df)

    rosters: list[dict[str, Any]] = []
    club_schedule_frames: list[pd.DataFrame] = []
    club_stat_skaters: list[pd.DataFrame] = []
    club_stat_goalies: list[pd.DataFrame] = []
    line_rows: list[dict[str, Any]] = []

    roster_by_team: dict[str, list[dict[str, Any]]] = {}
    for team in TEAMS:
        roster_json = request_json(
            f"nhl_roster_{team}",
            f"{NHL_WEB}/roster/{team}/current",
            f"nhl_roster_{team}_current.json",
        )
        team_roster = roster_rows(team, roster_json)
        roster_by_team[team] = team_roster
        rosters.extend(team_roster)

        schedule_json = request_json(
            f"nhl_club_schedule_{team}",
            f"{NHL_WEB}/club-schedule-season/{team}/{SEASON}",
            f"nhl_club_schedule_{team}_{SEASON}.json",
        )
        club_schedule_frames.append(normalize_schedule(team, schedule_json))

        for game_type in GAME_TYPES:
            club_stats = request_json(
                f"nhl_club_stats_{team}_{game_type}",
                f"{NHL_WEB}/club-stats/{team}/{SEASON}/{game_type}",
                f"nhl_club_stats_{team}_{SEASON}_{game_type}.json",
                sleep_s=0.05,
            )
            skaters, goalies = flatten_club_stats(team, game_type, club_stats)
            club_stat_skaters.append(skaters)
            club_stat_goalies.append(goalies)

        lineup_html = request_text(
            f"dailyfaceoff_lines_{team}",
            f"https://www.dailyfaceoff.com/teams/{TEAM_SLUGS[team]}/line-combinations",
            f"dailyfaceoff_{team}_line_combinations.html",
        )
        line_rows.extend(parse_dailyfaceoff_lines(team, lineup_html, team_roster))

    rosters_df = pd.DataFrame(rosters)
    write_df("rosters", rosters_df)
    write_df("projected_lines", pd.DataFrame(line_rows))

    starting_goalies_html = request_text(
        "dailyfaceoff_starting_goalies",
        "https://www.dailyfaceoff.com/starting-goalies",
        "dailyfaceoff_starting_goalies.html",
    )
    write_df("starting_goalie_mentions", parse_starting_goalie_mentions(starting_goalies_html, rosters_df))

    club_schedules = concat_or_empty(club_schedule_frames)
    write_df("team_schedule_games", club_schedules)
    write_df("head_to_head_games", head_to_head_from_schedules(club_schedules))

    write_df(
        "official_club_skaters",
        concat_or_empty(club_stat_skaters),
    )
    write_df(
        "official_club_goalies",
        concat_or_empty(club_stat_goalies),
    )

    stats_frames: dict[str, list[pd.DataFrame]] = {
        "nhl_team_summary": [],
        "nhl_skater_summary": [],
        "nhl_goalie_summary": [],
    }
    for report_name, report_path in [
        ("nhl_team_summary", "team/summary"),
        ("nhl_skater_summary", "skater/summary"),
        ("nhl_goalie_summary", "goalie/summary"),
    ]:
        for game_type in GAME_TYPES:
            data = request_json(
                f"{report_name}_{game_type}",
                f"{NHL_STATS}/{report_path}?limit=-1&cayenneExp=seasonId={SEASON}%20and%20gameTypeId={game_type}",
                f"{report_name}_{SEASON}_{game_type}.json",
                sleep_s=0.05,
            )
            stats_frames[report_name].append(flatten_stats_report(data, report_name, game_type))

    for name, frames in stats_frames.items():
        write_df(name, concat_or_empty(frames))

    mp_frames: dict[str, list[pd.DataFrame]] = {
        "moneypuck_skaters": [],
        "moneypuck_goalies": [],
        "moneypuck_teams": [],
    }
    for season_type in ["regular", "playoffs"]:
        for entity in ["skaters", "goalies", "teams"]:
            df = request_csv(
                f"moneypuck_{season_type}_{entity}",
                f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{SEASON_START_YEAR}/{season_type}/{entity}.csv",
                f"moneypuck_{SEASON_START_YEAR}_{season_type}_{entity}.csv",
                sleep_s=0.05,
            )
            mp_frames[f"moneypuck_{entity}"].append(normalize_moneypuck(entity, season_type, df))

    for name, frames in mp_frames.items():
        write_df(name, concat_or_empty(frames))

    player_logs_raw: dict[str, Any] = {}
    for team, team_roster in roster_by_team.items():
        player_logs_raw[team] = {}
        for player in team_roster:
            player_id = str(player["player_id"])
            player_logs_raw[team][player_id] = {
                "player": player,
                "logs_by_game_type": {},
            }
            for game_type in GAME_TYPES:
                data = request_json(
                    f"nhl_player_game_log_{team}_{player_id}_{game_type}",
                    f"{NHL_WEB}/player/{player_id}/game-log/{SEASON}/{game_type}",
                    f"nhl_player_game_log_{team}_{player_id}_{game_type}.json",
                    sleep_s=0.02,
                )
                player_logs_raw[team][player_id]["logs_by_game_type"][str(game_type)] = (
                    data.get("gameLog", []) if data else []
                )

    write_json(RAW_DIR / "nhl_player_game_logs_all.json", player_logs_raw)
    player_logs_df = flatten_player_logs(player_logs_raw)
    write_df("player_game_logs", player_logs_df)
    write_df("player_recent_summaries", summarize_player_logs(player_logs_df))

    injury_rows: list[dict[str, Any]] = []
    for team, espn_team_id in ESPN_TEAM_IDS.items():
        injury_index = request_json(
            f"espn_injuries_{team}",
            f"{ESPN_CORE}/teams/{espn_team_id}/injuries?lang=en&region=us",
            f"espn_injuries_{team}.json",
        )
        injury_rows.extend(normalize_injuries(team, injury_index))
    write_df("injuries", pd.DataFrame(injury_rows, columns=INJURY_COLUMNS))

    event_game_ids = playoff_and_h2h_game_ids(club_schedules, current_game_id)
    shift_frames: list[pd.DataFrame] = []
    pbp_frames: list[pd.DataFrame] = []
    for game_id in event_game_ids:
        shift_data = request_json(
            f"nhl_shiftcharts_{game_id}",
            f"{NHL_STATS}/shiftcharts?limit=-1&cayenneExp=gameId={game_id}",
            f"nhl_shiftcharts_{game_id}.json",
            sleep_s=0.05,
        )
        shift_frames.append(flatten_shiftcharts(game_id, shift_data))

        pbp_data = request_json(
            f"nhl_gamecenter_pbp_{game_id}",
            f"{NHL_WEB}/gamecenter/{game_id}/play-by-play",
            f"nhl_gamecenter_{game_id}_play_by_play.json",
            sleep_s=0.05,
        )
        pbp_frames.append(flatten_play_by_play(game_id, pbp_data))

    write_df("shiftcharts_playoff_and_h2h", concat_or_empty(shift_frames))
    write_df("play_by_play_playoff_and_h2h", concat_or_empty(pbp_frames))

    write_df("news_source_pages", fetch_news_sources())

    manifest = {
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": SEASON,
        "game_date": GAME_DATE,
        "teams": TEAMS,
        "game_id": current_game_id,
        "espn_event_id": espn_event_id,
        "raw_dir": str(RAW_DIR.relative_to(ROOT)),
        "processed_dir": str(PROCESSED_DIR.relative_to(ROOT)),
        "latest_dir": str(LATEST_DIR.relative_to(ROOT)),
        "sources": sources,
        "artifacts": artifacts,
    }
    write_json(PROCESSED_DIR / "manifest.json", manifest)
    write_json(LATEST_DIR / "manifest.json", manifest)
    print(f"Finished. Processed artifacts: {len(artifacts)}. Sources attempted: {len(sources)}.")
    print(f"Latest CSVs are in {LATEST_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
