from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_SEASONS = [2024, 2025, 2026]
GAME_DATE = "2026-06-03"
GAME_DATE_COMPACT = GAME_DATE.replace("-", "")
SEASON_TYPES = [2, 3]
MAX_WORKERS = 8

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "nba"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RAW_DIR = DATA_ROOT / "raw" / RUN_ID
PROCESSED_DIR = DATA_ROOT / "processed" / RUN_ID
LATEST_DIR = DATA_ROOT / "latest"

ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_WEB = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba"

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
USE_RAW_CACHE = True
REFRESH_RAW_CACHE = False
_RAW_CACHE_DIRS: list[Path] | None = None


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, LATEST_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


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


def raw_cache_dirs() -> list[Path]:
    global _RAW_CACHE_DIRS
    if _RAW_CACHE_DIRS is not None:
        return _RAW_CACHE_DIRS
    raw_root = DATA_ROOT / "raw"
    if not raw_root.exists():
        _RAW_CACHE_DIRS = []
        return _RAW_CACHE_DIRS
    _RAW_CACHE_DIRS = sorted(
        [path for path in raw_root.iterdir() if path.is_dir() and path.resolve() != RAW_DIR.resolve()],
        key=lambda path: path.name,
        reverse=True,
    )
    return _RAW_CACHE_DIRS


def find_cached_raw(filename: str) -> Path | None:
    for raw_dir in raw_cache_dirs():
        candidate = raw_dir / filename
        if candidate.exists():
            return candidate
    return None


def load_cached_json(path: Path, destination: Path) -> dict[str, Any] | None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        return json.loads(destination.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not reuse cached {path.name}: {exc}", flush=True)
        return None


def request_json(
    name: str,
    url: str,
    filename: str,
    timeout: int = 25,
    use_cache: bool = True,
) -> dict[str, Any] | None:
    path = RAW_DIR / filename
    cached_path = find_cached_raw(filename) if USE_RAW_CACHE and use_cache and not REFRESH_RAW_CACHE else None
    if cached_path is not None:
        data = load_cached_json(cached_path, path)
        if data is not None:
            sources.append(
                {
                    "name": name,
                    "url": url,
                    "path": str(path.relative_to(ROOT)),
                    "ok": True,
                    "cached": True,
                    "cache_path": str(cached_path.relative_to(ROOT)),
                }
            )
            return data

    try:
        response = SESSION.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        write_json(path, data)
        sources.append({"name": name, "url": url, "path": str(path.relative_to(ROOT)), "ok": True, "cached": False})
        return data
    except Exception as exc:
        fallback_path = find_cached_raw(filename) if USE_RAW_CACHE and use_cache else None
        if fallback_path is not None:
            data = load_cached_json(fallback_path, path)
            if data is not None:
                sources.append(
                    {
                        "name": name,
                        "url": url,
                        "path": str(path.relative_to(ROOT)),
                        "ok": True,
                        "cached": True,
                        "cache_path": str(fallback_path.relative_to(ROOT)),
                        "network_error": str(exc),
                    }
                )
                return data
        sources.append({"name": name, "url": url, "path": str(path.relative_to(ROOT)), "ok": False, "error": str(exc)})
        print(f"Could not fetch {name}: {exc}", flush=True)
        return None


def parse_seasons(value: str) -> list[int]:
    seasons: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            seasons.extend(range(start, end + 1))
        else:
            seasons.append(int(item))
    return sorted(set(seasons))


def normalize_teams(teams_data: dict[str, Any]) -> pd.DataFrame:
    teams = (((teams_data.get("sports") or [{}])[0].get("leagues") or [{}])[0].get("teams")) or []
    rows: list[dict[str, Any]] = []
    for item in teams:
        team = item.get("team") or item
        if team.get("isAllStar"):
            continue
        rows.append(
            {
                "team_id": str(team.get("id")),
                "team": team.get("displayName"),
                "abbrev": team.get("abbreviation"),
                "location": team.get("location"),
                "name": team.get("name"),
                "slug": team.get("slug"),
                "is_active": team.get("isActive"),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["team_id"]).sort_values("team").reset_index(drop=True)


def nested_default(value: Any, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get("displayValue") or value.get("name") or value.get("description") or value.get("id") or default
    return value if value is not None else default


def nested_type(value: Any, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get("type") or value.get("id") or default
    return value if value is not None else default


def season_type_from_event(event: dict[str, Any]) -> Any:
    return nested_type(event.get("seasonType")) or nested_type(event.get("season"))


def season_type_name_from_event(event: dict[str, Any]) -> Any:
    season_type = event.get("seasonType") if isinstance(event.get("seasonType"), dict) else {}
    season = event.get("season") if isinstance(event.get("season"), dict) else {}
    return nested_default(season_type) or season.get("displayName") or season.get("slug") or season.get("name")


def round_type_from_competition(comp: dict[str, Any]) -> Any:
    comp_type = comp.get("type") if isinstance(comp.get("type"), dict) else {}
    return (
        comp_type.get("text")
        or comp_type.get("name")
        or comp_type.get("description")
        or comp_type.get("abbreviation")
    )


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "--", "DNP", "None"}:
        return None
    if text.startswith("+"):
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        return None


def minutes_to_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text or text in {"--", "DNP"}:
        return None
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        base = to_float(minutes) or 0.0
        return base + (to_float(seconds) or 0.0) / 60.0
    return to_float(text)


def parse_made_attempted(value: Any) -> tuple[float | None, float | None]:
    text = str(value).strip()
    if "-" not in text:
        return None, None
    left, right = text.split("-", 1)
    return to_float(left), to_float(right)


def sanitize_column(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return name.lower()


def american_to_implied(odds: float | int | None) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def no_vig_prob(home_odds: float | int | None, away_odds: float | int | None) -> float | None:
    home = american_to_implied(home_odds)
    away = american_to_implied(away_odds)
    if home is None or away is None or home + away <= 0:
        return None
    return home / (home + away)


def team_record_from_competitor(comp: dict[str, Any]) -> str | None:
    records = comp.get("records") or []
    for record in records:
        if record.get("type") == "total":
            return record.get("summary")
    return records[0].get("summary") if records else None


def normalize_scoreboard(scoreboard: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in scoreboard.get("events", []) or []:
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors") or []
        home = next((item for item in competitors if item.get("homeAway") == "home"), {})
        away = next((item for item in competitors if item.get("homeAway") == "away"), {})
        rows.append(
            {
                "event_id": event.get("id"),
                "event_name": event.get("name"),
                "short_name": event.get("shortName"),
                "game_datetime": event.get("date"),
                "game_date": str(event.get("date", ""))[:10],
                "season_year": (event.get("season") or {}).get("year") if isinstance(event.get("season"), dict) else None,
                "season_type": season_type_from_event(event),
                "season_type_name": season_type_name_from_event(event),
                "round_type": round_type_from_competition(comp),
                "status_state": nested_default((event.get("status") or {}).get("type")),
                "status_detail": (event.get("status") or {}).get("type", {}).get("detail"),
                "venue": (comp.get("venue") or {}).get("fullName"),
                "neutral_site": comp.get("neutralSite"),
                "home_team_id": (home.get("team") or {}).get("id") or home.get("id"),
                "home_team": (home.get("team") or {}).get("displayName"),
                "home_abbrev": (home.get("team") or {}).get("abbreviation"),
                "home_record": team_record_from_competitor(home),
                "away_team_id": (away.get("team") or {}).get("id") or away.get("id"),
                "away_team": (away.get("team") or {}).get("displayName"),
                "away_abbrev": (away.get("team") or {}).get("abbreviation"),
                "away_record": team_record_from_competitor(away),
            }
        )
    return pd.DataFrame(rows)


def normalize_schedule_event(event: dict[str, Any]) -> dict[str, Any]:
    comp = (event.get("competitions") or [{}])[0]
    competitors = comp.get("competitors") or []
    home = next((item for item in competitors if item.get("homeAway") == "home"), {})
    away = next((item for item in competitors if item.get("homeAway") == "away"), {})
    event_status = event.get("status") if isinstance(event.get("status"), dict) else {}
    comp_status = comp.get("status") if isinstance(comp.get("status"), dict) else {}
    status = (event_status.get("type") if isinstance(event_status.get("type"), dict) else event_status) or (
        comp_status.get("type") if isinstance(comp_status.get("type"), dict) else comp_status
    ) or {}
    if not isinstance(status, dict):
        status = {"state": str(status), "detail": str(status)}

    def score(item: dict[str, Any]) -> float | None:
        value = item.get("score")
        if isinstance(value, dict):
            value = value.get("value") if value.get("value") is not None else value.get("displayValue")
        return to_float(value)

    home_score = score(home)
    away_score = score(away)
    completed = bool(status.get("completed")) or status.get("state") == "post"
    return {
        "event_id": event.get("id"),
        "game_datetime": event.get("date"),
        "game_date": str(event.get("date", ""))[:10],
        "name": event.get("name"),
        "short_name": event.get("shortName"),
        "season_type": season_type_from_event(event),
        "season_type_name": season_type_name_from_event(event),
        "round_type": round_type_from_competition(comp),
        "status_state": status.get("state"),
        "status_detail": status.get("detail") or status.get("description"),
        "completed": completed,
        "neutral_site": comp.get("neutralSite"),
        "venue": (comp.get("venue") or {}).get("fullName"),
        "home_team_id": (home.get("team") or {}).get("id") or home.get("id"),
        "home_team": (home.get("team") or {}).get("displayName"),
        "home_abbrev": (home.get("team") or {}).get("abbreviation"),
        "home_score": home_score,
        "away_team_id": (away.get("team") or {}).get("id") or away.get("id"),
        "away_team": (away.get("team") or {}).get("displayName"),
        "away_abbrev": (away.get("team") or {}).get("abbreviation"),
        "away_score": away_score,
        "home_win": int(home_score > away_score) if home_score is not None and away_score is not None and completed else None,
        "home_margin": home_score - away_score if home_score is not None and away_score is not None and completed else None,
    }


def normalize_roster(team_id: str, roster: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    team = roster.get("team") or {}
    for athlete in roster.get("athletes", []) or []:
        rows.append(
            {
                "team_id": team_id,
                "team": team.get("displayName"),
                "team_abbrev": team.get("abbreviation"),
                "athlete_id": athlete.get("id"),
                "player": athlete.get("displayName") or athlete.get("fullName"),
                "position": nested_default(athlete.get("position")),
                "jersey": athlete.get("jersey"),
                "age": athlete.get("age"),
                "height_inches": athlete.get("height"),
                "weight_lbs": athlete.get("weight"),
                "experience_years": athlete.get("experience", {}).get("years") if isinstance(athlete.get("experience"), dict) else None,
                "status": nested_default(athlete.get("status")),
                "headshot": (athlete.get("headshot") or {}).get("href"),
            }
        )
    return pd.DataFrame(rows)


def normalize_team_statistics(team_id: str, stats_data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stats = (((stats_data.get("results") or {}).get("stats") or {}).get("categories")) or []
    for category in stats:
        for stat in category.get("stats", []) or []:
            rows.append(
                {
                    "team_id": team_id,
                    "category": category.get("name"),
                    "stat_name": stat.get("name"),
                    "display_name": stat.get("displayName"),
                    "abbrev": stat.get("abbreviation"),
                    "value": stat.get("value"),
                    "display_value": stat.get("displayValue"),
                    "per_game_value": stat.get("perGameValue"),
                    "rank": stat.get("rank"),
                    "rank_display_value": stat.get("rankDisplayValue"),
                }
            )
    return pd.DataFrame(rows)


def extract_pickcenter(summary: dict[str, Any], current_game: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in summary.get("pickcenter", []) or []:
        home_odds = (item.get("homeTeamOdds") or {}).get("moneyLine")
        away_odds = (item.get("awayTeamOdds") or {}).get("moneyLine")
        rows.append(
            {
                "event_id": current_game.get("event_id"),
                "provider": (item.get("provider") or {}).get("name"),
                "details": item.get("details"),
                "spread": item.get("spread"),
                "over_under": item.get("overUnder"),
                "over_odds": item.get("overOdds"),
                "under_odds": item.get("underOdds"),
                "home_team_id": current_game.get("home_team_id"),
                "away_team_id": current_game.get("away_team_id"),
                "home_moneyline": home_odds,
                "away_moneyline": away_odds,
                "home_spread_odds": (item.get("homeTeamOdds") or {}).get("spreadOdds"),
                "away_spread_odds": (item.get("awayTeamOdds") or {}).get("spreadOdds"),
                "home_raw_implied_prob": american_to_implied(home_odds),
                "away_raw_implied_prob": american_to_implied(away_odds),
                "home_no_vig_prob": no_vig_prob(home_odds, away_odds),
            }
        )
    return pd.DataFrame(rows)


def extract_predictor(summary: dict[str, Any], current_game: dict[str, Any]) -> pd.DataFrame:
    predictor = summary.get("predictor") or {}
    home = predictor.get("homeTeam") or {}
    away = predictor.get("awayTeam") or {}
    return pd.DataFrame(
        [
            {
                "event_id": current_game.get("event_id"),
                "home_team_id": home.get("id"),
                "away_team_id": away.get("id"),
                "home_projection_pct": to_float(home.get("gameProjection")),
                "away_projection_pct": to_float(away.get("gameProjection")),
                "home_loss_pct": to_float(home.get("teamChanceLoss")),
                "away_loss_pct": to_float(away.get("teamChanceLoss")),
            }
        ]
    )


def extract_injuries(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for team_group in summary.get("injuries", []) or []:
        team = team_group.get("team") or {}
        for item in team_group.get("injuries", []) or []:
            athlete = item.get("athlete") or {}
            rows.append(
                {
                    "team_id": team.get("id"),
                    "team": team.get("displayName"),
                    "team_abbrev": team.get("abbreviation"),
                    "athlete_id": athlete.get("id"),
                    "player": athlete.get("displayName") or athlete.get("fullName"),
                    "status": item.get("status"),
                    "date": item.get("date"),
                    "type": item.get("type"),
                    "details": item.get("details") or item.get("detail"),
                    "description": item.get("description"),
                }
            )
    return pd.DataFrame(rows)


def extract_summary_context(summary: dict[str, Any], current_game: dict[str, Any]) -> dict[str, pd.DataFrame]:
    game_info = summary.get("gameInfo") or {}
    venue = game_info.get("venue") or {}
    seat = (summary.get("ticketsInfo") or {}).get("seatSituation") or {}
    game_info_rows = [
        {
            "event_id": current_game.get("event_id"),
            "venue_id": venue.get("id"),
            "venue_name": venue.get("fullName"),
            "venue_short_name": venue.get("shortName"),
            "city": (venue.get("address") or {}).get("city"),
            "state": (venue.get("address") or {}).get("state"),
            "ticket_summary": seat.get("summary"),
            "ticket_event_link": seat.get("eventLink"),
            "tickets_left_text": next(
                (item.get("ticketName") for item in (summary.get("ticketsInfo") or {}).get("tickets", []) if item.get("type") == "event"),
                None,
            ),
        }
    ]

    broadcast_rows = [
        {
            "event_id": current_game.get("event_id"),
            "name": item.get("name"),
            "type": item.get("type"),
            "market": item.get("market"),
            "media": json.dumps(item.get("media"), default=str),
        }
        for item in summary.get("broadcasts", []) or []
    ]

    news_rows: list[dict[str, Any]] = []
    article_items = list((summary.get("news") or {}).get("articles", []) or [])
    if summary.get("article"):
        article_items.append(summary["article"])
    for item in article_items:
        links = item.get("links") or {}
        web_href = ((links.get("web") or {}).get("href")) if isinstance(links.get("web"), dict) else None
        news_rows.append(
            {
                "event_id": current_game.get("event_id"),
                "headline": item.get("headline"),
                "description": item.get("description"),
                "published": item.get("published"),
                "last_modified": item.get("lastModified"),
                "source": item.get("source"),
                "byline": item.get("byline"),
                "link": web_href,
            }
        )

    standings_rows: list[dict[str, Any]] = []
    for group in (summary.get("standings") or {}).get("groups", []) or []:
        for entry in ((group.get("standings") or {}).get("entries") or []):
            row = {
                "team_id": entry.get("id"),
                "team": entry.get("name") or entry.get("displayName"),
                "abbrev": entry.get("abbreviation"),
                "group_header": group.get("header"),
                "conference_header": group.get("conferenceHeader"),
                "division_header": group.get("divisionHeader"),
            }
            for stat in entry.get("stats", []) or []:
                row[sanitize_column(stat.get("name") or stat.get("abbreviation") or "")] = stat.get("displayValue")
            standings_rows.append(row)

    season_rows: list[dict[str, Any]] = []
    h2h_events: list[dict[str, Any]] = []
    for series in summary.get("seasonseries", []) or []:
        season_rows.append(
            {
                "event_id": current_game.get("event_id"),
                "type": series.get("type"),
                "title": series.get("title"),
                "description": series.get("description"),
                "summary": series.get("summary"),
                "series_score": series.get("seriesScore"),
                "total_competitions": series.get("totalCompetitions"),
                "completed": series.get("completed"),
            }
        )
        for event in series.get("events", []) or []:
            h2h_events.append(normalize_schedule_event(event))

    last5_rows: list[dict[str, Any]] = []
    for team_group in summary.get("lastFiveGames", []) or []:
        team = team_group.get("team") or {}
        for event in team_group.get("events", []) or team_group.get("games", []) or []:
            row = normalize_schedule_event(event)
            row["context_team_id"] = team.get("id")
            row["context_team"] = team.get("displayName")
            last5_rows.append(row)

    leader_rows: list[dict[str, Any]] = []
    for team_group in summary.get("leaders", []) or []:
        team = team_group.get("team") or {}
        for category in team_group.get("leaders", []) or []:
            for leader in category.get("leaders", []) or []:
                athlete = leader.get("athlete") or {}
                leader_rows.append(
                    {
                        "team_id": team.get("id"),
                        "team": team.get("displayName"),
                        "team_abbrev": team.get("abbreviation"),
                        "category": category.get("name") or category.get("displayName"),
                        "athlete_id": athlete.get("id"),
                        "player": athlete.get("displayName") or athlete.get("fullName"),
                        "value": leader.get("value"),
                        "display_value": leader.get("displayValue"),
                    }
                )

    ats_rows: list[dict[str, Any]] = []
    for item in summary.get("againstTheSpread", []) or []:
        team = item.get("team") or {}
        ats_rows.append(
            {
                "team_id": team.get("id"),
                "team": team.get("displayName"),
                "team_abbrev": team.get("abbreviation"),
                "records": json.dumps(item.get("records", []), default=str),
            }
        )

    return {
        "game_info": pd.DataFrame(game_info_rows),
        "broadcasts": pd.DataFrame(broadcast_rows),
        "news_articles": pd.DataFrame(news_rows),
        "standings_snapshot": pd.DataFrame(standings_rows),
        "season_series": pd.DataFrame(season_rows),
        "season_series_events": pd.DataFrame(h2h_events),
        "last_five_from_summary": pd.DataFrame(last5_rows),
        "leaders": pd.DataFrame(leader_rows),
        "against_the_spread": pd.DataFrame(ats_rows),
    }


def parse_team_boxscores(summary: dict[str, Any], schedule_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_box in (summary.get("boxscore") or {}).get("teams", []) or []:
        team = team_box.get("team") or {}
        row: dict[str, Any] = {
            "event_id": schedule_row.get("event_id"),
            "game_date": schedule_row.get("game_date"),
            "game_datetime": schedule_row.get("game_datetime"),
            "season_type": schedule_row.get("season_type"),
            "team_id": str(team.get("id")),
            "team": team.get("displayName"),
            "team_abbrev": team.get("abbreviation"),
        }
        is_home = str(team.get("id")) == str(schedule_row.get("home_team_id"))
        row["home_away"] = "home" if is_home else "away"
        row["opponent_team_id"] = schedule_row.get("away_team_id") if is_home else schedule_row.get("home_team_id")
        row["opponent"] = schedule_row.get("away_team") if is_home else schedule_row.get("home_team")
        row["points_for"] = schedule_row.get("home_score") if is_home else schedule_row.get("away_score")
        row["points_against"] = schedule_row.get("away_score") if is_home else schedule_row.get("home_score")
        row["won"] = int(row["points_for"] > row["points_against"]) if row["points_for"] is not None and row["points_against"] is not None else None
        row["margin"] = row["points_for"] - row["points_against"] if row["points_for"] is not None and row["points_against"] is not None else None

        for stat in team_box.get("statistics", []) or []:
            name = sanitize_column(stat.get("name") or stat.get("label") or "")
            value = stat.get("displayValue")
            made, attempted = parse_made_attempted(value)
            if attempted is not None:
                row[f"{name}_made"] = made
                row[f"{name}_attempted"] = attempted
            else:
                row[name] = to_float(value)
        rows.append(row)
    return rows


def parse_player_boxscores(summary: dict[str, Any], schedule_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_group in (summary.get("boxscore") or {}).get("players", []) or []:
        team = team_group.get("team") or {}
        is_home = str(team.get("id")) == str(schedule_row.get("home_team_id"))
        for stat_group in team_group.get("statistics", []) or []:
            keys = stat_group.get("keys") or []
            for item in stat_group.get("athletes", []) or []:
                athlete = item.get("athlete") or {}
                row: dict[str, Any] = {
                    "event_id": schedule_row.get("event_id"),
                    "game_date": schedule_row.get("game_date"),
                    "game_datetime": schedule_row.get("game_datetime"),
                    "season_type": schedule_row.get("season_type"),
                    "team_id": str(team.get("id")),
                    "team": team.get("displayName"),
                    "team_abbrev": team.get("abbreviation"),
                    "home_away": "home" if is_home else "away",
                    "opponent_team_id": schedule_row.get("away_team_id") if is_home else schedule_row.get("home_team_id"),
                    "opponent": schedule_row.get("away_team") if is_home else schedule_row.get("home_team"),
                    "team_points": schedule_row.get("home_score") if is_home else schedule_row.get("away_score"),
                    "opponent_points": schedule_row.get("away_score") if is_home else schedule_row.get("home_score"),
                    "won": schedule_row.get("home_win") if is_home else (1 - int(schedule_row.get("home_win")) if schedule_row.get("home_win") is not None else None),
                    "athlete_id": athlete.get("id"),
                    "player": athlete.get("displayName"),
                    "position": nested_default(athlete.get("position")),
                    "starter": item.get("starter"),
                    "active": item.get("active"),
                    "did_not_play": item.get("didNotPlay"),
                    "reason": item.get("reason"),
                }
                for key, value in zip(keys, item.get("stats", []) or []):
                    col = sanitize_column(key)
                    if col == "minutes":
                        row[col] = minutes_to_float(value)
                        continue
                    made, attempted = parse_made_attempted(value)
                    if attempted is not None:
                        row[f"{col}_made"] = made
                        row[f"{col}_attempted"] = attempted
                    else:
                        row[col] = to_float(value)
                rows.append(row)
    return rows


def add_derived_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    fga = out.get("field_goals_made_field_goals_attempted_attempted", pd.Series(index=out.index, dtype=float))
    fgm = out.get("field_goals_made_field_goals_attempted_made", pd.Series(index=out.index, dtype=float))
    tpa = out.get("three_point_field_goals_made_three_point_field_goals_attempted_attempted", pd.Series(index=out.index, dtype=float))
    tpm = out.get("three_point_field_goals_made_three_point_field_goals_attempted_made", pd.Series(index=out.index, dtype=float))
    fta = out.get("free_throws_made_free_throws_attempted_attempted", pd.Series(index=out.index, dtype=float))
    tov = out.get("turnovers", out.get("total_turnovers", pd.Series(index=out.index, dtype=float)))
    orb = out.get("offensive_rebounds", pd.Series(index=out.index, dtype=float))
    reb = out.get("total_rebounds", pd.Series(index=out.index, dtype=float))
    points = out.get("points_for", pd.Series(index=out.index, dtype=float))

    out["estimated_possessions"] = fga + 0.44 * fta - orb + tov
    out["off_rating_est"] = 100 * points / out["estimated_possessions"].replace(0, math.nan)
    out["efg_pct_est"] = (fgm + 0.5 * tpm) / fga.replace(0, math.nan)
    out["three_pa_rate"] = tpa / fga.replace(0, math.nan)
    out["ft_rate"] = fta / fga.replace(0, math.nan)
    out["turnover_rate"] = tov / out["estimated_possessions"].replace(0, math.nan)
    out["oreb_rate_proxy"] = orb / reb.replace(0, math.nan)
    return out


def add_derived_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    minutes = out.get("minutes", pd.Series(index=out.index, dtype=float)).replace(0, math.nan)
    for stat in ["points", "rebounds", "assists", "turnovers", "steals", "blocks"]:
        if stat in out.columns:
            out[f"{stat}_per36"] = out[stat] * 36 / minutes
    fga = out.get("field_goals_made_field_goals_attempted_attempted", pd.Series(index=out.index, dtype=float)).fillna(0)
    fta = out.get("free_throws_made_free_throws_attempted_attempted", pd.Series(index=out.index, dtype=float)).fillna(0)
    tov = out.get("turnovers", pd.Series(index=out.index, dtype=float)).fillna(0)
    out["usage_proxy"] = fga + 0.44 * fta + tov
    return out


def aggregate_team_split(rows: pd.DataFrame, team_id: str, target_date: str, split: str, opponent_id: str | None = None) -> dict[str, Any]:
    subset = rows[(rows["team_id"].astype(str) == str(team_id)) & (rows["game_date"] < target_date)].copy()
    if opponent_id is not None:
        subset = subset[subset["opponent_team_id"].astype(str) == str(opponent_id)]
    if split == "regular":
        subset = subset[subset["season_type"] == 2]
    elif split == "playoffs":
        subset = subset[subset["season_type"] == 3]
    elif split.startswith("last"):
        n = int(split.replace("last", ""))
        subset = subset.sort_values("game_date").tail(n)

    numeric_cols = [
        "points_for",
        "points_against",
        "margin",
        "estimated_possessions",
        "off_rating_est",
        "efg_pct_est",
        "three_pa_rate",
        "ft_rate",
        "turnover_rate",
        "offensive_rebounds",
        "defensive_rebounds",
        "total_rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "fouls",
    ]
    out: dict[str, Any] = {
        "games": int(len(subset)),
        "win_pct": float(subset["won"].mean()) if not subset.empty and "won" in subset else None,
    }
    if not subset.empty:
        last_game = pd.to_datetime(subset["game_date"]).max()
        out["rest_days"] = max(float((pd.to_datetime(target_date) - last_game).days), 0.0)
    else:
        out["rest_days"] = None

    for col in numeric_cols:
        if col in subset.columns:
            out[f"avg_{col}"] = float(pd.to_numeric(subset[col], errors="coerce").mean()) if not subset.empty else None
    return out


def player_recent_summaries(player_logs: pd.DataFrame, roster: pd.DataFrame, injuries: pd.DataFrame, target_date: str) -> pd.DataFrame:
    if roster.empty:
        return pd.DataFrame()
    logs = player_logs[player_logs["game_date"] < target_date].copy() if not player_logs.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    injury_status = {}
    if not injuries.empty:
        injury_status = {
            str(row["athlete_id"]): row.get("status")
            for _, row in injuries.dropna(subset=["athlete_id"]).iterrows()
        }

    for _, player in roster.iterrows():
        athlete_id = str(player.get("athlete_id"))
        p_logs = logs[logs["athlete_id"].astype(str) == athlete_id].sort_values("game_date") if not logs.empty else pd.DataFrame()
        row: dict[str, Any] = player.to_dict()
        row["injury_status"] = injury_status.get(athlete_id)
        row["available_flag"] = 0 if str(row["injury_status"]).lower() == "out" else 1
        for label, n in [("last3", 3), ("last5", 5), ("last10", 10), ("season", 999)]:
            subset = p_logs.tail(n) if n != 999 else p_logs
            row[f"{label}_games_logged"] = int(len(subset))
            row[f"{label}_starts"] = int(pd.to_numeric(subset.get("starter", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not subset.empty else 0
            for col in [
                "minutes",
                "points",
                "rebounds",
                "assists",
                "turnovers",
                "steals",
                "blocks",
                "plus_minus",
                "usage_proxy",
                "points_per36",
                "rebounds_per36",
                "assists_per36",
            ]:
                if col in subset.columns:
                    row[f"{label}_avg_{col}"] = float(pd.to_numeric(subset[col], errors="coerce").mean()) if not subset.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def team_player_aggregates(player_summary: pd.DataFrame, team_id: str) -> dict[str, Any]:
    subset = player_summary[player_summary["team_id"].astype(str) == str(team_id)].copy()
    out: dict[str, Any] = {
        "roster_count": int(len(subset)),
        "injury_out_count": int((subset["injury_status"].astype(str).str.lower() == "out").sum()) if "injury_status" in subset else 0,
        "available_count": int(subset.get("available_flag", pd.Series(dtype=float)).fillna(1).sum()) if not subset.empty else 0,
    }
    if subset.empty:
        return out

    subset["sort_minutes"] = pd.to_numeric(subset.get("last10_avg_minutes"), errors="coerce").fillna(0)
    top = subset.sort_values("sort_minutes", ascending=False).head(8)
    for label, frame in [("top5", top.head(5)), ("top8", top)]:
        for col in ["last5_avg_minutes", "last5_avg_points", "last5_avg_rebounds", "last5_avg_assists", "last5_avg_usage_proxy", "last10_avg_points"]:
            if col in frame.columns:
                out[f"{label}_sum_{col}"] = float(pd.to_numeric(frame[col], errors="coerce").sum())
                out[f"{label}_avg_{col}"] = float(pd.to_numeric(frame[col], errors="coerce").mean())
    subset["sort_points"] = pd.to_numeric(subset.get("last10_avg_points"), errors="coerce").fillna(0)
    scorers = subset.sort_values("sort_points", ascending=False).head(3)
    for rank, (_, row) in enumerate(scorers.iterrows(), start=1):
        out[f"scorer{rank}_player"] = row.get("player")
        out[f"scorer{rank}_last10_ppg"] = row.get("last10_avg_points")
        out[f"scorer{rank}_last10_mpg"] = row.get("last10_avg_minutes")
    return out


def prefixed(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in data.items()}


def build_feature_vector(
    current_game: dict[str, Any],
    team_boxscores: pd.DataFrame,
    player_summary: pd.DataFrame,
    odds: pd.DataFrame,
    predictor: pd.DataFrame,
) -> pd.DataFrame:
    home_id = str(current_game["home_team_id"])
    away_id = str(current_game["away_team_id"])
    target_date = current_game["game_date"]
    row: dict[str, Any] = dict(current_game)

    odds_row = odds.iloc[0].to_dict() if not odds.empty else {}
    predictor_row = predictor.iloc[0].to_dict() if not predictor.empty else {}
    row.update({f"market_{key}": value for key, value in odds_row.items() if key not in {"event_id", "home_team_id", "away_team_id"}})
    row.update({f"espn_predictor_{key}": value for key, value in predictor_row.items() if key not in {"event_id", "home_team_id", "away_team_id"}})

    split_specs = ["regular", "playoffs", "last5", "last10"]
    for split in split_specs:
        home = aggregate_team_split(team_boxscores, home_id, target_date, split)
        away = aggregate_team_split(team_boxscores, away_id, target_date, split)
        row.update(prefixed(f"home_{split}", home))
        row.update(prefixed(f"away_{split}", away))
        for key in home:
            if isinstance(home.get(key), (int, float)) and isinstance(away.get(key), (int, float)):
                row[f"edge_{split}_{key}"] = home[key] - away[key]

    home_h2h = aggregate_team_split(team_boxscores, home_id, target_date, "all", opponent_id=away_id)
    away_h2h = aggregate_team_split(team_boxscores, away_id, target_date, "all", opponent_id=home_id)
    row.update(prefixed("home_h2h", home_h2h))
    row.update(prefixed("away_h2h", away_h2h))
    for key in home_h2h:
        if isinstance(home_h2h.get(key), (int, float)) and isinstance(away_h2h.get(key), (int, float)):
            row[f"edge_h2h_{key}"] = home_h2h[key] - away_h2h[key]

    home_players = team_player_aggregates(player_summary, home_id)
    away_players = team_player_aggregates(player_summary, away_id)
    row.update(prefixed("home_players", home_players))
    row.update(prefixed("away_players", away_players))
    for key in home_players:
        if isinstance(home_players.get(key), (int, float)) and isinstance(away_players.get(key), (int, float)):
            row[f"edge_players_{key}"] = home_players[key] - away_players[key]

    return pd.DataFrame([row])


def main() -> None:
    global GAME_DATE, GAME_DATE_COMPACT, USE_RAW_CACHE, REFRESH_RAW_CACHE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        default=",".join(str(season) for season in DEFAULT_SEASONS),
        help="Comma-separated ESPN NBA season years, e.g. 2024,2025,2026.",
    )
    parser.add_argument("--game-date", default=GAME_DATE, help="ESPN scoreboard date in YYYY-MM-DD format.")
    parser.add_argument(
        "--target-teams-only",
        action="store_true",
        help="Only collect schedules/rosters/stats for the two target teams.",
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--no-raw-cache",
        dest="raw_cache",
        action="store_false",
        help="Do not reuse JSON files from previous data/nba/raw runs.",
    )
    parser.add_argument(
        "--refresh-raw-cache",
        action="store_true",
        help="Force network refresh even when matching raw JSON files exist from prior runs.",
    )
    args = parser.parse_args()
    GAME_DATE = args.game_date
    GAME_DATE_COMPACT = GAME_DATE.replace("-", "")
    USE_RAW_CACHE = bool(args.raw_cache)
    REFRESH_RAW_CACHE = bool(args.refresh_raw_cache)
    seasons = parse_seasons(args.seasons)

    ensure_dirs()
    print(f"Collecting NBA Finals data into {RAW_DIR.relative_to(ROOT)} and {PROCESSED_DIR.relative_to(ROOT)}", flush=True)
    print(f"Schedule seasons: {', '.join(str(season) for season in seasons)}", flush=True)

    scoreboard = request_json(
        "espn_scoreboard",
        f"{ESPN_SITE}/scoreboard?dates={GAME_DATE_COMPACT}",
        f"espn_scoreboard_{GAME_DATE_COMPACT}.json",
        use_cache=False,
    )
    if not scoreboard:
        raise RuntimeError("Could not fetch ESPN scoreboard.")

    current_games = normalize_scoreboard(scoreboard)
    if current_games.empty:
        raise RuntimeError(f"No NBA games found for {GAME_DATE}.")
    write_df("current_scoreboard_games", current_games)

    current_game = current_games.iloc[0].to_dict()
    event_id = str(current_game["event_id"])
    target_game_date = str(current_game.get("game_date") or GAME_DATE)
    print(f"Target game: {current_game['away_team']} at {current_game['home_team']} ({event_id})", flush=True)

    current_summary = request_json(
        "espn_current_summary",
        f"{ESPN_SITE}/summary?event={event_id}",
        f"espn_summary_{event_id}.json",
        use_cache=False,
    ) or {}

    current_game_df = pd.DataFrame([current_game])
    write_df("current_game", current_game_df)
    odds = extract_pickcenter(current_summary, current_game)
    predictor = extract_predictor(current_summary, current_game)
    injuries = extract_injuries(current_summary)
    write_df("odds_market_snapshot", odds)
    write_df("espn_predictor", predictor)
    write_df("injuries", injuries)

    for name, df in extract_summary_context(current_summary, current_game).items():
        write_df(name, df)

    teams_data = request_json(
        "espn_teams",
        f"{ESPN_SITE}/teams",
        "espn_teams.json",
    ) or {}
    teams_df = normalize_teams(teams_data)
    write_df("teams", teams_df)

    target_team_ids = [str(current_game["home_team_id"]), str(current_game["away_team_id"])]
    if args.target_teams_only or teams_df.empty:
        team_ids = target_team_ids
    else:
        team_ids = teams_df["team_id"].astype(str).tolist()
    print(f"Collecting schedules for {len(team_ids)} teams", flush=True)

    roster_frames: list[pd.DataFrame] = []
    team_stat_frames: list[pd.DataFrame] = []
    schedule_rows: list[dict[str, Any]] = []

    for team_id in team_ids:
        roster = request_json(
            f"espn_roster_{team_id}",
            f"{ESPN_SITE}/teams/{team_id}/roster",
            f"espn_roster_{team_id}.json",
        ) or {}
        roster_frames.append(normalize_roster(team_id, roster))

        team_stats = request_json(
            f"espn_team_stats_{team_id}",
            f"{ESPN_SITE}/teams/{team_id}/statistics",
            f"espn_team_stats_{team_id}.json",
        ) or {}
        team_stat_frames.append(normalize_team_statistics(team_id, team_stats))

        for season in seasons:
            for season_type in SEASON_TYPES:
                schedule = request_json(
                    f"espn_schedule_{team_id}_{season}_{season_type}",
                    f"{ESPN_WEB}/teams/{team_id}/schedule?season={season}&seasontype={season_type}",
                    f"espn_schedule_{team_id}_{season}_{season_type}.json",
                ) or {}
                for event in schedule.get("events", []) or []:
                    row = normalize_schedule_event(event)
                    row["schedule_context_team_id"] = team_id
                    row["schedule_context_season"] = season
                    schedule_rows.append(row)

    roster_df = pd.concat(roster_frames, ignore_index=True) if roster_frames else pd.DataFrame()
    team_stats_df = pd.concat(team_stat_frames, ignore_index=True) if team_stat_frames else pd.DataFrame()
    schedule_df = pd.DataFrame(schedule_rows).drop_duplicates(subset=["event_id"]).sort_values(["game_date", "event_id"]).reset_index(drop=True)
    write_df("rosters", roster_df)
    write_df("espn_team_statistics", team_stats_df)
    write_df("team_schedule_games", schedule_df)

    completed = schedule_df[(schedule_df["completed"] == True) & (schedule_df["game_date"] < target_game_date)].copy()
    event_map = {str(row["event_id"]): row.to_dict() for _, row in completed.iterrows()}
    print(f"Fetching {len(event_map)} completed game summaries for team/player logs...", flush=True)

    def fetch_summary(event: str) -> tuple[str, dict[str, Any] | None]:
        data = request_json(
            f"espn_completed_summary_{event}",
            f"{ESPN_SITE}/summary?event={event}",
            f"espn_completed_summary_{event}.json",
        )
        time.sleep(0.02)
        return event, data

    summaries: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(fetch_summary, event) for event in event_map]
        for idx, future in enumerate(as_completed(futures), start=1):
            event, data = future.result()
            if data:
                summaries[event] = data
            if idx % 100 == 0 or idx == len(futures):
                print(f"  summaries fetched: {idx}/{len(futures)}", flush=True)

    team_box_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    for event, summary in summaries.items():
        schedule_row = event_map[event]
        team_box_rows.extend(parse_team_boxscores(summary, schedule_row))
        player_rows.extend(parse_player_boxscores(summary, schedule_row))

    team_boxscores = add_derived_team_stats(pd.DataFrame(team_box_rows))
    player_logs = add_derived_player_stats(pd.DataFrame(player_rows))
    write_df("team_boxscores", team_boxscores)
    write_df("player_game_logs", player_logs)

    current_roster_df = roster_df[roster_df["team_id"].astype(str).isin(target_team_ids)].copy()
    player_summary = player_recent_summaries(player_logs, current_roster_df, injuries, target_game_date)
    write_df("player_recent_summaries", player_summary)

    h2h_games = schedule_df[
        (
            (schedule_df["home_team_id"].astype(str).isin(target_team_ids))
            & (schedule_df["away_team_id"].astype(str).isin(target_team_ids))
        )
    ].sort_values(["game_date", "event_id"])
    write_df("head_to_head_games", h2h_games)

    feature_vector = build_feature_vector(current_game, team_boxscores, player_summary, odds, predictor)
    write_df("current_game_feature_vector", feature_vector)

    manifest = {
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game_date": GAME_DATE,
        "target_game_date_espn_utc": target_game_date,
        "schedule_seasons": seasons,
        "raw_cache_enabled": bool(USE_RAW_CACHE),
        "raw_cache_refresh_forced": bool(REFRESH_RAW_CACHE),
        "raw_cache_hits": int(sum(1 for source in sources if source.get("cached"))),
        "target_teams_only": bool(args.target_teams_only),
        "teams_collected": int(len(team_ids)),
        "completed_events_requested": int(len(event_map)),
        "completed_summaries_fetched": int(len(summaries)),
        "target_event_id": event_id,
        "target_matchup": f"{current_game['away_team']} at {current_game['home_team']}",
        "data_root": str(DATA_ROOT.relative_to(ROOT)),
        "sources": sources,
        "artifacts": artifacts,
        "notes": [
            "Historical schedule, team boxscore, and player logs are from ESPN public APIs.",
            "Player recent summaries are computed from completed game boxscores before the target date.",
            "The current feature vector includes team split edges, H2H edges, market/predictor features, and roster/player aggregate edges.",
            "stats.nba.com was probed but timed out in this environment, so this collector avoids a hard dependency on it.",
        ],
    }
    write_json(PROCESSED_DIR / "manifest.json", manifest)
    write_json(LATEST_DIR / "manifest.json", manifest)
    print(f"Finished. Artifacts: {len(artifacts)}. Latest CSVs are in {LATEST_DIR.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
