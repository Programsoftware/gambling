import re
import io
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from zoneinfo import ZoneInfo
from difflib import get_close_matches
import unicodedata

TEAM = "MTL"
SEASON = 20252026
TZ = ZoneInfo("America/Toronto")
BASE = "https://api-web.nhle.com/v1"

REGULAR_SEASON_GAME_TYPE = 2
OPPONENT_OVERRIDE = "CAR"

# For line xG/GA, 5on5 is usually cleaner.
MONEYPUCK_SITUATION = "5on5"

TEAM_SLUGS = {
    "MTL": "montreal-canadiens",
    "CAR": "carolina-hurricanes",
    "TOR": "toronto-maple-leafs",
    "BOS": "boston-bruins",
    "FLA": "florida-panthers",
    "TBL": "tampa-bay-lightning",
    "BUF": "buffalo-sabres",
    "OTT": "ottawa-senators",
    "DET": "detroit-red-wings",
    "NYR": "new-york-rangers",
    "NYI": "new-york-islanders",
    "NJD": "new-jersey-devils",
    "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins",
    "WSH": "washington-capitals",
    "CBJ": "columbus-blue-jackets",
}

MANUAL_LINES = {
    "MTL": {
        "F1": ["Cole Caufield", "Nick Suzuki", "Juraj Slafkovsky"],
        "F2": ["Alex Newhook", "Kirby Dach", "Patrik Laine"],
        "F3": ["Josh Anderson", "Jake Evans", "Brendan Gallagher"],
        "F4": ["Michael Pezzetta", "Christian Dvorak", "Joel Armia"],
        "D1": ["Mike Matheson", "Noah Dobson"],
        "D2": ["Kaiden Guhle", "Alexandre Carrier"],
        "D3": ["Lane Hutson", "David Savard"],
    },
    "CAR": {
        "F1": ["Andrei Svechnikov", "Sebastian Aho", "Seth Jarvis"],
        "F2": ["Nikolaj Ehlers", "Logan Stankoven", "Jackson Blake"],
        "F3": ["Taylor Hall", "Jordan Staal", "Jordan Martinook"],
        "F4": ["William Carrier", "Jesperi Kotkaniemi", "Eric Robinson"],
        "D1": ["Jaccob Slavin", "K'Andre Miller"],
        "D2": ["Shayne Gostisbehere", "Sean Walker"],
        "D3": ["Alexander Nikishin", "Dmitry Orlov"],
    },
}

PLAYER_COLUMNS = [
    "team", "line", "player", "player_id", "pos",

    "GP",
    "G_per_gp", "A_per_gp", "PTS_per_gp", "SOG_per_gp",
    "HIT_per_gp", "BLK_per_gp", "TOI_per_gp",
    "ixG_per_gp", "xGF_per_gp", "xGA_per_gp",
    "GF_onice_per_gp", "GA_onice_per_gp",
    "last10_G_per_gp", "last10_PTS_per_gp", "last10_SOG_per_gp",

    "current_game_type", "current_GP",
    "current_G_per_gp", "current_A_per_gp", "current_PTS_per_gp",
    "current_SOG_per_gp", "current_HIT_per_gp", "current_BLK_per_gp",
    "current_TOI_per_gp",
    "current_ixG_per_gp", "current_xGF_per_gp", "current_xGA_per_gp",
    "current_GF_onice_per_gp", "current_GA_onice_per_gp",
    "current_last10_G_per_gp", "current_last10_PTS_per_gp",
    "current_last10_SOG_per_gp",
]

LINE_COLUMNS = [
    "team", "line", "players", "player_count", "expected_count",

    "GP_avg",
    "G_per_gp", "A_per_gp", "PTS_per_gp", "SOG_per_gp",
    "HIT_per_gp", "BLK_per_gp", "TOI_per_gp",
    "ixG_per_gp", "xGF_per_gp", "xGA_per_gp",
    "GF_onice_per_gp", "GA_onice_per_gp",
    "last10_G_per_gp", "last10_PTS_per_gp", "last10_SOG_per_gp",

    "current_game_type", "current_GP_avg",
    "current_G_per_gp", "current_A_per_gp", "current_PTS_per_gp",
    "current_SOG_per_gp", "current_HIT_per_gp", "current_BLK_per_gp",
    "current_TOI_per_gp",
    "current_ixG_per_gp", "current_xGF_per_gp", "current_xGA_per_gp",
    "current_GF_onice_per_gp", "current_GA_onice_per_gp",
    "current_last10_G_per_gp", "current_last10_PTS_per_gp",
    "current_last10_SOG_per_gp",
]


def get_json(url):
    r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


def clean_name(name):
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.replace("’", "'").replace("`", "'").replace("´", "'")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def today_game(team=TEAM):
    today = datetime.now(TZ).date().isoformat()
    data = get_json(f"{BASE}/club-schedule-season/{team}/{SEASON}")

    for g in data.get("games", []):
        if g.get("gameDate") == today:
            away = g["awayTeam"]["abbrev"]
            home = g["homeTeam"]["abbrev"]
            return {
                "game_id": g["id"],
                "date": g["gameDate"],
                "away": away,
                "home": home,
                "opponent": home if away == team else away,
                "game_type": g.get("gameType", 2),
            }

    if OPPONENT_OVERRIDE:
        return {
            "game_id": None,
            "date": today,
            "away": None,
            "home": None,
            "opponent": OPPONENT_OVERRIDE,
            "game_type": REGULAR_SEASON_GAME_TYPE,
        }

    raise RuntimeError(f"No {team} game found today. Set OPPONENT_OVERRIDE.")


def roster_map(team):
    data = get_json(f"{BASE}/roster/{team}/current")
    players = []

    for group in ["forwards", "defensemen", "goalies"]:
        for p in data.get(group, []):
            name = f"{p['firstName']['default']} {p['lastName']['default']}"
            players.append({
                "id": p["id"],
                "name": name,
                "clean": clean_name(name),
                "pos": p.get("positionCode"),
            })

    return players


def match_player(name, roster):
    name_clean = clean_name(name)

    for p in roster:
        if p["clean"].lower() == name_clean.lower():
            return p

    roster_clean_names = [p["clean"] for p in roster]
    hit = get_close_matches(name_clean, roster_clean_names, n=1, cutoff=0.72)

    if not hit:
        return None

    return next(p for p in roster if p["clean"] == hit[0])


def scrape_daily_faceoff_lines(team, team_slug, roster):
    url = f"https://www.dailyfaceoff.com/teams/{team_slug}/line-combinations"

    try:
        html = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"}).text
    except Exception:
        return None

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")

    names = re.findall(r"\b[A-Z][a-zA-ZÀ-ÿ.'’-]+ [A-Z][a-zA-ZÀ-ÿ.'’-]+\b", text)

    bad_words = {
        "Daily Faceoff", "Line Combinations", "Power Play", "Penalty Kill",
        "Starting Goalies", "Projected Lineups", "Player News", "Latest News",
        "Betting Odds", "NHL News",
    }

    matched = []
    seen_ids = set()

    for name in names:
        name = clean_name(name)

        if name in bad_words:
            continue

        p = match_player(name, roster)

        if p is None:
            continue

        if p["id"] in seen_ids:
            continue

        seen_ids.add(p["id"])
        matched.append(p["name"])

    forwards = []
    defense = []

    for name in matched:
        p = match_player(name, roster)
        if p is None:
            continue

        if p["pos"] == "D":
            defense.append(name)
        elif p["pos"] != "G":
            forwards.append(name)

    if len(forwards) < 9:
        return None

    return {
        "F1": forwards[0:3],
        "F2": forwards[3:6],
        "F3": forwards[6:9],
        "F4": forwards[9:12],
        "D1": defense[0:2],
        "D2": defense[2:4],
        "D3": defense[4:6],
    }


def get_lines(team, roster):
    slug = TEAM_SLUGS.get(team)

    if slug:
        lines = scrape_daily_faceoff_lines(team, slug, roster)
        if lines:
            return lines, "dailyfaceoff"

    if team in MANUAL_LINES:
        return MANUAL_LINES[team], "manual_fallback"

    raise RuntimeError(f"No lineup source for {team}. Add {team} to MANUAL_LINES.")


def player_logs(player_id, game_type):
    url = f"{BASE}/player/{player_id}/game-log/{SEASON}/{game_type}"

    try:
        data = get_json(url)
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame(data.get("gameLog", []))


def mean_col(df, col):
    if df.empty or col not in df.columns:
        return 0.0

    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())


def toi_to_minutes(value):
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float)):
        return float(value) / 60.0

    value = str(value).strip()

    if ":" in value:
        parts = value.split(":")
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60.0

    return 0.0


def mean_toi(df):
    if df.empty:
        return 0.0

    toi_col = None

    for col in ["toi", "timeOnIce"]:
        if col in df.columns:
            toi_col = col
            break

    if toi_col is None:
        return 0.0

    return float(df[toi_col].apply(toi_to_minutes).mean())


def season_start_year():
    return int(str(SEASON)[:4])


def moneypuck_season_type(game_type):
    return "playoffs" if game_type == 3 else "regular"


def load_moneypuck(game_type):
    year = season_start_year()
    season_type = moneypuck_season_type(game_type)
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/{season_type}/skaters.csv"

    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    if "situation" in df.columns:
        filtered = df[df["situation"].astype(str).str.lower() == MONEYPUCK_SITUATION.lower()]
        if not filtered.empty:
            df = filtered.copy()

    return df


def find_col(df, candidates):
    if df.empty:
        return None

    col_map = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]

    return None


def num_value(row, col):
    if col is None:
        return 0.0

    try:
        return float(pd.to_numeric(row.get(col, 0), errors="coerce"))
    except Exception:
        return 0.0


def moneypuck_player_stats(player, team, mp_df):
    empty = {
        "ixG_per_gp": 0.0,
        "xGF_per_gp": 0.0,
        "xGA_per_gp": 0.0,
        "GF_onice_per_gp": 0.0,
        "GA_onice_per_gp": 0.0,
    }

    if mp_df.empty:
        return empty

    player_id_col = find_col(mp_df, ["playerId", "player_id"])
    name_col = find_col(mp_df, ["name", "playerName", "player"])
    team_col = find_col(mp_df, ["team"])

    row_df = pd.DataFrame()

    if player_id_col:
        row_df = mp_df[pd.to_numeric(mp_df[player_id_col], errors="coerce") == int(player["id"])]

    if row_df.empty and name_col:
        clean_target = clean_name(player["name"]).lower()
        row_df = mp_df[mp_df[name_col].astype(str).map(lambda x: clean_name(x).lower()) == clean_target]

        if not row_df.empty and team_col:
            team_filtered = row_df[row_df[team_col].astype(str).str.upper() == team.upper()]
            if not team_filtered.empty:
                row_df = team_filtered

    if row_df.empty:
        return empty

    row = row_df.iloc[0]

    gp_col = find_col(mp_df, ["games_played", "gamesPlayed", "GP"])
    gp = num_value(row, gp_col)

    if gp <= 0:
        gp = 1.0

    ixg_col = find_col(mp_df, ["I_F_xGoals", "ixG", "individual_xGoals"])
    xgf_col = find_col(mp_df, ["OnIce_F_xGoals", "onIce_F_xGoals"])
    xga_col = find_col(mp_df, ["OnIce_A_xGoals", "onIce_A_xGoals"])
    gf_col = find_col(mp_df, ["OnIce_F_goals", "onIce_F_goals"])
    ga_col = find_col(mp_df, ["OnIce_A_goals", "onIce_A_goals"])

    return {
        "ixG_per_gp": num_value(row, ixg_col) / gp,
        "xGF_per_gp": num_value(row, xgf_col) / gp,
        "xGA_per_gp": num_value(row, xga_col) / gp,
        "GF_onice_per_gp": num_value(row, gf_col) / gp,
        "GA_onice_per_gp": num_value(row, ga_col) / gp,
    }


def empty_stats():
    return {
        "GP": 0,
        "G_per_gp": 0.0,
        "A_per_gp": 0.0,
        "PTS_per_gp": 0.0,
        "SOG_per_gp": 0.0,
        "HIT_per_gp": 0.0,
        "BLK_per_gp": 0.0,
        "TOI_per_gp": 0.0,
        "last10_G_per_gp": 0.0,
        "last10_PTS_per_gp": 0.0,
        "last10_SOG_per_gp": 0.0,
    }


def summarize_player_game_type(player_id, game_type):
    df = player_logs(player_id, game_type)

    if df.empty:
        return empty_stats()

    if "gameDate" in df.columns:
        df = df.sort_values("gameDate", ascending=False)

    last10 = df.head(10)

    return {
        "GP": len(df),
        "G_per_gp": mean_col(df, "goals"),
        "A_per_gp": mean_col(df, "assists"),
        "PTS_per_gp": mean_col(df, "points"),
        "SOG_per_gp": mean_col(df, "shots"),
        "HIT_per_gp": mean_col(df, "hits"),
        "BLK_per_gp": mean_col(df, "blockedShots"),
        "TOI_per_gp": mean_toi(df),
        "last10_G_per_gp": mean_col(last10, "goals"),
        "last10_PTS_per_gp": mean_col(last10, "points"),
        "last10_SOG_per_gp": mean_col(last10, "shots"),
    }


def summarize_player(player, team, current_game_type, regular_mp, current_mp):
    regular = summarize_player_game_type(player["id"], REGULAR_SEASON_GAME_TYPE)
    regular_mp_stats = moneypuck_player_stats(player, team, regular_mp)

    if current_game_type == REGULAR_SEASON_GAME_TYPE:
        current = regular.copy()
        current_mp_stats = regular_mp_stats.copy()
    else:
        current = summarize_player_game_type(player["id"], current_game_type)
        current_mp_stats = moneypuck_player_stats(player, team, current_mp)

    row = {
        "player": player["name"],
        "player_id": player["id"],
        "pos": player["pos"],
    }

    row.update(regular)
    row.update(regular_mp_stats)

    row["current_game_type"] = current_game_type
    row["current_GP"] = current["GP"]

    for col in [
        "G_per_gp", "A_per_gp", "PTS_per_gp", "SOG_per_gp",
        "HIT_per_gp", "BLK_per_gp", "TOI_per_gp",
        "last10_G_per_gp", "last10_PTS_per_gp", "last10_SOG_per_gp",
    ]:
        row[f"current_{col}"] = current[col]

    for col in [
        "ixG_per_gp", "xGF_per_gp", "xGA_per_gp",
        "GF_onice_per_gp", "GA_onice_per_gp",
    ]:
        row[f"current_{col}"] = current_mp_stats[col]

    return row


def build_team_line_stats(team, current_game_type, regular_mp, current_mp):
    roster = roster_map(team)
    lines, source = get_lines(team, roster)

    rows = []
    missing = []

    for line_name, player_names in lines.items():
        for player_name in player_names:
            match = match_player(player_name, roster)

            if match is None:
                missing.append({
                    "team": team,
                    "line": line_name,
                    "player": player_name,
                })
                continue

            row = summarize_player(match, team, current_game_type, regular_mp, current_mp)
            row["team"] = team
            row["line"] = line_name
            rows.append(row)

    player_df = pd.DataFrame(rows)

    if player_df.empty:
        player_df = pd.DataFrame(columns=PLAYER_COLUMNS)
        line_df = pd.DataFrame(columns=LINE_COLUMNS)
        return player_df, line_df, missing, source, lines

    for col in PLAYER_COLUMNS:
        if col not in player_df.columns:
            player_df[col] = 0

    player_df = player_df[PLAYER_COLUMNS]

    expected_counts = {
        (team, line): len(players)
        for line, players in lines.items()
    }

    line_df = (
        player_df
        .groupby(["team", "line"], as_index=False)
        .agg(
            players=("player", lambda x: " / ".join(x)),
            player_count=("player", "count"),

            GP_avg=("GP", "mean"),
            G_per_gp=("G_per_gp", "sum"),
            A_per_gp=("A_per_gp", "sum"),
            PTS_per_gp=("PTS_per_gp", "sum"),
            SOG_per_gp=("SOG_per_gp", "sum"),
            HIT_per_gp=("HIT_per_gp", "sum"),
            BLK_per_gp=("BLK_per_gp", "sum"),
            TOI_per_gp=("TOI_per_gp", "sum"),

            ixG_per_gp=("ixG_per_gp", "sum"),
            xGF_per_gp=("xGF_per_gp", "mean"),
            xGA_per_gp=("xGA_per_gp", "mean"),
            GF_onice_per_gp=("GF_onice_per_gp", "mean"),
            GA_onice_per_gp=("GA_onice_per_gp", "mean"),

            last10_G_per_gp=("last10_G_per_gp", "sum"),
            last10_PTS_per_gp=("last10_PTS_per_gp", "sum"),
            last10_SOG_per_gp=("last10_SOG_per_gp", "sum"),

            current_game_type=("current_game_type", "first"),
            current_GP_avg=("current_GP", "mean"),
            current_G_per_gp=("current_G_per_gp", "sum"),
            current_A_per_gp=("current_A_per_gp", "sum"),
            current_PTS_per_gp=("current_PTS_per_gp", "sum"),
            current_SOG_per_gp=("current_SOG_per_gp", "sum"),
            current_HIT_per_gp=("current_HIT_per_gp", "sum"),
            current_BLK_per_gp=("current_BLK_per_gp", "sum"),
            current_TOI_per_gp=("current_TOI_per_gp", "sum"),

            current_ixG_per_gp=("current_ixG_per_gp", "sum"),
            current_xGF_per_gp=("current_xGF_per_gp", "mean"),
            current_xGA_per_gp=("current_xGA_per_gp", "mean"),
            current_GF_onice_per_gp=("current_GF_onice_per_gp", "mean"),
            current_GA_onice_per_gp=("current_GA_onice_per_gp", "mean"),

            current_last10_G_per_gp=("current_last10_G_per_gp", "sum"),
            current_last10_PTS_per_gp=("current_last10_PTS_per_gp", "sum"),
            current_last10_SOG_per_gp=("current_last10_SOG_per_gp", "sum"),
        )
    )

    line_df["expected_count"] = line_df.apply(
        lambda r: expected_counts.get((r["team"], r["line"]), r["player_count"]),
        axis=1,
    )

    line_df = line_df[LINE_COLUMNS]

    return player_df.round(3), line_df.round(3), missing, source, lines


def line_sort_key(line):
    m = re.match(r"([FD])(\d+)", str(line))

    if not m:
        return 999

    base = 0 if m.group(1) == "F" else 100
    return base + int(m.group(2))


def sort_lines(df):
    if df.empty:
        return df

    df = df.copy()
    df["_line_sort"] = df["line"].apply(line_sort_key)
    df = df.sort_values(["team", "_line_sort"]).drop(columns=["_line_sort"])
    return df


def main():
    game = today_game(TEAM)
    teams = [TEAM, game["opponent"]]

    regular_mp = load_moneypuck(REGULAR_SEASON_GAME_TYPE)
    current_mp = regular_mp if game["game_type"] == REGULAR_SEASON_GAME_TYPE else load_moneypuck(game["game_type"])

    print("\n=== GAME ===")
    print(game)

    print("\nNOTE:")
    print("G_per_gp / SOG_per_gp / xGF_per_gp / xGA_per_gp = regular season")
    print("current_G_per_gp / current_SOG_per_gp / current_xGF_per_gp / current_xGA_per_gp = current game type")
    print("ixG_per_gp = individual expected goals per game")
    print("xGF_per_gp = on-ice expected goals for per game")
    print("xGA_per_gp = on-ice expected goals against per game")
    print("GA_onice_per_gp = on-ice goals against per game")

    all_players = []
    all_lines = []

    for team in teams:
        print(f"\n=== {team} ===")

        player_df, line_df, missing, source, raw_lines = build_team_line_stats(
            team,
            game["game_type"],
            regular_mp,
            current_mp,
        )

        print(f"lineup_source: {source}")

        if missing:
            print("\nmissing matches:")
            for item in missing:
                print(f'{item["team"]} {item["line"]}: {item["player"]}')

        print("\nraw lines:")
        for line, players in raw_lines.items():
            print(line, ":", " / ".join(players))

        print("\nline stats:")
        if line_df.empty:
            print("No line stats found.")
        else:
            print(sort_lines(line_df).to_string(index=False))

        all_players.append(player_df)
        all_lines.append(line_df)

    players = pd.concat(all_players, ignore_index=True) if all_players else pd.DataFrame(columns=PLAYER_COLUMNS)
    lines = pd.concat(all_lines, ignore_index=True) if all_lines else pd.DataFrame(columns=LINE_COLUMNS)

    players = players[PLAYER_COLUMNS] if not players.empty else pd.DataFrame(columns=PLAYER_COLUMNS)
    lines = lines[LINE_COLUMNS] if not lines.empty else pd.DataFrame(columns=LINE_COLUMNS)

    players.to_csv("tonight_player_line_stats.csv", index=False)
    lines.to_csv("tonight_line_stats.csv", index=False)

    print("\n=== SAVED ===")
    print("tonight_player_line_stats.csv")
    print("tonight_line_stats.csv")


if __name__ == "__main__":
    main()