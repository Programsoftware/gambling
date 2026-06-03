import requests
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO

st.set_page_config(layout="wide")
st.title("Avs vs Vegas Matchup Model")

SEASON = 20252026
HOME = "COL"
AWAY = "VGK"

SKATERS_URLS = [
    "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/playoffs/skaters.csv",
    "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters.csv",
]

DEFAULT_AVS_LINES = """Gabriel Landeskog, Nathan MacKinnon, Martin Necas
Artturi Lehkonen, Brock Nelson, Nicolas Roy
Ross Colton, Nazem Kadri, Valeri Nichushkin
Parker Kelly, Jack Drury, Logan O'Connor"""

DEFAULT_VGK_LINES = """Ivan Barbashev, Jack Eichel, Pavel Dorofeyev
Brett Howden, William Karlsson, Mitch Marner
Brandon Saad, Tomas Hertl, Colton Sissons
Cole Smith, Nic Dowd, Keegan Kolesar"""

def no_vig_prob(home_odds, away_odds):
    home_raw = 1 / home_odds
    away_raw = 1 / away_odds
    total = home_raw + away_raw
    return home_raw / total, away_raw / total

def time_to_seconds(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    if ":" not in s:
        return float(s)
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)

def overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))

@st.cache_data(ttl=3600)
def load_skaters():
    dfs = []
    for url in SKATERS_URLS:
        df = pd.read_csv(url)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

@st.cache_data(ttl=3600)
def get_schedule(team, season):
    url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()["games"]

@st.cache_data(ttl=3600)
def get_shiftcharts(game_id):
    url = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    return pd.DataFrame(data)

def get_h2h_games():
    games = get_schedule(HOME, SEASON)
    rows = []
    for g in games:
        home = g["homeTeam"]["abbrev"]
        away = g["awayTeam"]["abbrev"]
        if {home, away} == {HOME, AWAY}:
            rows.append({
                "game_id": g["id"],
                "date": g["gameDate"],
                "home": home,
                "away": away,
                "state": g.get("gameState"),
                "home_score": g["homeTeam"].get("score"),
                "away_score": g["awayTeam"].get("score"),
            })
    return pd.DataFrame(rows)

def parse_lines(text, prefix):
    rows = []
    for i, line in enumerate(text.strip().split("\n"), start=1):
        players = [p.strip() for p in line.split(",") if p.strip()]
        rows.append({"unit": f"{prefix} L{i}", "players": players})
    return rows

def get_player_row(df, name):
    hits = df[df["name"].str.lower() == name.lower()]
    hits = hits[hits["situation"].isin(["5on5", "all"])]
    if hits.empty:
        return None
    if "5on5" in hits["situation"].values:
        return hits[hits["situation"] == "5on5"].iloc[0]
    return hits.iloc[0]

def per60(row, col):
    if row is None or col not in row or row["icetime"] == 0:
        return np.nan
    return 3600 * row[col] / row["icetime"]

def player_features(df, name):
    row = get_player_row(df, name)
    return {
        "player": name,
        "xGF60": per60(row, "OnIce_F_xGoals"),
        "xGA60": per60(row, "OnIce_A_xGoals"),
        "shotsF60": per60(row, "OnIce_F_shotsOnGoal"),
        "shotsA60": per60(row, "OnIce_A_shotsOnGoal"),
        "ixG60": per60(row, "I_F_xGoals"),
        "iShots60": per60(row, "I_F_shotsOnGoal"),
    }

def unit_features(df, unit):
    rows = [player_features(df, p) for p in unit["players"]]
    x = pd.DataFrame(rows)
    return {
        "unit": unit["unit"],
        "players": ", ".join(unit["players"]),
        "xGF60": x["xGF60"].mean(),
        "xGA60": x["xGA60"].mean(),
        "shotsF60": x["shotsF60"].mean(),
        "shotsA60": x["shotsA60"].mean(),
        "ixG60": x["ixG60"].sum(),
        "iShots60": x["iShots60"].sum(),
        "missing": ", ".join(x[x["xGF60"].isna()]["player"].tolist()),
    }

def matchup_edge(attacking_unit, defending_unit):
    return (
        0.55 * attacking_unit["xGF60"]
        + 0.35 * defending_unit["xGA60"]
        + 0.05 * attacking_unit["shotsF60"]
        + 0.05 * defending_unit["shotsA60"]
    )

def clean_shiftcharts(shifts):
    if shifts.empty:
        return shifts

    shifts = shifts.copy()

    name_col = None
    for c in ["playerName", "name"]:
        if c in shifts.columns:
            name_col = c

    if name_col is None:
        shifts["player"] = shifts["firstName"].astype(str) + " " + shifts["lastName"].astype(str)
    else:
        shifts["player"] = shifts[name_col].astype(str)

    team_col = "teamAbbrev" if "teamAbbrev" in shifts.columns else "teamAbbrevName"

    shifts["team"] = shifts[team_col]
    shifts["start_sec"] = shifts["startTime"].apply(time_to_seconds)
    shifts["end_sec"] = shifts["endTime"].apply(time_to_seconds)
    shifts["duration_sec"] = shifts["end_sec"] - shifts["start_sec"]

    return shifts[
        ["gameId", "period", "team", "player", "start_sec", "end_sec", "duration_sec"]
    ].dropna()

def line_matchup_seconds(shifts, avs_lines, vgk_lines, min_players_each_side=2):
    rows = []

    if shifts.empty:
        return pd.DataFrame()

    for period in sorted(shifts["period"].unique()):
        period_shifts = shifts[shifts["period"] == period]
        cutpoints = sorted(
            set(period_shifts["start_sec"].tolist() + period_shifts["end_sec"].tolist())
        )

        for a, b in zip(cutpoints[:-1], cutpoints[1:]):
            if b <= a:
                continue

            active = period_shifts[
                (period_shifts["start_sec"] < b) &
                (period_shifts["end_sec"] > a)
            ]

            active_avs = set(active[active["team"] == HOME]["player"])
            active_vgk = set(active[active["team"] == AWAY]["player"])

            for avs in avs_lines:
                avs_on = [p for p in avs["players"] if p in active_avs]
                if len(avs_on) < min_players_each_side:
                    continue

                for vgk in vgk_lines:
                    vgk_on = [p for p in vgk["players"] if p in active_vgk]
                    if len(vgk_on) < min_players_each_side:
                        continue

                    rows.append({
                        "game_id": int(period_shifts["gameId"].iloc[0]),
                        "period": period,
                        "start_sec": a,
                        "end_sec": b,
                        "seconds": b - a,
                        "avs_line": avs["unit"],
                        "vgk_line": vgk["unit"],
                        "avs_players_on": ", ".join(avs_on),
                        "vgk_players_on": ", ".join(vgk_on),
                    })

    return pd.DataFrame(rows)

def player_pair_seconds(shifts, avs_players, vgk_players):
    rows = []

    for avs_player in avs_players:
        a = shifts[(shifts["team"] == HOME) & (shifts["player"] == avs_player)]

        for vgk_player in vgk_players:
            b = shifts[(shifts["team"] == AWAY) & (shifts["player"] == vgk_player)]

            total = 0
            for _, x in a.iterrows():
                same_period = b[b["period"] == x["period"]]
                for _, y in same_period.iterrows():
                    total += overlap(x["start_sec"], x["end_sec"], y["start_sec"], y["end_sec"])

            rows.append({
                "avs_player": avs_player,
                "vgk_player": vgk_player,
                "matched_seconds": total,
                "matched_minutes": total / 60,
            })

    return pd.DataFrame(rows)

st.sidebar.header("Market")
avs_odds = st.sidebar.number_input("Avs decimal odds", value=1.94, min_value=1.01)
vgk_odds = st.sidebar.number_input("Vegas decimal odds", value=3.45, min_value=1.01)
avs_market_prob, vgk_market_prob = no_vig_prob(avs_odds, vgk_odds)

st.sidebar.write(f"Avs no-vig: **{avs_market_prob:.2%}**")
st.sidebar.write(f"VGK no-vig: **{vgk_market_prob:.2%}**")

st.sidebar.header("Historical matchup settings")
min_players_each_side = st.sidebar.slider("Minimum players from each line on ice", 1, 3, 2)
manual_game_ids = st.sidebar.text_input(
    "Extra game IDs, comma-separated",
    value=""
)

avs_lines_text = st.text_area("Avs projected lines", DEFAULT_AVS_LINES, height=130)
vgk_lines_text = st.text_area("Vegas projected lines", DEFAULT_VGK_LINES, height=130)

avs_lines = parse_lines(avs_lines_text, "Avs")
vgk_lines = parse_lines(vgk_lines_text, "VGK")

df = load_skaters()

avs_units = pd.DataFrame([unit_features(df, u) for u in avs_lines])
vgk_units = pd.DataFrame([unit_features(df, u) for u in vgk_lines])

st.subheader("Line strength")
c1, c2 = st.columns(2)
with c1:
    st.write("Avs")
    st.dataframe(avs_units, use_container_width=True)
with c2:
    st.write("Vegas")
    st.dataframe(vgk_units, use_container_width=True)

st.subheader("Current-line matchup matrix")

rows = []
for _, avs in avs_units.iterrows():
    for _, vgk in vgk_units.iterrows():
        avs_attack = matchup_edge(avs, vgk)
        vgk_attack = matchup_edge(vgk, avs)
        rows.append({
            "matchup": f"{avs['unit']} vs {vgk['unit']}",
            "Avs_attack_score": avs_attack,
            "VGK_attack_score": vgk_attack,
            "Avs_xG_edge": avs_attack - vgk_attack,
        })

matchups = pd.DataFrame(rows)
st.dataframe(matchups.sort_values("Avs_xG_edge", ascending=False), use_container_width=True)

st.subheader("Historical shift-chart matchup data")

h2h_games = get_h2h_games()

extra_ids = []
if manual_game_ids.strip():
    extra_ids = [int(x.strip()) for x in manual_game_ids.split(",") if x.strip()]

game_ids = sorted(set(h2h_games["game_id"].dropna().astype(int).tolist() + extra_ids))

st.write("Games used:")
st.dataframe(h2h_games, use_container_width=True)

all_line_intervals = []
all_player_pairs = []

avs_players = [p for u in avs_lines for p in u["players"]]
vgk_players = [p for u in vgk_lines for p in u["players"]]

for game_id in game_ids:
    try:
        raw = get_shiftcharts(game_id)
        shifts = clean_shiftcharts(raw)

        line_intervals = line_matchup_seconds(
            shifts,
            avs_lines,
            vgk_lines,
            min_players_each_side=min_players_each_side,
        )

        pair_seconds = player_pair_seconds(shifts, avs_players, vgk_players)
        pair_seconds["game_id"] = game_id

        all_line_intervals.append(line_intervals)
        all_player_pairs.append(pair_seconds)

    except Exception as e:
        st.warning(f"Could not load game {game_id}: {e}")

if all_line_intervals:
    line_intervals = pd.concat(all_line_intervals, ignore_index=True)

    if not line_intervals.empty:
        line_summary = (
            line_intervals
            .groupby(["avs_line", "vgk_line"], as_index=False)["seconds"]
            .sum()
        )
        line_summary["minutes"] = line_summary["seconds"] / 60

        st.write("Line-vs-line historical matchup minutes")
        st.dataframe(
            line_summary.sort_values("minutes", ascending=False),
            use_container_width=True
        )

        st.write("Exact historical intervals")
        st.dataframe(
            line_intervals.sort_values(["game_id", "period", "start_sec"]),
            use_container_width=True
        )
    else:
        st.info("No line matchup intervals found. Try lowering minimum players from each line to 1.")

if all_player_pairs:
    player_pairs = pd.concat(all_player_pairs, ignore_index=True)

    player_summary = (
        player_pairs
        .groupby(["avs_player", "vgk_player"], as_index=False)["matched_seconds"]
        .sum()
    )
    player_summary["matched_minutes"] = player_summary["matched_seconds"] / 60

    st.write("Player-vs-player historical matchup minutes")
    st.dataframe(
        player_summary.sort_values("matched_minutes", ascending=False),
        use_container_width=True
    )

st.subheader("Simple probability output")

avs_total_edge = matchups["Avs_xG_edge"].mean()
line_prob_adjustment = np.tanh(avs_total_edge / 2.5) * 0.12

market_weight = st.slider("Market weight", 0.0, 1.0, 0.65, 0.05)

avs_model_prob = (
    market_weight * avs_market_prob
    + (1 - market_weight) * (avs_market_prob + line_prob_adjustment)
)

avs_model_prob = min(max(avs_model_prob, 0.01), 0.99)

st.metric("Avs model probability", f"{avs_model_prob:.2%}")
st.metric("Vegas model probability", f"{1 - avs_model_prob:.2%}")
st.write("Line-based probability adjustment:", f"{line_prob_adjustment:.2%}")