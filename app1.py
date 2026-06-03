import os
import pandas as pd
import streamlit as st

LINE_FILE = "tonight_line_stats.csv"
PLAYER_FILE = "tonight_player_line_stats.csv"

TEXT_COLS = {"team", "line", "players", "player", "pos"}

st.set_page_config(page_title="NHL Line Stats", layout="wide")
st.title("NHL Line Stats Dashboard")


def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)

    for col in df.columns:
        if col not in TEXT_COLS:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass

    return df


def pick_existing(df, cols):
    return [c for c in cols if c in df.columns]


def safe_metric(label, df, sort_col, value_col="line", ascending=False):
    if df.empty or sort_col not in df.columns or value_col not in df.columns:
        st.metric(label, "N/A")
        return

    temp = df.dropna(subset=[sort_col])

    if temp.empty:
        st.metric(label, "N/A")
        return

    row = temp.sort_values(sort_col, ascending=ascending).iloc[0]
    st.metric(label, row[value_col], round(float(row[sort_col]), 3))


lines = load_csv(LINE_FILE)
players = load_csv(PLAYER_FILE)

if lines.empty:
    st.error("No tonight_line_stats.csv found. Run: python make_line_stats.py")
    st.stop()

st.sidebar.header("Filters")

teams = sorted(lines["team"].dropna().unique()) if "team" in lines.columns else []
selected_teams = st.sidebar.multiselect("Teams", teams, default=teams)

line_options = sorted(lines["line"].dropna().unique()) if "line" in lines.columns else []
selected_lines = st.sidebar.multiselect("Lines", line_options, default=line_options)

filtered_lines = lines.copy()

if selected_teams and "team" in filtered_lines.columns:
    filtered_lines = filtered_lines[filtered_lines["team"].isin(selected_teams)]

if selected_lines and "line" in filtered_lines.columns:
    filtered_lines = filtered_lines[filtered_lines["line"].isin(selected_lines)]

available_stats = pick_existing(
    filtered_lines,
    [
        "SOG_per_gp",
        "G_per_gp",
        "PTS_per_gp",
        "xGF_per_gp",
        "xGA_per_gp",
        "GF_onice_per_gp",
        "GA_onice_per_gp",
        "ixG_per_gp",
        "current_SOG_per_gp",
        "current_G_per_gp",
        "current_PTS_per_gp",
        "current_xGF_per_gp",
        "current_xGA_per_gp",
        "current_GF_onice_per_gp",
        "current_GA_onice_per_gp",
        "current_ixG_per_gp",
    ],
)

if not available_stats:
    st.error("No usable stat columns found in tonight_line_stats.csv.")
    st.dataframe(filtered_lines, use_container_width=True, hide_index=True)
    st.stop()

default_stat = "xGF_per_gp" if "xGF_per_gp" in available_stats else available_stats[0]

selected_stat = st.sidebar.selectbox(
    "Sort table by",
    available_stats,
    index=available_stats.index(default_stat),
)

st.subheader("Line Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    safe_metric("Best SOG Line", filtered_lines, "SOG_per_gp")

with c2:
    safe_metric("Best xGF Line", filtered_lines, "xGF_per_gp")

with c3:
    safe_metric("Lowest xGA Line", filtered_lines, "xGA_per_gp", ascending=True)

with c4:
    safe_metric("Most GA Against", filtered_lines, "GA_onice_per_gp")

line_display_cols = pick_existing(
    filtered_lines,
    [
        "team",
        "line",
        "players",
        "player_count",
        "expected_count",
        "GP_avg",
        "G_per_gp",
        "A_per_gp",
        "PTS_per_gp",
        "SOG_per_gp",
        "ixG_per_gp",
        "xGF_per_gp",
        "xGA_per_gp",
        "GF_onice_per_gp",
        "GA_onice_per_gp",
        "current_GP_avg",
        "current_G_per_gp",
        "current_A_per_gp",
        "current_PTS_per_gp",
        "current_SOG_per_gp",
        "current_ixG_per_gp",
        "current_xGF_per_gp",
        "current_xGA_per_gp",
        "current_GF_onice_per_gp",
        "current_GA_onice_per_gp",
    ],
)

ascending = selected_stat in ["xGA_per_gp", "GA_onice_per_gp", "current_xGA_per_gp", "current_GA_onice_per_gp"]

filtered_lines = filtered_lines.sort_values(selected_stat, ascending=ascending)

st.dataframe(
    filtered_lines[line_display_cols],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Meaning")

st.write(
    """
- no prefix = regular season
- `current_...` = current game type, usually playoffs
- `xGF_per_gp` = on-ice expected goals for per game
- `xGA_per_gp` = on-ice expected goals against per game
- `GA_onice_per_gp` = actual goals against while on ice
- `ixG_per_gp` = individual expected goals per game
"""
)

st.subheader("Player Line Stats")

if players.empty:
    st.warning("No tonight_player_line_stats.csv found.")
else:
    filtered_players = players.copy()

    if selected_teams and "team" in filtered_players.columns:
        filtered_players = filtered_players[filtered_players["team"].isin(selected_teams)]

    if selected_lines and "line" in filtered_players.columns:
        filtered_players = filtered_players[filtered_players["line"].isin(selected_lines)]

    player_display_cols = pick_existing(
        filtered_players,
        [
            "team",
            "line",
            "player",
            "pos",
            "GP",
            "G_per_gp",
            "A_per_gp",
            "PTS_per_gp",
            "SOG_per_gp",
            "ixG_per_gp",
            "xGF_per_gp",
            "xGA_per_gp",
            "GF_onice_per_gp",
            "GA_onice_per_gp",
            "current_GP",
            "current_G_per_gp",
            "current_A_per_gp",
            "current_PTS_per_gp",
            "current_SOG_per_gp",
            "current_ixG_per_gp",
            "current_xGF_per_gp",
            "current_xGA_per_gp",
            "current_GF_onice_per_gp",
            "current_GA_onice_per_gp",
        ],
    )

    st.dataframe(
        filtered_players[player_display_cols],
        use_container_width=True,
        hide_index=True,
    )