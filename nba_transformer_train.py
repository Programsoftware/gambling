from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
NBA_DIR = ROOT / "data" / "nba"
LATEST_DIR = NBA_DIR / "latest"
MODEL_DIR = NBA_DIR / "models"
PREDICTION_DIR = NBA_DIR / "predictions"
LATEST_CHECKPOINT = MODEL_DIR / "nba_transformer_latest.pt"

IDENTITY_COLUMNS = {
    "event_id",
    "game_datetime",
    "game_date",
    "name",
    "short_name",
    "event_name",
    "status_state",
    "status_detail",
    "venue",
    "home_team_id",
    "away_team_id",
    "home_team",
    "home_abbrev",
    "away_team",
    "away_abbrev",
    "round_type",
    "season_type_name",
    "home_record",
    "away_record",
    "target_home_win",
    "target_home_margin",
}

TEAM_AVG_STATS = [
    "points_for",
    "points_against",
    "margin",
    "estimated_possessions",
    "off_rating_est",
    "def_rating_est",
    "net_rating_est",
    "efg_pct_est",
    "three_pa_rate",
    "ft_rate",
    "turnover_rate",
    "oreb_rate_proxy",
    "steal_rate_est",
    "block_rate_est",
    "foul_rate_est",
    "paint_point_rate_est",
    "field_goal_pct",
    "three_point_field_goal_pct",
    "free_throw_pct",
    "total_rebounds",
    "offensive_rebounds",
    "defensive_rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
    "fast_break_points",
    "points_in_paint",
]

PLAYER_MEAN_STATS = [
    "minutes",
    "points",
    "rebounds",
    "offensive_rebounds",
    "defensive_rebounds",
    "assists",
    "turnovers",
    "steals",
    "blocks",
    "fouls",
    "plus_minus",
    "usage_proxy",
    "points_per36",
    "rebounds_per36",
    "assists_per36",
    "turnovers_per36",
    "steals_per36",
    "blocks_per36",
    "field_goals_made_field_goals_attempted_made",
    "field_goals_made_field_goals_attempted_attempted",
    "three_point_field_goals_made_three_point_field_goals_attempted_made",
    "three_point_field_goals_made_three_point_field_goals_attempted_attempted",
    "free_throws_made_free_throws_attempted_made",
    "free_throws_made_free_throws_attempted_attempted",
]

HOME_ELO_ADVANTAGE = 65.0
ELO_K = 20.0
TEAM_TOTAL_COLUMNS = [
    "points_for",
    "points_against",
    "field_goals_made_field_goals_attempted_made",
    "field_goals_made_field_goals_attempted_attempted",
    "three_point_field_goals_made_three_point_field_goals_attempted_made",
    "three_point_field_goals_made_three_point_field_goals_attempted_attempted",
    "free_throws_made_free_throws_attempted_made",
    "free_throws_made_free_throws_attempted_attempted",
    "estimated_possessions",
    "turnovers",
    "total_turnovers",
    "offensive_rebounds",
    "total_rebounds",
    "steals",
    "blocks",
    "fouls",
    "points_in_paint",
]
PLAYER_SHOT_COLUMNS = [
    "field_goals_made_field_goals_attempted_made",
    "field_goals_made_field_goals_attempted_attempted",
    "three_point_field_goals_made_three_point_field_goals_attempted_made",
    "three_point_field_goals_made_three_point_field_goals_attempted_attempted",
    "free_throws_made_free_throws_attempted_made",
    "free_throws_made_free_throws_attempted_attempted",
]
REQUIRED_STRENGTH_COLUMNS = {
    "home_strength_elo",
    "away_strength_elo",
    "edge_strength_elo",
    "home_strength_current_season_net_rating",
    "away_strength_current_season_net_rating",
    "edge_strength_current_season_net_rating",
    "home_strength_sos_adjusted_net_rating",
    "away_strength_sos_adjusted_net_rating",
    "edge_strength_sos_adjusted_net_rating",
    "home_strength_power_top10",
    "away_strength_power_top10",
    "edge_strength_power_top10",
    "home_strength_power_rating",
    "away_strength_power_rating",
    "edge_strength_power_rating",
    "home_players_top8_minutes_weighted_points_per36",
    "away_players_top8_minutes_weighted_points_per36",
    "edge_players_top8_minutes_weighted_points_per36",
    "home_schedule_back_to_back",
    "away_schedule_back_to_back",
    "edge_schedule_back_to_back",
    "edge_style_last20_scoring_matchup",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


class FeatureTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(8, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, d_model // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_projection(x.unsqueeze(-1)) + self.feature_embedding.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(-1)


class TabularMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        second_dim = max(16, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_model(n_features: int, args: argparse.Namespace) -> nn.Module:
    model_type = getattr(args, "model_type", "transformer")
    if model_type == "mlp":
        return TabularMLP(
            n_features=n_features,
            hidden_dim=int(getattr(args, "mlp_hidden_dim", 128)),
            dropout=float(args.dropout),
        )
    return FeatureTransformer(
        n_features=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )


def output_layer(model: nn.Module) -> nn.Linear | None:
    if isinstance(model, TabularMLP):
        layer = model.net[-1]
        return layer if isinstance(layer, nn.Linear) else None
    if isinstance(model, FeatureTransformer):
        layer = model.head[-1]
        return layer if isinstance(layer, nn.Linear) else None
    return None


def initialize_output_bias(model: nn.Module, positive_rate: float) -> None:
    layer = output_layer(model)
    if layer is None:
        return
    p = min(max(float(positive_rate), 1e-4), 1 - 1e-4)
    with torch.no_grad():
        layer.bias.fill_(math.log(p / (1.0 - p)))
        layer.weight.mul_(0.05)


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    path = LATEST_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing NBA data artifact: {path}")
    return pd.read_csv(path)


def load_cached_historical_moneylines() -> pd.DataFrame | None:
    candidates = [
        NBA_DIR / "backtests" / "transformer" / "nba_historical_moneylines.csv",
        LATEST_DIR / "historical_moneylines.csv",
    ]
    for path in candidates:
        if path.exists():
            moneylines = pd.read_csv(path)
            if "event_id" in moneylines.columns:
                moneylines["event_id"] = moneylines["event_id"].astype(str)
            return moneylines
    return None


def numeric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def fast_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return None if math.isnan(value) else value
    return numeric(value)


def truthy_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "1.0", "yes", "y"}
    return bool(value)


def context_text(game_dict: dict[str, Any]) -> str:
    fields = [
        "round_type",
        "season_type_name",
        "status_detail",
        "event_name",
        "name",
        "short_name",
    ]
    return " ".join(str(game_dict.get(field) or "") for field in fields).lower()


def infer_season_type(game_dict: dict[str, Any]) -> int | None:
    value = coerce_int(game_dict.get("season_type"))
    if value is not None:
        return value

    text = context_text(game_dict)
    playoff_markers = ["postseason", "post-season", "playoff", "final", "semifinal", "quarterfinal", "round of 16"]
    if any(marker in text for marker in playoff_markers):
        return 3

    game_date = pd.to_datetime(game_dict.get("game_date"), errors="coerce")
    if not pd.isna(game_date) and int(game_date.month) in {5, 6}:
        return 3
    return None


def infer_round_type(game_dict: dict[str, Any], season_type: int | None) -> Any:
    raw = game_dict.get("round_type")
    if raw is not None and not pd.isna(raw) and str(raw).strip():
        return raw
    text = context_text(game_dict)
    if "semifinal" in text or "semi-final" in text:
        return "Semifinal"
    if "quarterfinal" in text or "quarter-final" in text:
        return "Quarterfinal"
    if "nba final" in text or "championship" in text:
        return "Final"
    game_date = pd.to_datetime(game_dict.get("game_date"), errors="coerce")
    if season_type == 3 and not pd.isna(game_date) and int(game_date.month) == 6:
        return "Final"
    return raw


def is_playoff_round(round_type: Any) -> int:
    text = str(round_type or "").lower()
    return int("final" in text or "playoff" in text or "round" in text)


def is_finals_round(round_type: Any) -> int:
    text = str(round_type or "").strip().lower()
    if not text:
        return 0
    non_finals = ["semifinal", "semi-final", "quarterfinal", "quarter-final", "conference"]
    if any(marker in text for marker in non_finals):
        return 0
    return int(text in {"final", "finals", "nba finals", "championship", "championship round"} or "nba final" in text)


def prefix(prefix_text: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix_text}_{key}": value for key, value in values.items()}


def edge(prefix_text: str, home: dict[str, Any], away: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, home_value in home.items():
        away_value = away.get(key)
        if isinstance(home_value, (int, float, np.integer, np.floating)) and isinstance(
            away_value, (int, float, np.integer, np.floating)
        ):
            out[f"{prefix_text}_{key}"] = float(home_value) - float(away_value)
    return out


def safe_div(numerator: Any, denominator: Any, scale: float = 1.0) -> float | None:
    top = fast_float(numerator)
    bottom = fast_float(denominator)
    if top is None or bottom is None or abs(bottom) < 1e-12:
        return None
    return float(scale) * top / bottom


def numeric_series(rows: pd.DataFrame, column: str) -> pd.Series:
    if rows.empty or column not in rows.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def sum_numeric_column(rows: pd.DataFrame, column: str) -> float | None:
    values = numeric_series(rows, column).dropna()
    return float(values.sum()) if not values.empty else None


def mean_numeric_column(rows: pd.DataFrame, column: str) -> float | None:
    values = numeric_series(rows, column).dropna()
    return float(values.mean()) if not values.empty else None


def mean_record_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [fast_float(row.get(key)) for row in records]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def sum_record_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [fast_float(row.get(key)) for row in records]
    values = [value for value in values if value is not None]
    return float(sum(values)) if values else None


def weighted_record_average(
    records: list[dict[str, Any]],
    value_key: str,
    weight_key: str | None = None,
    weight_func: Any | None = None,
) -> float | None:
    total = 0.0
    total_weight = 0.0
    for row in records:
        value = fast_float(row.get(value_key))
        if value is None:
            continue
        if weight_func is not None:
            weight = fast_float(weight_func(row))
        elif weight_key is not None:
            weight = fast_float(row.get(weight_key))
        else:
            weight = 1.0
        if weight is None or weight <= 0:
            continue
        total += value * weight
        total_weight += weight
    return total / total_weight if total_weight > 0 else None


def diff_value(left: Any, right: Any) -> float | None:
    left_value = fast_float(left)
    right_value = fast_float(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def safe_logit(probability: float | None) -> float | None:
    if probability is None:
        return None
    p = min(max(float(probability), 1e-4), 1 - 1e-4)
    return math.log(p / (1.0 - p))


def initial_strength_state() -> dict[str, Any]:
    return {
        "elo": 1500.0,
        "games": 0,
        "wins": 0,
        "losses": 0,
        "margin_sum": 0.0,
        "last20_margins": deque(maxlen=20),
        "last20_opp_net_ratings": deque(maxlen=20),
        "last20_opp_elos": deque(maxlen=20),
        "sos_opp_net_sum": 0.0,
        "sos_games": 0,
        "quality_games": 0,
        "quality_wins": 0,
        "bad_games": 0,
        "bad_losses": 0,
        "market_games": 0,
        "market_expected_wins": 0.0,
        "market_actual_wins": 0.0,
        "market_overperformance_sum": 0.0,
        "market_prob_sum": 0.0,
        "market_logit_sum": 0.0,
        "market_spread_games": 0,
        "market_spread_sum": 0.0,
        "market_margin_vs_spread_sum": 0.0,
    }


def strength_win_pct(state: dict[str, Any]) -> float:
    games = int(state.get("games") or 0)
    return float(state.get("wins") or 0) / games if games else 0.5


def strength_avg_margin(state: dict[str, Any]) -> float:
    games = int(state.get("games") or 0)
    return float(state.get("margin_sum") or 0.0) / games if games else 0.0


def strength_last20_margin(state: dict[str, Any]) -> float:
    margins = list(state.get("last20_margins") or [])
    return float(np.mean(margins)) if margins else 0.0


def strength_sos_margin(state: dict[str, Any]) -> float:
    games = int(state.get("sos_games") or 0)
    return float(state.get("sos_opp_net_sum") or 0.0) / games if games else 0.0


def strength_market_power_rating(state: dict[str, Any]) -> float | None:
    market_games = int(state.get("market_games") or 0)
    if not market_games:
        return None
    market_avg_logit = float(state.get("market_logit_sum") or 0.0) / market_games
    return (400.0 / math.log(10.0)) * market_avg_logit


def strength_power_rating(state: dict[str, Any]) -> float:
    elo = float(state.get("elo") or 1500.0)
    avg_margin = strength_avg_margin(state)
    last20 = strength_last20_margin(state)
    sos = strength_sos_margin(state)
    return elo + (6.0 * last20) + (3.0 * avg_margin) + (2.0 * sos)


def moneylines_by_event(moneylines: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if moneylines is None or moneylines.empty or "event_id" not in moneylines.columns:
        return {}
    rows = moneylines.copy()
    rows["event_id"] = rows["event_id"].astype(str)
    return {str(row["event_id"]): row for row in rows.to_dict("records")}


def market_probability_for_side(odds_row: dict[str, Any] | None, side: str) -> float | None:
    if not odds_row:
        return None
    home_prob = numeric(odds_row.get("market_home_no_vig_prob"))
    if home_prob is None:
        return None
    return home_prob if side == "home" else 1.0 - home_prob


def rank_maps(team_states: dict[str, dict[str, Any]], metric: str) -> dict[str, int]:
    if metric == "elo":
        key = lambda item: (float(item[1].get("elo") or 1500.0), strength_win_pct(item[1]), strength_avg_margin(item[1]))
    elif metric == "power":
        key = lambda item: (strength_power_rating(item[1]), float(item[1].get("elo") or 1500.0), strength_win_pct(item[1]))
    else:
        key = lambda item: (strength_win_pct(item[1]), strength_avg_margin(item[1]), float(item[1].get("elo") or 1500.0))
    ranked = sorted(team_states.items(), key=key, reverse=True)
    return {team_id: rank for rank, (team_id, _) in enumerate(ranked, start=1)}


def strength_snapshot(
    team_id: str,
    team_states: dict[str, dict[str, Any]],
    elo_ranks: dict[str, int],
    win_ranks: dict[str, int],
    power_ranks: dict[str, int],
) -> dict[str, Any]:
    state = team_states.setdefault(str(team_id), initial_strength_state())
    teams_count = max(len(team_states), 1)
    elo = float(state.get("elo") or 1500.0)
    games = int(state.get("games") or 0)
    avg_margin = strength_avg_margin(state)
    last20 = strength_last20_margin(state)
    sos = strength_sos_margin(state)
    last20_opp_net = list(state.get("last20_opp_net_ratings") or [])
    last20_opp_elo = list(state.get("last20_opp_elos") or [])
    last20_sos = float(np.mean(last20_opp_net)) if last20_opp_net else 0.0
    market_games = int(state.get("market_games") or 0)
    market_expected_win_pct = float(state.get("market_expected_wins") or 0.0) / market_games if market_games else None
    market_actual_win_pct = float(state.get("market_actual_wins") or 0.0) / market_games if market_games else None
    market_avg_prob = float(state.get("market_prob_sum") or 0.0) / market_games if market_games else None
    market_avg_logit = float(state.get("market_logit_sum") or 0.0) / market_games if market_games else None
    market_power = strength_market_power_rating(state)
    market_spread_games = int(state.get("market_spread_games") or 0)
    elo_rank = int(elo_ranks.get(str(team_id), teams_count))
    win_rank = int(win_ranks.get(str(team_id), teams_count))
    power_rating = strength_power_rating(state)
    power_rank = int(power_ranks.get(str(team_id), teams_count))
    quality_games = int(state.get("quality_games") or 0)
    bad_games = int(state.get("bad_games") or 0)
    return {
        "elo": elo,
        "elo_rating_over_1500": elo - 1500.0,
        "elo_rank": elo_rank,
        "elo_percentile": 1.0 - ((elo_rank - 1) / max(teams_count - 1, 1)),
        "elo_top3": int(elo_rank <= 3),
        "elo_top5": int(elo_rank <= 5),
        "elo_top10": int(elo_rank <= 10),
        "power_rating": power_rating,
        "power_rating_over_1500": power_rating - 1500.0,
        "power_rating_rank": power_rank,
        "power_rating_percentile": 1.0 - ((power_rank - 1) / max(teams_count - 1, 1)),
        "power_top3": int(power_rank <= 3),
        "power_top5": int(power_rank <= 5),
        "power_top10": int(power_rank <= 10),
        "current_season_games": games,
        "current_season_wins": int(state.get("wins") or 0),
        "current_season_losses": int(state.get("losses") or 0),
        "current_season_win_pct": strength_win_pct(state),
        "current_season_net_rating": avg_margin,
        "last20_net_rating": last20,
        "last20_opponent_avg_elo": float(np.mean(last20_opp_elo)) if last20_opp_elo else None,
        "last20_opponent_avg_net_rating": last20_sos if last20_opp_net else None,
        "last20_sos_adjusted_net_rating": last20 + last20_sos,
        "sos_net_rating": sos,
        "sos_adjusted_net_rating": avg_margin + sos,
        "overall_seed": win_rank,
        "win_pct_rank": win_rank,
        "win_pct_percentile": 1.0 - ((win_rank - 1) / max(teams_count - 1, 1)),
        "quality_games": quality_games,
        "quality_wins": int(state.get("quality_wins") or 0),
        "quality_win_rate": float(state.get("quality_wins") or 0) / quality_games if quality_games else None,
        "bad_games": bad_games,
        "bad_losses": int(state.get("bad_losses") or 0),
        "bad_loss_rate": float(state.get("bad_losses") or 0) / bad_games if bad_games else None,
        "market_games": market_games,
        "market_expected_win_pct": market_expected_win_pct,
        "market_actual_win_pct": market_actual_win_pct,
        "market_overperformance": float(state.get("market_overperformance_sum") or 0.0) / market_games if market_games else None,
        "market_avg_implied_prob": market_avg_prob,
        "market_avg_logit_prob": market_avg_logit,
        "market_power_rating": market_power,
        "market_implied_elo": 1500.0 + market_power if market_power is not None else None,
        "market_spread_games": market_spread_games,
        "market_avg_spread_prior": float(state.get("market_spread_sum") or 0.0) / market_spread_games if market_spread_games else None,
        "market_margin_vs_spread": float(state.get("market_margin_vs_spread_sum") or 0.0) / market_spread_games if market_spread_games else None,
    }


def series_side_snapshot(
    team_id: str,
    opponent_id: str,
    series_state: dict[str, Any] | None,
    current_home_id: str,
) -> dict[str, Any]:
    if not series_state:
        return {
            "games_played": 0,
            "game_number": 1,
            "wins": 0,
            "losses": 0,
            "win_pct": None,
            "lead": 0,
            "home_court": int(str(team_id) == str(current_home_id)),
            "prev_game_margin": None,
            "elimination_risk": 0,
            "clinching_opportunity": 0,
        }
    wins = int(series_state.get("wins", {}).get(str(team_id), 0))
    losses = int(series_state.get("wins", {}).get(str(opponent_id), 0))
    games = int(series_state.get("games") or 0)
    home_court_team = str(series_state.get("home_court_team") or current_home_id)
    return {
        "games_played": games,
        "game_number": games + 1,
        "wins": wins,
        "losses": losses,
        "win_pct": wins / games if games else None,
        "lead": wins - losses,
        "home_court": int(str(team_id) == home_court_team),
        "prev_game_margin": series_state.get("last_margin_by_team", {}).get(str(team_id)),
        "elimination_risk": int(losses >= 3),
        "clinching_opportunity": int(wins >= 3),
    }


def elo_expected(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (((away_elo) - (home_elo + HOME_ELO_ADVANTAGE)) / 400.0))


def update_elo(home_state: dict[str, Any], away_state: dict[str, Any], home_win: int, home_margin: float) -> None:
    home_elo = float(home_state.get("elo") or 1500.0)
    away_elo = float(away_state.get("elo") or 1500.0)
    expected = elo_expected(home_elo, away_elo)
    actual = float(home_win)
    elo_diff = (home_elo + HOME_ELO_ADVANTAGE) - away_elo
    mov = math.log(abs(float(home_margin)) + 1.0)
    multiplier = mov * (2.2 / ((abs(elo_diff) * 0.001) + 2.2))
    change = ELO_K * multiplier * (actual - expected)
    home_state["elo"] = home_elo + change
    away_state["elo"] = away_elo - change


def update_strength_state(
    team_state: dict[str, Any],
    opponent_pre_margin: float,
    opponent_pre_elo: float,
    opponent_pre_power_rank: int,
    teams_count: int,
    margin: float,
    won: bool,
    market_prob: float | None,
    market_spread: float | None,
) -> None:
    team_state["games"] = int(team_state.get("games") or 0) + 1
    team_state["wins"] = int(team_state.get("wins") or 0) + int(bool(won))
    team_state["losses"] = int(team_state.get("losses") or 0) + int(not bool(won))
    team_state["margin_sum"] = float(team_state.get("margin_sum") or 0.0) + float(margin)
    team_state.setdefault("last20_margins", deque(maxlen=20)).append(float(margin))
    team_state.setdefault("last20_opp_net_ratings", deque(maxlen=20)).append(float(opponent_pre_margin))
    team_state.setdefault("last20_opp_elos", deque(maxlen=20)).append(float(opponent_pre_elo))
    team_state["sos_opp_net_sum"] = float(team_state.get("sos_opp_net_sum") or 0.0) + float(opponent_pre_margin)
    team_state["sos_games"] = int(team_state.get("sos_games") or 0) + 1
    if opponent_pre_power_rank <= 10:
        team_state["quality_games"] = int(team_state.get("quality_games") or 0) + 1
        team_state["quality_wins"] = int(team_state.get("quality_wins") or 0) + int(bool(won))
    if opponent_pre_power_rank >= max(teams_count - 9, 1):
        team_state["bad_games"] = int(team_state.get("bad_games") or 0) + 1
        team_state["bad_losses"] = int(team_state.get("bad_losses") or 0) + int(not bool(won))
    if market_prob is not None:
        team_state["market_games"] = int(team_state.get("market_games") or 0) + 1
        team_state["market_expected_wins"] = float(team_state.get("market_expected_wins") or 0.0) + float(market_prob)
        team_state["market_actual_wins"] = float(team_state.get("market_actual_wins") or 0.0) + float(bool(won))
        team_state["market_overperformance_sum"] = (
            float(team_state.get("market_overperformance_sum") or 0.0) + float(bool(won)) - float(market_prob)
        )
        team_state["market_prob_sum"] = float(team_state.get("market_prob_sum") or 0.0) + float(market_prob)
        logit = safe_logit(market_prob)
        if logit is not None:
            team_state["market_logit_sum"] = float(team_state.get("market_logit_sum") or 0.0) + logit
    if market_spread is not None:
        team_state["market_spread_games"] = int(team_state.get("market_spread_games") or 0) + 1
        team_state["market_spread_sum"] = float(team_state.get("market_spread_sum") or 0.0) + float(market_spread)
        team_state["market_margin_vs_spread_sum"] = (
            float(team_state.get("market_margin_vs_spread_sum") or 0.0) + float(margin) + float(market_spread)
        )


def prepare_strength_context(schedule: pd.DataFrame, moneylines: pd.DataFrame | None = None) -> dict[str, Any]:
    if schedule.empty:
        return {"event_features": {}}

    games = schedule.copy()
    games["event_id"] = games["event_id"].astype(str)
    games["_sort_datetime"] = pd.to_datetime(games.get("game_datetime"), errors="coerce")
    games["_sort_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games = games.drop_duplicates(subset=["event_id"]).sort_values(["_sort_datetime", "_sort_date", "event_id"], na_position="last")
    all_team_ids = sorted(set(games["home_team_id"].dropna().astype(str)) | set(games["away_team_id"].dropna().astype(str)))
    odds_by_event = moneylines_by_event(moneylines)
    season_states: dict[int, dict[str, dict[str, Any]]] = {}
    series_states: dict[tuple[int, tuple[str, str]], dict[str, Any]] = {}
    event_features: dict[str, dict[str, Any]] = {}

    for _, game in games.iterrows():
        event_id = str(game["event_id"])
        home_id = str(game["home_team_id"])
        away_id = str(game["away_team_id"])
        game_date = str(game["game_date"])
        season = nba_season_from_game_date(game_date)
        if season is None:
            continue
        team_states = season_states.setdefault(season, {team_id: initial_strength_state() for team_id in all_team_ids})
        home_state = team_states.setdefault(home_id, initial_strength_state())
        away_state = team_states.setdefault(away_id, initial_strength_state())
        elo_ranks = rank_maps(team_states, "elo")
        win_ranks = rank_maps(team_states, "wins")
        power_ranks = rank_maps(team_states, "power")
        home_strength = strength_snapshot(home_id, team_states, elo_ranks, win_ranks, power_ranks)
        away_strength = strength_snapshot(away_id, team_states, elo_ranks, win_ranks, power_ranks)

        row_features: dict[str, Any] = {}
        row_features.update(prefix("home_strength", home_strength))
        row_features.update(prefix("away_strength", away_strength))
        row_features.update(edge("edge_strength", home_strength, away_strength))

        season_type = coerce_int(game.get("season_type"))
        if season_type == 3:
            pair = tuple(sorted([home_id, away_id]))
            key = (season, pair)
            series_state = series_states.get(key)
            home_series = series_side_snapshot(home_id, away_id, series_state, home_id)
            away_series = series_side_snapshot(away_id, home_id, series_state, home_id)
            row_features.update(prefix("home_series", home_series))
            row_features.update(prefix("away_series", away_series))
            row_features.update(edge("edge_series", home_series, away_series))
        event_features[event_id] = row_features

        completed = truthy_value(game.get("completed")) and numeric(game.get("home_win")) is not None and numeric(game.get("home_margin")) is not None
        if not completed:
            continue

        home_margin = float(numeric(game.get("home_margin")) or 0.0)
        home_win = int(float(game.get("home_win")))
        home_pre_margin = strength_avg_margin(home_state)
        away_pre_margin = strength_avg_margin(away_state)
        home_pre_elo = float(home_state.get("elo") or 1500.0)
        away_pre_elo = float(away_state.get("elo") or 1500.0)
        home_pre_power_rank = int(power_ranks.get(home_id, len(team_states)))
        away_pre_power_rank = int(power_ranks.get(away_id, len(team_states)))
        teams_count = max(len(team_states), 1)
        odds_row = odds_by_event.get(event_id)
        home_market_prob = market_probability_for_side(odds_row, "home")
        away_market_prob = market_probability_for_side(odds_row, "away")
        home_market_spread = numeric(odds_row.get("market_spread")) if odds_row else None
        away_market_spread = -home_market_spread if home_market_spread is not None else None

        update_elo(home_state, away_state, home_win, home_margin)
        update_strength_state(
            home_state,
            away_pre_margin,
            away_pre_elo,
            away_pre_power_rank,
            teams_count,
            home_margin,
            home_win == 1,
            home_market_prob,
            home_market_spread,
        )
        update_strength_state(
            away_state,
            home_pre_margin,
            home_pre_elo,
            home_pre_power_rank,
            teams_count,
            -home_margin,
            home_win == 0,
            away_market_prob,
            away_market_spread,
        )

        if season_type == 3:
            pair = tuple(sorted([home_id, away_id]))
            key = (season, pair)
            series_state = series_states.setdefault(
                key,
                {
                    "games": 0,
                    "wins": {home_id: 0, away_id: 0},
                    "home_court_team": home_id,
                    "last_margin_by_team": {},
                },
            )
            series_state["games"] = int(series_state.get("games") or 0) + 1
            winner_id = home_id if home_win == 1 else away_id
            series_state.setdefault("wins", {}).setdefault(home_id, 0)
            series_state.setdefault("wins", {}).setdefault(away_id, 0)
            series_state["wins"][winner_id] = int(series_state["wins"].get(winner_id, 0)) + 1
            series_state.setdefault("last_margin_by_team", {})[home_id] = home_margin
            series_state.setdefault("last_margin_by_team", {})[away_id] = -home_margin
            if not series_state.get("home_court_team"):
                series_state["home_court_team"] = home_id

    return {"event_features": event_features}


def strength_features_for_game(game_dict: dict[str, Any], strength_context: dict[str, Any] | None) -> dict[str, Any]:
    if not strength_context:
        return {}
    return dict((strength_context.get("event_features") or {}).get(str(game_dict.get("event_id")), {}))


def team_games_before(team_boxscores: pd.DataFrame, team_id: str, game_date: str, opponent_id: str | None = None) -> pd.DataFrame:
    rows = team_boxscores[
        (team_boxscores["team_id"].astype(str) == str(team_id))
        & (team_boxscores["game_date"].astype(str) < str(game_date))
    ].copy()
    if opponent_id is not None:
        rows = rows[rows["opponent_team_id"].astype(str) == str(opponent_id)]
    return rows.sort_values(["game_date", "event_id"])


def current_season_rows(rows: pd.DataFrame, game_date: str) -> pd.DataFrame:
    if rows.empty or "game_date" not in rows.columns:
        return rows
    target_season = nba_season_from_game_date(game_date)
    if target_season is None:
        return rows
    season_keys = pd.to_datetime(rows["game_date"], errors="coerce").apply(nba_season_from_game_date)
    return rows[season_keys == target_season].copy()


def team_aggregate_from_rows(rows: pd.DataFrame, game_date: str) -> dict[str, Any]:
    games = int(len(rows))
    out: dict[str, Any] = {
        "games": games,
        "win_pct": mean_numeric_column(rows, "won") if games else None,
    }
    if not rows.empty:
        if "_game_date_dt" in rows.columns:
            last_game_date = rows["_game_date_dt"].max()
        else:
            last_game_date = pd.to_datetime(rows["game_date"], errors="coerce").max()
        target_date = pd.to_datetime(game_date, errors="coerce")
        out["rest_days"] = max(float((target_date - last_game_date).days), 0.0) if not pd.isna(target_date) else None
    else:
        out["rest_days"] = None

    for stat in TEAM_AVG_STATS:
        out[f"avg_{stat}"] = mean_numeric_column(rows, stat)

    fgm = sum_numeric_column(rows, "field_goals_made_field_goals_attempted_made")
    fga = sum_numeric_column(rows, "field_goals_made_field_goals_attempted_attempted")
    tpm = sum_numeric_column(rows, "three_point_field_goals_made_three_point_field_goals_attempted_made")
    tpa = sum_numeric_column(rows, "three_point_field_goals_made_three_point_field_goals_attempted_attempted")
    ftm = sum_numeric_column(rows, "free_throws_made_free_throws_attempted_made")
    fta = sum_numeric_column(rows, "free_throws_made_free_throws_attempted_attempted")
    points_for = sum_numeric_column(rows, "points_for")
    points_against = sum_numeric_column(rows, "points_against")
    possessions = sum_numeric_column(rows, "estimated_possessions")
    turnovers = sum_numeric_column(rows, "total_turnovers")
    if turnovers is None:
        turnovers = sum_numeric_column(rows, "turnovers")
    offensive_rebounds = sum_numeric_column(rows, "offensive_rebounds")
    total_rebounds = sum_numeric_column(rows, "total_rebounds")
    steals = sum_numeric_column(rows, "steals")
    blocks = sum_numeric_column(rows, "blocks")
    fouls = sum_numeric_column(rows, "fouls")
    points_in_paint = sum_numeric_column(rows, "points_in_paint")

    off_rating = safe_div(points_for, possessions, 100.0)
    def_rating = safe_div(points_against, possessions, 100.0)
    recomputed = {
        "avg_estimated_possessions": safe_div(possessions, games) if games else None,
        "avg_field_goal_pct": safe_div(fgm, fga, 100.0),
        "avg_three_point_field_goal_pct": safe_div(tpm, tpa, 100.0),
        "avg_free_throw_pct": safe_div(ftm, fta, 100.0),
        "avg_efg_pct_est": safe_div((fgm or 0.0) + (0.5 * (tpm or 0.0)), fga),
        "avg_three_pa_rate": safe_div(tpa, fga),
        "avg_ft_rate": safe_div(fta, fga),
        "avg_off_rating_est": off_rating,
        "avg_def_rating_est": def_rating,
        "avg_net_rating_est": diff_value(off_rating, def_rating),
        "avg_turnover_rate": safe_div(turnovers, possessions),
        "avg_oreb_rate_proxy": safe_div(offensive_rebounds, total_rebounds),
        "avg_steal_rate_est": safe_div(steals, possessions),
        "avg_block_rate_est": safe_div(blocks, possessions),
        "avg_foul_rate_est": safe_div(fouls, possessions),
        "avg_paint_point_rate_est": safe_div(points_in_paint, possessions),
    }
    for key, value in recomputed.items():
        if value is not None:
            out[key] = value
    return out


def schedule_context_from_rows(rows: pd.DataFrame, game_date: str, current_home_away: str) -> dict[str, Any]:
    target_date = pd.to_datetime(game_date, errors="coerce")
    current_side = str(current_home_away or "").strip().lower()
    out: dict[str, Any] = {
        "rest_days": None,
        "back_to_back": 0,
        "three_in_four": 0,
        "four_in_six": 0,
        "games_last_7_days": 0,
        "games_last_14_days": 0,
        "road_trip_game_number": 1 if current_side == "away" else 0,
        "home_stand_game_number": 1 if current_side == "home" else 0,
        "travel_flip_from_last_game": 0,
    }
    if rows.empty or pd.isna(target_date):
        return out

    sort_cols = ["_game_date_dt", "event_id"] if "_game_date_dt" in rows.columns else ["game_date", "event_id"]
    rows = rows.sort_values(sort_cols)
    date_values = rows["_game_date_dt"] if "_game_date_dt" in rows.columns else pd.to_datetime(rows["game_date"], errors="coerce")
    dates = [date for date in date_values if not pd.isna(date)]
    if not dates:
        return out

    last_game_date = max(dates)
    rest_days = max(float((target_date - last_game_date).days), 0.0)
    days_since = [int((target_date - date).days) for date in dates]
    out["rest_days"] = rest_days
    out["back_to_back"] = int(rest_days <= 1.0)
    out["three_in_four"] = int(sum(1 for days in days_since if 1 <= days <= 3) >= 2)
    out["four_in_six"] = int(sum(1 for days in days_since if 1 <= days <= 5) >= 3)
    out["games_last_7_days"] = int(sum(1 for days in days_since if 1 <= days <= 7))
    out["games_last_14_days"] = int(sum(1 for days in days_since if 1 <= days <= 14))

    if "home_away" in rows.columns:
        sides = rows["home_away"].astype(str).str.lower().tolist()
        last_side = sides[-1] if sides else ""
        out["travel_flip_from_last_game"] = int(bool(last_side) and bool(current_side) and last_side != current_side)
        streak = 1
        for side in reversed(sides):
            if side == current_side:
                streak += 1
            else:
                break
        if current_side == "away":
            out["road_trip_game_number"] = streak
        elif current_side == "home":
            out["home_stand_game_number"] = streak
    return out


def team_schedule_context(team_boxscores: pd.DataFrame, team_id: str, game_date: str, current_home_away: str) -> dict[str, Any]:
    rows = team_games_before(team_boxscores, team_id, game_date)
    rows = current_season_rows(rows, game_date)
    return schedule_context_from_rows(rows, game_date, current_home_away)


def team_schedule_context_indexed(context: dict[str, Any], team_id: str, game_date: str, current_home_away: str) -> dict[str, Any]:
    rows = indexed_rows_before(context["team_by_id"].get(str(team_id)), game_date)
    target_season = nba_season_from_game_date(game_date)
    rows = rows[rows["_season_key"] == target_season] if not rows.empty else rows
    return schedule_context_from_rows(rows, game_date, current_home_away)


def style_matchup_features(home_splits: dict[str, dict[str, Any]], away_splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["all", "regular", "playoffs", "last5", "last10", "last20", "h2h"]:
        home = home_splits.get(split, {})
        away = away_splits.get(split, {})
        home_scoring = diff_value(home.get("avg_points_for"), away.get("avg_points_against"))
        away_scoring = diff_value(away.get("avg_points_for"), home.get("avg_points_against"))
        out[f"home_style_{split}_points_for_vs_away_points_against"] = home_scoring
        out[f"away_style_{split}_points_for_vs_home_points_against"] = away_scoring
        out[f"edge_style_{split}_scoring_matchup"] = diff_value(home_scoring, away_scoring)

        home_pace = fast_float(home.get("avg_estimated_possessions"))
        away_pace = fast_float(away.get("avg_estimated_possessions"))
        pace_values = [value for value in [home_pace, away_pace] if value is not None]
        out[f"edge_style_{split}_expected_pace"] = float(np.mean(pace_values)) if pace_values else None
        out[f"edge_style_{split}_pace"] = diff_value(home_pace, away_pace)
        out[f"edge_style_{split}_net_rating"] = diff_value(home.get("avg_net_rating_est"), away.get("avg_net_rating_est"))
        out[f"edge_style_{split}_three_pa_rate"] = diff_value(home.get("avg_three_pa_rate"), away.get("avg_three_pa_rate"))

        home_ft_pressure = diff_value(home.get("avg_ft_rate"), away.get("avg_foul_rate_est"))
        away_ft_pressure = diff_value(away.get("avg_ft_rate"), home.get("avg_foul_rate_est"))
        out[f"home_style_{split}_ft_pressure"] = home_ft_pressure
        out[f"away_style_{split}_ft_pressure"] = away_ft_pressure
        out[f"edge_style_{split}_ft_pressure"] = diff_value(home_ft_pressure, away_ft_pressure)

        home_turnover_risk = None
        away_turnover_risk = None
        home_tov = fast_float(home.get("avg_turnover_rate"))
        away_tov = fast_float(away.get("avg_turnover_rate"))
        home_steal = fast_float(home.get("avg_steal_rate_est"))
        away_steal = fast_float(away.get("avg_steal_rate_est"))
        if home_tov is not None and away_steal is not None:
            home_turnover_risk = home_tov + away_steal
        if away_tov is not None and home_steal is not None:
            away_turnover_risk = away_tov + home_steal
        out[f"home_style_{split}_turnover_risk"] = home_turnover_risk
        out[f"away_style_{split}_turnover_risk"] = away_turnover_risk
        out[f"edge_style_{split}_turnover_pressure"] = diff_value(away_turnover_risk, home_turnover_risk)

        home_paint_pressure = diff_value(home.get("avg_paint_point_rate_est"), away.get("avg_block_rate_est"))
        away_paint_pressure = diff_value(away.get("avg_paint_point_rate_est"), home.get("avg_block_rate_est"))
        out[f"home_style_{split}_paint_pressure"] = home_paint_pressure
        out[f"away_style_{split}_paint_pressure"] = away_paint_pressure
        out[f"edge_style_{split}_paint_pressure"] = diff_value(home_paint_pressure, away_paint_pressure)
    return out


def aggregate_team(
    team_boxscores: pd.DataFrame,
    team_id: str,
    game_date: str,
    split: str,
    opponent_id: str | None = None,
) -> dict[str, Any]:
    rows = team_games_before(team_boxscores, team_id, game_date, opponent_id=opponent_id)
    if split == "season":
        rows = current_season_rows(rows, game_date)
    elif split == "regular":
        rows = current_season_rows(rows, game_date)
        rows = rows[pd.to_numeric(rows["season_type"], errors="coerce") == 2]
    elif split == "playoffs":
        rows = current_season_rows(rows, game_date)
        rows = rows[pd.to_numeric(rows["season_type"], errors="coerce") == 3]
    elif split.startswith("last"):
        rows = current_season_rows(rows, game_date)
        rows = rows.tail(int(split.replace("last", "")))

    return team_aggregate_from_rows(rows, game_date)


def player_usage_minutes_weight(row: dict[str, Any]) -> float:
    minutes = fast_float(row.get("minutes")) or 0.0
    usage = fast_float(row.get("usage_proxy")) or 0.0
    return max(minutes, 0.0) * max(usage, 0.0)


def player_records(frame_or_records: Any) -> list[dict[str, Any]]:
    if isinstance(frame_or_records, pd.DataFrame):
        return frame_or_records.to_dict("records")
    return list(frame_or_records or [])


def add_player_group_features(out: dict[str, Any], label: str, frame_or_records: Any) -> None:
    records = player_records(frame_or_records)
    out[f"{label}_count"] = int(len(records))
    for stat in PLAYER_MEAN_STATS:
        values = [fast_float(player.get(stat)) for player in records]
        values = [value for value in values if value is not None]
        out[f"{label}_sum_{stat}"] = float(sum(values)) if records else None
        out[f"{label}_avg_{stat}"] = float(sum(values) / len(values)) if values else np.nan if records else None

    minutes_sum = sum_record_value(records, "minutes")
    out[f"{label}_minutes_share"] = safe_div(minutes_sum, 240.0)
    out[f"{label}_minutes_weighted_points_per36"] = weighted_record_average(records, "points_per36", "minutes")
    out[f"{label}_minutes_weighted_rebounds_per36"] = weighted_record_average(records, "rebounds_per36", "minutes")
    out[f"{label}_minutes_weighted_assists_per36"] = weighted_record_average(records, "assists_per36", "minutes")
    out[f"{label}_minutes_weighted_turnovers_per36"] = weighted_record_average(records, "turnovers_per36", "minutes")
    out[f"{label}_minutes_weighted_plus_minus"] = weighted_record_average(records, "plus_minus", "minutes")
    out[f"{label}_minutes_weighted_usage_proxy"] = weighted_record_average(records, "usage_proxy", "minutes")

    out[f"{label}_usage_weighted_points_per36"] = weighted_record_average(
        records,
        "points_per36",
        weight_func=player_usage_minutes_weight,
    )
    out[f"{label}_usage_weighted_assists_per36"] = weighted_record_average(
        records,
        "assists_per36",
        weight_func=player_usage_minutes_weight,
    )
    out[f"{label}_usage_weighted_turnovers_per36"] = weighted_record_average(
        records,
        "turnovers_per36",
        weight_func=player_usage_minutes_weight,
    )
    out[f"{label}_usage_weighted_plus_minus"] = weighted_record_average(
        records,
        "plus_minus",
        weight_func=player_usage_minutes_weight,
    )

    fgm = sum_record_value(records, "field_goals_made_field_goals_attempted_made")
    fga = sum_record_value(records, "field_goals_made_field_goals_attempted_attempted")
    tpm = sum_record_value(records, "three_point_field_goals_made_three_point_field_goals_attempted_made")
    tpa = sum_record_value(records, "three_point_field_goals_made_three_point_field_goals_attempted_attempted")
    ftm = sum_record_value(records, "free_throws_made_free_throws_attempted_made")
    fta = sum_record_value(records, "free_throws_made_free_throws_attempted_attempted")
    out[f"{label}_fga_weighted_efg_pct"] = safe_div((fgm or 0.0) + (0.5 * (tpm or 0.0)), fga)
    out[f"{label}_3pa_weighted_three_pct"] = safe_div(tpm, tpa)
    out[f"{label}_fta_weighted_ft_pct"] = safe_div(ftm, fta)

    minutes_values = [fast_float(player.get("minutes")) for player in records]
    minutes_values = [value for value in minutes_values if value is not None]
    out[f"{label}_minutes_volatility"] = float(np.std(minutes_values)) if len(minutes_values) >= 2 else 0.0 if records else None
    out[f"{label}_dnp_games_last10"] = sum_record_value(records, "dnp_games")
    out[f"{label}_dnp_rate_last10"] = weighted_record_average(records, "dnp_rate", "minutes")
    availability_risk_minutes = 0.0
    availability_risk_usage = 0.0
    availability_risk_points = 0.0
    has_availability = False
    for player in records:
        dnp_rate = fast_float(player.get("dnp_rate"))
        if dnp_rate is None:
            continue
        has_availability = True
        minutes = fast_float(player.get("minutes")) or 0.0
        usage = fast_float(player.get("usage_proxy")) or 0.0
        points = fast_float(player.get("points")) or 0.0
        availability_risk_minutes += minutes * dnp_rate
        availability_risk_usage += usage * minutes * dnp_rate
        availability_risk_points += points * dnp_rate
    out[f"{label}_availability_risk_expected_minutes"] = availability_risk_minutes if has_availability else None
    out[f"{label}_availability_risk_expected_usage"] = availability_risk_usage if has_availability else None
    out[f"{label}_availability_risk_expected_points"] = availability_risk_points if has_availability else None


def aggregate_players(player_logs: pd.DataFrame, team_id: str, game_date: str) -> dict[str, Any]:
    rows = player_logs[
        (player_logs["team_id"].astype(str) == str(team_id))
        & (player_logs["game_date"].astype(str) < str(game_date))
    ].copy()
    rows = current_season_rows(rows, game_date)
    if rows.empty:
        return {
            "logged_players": 0,
            "rotation_players_last10": 0,
        }

    rows = rows.sort_values(["athlete_id", "game_date"])
    recent = rows.groupby("athlete_id", group_keys=False).tail(10)
    recent = recent.copy()
    if "did_not_play" in recent.columns:
        recent["_did_not_play_num"] = recent["did_not_play"].apply(truthy_value).astype(float)
    agg_map = {
        "player": ("player", "last"),
        "games": ("event_id", "nunique"),
    }
    for stat in PLAYER_MEAN_STATS:
        if stat in recent.columns:
            agg_map[stat] = (stat, "mean")
    if "_did_not_play_num" in recent.columns:
        agg_map["dnp_games"] = ("_did_not_play_num", "sum")
        agg_map["dnp_rate"] = ("_did_not_play_num", "mean")
    player_means = recent.groupby("athlete_id").agg(**agg_map).reset_index()
    if "minutes" in player_means.columns:
        player_means["minutes"] = pd.to_numeric(player_means["minutes"], errors="coerce").fillna(0)
    else:
        player_means["minutes"] = 0.0
    if "points" in player_means.columns:
        player_means["points"] = pd.to_numeric(player_means["points"], errors="coerce").fillna(0)
    else:
        player_means["points"] = 0.0
    rotation = player_means[player_means["minutes"] >= 8].copy()
    minute_ranked = rotation.sort_values("minutes", ascending=False)
    scoring_ranked = rotation.sort_values("points", ascending=False)

    out: dict[str, Any] = {
        "logged_players": int(player_means["athlete_id"].nunique()),
        "rotation_players_last10": int(len(rotation)),
    }
    for label, frame in [("top5", minute_ranked.head(5)), ("top8", minute_ranked.head(8))]:
        add_player_group_features(out, label, frame)
    minutes_values = pd.to_numeric(rotation.get("minutes"), errors="coerce").dropna() if not rotation.empty else pd.Series(dtype=float)
    out["rotation_minutes_volatility"] = float(minutes_values.std(ddof=0)) if len(minutes_values) >= 2 else 0.0 if len(rotation) else None

    for rank, (_, player) in enumerate(scoring_ranked.head(3).iterrows(), start=1):
        out[f"scorer{rank}_ppg"] = numeric(player.get("points"))
        out[f"scorer{rank}_mpg"] = numeric(player.get("minutes"))
        out[f"scorer{rank}_usage"] = numeric(player.get("usage_proxy"))
    return out


def prepare_feature_context(team_boxscores: pd.DataFrame, player_logs: pd.DataFrame) -> dict[str, Any]:
    teams = team_boxscores.copy()
    if not teams.empty:
        teams["_team_id_str"] = teams["team_id"].astype(str)
        teams["_opponent_team_id_str"] = teams["opponent_team_id"].astype(str) if "opponent_team_id" in teams.columns else ""
        teams["_game_date_dt"] = pd.to_datetime(teams["game_date"], errors="coerce")
        teams["_season_key"] = teams["_game_date_dt"].apply(nba_season_from_game_date)
        if "season_type" in teams.columns:
            teams["_season_type_num"] = pd.to_numeric(teams["season_type"], errors="coerce")
        else:
            teams["_season_type_num"] = np.nan
        for col in ["won", *TEAM_AVG_STATS, *TEAM_TOTAL_COLUMNS]:
            if col in teams.columns:
                teams[col] = pd.to_numeric(teams[col], errors="coerce")
        teams = teams.sort_values(["_team_id_str", "_game_date_dt", "event_id"]).reset_index(drop=True)

    players = player_logs.copy()
    if not players.empty:
        players["_team_id_str"] = players["team_id"].astype(str)
        players["_game_date_dt"] = pd.to_datetime(players["game_date"], errors="coerce")
        players["_season_key"] = players["_game_date_dt"].apply(nba_season_from_game_date)
        for col in ["minutes", "points", *PLAYER_MEAN_STATS, *PLAYER_SHOT_COLUMNS]:
            if col in players.columns:
                players[col] = pd.to_numeric(players[col], errors="coerce")
        players = players.sort_values(["_team_id_str", "_game_date_dt", "event_id", "athlete_id"]).reset_index(drop=True)

    team_by_id = {
        str(team_id): frame.sort_values(["_game_date_dt", "event_id"]).reset_index(drop=True)
        for team_id, frame in teams.groupby("_team_id_str", sort=False)
    } if not teams.empty else {}
    team_by_pair = {
        (str(team_id), str(opponent_id)): frame.sort_values(["_game_date_dt", "event_id"]).reset_index(drop=True)
        for (team_id, opponent_id), frame in teams.groupby(["_team_id_str", "_opponent_team_id_str"], sort=False)
    } if not teams.empty and "_opponent_team_id_str" in teams.columns else {}
    players_by_team = {
        str(team_id): frame.sort_values(["_game_date_dt", "event_id", "athlete_id"]).reset_index(drop=True)
        for team_id, frame in players.groupby("_team_id_str", sort=False)
    } if not players.empty else {}
    return {
        "team_by_id": team_by_id,
        "team_by_pair": team_by_pair,
        "players_by_team": players_by_team,
    }


def indexed_rows_before(frame: pd.DataFrame | None, game_date: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    target_date = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(target_date):
        return frame.iloc[0:0]
    dates = frame["_game_date_dt"].to_numpy()
    try:
        cutoff = int(np.searchsorted(dates, np.datetime64(target_date), side="left"))
    except TypeError:
        cutoff = int((frame["_game_date_dt"] < target_date).sum())
    return frame.iloc[:cutoff]


def aggregate_team_indexed(
    context: dict[str, Any],
    team_id: str,
    game_date: str,
    split: str,
    opponent_id: str | None = None,
) -> dict[str, Any]:
    if opponent_id is None:
        frame = context["team_by_id"].get(str(team_id))
    else:
        frame = context["team_by_pair"].get((str(team_id), str(opponent_id)))
    rows = indexed_rows_before(frame, game_date)
    target_season = nba_season_from_game_date(game_date)
    if split == "season":
        rows = rows[rows["_season_key"] == target_season]
    elif split == "regular":
        rows = rows[(rows["_season_key"] == target_season) & (rows["_season_type_num"] == 2)]
    elif split == "playoffs":
        rows = rows[(rows["_season_key"] == target_season) & (rows["_season_type_num"] == 3)]
    elif split.startswith("last"):
        rows = rows[rows["_season_key"] == target_season].tail(int(split.replace("last", "")))

    return team_aggregate_from_rows(rows, game_date)


def aggregate_players_indexed(context: dict[str, Any], team_id: str, game_date: str) -> dict[str, Any]:
    rows = indexed_rows_before(context["players_by_team"].get(str(team_id)), game_date)
    target_season = nba_season_from_game_date(game_date)
    rows = rows[rows["_season_key"] == target_season] if not rows.empty else rows
    if rows.empty:
        return {
            "logged_players": 0,
            "rotation_players_last10": 0,
        }

    rows = rows.sort_values(["athlete_id", "_game_date_dt", "event_id"])
    recent = rows.groupby("athlete_id", group_keys=False).tail(10)
    recent = recent.copy()
    if "did_not_play" in recent.columns:
        recent["_did_not_play_num"] = recent["did_not_play"].apply(truthy_value).astype(float)
    agg_map = {
        "player": ("player", "last"),
        "games": ("event_id", "nunique"),
    }
    for stat in PLAYER_MEAN_STATS:
        if stat in recent.columns:
            agg_map[stat] = (stat, "mean")
    if "_did_not_play_num" in recent.columns:
        agg_map["dnp_games"] = ("_did_not_play_num", "sum")
        agg_map["dnp_rate"] = ("_did_not_play_num", "mean")
    player_means = recent.groupby("athlete_id").agg(**agg_map).reset_index()
    if "minutes" in player_means.columns:
        player_means["minutes"] = pd.to_numeric(player_means["minutes"], errors="coerce").fillna(0)
    else:
        player_means["minutes"] = 0.0
    if "points" in player_means.columns:
        player_means["points"] = pd.to_numeric(player_means["points"], errors="coerce").fillna(0)
    else:
        player_means["points"] = 0.0
    rotation = player_means[player_means["minutes"] >= 8].copy()
    minute_ranked = rotation.sort_values("minutes", ascending=False)
    scoring_ranked = rotation.sort_values("points", ascending=False)

    out: dict[str, Any] = {
        "logged_players": int(player_means["athlete_id"].nunique()),
        "rotation_players_last10": int(len(rotation)),
    }
    for label, frame in [("top5", minute_ranked.head(5)), ("top8", minute_ranked.head(8))]:
        add_player_group_features(out, label, frame)
    minutes_values = pd.to_numeric(rotation.get("minutes"), errors="coerce").dropna() if not rotation.empty else pd.Series(dtype=float)
    out["rotation_minutes_volatility"] = float(minutes_values.std(ddof=0)) if len(minutes_values) >= 2 else 0.0 if len(rotation) else None

    for rank, (_, player) in enumerate(scoring_ranked.head(3).iterrows(), start=1):
        out[f"scorer{rank}_ppg"] = numeric(player.get("points"))
        out[f"scorer{rank}_mpg"] = numeric(player.get("minutes"))
        out[f"scorer{rank}_usage"] = numeric(player.get("usage_proxy"))
    return out


def player_summary_from_recent_rows(player_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not player_rows:
        return {
            "logged_players": 0,
            "rotation_players_last10": 0,
        }

    by_player: dict[Any, list[dict[str, Any]]] = {}
    for row in player_rows:
        athlete_id = row.get("athlete_id")
        if athlete_id is None:
            continue
        by_player.setdefault(athlete_id, []).append(row)

    mean_rows: list[dict[str, Any]] = []
    for athlete_id, rows in by_player.items():
        event_ids = {row.get("event_id") for row in rows if row.get("event_id") is not None}
        item: dict[str, Any] = {
            "athlete_id": athlete_id,
            "player": rows[-1].get("player"),
            "games": len(event_ids),
        }
        for stat in PLAYER_MEAN_STATS:
            values = [fast_float(row.get(stat)) for row in rows]
            values = [value for value in values if value is not None]
            item[stat] = float(np.mean(values)) if values else np.nan
        dnp_values = [truthy_value(row.get("did_not_play")) for row in rows if row.get("did_not_play") is not None]
        item["dnp_games"] = float(sum(1 for value in dnp_values if value))
        item["dnp_rate"] = float(item["dnp_games"] / len(dnp_values)) if dnp_values else np.nan
        item["minutes"] = fast_float(item.get("minutes")) or 0.0
        item["points"] = fast_float(item.get("points")) or 0.0
        mean_rows.append(item)

    rotation = [row for row in mean_rows if float(row.get("minutes") or 0.0) >= 8]
    minute_ranked = sorted(rotation, key=lambda row: float(row.get("minutes") or 0.0), reverse=True)
    scoring_ranked = sorted(rotation, key=lambda row: float(row.get("points") or 0.0), reverse=True)

    out: dict[str, Any] = {
        "logged_players": int(len(mean_rows)),
        "rotation_players_last10": int(len(rotation)),
    }
    for label, frame in [("top5", minute_ranked[:5]), ("top8", minute_ranked[:8])]:
        add_player_group_features(out, label, frame)
    minutes_values = [fast_float(player.get("minutes")) for player in rotation]
    minutes_values = [value for value in minutes_values if value is not None]
    out["rotation_minutes_volatility"] = float(np.std(minutes_values)) if len(minutes_values) >= 2 else 0.0 if rotation else None

    for rank, player in enumerate(scoring_ranked[:3], start=1):
        out[f"scorer{rank}_ppg"] = fast_float(player.get("points"))
        out[f"scorer{rank}_mpg"] = fast_float(player.get("minutes"))
        out[f"scorer{rank}_usage"] = fast_float(player.get("usage_proxy"))
    return out


def needed_team_dates(completed: pd.DataFrame) -> dict[str, set[str]]:
    needed: dict[str, set[str]] = {}
    for _, game in completed.iterrows():
        date = str(game["game_date"])
        needed.setdefault(str(game["home_team_id"]), set()).add(date)
        needed.setdefault(str(game["away_team_id"]), set()).add(date)
    return needed


def precompute_player_cache(context: dict[str, Any], completed: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for team_id, dates in needed_team_dates(completed).items():
        team_rows = context["players_by_team"].get(str(team_id))
        sorted_dates = sorted(dates)
        if team_rows is None or team_rows.empty:
            for date in sorted_dates:
                cache[(str(team_id), str(date))] = {"logged_players": 0, "rotation_players_last10": 0}
            continue

        rows = team_rows.sort_values(["_game_date_dt", "event_id", "athlete_id"]).reset_index(drop=True)
        records = rows.to_dict("records")
        position = 0
        active_season: int | None = None
        recent_by_player: dict[Any, deque[dict[str, Any]]] = {}
        for date in sorted_dates:
            target_date = pd.to_datetime(date, errors="coerce")
            target_season = nba_season_from_game_date(date)
            if active_season != target_season:
                recent_by_player = {}
                active_season = target_season
            while position < len(records):
                row = records[position]
                row_date = row.get("_game_date_dt")
                if pd.isna(target_date) or pd.isna(row_date) or row_date >= target_date:
                    break
                if row.get("_season_key") == target_season:
                    athlete_id = row.get("athlete_id")
                    recent_by_player.setdefault(athlete_id, deque(maxlen=10)).append(row)
                position += 1
            recent_rows = [row for rows_for_player in recent_by_player.values() for row in rows_for_player]
            cache[(str(team_id), str(date))] = player_summary_from_recent_rows(recent_rows)
    return cache


def cached_aggregate_team(
    cache: dict[tuple[str, str, str, str], dict[str, Any]] | None,
    team_boxscores: pd.DataFrame,
    team_id: str,
    game_date: str,
    split: str,
    opponent_id: str | None = None,
) -> dict[str, Any]:
    if cache is None:
        return aggregate_team(team_boxscores, team_id, game_date, split, opponent_id=opponent_id)
    key = (str(team_id), str(game_date), split, str(opponent_id or ""))
    if key not in cache:
        cache[key] = aggregate_team(team_boxscores, team_id, game_date, split, opponent_id=opponent_id)
    return cache[key]


def cached_schedule_context(
    cache: dict[tuple[str, str, str, str], dict[str, Any]] | None,
    team_boxscores: pd.DataFrame,
    team_id: str,
    game_date: str,
    current_home_away: str,
) -> dict[str, Any]:
    if cache is None:
        return team_schedule_context(team_boxscores, team_id, game_date, current_home_away)
    key = (str(team_id), str(game_date), f"schedule_{current_home_away}", "")
    if key not in cache:
        cache[key] = team_schedule_context(team_boxscores, team_id, game_date, current_home_away)
    return cache[key]


def cached_aggregate_players(
    cache: dict[tuple[str, str], dict[str, Any]] | None,
    player_logs: pd.DataFrame,
    team_id: str,
    game_date: str,
) -> dict[str, Any]:
    if cache is None:
        return aggregate_players(player_logs, team_id, game_date)
    key = (str(team_id), str(game_date))
    if key not in cache:
        cache[key] = aggregate_players(player_logs, team_id, game_date)
    return cache[key]


def cached_aggregate_team_indexed(
    cache: dict[tuple[str, str, str, str], dict[str, Any]],
    context: dict[str, Any],
    team_id: str,
    game_date: str,
    split: str,
    opponent_id: str | None = None,
) -> dict[str, Any]:
    key = (str(team_id), str(game_date), split, str(opponent_id or ""))
    if key not in cache:
        cache[key] = aggregate_team_indexed(context, team_id, game_date, split, opponent_id=opponent_id)
    return cache[key]


def cached_schedule_context_indexed(
    cache: dict[tuple[str, str, str, str], dict[str, Any]],
    context: dict[str, Any],
    team_id: str,
    game_date: str,
    current_home_away: str,
) -> dict[str, Any]:
    key = (str(team_id), str(game_date), f"schedule_{current_home_away}", "")
    if key not in cache:
        cache[key] = team_schedule_context_indexed(context, team_id, game_date, current_home_away)
    return cache[key]


def cached_aggregate_players_indexed(
    cache: dict[tuple[str, str], dict[str, Any]],
    context: dict[str, Any],
    team_id: str,
    game_date: str,
) -> dict[str, Any]:
    key = (str(team_id), str(game_date))
    if key not in cache:
        cache[key] = aggregate_players_indexed(context, team_id, game_date)
    return cache[key]


def build_feature_row(
    game: pd.Series | dict[str, Any],
    team_boxscores: pd.DataFrame,
    player_logs: pd.DataFrame,
    include_target: bool,
    team_cache: dict[tuple[str, str, str, str], dict[str, Any]] | None = None,
    player_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    strength_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    game_dict = dict(game)
    home_id = str(game_dict["home_team_id"])
    away_id = str(game_dict["away_team_id"])
    game_date = str(game_dict["game_date"])
    season_type = infer_season_type(game_dict)
    round_type = infer_round_type(game_dict, season_type)
    row: dict[str, Any] = {
        "event_id": game_dict.get("event_id"),
        "game_datetime": game_dict.get("game_datetime"),
        "game_date": game_date,
        "name": game_dict.get("name") or game_dict.get("event_name"),
        "short_name": game_dict.get("short_name"),
        "season_type": season_type,
        "round_type": round_type,
        "neutral_site": int(bool(game_dict.get("neutral_site"))) if game_dict.get("neutral_site") is not None else 0,
        "is_playoff": int(season_type == 3 or is_playoff_round(round_type)),
        "is_finals": is_finals_round(round_type),
        "home_team_id": home_id,
        "home_team": game_dict.get("home_team"),
        "away_team_id": away_id,
        "away_team": game_dict.get("away_team"),
    }

    home_splits: dict[str, dict[str, Any]] = {}
    away_splits: dict[str, dict[str, Any]] = {}
    for split in ["all", "regular", "playoffs", "last3", "last5", "last10", "last20"]:
        home = cached_aggregate_team(team_cache, team_boxscores, home_id, game_date, split)
        away = cached_aggregate_team(team_cache, team_boxscores, away_id, game_date, split)
        home_splits[split] = home
        away_splits[split] = away
        row.update(prefix(f"home_{split}", home))
        row.update(prefix(f"away_{split}", away))
        row.update(edge(f"edge_{split}", home, away))

    home_schedule = cached_schedule_context(team_cache, team_boxscores, home_id, game_date, "home")
    away_schedule = cached_schedule_context(team_cache, team_boxscores, away_id, game_date, "away")
    row.update(prefix("home_schedule", home_schedule))
    row.update(prefix("away_schedule", away_schedule))
    row.update(edge("edge_schedule", home_schedule, away_schedule))

    home_h2h = cached_aggregate_team(team_cache, team_boxscores, home_id, game_date, "season", opponent_id=away_id)
    away_h2h = cached_aggregate_team(team_cache, team_boxscores, away_id, game_date, "season", opponent_id=home_id)
    home_splits["h2h"] = home_h2h
    away_splits["h2h"] = away_h2h
    row.update(prefix("home_h2h", home_h2h))
    row.update(prefix("away_h2h", away_h2h))
    row.update(edge("edge_h2h", home_h2h, away_h2h))
    row.update(style_matchup_features(home_splits, away_splits))

    home_players = cached_aggregate_players(player_cache, player_logs, home_id, game_date)
    away_players = cached_aggregate_players(player_cache, player_logs, away_id, game_date)
    row.update(prefix("home_players", home_players))
    row.update(prefix("away_players", away_players))
    row.update(edge("edge_players", home_players, away_players))
    row.update(strength_features_for_game(game_dict, strength_context))

    if include_target:
        row["target_home_win"] = int(game_dict["home_win"])
        row["target_home_margin"] = numeric(game_dict.get("home_margin"))
    return row


def build_feature_row_indexed(
    game: pd.Series | dict[str, Any],
    context: dict[str, Any],
    include_target: bool,
    team_cache: dict[tuple[str, str, str, str], dict[str, Any]],
    player_cache: dict[tuple[str, str], dict[str, Any]],
    strength_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    game_dict = dict(game)
    home_id = str(game_dict["home_team_id"])
    away_id = str(game_dict["away_team_id"])
    game_date = str(game_dict["game_date"])
    season_type = infer_season_type(game_dict)
    round_type = infer_round_type(game_dict, season_type)
    row: dict[str, Any] = {
        "event_id": game_dict.get("event_id"),
        "game_datetime": game_dict.get("game_datetime"),
        "game_date": game_date,
        "name": game_dict.get("name") or game_dict.get("event_name"),
        "short_name": game_dict.get("short_name"),
        "season_type": season_type,
        "round_type": round_type,
        "neutral_site": int(bool(game_dict.get("neutral_site"))) if game_dict.get("neutral_site") is not None else 0,
        "is_playoff": int(season_type == 3 or is_playoff_round(round_type)),
        "is_finals": is_finals_round(round_type),
        "home_team_id": home_id,
        "home_team": game_dict.get("home_team"),
        "away_team_id": away_id,
        "away_team": game_dict.get("away_team"),
    }

    home_splits: dict[str, dict[str, Any]] = {}
    away_splits: dict[str, dict[str, Any]] = {}
    for split in ["all", "regular", "playoffs", "last3", "last5", "last10", "last20"]:
        home = cached_aggregate_team_indexed(team_cache, context, home_id, game_date, split)
        away = cached_aggregate_team_indexed(team_cache, context, away_id, game_date, split)
        home_splits[split] = home
        away_splits[split] = away
        row.update(prefix(f"home_{split}", home))
        row.update(prefix(f"away_{split}", away))
        row.update(edge(f"edge_{split}", home, away))

    home_schedule = cached_schedule_context_indexed(team_cache, context, home_id, game_date, "home")
    away_schedule = cached_schedule_context_indexed(team_cache, context, away_id, game_date, "away")
    row.update(prefix("home_schedule", home_schedule))
    row.update(prefix("away_schedule", away_schedule))
    row.update(edge("edge_schedule", home_schedule, away_schedule))

    home_h2h = cached_aggregate_team_indexed(team_cache, context, home_id, game_date, "season", opponent_id=away_id)
    away_h2h = cached_aggregate_team_indexed(team_cache, context, away_id, game_date, "season", opponent_id=home_id)
    home_splits["h2h"] = home_h2h
    away_splits["h2h"] = away_h2h
    row.update(prefix("home_h2h", home_h2h))
    row.update(prefix("away_h2h", away_h2h))
    row.update(edge("edge_h2h", home_h2h, away_h2h))
    row.update(style_matchup_features(home_splits, away_splits))

    home_players = cached_aggregate_players_indexed(player_cache, context, home_id, game_date)
    away_players = cached_aggregate_players_indexed(player_cache, context, away_id, game_date)
    row.update(prefix("home_players", home_players))
    row.update(prefix("away_players", away_players))
    row.update(edge("edge_players", home_players, away_players))
    row.update(strength_features_for_game(game_dict, strength_context))

    if include_target:
        row["target_home_win"] = int(game_dict["home_win"])
        row["target_home_margin"] = numeric(game_dict.get("home_margin"))
    return row


def build_training_table(
    schedule: pd.DataFrame,
    team_boxscores: pd.DataFrame,
    player_logs: pd.DataFrame,
    min_prior_games: int,
    moneylines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    completed = schedule[
        (schedule["completed"] == True)
        & schedule["home_win"].notna()
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    ].copy()
    completed = completed.sort_values(["game_date", "event_id"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    context = prepare_feature_context(team_boxscores, player_logs)
    strength_context = prepare_strength_context(schedule, moneylines)
    team_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    player_cache = precompute_player_cache(context, completed)
    for _, game in completed.iterrows():
        home_prior = cached_aggregate_team_indexed(
            team_cache,
            context,
            str(game["home_team_id"]),
            str(game["game_date"]),
            "all",
        )["games"]
        away_prior = cached_aggregate_team_indexed(
            team_cache,
            context,
            str(game["away_team_id"]),
            str(game["game_date"]),
            "all",
        )["games"]
        if min(home_prior, away_prior) < min_prior_games:
            continue
        rows.append(
            build_feature_row_indexed(
                game,
                context,
                include_target=True,
                team_cache=team_cache,
                player_cache=player_cache,
                strength_context=strength_context,
            )
        )
        if len(rows) % 250 == 0:
            print(f"  feature rows built: {len(rows)}", flush=True)
    return pd.DataFrame(rows).sort_values(["game_date", "event_id"]).reset_index(drop=True)


def nba_season_from_game_date(value: Any) -> int | None:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return None
    return int(date.year + 1 if date.month >= 7 else date.year)


def playoff_teams_by_season(reference_df: pd.DataFrame) -> dict[int, set[str]]:
    required = {"game_date", "season_type", "home_team_id", "away_team_id"}
    if not required.issubset(reference_df.columns):
        return {}

    season_type = pd.to_numeric(reference_df["season_type"], errors="coerce")
    playoff_rows = reference_df[season_type.eq(3)].copy()
    if playoff_rows.empty:
        return {}

    playoff_dates = pd.to_datetime(playoff_rows["game_date"], errors="coerce")
    playoff_rows["_season_key"] = playoff_dates.apply(nba_season_from_game_date)
    out: dict[int, set[str]] = {}
    for _, row in playoff_rows.dropna(subset=["_season_key"]).iterrows():
        season_key = int(row["_season_key"])
        teams = out.setdefault(season_key, set())
        teams.add(str(row.get("home_team_id")))
        teams.add(str(row.get("away_team_id")))
    return out


def row_team_win_pct(row: pd.Series, side: str) -> float | None:
    for col in [f"{side}_regular_win_pct", f"{side}_all_win_pct"]:
        value = numeric(row.get(col))
        if value is not None:
            return value
    return None


def late_regular_rest_mask(
    df: pd.DataFrame,
    start_month: int = 4,
    start_day: int = 1,
    playoff_teams_only: bool = True,
    playoff_reference_df: pd.DataFrame | None = None,
    tank_win_pct_threshold: float | None = 0.40,
    include_tank_risk: bool = True,
) -> pd.Series:
    if df.empty or "game_date" not in df.columns or "season_type" not in df.columns:
        return pd.Series(False, index=df.index)

    dates = pd.to_datetime(df["game_date"], errors="coerce")
    season_type = pd.to_numeric(df["season_type"], errors="coerce")
    after_cutoff = (
        ((dates.dt.month == start_month) & (dates.dt.day >= start_day))
        | ((dates.dt.month > start_month) & (dates.dt.month < 7))
    )
    mask = season_type.eq(2) & after_cutoff.fillna(False)

    if not playoff_teams_only or not {"home_team_id", "away_team_id"}.issubset(df.columns):
        return mask.fillna(False)

    season_keys = dates.apply(nba_season_from_game_date)
    playoff_source = playoff_reference_df if playoff_reference_df is not None else df
    playoff_teams = playoff_teams_by_season(playoff_source)
    if not playoff_teams:
        return pd.Series(False, index=df.index)

    threshold = numeric(tank_win_pct_threshold)
    tank_filter_on = include_tank_risk and threshold is not None and threshold > 0
    incentive_risk_game = []
    for index, row in df.iterrows():
        season_key = season_keys.loc[index]
        if season_key is None or pd.isna(season_key):
            incentive_risk_game.append(False)
            continue
        season_playoff_teams = playoff_teams.get(int(season_key), set())
        home_id = str(row.get("home_team_id"))
        away_id = str(row.get("away_team_id"))
        playoff_rest_risk = home_id in season_playoff_teams or away_id in season_playoff_teams
        tank_risk = False
        if tank_filter_on:
            for side, team_id in [("home", home_id), ("away", away_id)]:
                if team_id in season_playoff_teams:
                    continue
                win_pct = row_team_win_pct(row, side)
                if win_pct is not None and win_pct <= float(threshold):
                    tank_risk = True
                    break
        incentive_risk_game.append(playoff_rest_risk or tank_risk)

    return (mask & pd.Series(incentive_risk_game, index=df.index)).fillna(False)


def filter_late_regular_training_rows(
    df: pd.DataFrame,
    start_month: int = 4,
    start_day: int = 1,
    playoff_teams_only: bool = True,
    playoff_reference_df: pd.DataFrame | None = None,
    tank_win_pct_threshold: float | None = 0.40,
    include_tank_risk: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = late_regular_rest_mask(
        df,
        start_month=start_month,
        start_day=start_day,
        playoff_teams_only=playoff_teams_only,
        playoff_reference_df=playoff_reference_df,
        tank_win_pct_threshold=tank_win_pct_threshold,
        include_tank_risk=include_tank_risk,
    )
    filtered = df[~mask].copy().reset_index(drop=True)
    removed = df[mask].copy()
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "rule": "drop late regular-season incentive-risk games involving playoff rest or tank risk",
        "start_month": int(start_month),
        "start_day": int(start_day),
        "playoff_teams_only": bool(playoff_teams_only),
        "include_tank_risk": bool(include_tank_risk),
        "tank_win_pct_threshold": numeric(tank_win_pct_threshold),
        "rows_before": int(len(df)),
        "rows_removed": int(len(removed)),
        "rows_after": int(len(filtered)),
    }
    if not removed.empty:
        diagnostics["removed_date_range"] = {
            "start": str(removed["game_date"].min()),
            "end": str(removed["game_date"].max()),
        }
        diagnostics["removed_by_season"] = {
            str(season): int(count)
            for season, count in removed.assign(_season=removed["game_date"].apply(nba_season_from_game_date))["_season"]
            .value_counts()
            .sort_index()
            .items()
        }
    return filtered, diagnostics


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if col in IDENTITY_COLUMNS:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            columns.append(col)
    return columns


def selected_feature_columns(df: pd.DataFrame, mode: str) -> list[str]:
    columns = numeric_feature_columns(df)
    if mode == "full":
        return columns

    exact = {"season_type", "neutral_site", "is_playoff", "is_finals"}
    if mode == "wide":
        return [col for col in columns if col in exact or col.startswith(("home_", "away_", "edge_"))]
    if mode == "edge":
        return [col for col in columns if col in exact or col.startswith("edge_")]

    compact: list[str] = []
    for col in columns:
        if col in exact or col.startswith("edge_"):
            compact.append(col)
            continue
        if re.match(r"^(home|away)_(all|regular|playoffs|last3|last5|last10|last20|h2h)_(games|win_pct|rest_days)$", col):
            compact.append(col)
            continue
        if col.startswith(("home_players_", "away_players_")) and any(
            marker in col
            for marker in [
                "logged_players",
                "rotation_players",
                "top5_sum_minutes",
                "top5_sum_points",
                "top5_sum_usage",
                "top8_sum_minutes",
                "top8_sum_points",
                "top8_sum_usage",
                "scorer",
            ]
        ):
            compact.append(col)
    return compact


def is_market_derived_feature(column: str) -> bool:
    text = column.lower()
    if any(term in text for term in ("market", "odds", "implied", "no_vig")):
        return True
    return bool(re.search(r"(^|_)spread(_|$)", text))


def filter_model_feature_columns(
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    if not bool(getattr(args, "no_market_features", False)):
        return feature_columns, {
            "enabled": False,
            "removed_feature_count": 0,
            "removed_features": [],
        }

    removed = [col for col in feature_columns if is_market_derived_feature(col)]
    kept = [col for col in feature_columns if col not in set(removed)]
    return kept, {
        "enabled": True,
        "rule": "remove market, odds, implied probability, no-vig, and spread-derived columns from model inputs",
        "removed_feature_count": int(len(removed)),
        "removed_features": removed,
    }


def feature_group(column: str) -> str:
    if column in {"season_type", "neutral_site", "is_playoff", "is_finals"}:
        return "context"
    if "_strength_" in column:
        return "strength"
    if "_series_" in column:
        return "series"
    if "_schedule_" in column:
        return "schedule"
    if "_style_" in column:
        return "style_matchup"
    if "_players_" in column:
        return "players"
    if "_h2h_" in column:
        return "h2h"
    if re.match(r"^(home|away|edge)_(last3|last5|last10|last20)_", column):
        return "recent_team"
    if re.match(r"^(home|away|edge)_(regular|playoffs|all)_", column):
        return "team_profile"
    return "other"


def feature_score(train_df: pd.DataFrame, column: str, target: pd.Series, max_missing_fraction: float) -> float:
    values = pd.to_numeric(train_df[column], errors="coerce")
    missing_fraction = float(values.isna().mean()) if len(values) else 1.0
    if missing_fraction > max_missing_fraction:
        return 0.0
    mask = values.notna() & target.notna()
    if int(mask.sum()) < 40:
        return 0.0
    x = values[mask].astype(float).to_numpy()
    y = target[mask].astype(float).to_numpy()
    if float(np.nanstd(x)) < 1e-9 or float(np.nanstd(y)) < 1e-9:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return abs(float(corr)) if np.isfinite(corr) else 0.0


def select_feature_subset(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    mode = getattr(args, "feature_select", "grouped")
    max_features = int(getattr(args, "max_features", 0) or 0)
    if mode == "none" or max_features <= 0 or len(feature_columns) <= max_features:
        return feature_columns, {
            "enabled": False,
            "mode": mode,
            "base_feature_count": int(len(feature_columns)),
            "selected_feature_count": int(len(feature_columns)),
        }

    target = pd.to_numeric(train_df["target_home_win"], errors="coerce")
    max_missing = float(getattr(args, "feature_max_missing", 0.45))
    exact_keep = [col for col in feature_columns if col in {"season_type", "neutral_site", "is_playoff", "is_finals"}]
    scored: list[dict[str, Any]] = []
    for col in feature_columns:
        if col in exact_keep:
            continue
        scored.append(
            {
                "column": col,
                "group": feature_group(col),
                "score": feature_score(train_df, col, target, max_missing),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in scored:
        grouped.setdefault(str(item["group"]), []).append(item)
    for items in grouped.values():
        items.sort(key=lambda item: (float(item["score"]), item["column"]), reverse=True)

    group_names = sorted(grouped)
    group_quota = int(getattr(args, "feature_group_quota", 0) or 0)
    if group_quota <= 0:
        group_quota = max(4, max_features // max(len(group_names), 1))

    selected: list[str] = list(exact_keep)
    selected_set = set(selected)
    selected_by_group: dict[str, int] = {}
    for group in group_names:
        group_selected = 0
        for item in grouped[group]:
            if len(selected) >= max_features:
                break
            if group_selected >= group_quota:
                break
            if float(item["score"]) <= 0 and group_selected >= 2:
                continue
            col = str(item["column"])
            if col not in selected_set:
                selected.append(col)
                selected_set.add(col)
                group_selected += 1
        selected_by_group[group] = group_selected

    overall = sorted(scored, key=lambda item: (float(item["score"]), item["column"]), reverse=True)
    for item in overall:
        if len(selected) >= max_features:
            break
        col = str(item["column"])
        if col not in selected_set and float(item["score"]) > 0:
            selected.append(col)
            selected_set.add(col)
            selected_by_group[str(item["group"])] = selected_by_group.get(str(item["group"]), 0) + 1

    top_scores = [
        {"column": str(item["column"]), "group": str(item["group"]), "score": float(item["score"])}
        for item in overall[:20]
    ]
    return selected, {
        "enabled": True,
        "mode": mode,
        "base_feature_count": int(len(feature_columns)),
        "selected_feature_count": int(len(selected)),
        "max_features": int(max_features),
        "max_missing_fraction": max_missing,
        "group_quota": int(group_quota),
        "selected_by_group": selected_by_group,
        "top_scores": top_scores,
    }


def scale_arrays(train_df: pd.DataFrame, val_df: pd.DataFrame, pred_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    train = train_df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    val = val_df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    pred = pred_df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    train = np.where(np.isfinite(train), train, med)
    val = np.where(np.isfinite(val), val, med)
    pred = np.where(np.isfinite(pred), pred, med)

    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    return {
        "train": (train - mean) / std,
        "val": (val - mean) / std,
        "pred": (pred - mean) / std,
        "median": med,
        "mean": mean,
        "std": std,
    }


def transform_with_scaler(df: pd.DataFrame, feature_columns: list[str], scaler: dict[str, Any]) -> np.ndarray:
    x = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    med = np.asarray(scaler["median"], dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    x = np.where(np.isfinite(x), x, med)
    return (x - mean) / std


def sample_weights(df: pd.DataFrame, half_life_days: float, playoff_multiplier: float, finals_multiplier: float) -> np.ndarray:
    dates = pd.to_datetime(df["game_date"])
    anchor = dates.max()
    age_days = (anchor - dates).dt.days.astype(float).to_numpy()
    weights = np.exp(-age_days / half_life_days)
    weights *= np.where(pd.to_numeric(df.get("season_type"), errors="coerce").to_numpy() == 3, playoff_multiplier, 1.0)
    if "is_finals" in df.columns:
        weights *= np.where(pd.to_numeric(df["is_finals"], errors="coerce").fillna(0).to_numpy() == 1, finals_multiplier, 1.0)
    return weights.astype(np.float32)


def probability_distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {}
    quantiles = np.quantile(values, [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def binary_auc_score(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.float32)
    p = np.asarray(probabilities, dtype=np.float32)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    unique_values, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    if len(unique_values) < len(p):
        rank_sums = np.bincount(inverse, weights=ranks)
        avg_ranks = rank_sums / counts
        ranks = avg_ranks[inverse]
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def weighted_target_mean(y: np.ndarray, weights: np.ndarray) -> float | None:
    if len(y) == 0 or len(weights) == 0:
        return None
    denominator = float(np.sum(weights))
    if denominator <= 1e-12:
        return None
    return float(np.sum(y * weights) / denominator)


def fit_platt_calibration(
    logits: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray,
    min_auc: float,
) -> dict[str, Any]:
    raw_probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    auc = binary_auc_score(y_true, raw_probs)
    base = {
        "method": "platt",
        "applied": False,
        "validation_auc": auc,
        "min_auc": float(min_auc),
        "scale": 1.0,
        "bias": 0.0,
    }
    if auc is None:
        base["reason"] = "validation set has one class"
        return base
    if auc < float(min_auc):
        base["reason"] = "validation AUC below calibration threshold"
        return base

    z = torch.tensor(np.asarray(logits, dtype=np.float32), dtype=torch.float32)
    y = torch.tensor(np.asarray(y_true, dtype=np.float32), dtype=torch.float32)
    w = torch.tensor(np.asarray(weights, dtype=np.float32), dtype=torch.float32)
    raw_scale = torch.nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0), dtype=torch.float32))
    bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([raw_scale, bias], lr=0.5, max_iter=80, line_search_fn="strong_wolfe")
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scale = torch.nn.functional.softplus(raw_scale).clamp_min(1e-3)
        loss_values = loss_fn((scale * z) + bias, y)
        loss = (loss_values * w).sum() / w.sum().clamp_min(1e-6)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        scale_value = float(torch.nn.functional.softplus(raw_scale).detach().clamp_min(1e-3).item())
        bias_value = float(bias.detach().item())
    except RuntimeError as exc:
        base["reason"] = f"platt optimizer failed: {exc}"
        return base

    base.update(
        {
            "applied": True,
            "reason": "validation AUC met threshold",
            "scale": scale_value,
            "bias": bias_value,
        }
    )
    return base


def apply_probability_calibration(logits: np.ndarray, calibration: dict[str, Any] | None) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32)
    if calibration and calibration.get("applied") and calibration.get("method") == "platt":
        z = (float(calibration.get("scale", 1.0)) * z) + float(calibration.get("bias", 0.0))
    return 1.0 / (1.0 + np.exp(-z))


def predict_probabilities(
    model: nn.Module,
    x: np.ndarray,
    calibration: dict[str, Any] | None = None,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    probs = apply_probability_calibration(logits, calibration)
    return np.clip(probs, 0.02, 0.98)


def split_train_validation(df: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df["game_date"].astype(str).unique())
    val_dates = set(unique_dates[-max(1, int(math.ceil(len(unique_dates) * validation_fraction))) :])
    val_mask = df["game_date"].astype(str).isin(val_dates)
    if val_mask.sum() < 10 or (~val_mask).sum() < 30:
        cutoff = max(1, int(len(df) * (1 - validation_fraction)))
        val_mask = pd.Series(False, index=df.index)
        val_mask.iloc[cutoff:] = True
    return df[~val_mask].copy(), df[val_mask].copy()


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    set_seed(args.seed)
    scaled = scale_arrays(train_df, val_df, pred_df, feature_columns)
    y_train = train_df["target_home_win"].astype(float).to_numpy(dtype=np.float32)
    y_val = val_df["target_home_win"].astype(float).to_numpy(dtype=np.float32)
    w_train = sample_weights(train_df, args.half_life_days, args.playoff_multiplier, args.finals_multiplier)
    w_val = sample_weights(val_df, args.half_life_days, args.playoff_multiplier, args.finals_multiplier)

    device = torch.device("cpu")
    positive_rate = max(float(y_train.mean()), 1e-4)
    model = build_model(len(feature_columns), args).to(device)
    if bool(getattr(args, "init_output_bias", True)):
        initialize_output_bias(model, positive_rate)

    if getattr(args, "class_balance", "none") == "pos_weight":
        pos_weight = torch.tensor([(1 - positive_rate) / positive_rate], dtype=torch.float32, device=device)
    else:
        pos_weight = None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = TensorDataset(
        torch.tensor(scaled["train"], dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(w_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_x = torch.tensor(scaled["val"], dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=device)
    val_w = torch.tensor(w_val, dtype=torch.float32, device=device)

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad()
            raw_loss = loss_fn(model(xb), yb)
            loss = (raw_loss * wb).sum() / wb.sum().clamp_min(1e-6)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_raw = loss_fn(val_logits, val_y)
            val_loss = float(((val_raw * val_w).sum() / val_w.sum().clamp_min(1e-6)).detach().cpu().item())
            val_probs_epoch = torch.sigmoid(val_logits).detach().cpu().numpy()
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_dist = probability_distribution(val_probs_epoch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_prob_mean": val_dist.get("mean"),
                "validation_prob_std": val_dist.get("std"),
                "validation_prob_min": val_dist.get("min"),
                "validation_prob_max": val_dist.get("max"),
            }
        )

        if args.verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_prob_mean={val_dist.get('mean', float('nan')):.3f} "
                f"val_prob_std={val_dist.get('std', float('nan')):.3f} "
                f"range={val_dist.get('min', float('nan')):.3f}-{val_dist.get('max', float('nan')):.3f}",
                flush=True,
            )

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_logits = model(torch.tensor(scaled["train"], dtype=torch.float32, device=device)).cpu().numpy()
        val_logits = model(val_x).cpu().numpy()

    raw_train_probs = apply_probability_calibration(train_logits, None)
    raw_val_probs = apply_probability_calibration(val_logits, None)
    calibration_mode = getattr(args, "calibration", "platt")
    if calibration_mode == "platt":
        calibration = fit_platt_calibration(
            val_logits,
            y_val,
            w_val,
            min_auc=float(getattr(args, "calibration_min_auc", 0.52)),
        )
    else:
        calibration = {
            "method": calibration_mode,
            "applied": False,
            "reason": "calibration disabled",
            "scale": 1.0,
            "bias": 0.0,
            "validation_auc": binary_auc_score(y_val, raw_val_probs),
        }
    train_probs = apply_probability_calibration(train_logits, calibration)
    val_probs = apply_probability_calibration(val_logits, calibration)

    diag = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "positive_rate": float(positive_rate),
        "model_type": getattr(args, "model_type", "transformer"),
        "class_balance": getattr(args, "class_balance", "none"),
        "average_sample_weight": float(w_train.mean()) if len(w_train) else None,
        "weighted_train_mean": weighted_target_mean(y_train, w_train),
        "weighted_train_target_mean": weighted_target_mean(y_train, w_train),
        "raw_train_probability_distribution": probability_distribution(raw_train_probs),
        "raw_validation_probability_distribution": probability_distribution(raw_val_probs),
        "train_probability_distribution": probability_distribution(train_probs),
        "validation_probability_distribution": probability_distribution(val_probs),
        "validation_auc": binary_auc_score(y_val, val_probs),
        "raw_validation_auc": binary_auc_score(y_val, raw_val_probs),
        "validation_brier": float(np.mean((val_probs - y_val) ** 2)) if len(y_val) else None,
        "validation_accuracy_0_50": float(((val_probs >= 0.5) == y_val).mean()) if len(y_val) else None,
        "calibration": calibration,
        "history_tail": history[-10:],
    }
    scaler = {
        "median": scaled["median"].tolist(),
        "mean": scaled["mean"].tolist(),
        "std": scaled["std"].tolist(),
        "calibration": calibration,
    }
    return model, scaler, diag


def predict_probability(model: nn.Module, x: np.ndarray, calibration: dict[str, Any] | None = None) -> float:
    prob = float(predict_probabilities(model, x, calibration)[0])
    return min(max(prob, 0.02), 0.98)


def ensure_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> None:
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan


def feature_table_has_required_strength(df: pd.DataFrame) -> bool:
    return REQUIRED_STRENGTH_COLUMNS.issubset(set(df.columns))


def swapped_current_game(game: pd.Series | dict[str, Any]) -> dict[str, Any]:
    row = dict(game)
    for suffix in ["team_id", "team", "abbrev", "record", "score"]:
        home_key = f"home_{suffix}"
        away_key = f"away_{suffix}"
        if home_key in row or away_key in row:
            row[home_key], row[away_key] = row.get(away_key), row.get(home_key)
    if row.get("home_team") and row.get("away_team"):
        row["event_name"] = f"{row['away_team']} at {row['home_team']}"
    if row.get("home_abbrev") and row.get("away_abbrev"):
        row["short_name"] = f"{row['away_abbrev']} @ {row['home_abbrev']}"
    return row


def prediction_sanity_report(features: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    row = features.iloc[0]
    numeric_row = features[feature_columns].apply(pd.to_numeric, errors="coerce").iloc[0]
    missing_features = [col for col in feature_columns if pd.isna(numeric_row[col])]
    return {
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "season_type": coerce_int(row.get("season_type")),
        "round_type": row.get("round_type"),
        "is_playoff": coerce_int(row.get("is_playoff")),
        "is_finals": coerce_int(row.get("is_finals")),
        "home_prior_games": numeric(row.get("home_all_games")),
        "away_prior_games": numeric(row.get("away_all_games")),
        "home_playoff_prior_games": numeric(row.get("home_playoffs_games")),
        "away_playoff_prior_games": numeric(row.get("away_playoffs_games")),
        "feature_count": len(feature_columns),
        "feature_missing_count": len(missing_features),
        "feature_missing_sample": missing_features[:20],
    }


def checkpoint_payload(
    model: nn.Module,
    scaler: dict[str, Any],
    feature_columns: list[str],
    training_diagnostics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "feature_columns": feature_columns,
        "scaler": scaler,
        "model_config": {
            "model_type": getattr(args, "model_type", "transformer"),
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "mlp_hidden_dim": getattr(args, "mlp_hidden_dim", 128),
            "dropout": args.dropout,
        },
        "training_args": {
            "model_type": getattr(args, "model_type", "transformer"),
            "feature_mode": args.feature_mode,
            "feature_select": getattr(args, "feature_select", "none"),
            "max_features": getattr(args, "max_features", 0),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "class_balance": args.class_balance,
            "calibration": getattr(args, "calibration", "platt"),
            "calibration_min_auc": getattr(args, "calibration_min_auc", 0.52),
            "validation_fraction": args.validation_fraction,
            "half_life_days": args.half_life_days,
            "playoff_multiplier": args.playoff_multiplier,
            "finals_multiplier": args.finals_multiplier,
            "min_prior_games": args.min_prior_games,
            "include_late_regular_training": args.include_late_regular_training,
            "late_regular_start_month": args.late_regular_start_month,
            "late_regular_start_day": args.late_regular_start_day,
            "late_regular_all_teams": args.late_regular_all_teams,
            "late_regular_tank_filter": args.late_regular_tank_filter,
            "late_regular_tank_win_pct": args.late_regular_tank_win_pct,
        },
        "training_diagnostics": training_diagnostics,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_checkpoint(payload: dict[str, Any]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MODEL_DIR / f"nba_transformer_{timestamp}.pt"
    torch.save(payload, path)
    shutil.copy2(path, LATEST_CHECKPOINT)
    metadata = {key: value for key, value in payload.items() if key != "model_state"}
    (MODEL_DIR / f"nba_transformer_{timestamp}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (MODEL_DIR / "nba_transformer_latest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model_args = argparse.Namespace(
        model_type=config.get("model_type", "transformer"),
        d_model=int(config.get("d_model", 16)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 1)),
        mlp_hidden_dim=int(config.get("mlp_hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.0)),
    )
    model = build_model(len(payload["feature_columns"]), model_args)
    model.load_state_dict(payload["model_state"])
    return model, payload


def write_prediction(
    prediction_row: pd.DataFrame,
    home_prob: float,
    payload: dict[str, Any],
    trained: bool,
    sanity_report: dict[str, Any],
    swap_diagnostic: dict[str, Any],
) -> None:
    out = prediction_row.copy()
    out["model_home_win_prob"] = home_prob
    out["model_away_win_prob"] = 1 - home_prob
    out["checkpoint_created_at"] = payload.get("created_at")
    out["checkpoint_retrained_this_run"] = trained
    out_path = PREDICTION_DIR / "nba_current_prediction.csv"
    out.to_csv(out_path, index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_created_at": payload.get("created_at"),
        "checkpoint_retrained_this_run": trained,
        "home_team": out.iloc[0].get("home_team"),
        "away_team": out.iloc[0].get("away_team"),
        "model_home_win_prob": home_prob,
        "model_away_win_prob": 1 - home_prob,
        "feature_count": len(payload["feature_columns"]),
        "sanity_report": sanity_report,
        "home_away_swap_diagnostic": swap_diagnostic,
        "training_diagnostics": payload.get("training_diagnostics"),
    }
    (PREDICTION_DIR / "nba_current_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Train a new checkpoint even if one already exists.")
    parser.add_argument("--checkpoint", default=str(LATEST_CHECKPOINT))
    parser.add_argument("--reuse-feature-table", action="store_true")
    parser.add_argument("--feature-mode", choices=["edge", "compact", "wide", "full"], default="wide")
    parser.add_argument("--feature-select", choices=["none", "grouped"], default="grouped")
    parser.add_argument(
        "--no-market-features",
        action="store_true",
        help="Remove market/odds/implied/spread-derived columns from model inputs.",
    )
    parser.add_argument("--max-features", type=int, default=320)
    parser.add_argument("--feature-group-quota", type=int, default=0)
    parser.add_argument("--feature-max-missing", type=float, default=0.45)
    parser.add_argument("--model-type", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--min-prior-games", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no-init-output-bias", dest="init_output_bias", action="store_false")
    parser.add_argument(
        "--class-balance",
        choices=["none", "pos_weight"],
        default="none",
        help="Loss class balancing. Use none for calibrated probabilities; pos_weight for balanced classification.",
    )
    parser.add_argument("--calibration", choices=["none", "platt"], default="platt")
    parser.add_argument("--calibration-min-auc", type=float, default=0.52)
    parser.add_argument("--half-life-days", type=float, default=365.0)
    parser.add_argument("--playoff-multiplier", type=float, default=1.35)
    parser.add_argument("--finals-multiplier", type=float, default=1.60)
    parser.add_argument(
        "--include-late-regular-training",
        action="store_true",
        help="Keep late regular-season games in supervised training labels.",
    )
    parser.add_argument("--late-regular-start-month", type=int, default=4)
    parser.add_argument("--late-regular-start-day", type=int, default=1)
    parser.add_argument(
        "--late-regular-all-teams",
        action="store_true",
        help="Drop all late regular-season games instead of only games involving playoff teams.",
    )
    parser.add_argument(
        "--no-late-regular-tank-filter",
        dest="late_regular_tank_filter",
        action="store_false",
        help="Disable the non-playoff low-win tank-risk part of the late regular-season filter.",
    )
    parser.add_argument(
        "--late-regular-tank-win-pct",
        type=float,
        default=0.40,
        help="Pre-game win percentage threshold for treating a non-playoff team as tank-risk late in the regular season.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    ensure_dirs()
    schedule = load_csv("team_schedule_games.csv")
    team_boxscores = load_csv("team_boxscores.csv")
    player_logs = load_csv("player_game_logs.csv")
    current_game = load_csv("current_game.csv")

    for df in [schedule, team_boxscores, player_logs, current_game]:
        if "game_date" in df.columns:
            df["game_date"] = df["game_date"].astype(str)

    historical_moneylines = load_cached_historical_moneylines()
    strength_context = prepare_strength_context(schedule, historical_moneylines)
    current_row = build_feature_row(
        current_game.iloc[0],
        team_boxscores,
        player_logs,
        include_target=False,
        strength_context=strength_context,
    )
    current_features = pd.DataFrame([current_row])

    checkpoint_path = Path(args.checkpoint)
    trained = False
    if checkpoint_path.exists() and not args.retrain:
        print(f"Loading saved NBA transformer checkpoint: {checkpoint_path}", flush=True)
        model, payload = load_checkpoint(checkpoint_path)
        feature_columns = payload["feature_columns"]
    else:
        feature_table_path = MODEL_DIR / "nba_historical_feature_table.csv"
        if args.reuse_feature_table and feature_table_path.exists():
            print(f"Loading cached NBA historical feature table: {feature_table_path}", flush=True)
            feature_table = pd.read_csv(feature_table_path)
            if not feature_table_has_required_strength(feature_table):
                print("Cached NBA feature table is missing new strength features; rebuilding.", flush=True)
                feature_table = build_training_table(
                    schedule,
                    team_boxscores,
                    player_logs,
                    args.min_prior_games,
                    moneylines=historical_moneylines,
                )
        else:
            print("Building NBA historical feature table...", flush=True)
            feature_table = build_training_table(
                schedule,
                team_boxscores,
                player_logs,
                args.min_prior_games,
                moneylines=historical_moneylines,
            )
        feature_table.to_csv(feature_table_path, index=False)
        training_filter = {"enabled": False, "rows_before": int(len(feature_table)), "rows_removed": 0, "rows_after": int(len(feature_table))}
        if not args.include_late_regular_training:
            feature_table, training_filter = filter_late_regular_training_rows(
                feature_table,
                start_month=args.late_regular_start_month,
                start_day=args.late_regular_start_day,
                playoff_teams_only=not args.late_regular_all_teams,
                tank_win_pct_threshold=args.late_regular_tank_win_pct,
                include_tank_risk=args.late_regular_tank_filter,
            )
            print(
                "Late regular-season training filter: "
                f"removed={training_filter['rows_removed']} "
                f"rows_after={training_filter['rows_after']} "
                f"cutoff={args.late_regular_start_month:02d}-{args.late_regular_start_day:02d} "
                f"playoff_teams_only={not args.late_regular_all_teams} "
                f"tank_filter={args.late_regular_tank_filter} "
                f"tank_win_pct<={args.late_regular_tank_win_pct:.3f}",
                flush=True,
            )
        if len(feature_table) < 40:
            raise RuntimeError(
                f"Only {len(feature_table)} training rows after min-prior-games filtering. "
                "Collect more seasons/full-league data before training."
            )
        feature_columns = selected_feature_columns(feature_table, args.feature_mode)
        feature_columns, feature_blacklist = filter_model_feature_columns(feature_columns, args)
        train_df, val_df = split_train_validation(feature_table, args.validation_fraction)
        feature_columns, feature_selection = select_feature_subset(train_df, feature_columns, args)
        print(
            f"Training rows={len(feature_table)} features={len(feature_columns)} "
            f"mode={args.feature_mode} select={args.feature_select}",
            flush=True,
        )
        ensure_feature_columns(current_features, feature_columns)
        model, scaler, diagnostics = train_model(train_df, val_df, current_features, feature_columns, args)
        diagnostics["training_row_filter"] = training_filter
        diagnostics["feature_blacklist"] = feature_blacklist
        diagnostics["feature_selection"] = feature_selection
        diagnostics["feature_count"] = len(feature_columns)
        payload = checkpoint_payload(model, scaler, feature_columns, diagnostics, args)
        saved = save_checkpoint(payload)
        print(f"Saved NBA transformer checkpoint: {saved}", flush=True)
        trained = True

    swapped_row = build_feature_row(
        swapped_current_game(current_game.iloc[0]),
        team_boxscores,
        player_logs,
        include_target=False,
        strength_context=strength_context,
    )
    swapped_features = pd.DataFrame([swapped_row])
    ensure_feature_columns(current_features, feature_columns)
    ensure_feature_columns(swapped_features, feature_columns)
    x_pred = transform_with_scaler(current_features, feature_columns, payload["scaler"])
    x_swapped = transform_with_scaler(swapped_features, feature_columns, payload["scaler"])
    calibration = payload.get("scaler", {}).get("calibration")
    home_prob = predict_probability(model, x_pred, calibration)
    swapped_home_prob = predict_probability(model, x_swapped, calibration)
    sanity_report = prediction_sanity_report(current_features, feature_columns)
    swap_diagnostic = {
        "swapped_home_team": swapped_features.iloc[0].get("home_team"),
        "swapped_away_team": swapped_features.iloc[0].get("away_team"),
        "swapped_home_win_prob": swapped_home_prob,
        "swapped_away_win_prob": 1 - swapped_home_prob,
        "original_home_prob_plus_swapped_home_prob": home_prob + swapped_home_prob,
        "interpretation": "Swapped row treats the original away team as home; it is a sanity check, not a neutral-site identity test.",
    }
    write_prediction(current_features, home_prob, payload, trained, sanity_report, swap_diagnostic)
    print(
        f"Prediction: {current_features.iloc[0].get('away_team')} at {current_features.iloc[0].get('home_team')} "
        f"home_win_prob={home_prob:.3f} away_win_prob={1 - home_prob:.3f}",
        flush=True,
    )
    print(
        f"Swap diagnostic: {swapped_features.iloc[0].get('away_team')} at {swapped_features.iloc[0].get('home_team')} "
        f"home_win_prob={swapped_home_prob:.3f} away_win_prob={1 - swapped_home_prob:.3f}",
        flush=True,
    )
    print(f"Prediction saved to {PREDICTION_DIR / 'nba_current_prediction.csv'}", flush=True)


if __name__ == "__main__":
    main()
