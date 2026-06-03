from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
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


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    path = LATEST_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing NBA data artifact: {path}")
    return pd.read_csv(path)


def numeric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_playoff_round(round_type: Any) -> int:
    text = str(round_type or "").lower()
    return int("final" in text or "playoff" in text or "round" in text)


def is_finals_round(round_type: Any) -> int:
    return int("final" in str(round_type or "").lower())


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


def team_games_before(team_boxscores: pd.DataFrame, team_id: str, game_date: str, opponent_id: str | None = None) -> pd.DataFrame:
    rows = team_boxscores[
        (team_boxscores["team_id"].astype(str) == str(team_id))
        & (team_boxscores["game_date"].astype(str) < str(game_date))
    ].copy()
    if opponent_id is not None:
        rows = rows[rows["opponent_team_id"].astype(str) == str(opponent_id)]
    return rows.sort_values(["game_date", "event_id"])


def aggregate_team(
    team_boxscores: pd.DataFrame,
    team_id: str,
    game_date: str,
    split: str,
    opponent_id: str | None = None,
) -> dict[str, Any]:
    rows = team_games_before(team_boxscores, team_id, game_date, opponent_id=opponent_id)
    if split == "regular":
        rows = rows[rows["season_type"] == 2]
    elif split == "playoffs":
        rows = rows[rows["season_type"] == 3]
    elif split.startswith("last"):
        rows = rows.tail(int(split.replace("last", "")))

    out: dict[str, Any] = {
        "games": int(len(rows)),
        "win_pct": float(pd.to_numeric(rows.get("won"), errors="coerce").mean()) if not rows.empty else None,
    }
    if not rows.empty:
        last_game_date = pd.to_datetime(rows["game_date"]).max()
        out["rest_days"] = max(float((pd.to_datetime(game_date) - last_game_date).days), 0.0)
    else:
        out["rest_days"] = None

    stats = [
        "points_for",
        "points_against",
        "margin",
        "estimated_possessions",
        "off_rating_est",
        "efg_pct_est",
        "three_pa_rate",
        "ft_rate",
        "turnover_rate",
        "oreb_rate_proxy",
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
    for stat in stats:
        if stat in rows.columns:
            out[f"avg_{stat}"] = float(pd.to_numeric(rows[stat], errors="coerce").mean()) if not rows.empty else None
    return out


def aggregate_players(player_logs: pd.DataFrame, team_id: str, game_date: str) -> dict[str, Any]:
    rows = player_logs[
        (player_logs["team_id"].astype(str) == str(team_id))
        & (player_logs["game_date"].astype(str) < str(game_date))
    ].copy()
    if rows.empty:
        return {
            "logged_players": 0,
            "rotation_players_last10": 0,
        }

    rows = rows.sort_values(["athlete_id", "game_date"])
    recent = rows.groupby("athlete_id", group_keys=False).tail(10)
    player_means = recent.groupby("athlete_id").agg(
        player=("player", "last"),
        games=("event_id", "nunique"),
        minutes=("minutes", "mean"),
        points=("points", "mean"),
        rebounds=("rebounds", "mean"),
        assists=("assists", "mean"),
        turnovers=("turnovers", "mean"),
        steals=("steals", "mean"),
        blocks=("blocks", "mean"),
        plus_minus=("plus_minus", "mean"),
        usage_proxy=("usage_proxy", "mean"),
        points_per36=("points_per36", "mean"),
        rebounds_per36=("rebounds_per36", "mean"),
        assists_per36=("assists_per36", "mean"),
    )
    player_means = player_means.reset_index()
    player_means["minutes"] = pd.to_numeric(player_means["minutes"], errors="coerce").fillna(0)
    player_means["points"] = pd.to_numeric(player_means["points"], errors="coerce").fillna(0)
    rotation = player_means[player_means["minutes"] >= 8].copy()
    minute_ranked = rotation.sort_values("minutes", ascending=False)
    scoring_ranked = rotation.sort_values("points", ascending=False)

    out: dict[str, Any] = {
        "logged_players": int(player_means["athlete_id"].nunique()),
        "rotation_players_last10": int(len(rotation)),
    }
    for label, frame in [("top5", minute_ranked.head(5)), ("top8", minute_ranked.head(8))]:
        out[f"{label}_count"] = int(len(frame))
        for stat in [
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
            if stat in frame.columns:
                values = pd.to_numeric(frame[stat], errors="coerce")
                out[f"{label}_sum_{stat}"] = float(values.sum()) if not frame.empty else None
                out[f"{label}_avg_{stat}"] = float(values.mean()) if not frame.empty else None

    for rank, (_, player) in enumerate(scoring_ranked.head(3).iterrows(), start=1):
        out[f"scorer{rank}_ppg"] = numeric(player.get("points"))
        out[f"scorer{rank}_mpg"] = numeric(player.get("minutes"))
        out[f"scorer{rank}_usage"] = numeric(player.get("usage_proxy"))
    return out


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


def build_feature_row(
    game: pd.Series | dict[str, Any],
    team_boxscores: pd.DataFrame,
    player_logs: pd.DataFrame,
    include_target: bool,
    team_cache: dict[tuple[str, str, str, str], dict[str, Any]] | None = None,
    player_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    game_dict = dict(game)
    home_id = str(game_dict["home_team_id"])
    away_id = str(game_dict["away_team_id"])
    game_date = str(game_dict["game_date"])
    row: dict[str, Any] = {
        "event_id": game_dict.get("event_id"),
        "game_datetime": game_dict.get("game_datetime"),
        "game_date": game_date,
        "name": game_dict.get("name") or game_dict.get("event_name"),
        "short_name": game_dict.get("short_name"),
        "season_type": game_dict.get("season_type"),
        "round_type": game_dict.get("round_type"),
        "neutral_site": int(bool(game_dict.get("neutral_site"))) if game_dict.get("neutral_site") is not None else 0,
        "is_playoff": int(game_dict.get("season_type") == 3),
        "is_finals": is_finals_round(game_dict.get("round_type")),
        "home_team_id": home_id,
        "home_team": game_dict.get("home_team"),
        "away_team_id": away_id,
        "away_team": game_dict.get("away_team"),
    }

    for split in ["all", "regular", "playoffs", "last3", "last5", "last10", "last20"]:
        home = cached_aggregate_team(team_cache, team_boxscores, home_id, game_date, split)
        away = cached_aggregate_team(team_cache, team_boxscores, away_id, game_date, split)
        row.update(prefix(f"home_{split}", home))
        row.update(prefix(f"away_{split}", away))
        row.update(edge(f"edge_{split}", home, away))

    home_h2h = cached_aggregate_team(team_cache, team_boxscores, home_id, game_date, "all", opponent_id=away_id)
    away_h2h = cached_aggregate_team(team_cache, team_boxscores, away_id, game_date, "all", opponent_id=home_id)
    row.update(prefix("home_h2h", home_h2h))
    row.update(prefix("away_h2h", away_h2h))
    row.update(edge("edge_h2h", home_h2h, away_h2h))

    home_players = cached_aggregate_players(player_cache, player_logs, home_id, game_date)
    away_players = cached_aggregate_players(player_cache, player_logs, away_id, game_date)
    row.update(prefix("home_players", home_players))
    row.update(prefix("away_players", away_players))
    row.update(edge("edge_players", home_players, away_players))

    if include_target:
        row["target_home_win"] = int(game_dict["home_win"])
        row["target_home_margin"] = numeric(game_dict.get("home_margin"))
    return row


def build_training_table(schedule: pd.DataFrame, team_boxscores: pd.DataFrame, player_logs: pd.DataFrame, min_prior_games: int) -> pd.DataFrame:
    completed = schedule[
        (schedule["completed"] == True)
        & schedule["home_win"].notna()
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    ].copy()
    completed = completed.sort_values(["game_date", "event_id"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    team_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    player_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for _, game in completed.iterrows():
        home_prior = cached_aggregate_team(
            team_cache,
            team_boxscores,
            str(game["home_team_id"]),
            str(game["game_date"]),
            "all",
        )["games"]
        away_prior = cached_aggregate_team(
            team_cache,
            team_boxscores,
            str(game["away_team_id"]),
            str(game["game_date"]),
            "all",
        )["games"]
        if min(home_prior, away_prior) < min_prior_games:
            continue
        rows.append(
            build_feature_row(
                game,
                team_boxscores,
                player_logs,
                include_target=True,
                team_cache=team_cache,
                player_cache=player_cache,
            )
        )
        if len(rows) % 250 == 0:
            print(f"  feature rows built: {len(rows)}", flush=True)
    return pd.DataFrame(rows).sort_values(["game_date", "event_id"]).reset_index(drop=True)


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
) -> tuple[FeatureTransformer, dict[str, Any], dict[str, Any]]:
    set_seed(args.seed)
    scaled = scale_arrays(train_df, val_df, pred_df, feature_columns)
    y_train = train_df["target_home_win"].astype(float).to_numpy(dtype=np.float32)
    y_val = val_df["target_home_win"].astype(float).to_numpy(dtype=np.float32)
    w_train = sample_weights(train_df, args.half_life_days, args.playoff_multiplier, args.finals_multiplier)
    w_val = sample_weights(val_df, args.half_life_days, args.playoff_multiplier, args.finals_multiplier)

    device = torch.device("cpu")
    model = FeatureTransformer(
        n_features=len(feature_columns),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    positive_rate = max(float(y_train.mean()), 1e-4)
    pos_weight = torch.tensor([(1 - positive_rate) / positive_rate], dtype=torch.float32, device=device)
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
            val_raw = loss_fn(model(val_x), val_y)
            val_loss = float(((val_raw * val_w).sum() / val_w.sum().clamp_min(1e-6)).detach().cpu().item())
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})

        if args.verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

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

    diag = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "positive_rate": float(positive_rate),
        "weighted_train_mean": float(w_train.mean()) if len(w_train) else None,
        "history_tail": history[-10:],
    }
    scaler = {
        "median": scaled["median"].tolist(),
        "mean": scaled["mean"].tolist(),
        "std": scaled["std"].tolist(),
    }
    return model, scaler, diag


def predict_probability(model: FeatureTransformer, x: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        prob = float(torch.sigmoid(logits).cpu().numpy()[0])
    return min(max(prob, 0.02), 0.98)


def checkpoint_payload(
    model: FeatureTransformer,
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
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
        },
        "training_args": {
            "feature_mode": args.feature_mode,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "validation_fraction": args.validation_fraction,
            "half_life_days": args.half_life_days,
            "playoff_multiplier": args.playoff_multiplier,
            "finals_multiplier": args.finals_multiplier,
            "min_prior_games": args.min_prior_games,
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


def load_checkpoint(path: Path) -> tuple[FeatureTransformer, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = FeatureTransformer(
        n_features=len(payload["feature_columns"]),
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(payload["model_state"])
    return model, payload


def write_prediction(
    prediction_row: pd.DataFrame,
    home_prob: float,
    payload: dict[str, Any],
    trained: bool,
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
        "training_diagnostics": payload.get("training_diagnostics"),
    }
    (PREDICTION_DIR / "nba_current_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Train a new checkpoint even if one already exists.")
    parser.add_argument("--checkpoint", default=str(LATEST_CHECKPOINT))
    parser.add_argument("--reuse-feature-table", action="store_true")
    parser.add_argument("--feature-mode", choices=["compact", "full"], default="compact")
    parser.add_argument("--min-prior-games", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--half-life-days", type=float, default=365.0)
    parser.add_argument("--playoff-multiplier", type=float, default=1.35)
    parser.add_argument("--finals-multiplier", type=float, default=1.60)
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

    current_row = build_feature_row(current_game.iloc[0], team_boxscores, player_logs, include_target=False)
    current_features = pd.DataFrame([current_row])

    checkpoint_path = Path(args.checkpoint)
    trained = False
    if checkpoint_path.exists() and not args.retrain:
        print(f"Loading saved NBA transformer checkpoint: {checkpoint_path}", flush=True)
        model, payload = load_checkpoint(checkpoint_path)
        feature_columns = payload["feature_columns"]
        for col in feature_columns:
            if col not in current_features.columns:
                current_features[col] = np.nan
        x_pred = transform_with_scaler(current_features, feature_columns, payload["scaler"])
    else:
        feature_table_path = MODEL_DIR / "nba_historical_feature_table.csv"
        if args.reuse_feature_table and feature_table_path.exists():
            print(f"Loading cached NBA historical feature table: {feature_table_path}", flush=True)
            feature_table = pd.read_csv(feature_table_path)
        else:
            print("Building NBA historical feature table...", flush=True)
            feature_table = build_training_table(schedule, team_boxscores, player_logs, args.min_prior_games)
        if len(feature_table) < 40:
            raise RuntimeError(
                f"Only {len(feature_table)} training rows after min-prior-games filtering. "
                "Collect more seasons/full-league data before training."
            )
        feature_table.to_csv(feature_table_path, index=False)
        feature_columns = selected_feature_columns(feature_table, args.feature_mode)
        print(f"Training rows={len(feature_table)} features={len(feature_columns)} mode={args.feature_mode}", flush=True)
        train_df, val_df = split_train_validation(feature_table, args.validation_fraction)
        for col in feature_columns:
            if col not in current_features.columns:
                current_features[col] = np.nan
        model, scaler, diagnostics = train_model(train_df, val_df, current_features, feature_columns, args)
        payload = checkpoint_payload(model, scaler, feature_columns, diagnostics, args)
        saved = save_checkpoint(payload)
        print(f"Saved NBA transformer checkpoint: {saved}", flush=True)
        x_pred = transform_with_scaler(current_features, feature_columns, scaler)
        trained = True

    home_prob = predict_probability(model, x_pred)
    write_prediction(current_features, home_prob, payload, trained)
    print(
        f"Prediction: {current_features.iloc[0].get('away_team')} at {current_features.iloc[0].get('home_team')} "
        f"home_win_prob={home_prob:.3f} away_win_prob={1 - home_prob:.3f}",
        flush=True,
    )
    print(f"Prediction saved to {PREDICTION_DIR / 'nba_current_prediction.csv'}", flush=True)


if __name__ == "__main__":
    main()
