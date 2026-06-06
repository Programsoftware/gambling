from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kelly_moneyline_backtest import (
    DEFAULT_SEASONS,
    FEATURE_COLUMNS,
    OUT_DIR,
    bootstrap_paths,
    build_feature_table,
    fetch_nhl_schedule,
    fetch_statmuse_moneylines,
    parse_seasons,
    simulate_kelly,
)
from nhl_predict_tonight import append_recent_snapshot_games, load_historical_odds, moneyline_consensus


TRANSFORMER_DIR = OUT_DIR / "transformer"
DATA_ROOT = OUT_DIR.parent
LATEST_DIR = DATA_ROOT / "latest"
PREDICTION_DIR = DATA_ROOT / "predictions"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


class FeatureTransformer(nn.Module):
    """Small FT-Transformer-style classifier for tabular matchup features."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
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
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_projection(x.unsqueeze(-1)) + self.feature_embedding.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(-1)


def numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(train_x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    train = np.where(np.isfinite(train_x), train_x, med)
    test = np.where(np.isfinite(test_x), test_x, med)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    return (train - mean) / std, (test - mean) / std


def split_train_validation(
    train_df: pd.DataFrame,
    train_x: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_dates = sorted(train_df["game_date"].astype(str).unique())
    val_dates_count = max(1, int(math.ceil(len(unique_dates) * validation_fraction)))
    val_dates = set(unique_dates[-val_dates_count:])
    val_mask = train_df["game_date"].astype(str).isin(val_dates).to_numpy()

    if val_mask.sum() < 20 or (~val_mask).sum() < 40:
        cutoff = max(1, int(len(train_df) * (1 - validation_fraction)))
        val_mask = np.zeros(len(train_df), dtype=bool)
        val_mask[cutoff:] = True

    return train_x[~val_mask], y[~val_mask], train_x[val_mask], y[val_mask]


def train_transformer(
    train_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    validation_fraction: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    patience: int,
) -> tuple[FeatureTransformer, dict[str, Any]]:
    set_seed(seed)

    raw_x = numeric_matrix(train_df)
    y = train_df["home_win"].astype(float).to_numpy(dtype=np.float32)
    train_x, val_x = raw_x.copy(), raw_x.copy()
    train_x, _ = standardize(raw_x, raw_x)

    fit_x, fit_y, holdout_x, holdout_y = split_train_validation(
        train_df,
        train_x,
        y,
        validation_fraction,
    )

    device = torch.device("cpu")
    model = FeatureTransformer(
        n_features=len(FEATURE_COLUMNS),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    positive_rate = max(float(fit_y.mean()), 1e-4)
    pos_weight = torch.tensor([(1 - positive_rate) / positive_rate], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(
        torch.tensor(fit_x, dtype=torch.float32),
        torch.tensor(fit_y, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    holdout_tensor = torch.tensor(holdout_x, dtype=torch.float32, device=device)
    holdout_target = torch.tensor(holdout_y, dtype=torch.float32, device=device)

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            holdout_loss = float(loss_fn(model(holdout_tensor), holdout_target).item())

        if holdout_loss < best_loss - 1e-4:
            best_loss = holdout_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "train_rows": int(len(train_df)),
    }


def predict_with_model(model: FeatureTransformer, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    train_x = numeric_matrix(train_df)
    test_x = numeric_matrix(test_df)
    _, test_scaled = standardize(train_x, test_x)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(test_scaled, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()
    return np.clip(probs, 0.02, 0.98)


def scaler_payload(train_df: pd.DataFrame) -> dict[str, Any]:
    train_x = numeric_matrix(train_df)
    med = np.nanmedian(train_x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    train = np.where(np.isfinite(train_x), train_x, med)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    return {
        "median": med.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def train_current_transformer_prediction(
    features: pd.DataFrame,
    schedule: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.target_date:
        return {}

    current_path = LATEST_DIR / "current_game.csv"
    odds_path = LATEST_DIR / "odds_market_snapshot.csv"
    if not current_path.exists() or not odds_path.exists():
        raise RuntimeError("Run collect_nhl_data.py first so data/latest has current_game and odds_market_snapshot.")

    current = pd.read_csv(current_path).iloc[0].to_dict()
    current_date = str(current.get("game_date") or "")
    if current_date != args.target_date:
        raise RuntimeError(f"Latest current_game is {current_date}, expected {args.target_date}.")

    away_team = str(current.get("away_team") or "").upper()
    home_team = str(current.get("home_team") or "").upper()
    expected_teams = {item.strip().upper() for item in args.teams.split(",") if item.strip()}
    if expected_teams and {away_team, home_team} != expected_teams:
        raise RuntimeError(f"Latest current_game teams are {away_team},{home_team}, expected {sorted(expected_teams)}.")

    train_df = features[(features["game_date"].astype(str) < args.target_date) & features["home_win"].notna()].copy()
    if len(train_df) < args.min_train:
        raise RuntimeError(f"Only {len(train_df)} current-prediction training rows; need {args.min_train}.")

    odds_snapshot = pd.read_csv(odds_path)
    prices = moneyline_consensus(odds_snapshot, away_team=away_team, home_team=home_team)
    current_row = {
        "game_date": args.target_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_odds": prices["home_odds"],
        "away_odds": prices["away_odds"],
        "home_score": np.nan,
        "away_score": np.nan,
        "home_win": np.nan,
        "season": int(current.get("season") or 20252026),
        "source_query": "data/latest/odds_market_snapshot.csv",
        "source": "current_market_snapshot",
        "market_home_raw_prob": prices["market_home_raw_prob"],
        "market_away_raw_prob": prices["market_away_raw_prob"],
        "market_home_no_vig_prob": prices["market_home_no_vig_prob"],
        "home_decimal_odds": prices["home_decimal_odds"],
        "away_decimal_odds": prices["away_decimal_odds"],
    }
    current_features = build_feature_table(pd.DataFrame([current_row]), schedule)

    print(
        "training current-game transformer: "
        f"train_rows={len(train_df)} target={away_team} at {home_team} {args.target_date}",
        flush=True,
    )
    model, diag = train_transformer(
        train_df=train_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        patience=args.patience,
    )
    print(
        "finished current-game transformer: "
        f"best_epoch={diag.get('best_epoch')} "
        f"validation_loss={float(diag.get('best_validation_loss', float('nan'))):.6f}",
        flush=True,
    )
    home_prob = float(predict_with_model(model, train_df, current_features)[0])
    away_prob = 1.0 - home_prob

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = TRANSFORMER_DIR / "transformer_current_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "model_config": {
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
            },
            "scaler": scaler_payload(train_df),
            "training_diagnostics": diag,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        checkpoint_path,
    )

    prediction = {
        **current_row,
        "game_id": current.get("game_id"),
        "start_time_utc": current.get("start_time_utc"),
        "model_home_win_prob": home_prob,
        "model_away_win_prob": away_prob,
        "predicted_winner": home_team if home_prob >= away_prob else away_team,
        "training_rows": int(len(train_df)),
        "best_epoch": diag.get("best_epoch"),
        "best_validation_loss": diag.get("best_validation_loss"),
    }
    prediction_path = PREDICTION_DIR / "nhl_transformer_current_prediction.csv"
    pd.DataFrame([prediction]).to_csv(prediction_path, index=False)
    return {
        "prediction_csv": str(prediction_path),
        "checkpoint": str(checkpoint_path),
        "game_date": args.target_date,
        "game_id": current.get("game_id"),
        "away_team": away_team,
        "home_team": home_team,
        "home_win_probability": home_prob,
        "away_win_probability": away_prob,
        "predicted_winner": prediction["predicted_winner"],
        "training_rows": int(len(train_df)),
        "training_diagnostics": diag,
        "current_market": {
            "home_odds": int(prices["home_odds"]),
            "away_odds": int(prices["away_odds"]),
            "home_no_vig_probability": float(prices["market_home_no_vig_prob"]),
            "away_no_vig_probability": float(1.0 - prices["market_home_no_vig_prob"]),
        },
    }


def money_text(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def print_live_pnl(
    cumulative_predictions: pd.DataFrame,
    latest_predictions: pd.DataFrame,
    args: argparse.Namespace,
    label: str,
) -> None:
    if not getattr(args, "live_pnl", True) or cumulative_predictions.empty:
        return

    bets, summary = simulate_kelly(
        cumulative_predictions,
        starting_bankroll=args.bankroll,
        kelly_multiplier=args.kelly_multiplier,
        max_bet_fraction=args.max_bet_fraction,
        min_edge=0.0,
        max_market_disagreement=None,
        market_blend=0.0,
        min_model_prob=0.0,
        side_filter="all",
    )
    latest_bets = bets.tail(len(latest_predictions)) if not latest_predictions.empty else bets.iloc[0:0]
    latest_profit = float(latest_bets["profit"].sum()) if not latest_bets.empty else 0.0

    print(
        "pnl "
        f"{label}: "
        f"last={money_text(latest_profit)} "
        f"total={money_text(float(summary['profit']))} "
        f"bank=${float(summary['ending_bankroll']):,.2f} "
        f"bets={int(summary['bets_placed'])}/{int(summary['games_evaluated'])}",
        flush=True,
    )


def walk_forward_transformer_predictions(
    features: pd.DataFrame,
    requested_games: int,
    min_train: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    df = features.dropna(subset=["home_win", "home_odds", "away_odds"]).copy()
    df = df.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)

    candidates: list[int] = []
    for idx, row in df.iterrows():
        train_rows = df[df["game_date"] < row["game_date"]]
        if len(train_rows) >= min_train:
            candidates.append(idx)

    if not candidates:
        raise RuntimeError("Not enough rows for the transformer walk-forward test.")

    selected_indices = candidates[-requested_games:]
    selected = df.loc[selected_indices].copy()

    if args.fit_mode == "single":
        first_target_date = str(selected["game_date"].min())
        train_df = df[df["game_date"] < first_target_date].copy()
        print(
            "training one holdout transformer: "
            f"train_rows={len(train_df)} target_games={len(selected)} "
            f"target_start={first_target_date}",
            flush=True,
        )
        model, diag = train_transformer(
            train_df=train_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            patience=args.patience,
        )
        print(
            "finished holdout transformer: "
            f"best_epoch={diag.get('best_epoch')} "
            f"validation_loss={float(diag.get('best_validation_loss', float('nan'))):.6f}",
            flush=True,
        )
        probs = predict_with_model(model, train_df, selected)
        out = selected.copy()
        out["train_rows"] = len(train_df)
        out["model_home_win_prob_raw"] = probs
        out["model_home_win_prob"] = (
            args.market_blend * out["market_home_no_vig_prob"].astype(float)
            + (1 - args.market_blend) * out["model_home_win_prob_raw"].astype(float)
        )
        out["model_away_win_prob"] = 1 - out["model_home_win_prob"]
        out["market_away_no_vig_prob"] = 1 - out["market_home_no_vig_prob"].astype(float)
        if getattr(args, "live_pnl", True):
            cumulative: list[pd.DataFrame] = []
            for date, date_df in out.groupby("game_date", sort=True):
                cumulative.append(date_df)
                cumulative_df = pd.concat(cumulative, ignore_index=True)
                print_live_pnl(cumulative_df, date_df, args, str(date))
        diag["fit_mode"] = "single"
        diag["target_start_date"] = first_target_date
        diag["target_end_date"] = str(selected["game_date"].max())
        diag["target_games"] = int(len(selected))
        return out.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True), [diag]

    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for date, test_df in selected.groupby("game_date", sort=True):
        train_df = df[df["game_date"] < date].copy()
        print(
            f"training transformer for {date}: train_rows={len(train_df)} target_games={len(test_df)}",
            flush=True,
        )
        model, diag = train_transformer(
            train_df=train_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + len(diagnostics),
            validation_fraction=args.validation_fraction,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            patience=args.patience,
        )
        print(
            f"finished transformer for {date}: "
            f"best_epoch={diag.get('best_epoch')} "
            f"validation_loss={float(diag.get('best_validation_loss', float('nan'))):.6f}",
            flush=True,
        )
        probs = predict_with_model(model, train_df, test_df)
        out = test_df.copy()
        out["train_rows"] = len(train_df)
        out["model_home_win_prob_raw"] = probs
        out["model_home_win_prob"] = (
            args.market_blend * out["market_home_no_vig_prob"].astype(float)
            + (1 - args.market_blend) * out["model_home_win_prob_raw"].astype(float)
        )
        out["model_away_win_prob"] = 1 - out["model_home_win_prob"]
        out["market_away_no_vig_prob"] = 1 - out["market_home_no_vig_prob"].astype(float)
        predictions.append(out)
        cumulative_predictions = pd.concat(predictions, ignore_index=True)
        print_live_pnl(cumulative_predictions, out, args, str(date))
        diag["target_date"] = date
        diag["target_games"] = int(len(test_df))
        diagnostics.append(diag)

    pred_df = pd.concat(predictions, ignore_index=True)
    return pred_df.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--bankroll", type=float, default=1000)
    parser.add_argument("--min-train", type=int, default=1000)
    parser.add_argument("--kelly-multiplier", type=float, default=1.0)
    parser.add_argument("--max-bet-fraction", type=float, default=0.10)
    parser.add_argument("--market-blend", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--validation-fraction", type=float, default=0.18)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-live-pnl",
        dest="live_pnl",
        action="store_false",
        help="Disable live PnL progress lines during walk-forward training.",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["expanding", "single"],
        default="expanding",
        help="expanding retrains by target date; single trains once before the test window.",
    )
    parser.add_argument(
        "--seasons",
        default=",".join(str(season) for season in DEFAULT_SEASONS),
        help="Comma-separated NHL season ids, e.g. 20222023,20232024,20242025,20252026.",
    )
    parser.add_argument(
        "--target-date",
        help="Append saved processed odds snapshots before this game date, e.g. 2026-06-06.",
    )
    parser.add_argument(
        "--teams",
        default="VGK,CAR",
        help="Comma-separated matchup team abbreviations used when appending recent processed snapshots.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--predict-current",
        action="store_true",
        help="Train a final transformer on all rows before --target-date and predict data/latest/current_game.csv.",
    )
    args = parser.parse_args()

    TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)

    seasons = parse_seasons(args.seasons)
    odds = load_historical_odds(seasons=seasons, refresh=args.refresh)
    schedule = fetch_nhl_schedule(seasons=seasons, refresh=args.refresh)
    appended_rows: list[dict[str, Any]] = []
    if args.target_date:
        teams = {item.strip().upper() for item in args.teams.split(",") if item.strip()}
        odds, appended_rows = append_recent_snapshot_games(odds, schedule, args.target_date, teams)
    odds.to_csv(TRANSFORMER_DIR / "transformer_historical_moneylines.csv", index=False)
    features = build_feature_table(odds, schedule)
    features.to_csv(TRANSFORMER_DIR / "transformer_feature_table.csv", index=False)
    current_prediction = train_current_transformer_prediction(features, schedule, args) if args.predict_current else {}

    predictions, diagnostics = walk_forward_transformer_predictions(
        features,
        requested_games=args.games,
        min_train=args.min_train,
        args=args,
    )
    predictions.to_csv(TRANSFORMER_DIR / "transformer_walk_forward_predictions.csv", index=False)

    bets, summary = simulate_kelly(
        predictions,
        starting_bankroll=args.bankroll,
        kelly_multiplier=args.kelly_multiplier,
        max_bet_fraction=args.max_bet_fraction,
        min_edge=0.0,
        max_market_disagreement=None,
        market_blend=0.0,
        min_model_prob=0.0,
        side_filter="all",
    )
    bets.to_csv(TRANSFORMER_DIR / "transformer_kelly_bets.csv", index=False)

    summary.update(
        {
            "model_type": "feature_token_transformer_encoder",
            "historical_odds_rows": int(len(odds)),
            "feature_rows": int(len(features)),
            "prediction_rows": int(len(predictions)),
            "seasons": seasons,
            "target_date": args.target_date,
            "recent_snapshot_rows_appended": appended_rows,
            "current_prediction": current_prediction,
            "fit_mode": args.fit_mode,
            "market_blend": args.market_blend,
            "min_train": args.min_train,
            "epochs": args.epochs,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "training_diagnostics": diagnostics,
            "bootstrap": bootstrap_paths(
                bets,
                starting_bankroll=args.bankroll,
                trials=1000,
                games=min(args.games, max(int(summary["bets_placed"]), 1)),
            ),
            "notes": [
                "Each numeric matchup feature is treated as a transformer token.",
                "Feature values are standardized using only training rows before each target date.",
                "The walk-forward split excludes target-date games from training.",
                "Kelly betting uses the same simulator as the non-transformer baseline.",
                "No minimum model-probability skip gate is applied.",
            ],
        }
    )

    (TRANSFORMER_DIR / "transformer_kelly_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
