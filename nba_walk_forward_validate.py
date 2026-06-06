from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nba_transformer_backtest import BACKTEST_DIR, binary_metrics, probability_summary
from nba_transformer_train import (
    MODEL_DIR,
    ensure_feature_columns,
    filter_model_feature_columns,
    predict_probabilities,
    select_feature_subset,
    selected_feature_columns,
    split_train_validation,
    train_model,
    transform_with_scaler,
)


DEFAULT_FEATURE_TABLE = MODEL_DIR / "nba_historical_feature_table_no_market_clean.csv"


def safe_run_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.-")
    return text or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def with_season_end_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["game_date"], errors="coerce")
    out["_season_end_year"] = dates.dt.year + (dates.dt.month >= 10).astype(int)
    return out


def last_regular_window(feature_table: pd.DataFrame, season_end_year: int, tail_games: int) -> pd.DataFrame:
    rows = with_season_end_year(feature_table)
    season_type = pd.to_numeric(rows["season_type"], errors="coerce")
    rows = rows[(rows["_season_end_year"] == int(season_end_year)) & (season_type == 2)].copy()
    rows = rows.dropna(subset=["target_home_win"]).sort_values(["game_date", "event_id"]).reset_index(drop=True)
    if len(rows) < tail_games:
        raise RuntimeError(
            f"Season ending {season_end_year} has only {len(rows)} regular-season target rows; "
            f"need {tail_games}."
        )
    return rows.tail(tail_games).drop(columns=["_season_end_year"]).reset_index(drop=True)


def market_like_features(columns: list[str]) -> list[str]:
    out = []
    for column in columns:
        text = column.lower()
        if any(term in text for term in ("market", "odds", "implied", "no_vig")) or re.search(r"(^|_)spread(_|$)", text):
            out.append(column)
    return out


def run_validation_fold(args: argparse.Namespace) -> dict[str, Any]:
    feature_table = pd.read_csv(args.feature_table)
    feature_table["event_id"] = feature_table["event_id"].astype(str)
    feature_table["game_date"] = feature_table["game_date"].astype(str)
    feature_table = feature_table.dropna(subset=["target_home_win"]).sort_values(["game_date", "event_id"]).reset_index(drop=True)

    target_df = last_regular_window(feature_table, args.validation_season_end_year, args.tail_games)
    if args.sample_games and args.sample_games < len(target_df):
        target_df = target_df.sample(n=args.sample_games, replace=False, random_state=args.seed)
        target_df = target_df.sort_values(["game_date", "event_id"]).reset_index(drop=True)

    cutoff = str(target_df["game_date"].min())
    train_base = feature_table[feature_table["game_date"].astype(str) < cutoff].copy()
    if len(train_base) < args.min_train:
        raise RuntimeError(f"Only {len(train_base)} training rows before validation cutoff {cutoff}; need {args.min_train}.")

    base_features = selected_feature_columns(feature_table, args.feature_mode)
    filtered_features, feature_blacklist = filter_model_feature_columns(base_features, args)
    train_df, internal_val_df = split_train_validation(train_base, args.validation_fraction)
    feature_columns, feature_selection = select_feature_subset(train_df, filtered_features, args)
    ensure_feature_columns(target_df, feature_columns)

    print(
        "training validation fold: "
        f"season={args.validation_season_end_year} cutoff={cutoff} "
        f"train_rows={len(train_df)} internal_val_rows={len(internal_val_df)} "
        f"target_rows={len(target_df)} features={len(feature_columns)}/{len(base_features)}",
        flush=True,
    )
    model, scaler, diagnostics = train_model(train_df, internal_val_df, target_df, feature_columns, args)
    probabilities = predict_probabilities(
        model,
        transform_with_scaler(target_df, feature_columns, scaler),
        scaler.get("calibration"),
    )

    predictions = target_df.copy()
    predictions["model_home_win_prob"] = probabilities
    predictions["model_away_win_prob"] = 1.0 - predictions["model_home_win_prob"]

    selected_market_like = market_like_features(feature_columns)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": safe_run_id(args.run_id),
        "label": args.run_label,
        "feature_table": str(Path(args.feature_table)),
        "settings": {
            "validation_season_end_year": args.validation_season_end_year,
            "tail_games": args.tail_games,
            "sample_games": args.sample_games,
            "train_cutoff_date": cutoff,
            "min_train": args.min_train,
            "feature_mode": args.feature_mode,
            "feature_select": args.feature_select,
            "no_market_features": args.no_market_features,
            "max_features": args.max_features,
            "model_type": args.model_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "fold": {
            "role": "validation",
            "season_end_year": args.validation_season_end_year,
            "regular_season_only": True,
            "target_rows": int(len(predictions)),
            "target_date_range": {
                "start": str(predictions["game_date"].min()),
                "end": str(predictions["game_date"].max()),
            },
            "train_rows_before_cutoff": int(len(train_base)),
            "internal_train_rows": int(len(train_df)),
            "internal_validation_rows": int(len(internal_val_df)),
            "train_cutoff_date": cutoff,
        },
        "feature_audit": {
            "base_feature_count": int(len(base_features)),
            "post_blacklist_feature_count": int(len(filtered_features)),
            "selected_feature_count": int(len(feature_columns)),
            "selected_market_like_count": int(len(selected_market_like)),
            "selected_market_like_features": selected_market_like,
            "blacklist": feature_blacklist,
        },
        "model_metrics": binary_metrics(predictions, "model_home_win_prob"),
        "model_probability_distribution": probability_summary(predictions["model_home_win_prob"]),
        "training_diagnostics": {
            **diagnostics,
            "feature_selection": feature_selection,
            "feature_count": int(len(feature_columns)),
        },
        "notes": [
            "Validation targets are a full last-N regular-season window, not a latest tail across the whole dataset.",
            "Model training rows are restricted to games before the first validation target date.",
            "Market odds are not used as model features when --no-market-features is enabled.",
            "Older ESPN summaries in this workspace do not contain pickcenter odds, so this validation fold reports model accuracy/calibration but not betting P&L.",
        ],
    }

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    run_id = report["run_id"]
    predictions_path = BACKTEST_DIR / f"nba_walk_forward_validation_predictions_{run_id}.csv"
    report_path = BACKTEST_DIR / f"nba_walk_forward_validation_report_{run_id}.json"
    latest_predictions_path = BACKTEST_DIR / "nba_walk_forward_validation_predictions_latest.csv"
    latest_report_path = BACKTEST_DIR / "nba_walk_forward_validation_report_latest.json"
    predictions.to_csv(predictions_path, index=False)
    predictions.to_csv(latest_predictions_path, index=False)
    report["artifact_paths"] = {
        "predictions": str(predictions_path),
        "latest_predictions": str(latest_predictions_path),
        "report": str(report_path),
        "latest_report": str(latest_report_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-table", default=str(DEFAULT_FEATURE_TABLE))
    parser.add_argument("--validation-season-end-year", type=int, default=2025)
    parser.add_argument("--tail-games", type=int, default=230)
    parser.add_argument("--sample-games", type=int, default=0, help="0 means use the full validation window.")
    parser.add_argument("--min-train", type=int, default=1500)
    parser.add_argument("--feature-mode", choices=["edge", "compact", "wide", "full"], default="edge")
    parser.add_argument("--feature-select", choices=["none", "grouped"], default="grouped")
    parser.add_argument("--no-market-features", action="store_true")
    parser.add_argument("--max-features", type=int, default=320)
    parser.add_argument("--feature-group-quota", type=int, default=0)
    parser.add_argument("--feature-max-missing", type=float, default=0.45)
    parser.add_argument("--model-type", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--no-init-output-bias", dest="init_output_bias", action="store_false")
    parser.add_argument("--class-balance", choices=["none", "pos_weight"], default="none")
    parser.add_argument("--calibration", choices=["none", "platt"], default="platt")
    parser.add_argument("--calibration-min-auc", type=float, default=0.52)
    parser.add_argument("--half-life-days", type=float, default=365.0)
    parser.add_argument("--playoff-multiplier", type=float, default=1.35)
    parser.add_argument("--finals-multiplier", type=float, default=1.60)
    parser.add_argument("--seed", type=int, default=605)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-id", default="20260605_no_market_validation_2025_last230")
    parser.add_argument("--run-label", default="No-market validation: 2025 last-230 regular-season window")
    args = parser.parse_args()

    report = run_validation_fold(args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
