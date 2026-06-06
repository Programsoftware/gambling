from __future__ import annotations

import argparse
import gc
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from nba_transformer_train import (
    LATEST_DIR,
    MODEL_DIR,
    NBA_DIR,
    build_training_table,
    feature_table_has_required_strength,
    filter_model_feature_columns,
    filter_late_regular_training_rows,
    late_regular_rest_mask,
    load_csv,
    predict_probabilities,
    select_feature_subset,
    selected_feature_columns,
    split_train_validation,
    train_model,
    transform_with_scaler,
)


BACKTEST_DIR = NBA_DIR / "backtests" / "transformer"
DEFAULT_FEATURE_TABLE = MODEL_DIR / "nba_historical_feature_table.csv"


def ensure_dirs() -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


def safe_run_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    text = text.strip("_.-")
    return text or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def numeric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_american(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return int(value)
    text = str(value).replace("−", "-").replace("–", "-").strip()
    if text.lower() in {"even", "ev", "pk"}:
        return 100
    match = re.search(r"[+-]?\d+", text)
    if not match:
        return None
    parsed = int(match.group())
    return parsed if parsed != 0 else None


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def no_vig_home_prob(home_odds: int, away_odds: int) -> float:
    home = american_to_implied(home_odds)
    away = american_to_implied(away_odds)
    return home / (home + away)


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - probability
    return max((b * probability - q) / b, 0.0)


def max_drawdown(values: list[float], starting_bankroll: float) -> float:
    peak = starting_bankroll
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, (value - peak) / peak)
    return abs(worst)


def money_text(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def latest_raw_dir() -> Path:
    manifest_path = LATEST_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = manifest.get("run_id")
        if run_id:
            path = NBA_DIR / "raw" / str(run_id)
            if path.exists():
                return path

    raw_root = NBA_DIR / "raw"
    candidates = [path for path in raw_root.iterdir() if path.is_dir()] if raw_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No NBA raw data directories found under {raw_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def read_pickcenter_moneyline(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    event_match = re.search(r"_(\d+)\.json$", path.name)
    event_id = event_match.group(1) if event_match else str(data.get("header", {}).get("id") or "")
    pickcenter = data.get("pickcenter") or []
    if not isinstance(pickcenter, list):
        pickcenter = [pickcenter]

    for item in pickcenter:
        if not isinstance(item, dict):
            continue
        home_odds = item.get("homeTeamOdds") or {}
        away_odds = item.get("awayTeamOdds") or {}
        home_moneyline = parse_american(home_odds.get("moneyLine"))
        away_moneyline = parse_american(away_odds.get("moneyLine"))

        moneyline = item.get("moneyline") or {}
        if home_moneyline is None:
            home_moneyline = parse_american(((moneyline.get("home") or {}).get("close") or {}).get("odds"))
        if away_moneyline is None:
            away_moneyline = parse_american(((moneyline.get("away") or {}).get("close") or {}).get("odds"))

        if home_moneyline is None or away_moneyline is None:
            continue

        home_spread_odds = parse_american(home_odds.get("spreadOdds"))
        away_spread_odds = parse_american(away_odds.get("spreadOdds"))
        provider = item.get("provider") if isinstance(item.get("provider"), dict) else {}
        return {
            "event_id": event_id,
            "market_provider": provider.get("name"),
            "market_details": item.get("details"),
            "home_odds": int(home_moneyline),
            "away_odds": int(away_moneyline),
            "home_decimal_odds": american_to_decimal(int(home_moneyline)),
            "away_decimal_odds": american_to_decimal(int(away_moneyline)),
            "home_raw_implied_prob": american_to_implied(int(home_moneyline)),
            "away_raw_implied_prob": american_to_implied(int(away_moneyline)),
            "market_home_no_vig_prob": no_vig_home_prob(int(home_moneyline), int(away_moneyline)),
            "market_away_no_vig_prob": 1.0 - no_vig_home_prob(int(home_moneyline), int(away_moneyline)),
            "market_spread": numeric(item.get("spread")),
            "market_over_under": numeric(item.get("overUnder")),
            "home_spread_odds": home_spread_odds,
            "away_spread_odds": away_spread_odds,
        }
    return None


def extract_historical_moneylines(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.glob("espn_completed_summary_*.json")):
        row = read_pickcenter_moneyline(path)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["event_id"]).sort_values("event_id").reset_index(drop=True)


def load_or_build_feature_table(args: argparse.Namespace, moneylines: pd.DataFrame | None = None) -> pd.DataFrame:
    feature_path = Path(args.feature_table)
    if feature_path.exists() and not args.rebuild_features:
        table = pd.read_csv(feature_path)
        if not feature_table_has_required_strength(table):
            print("cached NBA feature table is missing new strength features; rebuilding", flush=True)
            args.rebuild_features = True
        else:
            table["event_id"] = table["event_id"].astype(str)
            table["game_date"] = table["game_date"].astype(str)
            table = table.sort_values(["game_date", "event_id"]).reset_index(drop=True)
            return table
    if feature_path.exists() and not args.rebuild_features:
        table = pd.read_csv(feature_path)
    else:
        schedule = load_csv("team_schedule_games.csv")
        team_boxscores = load_csv("team_boxscores.csv")
        player_logs = load_csv("player_game_logs.csv")
        for df in [schedule, team_boxscores, player_logs]:
            if "game_date" in df.columns:
                df["game_date"] = df["game_date"].astype(str)
        table = build_training_table(schedule, team_boxscores, player_logs, args.min_prior_games, moneylines=moneylines)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(feature_path, index=False)

    table["event_id"] = table["event_id"].astype(str)
    table["game_date"] = table["game_date"].astype(str)
    table = table.sort_values(["game_date", "event_id"]).reset_index(drop=True)
    return table


def disabled_filter_diag(df: pd.DataFrame) -> dict[str, Any]:
    return {"enabled": False, "rows_before": int(len(df)), "rows_removed": 0, "rows_after": int(len(df))}


def training_exclusion_ids_for_args(reference_df: pd.DataFrame, args: argparse.Namespace) -> set[str]:
    if args.include_late_regular_training or "event_id" not in reference_df.columns:
        return set()
    cached = getattr(args, "_late_regular_training_exclusion_ids", None)
    if cached is not None:
        return cached
    mask = late_regular_rest_mask(
        reference_df,
        start_month=args.late_regular_start_month,
        start_day=args.late_regular_start_day,
        playoff_teams_only=not args.late_regular_all_teams,
        playoff_reference_df=reference_df,
        tank_win_pct_threshold=args.late_regular_tank_win_pct,
        include_tank_risk=args.late_regular_tank_filter,
    )
    excluded = set(reference_df.loc[mask, "event_id"].astype(str))
    setattr(args, "_late_regular_training_exclusion_ids", excluded)
    return excluded


def filter_training_rows_for_args(
    df: pd.DataFrame,
    args: argparse.Namespace,
    playoff_reference_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.include_late_regular_training:
        return df.copy(), disabled_filter_diag(df)
    if playoff_reference_df is not None and "event_id" in df.columns and "event_id" in playoff_reference_df.columns:
        excluded = training_exclusion_ids_for_args(playoff_reference_df, args)
        mask = df["event_id"].astype(str).isin(excluded)
        filtered = df[~mask].copy().reset_index(drop=True)
        removed = df[mask].copy()
        diagnostics: dict[str, Any] = {
            "enabled": True,
            "rule": "drop late regular-season incentive-risk games involving playoff rest or tank risk",
            "start_month": int(args.late_regular_start_month),
            "start_day": int(args.late_regular_start_day),
            "playoff_teams_only": not args.late_regular_all_teams,
            "include_tank_risk": bool(args.late_regular_tank_filter),
            "tank_win_pct_threshold": args.late_regular_tank_win_pct,
            "rows_before": int(len(df)),
            "rows_removed": int(len(removed)),
            "rows_after": int(len(filtered)),
        }
        if not removed.empty:
            diagnostics["removed_date_range"] = {
                "start": str(removed["game_date"].min()),
                "end": str(removed["game_date"].max()),
            }
        return filtered, diagnostics
    return filter_late_regular_training_rows(
        df,
        start_month=args.late_regular_start_month,
        start_day=args.late_regular_start_day,
        playoff_teams_only=not args.late_regular_all_teams,
        playoff_reference_df=playoff_reference_df,
        tank_win_pct_threshold=args.late_regular_tank_win_pct,
        include_tank_risk=args.late_regular_tank_filter,
    )


def filter_target_rows_for_args(
    df: pd.DataFrame,
    args: argparse.Namespace,
    playoff_reference_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.include_late_regular_targets:
        return df.copy(), disabled_filter_diag(df)
    mask = late_regular_rest_mask(
        df,
        start_month=args.late_regular_start_month,
        start_day=args.late_regular_start_day,
        playoff_teams_only=not args.late_regular_all_teams,
        playoff_reference_df=playoff_reference_df,
        tank_win_pct_threshold=args.late_regular_tank_win_pct,
        include_tank_risk=args.late_regular_tank_filter,
    )
    filtered = df[~mask].copy().reset_index(drop=True)
    removed = df[mask].copy()
    diagnostics = {
        "enabled": True,
        "rule": "drop late regular-season incentive-risk games from backtest targets",
        "start_month": int(args.late_regular_start_month),
        "start_day": int(args.late_regular_start_day),
        "playoff_teams_only": not args.late_regular_all_teams,
        "include_tank_risk": bool(args.late_regular_tank_filter),
        "tank_win_pct_threshold": args.late_regular_tank_win_pct,
        "rows_before": int(len(df)),
        "rows_removed": int(len(removed)),
        "rows_after": int(len(filtered)),
    }
    if not removed.empty:
        diagnostics["removed_date_range"] = {
            "start": str(removed["game_date"].min()),
            "end": str(removed["game_date"].max()),
        }
    return filtered, diagnostics


def training_rows_before_date(feature_table: pd.DataFrame, game_date: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_rows = feature_table[feature_table["game_date"].astype(str) < str(game_date)].copy()
    return filter_training_rows_for_args(train_rows, args, playoff_reference_df=feature_table)


def add_season_end_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["game_date"], errors="coerce")
    out["_season_end_year"] = dates.dt.year + (dates.dt.month >= 10).astype(int)
    return out


def season_tail_sample(candidates: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "season_type" not in candidates.columns:
        raise RuntimeError("Season-tail target selection requires a season_type column.")

    regular = add_season_end_year(candidates)
    regular = regular[pd.to_numeric(regular["season_type"], errors="coerce") == 2].copy()
    if regular.empty:
        raise RuntimeError("No regular-season odds-backed rows available for season-tail target selection.")

    requested_season = args.target_season_end_year
    if requested_season is not None:
        regular = regular[regular["_season_end_year"] == int(requested_season)].copy()
        if regular.empty:
            raise RuntimeError(f"No regular-season rows available for season ending {requested_season}.")

    eligible: list[dict[str, Any]] = []
    for season, season_df in regular.groupby("_season_end_year", sort=True):
        season_df = season_df.sort_values(["game_date", "event_id"]).reset_index(drop=True)
        tail = season_df.tail(args.season_tail_games).copy().reset_index(drop=True)
        if len(tail) >= args.games:
            eligible.append(
                {
                    "season_end_year": int(season),
                    "season_rows": int(len(season_df)),
                    "tail_rows": int(len(tail)),
                    "tail_start_date": str(tail["game_date"].min()),
                    "tail_end_date": str(tail["game_date"].max()),
                    "tail": tail,
                }
            )

    if not eligible:
        raise RuntimeError(
            f"No season has at least {args.games} odds-backed games inside its last "
            f"{args.season_tail_games} regular-season rows."
        )

    rng = np.random.default_rng(args.seed)
    if requested_season is None:
        season_choice = eligible[int(rng.integers(0, len(eligible)))]
    else:
        season_choice = eligible[0]

    tail = season_choice["tail"]
    sample_seed = int(rng.integers(0, 2**31 - 1))
    selected = tail.sample(n=args.games, replace=False, random_state=sample_seed)
    selected = selected.sort_values(["game_date", "event_id"]).drop(columns=["_season_end_year"]).reset_index(drop=True)
    holdout_start = season_choice["tail_start_date"]
    args.holdout_train_cutoff_date = holdout_start

    diagnostics = {
        "mode": "season_tail_sample",
        "regular_season_only": True,
        "season_end_year": season_choice["season_end_year"],
        "eligible_seasons": [item["season_end_year"] for item in eligible],
        "season_rows_after_filters": season_choice["season_rows"],
        "tail_games": int(args.season_tail_games),
        "tail_rows_after_filters": season_choice["tail_rows"],
        "tail_date_range": {
            "start": season_choice["tail_start_date"],
            "end": season_choice["tail_end_date"],
        },
        "sampled_games": int(len(selected)),
        "sample_seed": sample_seed,
        "model_train_cutoff_date": holdout_start,
        "leakage_guard": "single-fit training rows are restricted to games before the first game in the last-N holdout window",
    }
    return selected, diagnostics


def select_target_rows(feature_table: pd.DataFrame, merged: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    pool = merged.dropna(subset=["target_home_win", "home_odds", "away_odds"]).copy()
    if args.start_date:
        pool = pool[pool["game_date"].astype(str) >= args.start_date]
    if args.end_date:
        pool = pool[pool["game_date"].astype(str) <= args.end_date]
    pool, target_filter = filter_target_rows_for_args(pool, args, playoff_reference_df=merged)
    pool = pool.sort_values(["game_date", "event_id"]).reset_index(drop=True)

    candidates = []
    for _, row in pool.iterrows():
        train_rows, _ = training_rows_before_date(feature_table, str(row["game_date"]), args)
        if len(train_rows) >= args.min_train:
            candidates.append(row)
    if not candidates:
        raise RuntimeError("Not enough odds-backed NBA rows after min-train filtering.")

    candidate_df = pd.DataFrame(candidates).reset_index(drop=True)
    if args.target_selection == "season_tail_sample":
        selected, selection_diag = season_tail_sample(candidate_df, args)
    else:
        selected = candidate_df.tail(args.games).reset_index(drop=True)
        selection_diag = {
            "mode": "latest_tail",
            "sampled_games": int(len(selected)),
            "leakage_guard": "training rows are restricted to games before each target date",
        }
    target_filter["target_selection"] = selection_diag
    return selected, target_filter


def predict_frame(model: torch.nn.Module, df: pd.DataFrame, feature_columns: list[str], scaler: dict[str, Any]) -> np.ndarray:
    x = transform_with_scaler(df, feature_columns, scaler)
    return predict_probabilities(model, x, scaler.get("calibration"))


def fit_predict_once(
    feature_table: pd.DataFrame,
    selected: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    first_target_date = str(selected["game_date"].min())
    train_cutoff_date = str(getattr(args, "holdout_train_cutoff_date", None) or first_target_date)
    train_base, training_filter = training_rows_before_date(feature_table, train_cutoff_date, args)
    train_df, val_df = split_train_validation(train_base, args.validation_fraction)
    fit_feature_columns, feature_selection = select_feature_subset(train_df, feature_columns, args)
    print(
        "training one NBA holdout transformer: "
        f"train_rows={len(train_df)} val_rows={len(val_df)} "
        f"target_games={len(selected)} target_start={first_target_date} "
        f"train_cutoff={train_cutoff_date} "
        f"features={len(fit_feature_columns)}/{len(feature_columns)}",
        flush=True,
    )
    model, scaler, diagnostics = train_model(train_df, val_df, selected, fit_feature_columns, args)
    out = selected.copy()
    raw_probs = predict_frame(model, out, fit_feature_columns, scaler)
    out["train_rows"] = len(train_df)
    out["model_home_win_prob_raw"] = raw_probs
    out["model_home_win_prob"] = (
        args.market_blend * out["market_home_no_vig_prob"].astype(float)
        + (1.0 - args.market_blend) * out["model_home_win_prob_raw"].astype(float)
    )
    out["model_away_win_prob"] = 1.0 - out["model_home_win_prob"]
    diagnostics["fit_mode"] = "single"
    diagnostics["training_row_filter"] = training_filter
    diagnostics["feature_selection"] = feature_selection
    diagnostics["feature_count"] = int(len(fit_feature_columns))
    diagnostics["base_feature_count"] = int(len(feature_columns))
    diagnostics["train_cutoff_date"] = train_cutoff_date
    diagnostics["target_start_date"] = first_target_date
    diagnostics["target_end_date"] = str(selected["game_date"].max())
    diagnostics["target_games"] = int(len(selected))
    return out.sort_values(["game_date", "event_id"]).reset_index(drop=True), [diagnostics]


def fit_predict_expanding(
    feature_table: pd.DataFrame,
    selected: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for fit_index, (date, test_df) in enumerate(selected.groupby("game_date", sort=True), start=1):
        train_base, training_filter = training_rows_before_date(feature_table, str(date), args)
        train_df, val_df = split_train_validation(train_base, args.validation_fraction)
        fit_feature_columns, feature_selection = select_feature_subset(train_df, feature_columns, args)
        print(
            f"training NBA transformer for {date}: train_rows={len(train_df)} "
            f"val_rows={len(val_df)} target_games={len(test_df)} "
            f"features={len(fit_feature_columns)}/{len(feature_columns)}",
            flush=True,
        )
        fit_args = argparse.Namespace(**vars(args))
        fit_args.seed = args.seed + fit_index - 1
        model, scaler, diag = train_model(train_df, val_df, test_df, fit_feature_columns, fit_args)
        out = test_df.copy()
        raw_probs = predict_frame(model, out, fit_feature_columns, scaler)
        out["train_rows"] = len(train_df)
        out["model_home_win_prob_raw"] = raw_probs
        out["model_home_win_prob"] = (
            args.market_blend * out["market_home_no_vig_prob"].astype(float)
            + (1.0 - args.market_blend) * out["model_home_win_prob_raw"].astype(float)
        )
        out["model_away_win_prob"] = 1.0 - out["model_home_win_prob"]
        predictions.append(out)
        diag["fit_mode"] = "expanding"
        diag["training_row_filter"] = training_filter
        diag["feature_selection"] = feature_selection
        diag["feature_count"] = int(len(fit_feature_columns))
        diag["base_feature_count"] = int(len(feature_columns))
        diag["target_date"] = str(date)
        diag["target_games"] = int(len(test_df))
        diagnostics.append(diag)
        if args.live_pnl:
            cumulative = pd.concat(predictions, ignore_index=True)
            print_live_pnl(cumulative, out, args, str(date))
        del model, scaler, train_base, train_df, val_df
        gc.collect()

    pred_df = pd.concat(predictions, ignore_index=True)
    return pred_df.sort_values(["game_date", "event_id"]).reset_index(drop=True), diagnostics


def binary_metrics(df: pd.DataFrame, prob_col: str) -> dict[str, Any]:
    clean = df.dropna(subset=["target_home_win", prob_col]).copy()
    if clean.empty:
        return {}
    y = clean["target_home_win"].astype(float).to_numpy()
    p = clean[prob_col].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    ranks = pd.Series(p).rank(method="average").to_numpy()
    positives = y == 1
    negatives = y == 0
    if positives.sum() and negatives.sum():
        auc = float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * negatives.sum()))
    else:
        auc = None
    return {
        "rows": int(len(clean)),
        "home_win_rate": float(y.mean()),
        "avg_probability": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy_0_50": float(((p >= 0.5) == y).mean()),
        "always_home_accuracy": float((y == 1).mean()),
        "auc": auc,
    }


def probability_summary(values: pd.Series) -> dict[str, Any]:
    probs = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(probs) == 0:
        return {}
    quantiles = np.quantile(probs, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    names = ["min", "q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99", "max"]
    return {
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
        **{name: float(value) for name, value in zip(names, quantiles)},
    }


def grouped_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    labels = {2: "regular", 3: "playoffs"}
    for season_type, group in predictions.groupby("season_type"):
        key = labels.get(int(season_type) if not pd.isna(season_type) else -1, str(season_type))
        out[key] = binary_metrics(group, "model_home_win_prob")
    return out


def calibration_bins(predictions: pd.DataFrame, bins: int) -> list[dict[str, Any]]:
    clean = predictions.dropna(subset=["model_home_win_prob", "target_home_win"]).copy()
    if clean.empty:
        return []
    clean["prob_bin"] = pd.cut(clean["model_home_win_prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    rows = []
    for interval, group in clean.groupby("prob_bin", observed=True):
        rows.append(
            {
                "bin": str(interval),
                "rows": int(len(group)),
                "avg_probability": float(group["model_home_win_prob"].mean()),
                "home_win_rate": float(group["target_home_win"].astype(float).mean()),
            }
        )
    return rows


def bet_only_metrics(bets: pd.DataFrame) -> dict[str, Any]:
    placed = bets[bets["bet_side"] != "skip"].dropna(subset=["model_prob", "market_prob", "break_even_prob", "won"]).copy()
    if placed.empty:
        return {}
    y = placed["won"].astype(float).to_numpy()
    model_prob = placed["model_prob"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    market_prob = placed["market_prob"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    break_even = placed["break_even_prob"].astype(float).to_numpy()
    by_side = {}
    for side, group in placed.groupby("bet_side"):
        by_side[str(side)] = {
            "bets": int(len(group)),
            "win_rate": float(group["won"].astype(float).mean()),
            "avg_model_prob": float(group["model_prob"].astype(float).mean()),
            "avg_market_prob": float(group["market_prob"].astype(float).mean()),
            "avg_break_even_prob": float(group["break_even_prob"].astype(float).mean()),
            "profit": float(group["profit"].astype(float).sum()),
            "roi_on_staked": float(group["profit"].sum() / group["stake"].sum()) if float(group["stake"].sum()) > 0 else 0.0,
        }
    return {
        "bets": int(len(placed)),
        "win_rate": float(y.mean()),
        "avg_model_prob": float(model_prob.mean()),
        "avg_market_prob": float(market_prob.mean()),
        "avg_break_even_prob": float(break_even.mean()),
        "avg_model_edge_vs_break_even": float((model_prob - break_even).mean()),
        "avg_model_edge_vs_market": float((model_prob - market_prob).mean()),
        "model_brier_on_bets": float(np.mean((model_prob - y) ** 2)),
        "market_brier_on_bets": float(np.mean((market_prob - y) ** 2)),
        "model_log_loss_on_bets": float(-np.mean(y * np.log(model_prob) + (1 - y) * np.log(1 - model_prob))),
        "market_log_loss_on_bets": float(-np.mean(y * np.log(market_prob) + (1 - y) * np.log(1 - market_prob))),
        "by_side": by_side,
    }


def skip_reason_counts(bets: pd.DataFrame) -> dict[str, int]:
    if "skip_reason" not in bets.columns:
        return {}
    skipped = bets[bets["bet_side"] == "skip"].copy()
    if skipped.empty:
        return {}
    return {str(reason): int(count) for reason, count in skipped["skip_reason"].fillna("unknown").value_counts().sort_index().items()}


def simulate_kelly(
    predictions: pd.DataFrame,
    starting_bankroll: float,
    kelly_multiplier: float,
    max_bet_fraction: float,
    min_edge: float,
    max_market_disagreement: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bankroll = float(starting_bankroll)
    bets: list[dict[str, Any]] = []
    for _, game in predictions.sort_values(["game_date", "event_id"]).iterrows():
        home_decimal = float(game["home_decimal_odds"])
        away_decimal = float(game["away_decimal_odds"])
        home_model_prob = float(game["model_home_win_prob"])
        away_model_prob = float(game["model_away_win_prob"])
        home_market_prob = float(game["market_home_no_vig_prob"])
        away_market_prob = 1.0 - home_market_prob
        home_kelly = kelly_fraction(home_model_prob, home_decimal)
        away_kelly = kelly_fraction(away_model_prob, away_decimal)
        skip_reason = None

        candidates = {
            "home": {
                "team": game["home_team"],
                "offered_odds": int(game["home_odds"]),
                "decimal_odds": home_decimal,
                "model_prob": home_model_prob,
                "market_prob": home_market_prob,
                "break_even_prob": 1.0 / home_decimal,
                "kelly": home_kelly,
                "won": int(game["target_home_win"]) == 1,
            },
            "away": {
                "team": game["away_team"],
                "offered_odds": int(game["away_odds"]),
                "decimal_odds": away_decimal,
                "model_prob": away_model_prob,
                "market_prob": away_market_prob,
                "break_even_prob": 1.0 / away_decimal,
                "kelly": away_kelly,
                "won": int(game["target_home_win"]) == 0,
            },
        }
        if home_kelly <= 0 and away_kelly <= 0:
            candidate_side = max(
                candidates,
                key=lambda candidate: candidates[candidate]["model_prob"] - candidates[candidate]["break_even_prob"],
            )
            skip_reason = "no_positive_kelly"
        else:
            candidate_side = "home" if home_kelly >= away_kelly else "away"
            candidate = candidates[candidate_side]
            candidate_edge = float(candidate["model_prob"]) - float(candidate["break_even_prob"])
            candidate_market_gap = abs(float(candidate["model_prob"]) - float(candidate["market_prob"]))
            if candidate_edge < min_edge:
                skip_reason = "below_min_edge"
            elif max_market_disagreement is not None and candidate_market_gap > max_market_disagreement:
                skip_reason = "market_disagreement"

        candidate = candidates[candidate_side]
        side = "skip" if skip_reason else candidate_side
        raw_fraction = float(candidate["kelly"])
        decimal_odds = float(candidate["decimal_odds"])
        offered_odds = int(candidate["offered_odds"])
        model_prob = float(candidate["model_prob"])
        market_prob = float(candidate["market_prob"])
        break_even_prob = float(candidate["break_even_prob"])
        candidate_won = bool(candidate["won"])

        if side == "skip":
            stake_fraction = 0.0
            stake = 0.0
            profit = 0.0
            won = None
        else:
            stake_fraction = min(raw_fraction * kelly_multiplier, max_bet_fraction)
            stake = bankroll * stake_fraction
            won = candidate_won
            profit = stake * (decimal_odds - 1.0) if won else -stake

        bankroll += profit
        bets.append(
            {
                "event_id": game["event_id"],
                "game_date": game["game_date"],
                "season_type": game.get("season_type"),
                "away_team": game["away_team"],
                "home_team": game["home_team"],
                "target_home_win": game["target_home_win"],
                "bet_side": side,
                "skip_reason": skip_reason,
                "candidate_side": candidate_side,
                "candidate_team": candidate["team"],
                "candidate_won": candidate_won,
                "bet_team": game["home_team"] if side == "home" else game["away_team"] if side == "away" else None,
                "offered_american_odds": offered_odds,
                "decimal_odds": decimal_odds,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "break_even_prob": break_even_prob,
                "model_edge_vs_break_even": model_prob - break_even_prob if model_prob is not None else None,
                "model_edge_vs_market": model_prob - market_prob if model_prob is not None else None,
                "market_home_no_vig_prob": game["market_home_no_vig_prob"],
                "raw_kelly_fraction": raw_fraction,
                "stake_fraction": stake_fraction,
                "stake": stake,
                "won": won,
                "profit": profit,
                "bankroll_after": bankroll,
            }
        )

    bets_df = pd.DataFrame(bets)
    placed = bets_df[bets_df["bet_side"] != "skip"].copy()
    total_staked = float(placed["stake"].sum()) if not placed.empty else 0.0
    total_profit = float(bets_df["profit"].sum()) if not bets_df.empty else 0.0
    return bets_df, {
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
        "max_drawdown": max_drawdown(bets_df["bankroll_after"].tolist(), starting_bankroll),
    }


def score_text(away_score: Any, home_score: Any) -> str:
    away = numeric(away_score)
    home = numeric(home_score)
    if away is None or home is None:
        return ""
    return f"{int(away)}-{int(home)}"


def build_readable_bet_history(bets: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "game_date",
        "event_id",
        "matchup",
        "final_score",
        "bet_side",
        "candidate_side",
        "candidate_team",
        "bet_team",
        "actual_winner",
        "result",
        "skip_reason",
        "model_prob_pct",
        "market_no_vig_prob_pct",
        "break_even_prob_pct",
        "model_edge_vs_break_even_pct",
        "model_edge_vs_market_pct",
        "american_odds",
        "decimal_odds",
        "raw_kelly_fraction_pct",
        "stake_fraction_pct",
        "stake",
        "profit",
        "bankroll_after",
    ]
    history_rows = bets.copy()
    if history_rows.empty:
        return pd.DataFrame(columns=columns)

    try:
        schedule = load_csv("team_schedule_games.csv")
    except FileNotFoundError:
        schedule = pd.DataFrame()

    history_rows["event_id"] = history_rows["event_id"].astype(str)
    if not schedule.empty and {"event_id", "away_score", "home_score"}.issubset(schedule.columns):
        scores = schedule[["event_id", "away_score", "home_score"]].copy()
        scores["event_id"] = scores["event_id"].astype(str)
        scores = scores.drop_duplicates(subset=["event_id"])
        history_rows = history_rows.merge(scores, on="event_id", how="left")
    else:
        history_rows["away_score"] = np.nan
        history_rows["home_score"] = np.nan

    home_won = pd.to_numeric(history_rows["target_home_win"], errors="coerce").fillna(0).astype(int) == 1
    placed_mask = history_rows["bet_side"].astype(str) != "skip"
    won_mask = history_rows["won"].fillna(False).astype(bool)
    candidate_side = history_rows["candidate_side"] if "candidate_side" in history_rows.columns else history_rows["bet_side"]
    candidate_team = history_rows["candidate_team"] if "candidate_team" in history_rows.columns else history_rows["bet_team"]
    history = pd.DataFrame(
        {
            "game_date": history_rows["game_date"].astype(str),
            "event_id": history_rows["event_id"].astype(str),
            "matchup": history_rows["away_team"].astype(str) + " at " + history_rows["home_team"].astype(str),
            "final_score": [score_text(away, home) for away, home in zip(history_rows["away_score"], history_rows["home_score"])],
            "bet_side": history_rows["bet_side"],
            "candidate_side": candidate_side,
            "candidate_team": candidate_team,
            "bet_team": history_rows["bet_team"],
            "actual_winner": np.where(home_won, history_rows["home_team"], history_rows["away_team"]),
            "result": np.where(placed_mask, np.where(won_mask, "WIN", "LOSS"), "SKIP"),
            "skip_reason": history_rows["skip_reason"].fillna(""),
            "model_prob_pct": (pd.to_numeric(history_rows["model_prob"], errors="coerce") * 100).round(2),
            "market_no_vig_prob_pct": (pd.to_numeric(history_rows["market_prob"], errors="coerce") * 100).round(2),
            "break_even_prob_pct": (pd.to_numeric(history_rows["break_even_prob"], errors="coerce") * 100).round(2),
            "model_edge_vs_break_even_pct": (pd.to_numeric(history_rows["model_edge_vs_break_even"], errors="coerce") * 100).round(2),
            "model_edge_vs_market_pct": (pd.to_numeric(history_rows["model_edge_vs_market"], errors="coerce") * 100).round(2),
            "american_odds": pd.to_numeric(history_rows["offered_american_odds"], errors="coerce").astype("Int64"),
            "decimal_odds": pd.to_numeric(history_rows["decimal_odds"], errors="coerce").round(3),
            "raw_kelly_fraction_pct": (pd.to_numeric(history_rows["raw_kelly_fraction"], errors="coerce") * 100).round(2),
            "stake_fraction_pct": (pd.to_numeric(history_rows["stake_fraction"], errors="coerce") * 100).round(2),
            "stake": pd.to_numeric(history_rows["stake"], errors="coerce").round(2),
            "profit": pd.to_numeric(history_rows["profit"], errors="coerce").round(2),
            "bankroll_after": pd.to_numeric(history_rows["bankroll_after"], errors="coerce").round(2),
        }
    )
    return history[columns]


def print_live_pnl(
    cumulative_predictions: pd.DataFrame,
    latest_predictions: pd.DataFrame,
    args: argparse.Namespace,
    label: str,
) -> None:
    if not args.live_pnl or cumulative_predictions.empty:
        return
    bets, summary = simulate_kelly(
        cumulative_predictions,
        starting_bankroll=args.bankroll,
        kelly_multiplier=args.kelly_multiplier,
        max_bet_fraction=args.max_bet_fraction,
        min_edge=args.min_edge,
        max_market_disagreement=None if args.max_market_disagreement < 0 else args.max_market_disagreement,
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


def bootstrap_paths(
    bets: pd.DataFrame,
    starting_bankroll: float,
    trials: int,
    games: int,
    seed: int,
) -> dict[str, Any]:
    placed = bets[bets["bet_side"] != "skip"].copy()
    if placed.empty:
        return {}
    rng = np.random.default_rng(seed)
    final_bankrolls = []
    for _ in range(trials):
        bankroll = float(starting_bankroll)
        sample = placed.sample(n=games, replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
        for _, row in sample.iterrows():
            stake = bankroll * float(row["stake_fraction"])
            profit = stake * (float(row["decimal_odds"]) - 1.0) if bool(row["won"]) else -stake
            bankroll += profit
        final_bankrolls.append(bankroll)
    values = np.asarray(final_bankrolls, dtype=float)
    return {
        "trials": int(trials),
        "sampled_games_per_trial": int(games),
        "mean_final_bankroll": float(values.mean()),
        "median_final_bankroll": float(np.median(values)),
        "p05_final_bankroll": float(np.percentile(values, 5)),
        "p95_final_bankroll": float(np.percentile(values, 95)),
        "probability_profit": float((values > starting_bankroll).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--min-train", type=int, default=1500)
    parser.add_argument("--kelly-multiplier", type=float, default=0.25)
    parser.add_argument("--max-bet-fraction", type=float, default=0.03)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument(
        "--max-market-disagreement",
        type=float,
        default=0.25,
        help="Skip bets where abs(model bet-side probability - no-vig market probability) exceeds this value. Use a negative value to disable.",
    )
    parser.add_argument("--market-blend", type=float, default=0.0)
    parser.add_argument("--fit-mode", choices=["single", "expanding"], default="expanding")
    parser.add_argument("--feature-table", default=str(DEFAULT_FEATURE_TABLE))
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--raw-dir")
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
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--target-selection",
        choices=["latest_tail", "season_tail_sample"],
        default="latest_tail",
        help="Choose the evaluated target set. season_tail_sample samples --games rows from a season's last --season-tail-games regular-season games.",
    )
    parser.add_argument("--season-tail-games", type=int, default=230)
    parser.add_argument(
        "--target-season-end-year",
        type=int,
        help="NBA season end year for season_tail_sample, e.g. 2026 for the 2025-26 season. If omitted, one eligible season is chosen by --seed.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--no-init-output-bias", dest="init_output_bias", action="store_false")
    parser.add_argument("--class-balance", choices=["none", "pos_weight"], default="none")
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
    parser.add_argument(
        "--include-late-regular-targets",
        action="store_true",
        help="Allow late regular-season games in the evaluated backtest target pool.",
    )
    parser.add_argument("--late-regular-start-month", type=int, default=4)
    parser.add_argument("--late-regular-start-day", type=int, default=1)
    parser.add_argument(
        "--late-regular-all-teams",
        action="store_true",
        help="Filter all late regular-season games instead of only games involving playoff teams.",
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
    parser.add_argument("--run-id", help="Stable id used for archived backtest artifact filenames.")
    parser.add_argument("--run-label", help="Human-readable label shown by the bet-history web app.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-live-pnl", dest="live_pnl", action="store_false")
    args = parser.parse_args()

    ensure_dirs()
    raw_dir = Path(args.raw_dir) if args.raw_dir else latest_raw_dir()
    odds = extract_historical_moneylines(raw_dir)
    if odds.empty:
        raise RuntimeError(f"No historical moneylines parsed from {raw_dir}")
    odds.to_csv(BACKTEST_DIR / "nba_historical_moneylines.csv", index=False)
    print(f"historical moneyline rows={len(odds)} raw_dir={raw_dir}", flush=True)

    print(f"loading NBA features from {args.feature_table}", flush=True)
    feature_table = load_or_build_feature_table(args, moneylines=odds)
    feature_columns = selected_feature_columns(feature_table, args.feature_mode)
    pre_blacklist_feature_count = len(feature_columns)
    feature_columns, feature_blacklist = filter_model_feature_columns(feature_columns, args)
    print(
        f"feature rows={len(feature_table)} features={len(feature_columns)} "
        f"mode={args.feature_mode} no_market={args.no_market_features}",
        flush=True,
    )

    merged = feature_table.merge(odds, on="event_id", how="left")
    merged.to_csv(BACKTEST_DIR / "nba_transformer_feature_table_with_odds.csv", index=False)
    selected, target_filter = select_target_rows(feature_table, merged, args)
    print(
        "Late regular-season target filter: "
        f"removed={target_filter['rows_removed']} "
        f"rows_after={target_filter['rows_after']} "
        f"cutoff={args.late_regular_start_month:02d}-{args.late_regular_start_day:02d} "
        f"playoff_teams_only={not args.late_regular_all_teams} "
        f"tank_filter={args.late_regular_tank_filter} "
        f"tank_win_pct<={args.late_regular_tank_win_pct:.3f}",
        flush=True,
    )
    print(
        f"selected target rows={len(selected)} "
        f"date_range={selected['game_date'].min()}..{selected['game_date'].max()} "
        f"odds_coverage={merged['home_odds'].notna().sum()}/{len(merged)}",
        flush=True,
    )

    if args.fit_mode == "expanding":
        predictions, diagnostics = fit_predict_expanding(feature_table, selected, feature_columns, args)
    else:
        predictions, diagnostics = fit_predict_once(feature_table, selected, feature_columns, args)
        if args.live_pnl:
            cumulative: list[pd.DataFrame] = []
            for date, date_df in predictions.groupby("game_date", sort=True):
                cumulative.append(date_df)
                cumulative_df = pd.concat(cumulative, ignore_index=True)
                print_live_pnl(cumulative_df, date_df, args, str(date))

    effective_feature_counts = [int(diag.get("feature_count", len(feature_columns))) for diag in diagnostics]
    effective_feature_count = int(max(effective_feature_counts)) if effective_feature_counts else len(feature_columns)

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = safe_run_id(args.run_id or f"{run_timestamp}_{args.model_type}_{args.feature_mode}_{args.fit_mode}")
    run_label = args.run_label or f"{args.model_type.title()} {args.feature_mode} {args.fit_mode} {run_timestamp}"

    latest_predictions_path = BACKTEST_DIR / "nba_transformer_backtest_predictions.csv"
    archived_predictions_path = BACKTEST_DIR / f"nba_transformer_backtest_predictions_{run_id}.csv"
    latest_bets_path = BACKTEST_DIR / "nba_transformer_backtest_bets.csv"
    archived_bets_path = BACKTEST_DIR / f"nba_transformer_backtest_bets_{run_id}.csv"
    latest_history_path = BACKTEST_DIR / "nba_transformer_bet_history_clean.csv"
    archived_history_path = BACKTEST_DIR / f"nba_transformer_bet_history_clean_{run_id}.csv"
    latest_summary_path = BACKTEST_DIR / "nba_transformer_backtest_summary.json"
    archived_summary_path = BACKTEST_DIR / f"nba_transformer_backtest_summary_{run_id}.json"

    predictions.to_csv(latest_predictions_path, index=False)
    predictions.to_csv(archived_predictions_path, index=False)
    bets, kelly_summary = simulate_kelly(
        predictions,
        starting_bankroll=args.bankroll,
        kelly_multiplier=args.kelly_multiplier,
        max_bet_fraction=args.max_bet_fraction,
        min_edge=args.min_edge,
        max_market_disagreement=None if args.max_market_disagreement < 0 else args.max_market_disagreement,
    )
    bets.to_csv(latest_bets_path, index=False)
    bets.to_csv(archived_bets_path, index=False)
    bet_history = build_readable_bet_history(bets)
    bet_history.to_csv(latest_history_path, index=False)
    bet_history.to_csv(archived_history_path, index=False)
    print(f"wrote readable bet history rows={len(bet_history)} path={latest_history_path}", flush=True)
    print(f"archived backtest run_id={run_id} bets={archived_bets_path}", flush=True)

    summary = {
        "created_at": created_at,
        "run_id": run_id,
        "label": run_label,
        "raw_dir": str(raw_dir),
        "fit_mode": args.fit_mode,
        "model_type": args.model_type,
        "feature_mode": args.feature_mode,
        "feature_select": args.feature_select,
        "settings": {
            "games": args.games,
            "bankroll": args.bankroll,
            "min_train": args.min_train,
            "kelly_multiplier": args.kelly_multiplier,
            "max_bet_fraction": args.max_bet_fraction,
            "min_edge": args.min_edge,
            "max_market_disagreement": args.max_market_disagreement,
            "market_blend": args.market_blend,
            "fit_mode": args.fit_mode,
            "feature_mode": args.feature_mode,
            "feature_select": args.feature_select,
            "no_market_features": args.no_market_features,
            "max_features": args.max_features,
            "model_type": args.model_type,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "target_selection": args.target_selection,
            "season_tail_games": args.season_tail_games,
            "target_season_end_year": args.target_season_end_year,
            "holdout_train_cutoff_date": getattr(args, "holdout_train_cutoff_date", None),
            "seed": args.seed,
        },
        "artifact_paths": {
            "latest_predictions": str(latest_predictions_path),
            "latest_bets": str(latest_bets_path),
            "latest_bet_history": str(latest_history_path),
            "latest_summary": str(latest_summary_path),
            "archived_predictions": str(archived_predictions_path),
            "archived_bets": str(archived_bets_path),
            "archived_bet_history": str(archived_history_path),
            "archived_summary": str(archived_summary_path),
        },
        "base_feature_count": len(feature_columns),
        "pre_blacklist_feature_count": int(pre_blacklist_feature_count),
        "feature_blacklist": feature_blacklist,
        "feature_count": effective_feature_count,
        "feature_rows": int(len(feature_table)),
        "feature_date_range": {
            "start": str(feature_table["game_date"].min()),
            "end": str(feature_table["game_date"].max()),
        },
        "historical_moneyline_rows": int(len(odds)),
        "odds_coverage_rows": int(merged["home_odds"].notna().sum()),
        "selected_games": int(len(predictions)),
        "selected_date_range": {
            "start": str(predictions["game_date"].min()),
            "end": str(predictions["game_date"].max()),
        },
        "late_regular_target_filter": target_filter,
        "model_metrics": binary_metrics(predictions, "model_home_win_prob"),
        "market_metrics": binary_metrics(predictions, "market_home_no_vig_prob"),
        "model_probability_distribution": probability_summary(predictions["model_home_win_prob"]),
        "raw_model_probability_distribution": probability_summary(predictions["model_home_win_prob_raw"]),
        "by_season_type": grouped_metrics(predictions),
        "calibration_bins": calibration_bins(predictions, bins=10),
        "kelly": kelly_summary,
        "bet_only_metrics": bet_only_metrics(bets),
        "skip_reason_counts": skip_reason_counts(bets),
        "bet_history_path": str(latest_history_path),
        "archived_bet_history_path": str(archived_history_path),
        "bootstrap": bootstrap_paths(
            bets,
            starting_bankroll=args.bankroll,
            trials=1000,
            games=min(int(kelly_summary["bets_placed"]), max(len(bets), 1)),
            seed=args.seed,
        ),
        "training_diagnostics": diagnostics,
        "notes": [
            "Model features are built from games before each target date.",
            "Grouped feature selection is fit on the training fold only when enabled.",
            "Platt calibration is applied only when validation AUC meets the calibration threshold.",
            "Moneyline odds are parsed from ESPN completed summary pickcenter blocks when available.",
            "Odds columns are used for betting simulation and market comparison, not as transformer input features.",
        "The default fit mode retrains date-by-date; use --fit-mode single only for a one-model holdout run.",
        ],
    }
    latest_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    archived_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
