from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import bet_history_app as base_app
from kelly_moneyline_backtest import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parent
BACKTEST_DIR = ROOT / "data" / "backtests"
PREDICTION_SUMMARY = ROOT / "data" / "predictions" / "nhl_tonight_prediction_summary.json"
ENRICHED_DEFAULT_CSV = BACKTEST_DIR / "nhl_kelly_backtest_bets_schedule_context_away_value.csv"
DEFAULT_CSV = ENRICHED_DEFAULT_CSV if ENRICHED_DEFAULT_CSV.exists() else BACKTEST_DIR / "nhl_kelly_backtest_bets_latest.csv"
BET_PREFIX = "nhl_kelly_backtest_bets"

HTML = (
    base_app.HTML.replace("NBA Backtest Bet History", "NHL Moneyline Backtest")
    .replace("Model Runs", "Model Snapshot")
    .replace("No saved transformer checkpoints found.", "No saved NHL model summary found.")
)


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def run_id_for_path(path: Path | None) -> str:
    if path is None:
        return ""
    stem = path.stem
    if stem == BET_PREFIX:
        return "latest"
    if stem.startswith(BET_PREFIX + "_"):
        return stem[len(BET_PREFIX) + 1 :]
    return stem


def summary_path_for_bets(path: Path) -> Path:
    return BACKTEST_DIR / f"nhl_kelly_backtest_summary_{run_id_for_path(path)}.json"


def load_json_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def label_for_bet_run(run_id: str, meta: dict[str, object]) -> str:
    label = meta.get("label")
    if isinstance(label, str) and label:
        return label
    if run_id == "latest":
        return "Latest NHL model default Kelly"
    if run_id == "quarter_kelly":
        return "Latest NHL model quarter Kelly"
    return run_id.replace("_", " ").title()


def bet_run_paths() -> dict[str, Path]:
    paths = sorted(BACKTEST_DIR.glob(f"{BET_PREFIX}*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
    return {run_id_for_path(path): path for path in paths}


def discover_bet_runs(default_path: Path | None = None) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    default_id = run_id_for_path(default_path)
    for run_id, path in bet_run_paths().items():
        rows = base_app.load_rows(path)
        summary = base_app.summarize(rows)
        meta = load_json_file(summary_path_for_bets(path))
        dates = [str(row.get("game_date") or "") for row in rows if row.get("game_date")]
        settings = meta.get("settings") if isinstance(meta.get("settings"), dict) else {}
        runs.append(
            {
                "id": run_id,
                "label": label_for_bet_run(run_id, meta),
                "path": relative_path(path),
                "summary_path": relative_path(summary_path_for_bets(path)) if summary_path_for_bets(path).exists() else "",
                "is_default": run_id == default_id,
                "created_at": meta.get("created_at") or path.stat().st_mtime,
                "date_start": min(dates) if dates else "",
                "date_end": max(dates) if dates else "",
                "settings": settings,
                "summary": summary,
            }
        )
    runs.sort(key=lambda run: (not bool(run.get("is_default")), str(run.get("label"))))
    return runs


def resolve_bet_run_csv(run_id: str | None, default_path: Path) -> Path:
    paths = bet_run_paths()
    if run_id and run_id in paths:
        return paths[run_id]
    return default_path


def discover_model_runs() -> list[dict[str, object]]:
    data = load_json_file(PREDICTION_SUMMARY)
    if not data:
        return []
    metrics = data.get("backtest_metrics") if isinstance(data.get("backtest_metrics"), dict) else {}
    prediction = data.get("prediction") if isinstance(data.get("prediction"), dict) else {}
    return [
        {
            "id": f"{data.get('away_team', '')} at {data.get('home_team', '')} / {data.get('game_date', '')}",
            "file": relative_path(PREDICTION_SUMMARY),
            "created_at": data.get("created_at") or "",
            "feature_count": len(FEATURE_COLUMNS),
            "feature_mode": "rolling moneyline",
            "model_type": data.get("model_type", ""),
            "epochs": "",
            "batch_size": "",
            "train_rows": data.get("training_rows", ""),
            "validation_rows": metrics.get("rows", ""),
            "best_epoch": "",
            "best_validation_loss": base_app.rounded_value(metrics.get("log_loss"), 4),
            "validation_auc": "",
            "validation_accuracy": base_app.rounded_value(metrics.get("accuracy_0_50"), 4),
            "predicted_winner": prediction.get("predicted_winner", ""),
        }
    ]


class NHLBetHistoryHandler(BaseHTTPRequestHandler):
    csv_path: Path = DEFAULT_CSV

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path in {"/", "/index.html"}:
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/runs":
            payload = {
                "default_run_id": run_id_for_path(self.csv_path),
                "bet_runs": discover_bet_runs(self.csv_path),
                "model_runs": discover_model_runs(),
            }
            self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            return

        if parsed.path == "/api/bets":
            run_id = qs.get("run", [None])[0]
            csv_path = resolve_bet_run_csv(run_id, self.csv_path)
            rows = base_app.load_rows(csv_path)
            run = next((item for item in discover_bet_runs(self.csv_path) if item["id"] == run_id_for_path(csv_path)), {})
            payload = {
                "path": relative_path(csv_path),
                "run": run,
                "summary": base_app.summarize(rows),
                "rows": rows,
            }
            self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            return

        if parsed.path == "/api/export.csv":
            run_id = qs.get("run", [None])[0]
            csv_path = resolve_bet_run_csv(run_id, self.csv_path)
            if not csv_path.exists():
                self.send_bytes(b"missing bet history csv\n", "text/plain; charset=utf-8", status=404)
                return
            body = csv_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{csv_path.name}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_bytes(b"not found\n", "text/plain; charset=utf-8", status=404)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    args = parser.parse_args()

    NHLBetHistoryHandler.csv_path = Path(args.csv).resolve()
    server = ThreadingHTTPServer((args.host, args.port), NHLBetHistoryHandler)
    print(f"NHL bet history app running at http://{args.host}:{args.port}", flush=True)
    print(f"Reading {NHLBetHistoryHandler.csv_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
