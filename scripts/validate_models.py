"""Build canonical data, run walk-forward validation, and fit production models."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_historical_dataset import (
    DEFAULT_OUTPUT_PATH,
    build_canonical_pregame_rows,
    save_canonical_dataset,
)
from src.data.fetch_current_stats import (
    fetch_historical_playoff_logs,
    fetch_historical_team_ratings,
)
from src.models.game_model import save_model, train, walk_forward_backtest
from src.models.meta_model import save_meta_model, train_meta_model
from src.models.uncertainty import (
    fit_empirical_probability_uncertainty,
    save_empirical_probability_uncertainty,
)


def main() -> None:
    logs = fetch_historical_playoff_logs()
    ratings = fetch_historical_team_ratings()
    rows = build_canonical_pregame_rows(logs, ratings)
    dataset_metadata = save_canonical_dataset(rows)

    report = walk_forward_backtest(rows)
    oof_rows = report.pop("oof_predictions")
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "walk_forward_backtest.json"
    report_path.write_text(
        json.dumps({**report, "dataset": dataset_metadata}, indent=2),
        encoding="utf-8",
    )

    oof_path = DEFAULT_OUTPUT_PATH.parent / "oof_predictions.csv"
    with oof_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(oof_rows[0]))
        writer.writeheader()
        writer.writerows(oof_rows)

    meta_bundle = train_meta_model(oof_rows)
    meta_bundle["dataset_sha256"] = dataset_metadata["sha256"]
    save_meta_model(meta_bundle)
    save_empirical_probability_uncertainty(
        fit_empirical_probability_uncertainty(oof_rows)
    )

    final_game_model = train(rows, holdout_seasons=[])
    final_game_model["validation_metrics"] = report.get("overall", {}).get("model", {})
    final_game_model["dataset_sha256"] = dataset_metadata["sha256"]
    save_model(final_game_model)

    print(f"Canonical dataset: {dataset_metadata['games']} games")
    print(json.dumps(report["overall"], indent=2))
    print(f"Meta-model rows: {meta_bundle['training_rows']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
