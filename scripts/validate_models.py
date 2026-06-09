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
from src.models.game_model import (
    save_model,
    save_xgb_margin_model,
    save_xgb_model,
    train,
    train_xgb,
    train_xgb_margin,
    walk_forward_backtest,
)
from src.models.meta_model import save_meta_model, train_meta_model
from src.models.uncertainty import (
    fit_empirical_probability_uncertainty,
    save_empirical_probability_uncertainty,
)


def main() -> None:
    logs = fetch_historical_playoff_logs()
    ratings = fetch_historical_team_ratings()
    from src.data.fetch_injury_proxy import _load_cache
    injury_cache = _load_cache()
    if not injury_cache.get("_seasons_fetched"):
        print("No injury cache found — run: python -m src.data.fetch_injury_proxy")
        print("Training without injury features.")
        injury_cache = None
    from src.data.fetch_clutch_stats import _load_cache as _load_clutch
    clutch_cache = _load_clutch()
    if not clutch_cache:
        print("No clutch cache found — run: python -m src.data.fetch_clutch_stats")
        print("Training without clutch features.")
        clutch_cache = None
    rows = build_canonical_pregame_rows(
        logs, ratings, injury_cache=injury_cache, clutch_cache=clutch_cache
    )
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

    print("Training XGBoost classifier with injury features...")
    xgb_model = train_xgb(rows, holdout_seasons=[])
    xgb_model["dataset_sha256"] = dataset_metadata["sha256"]
    save_xgb_model(xgb_model)

    print("Training XGBoost margin (spread) model...")
    margin_model = train_xgb_margin(rows, holdout_seasons=[])
    margin_model["dataset_sha256"] = dataset_metadata["sha256"]
    save_xgb_margin_model(margin_model)
    print(f"  Margin model — MAE: {margin_model['margin_mae']:.2f} pts  "
          f"residual_std: {margin_model['residual_std']:.2f} pts")

    print(f"Canonical dataset: {dataset_metadata['games']} games")
    print(json.dumps(report["overall"], indent=2))
    print(f"Meta-model rows: {meta_bundle['training_rows']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
