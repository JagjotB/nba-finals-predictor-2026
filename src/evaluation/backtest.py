"""Rolling playoff backtests and calibration reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

from src.evaluation.calibration import compare_binary_and_confidence_calibration
from src.evaluation.metrics import (
    evaluate_game_predictions,
    normalize_prediction_rows,
    series_winner_accuracy,
    upset_detection_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKTEST_SPLITS = [
    {"train_start": 2014, "train_end": 2021, "test_season": 2022},
    {"train_start": 2014, "train_end": 2022, "test_season": 2023},
    {"train_start": 2014, "train_end": 2023, "test_season": 2024},
    {"train_start": 2014, "train_end": 2024, "test_season": 2025},
]


def _iter_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return [dict(row) for row in data.to_dict(orient="records")]
    if isinstance(data, dict):
        if "rows" in data and isinstance(data["rows"], list):
            return [dict(row) for row in data["rows"]]
        return [dict(data)]
    return [dict(row) for row in data]


def load_backtest_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load historical playoff prediction rows from CSV or JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing backtest data file: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    if path.suffix.lower() in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as file:
            if path.suffix.lower() == ".jsonl":
                return [json.loads(line) for line in file if line.strip()]
            loaded = json.load(file)
        return _iter_records(loaded)
    raise ValueError("Backtest data must be CSV, JSON, or JSONL.")


def season_end_year(season: Any) -> int | None:
    """Convert NBA season labels such as 2021-22 or 2022 into end year."""
    if season in (None, ""):
        return None
    text = str(season).strip()
    if "-" not in text:
        try:
            return int(float(text))
        except ValueError:
            return None

    start_text, end_text = text.split("-", 1)
    try:
        start_year = int(float(start_text))
    except ValueError:
        return None
    end_text = end_text.strip()
    if len(end_text) == 2:
        century = start_year // 100 * 100
        end_year = century + int(end_text)
        if end_year < start_year:
            end_year += 100
        return end_year
    try:
        return int(float(end_text))
    except ValueError:
        return None


def _row_season(row: dict[str, Any]) -> int | None:
    return season_end_year(row.get("season") or row.get("SEASON") or row.get("year") or row.get("test_season"))


def default_playoff_splits() -> list[dict[str, int]]:
    """Return the requested rolling playoff backtest splits."""
    return [dict(split) for split in DEFAULT_BACKTEST_SPLITS]


def split_rows(
    rows: Any,
    split: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train and test sets by NBA playoff season end year."""
    records = _iter_records(rows)
    train_rows = []
    test_rows = []
    for row in records:
        season = _row_season(row)
        if season is None:
            continue
        if split["train_start"] <= season <= split["train_end"]:
            train_rows.append(row)
        elif season == split["test_season"]:
            test_rows.append(row)
    return train_rows, test_rows


def _actual_team_a_win(row: dict[str, Any]) -> int | None:
    for key in ("actual_team_a_win", "team_a_actual_result", "win", "target"):
        if key not in row or row[key] in (None, ""):
            continue
        text = str(row[key]).strip().lower()
        if text in {"1", "true", "w", "win", "won"}:
            return 1
        if text in {"0", "false", "l", "loss", "lost"}:
            return 0
    actual_winner = str(row.get("actual_winner") or row.get("winner") or row.get("winning_team") or "").strip()
    team_a = str(row.get("team_a") or row.get("team") or "").strip()
    team_b = str(row.get("team_b") or row.get("opponent") or "").strip()
    if actual_winner and team_a and actual_winner == team_a:
        return 1
    if actual_winner and team_b and actual_winner == team_b:
        return 0
    score_a = row.get("actual_score_team_a") or row.get("team_a_score") or row.get("team_score")
    score_b = row.get("actual_score_team_b") or row.get("team_b_score") or row.get("opponent_score")
    if score_a not in (None, "") and score_b not in (None, ""):
        try:
            return int(float(score_a) > float(score_b))
        except ValueError:
            return None
    return None


def _as_team_model_row(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt game-centric playoff rows to the Phase 15 team-model shape."""
    adapted = dict(row)
    if "team" not in adapted and "team_a" in adapted:
        adapted["team"] = adapted.get("team_a")
    if "opponent" not in adapted and "team_b" in adapted:
        adapted["opponent"] = adapted.get("team_b")
    if "win" not in adapted and "target" not in adapted:
        actual = _actual_team_a_win(adapted)
        if actual is not None:
            adapted["win"] = actual
    return adapted


def _team_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_as_team_model_row(row) for row in rows]


def _transparent_baseline_predictions(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from src.models.train_team_model import predict_baseline_without_training

    model_rows = _team_model_rows(test_rows)
    predictions = predict_baseline_without_training(model_rows)
    output = []
    for row, prediction in zip(test_rows, predictions):
        merged = dict(row)
        if "team_a" in merged:
            merged["team_a_win_probability"] = prediction["baseline_win_probability"]
        else:
            merged["win_probability"] = prediction["baseline_win_probability"]
        return_target = row.get("actual_win") or row.get("win") or row.get("target")
        if return_target is not None:
            merged["actual_win"] = return_target
        output.append(merged)
    return output


def train_and_predict_baseline(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    model_types: list[str] | None = None,
    fallback_to_transparent: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train the Phase 15 baseline model for one split and predict the test rows."""
    metadata: dict[str, Any] = {"mode": "trained_baseline"}
    try:
        from src.models.train_team_model import train_team_model, predict_team_win_probability

        train_model_rows = _team_model_rows(train_rows)
        test_model_rows = _team_model_rows(test_rows)
        bundle = train_team_model(
            train_model_rows,
            model_types=model_types or ["logistic_regression", "random_forest"],
            calibrate=True,
        )
        predictions = predict_team_win_probability(bundle, test_model_rows)
        output = []
        for row, prediction in zip(test_rows, predictions):
            merged = dict(row)
            if "team_a" in merged:
                merged["team_a_win_probability"] = prediction["baseline_win_probability"]
            else:
                merged["win_probability"] = prediction["baseline_win_probability"]
            merged["model_probabilities"] = prediction.get("model_probabilities", {})
            output.append(merged)
        metadata.update(
            {
                "training_rows": bundle.get("training_rows"),
                "model_types": bundle.get("model_types"),
                "training_metrics": bundle.get("metrics"),
                "skipped_models": bundle.get("skipped_models"),
            }
        )
        return output, metadata
    except Exception as exc:
        if not fallback_to_transparent:
            raise
        metadata.update({"mode": "transparent_fallback", "training_error": str(exc)})
        return _transparent_baseline_predictions(test_rows), metadata


def _custom_model_predictions(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    model_builder: Callable[[list[dict[str, Any]]], Any],
    predictor: Callable[[Any, list[dict[str, Any]]], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = model_builder(train_rows)
    return predictor(model, test_rows), {"mode": "custom_model"}


def _split_label(split: dict[str, int]) -> str:
    return f"Train {split['train_start']}-{split['train_end']} / Test {split['test_season']} playoffs"


def backtest_split(
    rows: Any,
    split: dict[str, int],
    train_model: bool = False,
    model_types: list[str] | None = None,
    model_builder: Callable[[list[dict[str, Any]]], Any] | None = None,
    predictor: Callable[[Any, list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Run one train/test playoff split."""
    train_rows, test_rows = split_rows(rows, split)
    metadata: dict[str, Any] = {"mode": "provided_predictions"}

    if model_builder and predictor:
        prediction_rows, metadata = _custom_model_predictions(train_rows, test_rows, model_builder, predictor)
    elif train_model:
        prediction_rows, metadata = train_and_predict_baseline(train_rows, test_rows, model_types=model_types)
    else:
        prediction_rows = test_rows

    normalized = normalize_prediction_rows(prediction_rows)
    game_metrics = evaluate_game_predictions(normalized)
    series_metrics = series_winner_accuracy(normalized)
    calibration = compare_binary_and_confidence_calibration(normalized, n_bins=n_bins)

    return {
        "split": dict(split),
        "label": _split_label(split),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "scored_rows": len(normalized),
        "model_metadata": metadata,
        "game_metrics": game_metrics,
        "series_metrics": series_metrics,
        "calibration": calibration,
        "upset_detection": upset_detection_metrics(normalized),
        "prediction_rows": normalized,
    }


def _weighted_average(results: list[dict[str, Any]], metric_path: list[str]) -> float | None:
    total_weight = 0
    weighted = 0.0
    for result in results:
        current: Any = result
        for key in metric_path:
            current = current.get(key) if isinstance(current, dict) else None
            if current is None:
                break
        if current is None:
            continue
        weight = int(result.get("scored_rows", 0))
        total_weight += weight
        weighted += float(current) * weight
    return round(weighted / total_weight, 4) if total_weight else None


def summarize_backtest_results(split_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate split-level metrics into a compact report summary."""
    all_rows = [
        row
        for result in split_results
        for row in result.get("prediction_rows", [])
    ]
    total_games = len(all_rows)
    series_results = series_winner_accuracy(all_rows)
    calibration = compare_binary_and_confidence_calibration(all_rows)

    return {
        "splits": len(split_results),
        "total_scored_games": total_games,
        "accuracy": _weighted_average(split_results, ["game_metrics", "accuracy"]),
        "game_winner_accuracy": _weighted_average(split_results, ["game_metrics", "game_winner_accuracy"]),
        "log_loss": _weighted_average(split_results, ["game_metrics", "log_loss"]),
        "brier_score": _weighted_average(split_results, ["game_metrics", "brier_score"]),
        "roc_auc": _weighted_average(split_results, ["game_metrics", "roc_auc"]),
        "series_winner_accuracy": series_results["series_winner_accuracy"],
        "series_count": series_results["series_count"],
        "upset_detection": upset_detection_metrics(all_rows),
        "calibration": calibration,
        "confidence_bucket_performance": calibration["confidence"]["table"],
    }


def run_playoff_backtest(
    playoff_rows: Any,
    splits: list[dict[str, int]] | None = None,
    train_model: bool = False,
    model_types: list[str] | None = None,
    model_builder: Callable[[list[dict[str, Any]]], Any] | None = None,
    predictor: Callable[[Any, list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Run rolling playoff backtests across the default 2022-2025 splits."""
    rows = load_backtest_rows(playoff_rows) if isinstance(playoff_rows, (str, Path)) else _iter_records(playoff_rows)
    splits = splits or default_playoff_splits()
    split_results = [
        backtest_split(
            rows,
            split,
            train_model=train_model,
            model_types=model_types,
            model_builder=model_builder,
            predictor=predictor,
            n_bins=n_bins,
        )
        for split in splits
    ]
    return {
        "splits": split_results,
        "summary": summarize_backtest_results(split_results),
    }


def _synthetic_playoff_rows() -> list[dict[str, Any]]:
    rows = []
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for season in range(2014, 2026):
        for series_index in range(4):
            team_a = teams[(series_index * 2 + season) % len(teams)]
            team_b = teams[(series_index * 2 + season + 1) % len(teams)]
            series_strength = ((season + series_index) % 7 - 3) * 0.035
            team_a_wins = 0
            team_b_wins = 0
            for game_number in range(1, 8):
                if team_a_wins == 4 or team_b_wins == 4:
                    break
                home_edge = 0.025 if game_number in {1, 2, 5, 7} else -0.020
                probability = 0.54 + series_strength + home_edge
                noise = ((season * 17 + series_index * 5 + game_number * 3) % 11 - 5) * 0.018
                actual_team_a_win = int(probability + noise >= 0.50)
                if actual_team_a_win:
                    team_a_wins += 1
                else:
                    team_b_wins += 1
                rows.append(
                    {
                        "season": season,
                        "series_id": f"{season}-{team_a}-{team_b}",
                        "game_number": game_number,
                        "team_a": team_a,
                        "team_b": team_b,
                        "team_a_win_probability": round(max(min(probability, 0.86), 0.14), 3),
                        "actual_winner": team_a if actual_team_a_win else team_b,
                        "actual_series_winner": team_a if team_a_wins == 4 else team_b if team_b_wins == 4 else "",
                    }
                )
            winner = team_a if team_a_wins > team_b_wins else team_b
            for row in rows:
                if row["series_id"] == f"{season}-{team_a}-{team_b}":
                    row["actual_series_winner"] = winner
    return rows


if __name__ == "__main__":
    from src.data.build_historical_dataset import build_canonical_pregame_rows
    from src.data.fetch_current_stats import (
        fetch_historical_playoff_logs,
        fetch_historical_team_ratings,
    )
    from src.models.game_model import walk_forward_backtest

    report = walk_forward_backtest(
        build_canonical_pregame_rows(
            fetch_historical_playoff_logs(),
            fetch_historical_team_ratings(),
        )
    )
    for name, metrics in report["overall"].items():
        print(
            f"{name}: accuracy {metrics['accuracy']:.1%}, "
            f"Brier {metrics['brier_score']:.4f}, "
            f"log loss {metrics['log_loss']:.4f}, "
            f"ECE {metrics['expected_calibration_error']:.4f}"
        )
