"""Evaluation metrics for game and series win-probability predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isnan, log, sqrt
from typing import Any


PROBABILITY_ALIASES = [
    "team_a_win_probability",
    "win_probability",
    "predicted_probability",
    "prediction",
    "probability",
    "model_probability",
    "baseline_win_probability",
]

TARGET_ALIASES = [
    "actual_team_a_win",
    "actual_win",
    "actual",
    "team_a_actual_result",
    "win",
    "won",
    "target",
    "result",
    "is_win",
]

ACTUAL_WINNER_ALIASES = ["actual_winner", "winner", "winning_team", "game_winner"]
PREDICTED_WINNER_ALIASES = ["predicted_winner", "model_winner", "favorite"]

TEAM_A_ALIASES = ["team_a", "team", "TEAM", "TEAM_ABBREVIATION"]
TEAM_B_ALIASES = ["team_b", "opponent", "OPPONENT", "opp_team"]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if isnan(number):
        return default
    return number


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _first_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _iter_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return [dict(row) for row in data.to_dict(orient="records")]
    if isinstance(data, dict):
        if any(key in data for key in PROBABILITY_ALIASES + TARGET_ALIASES):
            return [dict(data)]
        records = []
        for key, value in data.items():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        record = dict(row)
                        record.setdefault("series_id", key)
                        records.append(record)
            elif isinstance(value, dict):
                record = dict(value)
                record.setdefault("series_id", key)
                records.append(record)
        return records
    return [dict(row) for row in data]


def parse_binary_result(value: Any) -> int | None:
    """Parse common win/loss target encodings."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"w", "win", "won", "true", "t", "yes", "y", "1"}:
        return 1
    if text in {"l", "loss", "lost", "false", "f", "no", "n", "0"}:
        return 0
    number = _as_float(value, -999.0)
    if number in {0.0, 1.0}:
        return int(number)
    return None


def _team_a(row: dict[str, Any]) -> str:
    return str(_first_value(row, TEAM_A_ALIASES) or "Team A").strip()


def _team_b(row: dict[str, Any]) -> str:
    return str(_first_value(row, TEAM_B_ALIASES) or "Team B").strip()


def _probability(row: dict[str, Any]) -> float | None:
    value = _first_value(row, PROBABILITY_ALIASES)
    if value is None and "team_b_win_probability" in row:
        value = 1.0 - _as_float(row["team_b_win_probability"], 0.5)
    if value is None:
        return None
    probability = _as_float(value, -1.0)
    if probability < 0:
        return None
    if probability > 1.0 and probability <= 100.0:
        probability /= 100.0
    return _clip(probability, 0.0001, 0.9999)


def _actual_from_scores(row: dict[str, Any], team_a: str, team_b: str) -> int | None:
    team_a_score = (
        row.get("actual_score_team_a")
        or row.get("team_a_score")
        or row.get("score_team_a")
        or row.get("team_score")
    )
    team_b_score = (
        row.get("actual_score_team_b")
        or row.get("team_b_score")
        or row.get("score_team_b")
        or row.get("opponent_score")
    )
    if team_a_score is None or team_b_score is None:
        return None
    score_a = _as_float(team_a_score)
    score_b = _as_float(team_b_score)
    if score_a == score_b:
        return None
    return int(score_a > score_b)


def _actual_target(row: dict[str, Any], team_a: str, team_b: str) -> int | None:
    direct = parse_binary_result(_first_value(row, TARGET_ALIASES))
    if direct is not None:
        return direct

    actual_winner = str(_first_value(row, ACTUAL_WINNER_ALIASES) or "").strip()
    if actual_winner:
        if actual_winner == team_a:
            return 1
        if actual_winner == team_b:
            return 0

    return _actual_from_scores(row, team_a, team_b)


def normalize_prediction_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one game prediction row for binary probability scoring."""
    team_a = _team_a(row)
    team_b = _team_b(row)
    probability = _probability(row)
    actual = _actual_target(row, team_a, team_b)
    if probability is None or actual is None:
        return None

    predicted_label = int(probability >= 0.50)
    predicted_winner = str(_first_value(row, PREDICTED_WINNER_ALIASES) or "").strip()
    if not predicted_winner:
        predicted_winner = team_a if predicted_label else team_b
    actual_winner = team_a if actual else team_b
    favorite_probability = max(probability, 1.0 - probability)
    favorite = team_a if probability >= 0.50 else team_b
    favorite_won = int(actual_winner == favorite)
    season = row.get("season") or row.get("SEASON") or row.get("year") or row.get("test_season")
    series_id = row.get("series_id") or row.get("series") or row.get("playoff_series")
    game_number = row.get("game_number") or row.get("game") or row.get("GAME_NUMBER")
    game_id = row.get("game_id") or row.get("GAME_ID") or f"{season}:{series_id}:{game_number}:{team_a}:{team_b}"

    return {
        **row,
        "game_id": game_id,
        "season": season,
        "series_id": series_id,
        "game_number": game_number,
        "team_a": team_a,
        "team_b": team_b,
        "probability": probability,
        "actual": actual,
        "predicted_label": predicted_label,
        "predicted_winner": predicted_winner,
        "actual_winner": actual_winner,
        "favorite": favorite,
        "favorite_probability": favorite_probability,
        "favorite_won": favorite_won,
        "underdog": team_b if favorite == team_a else team_a,
        "underdog_probability": 1.0 - favorite_probability,
    }


def normalize_prediction_rows(rows: Any) -> list[dict[str, Any]]:
    """Normalize all scorable rows and drop incomplete records."""
    normalized = []
    for row in _iter_records(rows):
        normalized_row = normalize_prediction_row(row)
        if normalized_row is not None:
            normalized.append(normalized_row)
    return normalized


def accuracy_score(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> float:
    if not y_true:
        return 0.0
    correct = sum(int(prob >= threshold) == actual for actual, prob in zip(y_true, y_prob))
    return correct / len(y_true)


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return 0.0
    return sum((prob - actual) ** 2 for actual, prob in zip(y_true, y_prob)) / len(y_true)


def log_loss_score(y_true: list[int], y_prob: list[float], epsilon: float = 1e-15) -> float:
    if not y_true:
        return 0.0
    total = 0.0
    for actual, probability in zip(y_true, y_prob):
        probability = _clip(probability, epsilon, 1.0 - epsilon)
        total += actual * log(probability) + (1 - actual) * log(1.0 - probability)
    return -total / len(y_true)


def roc_auc_score(y_true: list[int], y_prob: list[float]) -> float | None:
    """Compute binary ROC AUC using average ranks."""
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    sorted_pairs = sorted(enumerate(y_prob), key=lambda pair: pair[1])
    ranks = [0.0] * len(y_prob)
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][1] == sorted_pairs[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for rank_index in range(index, end):
            ranks[sorted_pairs[rank_index][0]] = average_rank
        index = end

    positive_rank_sum = sum(rank for rank, actual in zip(ranks, y_true) if actual == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def prediction_interval(values: list[float]) -> dict[str, float | None]:
    """Return mean and standard error for a list of metric values."""
    if not values:
        return {"mean": None, "standard_error": None}
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": round(mean, 4), "standard_error": None}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": round(mean, 4), "standard_error": round(sqrt(variance / len(values)), 4)}


def confidence_bucket_performance(
    rows: Any,
    buckets: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate favorite win rate by confidence bucket."""
    normalized = normalize_prediction_rows(rows)
    buckets = buckets or [
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 0.80),
        (0.80, 1.01),
    ]
    output = []
    for low, high in buckets:
        bucket_rows = [
            row
            for row in normalized
            if row["favorite_probability"] >= low and row["favorite_probability"] < high
        ]
        if bucket_rows:
            mean_confidence = sum(row["favorite_probability"] for row in bucket_rows) / len(bucket_rows)
            favorite_win_rate = sum(row["favorite_won"] for row in bucket_rows) / len(bucket_rows)
            accuracy = favorite_win_rate
        else:
            mean_confidence = None
            favorite_win_rate = None
            accuracy = None
        output.append(
            {
                "bucket": f"{low:.0%}-{min(high, 1.0):.0%}",
                "low": low,
                "high": min(high, 1.0),
                "count": len(bucket_rows),
                "mean_confidence": round(mean_confidence, 4) if mean_confidence is not None else None,
                "favorite_win_rate": round(favorite_win_rate, 4) if favorite_win_rate is not None else None,
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
                "calibration_error": (
                    round(favorite_win_rate - mean_confidence, 4)
                    if favorite_win_rate is not None and mean_confidence is not None
                    else None
                ),
            }
        )
    return output


def upset_detection_metrics(
    rows: Any,
    favorite_threshold: float = 0.55,
    alert_confidence_threshold: float = 0.60,
    high_confidence_threshold: float = 0.70,
) -> dict[str, Any]:
    """Evaluate how often modeled favorites lose and whether low-confidence games flag upset risk."""
    normalized = normalize_prediction_rows(rows)
    eligible = [row for row in normalized if row["favorite_probability"] >= favorite_threshold]
    upsets = [row for row in eligible if row["favorite_won"] == 0]
    alerts = [row for row in eligible if row["favorite_probability"] <= alert_confidence_threshold]
    alert_upsets = [row for row in alerts if row["favorite_won"] == 0]
    high_confidence = [row for row in eligible if row["favorite_probability"] >= high_confidence_threshold]
    high_confidence_upsets = [row for row in high_confidence if row["favorite_won"] == 0]

    return {
        "favorite_threshold": favorite_threshold,
        "upset_count": len(upsets),
        "eligible_games": len(eligible),
        "upset_rate": round(len(upsets) / len(eligible), 4) if eligible else None,
        "upset_warning_games": len(alerts),
        "upset_warning_precision": round(len(alert_upsets) / len(alerts), 4) if alerts else None,
        "upset_warning_recall": round(len(alert_upsets) / len(upsets), 4) if upsets else None,
        "high_confidence_games": len(high_confidence),
        "high_confidence_upset_count": len(high_confidence_upsets),
        "high_confidence_upset_rate": (
            round(len(high_confidence_upsets) / len(high_confidence), 4)
            if high_confidence
            else None
        ),
    }


def game_winner_accuracy(rows: Any) -> float:
    normalized = normalize_prediction_rows(rows)
    if not normalized:
        return 0.0
    return sum(row["predicted_winner"] == row["actual_winner"] for row in normalized) / len(normalized)


def _series_groups(rows: Any) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in normalize_prediction_rows(rows):
        key = row.get("series_id") or f"{row.get('season')}:{row['team_a']}:{row['team_b']}"
        groups[key].append(row)
    return groups


def _series_actual_winner(rows: list[dict[str, Any]]) -> str | None:
    direct = [row.get("actual_series_winner") or row.get("series_winner") for row in rows]
    direct = [str(value).strip() for value in direct if value not in (None, "")]
    if direct:
        return Counter(direct).most_common(1)[0][0]

    wins: dict[str, int] = defaultdict(int)
    for row in rows:
        wins[row["actual_winner"]] += 1
    return max(wins, key=wins.get) if wins else None


def _series_predicted_winner(rows: list[dict[str, Any]]) -> str | None:
    direct = [row.get("predicted_series_winner") for row in rows]
    direct = [str(value).strip() for value in direct if value not in (None, "")]
    if direct:
        return Counter(direct).most_common(1)[0][0]

    expected_wins: dict[str, float] = defaultdict(float)
    for row in rows:
        expected_wins[row["team_a"]] += row["probability"]
        expected_wins[row["team_b"]] += 1.0 - row["probability"]
    return max(expected_wins, key=expected_wins.get) if expected_wins else None


def series_winner_accuracy(rows: Any) -> dict[str, Any]:
    """Evaluate predicted vs actual series winner by series_id."""
    series_rows = _series_groups(rows)
    evaluated = []
    for series_id, group in series_rows.items():
        actual = _series_actual_winner(group)
        predicted = _series_predicted_winner(group)
        if not actual or not predicted:
            continue
        evaluated.append(
            {
                "series_id": series_id,
                "predicted_series_winner": predicted,
                "actual_series_winner": actual,
                "correct": predicted == actual,
                "games": len(group),
            }
        )

    return {
        "series_count": len(evaluated),
        "series_winner_accuracy": (
            round(sum(row["correct"] for row in evaluated) / len(evaluated), 4)
            if evaluated
            else None
        ),
        "series_results": evaluated,
    }


def evaluate_game_predictions(rows: Any) -> dict[str, Any]:
    """Compute core game-level metrics."""
    normalized = normalize_prediction_rows(rows)
    y_true = [int(row["actual"]) for row in normalized]
    y_prob = [float(row["probability"]) for row in normalized]
    auc = roc_auc_score(y_true, y_prob)

    return {
        "count": len(normalized),
        "accuracy": round(accuracy_score(y_true, y_prob), 4) if normalized else None,
        "game_winner_accuracy": round(game_winner_accuracy(normalized), 4) if normalized else None,
        "log_loss": round(log_loss_score(y_true, y_prob), 4) if normalized else None,
        "brier_score": round(brier_score(y_true, y_prob), 4) if normalized else None,
        "roc_auc": round(auc, 4) if auc is not None else None,
        "average_predicted_probability": round(sum(y_prob) / len(y_prob), 4) if y_prob else None,
        "observed_win_rate": round(sum(y_true) / len(y_true), 4) if y_true else None,
        "upset_detection": upset_detection_metrics(normalized),
        "confidence_bucket_performance": confidence_bucket_performance(normalized),
    }


def evaluate_predictions(rows: Any) -> dict[str, Any]:
    """Compute full game, series, confidence, and upset metrics."""
    return {
        "game_metrics": evaluate_game_predictions(rows),
        "series_metrics": series_winner_accuracy(rows),
    }


if __name__ == "__main__":
    sample = [
        {"team_a": "A", "team_b": "B", "team_a_win_probability": 0.60, "actual_winner": "A", "series_id": "S1"},
        {"team_a": "A", "team_b": "B", "team_a_win_probability": 0.58, "actual_winner": "B", "series_id": "S1"},
        {"team_a": "C", "team_b": "D", "team_a_win_probability": 0.42, "actual_winner": "D", "series_id": "S2"},
        {"team_a": "C", "team_b": "D", "team_a_win_probability": 0.48, "actual_winner": "C", "series_id": "S2"},
    ]
    print(evaluate_predictions(sample))
