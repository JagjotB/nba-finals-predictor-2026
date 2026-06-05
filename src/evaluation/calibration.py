"""Calibration analysis for playoff win-probability predictions."""

from __future__ import annotations

from typing import Any

from src.evaluation.metrics import normalize_prediction_rows


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bin_edges(n_bins: int, low: float = 0.0, high: float = 1.0) -> list[tuple[float, float]]:
    width = (high - low) / n_bins
    edges = []
    for index in range(n_bins):
        start = low + width * index
        end = high if index == n_bins - 1 else low + width * (index + 1)
        edges.append((start, end))
    return edges


def _quantile_edges(values: list[float], n_bins: int) -> list[tuple[float, float]]:
    if not values:
        return _bin_edges(n_bins)
    sorted_values = sorted(values)
    edges = []
    for index in range(n_bins):
        start_index = int(index * len(sorted_values) / n_bins)
        end_index = int((index + 1) * len(sorted_values) / n_bins) - 1
        start = sorted_values[max(start_index, 0)]
        end = sorted_values[min(max(end_index, start_index), len(sorted_values) - 1)]
        if index == 0:
            start = 0.0
        if index == n_bins - 1:
            end = 1.0
        edges.append((start, end + 1e-12))
    return edges


def _rows_for_mode(rows: Any, mode: str) -> list[dict[str, float]]:
    normalized = normalize_prediction_rows(rows)
    output = []
    for row in normalized:
        if mode == "confidence":
            output.append(
                {
                    "probability": float(row["favorite_probability"]),
                    "actual": float(row["favorite_won"]),
                }
            )
        else:
            output.append(
                {
                    "probability": float(row["probability"]),
                    "actual": float(row["actual"]),
                }
            )
    return output


def calibration_table(
    rows: Any,
    n_bins: int = 10,
    strategy: str = "uniform",
    mode: str = "binary",
) -> list[dict[str, Any]]:
    """Build observed vs predicted calibration buckets.

    mode="binary" evaluates the probability assigned to the row's target team.
    mode="confidence" evaluates whether favorites win at their stated confidence.
    """
    prepared = _rows_for_mode(rows, mode)
    probabilities = [row["probability"] for row in prepared]
    edges = _quantile_edges(probabilities, n_bins) if strategy == "quantile" else _bin_edges(n_bins)
    table = []

    for index, (low, high) in enumerate(edges):
        bucket = [
            row
            for row in prepared
            if row["probability"] >= low
            and (row["probability"] < high or (index == len(edges) - 1 and row["probability"] <= high))
        ]
        if bucket:
            mean_predicted = sum(row["probability"] for row in bucket) / len(bucket)
            observed = sum(row["actual"] for row in bucket) / len(bucket)
            error = observed - mean_predicted
        else:
            mean_predicted = None
            observed = None
            error = None

        table.append(
            {
                "bucket": f"{low:.0%}-{min(high, 1.0):.0%}",
                "bin_start": round(low, 4),
                "bin_end": round(min(high, 1.0), 4),
                "count": len(bucket),
                "mean_predicted_probability": round(mean_predicted, 4) if mean_predicted is not None else None,
                "observed_win_rate": round(observed, 4) if observed is not None else None,
                "calibration_error": round(error, 4) if error is not None else None,
                "absolute_calibration_error": round(abs(error), 4) if error is not None else None,
            }
        )
    return table


def expected_calibration_error(table: list[dict[str, Any]]) -> float | None:
    total = sum(row["count"] for row in table)
    if total <= 0:
        return None
    weighted = sum(
        row["count"] * float(row["absolute_calibration_error"])
        for row in table
        if row["absolute_calibration_error"] is not None
    )
    return round(weighted / total, 4)


def maximum_calibration_error(table: list[dict[str, Any]]) -> float | None:
    errors = [
        float(row["absolute_calibration_error"])
        for row in table
        if row["absolute_calibration_error"] is not None
    ]
    return round(max(errors), 4) if errors else None


def calibration_slope(rows: Any, mode: str = "binary") -> float | None:
    """Estimate observed outcome slope against predicted probability."""
    prepared = _rows_for_mode(rows, mode)
    if len(prepared) < 2:
        return None
    x_values = [row["probability"] for row in prepared]
    y_values = [row["actual"] for row in prepared]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    return round(numerator / denominator, 4)


def calibration_label(ece: float | None) -> str:
    if ece is None:
        return "not enough data"
    if ece <= 0.025:
        return "well calibrated"
    if ece <= 0.060:
        return "usable calibration"
    return "needs recalibration"


def calibration_summary(
    rows: Any,
    n_bins: int = 10,
    strategy: str = "uniform",
    mode: str = "binary",
) -> dict[str, Any]:
    """Summarize calibration quality."""
    table = calibration_table(rows, n_bins=n_bins, strategy=strategy, mode=mode)
    ece = expected_calibration_error(table)
    mce = maximum_calibration_error(table)
    return {
        "mode": mode,
        "n_bins": n_bins,
        "strategy": strategy,
        "sample_count": sum(row["count"] for row in table),
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "calibration_slope": calibration_slope(rows, mode=mode),
        "label": calibration_label(ece),
        "table": table,
    }


def compare_binary_and_confidence_calibration(
    rows: Any,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Return target-team calibration and favorite-confidence calibration."""
    return {
        "binary": calibration_summary(rows, n_bins=n_bins, strategy=strategy, mode="binary"),
        "confidence": calibration_summary(rows, n_bins=n_bins, strategy=strategy, mode="confidence"),
    }


def fit_bin_calibrator(
    rows: Any,
    n_bins: int = 10,
    shrinkage: float = 20.0,
    mode: str = "binary",
) -> dict[str, Any]:
    """Fit a simple binned probability calibrator with shrinkage toward bucket center."""
    table = calibration_table(rows, n_bins=n_bins, strategy="uniform", mode=mode)
    calibrated_bins = []
    for row in table:
        center = (row["bin_start"] + row["bin_end"]) / 2.0
        if row["count"] <= 0 or row["observed_win_rate"] is None:
            calibrated = center
        else:
            weight = row["count"] / (row["count"] + shrinkage)
            calibrated = row["observed_win_rate"] * weight + center * (1.0 - weight)
        calibrated_bins.append(
            {
                "bin_start": row["bin_start"],
                "bin_end": row["bin_end"],
                "count": row["count"],
                "calibrated_probability": round(_clip(calibrated, 0.001, 0.999), 4),
            }
        )
    return {
        "mode": mode,
        "n_bins": n_bins,
        "shrinkage": shrinkage,
        "bins": calibrated_bins,
    }


def apply_bin_calibration(probability: float, calibrator: dict[str, Any]) -> float:
    """Apply a fitted binned calibrator to one probability."""
    probability = _clip(float(probability), 0.001, 0.999)
    bins = calibrator.get("bins", [])
    for index, row in enumerate(bins):
        if probability >= row["bin_start"] and (
            probability < row["bin_end"] or index == len(bins) - 1
        ):
            return float(row["calibrated_probability"])
    return probability


if __name__ == "__main__":
    sample = [
        {"team_a": "A", "team_b": "B", "team_a_win_probability": 0.60, "actual_winner": "A"},
        {"team_a": "A", "team_b": "B", "team_a_win_probability": 0.62, "actual_winner": "B"},
        {"team_a": "A", "team_b": "B", "team_a_win_probability": 0.70, "actual_winner": "A"},
        {"team_a": "A", "team_b": "B", "team_a_win_probability": 0.40, "actual_winner": "B"},
    ]
    print(compare_binary_and_confidence_calibration(sample, n_bins=5))
