"""Regularized meta-model for combining validated prediction components."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "stats_cache" / "meta_model.json"
MODEL_BINARY_PATH = MODEL_PATH.with_suffix(".joblib")

COMPONENT_COLUMNS = [
    "baseline_logit",
    "net_rating_logit",
    "player_edge",
    "matchup_edge",
    "lineup_edge",
    "clutch_edge",
    "injury_edge",
    "coaching_edge",
    "player_data_available",
    "matchup_data_available",
    "lineup_data_available",
    "clutch_data_available",
    "injury_data_available",
    "coaching_data_available",
]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _logit(probability: float) -> float:
    probability = _clip(probability, 1e-6, 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def _feature_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        "baseline_logit": _logit(float(
            row.get("ensemble_probability") or row.get("baseline_probability") or 0.5
        )),
        "net_rating_logit": _logit(float(row.get("net_rating_probability", row.get("baseline_probability", 0.5)))),
        "player_edge": float(row.get("player_edge", 0.0)) / 11.5,
        "matchup_edge": float(row.get("matchup_edge", 0.0)) / 11.5,
        "lineup_edge": float(row.get("lineup_edge", 0.0)) / 11.5,
        "clutch_edge": float(row.get("clutch_edge", 0.0)) / 11.5,
        "injury_edge": float(row.get("injury_edge", 0.0)) / 11.5,
        "coaching_edge": float(row.get("coaching_edge", 0.0)) / 11.5,
        "player_data_available": float(row.get("player_data_available", 0.0)),
        "matchup_data_available": float(row.get("matchup_data_available", 0.0)),
        "lineup_data_available": float(row.get("lineup_data_available", 0.0)),
        "clutch_data_available": float(row.get("clutch_data_available", 0.0)),
        "injury_data_available": float(row.get("injury_data_available", 0.0)),
        "coaching_data_available": float(row.get("coaching_data_available", 0.0)),
    }


def train_meta_model(oof_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit coefficients only from predictions made out of sample."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if len(oof_rows) < 100:
        raise ValueError("At least 100 out-of-fold rows are required for the meta-model.")
    x = np.asarray([
        [_feature_row(row)[column] for column in COMPONENT_COLUMNS]
        for row in oof_rows
    ])
    y = np.asarray([int(row["actual_team_a_win"]) for row in oof_rows])
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=0.25, max_iter=2000, random_state=42)),
    ])
    seasons = sorted({str(row.get("season")) for row in oof_rows})
    validation_probabilities = []
    validation_targets = []
    for test_season in seasons[1:]:
        train_indices = [
            index for index, row in enumerate(oof_rows)
            if str(row.get("season")) < test_season
        ]
        test_indices = [
            index for index, row in enumerate(oof_rows)
            if str(row.get("season")) == test_season
        ]
        if len(train_indices) < 100 or not test_indices:
            continue
        fold_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=0.25, max_iter=2000, random_state=42)),
        ])
        fold_model.fit(x[train_indices], y[train_indices])
        validation_probabilities.extend(
            fold_model.predict_proba(x[test_indices])[:, 1].tolist()
        )
        validation_targets.extend(y[test_indices].tolist())

    model.fit(x, y)
    probabilities = model.predict_proba(x)[:, 1]
    classifier = model.named_steps["classifier"]
    coefficients = {
        column: round(float(value), 6)
        for column, value in zip(COMPONENT_COLUMNS, classifier.coef_[0])
    }
    return {
        "feature_columns": COMPONENT_COLUMNS,
        "training_rows": len(oof_rows),
        "training_seasons": sorted({str(row.get("season")) for row in oof_rows}),
        "metrics": {
            "log_loss": round(float(log_loss(y, probabilities)), 4),
            "brier_score": round(float(brier_score_loss(y, probabilities)), 4),
            "walk_forward_log_loss": (
                round(float(log_loss(validation_targets, validation_probabilities)), 4)
                if validation_targets else None
            ),
            "walk_forward_brier_score": (
                round(float(brier_score_loss(validation_targets, validation_probabilities)), 4)
                if validation_targets else None
            ),
            "walk_forward_games": len(validation_targets),
        },
        "coefficients_standardized": coefficients,
        "validated_components": [
            column.removesuffix("_data_available")
            for column in COMPONENT_COLUMNS
            if column.endswith("_data_available")
            and any(float(row.get(column, 0.0)) > 0 for row in oof_rows)
        ],
        "_model_ref": model,
    }


def save_meta_model(bundle: dict[str, Any], path: Path = MODEL_PATH) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path.with_suffix(".joblib"))
    metadata = {key: value for key, value in bundle.items() if not key.startswith("_")}
    metadata["model_format"] = "joblib_regularized_meta_model"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_meta_model(path: Path = MODEL_PATH) -> dict[str, Any] | None:
    binary = path.with_suffix(".joblib")
    if not binary.exists():
        return None
    import joblib
    return joblib.load(binary)


def predict_meta_probability(
    baseline_probability: float,
    component_edges: dict[str, float],
    availability: dict[str, bool] | None = None,
    model_bundle: dict[str, Any] | None = None,
    net_rating_probability: float | None = None,
) -> float:
    """Predict from the learned stack; unvalidated components receive zero effect."""
    import numpy as np

    bundle = model_bundle or load_meta_model()
    if not bundle:
        return round(_clip(baseline_probability, 0.05, 0.95), 4)
    availability = availability or {}
    row = {
        "baseline_probability": baseline_probability,
        "net_rating_probability": (
            baseline_probability if net_rating_probability is None else net_rating_probability
        ),
        **component_edges,
        **{
            f"{component}_data_available": int(bool(availability.get(component, False)))
            for component in ("player", "matchup", "lineup", "clutch", "injury", "coaching")
        },
    }
    features = _feature_row(row)
    x = np.asarray([[features[column] for column in bundle["feature_columns"]]])
    probability = float(bundle["_model_ref"].predict_proba(x)[0, 1])
    return round(_clip(probability, 0.05, 0.95), 4)
