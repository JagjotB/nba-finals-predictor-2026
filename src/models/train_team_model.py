"""Team baseline win-probability model.

This is the grounding layer before player, matchup, lineup, clutch, and
scenario adjustments are applied.
"""

from __future__ import annotations

from math import isnan
from typing import Any


TARGET_ALIASES = [
    "win",
    "won",
    "is_win",
    "target",
    "game_win",
    "team_win",
    "WL",
    "W_L",
    "result",
    "game_result",
]

TEAM_KEYS = ["team", "TEAM", "TEAM_ABBREVIATION", "TEAM_NAME"]
OPPONENT_KEYS = ["opponent", "OPPONENT", "opp_team", "OPP_TEAM"]

ALIASES = {
    "team_net_rating": ["team_net_rating", "net_rating", "NET_RATING"],
    "opponent_net_rating": ["opponent_net_rating", "opp_net_rating", "OPP_NET_RATING"],
    "offensive_rating": ["offensive_rating", "OFF_RATING", "ortg"],
    "defensive_rating": ["defensive_rating", "DEF_RATING", "drtg"],
    "opponent_offensive_rating": ["opponent_offensive_rating", "opp_off_rating", "OPP_OFF_RATING"],
    "opponent_defensive_rating": ["opponent_defensive_rating", "opp_def_rating", "OPP_DEF_RATING"],
    "home_court": ["home_court", "is_home", "HOME", "home"],
    "rest_days": ["rest_days", "days_rest", "REST"],
    "opponent_rest_days": ["opponent_rest_days", "opp_rest_days", "OPP_REST"],
    "recent_form": ["recent_form", "recent_win_pct", "last_10_win_pct", "recent_net_rating"],
    "opponent_recent_form": ["opponent_recent_form", "opp_recent_form", "opp_recent_net_rating"],
    "playoff_form": ["playoff_form", "playoff_net_rating", "postseason_form"],
    "opponent_playoff_form": ["opponent_playoff_form", "opp_playoff_form", "opp_playoff_net_rating"],
    "pace": ["pace", "PACE"],
    "opponent_pace": ["opponent_pace", "opp_pace", "OPP_PACE"],
    "efg_pct": ["efg_pct", "EFG_PCT", "eFG%", "effective_fg_pct"],
    "opponent_efg_pct": ["opponent_efg_pct", "opp_efg_pct", "OPP_EFG_PCT"],
    "turnover_pct": ["turnover_pct", "TOV_PCT", "TOV%", "tov_pct"],
    "opponent_turnover_pct": ["opponent_turnover_pct", "opp_tov_pct", "OPP_TOV_PCT"],
    "offensive_rebound_pct": ["offensive_rebound_pct", "OREB_PCT", "OREB%", "oreb_pct"],
    "opponent_offensive_rebound_pct": ["opponent_offensive_rebound_pct", "opp_oreb_pct", "OPP_OREB_PCT"],
    "free_throw_rate": ["free_throw_rate", "FTA_RATE", "FTR", "fta_rate"],
    "opponent_free_throw_rate": ["opponent_free_throw_rate", "opp_fta_rate", "OPP_FTA_RATE"],
    "injury_adjusted_team_strength": ["injury_adjusted_team_strength", "injury_strength", "available_strength"],
    "opponent_injury_adjusted_team_strength": [
        "opponent_injury_adjusted_team_strength",
        "opp_injury_strength",
        "opp_available_strength",
    ],
    "team_score": ["team_score", "PTS", "points"],
    "opponent_score": ["opponent_score", "OPP_PTS", "opp_points"],
}

BASE_FEATURE_COLUMNS = [
    "team_net_rating",
    "opponent_net_rating",
    "net_rating_edge",
    "offensive_rating",
    "defensive_rating",
    "opponent_offensive_rating",
    "opponent_defensive_rating",
    "offense_vs_opponent_defense",
    "defense_vs_opponent_offense",
    "home_court",
    "rest_days",
    "opponent_rest_days",
    "rest_edge",
    "recent_form",
    "opponent_recent_form",
    "recent_form_edge",
    "playoff_form",
    "opponent_playoff_form",
    "playoff_form_edge",
    "pace",
    "opponent_pace",
    "pace_edge",
    "efg_pct",
    "opponent_efg_pct",
    "efg_edge",
    "turnover_pct",
    "opponent_turnover_pct",
    "turnover_edge",
    "offensive_rebound_pct",
    "opponent_offensive_rebound_pct",
    "offensive_rebound_edge",
    "free_throw_rate",
    "opponent_free_throw_rate",
    "free_throw_edge",
    "four_factor_edge",
    "injury_adjusted_team_strength",
    "opponent_injury_adjusted_team_strength",
    "injury_strength_edge",
]


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


def _as_bool_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "home", "h", "1"}:
        return 1.0
    if text in {"false", "no", "n", "away", "a", "0"}:
        return 0.0
    return _as_float(value, default)


def _rate(value: Any, default: float) -> float:
    number = _as_float(value, default)
    if 0.0 <= number <= 1.5:
        return number * 100.0
    return number


def _first_value(row: dict[str, Any] | None, keys: list[str]) -> Any:
    if not row:
        return None
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
        if any(key in data for key in TEAM_KEYS + TARGET_ALIASES):
            return [dict(data)]
        records = []
        for key, value in data.items():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        record = dict(row)
                        record.setdefault("team", key)
                        records.append(record)
            elif isinstance(value, dict):
                record = dict(value)
                record.setdefault("team", key)
                records.append(record)
        return records
    return [dict(row) for row in data]


def _metric(row: dict[str, Any], metric: str, default: float = 0.0) -> float:
    return _as_float(_first_value(row, ALIASES[metric]), default)


def _rate_metric(row: dict[str, Any], metric: str, default: float) -> float:
    return _rate(_first_value(row, ALIASES[metric]), default)


def _target_value(row: dict[str, Any], target_col: str | None = None) -> int | None:
    raw_value = row.get(target_col) if target_col else _first_value(row, TARGET_ALIASES)
    if raw_value is None:
        team_score = _first_value(row, ALIASES["team_score"])
        opponent_score = _first_value(row, ALIASES["opponent_score"])
        if team_score is not None and opponent_score is not None:
            return int(_as_float(team_score) > _as_float(opponent_score))
        return None

    if isinstance(raw_value, bool):
        return int(raw_value)
    text = str(raw_value).strip().lower()
    if text in {"w", "win", "won", "true", "1"}:
        return 1
    if text in {"l", "loss", "lost", "false", "0"}:
        return 0
    numeric = _as_float(raw_value, -1.0)
    if numeric in {0.0, 1.0}:
        return int(numeric)
    return None


def _home_court(row: dict[str, Any]) -> float:
    direct = _first_value(row, ALIASES["home_court"])
    if direct is not None:
        return _as_bool_float(direct)

    team = str(_first_value(row, TEAM_KEYS) or "").strip()
    home_team = str(row.get("home_team") or row.get("HOME_TEAM") or "").strip()
    if team and home_team:
        return 1.0 if team == home_team else 0.0
    return 0.0


def build_team_feature_row(row: dict[str, Any], include_target: bool = True, target_col: str | None = None) -> dict[str, Any]:
    """Normalize one team-game row into baseline model features."""
    team_net = _metric(row, "team_net_rating", _metric(row, "offensive_rating", 115.0) - _metric(row, "defensive_rating", 115.0))
    opponent_net = _metric(
        row,
        "opponent_net_rating",
        _metric(row, "opponent_offensive_rating", 115.0) - _metric(row, "opponent_defensive_rating", 115.0),
    )
    offensive_rating = _metric(row, "offensive_rating", team_net + 115.0)
    defensive_rating = _metric(row, "defensive_rating", 115.0)
    opponent_offensive_rating = _metric(row, "opponent_offensive_rating", opponent_net + 115.0)
    opponent_defensive_rating = _metric(row, "opponent_defensive_rating", 115.0)
    rest_days = _metric(row, "rest_days", 2.0)
    opponent_rest_days = _metric(row, "opponent_rest_days", 2.0)
    recent_form = _metric(row, "recent_form", team_net)
    opponent_recent_form = _metric(row, "opponent_recent_form", opponent_net)
    playoff_form = _metric(row, "playoff_form", team_net)
    opponent_playoff_form = _metric(row, "opponent_playoff_form", opponent_net)
    pace = _metric(row, "pace", 98.5)
    opponent_pace = _metric(row, "opponent_pace", 98.5)
    efg = _rate_metric(row, "efg_pct", 54.0)
    opponent_efg = _rate_metric(row, "opponent_efg_pct", 54.0)
    tov = _rate_metric(row, "turnover_pct", 13.5)
    opponent_tov = _rate_metric(row, "opponent_turnover_pct", 13.5)
    oreb = _rate_metric(row, "offensive_rebound_pct", 27.0)
    opponent_oreb = _rate_metric(row, "opponent_offensive_rebound_pct", 27.0)
    ftr = _rate_metric(row, "free_throw_rate", 25.0)
    opponent_ftr = _rate_metric(row, "opponent_free_throw_rate", 25.0)
    injury_strength = _metric(row, "injury_adjusted_team_strength", team_net)
    opponent_injury_strength = _metric(row, "opponent_injury_adjusted_team_strength", opponent_net)

    feature_row = {
        "team": str(_first_value(row, TEAM_KEYS) or ""),
        "opponent": str(_first_value(row, OPPONENT_KEYS) or ""),
        "team_net_rating": team_net,
        "opponent_net_rating": opponent_net,
        "net_rating_edge": team_net - opponent_net,
        "offensive_rating": offensive_rating,
        "defensive_rating": defensive_rating,
        "opponent_offensive_rating": opponent_offensive_rating,
        "opponent_defensive_rating": opponent_defensive_rating,
        "offense_vs_opponent_defense": offensive_rating - opponent_defensive_rating,
        "defense_vs_opponent_offense": opponent_offensive_rating - defensive_rating,
        "home_court": _home_court(row),
        "rest_days": rest_days,
        "opponent_rest_days": opponent_rest_days,
        "rest_edge": rest_days - opponent_rest_days,
        "recent_form": recent_form,
        "opponent_recent_form": opponent_recent_form,
        "recent_form_edge": recent_form - opponent_recent_form,
        "playoff_form": playoff_form,
        "opponent_playoff_form": opponent_playoff_form,
        "playoff_form_edge": playoff_form - opponent_playoff_form,
        "pace": pace,
        "opponent_pace": opponent_pace,
        "pace_edge": pace - opponent_pace,
        "efg_pct": efg,
        "opponent_efg_pct": opponent_efg,
        "efg_edge": efg - opponent_efg,
        "turnover_pct": tov,
        "opponent_turnover_pct": opponent_tov,
        "turnover_edge": opponent_tov - tov,
        "offensive_rebound_pct": oreb,
        "opponent_offensive_rebound_pct": opponent_oreb,
        "offensive_rebound_edge": oreb - opponent_oreb,
        "free_throw_rate": ftr,
        "opponent_free_throw_rate": opponent_ftr,
        "free_throw_edge": ftr - opponent_ftr,
        "four_factor_edge": (
            (efg - opponent_efg) * 0.40
            + (opponent_tov - tov) * 0.25
            + (oreb - opponent_oreb) * 0.20
            + (ftr - opponent_ftr) * 0.15
        ),
        "injury_adjusted_team_strength": injury_strength,
        "opponent_injury_adjusted_team_strength": opponent_injury_strength,
        "injury_strength_edge": injury_strength - opponent_injury_strength,
    }

    target = _target_value(row, target_col)
    if include_target and target is not None:
        feature_row["target"] = target
    return feature_row


def prepare_team_training_frame(
    game_rows: Any,
    target_col: str | None = None,
    require_target: bool = True,
) -> Any:
    """Convert raw team-game rows into a pandas training/prediction frame."""
    import pandas as pd

    rows = [
        build_team_feature_row(row, include_target=require_target, target_col=target_col)
        for row in _iter_records(game_rows)
    ]
    frame = pd.DataFrame(rows)
    if require_target:
        frame = frame.dropna(subset=["target"])
        frame["target"] = frame["target"].astype(int)
    return frame


def _make_preprocessor() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _calibrator(estimator: Any, cv: int) -> Any:
    from sklearn.calibration import CalibratedClassifierCV

    try:
        return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv=cv)


def _make_classifier(model_type: str, random_state: int = 42) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if model_type == "logistic_regression":
        return Pipeline(
            steps=[
                ("preprocessor", _make_preprocessor()),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is optional and is not installed.") from exc

        return XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.90,
            colsample_bytree=0.90,
            eval_metric="logloss",
            random_state=random_state,
        )
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is optional and is not installed.") from exc

        return LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=15,
            subsample=0.90,
            colsample_bytree=0.90,
            random_state=random_state,
            verbose=-1,
        )
    raise ValueError("Unknown model_type.")


def _can_calibrate(y: Any, max_cv: int = 5) -> int | None:
    import pandas as pd

    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        return None
    min_count = int(counts.min())
    if min_count < 3:
        return None
    return min(max_cv, min_count)


def _positive_class_probability(model: Any, x: Any) -> Any:
    probabilities = model.predict_proba(x)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    return probabilities[:, 1]


def evaluate_team_model(y_true: Any, y_prob: Any, n_bins: int = 10) -> dict[str, Any]:
    """Evaluate win-probability predictions."""
    import numpy as np
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.50).astype(int)
    has_both_classes = len(set(y_true.tolist())) == 2

    if has_both_classes:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        calibration_points = [
            {"mean_predicted_probability": round(float(pred), 4), "observed_win_rate": round(float(true), 4)}
            for pred, true in zip(prob_pred, prob_true)
        ]
        roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
        loss = round(float(log_loss(y_true, y_prob, labels=[0, 1])), 4)
    else:
        calibration_points = []
        roc_auc = None
        loss = None

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "log_loss": loss,
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "roc_auc": roc_auc,
        "calibration_curve": calibration_points,
    }


def _train_one_model(
    x_train: Any,
    y_train: Any,
    model_type: str,
    random_state: int,
    calibrate: bool,
) -> Any:
    estimator = _make_classifier(model_type, random_state)
    cv = _can_calibrate(y_train)
    if calibrate and cv:
        model = _calibrator(estimator, cv)
    else:
        model = estimator
    model.fit(x_train, y_train)
    return model


def _model_types_with_available_optionals(model_types: list[str]) -> tuple[list[str], dict[str, str]]:
    available = []
    skipped = {}
    for model_type in model_types:
        try:
            _make_classifier(model_type)
        except ImportError as exc:
            skipped[model_type] = str(exc)
            continue
        available.append(model_type)
    return available, skipped


def train_team_model(
    game_rows: Any,
    model_types: list[str] | None = None,
    target_col: str | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
    calibrate: bool = True,
) -> dict[str, Any]:
    """Train baseline team win/loss models and a calibrated ensemble."""
    from sklearn.model_selection import train_test_split

    frame = prepare_team_training_frame(game_rows, target_col=target_col, require_target=True)
    if frame.empty:
        raise ValueError("Training data must include rows with a win/loss target.")
    if frame["target"].nunique() < 2:
        raise ValueError("Training data must include both wins and losses.")

    model_types = model_types or ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    model_types, skipped_models = _model_types_with_available_optionals(model_types)
    if not model_types:
        raise ValueError("No requested model types are available.")

    x = frame[BASE_FEATURE_COLUMNS]
    y = frame["target"]
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    trained_models = {}
    metrics = {}
    holdout_probabilities = {}
    for model_type in model_types:
        model = _train_one_model(x_train, y_train, model_type, random_state, calibrate)
        probabilities = _positive_class_probability(model, x_test)
        trained_models[model_type] = model
        holdout_probabilities[model_type] = probabilities
        metrics[model_type] = evaluate_team_model(y_test, probabilities)

    ensemble_probabilities = sum(holdout_probabilities.values()) / len(holdout_probabilities)
    metrics["calibrated_ensemble"] = evaluate_team_model(y_test, ensemble_probabilities)

    return {
        "feature_columns": BASE_FEATURE_COLUMNS,
        "models": trained_models,
        "model_types": model_types,
        "skipped_models": skipped_models,
        "metrics": metrics,
        "holdout": {
            "y_true": [int(value) for value in y_test.tolist()],
            "probabilities": {
                model_type: [float(value) for value in values]
                for model_type, values in holdout_probabilities.items()
            },
            "calibrated_ensemble": [float(value) for value in ensemble_probabilities],
        },
        "training_rows": int(len(frame)),
    }


def predict_team_win_probability(
    model_bundle: dict[str, Any],
    game_rows: Any,
) -> list[dict[str, Any]]:
    """Predict baseline win probability for one or more team-game rows."""
    frame = prepare_team_training_frame(game_rows, require_target=False)
    x = frame[model_bundle["feature_columns"]]
    per_model = {
        model_type: _positive_class_probability(model, x)
        for model_type, model in model_bundle["models"].items()
    }
    ensemble = sum(per_model.values()) / len(per_model)

    predictions = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        model_probabilities = {
            model_type: round(float(values[idx]), 4)
            for model_type, values in per_model.items()
        }
        predictions.append(
            {
                "team": row.get("team", ""),
                "opponent": row.get("opponent", ""),
                "baseline_win_probability": round(float(ensemble[idx]), 4),
                "model_probabilities": model_probabilities,
            }
        )
    return predictions


def baseline_probability_from_features(feature_row: dict[str, Any]) -> float:
    """Transparent fallback baseline when no trained model is available yet."""
    margin_signal = (
        feature_row.get("net_rating_edge", 0.0) * 0.16
        + feature_row.get("recent_form_edge", 0.0) * 0.07
        + feature_row.get("playoff_form_edge", 0.0) * 0.08
        + feature_row.get("four_factor_edge", 0.0) * 0.04
        + feature_row.get("injury_strength_edge", 0.0) * 0.09
        + feature_row.get("home_court", 0.0) * 2.2
        + feature_row.get("rest_edge", 0.0) * 0.45
    )
    probability = 1.0 / (1.0 + pow(2.718281828, -margin_signal / 6.5))
    return round(_clip(probability, 0.05, 0.95), 4)


def predict_baseline_without_training(game_rows: Any) -> list[dict[str, Any]]:
    """Use the transparent formula before historical training data exists."""
    frame = prepare_team_training_frame(game_rows, require_target=False)
    predictions = []
    for _, row in frame.iterrows():
        feature_row = row.to_dict()
        predictions.append(
            {
                "team": feature_row.get("team", ""),
                "opponent": feature_row.get("opponent", ""),
                "baseline_win_probability": baseline_probability_from_features(feature_row),
                "model_probabilities": {"transparent_formula": baseline_probability_from_features(feature_row)},
            }
        )
    return predictions


if __name__ == "__main__":
    sample_rows = []
    for i in range(80):
        edge = (i % 16) - 7.5
        home = i % 2
        win = int(edge + home * 1.5 + (i % 5 - 2) * 0.4 > 0)
        sample_rows.append(
            {
                "team": "NYK" if i % 2 == 0 else "SAS",
                "opponent": "SAS" if i % 2 == 0 else "NYK",
                "team_net_rating": edge,
                "opponent_net_rating": -edge / 2.0,
                "offensive_rating": 115 + edge / 2.0,
                "defensive_rating": 115 - edge / 3.0,
                "home_court": home,
                "rest_days": 2 + (i % 3),
                "opponent_rest_days": 2,
                "recent_form": edge + 1,
                "playoff_form": edge,
                "efg_pct": 54 + edge / 6.0,
                "opponent_efg_pct": 54 - edge / 7.0,
                "turnover_pct": 13 - edge / 18.0,
                "opponent_turnover_pct": 13 + edge / 20.0,
                "offensive_rebound_pct": 27 + edge / 10.0,
                "opponent_offensive_rebound_pct": 27 - edge / 12.0,
                "free_throw_rate": 25 + edge / 8.0,
                "opponent_free_throw_rate": 25 - edge / 9.0,
                "injury_adjusted_team_strength": edge,
                "win": win,
            }
        )

    bundle = train_team_model(sample_rows, model_types=["logistic_regression", "random_forest"])
    print(bundle["metrics"]["calibrated_ensemble"])
    print(predict_team_win_probability(bundle, sample_rows[:2]))
