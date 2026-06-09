"""Trained game win-probability model: calibrated structural logistic model.

Replaces the hardcoded sigmoid formula in train_team_model.py with a model
trained on historical playoff game logs using structural features that
are stable across roster changes.

Training features (all team-differential, team_a minus team_b):
  net_rating_diff, efg_diff, tov_pct_diff, oreb_pct_diff, fta_rate_diff,
  pace_diff

Home court and rest are explicit context adjustments instead of learned
features. This avoids treating playoff seeding strength as home-court value.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.data.build_historical_dataset import build_canonical_pregame_rows

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "stats_cache" / "game_model.json"
MODEL_BINARY_PATH = PROJECT_ROOT / "data" / "processed" / "stats_cache" / "game_model.joblib"

FEATURE_COLS = [
    "regular_net_rating_diff",
    "blended_net_rating_diff",
    "regular_efg_pct_diff",
    "blended_efg_pct_diff",
    "regular_tov_pct_diff",
    "blended_tov_pct_diff",
    "regular_oreb_pct_diff",
    "blended_oreb_pct_diff",
    "regular_fta_rate_diff",
    "blended_fta_rate_diff",
    "regular_pace_diff",
    "blended_pace_diff",
    "recent_net_rating_diff",
    "travel_miles_diff",
    "travel_data_available",
    "home_court",
]

# XGBoost uses all LR features plus injury signals.
# LR deliberately excludes injury features — validation showed LR is hurt by
# sparse injury data while XGBoost handles it correctly via tree splits.
XGB_FEATURE_COLS = FEATURE_COLS + [
    "injury_strength_diff",
    "injury_data_available",
    # In-series features (series_games_played, series_score_diff, etc.) computed
    # but excluded: validation showed they hurt ECE (0.058 → 0.069) due to thin
    # sample per series-state. Re-evaluate with more historical data.
]

XGB_MODEL_PATH = MODEL_PATH.parent / "xgb_game_model.json"
XGB_MODEL_BINARY_PATH = MODEL_PATH.parent / "xgb_game_model.joblib"

HOME_MARGIN_POINTS = 2.2
REST_MARGIN_POINTS_PER_DAY = 0.35
MAX_REST_MARGIN_POINTS = 2.0
GAME_MARGIN_LOGIT_SCALE = 11.5


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _home_from_matchup(matchup: str, team_abbr: str) -> float:
    """1.0 if team played at home, 0.0 if away."""
    if " vs. " in matchup:
        return 1.0
    return 0.0


def _rest_days(dates: list[str], idx: int) -> float:
    """Days since previous game for team at position idx."""
    if idx == 0:
        return 3.0
    try:
        from datetime import date
        d1 = date.fromisoformat(str(dates[idx])[:10])
        d0 = date.fromisoformat(str(dates[idx - 1])[:10])
        return float(max((d1 - d0).days, 1))
    except (ValueError, TypeError):
        return 2.0


def build_training_rows(
    game_logs: list[dict[str, Any]],
    team_ratings: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build canonical rows containing only information known before tipoff."""
    return build_canonical_pregame_rows(game_logs, team_ratings)


# ---------------------------------------------------------------------------
# Model training and persistence
# ---------------------------------------------------------------------------

def train(
    rows: list[dict[str, Any]],
    holdout_seasons: list[str] | None = None,
) -> dict[str, Any]:
    """Train a calibrated logistic model on historical game rows.

    Returns a model bundle with a joblib classifier and JSON metadata.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    if holdout_seasons is None:
        holdout_seasons = ["2024-25"]

    train_rows = [r for r in rows if r.get("season") not in holdout_seasons]
    test_rows = [r for r in rows if r.get("season") in holdout_seasons]

    if len(train_rows) < 50:
        raise ValueError(f"Not enough training rows: {len(train_rows)}")

    X_train = np.array([[r[f] for f in FEATURE_COLS] for r in train_rows])
    y_train = np.array([r["won"] for r in train_rows])
    groups = np.array([
        f"{row.get('season', '')}:{row.get('game_id', index)}"
        for index, row in enumerate(train_rows)
    ])

    base_lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
    )
    calibration_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups))
    lr_model = CalibratedClassifierCV(
        base_lr,
        cv=calibration_splits,
        method="sigmoid",
    )
    lr_model.fit(X_train, y_train)

    # Evaluate on holdout
    metrics: dict[str, Any] = {
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "holdout_seasons": holdout_seasons,
    }

    if test_rows:
        X_test = np.array([[r[f] for f in FEATURE_COLS] for r in test_rows])
        y_test = np.array([r["won"] for r in test_rows])
        structural_probs = lr_model.predict_proba(X_test)[:, 1]
        lr_probs = np.array([
            _apply_context_adjustments(
                float(probability),
                float(row.get("home_court", 0.0)),
                float(row.get("rest_diff", 0.0)),
            )
            for probability, row in zip(structural_probs, test_rows)
        ])

        correct = int(np.sum((lr_probs > 0.5) == y_test))
        metrics["holdout_accuracy"] = round(correct / len(y_test), 3)
        metrics["holdout_brier"] = round(
            float(np.mean((lr_probs - y_test) ** 2)), 4
        )

    return {
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "_lr_model_ref": lr_model,
    }


def _binary_model_path(path: Path) -> Path:
    if path == MODEL_PATH:
        return MODEL_BINARY_PATH
    return path.with_suffix(".joblib")


def save_model(bundle: dict[str, Any], path: Path = MODEL_PATH) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    binary_path = _binary_model_path(path)
    joblib.dump(bundle, binary_path)
    serializable = {
        k: v for k, v in bundle.items()
        if not k.startswith("_")
    }
    serializable["model_format"] = "joblib_calibrated_classifier"
    serializable["binary_artifact"] = binary_path.name
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_model(path: Path = MODEL_PATH) -> dict[str, Any] | None:
    binary_path = _binary_model_path(path)
    if binary_path.exists():
        import joblib
        return joblib.load(binary_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("model_format") == "joblib_calibrated_classifier":
        raise FileNotFoundError(
            f"Binary model artifact missing: {binary_path}. "
            "Run `python scripts/train_game_model.py` to rebuild."
        )
    return None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def train_xgb(
    rows: list[dict[str, Any]],
    holdout_seasons: list[str] | None = None,
) -> dict[str, Any]:
    """Train a calibrated XGBoost model using XGB_FEATURE_COLS (includes injury)."""
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier

    if holdout_seasons is None:
        holdout_seasons = ["2024-25"]

    train_rows = [r for r in rows if r.get("season") not in holdout_seasons]
    if len(train_rows) < 50:
        raise ValueError(f"Not enough training rows: {len(train_rows)}")

    X_train = np.array([[float(r.get(f, 0) or 0) for f in XGB_FEATURE_COLS] for r in train_rows])
    y_train = np.array([r["won"] for r in train_rows])
    groups = np.array([
        f"{r.get('season', '')}:{r.get('game_id', i)}"
        for i, r in enumerate(train_rows)
    ])
    cv_splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups))

    base_xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=2.0, random_state=42, verbosity=0,
    )
    xgb_model = CalibratedClassifierCV(base_xgb, cv=cv_splits, method="sigmoid")
    xgb_model.fit(X_train, y_train)

    return {"feature_cols": XGB_FEATURE_COLS, "_xgb_model_ref": xgb_model}


def save_xgb_model(bundle: dict[str, Any]) -> None:
    import joblib
    XGB_MODEL_BINARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, XGB_MODEL_BINARY_PATH)
    serializable = {k: v for k, v in bundle.items() if not k.startswith("_")}
    serializable["binary_artifact"] = XGB_MODEL_BINARY_PATH.name
    XGB_MODEL_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_xgb_model() -> dict[str, Any] | None:
    if not XGB_MODEL_BINARY_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(XGB_MODEL_BINARY_PATH)
    except Exception:
        return None


def predict_xgb_win_probability(
    row: dict[str, Any],
    bundle: dict[str, Any],
) -> float:
    import numpy as np
    xgb = bundle.get("_xgb_model_ref")
    if xgb is None:
        return 0.5
    feat = bundle.get("feature_cols", XGB_FEATURE_COLS)
    X = np.array([[float(row.get(f, 0) or 0) for f in feat]])
    return float(xgb.predict_proba(X)[0, 1])


def _logit(probability: float) -> float:
    probability = max(1e-6, min(1.0 - 1e-6, probability))
    return math.log(probability / (1.0 - probability))


def _apply_context_adjustments(
    probability: float,
    home_court: float,
    rest_diff: float,
) -> float:
    """Apply rest-advantage margin to a structural probability.

    Home court advantage is learned directly by the calibrated model via the
    home_court feature in FEATURE_COLS, so only the rest adjustment is applied
    here to avoid double-counting.
    """
    rest_margin = max(
        -MAX_REST_MARGIN_POINTS,
        min(MAX_REST_MARGIN_POINTS, rest_diff * REST_MARGIN_POINTS_PER_DAY),
    )
    adjusted_logit = _logit(probability) + rest_margin / GAME_MARGIN_LOGIT_SCALE
    return _sigmoid(adjusted_logit)


def predict_win_probability(
    features: dict[str, float],
    model_bundle: dict[str, Any],
    team_a_id: str | None = None,
    team_b_id: str | None = None,
) -> float:
    """P(team_a wins) given feature dict and loaded model bundle.

    Uses the complete calibrated classifier persisted by ``save_model``.
    Legacy JSON-only artifacts are rejected because they do not contain the
    calibration mapping that was evaluated during training.
    """
    import numpy as np

    feat_vec = np.array([[features.get(f, 0.0) for f in FEATURE_COLS]])

    # LR prediction
    lr_ref = model_bundle.get("_lr_model_ref")

    if lr_ref is not None:
        structural_prob = float(lr_ref.predict_proba(feat_vec)[0, 1])
        lr_prob = _apply_context_adjustments(
            structural_prob,
            float(features.get("home_court", 0.0)),
            float(features.get("rest_diff", 0.0)),
        )
    else:
        raise ValueError(
            "The game model is a legacy JSON-only artifact. "
            "Run `python scripts/train_game_model.py` to rebuild the calibrated model."
        )

    return round(max(0.05, min(0.95, lr_prob)), 4)


def _binary_metrics(y_true: list[int], probabilities: list[float]) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {
        "games": int(len(y)),
        "accuracy": round(float(accuracy_score(y, p >= 0.5)), 4),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
        "brier_score": round(float(brier_score_loss(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(set(y.tolist())) == 2 else None,
        "expected_calibration_error": round(ece, 4),
    }


def _net_rating_baseline_probability(row: dict[str, Any]) -> float:
    structural = _sigmoid(float(row.get("regular_net_rating_diff", 0.0)) / GAME_MARGIN_LOGIT_SCALE)
    return _apply_context_adjustments(
        structural,
        float(row.get("home_court", 0.0)),
        float(row.get("rest_diff", 0.0)),
    )


def _opening_series_accuracy(
    rows: list[dict[str, Any]],
    probability_key: str,
) -> dict[str, Any]:
    series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault(str(row.get("series_id")), []).append(row)
    correct = 0
    scored = 0
    for series_rows in series.values():
        home_rows = [
            row for row in series_rows
            if row.get("perspective") == "home"
        ]
        if not home_rows:
            continue
        opening = min(home_rows, key=lambda row: (str(row.get("game_date")), str(row.get("game_id"))))
        predicted = (
            str(opening.get("team_a"))
            if float(opening.get(probability_key, 0.5)) >= 0.5
            else str(opening.get("team_b"))
        )
        wins: dict[str, int] = {}
        for row in home_rows:
            winner = str(row.get("team_a")) if int(row.get("actual_team_a_win", row.get("won", 0))) else str(row.get("team_b"))
            wins[winner] = wins.get(winner, 0) + 1
        if not wins:
            continue
        actual = max(wins, key=wins.get)
        correct += int(predicted == actual)
        scored += 1
    return {
        "series": scored,
        "accuracy": round(correct / scored, 4) if scored else None,
    }


def _elo_predictions(
    rows: list[dict[str, Any]],
    test_season: str,
) -> dict[str, float]:
    """Generate sequential pregame Elo probabilities for one test season."""
    ratings: dict[str, float] = {}
    predictions: dict[str, float] = {}
    current_season = ""
    canonical = sorted(
        (row for row in rows if row.get("perspective") == "home"),
        key=lambda row: (str(row.get("season")), str(row.get("game_date")), str(row.get("game_id"))),
    )
    for row in canonical:
        season = str(row.get("season"))
        if season != current_season:
            ratings = {team: 0.75 * rating + 0.25 * 1500.0 for team, rating in ratings.items()}
            current_season = season
        home = str(row["actual_home_team_id"])
        away = str(row["actual_away_team_id"])
        home_rating = ratings.get(home, 1500.0)
        away_rating = ratings.get(away, 1500.0)
        home_probability = 1.0 / (
            1.0 + 10.0 ** ((away_rating - home_rating - 65.0) / 400.0)
        )
        if season == test_season:
            predictions[f"{season}:{row['game_id']}:home"] = home_probability
            predictions[f"{season}:{row['game_id']}:away"] = 1.0 - home_probability
        outcome = int(row["won"])
        margin = abs(
            float(row.get("team_score", 0.0))
            - float(row.get("opponent_score", 0.0))
        )
        multiplier = max(1.0, math.log(max(margin, 1.0) + 1.0))
        change = 20.0 * multiplier * (outcome - home_probability)
        ratings[home] = home_rating + change
        ratings[away] = away_rating - change
    return predictions


def walk_forward_backtest(
    rows: list[dict[str, Any]],
    test_seasons: list[str] | None = None,
) -> dict[str, Any]:
    """Train only on earlier seasons and score each later playoff season.

    Validates three models: LR (baseline), XGBoost (with injury), and their
    50/50 ensemble. The ensemble is the production model.
    """
    from src.data.fetch_injury_proxy import _load_cache as _load_injury_cache, get_injury_diff_for_row
    from src.data.fetch_player_projections import build_player_projections, get_player_projection_edge

    seasons = sorted({str(row.get("season")) for row in rows})
    test_seasons = test_seasons or seasons[-4:]

    # Load historical component data for meta model training
    injury_cache = _load_injury_cache()
    player_projections = build_player_projections(seasons, verbose=False)

    INJURY_PTS_SCALE = 15.0  # injury pts_share → pts margin conversion

    split_reports = []
    oof_rows: list[dict[str, Any]] = []

    for test_season in test_seasons:
        eligible_seasons = [season for season in seasons if season <= test_season]
        train_seasons = [season for season in eligible_seasons if season < test_season]
        eligible_rows = [
            row for row in rows
            if str(row.get("season")) in set(train_seasons + [test_season])
        ]
        test_rows = [
            row for row in eligible_rows
            if str(row.get("season")) == test_season
        ]
        if len(train_seasons) < 3 or not test_rows:
            continue

        # LR model (FEATURE_COLS only, no injury)
        lr_bundle = train(eligible_rows, holdout_seasons=[test_season])

        # XGBoost model (XGB_FEATURE_COLS including injury)
        xgb_bundle = train_xgb(eligible_rows, holdout_seasons=[test_season])

        elo = _elo_predictions(eligible_rows, test_season)
        targets, lr_probs, xgb_probs, ens_probs, net_probs, elo_probs = [], [], [], [], [], []

        for row in test_rows:
            lr_p = predict_win_probability(
                row, lr_bundle,
                str(row.get("team_a_id")), str(row.get("team_b_id")),
            )
            xgb_p = predict_xgb_win_probability(row, xgb_bundle)
            ens_p = round(0.5 * lr_p + 0.5 * xgb_p, 6)

            identity = f"{test_season}:{row['game_id']}:{row['perspective']}"
            elo_p = elo.get(identity, 0.5)
            net_p = _net_rating_baseline_probability(row)
            target = int(row["won"])

            targets.append(target)
            lr_probs.append(lr_p)
            xgb_probs.append(xgb_p)
            ens_probs.append(ens_p)
            net_probs.append(net_p)
            elo_probs.append(elo_p)

            # Compute historical component values for meta-model training
            inj_diff, inj_avail = get_injury_diff_for_row(
                row["game_id"], str(row.get("team_a", "")), injury_cache
            )
            # Positive inj_diff means team_a is more injured → negative edge
            inj_edge = round(-inj_diff * INJURY_PTS_SCALE, 3)

            player_edge_pts, player_avail = get_player_projection_edge(
                test_season,
                str(row.get("team_a", "")),
                str(row.get("team_b", "")),
                player_projections,
            )

            oof_rows.append({
                "season": test_season,
                "series_id": row.get("series_id"),
                "game_id": row["game_id"],
                "game_date": row["game_date"],
                "perspective": row["perspective"],
                "team_a": row.get("team_a"),
                "team_b": row.get("team_b"),
                "actual_team_a_win": target,
                # baseline_probability = LR only (original meaning for report clarity)
                # ensemble_probability used by meta model as its baseline logit
                "baseline_probability": lr_p,
                "xgb_probability": xgb_p,
                "ensemble_probability": ens_p,
                "net_rating_probability": round(net_p, 6),
                "elo_probability": round(elo_p, 6),
                "player_edge": player_edge_pts,
                "matchup_edge": 0.0,
                "lineup_edge": 0.0,
                "clutch_edge": 0.0,
                "injury_edge": inj_edge,
                "coaching_edge": 0.0,
                "player_data_available": player_avail,
                "matchup_data_available": 0,
                "lineup_data_available": int(row.get("lineup_data_available", 0)),
                "clutch_data_available": 0,
                "injury_data_available": inj_avail,
                "coaching_data_available": 0,
            })

        split_reports.append({
            "test_season": test_season,
            "train_seasons": train_seasons,
            "model": _binary_metrics(targets, lr_probs),
            "xgb_model": _binary_metrics(targets, xgb_probs),
            "ensemble": _binary_metrics(targets, ens_probs),
            "net_rating_baseline": _binary_metrics(targets, net_probs),
            "elo_baseline": _binary_metrics(targets, elo_probs),
        })

    targets = [int(row["actual_team_a_win"]) for row in oof_rows]
    overall: dict[str, Any] = {}
    if oof_rows:
        for key, prob_key in [
            ("model", "baseline_probability"),
            ("xgb_model", "xgb_probability"),
            ("ensemble", "ensemble_probability"),
            ("net_rating_baseline", "net_rating_probability"),
            ("elo_baseline", "elo_probability"),
        ]:
            probs = [float(row[prob_key]) for row in oof_rows]
            overall[key] = _binary_metrics(targets, probs)
            overall[key]["opening_series_winner"] = _opening_series_accuracy(oof_rows, prob_key)

    return {
        "validation": "chronological_walk_forward",
        "production_model": "ensemble",
        "splits": split_reports,
        "overall": overall,
        "oof_predictions": oof_rows,
    }
