"""Uncertainty ranges for game and player projections."""

from __future__ import annotations

import json
from math import ceil, isnan
from pathlib import Path
from statistics import pstdev
from typing import Any

from src.data.build_dataset import build_finals_context
from src.models.predict_game import predict_finals_games
from src.models.train_player_model import project_finals_players


COUNTING_STATS = ["points", "rebounds", "assists", "turnovers", "steals", "blocks"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMPIRICAL_UNCERTAINTY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stats_cache" / "probability_uncertainty.json"
)


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


def _iter_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return [dict(row) for row in data.to_dict(orient="records")]
    if isinstance(data, dict):
        if "player" in data or "team" in data:
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


def _format_probability_range(low: float, high: float) -> str:
    return f"{round(low * 100):.0f}%–{round(high * 100):.0f}%"


def _format_score_range(low: int, high: int) -> str:
    return f"{low}–{high}"


def _model_probability_spread(game_prediction: dict[str, Any]) -> float:
    baseline_predictions = game_prediction.get("baseline_model_predictions", [])
    probabilities = []
    for prediction in baseline_predictions:
        model_probabilities = prediction.get("model_probabilities", {})
        probabilities.extend(_as_float(value) for value in model_probabilities.values())
    if len(probabilities) < 2:
        return 0.0
    return pstdev(probabilities)


def _component_margin_spread(game_prediction: dict[str, Any]) -> float:
    margins = [
        _as_float(value)
        for value in game_prediction.get("component_margins", {}).values()
    ]
    if len(margins) < 2:
        return 0.0
    return pstdev(margins)


def fit_empirical_probability_uncertainty(
    oof_rows: list[dict[str, Any]],
    probability_key: str = "net_rating_probability",
) -> dict[str, Any]:
    """Fit probability-range widths from chronological out-of-fold residuals."""
    bins = []
    for index in range(10):
        low, high = index / 10.0, (index + 1) / 10.0
        rows = [
            row for row in oof_rows
            if low <= _as_float(row.get(probability_key), 0.5)
            < (high if high < 1.0 else 1.000001)
        ]
        if not rows:
            continue
        predicted = sum(_as_float(row.get(probability_key), 0.5) for row in rows) / len(rows)
        observed = sum(int(row.get("actual_team_a_win", 0)) for row in rows) / len(rows)
        standard_error = (max(observed * (1.0 - observed), 0.01) / len(rows)) ** 0.5
        width = _clip(max(abs(observed - predicted), 1.64 * standard_error, 0.035), 0.035, 0.12)
        bins.append({
            "low": low,
            "high": high,
            "count": len(rows),
            "mean_prediction": round(predicted, 4),
            "observed_rate": round(observed, 4),
            "half_width": round(width, 4),
        })
    return {
        "method": "walk_forward_probability_bucket_residuals",
        "probability_key": probability_key,
        "rows": len(oof_rows),
        "bins": bins,
    }


def save_empirical_probability_uncertainty(
    artifact: dict[str, Any],
    path: Path = EMPIRICAL_UNCERTAINTY_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def _empirical_probability_width(probability: float) -> float | None:
    if not EMPIRICAL_UNCERTAINTY_PATH.exists():
        return None
    artifact = json.loads(EMPIRICAL_UNCERTAINTY_PATH.read_text(encoding="utf-8"))
    for row in artifact.get("bins", []):
        if float(row["low"]) <= probability < float(row["high"]):
            return float(row["half_width"])
    return None


def estimate_win_probability_width(game_prediction: dict[str, Any]) -> float:
    """Estimate a realistic probability half-width for a game projection."""
    probability = _as_float(game_prediction.get("team_a_win_probability"), 0.5)
    empirical_width = _empirical_probability_width(probability)
    if empirical_width is not None:
        return round(empirical_width, 4)
    closeness_bonus = (0.5 - abs(probability - 0.5)) * 0.025
    component_bonus = min(_component_margin_spread(game_prediction) * 0.006, 0.020)
    model_bonus = min(_model_probability_spread(game_prediction) * 0.25, 0.020)
    x_factor_bonus = min(len(game_prediction.get("x_factors", [])) * 0.002, 0.012)
    baseline_width = 0.025
    return round(_clip(baseline_width + closeness_bonus + component_bonus + model_bonus + x_factor_bonus, 0.035, 0.095), 4)


def win_probability_range(game_prediction: dict[str, Any]) -> dict[str, Any]:
    """Build win-probability ranges for both teams."""
    team_a = game_prediction["team_a"]
    team_b = game_prediction["team_b"]
    team_a_center = _as_float(game_prediction.get("team_a_win_probability"), 0.5)
    width = estimate_win_probability_width(game_prediction)
    team_a_low = _clip(team_a_center - width, 0.01, 0.99)
    team_a_high = _clip(team_a_center + width, 0.01, 0.99)
    team_b_low = 1.0 - team_a_high
    team_b_high = 1.0 - team_a_low

    return {
        team_a: {
            "center": round(team_a_center, 3),
            "low": round(team_a_low, 3),
            "high": round(team_a_high, 3),
            "label": _format_probability_range(team_a_low, team_a_high),
        },
        team_b: {
            "center": round(1.0 - team_a_center, 3),
            "low": round(team_b_low, 3),
            "high": round(team_b_high, 3),
            "label": _format_probability_range(team_b_low, team_b_high),
        },
    }


def estimate_score_width(game_prediction: dict[str, Any]) -> int:
    """Estimate a likely score-range half-width."""
    pace = _as_float(game_prediction.get("projected_pace"), 98.5)
    pace_bonus = max(pace - 97.0, 0.0) * 0.05
    margin_spread_bonus = min(_component_margin_spread(game_prediction) * 0.45, 2.5)
    x_factor_bonus = min(len(game_prediction.get("x_factors", [])) * 0.25, 1.5)
    width = 7.5 + pace_bonus + margin_spread_bonus + x_factor_bonus
    return int(round(_clip(width, 7.0, 13.0)))


def score_range(game_prediction: dict[str, Any]) -> dict[str, Any]:
    """Build expected score ranges for both teams."""
    team_a = game_prediction["team_a"]
    team_b = game_prediction["team_b"]
    score_a = int(game_prediction["expected_score_team_a"])
    score_b = int(game_prediction["expected_score_team_b"])
    width = estimate_score_width(game_prediction)

    return {
        team_a: {
            "center": score_a,
            "low": max(score_a - width, 70),
            "high": score_a + width,
            "label": _format_score_range(max(score_a - width, 70), score_a + width),
        },
        team_b: {
            "center": score_b,
            "low": max(score_b - width, 70),
            "high": score_b + width,
            "label": _format_score_range(max(score_b - width, 70), score_b + width),
        },
    }


def add_game_uncertainty(game_prediction: dict[str, Any]) -> dict[str, Any]:
    """Attach probability and score ranges to one game prediction."""
    probability_ranges = win_probability_range(game_prediction)
    score_ranges = score_range(game_prediction)
    team_a = game_prediction["team_a"]
    team_b = game_prediction["team_b"]

    return {
        **game_prediction,
        "team_a_win_probability_range": probability_ranges[team_a],
        "team_b_win_probability_range": probability_ranges[team_b],
        "realistic_win_probability_range": probability_ranges,
        "likely_score_range": score_ranges,
        "expected_score_range_team_a": score_ranges[team_a],
        "expected_score_range_team_b": score_ranges[team_b],
        "uncertainty_drivers": uncertainty_drivers(game_prediction),
    }


def uncertainty_drivers(game_prediction: dict[str, Any]) -> list[str]:
    """Name the biggest reasons the game projection should be treated as a range."""
    drivers = []
    if abs(_as_float(game_prediction.get("team_a_win_probability"), 0.5) - 0.5) <= 0.06:
        drivers.append("Game projects close to a toss-up")
    if _component_margin_spread(game_prediction) >= 2.0:
        drivers.append("Model components disagree on the size of the edge")
    if _model_probability_spread(game_prediction) >= 0.04:
        drivers.append("Baseline model probabilities have meaningful spread")
    if game_prediction.get("x_factors"):
        drivers.append("X-factors can move the projection")
    if not drivers:
        drivers.append("Normal NBA game-to-game variance")
    return drivers


def _player_minutes(player_projection: dict[str, Any]) -> float:
    return _as_float(player_projection.get("minutes") or player_projection.get("projected_minutes"), 0.0)


def _minutes_range(player_projection: dict[str, Any]) -> tuple[float, float]:
    minutes = _player_minutes(player_projection)
    floor = _as_float(player_projection.get("minutes_floor"), minutes - 3.0)
    ceiling = _as_float(player_projection.get("minutes_ceiling"), minutes + 3.0)
    if player_projection.get("injury_status") not in (None, "", "Available", "Probable"):
        floor -= 2.0
        ceiling += 2.0
    return max(0.0, min(floor, minutes)), max(minutes, ceiling)


def player_stat_range(
    player_projection: dict[str, Any],
    stat: str,
) -> dict[str, Any]:
    """Estimate a realistic range for one player stat projection."""
    projected = _as_float(player_projection.get(stat), 0.0)
    minutes = max(_player_minutes(player_projection), 1.0)
    floor_minutes, ceiling_minutes = _minutes_range(player_projection)
    per_minute = projected / minutes

    if stat == "points":
        volatility = max(2.2, projected * 0.11)
    elif stat in {"rebounds", "assists"}:
        volatility = max(1.2, projected * 0.18)
    elif stat == "turnovers":
        volatility = max(0.8, projected * 0.22)
    else:
        volatility = max(0.6, projected * 0.28)

    if str(player_projection.get("projection_method", "rate_based")) == "rate_based":
        volatility *= 1.10
    if player_projection.get("injury_status") not in (None, "", "Available", "Probable"):
        volatility *= 1.20

    low = max(0.0, per_minute * floor_minutes - volatility)
    high = max(low, per_minute * ceiling_minutes + volatility)

    if stat == "points":
        low_value = int(round(low))
        high_value = int(ceil(high))
    else:
        low_value = round(low, 1)
        high_value = round(high, 1)

    return {
        "center": round(projected, 1),
        "low": low_value,
        "high": high_value,
        "label": _format_score_range(low_value, high_value),
    }


def add_player_uncertainty(player_projection: dict[str, Any]) -> dict[str, Any]:
    """Attach realistic stat ranges to one player projection."""
    ranges = {
        stat: player_stat_range(player_projection, stat)
        for stat in COUNTING_STATS
        if stat in player_projection
    }
    return {
        **player_projection,
        "stat_ranges": ranges,
        "points_range": ranges.get("points"),
    }


def add_player_projection_uncertainty(player_projections: Any) -> dict[str, list[dict[str, Any]]]:
    """Attach player uncertainty ranges while preserving team grouping."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _iter_records(player_projections):
        team = str(row.get("team") or "")
        grouped.setdefault(team, []).append(add_player_uncertainty(row))
    for team in grouped:
        grouped[team] = sorted(grouped[team], key=lambda row: _player_minutes(row), reverse=True)
    return grouped


def build_uncertainty_report(
    game_predictions: list[dict[str, Any]],
    player_projections: Any | None = None,
) -> dict[str, Any]:
    """Build a full uncertainty report for games and optional player projections."""
    games = [add_game_uncertainty(prediction) for prediction in game_predictions]
    report = {
        "games": games,
        "series_probability_range": series_probability_range(games),
    }
    if player_projections is not None:
        report["players"] = add_player_projection_uncertainty(player_projections)
    return report


def series_probability_range(game_predictions_with_uncertainty: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize game-level probability ranges into an average series signal range."""
    if not game_predictions_with_uncertainty:
        return {}
    team_a = game_predictions_with_uncertainty[0]["team_a"]
    team_b = game_predictions_with_uncertainty[0]["team_b"]
    lows = [game["team_a_win_probability_range"]["low"] for game in game_predictions_with_uncertainty]
    centers = [game["team_a_win_probability"] for game in game_predictions_with_uncertainty]
    highs = [game["team_a_win_probability_range"]["high"] for game in game_predictions_with_uncertainty]
    low = sum(lows) / len(lows)
    center = sum(centers) / len(centers)
    high = sum(highs) / len(highs)
    return {
        team_a: {
            "center": round(center, 3),
            "low": round(low, 3),
            "high": round(high, 3),
            "label": _format_probability_range(low, high),
        },
        team_b: {
            "center": round(1.0 - center, 3),
            "low": round(1.0 - high, 3),
            "high": round(1.0 - low, 3),
            "label": _format_probability_range(1.0 - high, 1.0 - low),
        },
    }


if __name__ == "__main__":
    context = build_finals_context()
    game_predictions = predict_finals_games(context)
    player_projections = project_finals_players(context)
    report = build_uncertainty_report(game_predictions, player_projections)
    first_game = report["games"][0]
    print(
        f"Game {first_game['game_number']} {first_game['team_a']} win probability: "
        f"{first_game['team_a_win_probability']:.1%}"
    )
    print(f"Realistic range: {first_game['team_a_win_probability_range']['label']}")
    print(
        f"Likely score range: {first_game['team_a']} "
        f"{first_game['expected_score_range_team_a']['label']}, "
        f"{first_game['team_b']} {first_game['expected_score_range_team_b']['label']}"
    )
