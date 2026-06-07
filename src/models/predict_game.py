"""Finals game prediction engine.

Combines the grounding team baseline with basketball adjustment engines:
player projections, matchup edges, lineup strength, clutch/closing lineup
edge, injuries, foul-trouble scenarios, home/rest context, and expected pace.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from math import exp, isnan, log
from pathlib import Path
from typing import Any

import yaml

from src.data.build_dataset import build_finals_context
from src.data.build_historical_dataset import METRIC_PRIOR_POSSESSIONS
from src.features.lineup_features import build_lineup_features
from src.features.matchup_features import build_matchup_edges
from src.features.playstyle_features import build_playstyle_profiles
from src.models.closing_lineup_model import predict_close_game_edge
from src.models.foul_trouble_simulator import simulate_foul_trouble_scenarios
from src.models.game_model import load_model as _load_game_model, predict_win_probability as _ml_win_prob
from src.models.meta_model import load_meta_model, predict_meta_probability
from src.models.train_player_model import project_finals_players, summarize_player_projections
from src.models.train_team_model import predict_baseline_without_training, predict_team_win_probability


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_WEIGHTS_PATH = PROJECT_ROOT / "config" / "model_weights.yaml"
MODEL_VERSION = "calibrated-finals-v3"

DEFAULT_WEIGHTS = {
    "team_baseline": 1.0,
    "player_projection": 0.45,
    "matchup_edge": 0.20,
    "lineup_edge": 0.25,
    "clutch_edge": 0.10,
    "injury_edge": 0.0,
    "foul_trouble_risk": 0.0,
}

RESIDUAL_MARGIN_CAP = 4.0
GAME_MARGIN_LOGIT_SCALE = 11.5
CLUTCH_GAME_RATE = 0.30
BLENDED_TEAM_METRICS = {
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "EFG_PCT",
    "TM_TOV_PCT",
    "OREB_PCT",
    "FTA_RATE",
    "PACE",
}


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


def _logit(probability: float) -> float:
    probability = _clip(probability, 0.01, 0.99)
    return log(probability / (1.0 - probability))


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def _margin_to_logit(margin_points: float) -> float:
    return margin_points / GAME_MARGIN_LOGIT_SCALE


def load_game_prediction_weights(
    model_weights_path: str | Path = DEFAULT_MODEL_WEIGHTS_PATH,
) -> dict[str, float]:
    """Load and normalize final game prediction weights."""
    path = Path(model_weights_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        weights = {
            **DEFAULT_WEIGHTS,
            **(loaded.get("final_game_prediction_weights") or {}),
        }
    else:
        weights = dict(DEFAULT_WEIGHTS)

    residual_keys = [key for key in weights if key != "team_baseline"]
    residual_total = sum(max(_as_float(weights[key]), 0.0) for key in residual_keys)
    if residual_total <= 0:
        return dict(DEFAULT_WEIGHTS)
    normalized = {
        key: max(_as_float(weights[key]), 0.0) / residual_total
        for key in residual_keys
    }
    return {"team_baseline": 1.0, **normalized}


def _combination_mode(model_weights_path: str | Path) -> str:
    path = Path(model_weights_path)
    if not path.exists():
        return "learned_meta"
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return str(loaded.get("combination_mode") or "learned_meta")


def _iter_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return [dict(row) for row in data.to_dict(orient="records")]
    if isinstance(data, dict):
        if "team" in data or "TEAM" in data:
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


def _team_index(data: Any) -> dict[str, dict[str, Any]]:
    index = {}
    for row in _iter_records(data):
        team = str(
            row.get("team")
            or row.get("TEAM")
            or row.get("TEAM_ABBREVIATION")
            or row.get("TEAM_NAME")
            or ""
        ).strip()
        if team:
            index[team] = row
    return index


def _prediction_team_stats(
    finals_context: dict[str, Any],
    team_stats: Any = None,
) -> dict[str, dict[str, Any]]:
    """Blend playoff form with the larger regular-season prior."""
    playoff = _team_index(
        team_stats
        or finals_context.get("playoff_team_stats")
        or finals_context.get("team_stats")
    )
    regular = _team_index(finals_context.get("regular_season_team_stats"))
    teams = [str(finals_context["team_a"]), str(finals_context["team_b"])]
    blended: dict[str, dict[str, Any]] = {}
    for team in teams:
        playoff_row = playoff.get(team, {})
        regular_row = regular.get(team, {})
        playoff_possessions = _as_float(
            playoff_row.get("POSS"),
            _as_float(playoff_row.get("GP"), 0.0) * _as_float(playoff_row.get("PACE"), 98.0),
        )
        row = {**regular_row, **playoff_row}
        for metric in BLENDED_TEAM_METRICS:
            playoff_value = playoff_row.get(metric)
            regular_value = regular_row.get(metric)
            metric_name = {
                "OFF_RATING": "net_rating",
                "DEF_RATING": "net_rating",
                "NET_RATING": "net_rating",
                "EFG_PCT": "efg_pct",
                "TM_TOV_PCT": "tov_pct",
                "OREB_PCT": "oreb_pct",
                "FTA_RATE": "fta_rate",
                "PACE": "pace",
            }[metric]
            prior_possessions = METRIC_PRIOR_POSSESSIONS[metric_name]
            playoff_weight = (
                playoff_possessions / (playoff_possessions + prior_possessions)
                if playoff_possessions > 0
                else 0.0
            )
            row[f"REGULAR_{metric}"] = regular_value
            row[f"PLAYOFF_{metric}"] = playoff_value
            row[f"{metric}_PLAYOFF_WEIGHT"] = round(playoff_weight, 4)
            if playoff_value is None and regular_value is None:
                continue
            if playoff_value is None:
                row[metric] = regular_value
            elif regular_value is None:
                row[metric] = playoff_value
            else:
                row[metric] = (
                    playoff_weight * _as_float(playoff_value)
                    + (1.0 - playoff_weight) * _as_float(regular_value)
                )
        row["PLAYOFF_POSSESSIONS"] = round(playoff_possessions, 1)
        blended[team] = row
    return blended


def _baseline_team_stats(
    finals_context: dict[str, Any],
    team_stats: Any = None,
) -> dict[str, dict[str, Any]]:
    """Return regular priors plus metric-specific playoff-shrunk ratings."""
    return _prediction_team_stats(finals_context, team_stats)


def _metric(row: dict[str, Any] | None, keys: list[str], default: float) -> float:
    if not row:
        return default
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return _as_float(row[key], default)
    return default


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value in (None, ""):
        return None
    return datetime.fromisoformat(str(value)).date()


def _game_rest_days(schedule: list[dict[str, Any]], game_number: int) -> float:
    if game_number <= 1:
        return 3.0
    current = next((game for game in schedule if int(game["game_number"]) == game_number), None)
    previous = next((game for game in schedule if int(game["game_number"]) == game_number - 1), None)
    if not current or not previous:
        return 2.0
    current_date = _parse_date(current.get("date"))
    previous_date = _parse_date(previous.get("date"))
    if not current_date or not previous_date:
        return 2.0
    return max(float((current_date - previous_date).days), 1.0)


def _team_rest_days(
    team: str,
    game_number: int,
    schedule: list[dict[str, Any]],
    finals_context: dict[str, Any],
) -> float:
    """Per-team rest days. Game 1 uses actual pre-series rest from context."""
    if game_number <= 1:
        pre_series = finals_context.get("pre_series_rest") or {}
        return float(pre_series.get(team, 3.0))
    return _game_rest_days(schedule, game_number)


def _rest_ortg_boost(rest_days: float) -> float:
    """ORTG adjustment vs the 3-day baseline. Each extra rest day ≈ +0.4 pts."""
    return round(_clip((rest_days - 3.0) * 0.4, -3.0, 3.0), 2)


def _series_pace_decay(game_number: int) -> float:
    """Finals series slow down as both staffs adjust. -0.25 pace per game."""
    return round((int(game_number) - 1) * -0.25, 2)


def _opponent(finals_context: dict[str, Any], team: str) -> str:
    team_a = str(finals_context.get("team_a") or "")
    team_b = str(finals_context.get("team_b") or "")
    return team_b if team == team_a else team_a


def _pace_from_profiles(
    playstyle_profiles: dict[str, dict[str, Any]],
    team_a: str,
    team_b: str,
) -> float:
    # Default 96.5: Finals pace is consistently slower than regular season
    # (recent Finals average ~95-97, vs league avg ~98-100).
    pace_a = _as_float(
        playstyle_profiles.get(team_a, {})
        .get("offense", {})
        .get("metrics", {})
        .get("pace", {})
        .get("value"),
        96.5,
    )
    pace_b = _as_float(
        playstyle_profiles.get(team_b, {})
        .get("offense", {})
        .get("metrics", {})
        .get("pace", {})
        .get("value"),
        96.5,
    )
    return round((pace_a + pace_b) / 2.0, 1)


def _schedule_context_rows(
    finals_context: dict[str, Any],
    game: dict[str, Any],
    team_stats: Any,
    player_summary: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    team_a = str(finals_context["team_a"])
    team_b = str(finals_context["team_b"])
    stats_index = _prediction_team_stats(finals_context, team_stats)
    rest_days = _game_rest_days(finals_context.get("schedule", []), int(game["game_number"]))

    rows = []
    for team in (team_a, team_b):
        opponent = _opponent(finals_context, team)
        row = stats_index.get(team, {})
        opponent_row = stats_index.get(opponent, {})
        team_points = player_summary.get(team, {}).get("points", 0.0)
        opponent_points = player_summary.get(opponent, {}).get("points", 0.0)
        projected_strength = (team_points - opponent_points) * 0.15
        home_court = 0.0
        if not bool(game.get("neutral_site")) and str(game.get("home_team")) == team:
            home_court = 1.0

        rows.append(
            {
                "team": team,
                "opponent": opponent,
                "team_net_rating": _metric(row, ["team_net_rating", "net_rating", "NET_RATING"], projected_strength),
                "opponent_net_rating": _metric(opponent_row, ["team_net_rating", "net_rating", "NET_RATING"], -projected_strength),
                "offensive_rating": _metric(
                    row, ["offensive_rating", "OFF_RATING"],
                    112.0 + projected_strength,
                ),
                "defensive_rating": _metric(
                    row, ["defensive_rating", "DEF_RATING"],
                    112.0 - projected_strength * 0.35,
                ),
                "opponent_offensive_rating": _metric(
                    opponent_row, ["offensive_rating", "OFF_RATING"],
                    112.0 - projected_strength,
                ),
                "opponent_defensive_rating": _metric(
                    opponent_row, ["defensive_rating", "DEF_RATING"],
                    112.0 + projected_strength * 0.35,
                ),
                "home_court": home_court,
                "rest_days": rest_days,
                "opponent_rest_days": rest_days,
                "recent_form": _metric(row, ["recent_form", "recent_net_rating"], projected_strength),
                "opponent_recent_form": _metric(opponent_row, ["recent_form", "recent_net_rating"], -projected_strength),
                "playoff_form": _metric(row, ["playoff_form", "playoff_net_rating"], projected_strength),
                "opponent_playoff_form": _metric(opponent_row, ["playoff_form", "playoff_net_rating"], -projected_strength),
                "pace": _metric(row, ["pace", "PACE"], 96.5),
                "opponent_pace": _metric(opponent_row, ["pace", "PACE"], 96.5),
                "efg_pct": _metric(row, ["efg_pct", "EFG_PCT"], 54.0),
                "opponent_efg_pct": _metric(opponent_row, ["efg_pct", "EFG_PCT"], 54.0),
                "turnover_pct": _metric(row, ["turnover_pct", "TOV_PCT"], 13.5),
                "opponent_turnover_pct": _metric(opponent_row, ["turnover_pct", "TOV_PCT"], 13.5),
                "offensive_rebound_pct": _metric(row, ["offensive_rebound_pct", "OREB_PCT"], 27.0),
                "opponent_offensive_rebound_pct": _metric(opponent_row, ["offensive_rebound_pct", "OREB_PCT"], 27.0),
                "free_throw_rate": _metric(row, ["free_throw_rate", "FTA_RATE"], 25.0),
                "opponent_free_throw_rate": _metric(opponent_row, ["free_throw_rate", "FTA_RATE"], 25.0),
                "injury_adjusted_team_strength": _metric(row, ["injury_adjusted_team_strength"], projected_strength),
                "opponent_injury_adjusted_team_strength": _metric(opponent_row, ["injury_adjusted_team_strength"], -projected_strength),
            }
        )
    return rows


_GAME_MODEL_BUNDLE: dict[str, Any] | None = None
_META_MODEL_BUNDLE: dict[str, Any] | None = None


def _get_game_model() -> dict[str, Any] | None:
    global _GAME_MODEL_BUNDLE
    if _GAME_MODEL_BUNDLE is None:
        _GAME_MODEL_BUNDLE = _load_game_model()
    return _GAME_MODEL_BUNDLE


def _get_meta_model() -> dict[str, Any] | None:
    global _META_MODEL_BUNDLE
    if _META_MODEL_BUNDLE is None:
        _META_MODEL_BUNDLE = load_meta_model()
    return _META_MODEL_BUNDLE


def _ml_baseline_probability(
    finals_context: dict[str, Any],
    game: dict[str, Any],
    team_stats: Any,
    rest_a: float,
    rest_b: float,
) -> float | None:
    """Use trained ML model for baseline probability if available."""
    bundle = _get_game_model()
    if not bundle:
        return None

    stats_index = _baseline_team_stats(finals_context, team_stats)

    team_a = str(finals_context["team_a"])
    team_b = str(finals_context["team_b"])
    row_a = stats_index.get(team_a, {})
    row_b = stats_index.get(team_b, {})

    def _r(d: dict[str, Any], key: str, default: float = 0.0) -> float:
        v = d.get(key)
        return float(v) if v is not None else default

    is_neutral = bool(game.get("neutral_site"))
    home_team = str(game.get("home_team", ""))
    home_court = 0.0 if is_neutral else (1.0 if home_team == team_a else -1.0)

    net_a = _r(row_a, "NET_RATING") or _r(row_a, "net_rating")
    net_b = _r(row_b, "NET_RATING") or _r(row_b, "net_rating")

    features = {
        "regular_net_rating_diff": _r(row_a, "REGULAR_NET_RATING") - _r(row_b, "REGULAR_NET_RATING"),
        "blended_net_rating_diff": net_a - net_b,
        "regular_efg_pct_diff": _r(row_a, "REGULAR_EFG_PCT") - _r(row_b, "REGULAR_EFG_PCT"),
        "blended_efg_pct_diff": _r(row_a, "EFG_PCT") - _r(row_b, "EFG_PCT"),
        "regular_tov_pct_diff": _r(row_a, "REGULAR_TM_TOV_PCT") - _r(row_b, "REGULAR_TM_TOV_PCT"),
        "blended_tov_pct_diff": _r(row_a, "TM_TOV_PCT") - _r(row_b, "TM_TOV_PCT"),
        "regular_oreb_pct_diff": _r(row_a, "REGULAR_OREB_PCT") - _r(row_b, "REGULAR_OREB_PCT"),
        "blended_oreb_pct_diff": _r(row_a, "OREB_PCT") - _r(row_b, "OREB_PCT"),
        "regular_fta_rate_diff": _r(row_a, "REGULAR_FTA_RATE") - _r(row_b, "REGULAR_FTA_RATE"),
        "blended_fta_rate_diff": _r(row_a, "FTA_RATE") - _r(row_b, "FTA_RATE"),
        "regular_pace_diff": _r(row_a, "REGULAR_PACE", 96.5) - _r(row_b, "REGULAR_PACE", 96.5),
        "blended_pace_diff": _r(row_a, "PACE", 96.5) - _r(row_b, "PACE", 96.5),
        "recent_net_rating_diff": _r(row_a, "PLAYOFF_NET_RATING") - _r(row_b, "PLAYOFF_NET_RATING"),
        "travel_miles_diff": 0.0,
        "travel_data_available": 0.0,
        "home_court": home_court,
        "rest_diff": rest_a - rest_b,
    }

    # Need at least net rating to trust the ML model
    if net_a == 0.0 and net_b == 0.0:
        return None

    team_a_id = str(1610612752 if team_a == "NYK" else 1610612759)
    team_b_id = str(1610612759 if team_a == "NYK" else 1610612752)

    return _ml_win_prob(features, bundle, team_a_id, team_b_id)


def _baseline_probability(
    finals_context: dict[str, Any],
    game: dict[str, Any],
    team_stats: Any,
    player_summary: dict[str, dict[str, float]],
    team_model_bundle: dict[str, Any] | None,
    rest_a: float = 3.0,
    rest_b: float = 3.0,
) -> tuple[float, list[dict[str, Any]]]:
    # Try trained ML model first
    ml_prob = _ml_baseline_probability(finals_context, game, team_stats, rest_a, rest_b)
    rows = _schedule_context_rows(finals_context, game, team_stats, player_summary)

    if ml_prob is not None:
        team_a = str(finals_context["team_a"])
        team_b = str(finals_context["team_b"])
        predictions = [
            {"team": team_a, "baseline_win_probability": ml_prob,
             "model_probabilities": {"ml_ensemble": ml_prob}},
            {"team": team_b, "baseline_win_probability": round(1.0 - ml_prob, 4),
             "model_probabilities": {"ml_ensemble": round(1.0 - ml_prob, 4)}},
        ]
        return ml_prob, predictions

    if team_model_bundle:
        predictions = predict_team_win_probability(team_model_bundle, rows)
    else:
        predictions = predict_baseline_without_training(rows)

    team_a = str(finals_context["team_a"])
    probability_by_team = {
        str(prediction["team"]): _as_float(prediction["baseline_win_probability"], 0.5)
        for prediction in predictions
    }
    team_a_raw = probability_by_team.get(team_a, 0.5)
    team_b_raw = probability_by_team.get(str(finals_context["team_b"]), 0.5)
    denominator = max(team_a_raw + team_b_raw, 0.01)
    return round(team_a_raw / denominator, 4), predictions


def _net_rating_probability(
    finals_context: dict[str, Any],
    game: dict[str, Any],
    team_stats: Any,
    rest_a: float,
    rest_b: float,
) -> float:
    stats = _baseline_team_stats(finals_context, team_stats)
    team_a, team_b = str(finals_context["team_a"]), str(finals_context["team_b"])
    regular_a = _as_float(stats.get(team_a, {}).get("REGULAR_NET_RATING"), 0.0)
    regular_b = _as_float(stats.get(team_b, {}).get("REGULAR_NET_RATING"), 0.0)
    structural = _logistic((regular_a - regular_b) / GAME_MARGIN_LOGIT_SCALE)
    home_context = 0.0
    if not bool(game.get("neutral_site")):
        home_context = 1.0 if str(game.get("home_team")) == team_a else -1.0
    context_margin = home_context * 2.2 + _clip((rest_a - rest_b) * 0.35, -2.0, 2.0)
    return round(
        _clip(
            _logistic(_logit(structural) + context_margin / GAME_MARGIN_LOGIT_SCALE),
            0.05,
            0.95,
        ),
        4,
    )


def _player_projection_margin(
    player_summary: dict[str, dict[str, float]],
    team_a: str,
    team_b: str,
) -> float:
    a = player_summary.get(team_a, {})
    b = player_summary.get(team_b, {})
    points = a.get("points", 0.0) - b.get("points", 0.0)
    rebounds = a.get("rebounds", 0.0) - b.get("rebounds", 0.0)
    assists = a.get("assists", 0.0) - b.get("assists", 0.0)
    turnovers = a.get("turnovers", 0.0) - b.get("turnovers", 0.0)
    return round(_clip(points * 0.45 + rebounds * 0.08 + assists * 0.10 - turnovers * 0.22, -8.0, 8.0), 3)


def _comparison_key(team: str, opponent: str) -> str:
    return f"{team}_offense_vs_{opponent}_defense"


def _matchup_margin(matchup_edges: dict[str, Any], team_a: str, team_b: str) -> float:
    comparisons = matchup_edges.get("comparisons", {})
    a_comparison = comparisons.get(_comparison_key(team_a, team_b), {})
    b_comparison = comparisons.get(_comparison_key(team_b, team_a), {})
    a_score = _as_float(a_comparison.get("average_matchup_score"), 0.0) + _as_float(a_comparison.get("overall_edge"), 0.0) * 0.35
    b_score = _as_float(b_comparison.get("average_matchup_score"), 0.0) + _as_float(b_comparison.get("overall_edge"), 0.0) * 0.35
    return round(_clip((a_score - b_score) * 2.0, -6.0, 6.0), 3)


def _lineup_by_type(lineups: list[dict[str, Any]], lineup_type: str) -> dict[str, Any] | None:
    for lineup in lineups:
        if lineup.get("lineup_type") == lineup_type:
            return lineup
    return None


def _lineup_margin(lineup_features: dict[str, list[dict[str, Any]]], team_a: str, team_b: str) -> float:
    a_lineups = lineup_features.get(team_a, [])
    b_lineups = lineup_features.get(team_b, [])
    a_closing = _lineup_by_type(a_lineups, "closing_lineup") or {}
    b_closing = _lineup_by_type(b_lineups, "closing_lineup") or {}
    a_best = max(a_lineups, key=lambda row: row["adjusted_lineup_net_rating"], default={})
    b_best = max(b_lineups, key=lambda row: row["adjusted_lineup_net_rating"], default={})
    closing_edge = (
        _as_float(a_closing.get("adjusted_lineup_net_rating"), 0.0)
        - _as_float(b_closing.get("adjusted_lineup_net_rating"), 0.0)
    )
    best_edge = (
        _as_float(a_best.get("adjusted_lineup_net_rating"), 0.0)
        - _as_float(b_best.get("adjusted_lineup_net_rating"), 0.0)
    )
    raw = closing_edge * 0.55 + best_edge * 0.20
    # When lineup ratings are derived from team net ratings (not specific lineup
    # data) they overlap with the ML baseline. Cap tighter to avoid double-counting.
    cap = 3.0 if abs(raw) > 4.0 else 6.0
    return round(_clip(raw, -cap, cap), 3)


def _clutch_margin(clutch_prediction: dict[str, Any], team_a: str) -> float:
    return round(_clip(_as_float(clutch_prediction.get("close_game_edges_per_100", {}).get(team_a), 0.0), -6.0, 6.0), 3)


def _injury_margin(finals_context: dict[str, Any], team_a: str, team_b: str) -> float:
    def team_adjustment(team: str) -> float:
        total = 0.0
        for injury in finals_context.get("injuries", {}).get(team, []):
            status = str(injury.get("status") or "").lower()
            minute_adjustment = _as_float(injury.get("expected_minutes_adjustment"), 0.0)
            if status == "out" and minute_adjustment == 0.0:
                minute_adjustment = -18.0
            elif status == "doubtful" and minute_adjustment == 0.0:
                minute_adjustment = -12.0
            total += minute_adjustment
        return total

    return round(_clip((team_adjustment(team_a) - team_adjustment(team_b)) * 0.24, -5.0, 5.0), 3)


def _foul_trouble_margin(foul_trouble_simulation: dict[str, Any], team_a: str, team_b: str) -> float:
    team_drags = {team_a: 0.0, team_b: 0.0}
    for scenario in foul_trouble_simulation.get("scenarios", []):
        team = str(scenario.get("team"))
        if team not in team_drags:
            continue
        normal_probability = _as_float(scenario.get("normal_win_probability"), 0.5)
        expected_probability = _as_float(scenario.get("expected_win_probability"), normal_probability)
        team_drags[team] += expected_probability - normal_probability
    return round(_clip((team_drags[team_a] - team_drags[team_b]) * 42.0, -4.0, 4.0), 3)


def _combine_probability(
    baseline_probability: float,
    margins: dict[str, float],
    weights: dict[str, float],
) -> float:
    residual_margin = (
        weights["player_projection"] * margins["player_projection"]
        + weights["matchup_edge"] * margins["matchup_edge"]
        + weights["lineup_edge"] * margins["lineup_edge"]
        + weights["clutch_edge"] * margins["clutch_edge"] * CLUTCH_GAME_RATE
        + weights["injury_edge"] * margins["injury_edge"]
        + weights["foul_trouble_risk"] * margins["foul_trouble_risk"]
    )
    residual_margin = _clip(
        residual_margin,
        -RESIDUAL_MARGIN_CAP,
        RESIDUAL_MARGIN_CAP,
    )
    logit_value = _logit(baseline_probability) + _margin_to_logit(residual_margin)
    return round(_clip(_logistic(logit_value), 0.05, 0.95), 4)


def _production_probability(
    baseline_probability: float,
    net_rating_probability: float,
    margins: dict[str, float],
    model_weights_path: str | Path,
) -> tuple[float, str]:
    mode = _combination_mode(model_weights_path)
    meta_model = _get_meta_model()
    if mode == "learned_meta" and meta_model:
        probability = predict_meta_probability(
            baseline_probability,
            {
                "player_edge": margins["player_projection"],
                "matchup_edge": margins["matchup_edge"],
                "lineup_edge": margins["lineup_edge"],
                "clutch_edge": margins["clutch_edge"] * CLUTCH_GAME_RATE,
                "injury_edge": margins["injury_edge"],
                "coaching_edge": 0.0,
            },
            availability={
                "player": True,
                "matchup": True,
                "lineup": True,
                "clutch": True,
                "injury": True,
                "coaching": False,
            },
            model_bundle=meta_model,
            net_rating_probability=net_rating_probability,
        )
        return probability, "learned_meta"
    weights = load_game_prediction_weights(model_weights_path)
    return _combine_probability(baseline_probability, margins, weights), "calibrated_ensemble"


_LEAGUE_AVG_ORTG = 113.0  # 2025-26 playoffs approximate
# NBA game-to-game margin std deviation (~11.5 pts).
# P(team_a wins) = sigmoid(expected_margin / NBA_MARGIN_SD)
_NBA_MARGIN_SD = GAME_MARGIN_LOGIT_SCALE


def _pythagorean_ortg(
    team_ortg: float,
    opp_drtg: float,
    league_avg: float = _LEAGUE_AVG_ORTG,
) -> float:
    """Expected points-per-100 for team against this specific opponent."""
    if league_avg <= 0:
        return team_ortg
    return team_ortg * (opp_drtg / league_avg)


def _score_from_prob_and_base(
    team_a_prob: float,
    base_ortg_a: float,
    base_ortg_b: float,
) -> tuple[float, float]:
    """Derive each team's ORTG so the score margin is consistent with win prob.

    Uses: P(A wins) = sigmoid(margin / NBA_MARGIN_SD)
    Solves for: margin = logit(P) × NBA_MARGIN_SD
    Applied around the midpoint of the two Pythagorean ORTGs.
    """
    p = _clip(team_a_prob, 0.02, 0.98)
    log_odds = log(p / (1.0 - p))
    expected_margin = log_odds * _NBA_MARGIN_SD

    midpoint = (base_ortg_a + base_ortg_b) / 2.0
    return round(midpoint + expected_margin / 2.0, 1), round(midpoint - expected_margin / 2.0, 1)


def _base_offensive_rating(
    team: str,
    team_stats: Any,
    player_summary: dict[str, dict[str, float]],
) -> float:
    row = _team_index(team_stats).get(team, {})
    player_points = player_summary.get(team, {}).get("points", 0.0)
    ortg_adj = _clip(player_points / 12.0, 0.0, 4.0)
    return _metric(row, ["offensive_rating", "OFF_RATING"], 108.0 + ortg_adj)


def _projected_offensive_ratings(
    team_a: str,
    team_b: str,
    team_stats: Any,
    player_summary: dict[str, dict[str, float]],
    margins: dict[str, float],
) -> tuple[float, float]:
    base_a = _base_offensive_rating(team_a, team_stats, player_summary)
    base_b = _base_offensive_rating(team_b, team_stats, player_summary)
    net_adjustment = (
        margins["player_projection"] * 0.35
        + margins["matchup_edge"] * 0.55
        + margins["lineup_edge"] * 0.25
        + margins["clutch_edge"] * 0.15
        + margins["injury_edge"] * 0.25
        + margins["foul_trouble_risk"] * 0.18
    )
    return round(base_a + net_adjustment / 2.0, 1), round(base_b - net_adjustment / 2.0, 1)


def _top_matchup_text(matchup_edges: dict[str, Any], limit: int = 4) -> list[str]:
    lines = []
    for edge in matchup_edges.get("top_series_edges", [])[:limit]:
        explanation = str(edge.get("explanation") or "")
        if explanation:
            first_sentence = explanation.split(". ", 1)[0]
            if not first_sentence.endswith("."):
                first_sentence += "."
            lines.append(first_sentence)
    return lines


def _lineup_edge_text(lineup_margin: float, team_a: str, team_b: str) -> str | None:
    if abs(lineup_margin) < 0.35:
        return None
    team = team_a if lineup_margin > 0 else team_b
    qualifier = "slightly better" if abs(lineup_margin) < 1.5 else "better"
    return f"{team} closing lineup projects {qualifier}."


def _clutch_edge_text(clutch_prediction: dict[str, Any]) -> str | None:
    favorite = clutch_prediction.get("favorite")
    edge = _as_float(clutch_prediction.get("favorite_edge_per_100"), 0.0)
    if not favorite or favorite == "Even" or edge < 0.25:
        return None
    return f"{favorite} has a close-game edge of {edge:+.1f} per 100 possessions."


def _top_edges(
    matchup_edges: dict[str, Any],
    lineup_margin: float,
    clutch_prediction: dict[str, Any],
    team_a: str,
    team_b: str,
) -> list[str]:
    lines = _top_matchup_text(matchup_edges)
    for line in (_lineup_edge_text(lineup_margin, team_a, team_b), _clutch_edge_text(clutch_prediction)):
        if line:
            lines.append(line)
    return lines[:6]


def _x_factors(
    finals_context: dict[str, Any],
    foul_trouble_simulation: dict[str, Any],
    lineup_features: dict[str, list[dict[str, Any]]],
    matchup_edges: dict[str, Any],
) -> list[str]:
    factors = []
    for scenario in foul_trouble_simulation.get("scenarios", [])[:3]:
        swing = _as_float(scenario.get("win_probability_swing"), 0.0)
        if swing >= 0.025:
            swing_pct = round(swing * 100, 1)
            factors.append(f"{scenario['team']} {scenario['player']} foul trouble ({swing_pct}% win-prob swing per game)")

    for team, lineups in lineup_features.items():
        non_star = _lineup_by_type(lineups, "non_star_minutes")
        if non_star:
            factors.append(f"{team} non-star minutes")

    for edge in matchup_edges.get("top_series_edges", []):
        key = str(edge.get("key") or "")
        offensive_team = edge.get("offensive_team")
        if "corner_3" in key and offensive_team:
            factors.append(f"{offensive_team} corner three shooting")
        if len(factors) >= 5:
            break

    for team, injuries in finals_context.get("uncertain_minutes", {}).items():
        for injury in injuries:
            factors.append(f"{team} {injury.get('player')} minutes uncertainty")
            if len(factors) >= 6:
                return _unique(factors)
    return _unique(factors)[:6]


def _unique(items: list[str]) -> list[str]:
    seen = set()
    unique_items = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def _build_engine_inputs(
    finals_context: dict[str, Any],
    player_projections: Any = None,
    playstyle_profiles: dict[str, dict[str, Any]] | None = None,
    matchup_edges: dict[str, Any] | None = None,
    lineup_features: dict[str, list[dict[str, Any]]] | None = None,
    clutch_prediction: dict[str, Any] | None = None,
    foul_trouble_simulation: dict[str, Any] | None = None,
    team_stats: Any = None,
    lineup_stats: Any = None,
    team_ratings: Any = None,
) -> dict[str, Any]:
    if player_projections is None:
        player_projections = project_finals_players(finals_context, matchup_adjustments=finals_context.get("active_players"))
    player_summary = summarize_player_projections(player_projections)

    if playstyle_profiles is None:
        playstyle_profiles = build_playstyle_profiles(finals_context, team_stats=team_stats)
    if matchup_edges is None:
        matchup_edges = build_matchup_edges(finals_context, playstyle_profiles=playstyle_profiles)
    if lineup_features is None:
        lineup_features = build_lineup_features(
            finals_context,
            player_projections=player_projections,
            lineup_stats=lineup_stats,
            team_ratings=team_ratings,
        )
    if clutch_prediction is None:
        clutch_prediction = predict_close_game_edge(
            finals_context,
            player_projections=player_projections,
            lineup_features=lineup_features,
            lineup_stats=lineup_stats,
            team_ratings=team_ratings,
        )
    if foul_trouble_simulation is None:
        foul_trouble_simulation = simulate_foul_trouble_scenarios(
            finals_context,
            player_projections=player_projections,
            playstyle_profiles=playstyle_profiles,
            base_win_probability={
                str(finals_context["team_a"]): 0.50,
                str(finals_context["team_b"]): 0.50,
            },
        )

    return {
        "player_projections": player_projections,
        "player_summary": player_summary,
        "playstyle_profiles": playstyle_profiles,
        "matchup_edges": matchup_edges,
        "lineup_features": lineup_features,
        "clutch_prediction": clutch_prediction,
        "foul_trouble_simulation": foul_trouble_simulation,
    }


def predict_game(
    finals_context: dict[str, Any],
    game: dict[str, Any],
    team_model_bundle: dict[str, Any] | None = None,
    team_stats: Any = None,
    player_projections: Any = None,
    playstyle_profiles: dict[str, dict[str, Any]] | None = None,
    matchup_edges: dict[str, Any] | None = None,
    lineup_features: dict[str, list[dict[str, Any]]] | None = None,
    clutch_prediction: dict[str, Any] | None = None,
    foul_trouble_simulation: dict[str, Any] | None = None,
    lineup_stats: Any = None,
    team_ratings: Any = None,
    model_weights_path: str | Path = DEFAULT_MODEL_WEIGHTS_PATH,
) -> dict[str, Any]:
    """Predict one Finals game."""
    team_a = str(finals_context["team_a"])
    team_b = str(finals_context["team_b"])
    weights = load_game_prediction_weights(model_weights_path)
    inputs = _build_engine_inputs(
        finals_context,
        player_projections,
        playstyle_profiles,
        matchup_edges,
        lineup_features,
        clutch_prediction,
        foul_trouble_simulation,
        team_stats,
        lineup_stats,
        team_ratings,
    )

    game_number = int(game["game_number"])
    schedule = finals_context.get("schedule", [])
    rest_a = _team_rest_days(team_a, game_number, schedule, finals_context)
    rest_b = _team_rest_days(team_b, game_number, schedule, finals_context)
    baseline_probability, baseline_predictions = _baseline_probability(
        finals_context,
        game,
        team_stats,
        inputs["player_summary"],
        team_model_bundle,
        rest_a=rest_a,
        rest_b=rest_b,
    )
    net_rating_probability = _net_rating_probability(
        finals_context,
        game,
        team_stats,
        rest_a,
        rest_b,
    )
    margins = {
        "player_projection": _player_projection_margin(inputs["player_summary"], team_a, team_b),
        "matchup_edge": _matchup_margin(inputs["matchup_edges"], team_a, team_b),
        "lineup_edge": _lineup_margin(inputs["lineup_features"], team_a, team_b),
        "clutch_edge": _clutch_margin(inputs["clutch_prediction"], team_a),
        "injury_edge": _injury_margin(finals_context, team_a, team_b),
        "foul_trouble_risk": _foul_trouble_margin(inputs["foul_trouble_simulation"], team_a, team_b),
    }
    team_a_probability, combination_method = _production_probability(
        baseline_probability,
        net_rating_probability,
        margins,
        model_weights_path,
    )
    team_b_probability = round(1.0 - team_a_probability, 4)
    projected_pace = _pace_from_profiles(inputs["playstyle_profiles"], team_a, team_b)

    # --- Score projection ---
    # Use Pythagorean ORTG (team_ORTG × opponent_DRTG / league_avg) when real
    # team stats are available, so the opponent's defense is factored in.
    # Then derive the margin from the win probability so the projected score
    # is always consistent with who the model thinks will win.
    live = _prediction_team_stats(finals_context, team_stats)
    ts_a = live.get(team_a, {})
    ts_b = live.get(team_b, {})
    ortg_a = _as_float(ts_a.get("OFF_RATING") or ts_a.get("offensive_rating"), 0)
    drtg_a = _as_float(ts_a.get("DEF_RATING") or ts_a.get("defensive_rating"), 0)
    ortg_b = _as_float(ts_b.get("OFF_RATING") or ts_b.get("offensive_rating"), 0)
    drtg_b = _as_float(ts_b.get("DEF_RATING") or ts_b.get("defensive_rating"), 0)

    if ortg_a > 0 and ortg_b > 0 and drtg_a > 0 and drtg_b > 0:
        # Pythagorean: each team's expected scoring adjusted for the specific opponent
        pyth_a = _pythagorean_ortg(ortg_a, drtg_b)
        pyth_b = _pythagorean_ortg(ortg_b, drtg_a)
    else:
        # Fall back to component-based formula
        pyth_a, pyth_b = _projected_offensive_ratings(
            team_a, team_b, team_stats, inputs["player_summary"], margins,
        )

    # Home court and rest shift the Pythagorean bases before we derive the margin
    is_neutral = bool(game.get("neutral_site"))
    home_team = str(game.get("home_team", ""))
    home_boost = 0.0 if is_neutral else 2.0
    if home_team == team_a:
        pyth_a = round(pyth_a + home_boost, 1)
        pyth_b = round(pyth_b - home_boost, 1)
    elif home_team == team_b:
        pyth_b = round(pyth_b + home_boost, 1)
        pyth_a = round(pyth_a - home_boost, 1)

    pyth_a = round(pyth_a + _rest_ortg_boost(rest_a), 1)
    pyth_b = round(pyth_b + _rest_ortg_boost(rest_b), 1)

    # Derive the actual score split from the win probability so projected
    # winner always has a higher projected score.
    off_a, off_b = _score_from_prob_and_base(team_a_probability, pyth_a, pyth_b)

    # Pace decay: later games in the series play slightly slower.
    adjusted_pace = max(projected_pace + _series_pace_decay(game_number), 90.0)

    score_a = round(off_a * adjusted_pace / 100.0)
    score_b = round(off_b * adjusted_pace / 100.0)

    # Projected winner must have a higher score after integer rounding.
    if score_a == score_b:
        if team_a_probability >= 0.5:
            score_a += 1
        else:
            score_b += 1

    return {
        "model_version": MODEL_VERSION,
        "prediction_generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "game_number": int(game["game_number"]),
        "date": game.get("date"),
        "home_team": game.get("home_team"),
        "away_team": game.get("away_team"),
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win_probability": team_a_probability,
        "team_b_win_probability": team_b_probability,
        "expected_score_team_a": int(score_a),
        "expected_score_team_b": int(score_b),
        "projected_pace": projected_pace,
        "projected_team_a_off_rating": off_a,
        "projected_team_b_off_rating": off_b,
        "top_edges": _top_edges(inputs["matchup_edges"], margins["lineup_edge"], inputs["clutch_prediction"], team_a, team_b),
        "x_factors": _x_factors(finals_context, inputs["foul_trouble_simulation"], inputs["lineup_features"], inputs["matchup_edges"]),
        "component_margins": margins,
        "baseline_probability_team_a": baseline_probability,
        "net_rating_probability_team_a": net_rating_probability,
        "baseline_model_predictions": baseline_predictions,
        "weights": weights,
        "combination_method": combination_method,
    }


def predict_finals_games(
    finals_context: dict[str, Any] | None = None,
    team_model_bundle: dict[str, Any] | None = None,
    team_stats: Any = None,
    player_projections: Any = None,
    playstyle_profiles: dict[str, dict[str, Any]] | None = None,
    matchup_edges: dict[str, Any] | None = None,
    lineup_features: dict[str, list[dict[str, Any]]] | None = None,
    clutch_prediction: dict[str, Any] | None = None,
    foul_trouble_simulation: dict[str, Any] | None = None,
    lineup_stats: Any = None,
    team_ratings: Any = None,
    model_weights_path: str | Path = DEFAULT_MODEL_WEIGHTS_PATH,
) -> list[dict[str, Any]]:
    """Predict every scheduled Finals game."""
    finals_context = finals_context or build_finals_context()
    inputs = _build_engine_inputs(
        finals_context,
        player_projections,
        playstyle_profiles,
        matchup_edges,
        lineup_features,
        clutch_prediction,
        foul_trouble_simulation,
        team_stats,
        lineup_stats,
        team_ratings,
    )

    return [
        predict_game(
            finals_context,
            game,
            team_model_bundle=team_model_bundle,
            team_stats=team_stats,
            player_projections=inputs["player_projections"],
            playstyle_profiles=inputs["playstyle_profiles"],
            matchup_edges=inputs["matchup_edges"],
            lineup_features=inputs["lineup_features"],
            clutch_prediction=inputs["clutch_prediction"],
            foul_trouble_simulation=inputs["foul_trouble_simulation"],
            lineup_stats=lineup_stats,
            team_ratings=team_ratings,
            model_weights_path=model_weights_path,
        )
        for game in finals_context.get("schedule", [])
    ]


if __name__ == "__main__":
    predictions = predict_finals_games()
    for prediction in predictions:
        print(
            f"Game {prediction['game_number']}: "
            f"{prediction['team_a']} {prediction['team_a_win_probability']:.1%}, "
            f"{prediction['team_b']} {prediction['team_b_win_probability']:.1%}, "
            f"score {prediction['expected_score_team_a']}-{prediction['expected_score_team_b']}"
        )
