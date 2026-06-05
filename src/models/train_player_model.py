"""Player box-score projection engine.

The first version is intentionally transparent:

    projected_stat = per_minute_rate * projected_minutes * matchup_adjustment

Historical model training hooks are included for Ridge, Random Forest, and
Gradient Boosting regressors once richer training rows are available.
"""

from __future__ import annotations

from math import isnan
from random import Random
from typing import Any

from src.features.player_features import (
    flatten_player_projections,
    project_rotation_minutes,
)


COUNTING_STATS = [
    "points",
    "rebounds",
    "assists",
    "turnovers",
    "steals",
    "blocks",
    "threes_made",
    "threes_attempted",
    "free_throw_attempts",
]

EFFICIENCY_STATS = ["usage_rate", "true_shooting_proxy"]
RISK_STATS = ["foul_risk"]
PLAYER_PROJECTION_TARGETS = ["minutes", *COUNTING_STATS, *EFFICIENCY_STATS, *RISK_STATS]

NEGATIVE_CONTEXT_STATS = {"turnovers", "foul_risk"}

SOURCE_WEIGHTS = {
    "season": 0.25,
    "playoff": 0.55,
    "recent_5": 0.10,
    "recent_10": 0.07,
    "recent_15": 0.03,
}

PLAYER_STAT_PRIOR_MINUTES = {
    "points": 800.0,
    "rebounds": 400.0,
    "assists": 500.0,
    "turnovers": 350.0,
    "steals": 900.0,
    "blocks": 900.0,
    "threes_made": 1000.0,
    "threes_attempted": 500.0,
    "free_throw_attempts": 650.0,
    "personal_fouls": 300.0,
    "usage_rate": 500.0,
    "true_shooting_proxy": 1200.0,
}

ROLE_BASELINE_RATES = {
    "starter": {
        "points": 0.62,
        "rebounds": 0.18,
        "assists": 0.13,
        "turnovers": 0.075,
        "steals": 0.035,
        "blocks": 0.025,
        "threes_made": 0.075,
        "threes_attempted": 0.20,
        "free_throw_attempts": 0.15,
        "personal_fouls": 0.075,
        "usage_rate": 25.0,
        "true_shooting_proxy": 0.580,
    },
    "bench": {
        "points": 0.42,
        "rebounds": 0.14,
        "assists": 0.09,
        "turnovers": 0.055,
        "steals": 0.030,
        "blocks": 0.020,
        "threes_made": 0.050,
        "threes_attempted": 0.145,
        "free_throw_attempts": 0.090,
        "personal_fouls": 0.085,
        "usage_rate": 18.0,
        "true_shooting_proxy": 0.555,
    },
    "default": {
        "points": 0.50,
        "rebounds": 0.16,
        "assists": 0.10,
        "turnovers": 0.060,
        "steals": 0.030,
        "blocks": 0.020,
        "threes_made": 0.060,
        "threes_attempted": 0.165,
        "free_throw_attempts": 0.110,
        "personal_fouls": 0.080,
        "usage_rate": 20.0,
        "true_shooting_proxy": 0.560,
    },
}

STAT_ALIASES = {
    "minutes": ["minutes", "min", "MIN", "MP", "avg_minutes"],
    "points": ["points", "PTS", "pts", "points_per_game"],
    "rebounds": ["rebounds", "REB", "TRB", "total_rebounds", "rebounds_per_game"],
    "assists": ["assists", "AST", "ast", "assists_per_game"],
    "turnovers": ["turnovers", "TOV", "TO", "turnovers_per_game"],
    "steals": ["steals", "STL", "stl", "steals_per_game"],
    "blocks": ["blocks", "BLK", "blk", "blocks_per_game"],
    "threes_made": ["threes_made", "FG3M", "3PM", "three_pointers_made"],
    "threes_attempted": ["threes_attempted", "FG3A", "3PA", "three_pointers_attempted"],
    "free_throw_attempts": ["free_throw_attempts", "FTA", "fta"],
    "usage_rate": ["usage_rate", "USG_PCT", "USG%", "usg_pct"],
    "true_shooting_proxy": ["true_shooting_proxy", "TS_PCT", "TS%", "ts_pct"],
    "field_goal_attempts": ["field_goal_attempts", "FGA", "fga"],
    "personal_fouls": ["personal_fouls", "PF", "fouls", "pf"],
}

PLAYER_NAME_KEYS = ["player", "PLAYER_NAME", "name", "Name"]
TEAM_KEYS = ["team", "TEAM_ABBREVIATION", "TEAM", "team_abbr"]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        value = value.strip().replace("%", "")
        if ":" in value:
            minutes, seconds = value.split(":", 1)
            try:
                return float(minutes) + float(seconds) / 60.0
            except ValueError:
                return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if isnan(number):
        return default
    return number


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _first_value(row: dict[str, Any] | None, keys: list[str]) -> Any:
    if not row:
        return None
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _first_number(row: dict[str, Any] | None, stat_name: str, default: float = 0.0) -> float:
    return _as_float(_first_value(row, STAT_ALIASES[stat_name]), default)


def _iter_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return [dict(row) for row in data.to_dict(orient="records")]
    if isinstance(data, dict):
        if "player" in data or "PLAYER_NAME" in data:
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
                record.setdefault("player", key)
                records.append(record)
        return records
    return [dict(row) for row in data]


def _player_name(row: dict[str, Any]) -> str:
    return str(_first_value(row, PLAYER_NAME_KEYS) or "").strip()


def _team(row: dict[str, Any]) -> str:
    return str(_first_value(row, TEAM_KEYS) or "").strip()


def _build_player_index(data: Any) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_records(data):
        player = _player_name(row)
        team = _team(row)
        if not player:
            continue
        if team:
            index[(team, player)] = row
        index[("", player)] = row
    return index


def _lookup_player(
    index: dict[tuple[str, str], dict[str, Any]],
    team: str,
    player: str,
) -> dict[str, Any] | None:
    return index.get((team, player)) or index.get(("", player))


def _role_key(minutes_projection: dict[str, Any]) -> str:
    role = str(minutes_projection.get("role") or "").strip().lower()
    if "bench" in role:
        return "bench"
    if minutes_projection.get("is_starter"):
        return "starter"
    return "default"


def _role_baseline(minutes_projection: dict[str, Any]) -> dict[str, float]:
    return ROLE_BASELINE_RATES[_role_key(minutes_projection)]


def _normalize_percentage(value: float, percentage_points: bool) -> float:
    if percentage_points:
        return value * 100.0 if 0.0 <= value <= 1.5 else value
    return value / 100.0 if value > 1.5 else value


def _minutes_from_row(row: dict[str, Any] | None) -> float:
    return _first_number(row, "minutes", 0.0)


def _stat_rate(row: dict[str, Any] | None, stat_name: str) -> float | None:
    if not row:
        return None
    minutes = _minutes_from_row(row)
    if minutes <= 0:
        return None
    value = _first_number(row, stat_name, 0.0)
    return value / minutes


def _usage_rate(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    usage = _first_number(row, "usage_rate", 0.0)
    if usage > 0:
        return _normalize_percentage(usage, percentage_points=True)

    minutes = _minutes_from_row(row)
    if minutes <= 0:
        return None

    offensive_load = (
        _first_number(row, "points")
        + 1.5 * _first_number(row, "assists")
        + 2.0 * _first_number(row, "turnovers")
    )
    return _clip((offensive_load / minutes) * 18.0, 8.0, 38.0)


def _true_shooting_proxy(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    ts_value = _first_number(row, "true_shooting_proxy", 0.0)
    if ts_value > 0:
        return _normalize_percentage(ts_value, percentage_points=False)

    points = _first_number(row, "points", 0.0)
    fga = _first_number(row, "field_goal_attempts", 0.0)
    fta = _first_number(row, "free_throw_attempts", 0.0)
    denominator = 2.0 * (fga + 0.44 * fta)
    if denominator <= 0:
        return None
    return _clip(points / denominator, 0.35, 0.75)


def _weighted_blend(
    values_by_source: dict[str, float | None],
    fallback: float,
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for source, value in values_by_source.items():
        if value is None:
            continue
        weight = SOURCE_WEIGHTS[source]
        weighted_sum += value * weight
        total_weight += weight
    if total_weight == 0:
        return fallback
    return weighted_sum / total_weight


def _sample_minutes(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    minutes = _minutes_from_row(row)
    games = _as_float(row.get("GP") or row.get("games") or row.get("G"), 0.0)
    if games <= 0:
        return 0.0
    return minutes * games


def _sample_size_blend(
    values_by_source: dict[str, float | None],
    source_rows: dict[str, dict[str, Any] | None],
    fallback: float,
    stat_name: str,
) -> float:
    """Use regular season as the prior and playoff minutes as new evidence."""
    regular = values_by_source.get("season")
    playoff = values_by_source.get("playoff")
    playoff_minutes = _sample_minutes(source_rows.get("playoff"))
    if regular is None and playoff is None:
        return fallback
    if regular is None:
        regular = fallback
    if playoff is None:
        playoff = regular
    if playoff_minutes <= 0:
        return _weighted_blend(values_by_source, fallback)

    prior_minutes = PLAYER_STAT_PRIOR_MINUTES[stat_name]
    playoff_weight = playoff_minutes / (playoff_minutes + prior_minutes)
    blended = playoff_weight * playoff + (1.0 - playoff_weight) * regular

    # Recent windows are a bounded role/form adjustment, not independent samples.
    recent_values = [
        values_by_source[source]
        for source in ("recent_5", "recent_10", "recent_15")
        if values_by_source.get(source) is not None
    ]
    if recent_values:
        recent = sum(recent_values) / len(recent_values)
        blended = 0.88 * blended + 0.12 * recent
    return blended


def _source_indexes(
    season_averages: Any = None,
    playoff_averages: Any = None,
    recent_averages: dict[str | int, Any] | None = None,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    recent_averages = recent_averages or {}
    return {
        "season": _build_player_index(season_averages),
        "playoff": _build_player_index(playoff_averages),
        "recent_5": _build_player_index(
            recent_averages.get(5) or recent_averages.get("5") or recent_averages.get("recent_5")
        ),
        "recent_10": _build_player_index(
            recent_averages.get(10) or recent_averages.get("10") or recent_averages.get("recent_10")
        ),
        "recent_15": _build_player_index(
            recent_averages.get(15) or recent_averages.get("15") or recent_averages.get("recent_15")
        ),
    }


def _source_rows(
    indexes: dict[str, dict[tuple[str, str], dict[str, Any]]],
    team: str,
    player: str,
) -> dict[str, dict[str, Any] | None]:
    return {
        source: _lookup_player(index, team, player)
        for source, index in indexes.items()
    }


def _matchup_adjustment_lookup(matchup_adjustments: Any) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for row in _iter_records(matchup_adjustments):
        team = str(
            row.get("team")
            or row.get("offensive_team")
            or row.get("TEAM_ABBREVIATION")
            or ""
        ).strip()
        player = str(
            row.get("player")
            or row.get("offensive_player")
            or row.get("PLAYER_NAME")
            or ""
        ).strip()
        raw_value = _raw_matchup_value(row)
        if player:
            lookup[(team, player)] = _coerce_matchup_multiplier(raw_value)
            lookup[("", player)] = lookup[(team, player)]
    return lookup


def _raw_matchup_value(row: dict[str, Any]) -> Any:
    for key in ("matchup_adjustment", "matchup_multiplier", "expected_impact"):
        if key in row and row[key] not in (None, ""):
            return row[key]

    matchup_roles = row.get("matchup_roles")
    if isinstance(matchup_roles, list):
        impacts = [
            _as_float(role.get("expected_impact"), 0.0)
            for role in matchup_roles
            if isinstance(role, dict) and role.get("side") == "offense"
        ]
        if not impacts:
            impacts = [
                _as_float(role.get("expected_impact"), 0.0)
                for role in matchup_roles
                if isinstance(role, dict)
            ]
        if impacts:
            return sum(impacts) / len(impacts)
    return 1.0


def _coerce_matchup_multiplier(raw_value: Any) -> float:
    # expected_impact uses a ±2 point scale where 0 = neutral.
    # Convert to a stat multiplier: each 1.0 impact point = 8% swing.
    value = _as_float(raw_value, 0.0)
    multiplier = 1.0 + _clip(value, -2.0, 2.0) * 0.08
    return _clip(multiplier, 0.80, 1.20)


def _opponent_defense_lookup(opponent_defense: Any) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in _iter_records(opponent_defense):
        team = _team(row) or str(row.get("opponent") or "").strip()
        if team:
            lookup[team] = row
    return lookup


def _opponent_for_team(finals_context: dict[str, Any], team: str) -> str:
    team_a = str(finals_context.get("team_a") or "")
    team_b = str(finals_context.get("team_b") or "")
    return team_b if team == team_a else team_a


def _opponent_stat_multiplier(
    opponent_row: dict[str, Any] | None,
    stat_name: str,
) -> float:
    if not opponent_row:
        return 1.0

    target_key = f"{stat_name}_allowed_multiplier"
    if target_key in opponent_row:
        return _clip(_as_float(opponent_row[target_key], 1.0), 0.80, 1.20)

    defensive_rating = _as_float(
        opponent_row.get("defensive_rating") or opponent_row.get("DEF_RATING"),
        0.0,
    )
    if defensive_rating <= 0:
        return 1.0

    multiplier = 1.0 + (defensive_rating - 115.0) * 0.004
    if stat_name in NEGATIVE_CONTEXT_STATS:
        multiplier = 2.0 - multiplier
    return _clip(multiplier, 0.90, 1.10)


def build_player_feature_row(
    minutes_projection: dict[str, Any],
    source_rows: dict[str, dict[str, Any] | None],
    opponent: str,
    opponent_defense_row: dict[str, Any] | None = None,
    matchup_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Create one player feature row for rate-based or ML projection."""
    baseline = _role_baseline(minutes_projection)
    feature_row: dict[str, Any] = {
        "player_key": minutes_projection.get("player_key"),
        "team": minutes_projection.get("team"),
        "player": minutes_projection.get("player"),
        "opponent": opponent,
        "role": minutes_projection.get("role"),
        "projected_minutes": minutes_projection.get("projected_minutes", 0.0),
        "minutes_floor": minutes_projection.get("minutes_floor", 0.0),
        "minutes_ceiling": minutes_projection.get("minutes_ceiling", 0.0),
        "is_starter": bool(minutes_projection.get("is_starter")),
        "is_closer": bool(minutes_projection.get("is_closer")),
        "injury_status": minutes_projection.get("injury_status", "Available"),
        "injury_adjustment": minutes_projection.get("injury_adjustment", 0.0),
        "rotation_confidence": minutes_projection.get("rotation_confidence", "medium"),
        "matchup_adjustment": matchup_multiplier,
        "opponent_defensive_rating": _as_float(
            (opponent_defense_row or {}).get("defensive_rating")
            or (opponent_defense_row or {}).get("DEF_RATING"),
            115.0,
        ),
        "opponent_pace": _as_float(
            (opponent_defense_row or {}).get("pace") or (opponent_defense_row or {}).get("PACE"),
            100.0,
        ),
    }

    for stat_name in COUNTING_STATS:
        values = {
            source: _stat_rate(row, stat_name)
            for source, row in source_rows.items()
        }
        fallback_key = "personal_fouls" if stat_name == "foul_risk" else stat_name
        feature_row[f"{stat_name}_per_minute"] = _sample_size_blend(
            values,
            source_rows,
            baseline.get(fallback_key, ROLE_BASELINE_RATES["default"][stat_name]),
            fallback_key,
        )

    foul_values = {
        source: _stat_rate(row, "personal_fouls")
        for source, row in source_rows.items()
    }
    feature_row["personal_fouls_per_minute"] = _sample_size_blend(
        foul_values,
        source_rows,
        baseline["personal_fouls"],
        "personal_fouls",
    )

    usage_values = {
        source: _usage_rate(row)
        for source, row in source_rows.items()
    }
    ts_values = {
        source: _true_shooting_proxy(row)
        for source, row in source_rows.items()
    }
    feature_row["usage_rate_base"] = _sample_size_blend(
        usage_values, source_rows, baseline["usage_rate"], "usage_rate",
    )
    feature_row["true_shooting_proxy_base"] = _sample_size_blend(
        ts_values, source_rows, baseline["true_shooting_proxy"], "true_shooting_proxy",
    )
    return feature_row


def build_player_feature_rows(
    finals_context: dict[str, Any],
    season_averages: Any = None,
    playoff_averages: Any = None,
    recent_averages: dict[str | int, Any] | None = None,
    opponent_defense: Any = None,
    matchup_adjustments: Any = None,
) -> list[dict[str, Any]]:
    """Build projection feature rows for every active Finals rotation player."""
    indexes = _source_indexes(season_averages, playoff_averages, recent_averages)
    matchup_lookup = _matchup_adjustment_lookup(matchup_adjustments)
    defense_lookup = _opponent_defense_lookup(opponent_defense)
    minute_rows = flatten_player_projections(project_rotation_minutes(finals_context))

    feature_rows = []
    for minutes_projection in minute_rows:
        team = str(minutes_projection.get("team"))
        player = str(minutes_projection.get("player"))
        opponent = _opponent_for_team(finals_context, team)
        matchup_multiplier = matchup_lookup.get((team, player), matchup_lookup.get(("", player), 1.0))
        feature_rows.append(
            build_player_feature_row(
                minutes_projection,
                _source_rows(indexes, team, player),
                opponent,
                defense_lookup.get(opponent),
                matchup_multiplier,
            )
        )
    return feature_rows


def _context_multiplier(feature_row: dict[str, Any], stat_name: str) -> float:
    matchup = _as_float(feature_row.get("matchup_adjustment"), 1.0)
    if stat_name in NEGATIVE_CONTEXT_STATS:
        matchup = 2.0 - matchup

    opponent_multiplier = _opponent_stat_multiplier(
        {
            "defensive_rating": feature_row.get("opponent_defensive_rating"),
            "pace": feature_row.get("opponent_pace"),
        },
        stat_name,
    )
    return _clip(matchup * opponent_multiplier, 0.75, 1.25)


def project_player_from_rates(feature_row: dict[str, Any]) -> dict[str, Any]:
    """Project one player using blended per-minute rates."""
    minutes = _as_float(feature_row.get("projected_minutes"), 0.0)
    projection = {
        "player_key": feature_row.get("player_key"),
        "team": feature_row.get("team"),
        "player": feature_row.get("player"),
        "opponent": feature_row.get("opponent"),
        "role": feature_row.get("role"),
        "minutes": round(minutes, 1),
        "minutes_floor": feature_row.get("minutes_floor"),
        "minutes_ceiling": feature_row.get("minutes_ceiling"),
        "injury_status": feature_row.get("injury_status"),
        "matchup_adjustment": round(_as_float(feature_row.get("matchup_adjustment"), 1.0), 3),
        "projection_method": "rate_based",
    }

    for stat_name in COUNTING_STATS:
        rate = _as_float(feature_row.get(f"{stat_name}_per_minute"), 0.0)
        multiplier = _context_multiplier(feature_row, stat_name)
        projection[stat_name] = round(rate * minutes * multiplier, 1)

    usage_multiplier = _clip(_context_multiplier(feature_row, "points"), 0.90, 1.10)
    ts_multiplier = _clip(_context_multiplier(feature_row, "points"), 0.94, 1.06)
    expected_fouls = (
        _as_float(feature_row.get("personal_fouls_per_minute"), 0.0)
        * minutes
        * _context_multiplier(feature_row, "foul_risk")
    )

    projection["usage_rate"] = round(
        _clip(_as_float(feature_row.get("usage_rate_base"), 20.0) * usage_multiplier, 5.0, 40.0),
        1,
    )
    projection["true_shooting_proxy"] = round(
        _clip(
            _as_float(feature_row.get("true_shooting_proxy_base"), 0.56) * ts_multiplier,
            0.35,
            0.75,
        ),
        3,
    )
    projection["foul_risk"] = round(_clip(expected_fouls / 6.0, 0.0, 1.0), 3)
    return projection


def project_players_from_rates(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project every player with the transparent rate-based formula."""
    return [project_player_from_rates(row) for row in feature_rows]


def _make_regressor(model_type: str, random_state: int = 42) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    if model_type == "ridge":
        return Ridge(alpha=1.0)
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(random_state=random_state)
    raise ValueError("model_type must be ridge, random_forest, or gradient_boosting.")


def _one_hot_encoder() -> Any:
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", _one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def _feature_columns(training_frame: Any, targets: list[str]) -> tuple[list[str], list[str], list[str]]:
    excluded = set(targets) | {"player_key"}
    candidate_columns = [
        column
        for column in training_frame.columns
        if column not in excluded
    ]
    numeric_columns = [
        column
        for column in candidate_columns
        if str(training_frame[column].dtype) != "object"
    ]
    categorical_columns = [
        column
        for column in candidate_columns
        if column not in numeric_columns
    ]
    return candidate_columns, numeric_columns, categorical_columns


def train_player_models(
    training_data: Any,
    model_type: str = "ridge",
    targets: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train one model type for every requested player projection target."""
    import pandas as pd
    from sklearn.pipeline import Pipeline

    targets = targets or PLAYER_PROJECTION_TARGETS
    training_frame = pd.DataFrame(_iter_records(training_data)).copy()
    if training_frame.empty:
        raise ValueError("training_data must contain at least one row.")

    available_targets = [target for target in targets if target in training_frame.columns]
    if not available_targets:
        raise ValueError("training_data does not contain any requested target columns.")

    feature_columns, numeric_columns, categorical_columns = _feature_columns(
        training_frame,
        available_targets,
    )
    if not feature_columns:
        raise ValueError("training_data must include feature columns in addition to targets.")

    target_models = {}
    x = training_frame[feature_columns]
    for target in available_targets:
        y = training_frame[target]
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _make_preprocessor(numeric_columns, categorical_columns)),
                ("regressor", _make_regressor(model_type, random_state=random_state)),
            ]
        )
        pipeline.fit(x, y)
        target_models[target] = pipeline

    return {
        "model_type": model_type,
        "targets": available_targets,
        "feature_columns": feature_columns,
        "target_models": target_models,
    }


def train_model_family(
    training_data: Any,
    targets: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Train Ridge, Random Forest, and Gradient Boosting player models."""
    return {
        model_type: train_player_models(training_data, model_type, targets, random_state)
        for model_type in ("ridge", "random_forest", "gradient_boosting")
    }


def predict_with_trained_models(
    feature_rows: list[dict[str, Any]],
    trained_models: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project players with trained sklearn target models."""
    import pandas as pd

    feature_frame = pd.DataFrame(feature_rows)
    feature_columns = trained_models["feature_columns"]
    base_predictions = project_players_from_rates(feature_rows)

    for target in trained_models["targets"]:
        model = trained_models["target_models"][target]
        values = model.predict(feature_frame[feature_columns])
        for projection, value in zip(base_predictions, values):
            if target == "true_shooting_proxy":
                projection[target] = round(_clip(float(value), 0.35, 0.75), 3)
            elif target == "foul_risk":
                projection[target] = round(_clip(float(value), 0.0, 1.0), 3)
            elif target == "usage_rate":
                projection[target] = round(_clip(float(value), 5.0, 40.0), 1)
            elif target == "minutes":
                projection[target] = round(_clip(float(value), 0.0, 48.0), 1)
            else:
                projection[target] = round(max(0.0, float(value)), 1)
            projection["projection_method"] = trained_models["model_type"]

    return base_predictions


def _build_player_efficiency_lookup(
    player_stats: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Build {player_name: {PIE, E_NET_RATING, USG_PCT}} from live stats."""
    lookup: dict[str, dict[str, float]] = {}
    for _team, players in player_stats.items():
        for p in players:
            name_x = str(p.get("PLAYER_NAME_x") or p.get("PLAYER_NAME") or "")
            name = name_x.strip()
            if not name:
                continue
            lookup[name] = {
                "PIE": float(p["PIE"]) if p.get("PIE") is not None else 0.0,
                "E_NET_RATING": float(p["E_NET_RATING"]) if p.get("E_NET_RATING") is not None else 0.0,
                "USG_PCT": float(p["USG_PCT"]) if p.get("USG_PCT") is not None else 0.0,
            }
    return lookup


def _apply_efficiency_boost(
    projection: dict[str, Any],
    efficiency_lookup: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Scale player projection by PIE relative to league average (0.10).

    PIE > 0.10 → positive player, scale up.
    PIE < 0.10 → below average, scale down slightly.
    Effect capped at ±12% so outliers don't dominate.
    """
    player = str(projection.get("player") or "")
    eff = efficiency_lookup.get(player)
    if not eff:
        return projection

    pie = eff.get("PIE", 0.10)
    league_avg_pie = 0.10
    # Each 0.01 above/below average = 1.2% stat multiplier
    boost = _clip((pie - league_avg_pie) * 1.2, -0.12, 0.12)
    multiplier = 1.0 + boost

    updated = dict(projection)
    for stat in ("points", "rebounds", "assists"):
        if stat in updated:
            updated[stat] = round(float(updated[stat]) * multiplier, 1)
    updated["efficiency_multiplier"] = round(multiplier, 3)
    updated["pie"] = round(pie, 3)
    return updated


def project_finals_players(
    finals_context: dict[str, Any],
    season_averages: Any = None,
    playoff_averages: Any = None,
    recent_averages: dict[str | int, Any] | None = None,
    opponent_defense: Any = None,
    matchup_adjustments: Any = None,
    trained_models: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Project every player in the Finals context."""
    if season_averages is None:
        season_averages = finals_context.get("regular_season_player_stats")
    if playoff_averages is None:
        playoff_averages = (
            finals_context.get("playoff_player_stats")
            or finals_context.get("player_stats")
        )
    if opponent_defense is None:
        opponent_defense = (
            finals_context.get("playoff_team_stats")
            or finals_context.get("team_stats")
        )

    live_player_stats = (
        finals_context.get("playoff_player_stats")
        or finals_context.get("player_stats")
        or {}
    )
    efficiency_lookup = _build_player_efficiency_lookup(live_player_stats)

    feature_rows = build_player_feature_rows(
        finals_context,
        season_averages=season_averages,
        playoff_averages=playoff_averages,
        recent_averages=recent_averages,
        opponent_defense=opponent_defense,
        matchup_adjustments=matchup_adjustments,
    )
    if trained_models:
        projections = predict_with_trained_models(feature_rows, trained_models)
    else:
        projections = project_players_from_rates(feature_rows)

    # Apply PIE-based efficiency boost when live player stats are available
    if efficiency_lookup:
        projections = [_apply_efficiency_boost(p, efficiency_lookup) for p in projections]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for projection in projections:
        grouped.setdefault(str(projection["team"]), []).append(projection)

    for team in grouped:
        grouped[team] = sorted(grouped[team], key=lambda row: row["minutes"], reverse=True)
    return grouped


def summarize_player_projections(
    player_projections: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Summarize projected player production by team."""
    summaries: dict[str, dict[str, float]] = {}
    for team, rows in player_projections.items():
        summaries[team] = {
            "minutes": round(sum(row["minutes"] for row in rows), 1),
            "points": round(sum(row["points"] for row in rows), 1),
            "rebounds": round(sum(row["rebounds"] for row in rows), 1),
            "assists": round(sum(row["assists"] for row in rows), 1),
            "turnovers": round(sum(row["turnovers"] for row in rows), 1),
            "foul_risk": round(sum(row["foul_risk"] for row in rows) / max(len(rows), 1), 3),
        }
    return summaries


def reconcile_player_projections_to_team_scores(
    player_projections: dict[str, list[dict[str, Any]]],
    expected_team_scores: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    """Scale player points so displayed player totals match the team forecast."""
    reconciled: dict[str, list[dict[str, Any]]] = {}
    for team, rows in player_projections.items():
        current_total = sum(_as_float(row.get("points"), 0.0) for row in rows)
        target = _as_float(expected_team_scores.get(team), current_total)
        multiplier = _clip(target / max(current_total, 1.0), 0.80, 1.20)
        reconciled[team] = []
        for row in rows:
            updated = dict(row)
            updated["points_before_team_reconciliation"] = row.get("points")
            updated["points"] = round(_as_float(row.get("points"), 0.0) * multiplier, 1)
            updated["team_score_reconciliation_factor"] = round(multiplier, 4)
            reconciled[team].append(updated)
    return reconciled


def simulate_correlated_player_box_scores(
    player_projections: dict[str, list[dict[str, Any]]],
    simulations: int = 2000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Simulate player outcomes with shared pace and team shooting factors."""
    rng = Random(random_seed)
    samples: dict[tuple[str, str], dict[str, list[float]]] = {}
    for team, rows in player_projections.items():
        for row in rows:
            samples[(team, str(row.get("player")))] = {
                "points": [], "rebounds": [], "assists": [],
            }

    for _ in range(simulations):
        pace_factor = rng.gauss(0.0, 0.035)
        for team, rows in player_projections.items():
            team_shooting = rng.gauss(0.0, 0.075)
            team_creation = rng.gauss(0.0, 0.045)
            for row in rows:
                player_noise = rng.gauss(0.0, 0.11)
                key = (team, str(row.get("player")))
                points = _as_float(row.get("points")) * (
                    1.0 + pace_factor + team_shooting + player_noise
                )
                rebounds = _as_float(row.get("rebounds")) * (
                    1.0 + pace_factor - 0.25 * team_shooting + rng.gauss(0.0, 0.10)
                )
                assists = _as_float(row.get("assists")) * (
                    1.0 + pace_factor + team_creation + 0.30 * team_shooting
                    + rng.gauss(0.0, 0.10)
                )
                samples[key]["points"].append(max(points, 0.0))
                samples[key]["rebounds"].append(max(rebounds, 0.0))
                samples[key]["assists"].append(max(assists, 0.0))

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        index = min(int((len(ordered) - 1) * fraction), len(ordered) - 1)
        return round(ordered[index], 1)

    players = []
    for (team, player), stats in samples.items():
        players.append({
            "team": team,
            "player": player,
            **{
                stat: {
                    "mean": round(sum(values) / len(values), 1),
                    "p10": percentile(values, 0.10),
                    "p90": percentile(values, 0.90),
                }
                for stat, values in stats.items()
            },
        })
    return {
        "simulations": simulations,
        "random_seed": random_seed,
        "correlation_structure": [
            "shared game pace",
            "shared team shooting",
            "shared team creation",
            "player-specific residual",
        ],
        "players": players,
    }


if __name__ == "__main__":
    from src.data.build_dataset import build_finals_context

    context = build_finals_context()
    projections_by_team = project_finals_players(
        context,
        matchup_adjustments=context.get("active_players"),
    )
    for team, projections in projections_by_team.items():
        print(team)
        for projection in projections:
            print(
                f"  {projection['player']}: "
                f"{projection['minutes']} min, "
                f"{projection['points']} pts, "
                f"{projection['assists']} ast, "
                f"{projection['rebounds']} reb, "
                f"{projection['turnovers']} tov, "
                f"{projection['true_shooting_proxy']:.3f} TS proxy"
            )
