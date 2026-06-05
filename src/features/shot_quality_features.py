"""Shot-quality features for postgame process evaluation."""

from __future__ import annotations

from math import isnan
from typing import Any


TEAM_KEYS = ["team", "TEAM", "TEAM_ABBREVIATION", "TEAM_NAME"]

ALIASES = {
    "points": ["points", "PTS", "pts"],
    "field_goals_made": ["field_goals_made", "FGM", "fgm"],
    "field_goal_attempts": ["field_goal_attempts", "FGA", "fga"],
    "threes_made": ["threes_made", "FG3M", "3PM", "fg3m"],
    "threes_attempted": ["threes_attempted", "FG3A", "3PA", "fg3a"],
    "free_throw_attempts": ["free_throw_attempts", "FTA", "fta"],
    "turnovers": ["turnovers", "TOV", "TO", "tov"],
    "offensive_rebounds": ["offensive_rebounds", "OREB", "oreb"],
    "assists": ["assists", "AST", "ast"],
    "assisted_field_goals": ["assisted_field_goals", "AST_FGM", "assisted_fgm"],
    "assisted_shot_rate": ["assisted_shot_rate", "assist_rate", "AST_PCT", "ast_pct"],
    "rim_attempts": ["rim_attempts", "restricted_area_attempts", "RA_FGA", "rim_fga"],
    "corner_threes": ["corner_threes", "corner_3_attempts", "corner_three_attempts", "C3A"],
    "wide_open_threes": [
        "wide_open_threes",
        "wide_open_3_attempts",
        "wide_open_three_attempts",
        "WIDE_OPEN_FG3A",
    ],
    "midrange_attempts": ["midrange_attempts", "mid_range_attempts", "MID_FGA"],
    "offensive_rebound_chances": [
        "offensive_rebound_chances",
        "OREB_CHANCES",
        "oreb_chances",
    ],
    "transition_opportunities": [
        "transition_opportunities",
        "transition_attempts",
        "fast_break_opportunities",
        "FB_FGA",
    ],
    "contested_field_goals_made": [
        "contested_field_goals_made",
        "contested_fgm",
        "CONTESTED_FGM",
    ],
    "contested_field_goal_attempts": [
        "contested_field_goal_attempts",
        "contested_fga",
        "CONTESTED_FGA",
    ],
    "contested_shot_making": [
        "contested_shot_making",
        "contested_fg_pct",
        "CONTESTED_FG_PCT",
    ],
    "expected_efg": ["expected_efg", "xEFG", "expected_effective_fg_pct"],
    "shot_quality_score": ["shot_quality_score", "shot_quality"],
}

REPEATABLE_WEIGHTS = {
    "rim_attempt_rate": 0.22,
    "free_throw_rate": 0.16,
    "corner_three_rate": 0.15,
    "wide_open_three_rate": 0.12,
    "assisted_shot_rate": 0.10,
    "offensive_rebound_chance_rate": 0.10,
    "transition_opportunity_rate": 0.10,
    "contested_attempt_rate": -0.05,
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


def _score(value: float, low: float, high: float, invert: bool = False) -> float:
    if high == low:
        return 50.0
    scaled = (value - low) / (high - low) * 100.0
    if invert:
        scaled = 100.0 - scaled
    return round(_clip(scaled, 0.0, 100.0), 1)


def _percentage(value: Any, default: float = 0.0) -> float:
    number = _as_float(value, default)
    if 0.0 <= number <= 1.5:
        return number
    return number / 100.0


def _first_value(row: dict[str, Any] | None, keys: list[str]) -> Any:
    if not row:
        return None
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _metric(row: dict[str, Any], metric: str, default: float = 0.0) -> float:
    return _as_float(_first_value(row, ALIASES[metric]), default)


def _metric_percentage(row: dict[str, Any], metric: str, default: float = 0.0) -> float:
    value = _first_value(row, ALIASES[metric])
    if value is None:
        return default
    return _percentage(value, default)


def _iter_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return [dict(row) for row in data.to_dict(orient="records")]
    if isinstance(data, dict):
        if any(key in data for key in TEAM_KEYS):
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


def _team(row: dict[str, Any]) -> str:
    return str(_first_value(row, TEAM_KEYS) or "").strip()


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _estimated_possessions(row: dict[str, Any]) -> float:
    fga = _metric(row, "field_goal_attempts")
    fta = _metric(row, "free_throw_attempts")
    turnovers = _metric(row, "turnovers")
    offensive_rebounds = _metric(row, "offensive_rebounds")
    return max(fga + 0.44 * fta + turnovers - offensive_rebounds, 1.0)


def _actual_efg(row: dict[str, Any]) -> float:
    fgm = _metric(row, "field_goals_made")
    fga = _metric(row, "field_goal_attempts")
    threes_made = _metric(row, "threes_made")
    return _safe_rate(fgm + 0.5 * threes_made, fga)


def _assisted_shot_rate(row: dict[str, Any]) -> float:
    direct = _first_value(row, ALIASES["assisted_shot_rate"])
    if direct is not None:
        return _percentage(direct)

    assisted_fgm = _metric(row, "assisted_field_goals")
    fgm = _metric(row, "field_goals_made")
    if assisted_fgm > 0 and fgm > 0:
        return _safe_rate(assisted_fgm, fgm)

    assists = _metric(row, "assists")
    return _safe_rate(assists, max(fgm, 1.0))


def _contested_shot_making(row: dict[str, Any]) -> tuple[float, float]:
    direct = _first_value(row, ALIASES["contested_shot_making"])
    contested_attempts = _metric(row, "contested_field_goal_attempts")
    if direct is not None:
        return _percentage(direct), contested_attempts

    contested_makes = _metric(row, "contested_field_goals_made")
    if contested_attempts <= 0:
        return 0.0, 0.0
    return _safe_rate(contested_makes, contested_attempts), contested_attempts


def _expected_efg_proxy(
    rim_attempt_rate: float,
    free_throw_rate: float,
    corner_three_rate: float,
    wide_open_three_rate: float,
    assisted_shot_rate: float,
    transition_rate: float,
    contested_attempt_rate: float,
    row: dict[str, Any],
) -> float:
    direct = _first_value(row, ALIASES["expected_efg"])
    if direct is not None:
        return round(_percentage(direct), 3)

    expected = 0.505
    expected += (rim_attempt_rate - 0.32) * 0.22
    expected += (corner_three_rate - 0.10) * 0.16
    expected += (wide_open_three_rate - 0.14) * 0.12
    expected += (free_throw_rate - 0.24) * 0.05
    expected += (assisted_shot_rate - 0.58) * 0.06
    expected += (transition_rate - 0.13) * 0.08
    expected -= max(contested_attempt_rate - 0.34, 0.0) * 0.14
    return round(_clip(expected, 0.430, 0.620), 3)


def _shot_quality_score(
    rim_attempt_rate: float,
    free_throw_rate: float,
    corner_three_rate: float,
    wide_open_three_rate: float,
    assisted_shot_rate: float,
    offensive_rebound_chance_rate: float,
    transition_rate: float,
    contested_attempt_rate: float,
    expected_efg: float,
    row: dict[str, Any],
) -> float:
    direct = _first_value(row, ALIASES["shot_quality_score"])
    if direct is not None:
        value = _as_float(direct)
        return round(_clip(value if value > 1.5 else value * 100.0, 0.0, 100.0), 1)

    components = [
        _score(expected_efg, 0.470, 0.580) * 0.30,
        _score(rim_attempt_rate, 0.22, 0.42) * 0.18,
        _score(free_throw_rate, 0.16, 0.34) * 0.12,
        _score(corner_three_rate, 0.05, 0.17) * 0.12,
        _score(wide_open_three_rate, 0.07, 0.22) * 0.10,
        _score(assisted_shot_rate, 0.45, 0.70) * 0.08,
        _score(offensive_rebound_chance_rate, 0.18, 0.35) * 0.05,
        _score(transition_rate, 0.07, 0.20) * 0.05,
        _score(contested_attempt_rate, 0.48, 0.24) * 0.05,
    ]
    return round(_clip(sum(components), 0.0, 100.0), 1)


def _repeatability_score(
    rim_attempt_rate: float,
    free_throw_rate: float,
    corner_three_rate: float,
    wide_open_three_rate: float,
    assisted_shot_rate: float,
    offensive_rebound_chance_rate: float,
    transition_rate: float,
    contested_attempt_rate: float,
) -> float:
    scores = {
        "rim_attempt_rate": _score(rim_attempt_rate, 0.22, 0.42),
        "free_throw_rate": _score(free_throw_rate, 0.16, 0.34),
        "corner_three_rate": _score(corner_three_rate, 0.05, 0.17),
        "wide_open_three_rate": _score(wide_open_three_rate, 0.07, 0.22),
        "assisted_shot_rate": _score(assisted_shot_rate, 0.45, 0.70),
        "offensive_rebound_chance_rate": _score(offensive_rebound_chance_rate, 0.18, 0.35),
        "transition_opportunity_rate": _score(transition_rate, 0.07, 0.20),
        "contested_attempt_rate": _score(contested_attempt_rate, 0.24, 0.48),
    }
    total_weight = sum(abs(weight) for weight in REPEATABLE_WEIGHTS.values())
    weighted = 0.0
    for key, weight in REPEATABLE_WEIGHTS.items():
        if weight >= 0:
            weighted += scores[key] * weight
        else:
            weighted += (100.0 - scores[key]) * abs(weight)
    return round(_clip(weighted / total_weight, 0.0, 100.0), 1)


def _process_label(score: float) -> str:
    if score >= 65.0:
        return "strong shot quality"
    if score <= 35.0:
        return "poor shot quality"
    return "solid shot quality"


def calculate_team_shot_quality(row: dict[str, Any]) -> dict[str, Any]:
    """Calculate one team's shot-quality profile for a game."""
    team = _team(row)
    fga = _metric(row, "field_goal_attempts")
    possessions = _estimated_possessions(row)
    rim_attempts = _metric(row, "rim_attempts")
    fta = _metric(row, "free_throw_attempts")
    corner_threes = _metric(row, "corner_threes")
    wide_open_threes = _metric(row, "wide_open_threes")
    offensive_rebound_chances = _metric(row, "offensive_rebound_chances")
    transition_opportunities = _metric(row, "transition_opportunities")
    contested_making, contested_attempts = _contested_shot_making(row)
    assisted_rate = _assisted_shot_rate(row)

    rim_attempt_rate = _safe_rate(rim_attempts, fga)
    free_throw_rate = _safe_rate(fta, fga)
    corner_three_rate = _safe_rate(corner_threes, fga)
    wide_open_three_rate = _safe_rate(wide_open_threes, fga)
    offensive_rebound_chance_rate = _safe_rate(offensive_rebound_chances, possessions)
    transition_rate = _safe_rate(transition_opportunities, possessions)
    contested_attempt_rate = _safe_rate(contested_attempts, fga)

    expected_efg = _expected_efg_proxy(
        rim_attempt_rate,
        free_throw_rate,
        corner_three_rate,
        wide_open_three_rate,
        assisted_rate,
        transition_rate,
        contested_attempt_rate,
        row,
    )
    actual_efg = _actual_efg(row)
    shot_making_over_expectation = round(actual_efg - expected_efg, 3)
    shot_making_score = _score(shot_making_over_expectation, -0.080, 0.080)
    shot_quality_score = _shot_quality_score(
        rim_attempt_rate,
        free_throw_rate,
        corner_three_rate,
        wide_open_three_rate,
        assisted_rate,
        offensive_rebound_chance_rate,
        transition_rate,
        contested_attempt_rate,
        expected_efg,
        row,
    )

    repeatability_score = _repeatability_score(
        rim_attempt_rate,
        free_throw_rate,
        corner_three_rate,
        wide_open_three_rate,
        assisted_rate,
        offensive_rebound_chance_rate,
        transition_rate,
        contested_attempt_rate,
    )

    return {
        "team": team,
        "points": round(_metric(row, "points"), 1),
        "possessions_estimate": round(possessions, 1),
        "field_goal_attempts": round(fga, 1),
        "rim_attempts": round(rim_attempts, 1),
        "rim_attempt_rate": round(rim_attempt_rate, 3),
        "free_throw_attempts": round(fta, 1),
        "free_throw_rate": round(free_throw_rate, 3),
        "corner_threes": round(corner_threes, 1),
        "corner_three_rate": round(corner_three_rate, 3),
        "wide_open_threes": round(wide_open_threes, 1),
        "wide_open_three_rate": round(wide_open_three_rate, 3),
        "assisted_shot_rate": round(assisted_rate, 3),
        "offensive_rebound_chances": round(offensive_rebound_chances, 1),
        "offensive_rebound_chance_rate": round(offensive_rebound_chance_rate, 3),
        "transition_opportunities": round(transition_opportunities, 1),
        "transition_opportunity_rate": round(transition_rate, 3),
        "contested_field_goal_attempts": round(contested_attempts, 1),
        "contested_attempt_rate": round(contested_attempt_rate, 3),
        "contested_shot_making": round(contested_making, 3),
        "expected_efg": expected_efg,
        "actual_efg": round(actual_efg, 3),
        "shot_making_over_expectation": shot_making_over_expectation,
        "shot_making_score": shot_making_score,
        "shot_quality_score": shot_quality_score,
        "repeatability_score": repeatability_score,
        "process_label": _process_label(shot_quality_score),
    }


def build_shot_quality_features(game_team_stats: Any) -> dict[str, dict[str, Any]]:
    """Build shot-quality profiles for all teams in a game."""
    profiles = {}
    for row in _iter_records(game_team_stats):
        profile = calculate_team_shot_quality(row)
        if profile["team"]:
            profiles[profile["team"]] = profile
    return profiles


def _winner_from_scores(actual_scores: dict[str, float] | None) -> str | None:
    if not actual_scores:
        return None
    if len(actual_scores) < 2:
        return None
    return max(actual_scores, key=actual_scores.get)


def _team_scored(profile: dict[str, Any], actual_scores: dict[str, float] | None) -> float:
    if actual_scores and profile["team"] in actual_scores:
        return actual_scores[profile["team"]]
    return float(profile.get("points", 0.0))


def _update_recommendation(
    profile: dict[str, Any],
    opponent: dict[str, Any],
    won: bool | None,
) -> str:
    process_edge = profile["shot_quality_score"] - opponent["shot_quality_score"]
    repeatable_edge = profile["repeatability_score"] - opponent["repeatability_score"]
    making = profile["shot_making_over_expectation"]

    if won is False and process_edge >= 4.0:
        return (
            f"{profile['team']} lost but generated better shot quality. "
            "Do not downgrade heavily."
        )
    if won is True and repeatable_edge >= 4.0 and making <= 0.030:
        return (
            f"{profile['team']} won with repeatable rim pressure, free throws, corner threes, "
            "or transition chances. Upgrade slightly."
        )
    if won is True and making >= 0.055 and process_edge <= 1.0:
        return (
            f"{profile['team']} won while making shots above expected quality. "
            "Be careful about upgrading too much."
        )
    if won is False and making <= -0.055 and process_edge >= -1.0:
        return (
            f"{profile['team']} lost partly from poor shot making against acceptable shot quality. "
            "Downgrade only slightly."
        )
    if repeatable_edge <= -5.0:
        return (
            f"{profile['team']} created fewer repeatable advantages than its opponent. "
            "Downgrade the offensive process slightly."
        )
    return (
        f"{profile['team']}'s shot-quality signal is close to neutral. "
        "Make only a small update."
    )


def compare_game_shot_quality(
    shot_quality_profiles: dict[str, dict[str, Any]],
    actual_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare shot quality for a completed game and recommend update strength."""
    teams = list(shot_quality_profiles)
    if len(teams) != 2:
        raise ValueError("compare_game_shot_quality expects exactly two team profiles.")

    team_a, team_b = teams
    profile_a = shot_quality_profiles[team_a]
    profile_b = shot_quality_profiles[team_b]
    actual_scores = actual_scores or {
        team_a: _team_scored(profile_a, None),
        team_b: _team_scored(profile_b, None),
    }
    winner = _winner_from_scores(actual_scores)
    quality_winner = (
        team_a
        if profile_a["shot_quality_score"] >= profile_b["shot_quality_score"]
        else team_b
    )
    repeatability_winner = (
        team_a
        if profile_a["repeatability_score"] >= profile_b["repeatability_score"]
        else team_b
    )

    recommendations = {
        team_a: _update_recommendation(profile_a, profile_b, winner == team_a if winner else None),
        team_b: _update_recommendation(profile_b, profile_a, winner == team_b if winner else None),
    }

    return {
        "teams": teams,
        "scoreboard_winner": winner,
        "shot_quality_winner": quality_winner,
        "repeatability_winner": repeatability_winner,
        "shot_quality_edge": {
            team_a: round(profile_a["shot_quality_score"] - profile_b["shot_quality_score"], 1),
            team_b: round(profile_b["shot_quality_score"] - profile_a["shot_quality_score"], 1),
        },
        "repeatability_edge": {
            team_a: round(profile_a["repeatability_score"] - profile_b["repeatability_score"], 1),
            team_b: round(profile_b["repeatability_score"] - profile_a["repeatability_score"], 1),
        },
        "shot_making_over_expectation": {
            team_a: profile_a["shot_making_over_expectation"],
            team_b: profile_b["shot_making_over_expectation"],
        },
        "recommendations": recommendations,
    }


def evaluate_postgame_shot_quality(
    game_team_stats: Any,
    actual_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build and compare shot-quality profiles for one completed game."""
    profiles = build_shot_quality_features(game_team_stats)
    return {
        "profiles": profiles,
        "comparison": compare_game_shot_quality(profiles, actual_scores),
    }


def shot_quality_feature_vector(
    shot_quality_profiles: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Flatten shot-quality profiles into numeric model inputs."""
    features: dict[str, float] = {}
    numeric_keys = [
        "rim_attempt_rate",
        "free_throw_rate",
        "corner_three_rate",
        "wide_open_three_rate",
        "assisted_shot_rate",
        "offensive_rebound_chance_rate",
        "transition_opportunity_rate",
        "contested_attempt_rate",
        "contested_shot_making",
        "expected_efg",
        "actual_efg",
        "shot_making_over_expectation",
        "shot_making_score",
        "shot_quality_score",
        "repeatability_score",
    ]
    for team, profile in shot_quality_profiles.items():
        for key in numeric_keys:
            features[f"{team}_{key}"] = float(profile[key])
    return features


if __name__ == "__main__":
    sample = [
        {
            "team": "NYK",
            "PTS": 104,
            "FGM": 38,
            "FGA": 88,
            "FG3M": 10,
            "FG3A": 34,
            "FTA": 24,
            "TOV": 11,
            "OREB": 13,
            "AST": 24,
            "rim_attempts": 33,
            "corner_threes": 11,
            "wide_open_threes": 14,
            "offensive_rebound_chances": 30,
            "transition_opportunities": 13,
            "contested_fgm": 8,
            "contested_fga": 25,
        },
        {
            "team": "SAS",
            "PTS": 109,
            "FGM": 41,
            "FGA": 86,
            "FG3M": 14,
            "FG3A": 38,
            "FTA": 14,
            "TOV": 10,
            "OREB": 8,
            "AST": 19,
            "rim_attempts": 23,
            "corner_threes": 6,
            "wide_open_threes": 8,
            "offensive_rebound_chances": 20,
            "transition_opportunities": 9,
            "contested_fgm": 13,
            "contested_fga": 27,
        },
    ]
    result = evaluate_postgame_shot_quality(sample)
    print(result["comparison"])
