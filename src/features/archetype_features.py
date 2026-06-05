"""Player and team archetype classification features."""

from __future__ import annotations

from math import isnan, sqrt
from typing import Any

from src.features.playstyle_features import build_playstyle_profiles


PLAYER_ARCHETYPES = {
    "rim_pressure_guard": "rim-pressure guard",
    "pull_up_shooting_guard": "pull-up shooting guard",
    "isolation_scorer": "isolation scorer",
    "spot_up_shooter": "spot-up shooter",
    "post_big": "post big",
    "rim_protector": "rim protector",
    "switchable_wing": "switchable wing",
    "stretch_big": "stretch big",
    "point_of_attack_defender": "point-of-attack defender",
    "foul_prone_defender": "foul-prone defender",
}

TEAM_ARCHETYPES = {
    "transition_heavy_team": "transition-heavy team",
    "slow_half_court_team": "slow half-court team",
    "three_point_volume_team": "three-point-volume team",
    "isolation_heavy_team": "isolation-heavy team",
    "pick_and_roll_heavy_team": "pick-and-roll-heavy team",
    "offensive_rebounding_team": "offensive-rebounding team",
    "switch_heavy_defense": "switch-heavy defense",
    "drop_coverage_defense": "drop-coverage defense",
    "rim_protection_defense": "rim-protection defense",
}

ALL_ARCHETYPES = {**PLAYER_ARCHETYPES, **TEAM_ARCHETYPES}

TEAM_KEYS = ["team", "TEAM", "TEAM_ABBREVIATION", "TEAM_NAME"]
PLAYER_KEYS = ["player", "PLAYER_NAME", "name", "Name"]

PLAYER_ALIASES = {
    "minutes": ["minutes", "MIN", "MP", "projected_minutes"],
    "points": ["points", "PTS", "pts"],
    "rebounds": ["rebounds", "REB", "TRB"],
    "assists": ["assists", "AST"],
    "turnovers": ["turnovers", "TOV", "TO"],
    "steals": ["steals", "STL"],
    "blocks": ["blocks", "BLK"],
    "threes_made": ["threes_made", "FG3M", "3PM"],
    "threes_attempted": ["threes_attempted", "FG3A", "3PA"],
    "free_throw_attempts": ["free_throw_attempts", "FTA"],
    "field_goal_attempts": ["field_goal_attempts", "FGA"],
    "personal_fouls": ["personal_fouls", "PF", "fouls"],
    "usage_rate": ["usage_rate", "USG_PCT", "USG%"],
    "true_shooting_proxy": ["true_shooting_proxy", "TS_PCT", "TS%"],
    "foul_risk": ["foul_risk"],
    "position": ["position", "POS", "player_position"],
    "height_inches": ["height_inches", "HEIGHT_INCHES"],
    "rim_frequency": ["rim_frequency", "rim_pressure", "PCT_FGA_RA"],
    "drives": ["drives", "DRIVES"],
    "pull_up_3_attempts": ["pull_up_3_attempts", "PULL_UP_FG3A"],
    "pull_up_frequency": ["pull_up_frequency", "pull_up_rate"],
    "catch_shoot_3_attempts": ["catch_shoot_3_attempts", "CATCH_SHOOT_FG3A"],
    "catch_shoot_frequency": ["catch_shoot_frequency", "spot_up_frequency"],
    "isolation_usage": ["isolation_usage", "iso_usage", "ISO_USAGE"],
    "post_up_usage": ["post_up_usage", "postup_usage", "POST_UP_USAGE"],
    "switchability": ["switchability", "switchability_score"],
    "point_of_attack_defense": ["point_of_attack_defense", "poa_defense"],
    "rim_protection": ["rim_protection", "rim_protection_score"],
    "deflections": ["deflections", "DEFLECTIONS"],
}

DIRECT_PLAYER_ARCHETYPE_ALIASES = {
    key: [key, f"{key}_score", PLAYER_ARCHETYPES[key], f"{PLAYER_ARCHETYPES[key]} score"]
    for key in PLAYER_ARCHETYPES
}


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


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _score(value: float, low: float, high: float, invert: bool = False) -> float:
    if high == low:
        return 50.0
    scaled = (value - low) / (high - low) * 100.0
    if invert:
        scaled = 100.0 - scaled
    return round(_clip(scaled), 1)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 1)


def _weighted_mean(parts: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in parts)
    if total_weight <= 0:
        return 0.0
    return round(sum(value * weight for value, weight in parts) / total_weight, 1)


def _percentage(value: Any, default: float = 0.0) -> float:
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


def _metric(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    return _as_float(_first_value(row, PLAYER_ALIASES[name]), default)


def _metric_percentage(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    return _percentage(_first_value(row, PLAYER_ALIASES[name]), default)


def _iter_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return [dict(row) for row in data.to_dict(orient="records")]
    if isinstance(data, dict):
        if any(key in data for key in PLAYER_KEYS + TEAM_KEYS):
            return [dict(data)]

        records: list[dict[str, Any]] = []
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


def _player(row: dict[str, Any]) -> str:
    return str(_first_value(row, PLAYER_KEYS) or "").strip()


def _position_group(row: dict[str, Any]) -> str:
    position = str(_first_value(row, PLAYER_ALIASES["position"]) or "").lower()
    height = _metric(row, "height_inches", 0.0) if "height_inches" in PLAYER_ALIASES else 0.0

    if any(token in position for token in ("c", "center", "big")) or height >= 80:
        return "big"
    if any(token in position for token in ("f", "wing", "sf", "pf")) or height >= 77:
        return "wing"
    if any(token in position for token in ("g", "guard", "pg", "sg")):
        return "guard"
    return "unknown"


def _position_score(position_group: str, target: str) -> float:
    if position_group == target:
        return 100.0
    if position_group == "unknown":
        return 50.0
    if {position_group, target} == {"wing", "guard"}:
        return 65.0
    if {position_group, target} == {"wing", "big"}:
        return 65.0
    return 20.0


def _direct_archetype_score(row: dict[str, Any], archetype_key: str) -> float | None:
    for key in DIRECT_PLAYER_ARCHETYPE_ALIASES[archetype_key]:
        if key in row and row[key] not in (None, ""):
            return _clip(_percentage(row[key]))
    return None


def _player_derived_metrics(row: dict[str, Any]) -> dict[str, float | str]:
    minutes = max(_metric(row, "minutes", 0.0), 1.0)
    points = _metric(row, "points", 0.0)
    assists = _metric(row, "assists", 0.0)
    rebounds = _metric(row, "rebounds", 0.0)
    steals = _metric(row, "steals", 0.0)
    blocks = _metric(row, "blocks", 0.0)
    threes_attempted = _metric(row, "threes_attempted", 0.0)
    free_throw_attempts = _metric(row, "free_throw_attempts", 0.0)
    personal_fouls = _metric(row, "personal_fouls", 0.0)

    fga = _metric(row, "field_goal_attempts", 0.0)
    if fga <= 0:
        fga = max(points / 1.35, threes_attempted, 1.0)

    position_group = _position_group(row)
    usage_rate = _metric_percentage(row, "usage_rate", 20.0)

    return {
        "minutes": minutes,
        "points_per_minute": points / minutes,
        "assists_per_minute": assists / minutes,
        "rebounds_per_minute": rebounds / minutes,
        "stocks_per_minute": (steals + blocks) / minutes,
        "blocks_per_minute": blocks / minutes,
        "steals_per_minute": steals / minutes,
        "threes_attempted_per_minute": threes_attempted / minutes,
        "free_throw_attempts_per_minute": free_throw_attempts / minutes,
        "free_throw_rate": free_throw_attempts / max(fga, 1.0),
        "personal_fouls_per_minute": personal_fouls / minutes,
        "usage_rate": usage_rate,
        "true_shooting_proxy": _metric_percentage(row, "true_shooting_proxy", 56.0),
        "foul_risk": _percentage(_first_value(row, PLAYER_ALIASES["foul_risk"]), 0.0),
        "rim_frequency": _metric_percentage(row, "rim_frequency", 0.0),
        "drives": _metric(row, "drives", 0.0),
        "pull_up_frequency": _metric_percentage(row, "pull_up_frequency", 0.0),
        "pull_up_3_attempts": _metric(row, "pull_up_3_attempts", 0.0),
        "catch_shoot_frequency": _metric_percentage(row, "catch_shoot_frequency", 0.0),
        "catch_shoot_3_attempts": _metric(row, "catch_shoot_3_attempts", 0.0),
        "isolation_usage": _metric_percentage(row, "isolation_usage", 0.0),
        "post_up_usage": _metric_percentage(row, "post_up_usage", 0.0),
        "switchability": _metric_percentage(row, "switchability", 0.0),
        "point_of_attack_defense": _metric_percentage(row, "point_of_attack_defense", 0.0),
        "rim_protection": _metric_percentage(row, "rim_protection", 0.0),
        "deflections": _metric(row, "deflections", 0.0),
        "position_group": position_group,
    }


def _player_archetype_scores(row: dict[str, Any]) -> dict[str, float]:
    metrics = _player_derived_metrics(row)
    guard_score = _position_score(str(metrics["position_group"]), "guard")
    wing_score = _position_score(str(metrics["position_group"]), "wing")
    big_score = _position_score(str(metrics["position_group"]), "big")

    rim_pressure = _mean(
        [
            _score(float(metrics["free_throw_attempts_per_minute"]), 0.04, 0.22),
            _score(float(metrics["free_throw_rate"]), 0.10, 0.55),
            _score(float(metrics["rim_frequency"]), 18.0, 48.0),
            _score(float(metrics["drives"]), 4.0, 18.0),
        ]
    )
    pull_up = _mean(
        [
            _score(float(metrics["pull_up_frequency"]), 6.0, 32.0),
            _score(float(metrics["pull_up_3_attempts"]), 1.0, 8.0),
            _score(float(metrics["threes_attempted_per_minute"]), 0.08, 0.30),
            _score(float(metrics["usage_rate"]), 17.0, 32.0),
        ]
    )
    isolation = _mean(
        [
            _score(float(metrics["isolation_usage"]), 4.0, 16.0),
            _score(float(metrics["usage_rate"]), 18.0, 34.0),
            _score(float(metrics["points_per_minute"]), 0.35, 0.85),
        ]
    )
    spot_up = _mean(
        [
            _score(float(metrics["catch_shoot_frequency"]), 10.0, 42.0),
            _score(float(metrics["catch_shoot_3_attempts"]), 1.5, 8.0),
            _score(float(metrics["threes_attempted_per_minute"]), 0.08, 0.32),
            _score(float(metrics["usage_rate"]), 28.0, 14.0),
        ]
    )
    post = _mean(
        [
            _score(float(metrics["post_up_usage"]), 3.0, 16.0),
            _score(float(metrics["rebounds_per_minute"]), 0.12, 0.34),
            big_score,
        ]
    )
    rim_protection = _mean(
        [
            _score(float(metrics["blocks_per_minute"]), 0.015, 0.085),
            _score(float(metrics["rim_protection"]), 45.0, 90.0),
            _score(float(metrics["rebounds_per_minute"]), 0.12, 0.32),
            big_score,
        ]
    )
    switchable = _mean(
        [
            _score(float(metrics["switchability"]), 45.0, 85.0),
            _score(float(metrics["stocks_per_minute"]), 0.035, 0.095),
            wing_score,
        ]
    )
    stretch = _mean(
        [
            big_score,
            _score(float(metrics["threes_attempted_per_minute"]), 0.08, 0.26),
            _score(float(metrics["true_shooting_proxy"]), 53.0, 63.0),
        ]
    )
    point_of_attack = _mean(
        [
            _score(float(metrics["point_of_attack_defense"]), 45.0, 85.0),
            _score(float(metrics["steals_per_minute"]), 0.015, 0.055),
            _score(float(metrics["deflections"]), 1.0, 4.0),
            _mean([guard_score, wing_score]),
        ]
    )
    foul_prone = _mean(
        [
            _score(float(metrics["foul_risk"]), 15.0, 75.0),
            _score(float(metrics["personal_fouls_per_minute"]), 0.045, 0.13),
        ]
    )

    scores = {
        "rim_pressure_guard": _weighted_mean([(rim_pressure, 0.75), (guard_score, 0.25)]),
        "pull_up_shooting_guard": _weighted_mean([(pull_up, 0.75), (guard_score, 0.25)]),
        "isolation_scorer": isolation,
        "spot_up_shooter": spot_up,
        "post_big": post,
        "rim_protector": rim_protection,
        "switchable_wing": switchable,
        "stretch_big": stretch,
        "point_of_attack_defender": point_of_attack,
        "foul_prone_defender": foul_prone,
    }

    for archetype_key in PLAYER_ARCHETYPES:
        direct_score = _direct_archetype_score(row, archetype_key)
        if direct_score is not None:
            scores[archetype_key] = round(direct_score, 1)

    return {key: round(_clip(value), 1) for key, value in scores.items()}


def _ranked_archetypes(scores: dict[str, float], labels: dict[str, str], min_score: float) -> list[dict[str, Any]]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {"key": key, "label": labels[key], "score": round(score, 1)}
        for key, score in ranked
        if score >= min_score
    ]


def classify_player_archetypes(
    player_row: dict[str, Any],
    min_secondary_score: float = 55.0,
) -> dict[str, Any]:
    """Classify one player into scored basketball archetypes."""
    scores = _player_archetype_scores(player_row)
    ranked = _ranked_archetypes(scores, PLAYER_ARCHETYPES, min_secondary_score)
    primary_key = max(scores, key=scores.get)

    return {
        "team": _team(player_row),
        "player": _player(player_row),
        "primary_archetype_key": primary_key,
        "primary_archetype": PLAYER_ARCHETYPES[primary_key],
        "primary_score": round(scores[primary_key], 1),
        "secondary_archetypes": [
            item for item in ranked if item["key"] != primary_key
        ],
        "scores": scores,
        "derived_metrics": _player_derived_metrics(player_row),
    }


def classify_players(
    player_rows: Any,
    min_secondary_score: float = 55.0,
) -> list[dict[str, Any]]:
    """Classify multiple players from a list, dict, or DataFrame."""
    classifications = [
        classify_player_archetypes(row, min_secondary_score)
        for row in _iter_records(player_rows)
    ]
    return sorted(classifications, key=lambda row: (row["team"], row["player"]))


def _profile_metric(team_profile: dict[str, Any], side: str, metric: str) -> float:
    return float(
        team_profile.get(side, {})
        .get("metrics", {})
        .get(metric, {})
        .get("score", 50.0)
    )


def _team_archetype_scores(team_profile: dict[str, Any]) -> dict[str, float]:
    offense = {
        metric: _profile_metric(team_profile, "offense", metric)
        for metric in [
            "pace",
            "half_court_reliance",
            "transition_frequency",
            "corner_3_frequency",
            "above_the_break_3_frequency",
            "isolation_usage",
            "pick_and_roll_usage",
            "offensive_rebounding",
        ]
    }
    defense = {
        metric: _profile_metric(team_profile, "defense", metric)
        for metric in [
            "switchability",
            "drop_coverage_strength",
            "rim_protection",
        ]
    }

    scores = {
        "transition_heavy_team": _weighted_mean(
            [(offense["transition_frequency"], 0.70), (offense["pace"], 0.30)]
        ),
        "slow_half_court_team": _weighted_mean(
            [(offense["half_court_reliance"], 0.65), (100.0 - offense["pace"], 0.35)]
        ),
        "three_point_volume_team": _mean(
            [offense["corner_3_frequency"], offense["above_the_break_3_frequency"]]
        ),
        "isolation_heavy_team": offense["isolation_usage"],
        "pick_and_roll_heavy_team": offense["pick_and_roll_usage"],
        "offensive_rebounding_team": offense["offensive_rebounding"],
        "switch_heavy_defense": defense["switchability"],
        "drop_coverage_defense": defense["drop_coverage_strength"],
        "rim_protection_defense": defense["rim_protection"],
    }
    return {key: round(_clip(value), 1) for key, value in scores.items()}


def classify_team_archetypes(
    team_profile: dict[str, Any],
    min_secondary_score: float = 60.0,
) -> dict[str, Any]:
    """Classify one team play-style profile into scored archetypes."""
    team = str(team_profile.get("team") or "").strip()
    scores = _team_archetype_scores(team_profile)
    ranked = _ranked_archetypes(scores, TEAM_ARCHETYPES, min_secondary_score)
    primary_key = max(scores, key=scores.get)

    return {
        "team": team,
        "primary_archetype_key": primary_key,
        "primary_archetype": TEAM_ARCHETYPES[primary_key],
        "primary_score": round(scores[primary_key], 1),
        "secondary_archetypes": [
            item for item in ranked if item["key"] != primary_key
        ],
        "scores": scores,
    }


def classify_teams(
    playstyle_profiles: dict[str, dict[str, Any]],
    min_secondary_score: float = 60.0,
) -> dict[str, dict[str, Any]]:
    """Classify every team play-style profile."""
    return {
        team: classify_team_archetypes(profile, min_secondary_score)
        for team, profile in playstyle_profiles.items()
    }


def _normalize_archetype_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    text = text.replace("__", "_")
    for key, label in ALL_ARCHETYPES.items():
        if text in {key, label.replace("-", "_").replace(" ", "_")}:
            return key
    return text


def _score_vector(profile: dict[str, Any]) -> dict[str, float]:
    return {
        _normalize_archetype_key(key): _as_float(value)
        for key, value in profile.get("scores", {}).items()
    }


def archetype_similarity(profile_a: dict[str, Any], profile_b: dict[str, Any]) -> float:
    """Calculate cosine similarity between two archetype score profiles."""
    vector_a = _score_vector(profile_a)
    vector_b = _score_vector(profile_b)
    keys = sorted(set(vector_a) | set(vector_b))
    if not keys:
        return 0.0

    dot = sum(vector_a.get(key, 0.0) * vector_b.get(key, 0.0) for key in keys)
    norm_a = sqrt(sum(vector_a.get(key, 0.0) ** 2 for key in keys))
    norm_b = sqrt(sum(vector_b.get(key, 0.0) ** 2 for key in keys))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return round(dot / (norm_a * norm_b), 3)


def _historical_profile_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    score_columns = {
        key: _as_float(row[column])
        for key in ALL_ARCHETYPES
        for column in (key, f"{key}_score")
        if column in row and row[column] not in (None, "")
    }
    if score_columns:
        return {"scores": score_columns}

    archetype_value = (
        row.get("opponent_archetype")
        or row.get("archetype")
        or row.get("primary_archetype_key")
        or row.get("primary_archetype")
    )
    key = _normalize_archetype_key(archetype_value)
    if key in ALL_ARCHETYPES:
        return {"scores": {key: 100.0}}
    return None


def _numeric_result_columns(rows: list[dict[str, Any]], excluded: set[str]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column, value in row.items():
            if column in excluded or column.endswith("_score"):
                continue
            if column in ALL_ARCHETYPES:
                continue
            try:
                _as_float(value)
            except Exception:
                continue
            if isinstance(value, (int, float)) or str(value).replace(".", "", 1).replace("-", "", 1).isdigit():
                if column not in columns:
                    columns.append(column)
    return columns


def compare_against_similar_archetypes(
    archetype_profile: dict[str, Any],
    historical_results: Any,
    result_columns: list[str] | None = None,
    min_similarity: float = 0.35,
) -> dict[str, Any]:
    """Summarize historical performance against similar archetype profiles."""
    rows = _iter_records(historical_results)
    excluded = {
        "team",
        "player",
        "opponent",
        "archetype",
        "opponent_archetype",
        "primary_archetype",
        "primary_archetype_key",
        "sample_size",
    }
    result_columns = result_columns or _numeric_result_columns(rows, excluded)

    matches = []
    for row in rows:
        historical_profile = _historical_profile_from_row(row)
        if not historical_profile:
            continue
        similarity = archetype_similarity(archetype_profile, historical_profile)
        if similarity < min_similarity:
            continue
        sample_size = max(_as_float(row.get("sample_size"), 1.0), 1.0)
        matches.append(
            {
                "row": row,
                "similarity": similarity,
                "weight": similarity * sample_size,
                "archetype": (
                    row.get("opponent_archetype")
                    or row.get("archetype")
                    or row.get("primary_archetype")
                    or row.get("primary_archetype_key")
                ),
            }
        )

    if not matches:
        return {
            "primary_archetype": archetype_profile.get("primary_archetype"),
            "matched_samples": 0,
            "weighted_sample_size": 0.0,
            "expected_results": {},
            "similar_archetypes": [],
            "interpretation": "No similar archetype history available yet.",
        }

    total_weight = sum(match["weight"] for match in matches)
    expected_results = {}
    for column in result_columns:
        weighted_values = [
            (_as_float(match["row"].get(column)), match["weight"])
            for match in matches
            if column in match["row"]
        ]
        if weighted_values:
            expected_results[column] = round(
                sum(value * weight for value, weight in weighted_values)
                / sum(weight for _, weight in weighted_values),
                3,
            )

    archetype_weights: dict[str, float] = {}
    for match in matches:
        archetype = str(match["archetype"] or "score-vector")
        archetype_weights[archetype] = archetype_weights.get(archetype, 0.0) + match["weight"]

    similar_archetypes = [
        {
            "archetype": archetype,
            "weight_share": round(weight / total_weight, 3),
        }
        for archetype, weight in sorted(archetype_weights.items(), key=lambda item: item[1], reverse=True)
    ]

    return {
        "primary_archetype": archetype_profile.get("primary_archetype"),
        "matched_samples": len(matches),
        "weighted_sample_size": round(total_weight, 3),
        "expected_results": expected_results,
        "similar_archetypes": similar_archetypes,
        "interpretation": (
            f"Matched {len(matches)} historical archetype rows most similar to "
            f"{archetype_profile.get('primary_archetype')}."
        ),
    }


def compare_player_against_similar_archetypes(
    player_archetype_profile: dict[str, Any],
    historical_results: Any,
    result_columns: list[str] | None = None,
    min_similarity: float = 0.35,
) -> dict[str, Any]:
    """Compare a player archetype profile to historical player-archetype results."""
    return compare_against_similar_archetypes(
        player_archetype_profile,
        historical_results,
        result_columns,
        min_similarity,
    )


def compare_team_against_similar_archetypes(
    team_archetype_profile: dict[str, Any],
    historical_results: Any,
    result_columns: list[str] | None = None,
    min_similarity: float = 0.35,
) -> dict[str, Any]:
    """Compare a team archetype profile to historical team-archetype results."""
    return compare_against_similar_archetypes(
        team_archetype_profile,
        historical_results,
        result_columns,
        min_similarity,
    )


def build_archetype_profiles(
    player_rows: Any = None,
    playstyle_profiles: dict[str, dict[str, Any]] | None = None,
    finals_context: dict[str, Any] | None = None,
    team_stats: Any = None,
) -> dict[str, Any]:
    """Build player and team archetype classifications for the Finals context."""
    if playstyle_profiles is None and finals_context is not None:
        playstyle_profiles = build_playstyle_profiles(finals_context, team_stats=team_stats)

    return {
        "players": classify_players(player_rows) if player_rows is not None else [],
        "teams": classify_teams(playstyle_profiles or {}),
    }


if __name__ == "__main__":
    from src.data.build_dataset import build_finals_context
    from src.models.train_player_model import project_finals_players

    context = build_finals_context()
    player_projections = project_finals_players(context)
    playstyle_profiles = build_playstyle_profiles(context)
    archetypes = build_archetype_profiles(player_projections, playstyle_profiles)

    for player in archetypes["players"]:
        print(f"{player['team']} {player['player']}: {player['primary_archetype']}")
    for team, profile in archetypes["teams"].items():
        print(f"{team}: {profile['primary_archetype']}")
