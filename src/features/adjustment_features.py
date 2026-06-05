"""Post-game adjustment features for model updates.

Compares one Finals prediction with the completed-game evidence and classifies
whether the result came from variance or repeatable basketball signals.
"""

from __future__ import annotations

from math import isnan
from typing import Any

from src.features.shot_quality_features import evaluate_postgame_shot_quality


TEAM_KEYS = ["team", "TEAM", "TEAM_ABBREVIATION", "TEAM_NAME"]
PLAYER_KEYS = ["player", "PLAYER_NAME", "name", "Name"]

TEAM_STAT_ALIASES = {
    "points": ["points", "PTS", "pts"],
    "pace": ["pace", "PACE", "possessions", "estimated_possessions"],
    "field_goal_attempts": ["field_goal_attempts", "FGA", "fga"],
    "threes_made": ["threes_made", "FG3M", "3PM", "fg3m"],
    "threes_attempted": ["threes_attempted", "FG3A", "3PA", "fg3a"],
    "offensive_rebounds": ["offensive_rebounds", "OREB", "oreb"],
    "defensive_rebounds": ["defensive_rebounds", "DREB", "dreb"],
    "total_rebounds": ["total_rebounds", "REB", "rebounds", "TRB"],
    "turnovers": ["turnovers", "TOV", "TO", "tov"],
    "personal_fouls": ["personal_fouls", "PF", "fouls"],
    "wide_open_threes": ["wide_open_threes", "wide_open_3_attempts", "wide_open_three_attempts"],
    "rim_attempts": ["rim_attempts", "restricted_area_attempts", "RA_FGA"],
    "free_throw_attempts": ["free_throw_attempts", "FTA", "fta"],
}

PLAYER_ALIASES = {
    "minutes": ["minutes", "MIN", "actual_minutes", "mp"],
    "projected_minutes": ["projected_minutes", "expected_minutes", "prediction_minutes"],
    "minutes_floor": ["minutes_floor", "floor_minutes"],
    "personal_fouls": ["personal_fouls", "PF", "fouls"],
}

LINEUP_ALIASES = {
    "minutes": ["minutes", "actual_minutes", "MIN"],
    "projected_minutes": ["projected_minutes", "expected_minutes"],
    "net_rating": ["net_rating", "NET_RATING"],
    "projected_net_rating": ["projected_net_rating", "expected_net_rating"],
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score(value: float, low: float, high: float, invert: bool = False) -> float:
    if high == low:
        return 50.0
    scaled = (value - low) / (high - low) * 100.0
    if invert:
        scaled = 100.0 - scaled
    return round(_clip(scaled, 0.0, 100.0), 1)


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


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
        if any(key in data for key in TEAM_KEYS + PLAYER_KEYS):
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


def _metric(row: dict[str, Any], aliases: dict[str, list[str]], metric: str, default: float = 0.0) -> float:
    return _as_float(_first_value(row, aliases[metric]), default)


def _team(row: dict[str, Any]) -> str:
    return str(_first_value(row, TEAM_KEYS) or "").strip()


def _player(row: dict[str, Any]) -> str:
    return str(_first_value(row, PLAYER_KEYS) or "").strip()


def _team_stat_rows(actual_game: dict[str, Any]) -> list[dict[str, Any]]:
    return _iter_records(
        actual_game.get("team_stats")
        or actual_game.get("box_score_team")
        or actual_game.get("team_box")
        or actual_game.get("teams")
    )


def _team_stat_index(actual_game: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {team: row for row in _team_stat_rows(actual_game) if (team := _team(row))}


def _actual_scores(
    actual_game: dict[str, Any],
    team_a: str,
    team_b: str,
) -> dict[str, float]:
    direct = actual_game.get("actual_scores") or actual_game.get("score") or {}
    if direct:
        return {
            team_a: _as_float(direct.get(team_a) or direct.get("team_a") or direct.get("home_score"), 0.0),
            team_b: _as_float(direct.get(team_b) or direct.get("team_b") or direct.get("away_score"), 0.0),
        }

    stats = _team_stat_index(actual_game)
    return {
        team_a: _metric(stats.get(team_a, {}), TEAM_STAT_ALIASES, "points", 0.0),
        team_b: _metric(stats.get(team_b, {}), TEAM_STAT_ALIASES, "points", 0.0),
    }


def _winner(scores: dict[str, float]) -> str | None:
    if len(scores) < 2:
        return None
    values = list(scores.values())
    if values[0] == values[1]:
        return None
    return max(scores, key=scores.get)


def _opponent(team: str, team_a: str, team_b: str) -> str:
    return team_b if team == team_a else team_a


def _team_order(
    predicted_game: dict[str, Any],
    actual_game: dict[str, Any],
    finals_context: dict[str, Any] | None,
) -> tuple[str, str]:
    team_a = str(
        predicted_game.get("team_a")
        or actual_game.get("team_a")
        or (finals_context or {}).get("team_a")
        or "Team A"
    )
    team_b = str(
        predicted_game.get("team_b")
        or actual_game.get("team_b")
        or (finals_context or {}).get("team_b")
        or "Team B"
    )
    return team_a, team_b


def result_comparison_features(
    predicted_game: dict[str, Any],
    actual_game: dict[str, Any],
    finals_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare expected score/probability with the actual result."""
    team_a, team_b = _team_order(predicted_game, actual_game, finals_context)
    scores = _actual_scores(actual_game, team_a, team_b)
    actual_winner = _winner(scores)
    predicted_probability = _clip(_as_float(predicted_game.get("team_a_win_probability"), 0.5), 0.01, 0.99)
    predicted_winner = team_a if predicted_probability >= 0.5 else team_b
    expected_margin = _as_float(predicted_game.get("expected_score_team_a"), 0.0) - _as_float(
        predicted_game.get("expected_score_team_b"),
        0.0,
    )
    actual_margin = scores.get(team_a, 0.0) - scores.get(team_b, 0.0)
    team_a_actual_result = 1.0 if actual_winner == team_a else 0.0

    return {
        "team_a": team_a,
        "team_b": team_b,
        "game_number": int(actual_game.get("game_number") or predicted_game.get("game_number") or 0),
        "actual_scores": scores,
        "actual_winner": actual_winner,
        "predicted_winner": predicted_winner,
        "prediction_was_correct": actual_winner == predicted_winner if actual_winner else None,
        "team_a_predicted_probability": round(predicted_probability, 4),
        "team_b_predicted_probability": round(1.0 - predicted_probability, 4),
        "team_a_actual_result": team_a_actual_result,
        "expected_margin_team_a": round(expected_margin, 1),
        "actual_margin_team_a": round(actual_margin, 1),
        "margin_error_team_a": round(actual_margin - expected_margin, 1),
        "signed_probability_surprise_team_a": round(team_a_actual_result - predicted_probability, 4),
        "absolute_probability_surprise": round(abs(team_a_actual_result - predicted_probability), 4),
    }


def shot_profile_adjustment_features(
    actual_game: dict[str, Any],
    actual_scores: dict[str, float],
) -> dict[str, Any]:
    """Analyze whether shot process or shot making drove the result."""
    rows = _team_stat_rows(actual_game)
    if len(rows) != 2:
        return {
            "available": False,
            "profiles": {},
            "comparison": {},
            "shot_quality_winner": None,
            "repeatability_winner": None,
        }

    report = evaluate_postgame_shot_quality(rows, actual_scores)
    comparison = report["comparison"]
    return {
        "available": True,
        "profiles": report["profiles"],
        "comparison": comparison,
        "shot_quality_winner": comparison.get("shot_quality_winner"),
        "repeatability_winner": comparison.get("repeatability_winner"),
        "shot_quality_edge": comparison.get("shot_quality_edge", {}),
        "repeatability_edge": comparison.get("repeatability_edge", {}),
        "shot_making_over_expectation": comparison.get("shot_making_over_expectation", {}),
        "recommendations": comparison.get("recommendations", {}),
    }


def box_score_adjustment_features(
    actual_game: dict[str, Any],
    team_a: str,
    team_b: str,
    predicted_game: dict[str, Any],
) -> dict[str, Any]:
    """Extract pace, rebounding, turnover, and three-point deltas."""
    stats = _team_stat_index(actual_game)
    row_a = stats.get(team_a, {})
    row_b = stats.get(team_b, {})
    actual_pace = _as_float(actual_game.get("actual_pace") or actual_game.get("pace"), 0.0)
    if actual_pace <= 0:
        actual_pace = (
            _metric(row_a, TEAM_STAT_ALIASES, "pace", 0.0)
            + _metric(row_b, TEAM_STAT_ALIASES, "pace", 0.0)
        ) / 2.0

    projected_pace = _as_float(predicted_game.get("projected_pace"), actual_pace or 98.0)
    team_a_oreb = _metric(row_a, TEAM_STAT_ALIASES, "offensive_rebounds", 0.0)
    team_b_oreb = _metric(row_b, TEAM_STAT_ALIASES, "offensive_rebounds", 0.0)
    team_a_reb = _metric(row_a, TEAM_STAT_ALIASES, "total_rebounds", 0.0)
    team_b_reb = _metric(row_b, TEAM_STAT_ALIASES, "total_rebounds", 0.0)
    team_a_tov = _metric(row_a, TEAM_STAT_ALIASES, "turnovers", 0.0)
    team_b_tov = _metric(row_b, TEAM_STAT_ALIASES, "turnovers", 0.0)
    team_a_3pa = _metric(row_a, TEAM_STAT_ALIASES, "threes_attempted", 0.0)
    team_b_3pa = _metric(row_b, TEAM_STAT_ALIASES, "threes_attempted", 0.0)

    return {
        "actual_pace": round(actual_pace, 1),
        "projected_pace": round(projected_pace, 1),
        "pace_delta": round(actual_pace - projected_pace, 1),
        "offensive_rebound_edge_team_a": round(team_a_oreb - team_b_oreb, 1),
        "total_rebound_edge_team_a": round(team_a_reb - team_b_reb, 1),
        "turnover_edge_team_a": round(team_b_tov - team_a_tov, 1),
        "team_a_three_point_pct": round(_safe_rate(_metric(row_a, TEAM_STAT_ALIASES, "threes_made"), team_a_3pa), 3),
        "team_b_three_point_pct": round(_safe_rate(_metric(row_b, TEAM_STAT_ALIASES, "threes_made"), team_b_3pa), 3),
        "team_a_wide_open_threes": round(_metric(row_a, TEAM_STAT_ALIASES, "wide_open_threes", 0.0), 1),
        "team_b_wide_open_threes": round(_metric(row_b, TEAM_STAT_ALIASES, "wide_open_threes", 0.0), 1),
        "team_a_rim_attempts": round(_metric(row_a, TEAM_STAT_ALIASES, "rim_attempts", 0.0), 1),
        "team_b_rim_attempts": round(_metric(row_b, TEAM_STAT_ALIASES, "rim_attempts", 0.0), 1),
        "team_a_free_throw_attempts": round(_metric(row_a, TEAM_STAT_ALIASES, "free_throw_attempts", 0.0), 1),
        "team_b_free_throw_attempts": round(_metric(row_b, TEAM_STAT_ALIASES, "free_throw_attempts", 0.0), 1),
    }


def _rotation_minutes_lookup(finals_context: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    lookup = {}
    for team, rotations in (finals_context or {}).get("rotations", {}).items():
        for rotation in rotations:
            player = str(rotation.get("player") or "").strip()
            if player:
                lookup[(team, player)] = rotation
    return lookup


def minutes_adjustment_features(
    actual_game: dict[str, Any],
    finals_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare actual player minutes with the projected rotation."""
    player_rows = _iter_records(
        actual_game.get("player_minutes")
        or actual_game.get("box_score_players")
        or actual_game.get("player_box")
        or actual_game.get("players")
    )
    rotation_lookup = _rotation_minutes_lookup(finals_context)
    major_changes = []
    total_abs_delta = 0.0

    for row in player_rows:
        team = _team(row)
        player = _player(row)
        if not team or not player:
            continue
        rotation = rotation_lookup.get((team, player), {})
        actual_minutes = _metric(row, PLAYER_ALIASES, "minutes", 0.0)
        projected_minutes = _metric(
            row,
            PLAYER_ALIASES,
            "projected_minutes",
            _as_float(rotation.get("projected_minutes"), actual_minutes),
        )
        minutes_floor = _metric(row, PLAYER_ALIASES, "minutes_floor", _as_float(rotation.get("minutes_floor"), projected_minutes - 4.0))
        delta = actual_minutes - projected_minutes
        total_abs_delta += abs(delta)
        if abs(delta) >= 6.0 or actual_minutes < minutes_floor - 2.0:
            major_changes.append(
                {
                    "team": team,
                    "player": player,
                    "actual_minutes": round(actual_minutes, 1),
                    "projected_minutes": round(projected_minutes, 1),
                    "minutes_delta": round(delta, 1),
                    "reason": "below floor" if actual_minutes < minutes_floor - 2.0 else "large minutes change",
                }
            )

    score = _score(total_abs_delta, 10.0, 58.0)
    return {
        "score": score,
        "total_abs_minutes_delta": round(total_abs_delta, 1),
        "major_changes": sorted(major_changes, key=lambda row: abs(row["minutes_delta"]), reverse=True),
    }


def lineup_adjustment_features(actual_game: dict[str, Any]) -> dict[str, Any]:
    """Detect meaningful lineup usage or lineup-performance changes."""
    rows = _iter_records(actual_game.get("lineups") or actual_game.get("actual_lineups"))
    major_changes = []
    total_minutes_delta = 0.0
    biggest_underperformance = 0.0

    for row in rows:
        team = _team(row)
        lineup_name = str(row.get("lineup_type") or row.get("lineup") or row.get("players") or "lineup")
        actual_minutes = _metric(row, LINEUP_ALIASES, "minutes", 0.0)
        projected_minutes = _metric(row, LINEUP_ALIASES, "projected_minutes", actual_minutes)
        actual_net = _metric(row, LINEUP_ALIASES, "net_rating", 0.0)
        projected_net = _metric(row, LINEUP_ALIASES, "projected_net_rating", actual_net)
        minutes_delta = actual_minutes - projected_minutes
        net_delta = actual_net - projected_net
        total_minutes_delta += abs(minutes_delta)
        biggest_underperformance = min(biggest_underperformance, net_delta)
        if abs(minutes_delta) >= 4.0 or abs(net_delta) >= 8.0:
            major_changes.append(
                {
                    "team": team,
                    "lineup": lineup_name,
                    "minutes_delta": round(minutes_delta, 1),
                    "net_rating_delta": round(net_delta, 1),
                }
            )

    score = _clip(_score(total_minutes_delta, 4.0, 28.0) * 0.65 + _score(abs(biggest_underperformance), 4.0, 22.0) * 0.35, 0.0, 100.0)
    return {
        "score": round(score, 1),
        "total_lineup_minutes_delta": round(total_minutes_delta, 1),
        "biggest_lineup_underperformance": round(biggest_underperformance, 1),
        "major_changes": major_changes,
    }


def foul_trouble_adjustment_features(
    actual_game: dict[str, Any],
    finals_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score whether foul trouble materially changed expected minutes."""
    direct_rows = _iter_records(actual_game.get("foul_trouble") or actual_game.get("foul_trouble_events"))
    player_rows = _iter_records(
        actual_game.get("player_minutes")
        or actual_game.get("box_score_players")
        or actual_game.get("player_box")
        or actual_game.get("players")
    )
    rotation_lookup = _rotation_minutes_lookup(finals_context)
    events = []

    for row in direct_rows:
        team = _team(row)
        player = _player(row)
        minutes_lost = _as_float(row.get("minutes_lost") or row.get("expected_minutes_lost"), 0.0)
        fouls = _as_float(row.get("fouls") or row.get("personal_fouls") or row.get("PF"), 0.0)
        primary = _as_bool(row.get("primary_rim_protector") or row.get("primary_defender"))
        events.append(
            {
                "team": team,
                "player": player,
                "fouls": round(fouls, 1),
                "minutes_lost": round(minutes_lost, 1),
                "primary_defensive_role": primary,
            }
        )

    for row in player_rows:
        team = _team(row)
        player = _player(row)
        rotation = rotation_lookup.get((team, player), {})
        actual_minutes = _metric(row, PLAYER_ALIASES, "minutes", 0.0)
        projected_minutes = _metric(
            row,
            PLAYER_ALIASES,
            "projected_minutes",
            _as_float(rotation.get("projected_minutes"), actual_minutes),
        )
        fouls = _metric(row, PLAYER_ALIASES, "personal_fouls", 0.0)
        if fouls >= 5.0 and projected_minutes - actual_minutes >= 4.0:
            events.append(
                {
                    "team": team,
                    "player": player,
                    "fouls": round(fouls, 1),
                    "minutes_lost": round(projected_minutes - actual_minutes, 1),
                    "primary_defensive_role": "center" in str(rotation.get("role", "")).lower(),
                }
            )

    score = 0.0
    for event in events:
        score += event["minutes_lost"] * 4.0 + event["fouls"] * 4.0
        if event["primary_defensive_role"]:
            score += 14.0
    return {
        "score": round(_clip(score, 0.0, 100.0), 1),
        "events": sorted(events, key=lambda row: row["minutes_lost"], reverse=True),
    }


def injury_adjustment_features(actual_game: dict[str, Any]) -> dict[str, Any]:
    """Detect injury or availability changes after the game."""
    rows = _iter_records(actual_game.get("injuries") or actual_game.get("injury_updates"))
    events = []
    score = 0.0
    for row in rows:
        team = _team(row)
        player = _player(row)
        status_change = str(row.get("status_change") or row.get("status") or "").strip()
        minutes_adjustment = _as_float(row.get("expected_minutes_adjustment") or row.get("minutes_adjustment"), 0.0)
        if not status_change and minutes_adjustment == 0:
            continue
        event_score = min(abs(minutes_adjustment) * 4.0, 48.0)
        if status_change.lower() in {"out", "doubtful", "questionable", "left game"}:
            event_score += 30.0
        score += event_score
        events.append(
            {
                "team": team,
                "player": player,
                "status_change": status_change,
                "expected_minutes_adjustment": round(minutes_adjustment, 1),
                "event_score": round(event_score, 1),
            }
        )
    return {"score": round(_clip(score, 0.0, 100.0), 1), "events": events}


def coaching_adjustment_features(actual_game: dict[str, Any]) -> dict[str, Any]:
    """Score coaching and defensive assignment changes."""
    coaching_rows = _iter_records(actual_game.get("coaching_adjustments") or actual_game.get("coaching_notes"))
    assignment_rows = _iter_records(actual_game.get("defensive_assignments") or actual_game.get("assignments"))
    events = []
    score_by_team: dict[str, float] = {}

    for row in coaching_rows:
        team = _team(row)
        impact = _as_float(row.get("actual_impact") or row.get("expected_impact") or row.get("impact"), 0.0)
        success = _as_float(row.get("success_score"), 0.0)
        event_score = abs(impact) * 18.0 + success * 0.35 + 16.0
        score_by_team[team] = score_by_team.get(team, 0.0) + event_score
        events.append(
            {
                "team": team,
                "type": row.get("adjustment_type") or row.get("type"),
                "description": row.get("description") or row.get("notes"),
                "score": round(event_score, 1),
            }
        )

    for row in assignment_rows:
        team = _team(row)
        changed = _as_bool(row.get("changed") or row.get("new_assignment"))
        success = _as_float(row.get("success_score"), 0.0)
        possessions = _as_float(row.get("possessions"), 0.0)
        if changed or success:
            event_score = (18.0 if changed else 5.0) + success * 0.35 + min(possessions * 0.25, 12.0)
            score_by_team[team] = score_by_team.get(team, 0.0) + event_score
            events.append(
                {
                    "team": team,
                    "type": "defensive_assignment",
                    "description": row.get("description") or row.get("notes"),
                    "score": round(event_score, 1),
                }
            )

    total_score = _clip(sum(score_by_team.values()), 0.0, 100.0)
    leading_team = max(score_by_team, key=score_by_team.get) if score_by_team else None
    return {
        "score": round(total_score, 1),
        "leading_team": leading_team,
        "score_by_team": {team: round(score, 1) for team, score in score_by_team.items()},
        "events": sorted(events, key=lambda row: row["score"], reverse=True),
    }


def _edge_for_team(edge_map: dict[str, Any], team: str) -> float:
    return _as_float(edge_map.get(team), 0.0)


def _team_from_edge(edge_map: dict[str, Any]) -> str | None:
    if not edge_map:
        return None
    return max(edge_map, key=lambda team: _as_float(edge_map.get(team), 0.0))


def classify_adjustment_causes(features: dict[str, Any]) -> list[dict[str, Any]]:
    """Classify why the result happened."""
    result = features["result"]
    team_a = result["team_a"]
    team_b = result["team_b"]
    winner = result.get("actual_winner")
    loser = _opponent(winner, team_a, team_b) if winner else None
    shot = features["shot_profile"]
    box = features["box_score"]
    shot_profiles = shot.get("profiles", {})
    comparison = shot.get("comparison", {})

    winner_making = _as_float(shot.get("shot_making_over_expectation", {}).get(winner), 0.0)
    loser_making = _as_float(shot.get("shot_making_over_expectation", {}).get(loser), 0.0)
    winner_quality_edge = _edge_for_team(shot.get("shot_quality_edge", {}), winner or "")
    winner_repeatability_edge = _edge_for_team(shot.get("repeatability_edge", {}), winner or "")
    winner_profile = shot_profiles.get(winner or "", {})
    winner_three_pct = box.get("team_a_three_point_pct") if winner == team_a else box.get("team_b_three_point_pct")
    winner_wide_open = box.get("team_a_wide_open_threes") if winner == team_a else box.get("team_b_wide_open_threes")
    winner_oreb_edge = box.get("offensive_rebound_edge_team_a", 0.0)
    winner_turnover_edge = box.get("turnover_edge_team_a", 0.0)
    if winner == team_b:
        winner_oreb_edge *= -1.0
        winner_turnover_edge *= -1.0

    difficult_attempt_bonus = 0.0
    if winner_profile:
        difficult_attempt_bonus = _score(
            _as_float(winner_profile.get("contested_attempt_rate"), 0.0),
            0.28,
            0.48,
        )
        if _as_float(winner_profile.get("wide_open_three_rate"), 0.0) >= 0.18:
            difficult_attempt_bonus *= 0.25

    shooting_variance_score = 0.0
    if winner:
        shooting_variance_score += _score(winner_making, 0.025, 0.090) * 0.45
        shooting_variance_score += _score(_as_float(winner_three_pct, 0.0), 0.380, 0.500) * 0.20
        shooting_variance_score += _score(max(0.0, -winner_quality_edge), 0.0, 8.0) * 0.15
        shooting_variance_score += _score(max(0.0, -winner_repeatability_edge), 0.0, 8.0) * 0.10
        shooting_variance_score += difficult_attempt_bonus * 0.10
        if loser_making <= -0.045:
            shooting_variance_score += 8.0

    real_matchup_score = 0.0
    if winner:
        real_matchup_score += _score(winner_repeatability_edge, 2.0, 12.0) * 0.32
        real_matchup_score += _score(winner_quality_edge, 2.0, 12.0) * 0.28
        real_matchup_score += _score(_as_float(winner_wide_open, 0.0), 10.0, 22.0) * 0.16
        real_matchup_score += _score(winner_oreb_edge, 2.0, 9.0) * 0.12
        real_matchup_score += _score(winner_turnover_edge, 2.0, 8.0) * 0.12

    coaching = features["coaching"]
    injuries = features["injuries"]
    minutes = features["minutes"]
    lineups = features["lineups"]
    foul_trouble = features["foul_trouble"]
    injury_minutes_score = _clip(injuries["score"] * 0.58 + minutes["score"] * 0.42, 0.0, 100.0)

    causes = [
        {
            "cause": "shooting_variance",
            "label": "Shooting variance",
            "score": round(_clip(shooting_variance_score, 0.0, 100.0), 1),
            "evidence_team": winner,
            "reason": _shooting_variance_reason(winner, winner_making, winner_three_pct, winner_quality_edge),
        },
        {
            "cause": "real_matchup_advantage",
            "label": "Real matchup advantage",
            "score": round(_clip(real_matchup_score, 0.0, 100.0), 1),
            "evidence_team": comparison.get("repeatability_winner") or comparison.get("shot_quality_winner"),
            "reason": _matchup_reason(comparison.get("repeatability_winner"), winner_repeatability_edge, winner_wide_open),
        },
        {
            "cause": "coaching_adjustment",
            "label": "Coaching adjustment",
            "score": coaching["score"],
            "evidence_team": coaching.get("leading_team"),
            "reason": _first_event_reason(coaching.get("events"), "coaching or assignment change"),
        },
        {
            "cause": "injury_minutes_change",
            "label": "Injury/minutes change",
            "score": round(injury_minutes_score, 1),
            "evidence_team": _event_team(injuries.get("events")) or _event_team(minutes.get("major_changes")),
            "reason": _first_event_reason(injuries.get("events") or minutes.get("major_changes"), "rotation minutes changed"),
        },
        {
            "cause": "lineup_change",
            "label": "Lineup change",
            "score": lineups["score"],
            "evidence_team": _event_team(lineups.get("major_changes")),
            "reason": _first_event_reason(lineups.get("major_changes"), "lineup usage changed"),
        },
        {
            "cause": "foul_trouble",
            "label": "Foul trouble",
            "score": foul_trouble["score"],
            "evidence_team": _event_team(foul_trouble.get("events")),
            "reason": _first_event_reason(foul_trouble.get("events"), "foul trouble changed minutes"),
        },
    ]
    return sorted(causes, key=lambda row: row["score"], reverse=True)


def _shooting_variance_reason(
    winner: str | None,
    making: float,
    three_pct: Any,
    quality_edge: float,
) -> str:
    if not winner:
        return "No clear winner signal."
    return (
        f"{winner} finished {making:+.1%} above expected shot quality, "
        f"shot {_as_float(three_pct, 0.0):.1%} from three, "
        f"and had a shot-quality edge of {quality_edge:+.1f}."
    )


def _matchup_reason(
    repeatability_winner: str | None,
    repeatability_edge: float,
    wide_open_threes: Any,
) -> str:
    if not repeatability_winner:
        return "No strong repeatable process edge was detected."
    return (
        f"{repeatability_winner} had the repeatability signal; "
        f"winner repeatability edge was {repeatability_edge:+.1f} "
        f"with {_as_float(wide_open_threes, 0.0):.0f} wide-open threes."
    )


def _first_event_reason(events: list[dict[str, Any]] | None, fallback: str) -> str:
    if not events:
        return fallback
    first = events[0]
    parts = [str(first.get("team") or "").strip(), str(first.get("player") or first.get("lineup") or first.get("type") or "").strip()]
    descriptor = " ".join(part for part in parts if part)
    detail = first.get("description") or first.get("reason") or fallback
    return f"{descriptor}: {detail}" if descriptor else str(detail)


def _event_team(events: list[dict[str, Any]] | None) -> str | None:
    if not events:
        return None
    return str(events[0].get("team") or "") or None


def update_strength_label(score: float) -> str:
    """Convert a 0-100 evidence score into a readable update label."""
    if score >= 70.0:
        return "strong update"
    if score >= 40.0:
        return "moderate update"
    return "small update"


def recommend_probability_update(features: dict[str, Any], causes: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate how much Team A probability should move after this game."""
    result = features["result"]
    team_a = result["team_a"]
    team_b = result["team_b"]
    primary = causes[0] if causes else {"cause": "neutral", "score": 0.0}
    signed_outcome_surprise = _as_float(result["signed_probability_surprise_team_a"], 0.0)
    predicted_probability = _as_float(result["team_a_predicted_probability"], 0.5)

    process_team = features["shot_profile"].get("repeatability_winner") or features["shot_profile"].get("shot_quality_winner")
    if process_team == team_a:
        process_target = 1.0
    elif process_team == team_b:
        process_target = 0.0
    else:
        process_target = predicted_probability
    process_surprise = process_target - predicted_probability

    primary_score = _as_float(primary.get("score"), 0.0)
    if primary["cause"] == "shooting_variance" and primary_score >= 45.0:
        scoreboard_weight = 0.045
        process_weight = 0.055
    elif primary["cause"] == "real_matchup_advantage":
        scoreboard_weight = 0.105
        process_weight = 0.105
    elif primary["cause"] in {"injury_minutes_change", "foul_trouble"}:
        scoreboard_weight = 0.105
        process_weight = 0.055
    elif primary["cause"] in {"coaching_adjustment", "lineup_change"}:
        scoreboard_weight = 0.085
        process_weight = 0.070
    else:
        scoreboard_weight = 0.060
        process_weight = 0.050

    raw_move = signed_outcome_surprise * scoreboard_weight + process_surprise * process_weight
    capped_move = _clip(raw_move, -0.085, 0.085)
    if primary["cause"] == "shooting_variance":
        if abs(capped_move) < 0.015:
            strength = "small update"
        elif abs(capped_move) < 0.035:
            strength = "moderate update"
        else:
            strength = "strong update"
    else:
        evidence_score = max(primary_score, abs(capped_move) * 1000.0)
        strength = update_strength_label(evidence_score)

    return {
        "team_a_probability_shift": round(capped_move, 4),
        "team_b_probability_shift": round(-capped_move, 4),
        "primary_cause": primary["cause"],
        "primary_cause_label": primary.get("label"),
        "primary_evidence_score": round(primary_score, 1),
        "update_strength": strength,
        "scoreboard_weight": scoreboard_weight,
        "process_weight": process_weight,
        "process_team": process_team,
        "recommendation": _update_recommendation_text(primary, capped_move, team_a, team_b),
    }


def _update_recommendation_text(
    primary: dict[str, Any],
    team_a_move: float,
    team_a: str,
    team_b: str,
) -> str:
    direction = team_a if team_a_move > 0 else team_b if team_a_move < 0 else "neither team"
    if primary["cause"] == "shooting_variance":
        return f"Treat the result as noisy shot making; move only slightly toward {direction}."
    if primary["cause"] == "real_matchup_advantage":
        return f"Repeatable process showed up; move future games more meaningfully toward {direction}."
    if primary["cause"] == "coaching_adjustment":
        return f"Coaching or assignment changes looked material; bake a moderate move toward {direction}."
    if primary["cause"] == "injury_minutes_change":
        return f"Availability or rotation changes altered the baseline; update minutes before trusting the old forecast."
    if primary["cause"] == "lineup_change":
        return f"Lineup usage changed the game environment; make a moderate lineup-sensitive update."
    if primary["cause"] == "foul_trouble":
        return f"Foul trouble affected key minutes; adjust cautiously unless it is repeatable matchup pressure."
    return "Evidence was mixed; keep the model close to the pre-game baseline."


def build_adjustment_features(
    predicted_game: dict[str, Any],
    actual_game: dict[str, Any],
    finals_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full post-game adjustment feature bundle."""
    result = result_comparison_features(predicted_game, actual_game, finals_context)
    team_a = result["team_a"]
    team_b = result["team_b"]
    shot_profile = shot_profile_adjustment_features(actual_game, result["actual_scores"])
    features = {
        "result": result,
        "shot_profile": shot_profile,
        "box_score": box_score_adjustment_features(actual_game, team_a, team_b, predicted_game),
        "minutes": minutes_adjustment_features(actual_game, finals_context),
        "lineups": lineup_adjustment_features(actual_game),
        "foul_trouble": foul_trouble_adjustment_features(actual_game, finals_context),
        "injuries": injury_adjustment_features(actual_game),
        "coaching": coaching_adjustment_features(actual_game),
    }
    causes = classify_adjustment_causes(features)
    features["cause_classification"] = causes
    features["primary_cause"] = causes[0] if causes else None
    features["model_update"] = recommend_probability_update(features, causes)
    features["feature_vector"] = adjustment_feature_vector(features)
    return features


def adjustment_feature_vector(features: dict[str, Any]) -> dict[str, float]:
    """Flatten adjustment features into numeric model inputs."""
    result = features["result"]
    box = features["box_score"]
    vector = {
        "absolute_probability_surprise": float(result["absolute_probability_surprise"]),
        "signed_probability_surprise_team_a": float(result["signed_probability_surprise_team_a"]),
        "margin_error_team_a": float(result["margin_error_team_a"]),
        "pace_delta": float(box["pace_delta"]),
        "offensive_rebound_edge_team_a": float(box["offensive_rebound_edge_team_a"]),
        "turnover_edge_team_a": float(box["turnover_edge_team_a"]),
        "minutes_change_score": float(features["minutes"]["score"]),
        "lineup_change_score": float(features["lineups"]["score"]),
        "foul_trouble_score": float(features["foul_trouble"]["score"]),
        "injury_change_score": float(features["injuries"]["score"]),
        "coaching_adjustment_score": float(features["coaching"]["score"]),
        "team_a_probability_shift": float(features["model_update"]["team_a_probability_shift"]),
    }
    for cause in features.get("cause_classification", []):
        vector[f"cause_{cause['cause']}_score"] = float(cause["score"])
    return vector


if __name__ == "__main__":
    predicted = {
        "game_number": 1,
        "team_a": "NYK",
        "team_b": "SAS",
        "team_a_win_probability": 0.574,
        "expected_score_team_a": 113,
        "expected_score_team_b": 110,
        "projected_pace": 96.8,
    }
    actual = {
        "game_number": 1,
        "actual_scores": {"NYK": 104, "SAS": 109},
        "team_stats": [
            {"team": "NYK", "PTS": 104, "FGM": 38, "FGA": 88, "FG3M": 10, "FG3A": 34, "FTA": 24, "TOV": 11, "OREB": 13, "REB": 45, "AST": 24, "rim_attempts": 33, "corner_threes": 11, "wide_open_threes": 14, "offensive_rebound_chances": 30, "transition_opportunities": 13, "contested_fgm": 8, "contested_fga": 25},
            {"team": "SAS", "PTS": 109, "FGM": 41, "FGA": 86, "FG3M": 14, "FG3A": 29, "FTA": 14, "TOV": 10, "OREB": 8, "REB": 39, "AST": 19, "rim_attempts": 20, "corner_threes": 5, "wide_open_threes": 5, "offensive_rebound_chances": 18, "transition_opportunities": 8, "contested_fgm": 17, "contested_fga": 36},
        ],
    }
    report = build_adjustment_features(predicted, actual)
    print(report["primary_cause"])
    print(report["model_update"])
