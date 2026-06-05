"""Validation logic for all Finals manual input files."""

from __future__ import annotations

import pandas as pd


VALID_INJURY_STATUSES = {"Available", "Probable", "Questionable", "Doubtful", "Out"}
VALID_MATCHUP_TYPES = {
    "pick_and_roll", "isolation", "post_up", "rim_pressure", "transition",
    "three_point", "rebounding", "foul_pressure", "defensive_hunting",
}
VALID_COACHING_ADJUSTMENT_TYPES = {
    "lineup", "coverage", "pace", "rotation", "matchup",
    "shot_profile", "rebounding", "foul_strategy",
}
VALID_ROTATION_CONFIDENCES = {"high", "medium", "low"}


def validate_schedule(df: pd.DataFrame, team_a: str, team_b: str) -> list[str]:
    warnings: list[str] = []
    required = {"game_number", "date", "home_team", "away_team", "neutral_site"}
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"finals_schedule.csv missing columns: {missing}")
        return warnings

    game_numbers = sorted(df["game_number"].tolist())
    if game_numbers != list(range(1, 8)):
        warnings.append(f"finals_schedule.csv should have games 1-7, found: {game_numbers}")

    valid_teams = {team_a, team_b}
    bad_home = df[~df["home_team"].isin(valid_teams)]["home_team"].tolist()
    if bad_home:
        warnings.append(f"finals_schedule.csv invalid home_team values: {bad_home}")

    bad_away = df[~df["away_team"].isin(valid_teams)]["away_team"].tolist()
    if bad_away:
        warnings.append(f"finals_schedule.csv invalid away_team values: {bad_away}")

    if df["game_number"].duplicated().any():
        warnings.append("finals_schedule.csv has duplicate game_number values.")

    return warnings


def validate_rotations(df: pd.DataFrame, team_a: str, team_b: str) -> list[str]:
    warnings: list[str] = []
    required = {
        "team", "player", "role", "projected_minutes", "minutes_floor",
        "minutes_ceiling", "is_starter", "is_closer", "rotation_confidence", "notes",
    }
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"rotations.csv missing columns: {missing}")
        return warnings

    placeholder_rows = (df["player"] == "Player Name").sum()
    if placeholder_rows > 0:
        warnings.append(
            f"rotations.csv still has {placeholder_rows} placeholder 'Player Name' rows."
        )

    for team in [team_a, team_b]:
        team_df = df[df["team"] == team]
        if team_df.empty:
            warnings.append(f"rotations.csv has no players for {team}.")
            continue

        if len(team_df) < 7:
            warnings.append(
                f"rotations.csv: {team} has only {len(team_df)} players (expected >= 7)."
            )

        total_min = team_df["projected_minutes"].sum()
        if not (220 <= total_min <= 260):
            warnings.append(
                f"rotations.csv: {team} projected minutes sum to {total_min:.1f} (expected ~240)."
            )

        starters = team_df[team_df["is_starter"] == True]
        if len(starters) != 5:
            warnings.append(
                f"rotations.csv: {team} has {len(starters)} starters (expected 5)."
            )

        closers = team_df[team_df["is_closer"] == True]
        if len(closers) < 4:
            warnings.append(
                f"rotations.csv: {team} has only {len(closers)} closers marked (expected ~5)."
            )

    invalid_conf = df[
        ~df["rotation_confidence"].astype(str).str.lower().isin(VALID_ROTATION_CONFIDENCES)
    ]
    if not invalid_conf.empty:
        warnings.append(
            f"rotations.csv invalid rotation_confidence values: "
            f"{invalid_conf['rotation_confidence'].tolist()}"
        )

    numeric_df = df[["minutes_floor", "projected_minutes", "minutes_ceiling"]].apply(
        pd.to_numeric, errors="coerce"
    )
    bad_floor = numeric_df[numeric_df["minutes_floor"] > numeric_df["projected_minutes"]]
    if not bad_floor.empty:
        warnings.append(
            f"rotations.csv: {len(bad_floor)} rows have minutes_floor > projected_minutes."
        )

    bad_ceil = numeric_df[numeric_df["projected_minutes"] > numeric_df["minutes_ceiling"]]
    if not bad_ceil.empty:
        warnings.append(
            f"rotations.csv: {len(bad_ceil)} rows have projected_minutes > minutes_ceiling."
        )

    return warnings


def validate_injuries(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if df.empty:
        return warnings

    required = {"date", "team", "player", "status", "injury", "notes", "expected_minutes_adjustment"}
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"injuries.csv missing columns: {missing}")
        return warnings

    invalid_status = df[~df["status"].isin(VALID_INJURY_STATUSES)]
    if not invalid_status.empty:
        warnings.append(
            f"injuries.csv invalid statuses: {invalid_status['status'].tolist()}"
        )

    placeholder_rows = (df["player"] == "Player Name").sum()
    if placeholder_rows > 0:
        warnings.append(
            f"injuries.csv still has {placeholder_rows} placeholder 'Player Name' rows."
        )

    return warnings


def validate_player_matchups(df: pd.DataFrame, team_a: str, team_b: str) -> list[str]:
    warnings: list[str] = []
    if df.empty:
        return warnings

    required = {
        "offensive_team", "offensive_player", "defensive_team",
        "primary_defender", "matchup_type", "expected_impact", "notes",
    }
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"player_matchups.csv missing columns: {missing}")
        return warnings

    impact_col = pd.to_numeric(df["expected_impact"], errors="coerce")
    out_of_range = df[(impact_col < -2) | (impact_col > 2)]
    if not out_of_range.empty:
        warnings.append(
            f"player_matchups.csv: {len(out_of_range)} rows have expected_impact outside [-2, 2]."
        )

    valid_teams = {team_a, team_b}
    bad_off = df[~df["offensive_team"].isin(valid_teams)]["offensive_team"].tolist()
    if bad_off:
        warnings.append(f"player_matchups.csv invalid offensive_team: {bad_off}")

    invalid_types = df[~df["matchup_type"].isin(VALID_MATCHUP_TYPES)]
    if not invalid_types.empty:
        warnings.append(
            f"player_matchups.csv unrecognized matchup_type values: "
            f"{invalid_types['matchup_type'].tolist()}"
        )

    return warnings


def validate_coaching_notes(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if df.empty:
        return warnings

    required = {"game_number", "team", "adjustment_type", "description", "expected_impact"}
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"coaching_notes.csv missing columns: {missing}")
        return warnings

    game_col = pd.to_numeric(df["game_number"], errors="coerce")
    bad_games = df[(game_col < 1) | (game_col > 7)]
    if not bad_games.empty:
        warnings.append(
            f"coaching_notes.csv: {len(bad_games)} rows have game_number outside [1, 7]."
        )

    impact_col = pd.to_numeric(df["expected_impact"], errors="coerce")
    out_of_range = df[(impact_col < -2) | (impact_col > 2)]
    if not out_of_range.empty:
        warnings.append(
            f"coaching_notes.csv: {len(out_of_range)} rows have expected_impact outside [-2, 2]."
        )

    invalid_types = df[~df["adjustment_type"].isin(VALID_COACHING_ADJUSTMENT_TYPES)]
    if not invalid_types.empty:
        warnings.append(
            f"coaching_notes.csv unrecognized adjustment_type: "
            f"{invalid_types['adjustment_type'].tolist()}"
        )

    return warnings


def validate_all(
    schedule: pd.DataFrame,
    rotations: pd.DataFrame,
    injuries: pd.DataFrame,
    player_matchups: pd.DataFrame,
    coaching_notes: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> list[str]:
    """Run all validators and return combined warning list."""
    warnings: list[str] = []
    warnings.extend(validate_schedule(schedule, team_a, team_b))
    warnings.extend(validate_rotations(rotations, team_a, team_b))
    warnings.extend(validate_injuries(injuries))
    warnings.extend(validate_player_matchups(player_matchups, team_a, team_b))
    warnings.extend(validate_coaching_notes(coaching_notes))
    return warnings
