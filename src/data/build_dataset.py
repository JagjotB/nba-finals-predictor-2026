"""Build the Finals context object from config and manual data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.load_manual_data import load_all_manual_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

QUESTIONABLE_STATUSES = {"questionable", "doubtful", "out"}


def load_settings(settings_path: str | Path = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    """Load project settings from YAML."""
    path = Path(settings_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing settings file: {path}")

    with path.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file) or {}

    if "finals" not in settings:
        raise ValueError("settings.yaml must include a `finals` section.")
    return settings


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _json_safe(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _group_records(frame: pd.DataFrame, group_column: str) -> dict[str, list[dict[str, Any]]]:
    if frame.empty or group_column not in frame.columns:
        return {}

    return {
        str(group): _records(group_frame.drop(columns=[group_column]))
        for group, group_frame in frame.groupby(group_column, sort=False)
    }


def _schedule_records(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    schedule = schedule.copy()
    schedule["site_team"] = schedule.apply(
        lambda row: None if bool(row["neutral_site"]) else row["home_team"],
        axis=1,
    )
    return _records(schedule)


def _players_from_rotations(rotations: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    players: dict[str, dict[str, dict[str, Any]]] = {}
    if rotations.empty:
        return players

    for row in _records(rotations):
        team = str(row.pop("team"))
        player = str(row.pop("player"))
        players.setdefault(team, {})
        players[team].setdefault("players", {})
        players[team]["players"].setdefault(player, {"player": player})
        players[team]["players"][player].update(row)

    return players


def _attach_injury_context(
    active_players: dict[str, dict[str, dict[str, Any]]],
    injuries: pd.DataFrame,
) -> None:
    if injuries.empty:
        return

    for row in _records(injuries):
        team = str(row["team"])
        player = str(row["player"])
        player_record = active_players.setdefault(team, {}).setdefault("players", {}).setdefault(
            player,
            {"player": player},
        )
        player_record["injury_status"] = row.get("status")
        player_record["injury"] = row.get("injury")
        player_record["expected_minutes_adjustment"] = row.get("expected_minutes_adjustment")


def _attach_matchup_context(
    active_players: dict[str, dict[str, dict[str, Any]]],
    player_matchups: pd.DataFrame,
) -> None:
    if player_matchups.empty:
        return

    for row in _records(player_matchups):
        offensive_team = str(row["offensive_team"])
        offensive_player = str(row["offensive_player"])
        defensive_team = str(row["defensive_team"])
        primary_defender = str(row["primary_defender"])

        offensive_record = active_players.setdefault(offensive_team, {}).setdefault(
            "players",
            {},
        ).setdefault(offensive_player, {"player": offensive_player})
        offensive_record.setdefault("matchup_roles", []).append(
            {
                "side": "offense",
                "opponent": defensive_team,
                "primary_defender": primary_defender,
                "matchup_type": row.get("matchup_type"),
                "expected_impact": row.get("expected_impact"),
            }
        )

        defensive_record = active_players.setdefault(defensive_team, {}).setdefault(
            "players",
            {},
        ).setdefault(primary_defender, {"player": primary_defender})
        defensive_record.setdefault("matchup_roles", []).append(
            {
                "side": "defense",
                "opponent": offensive_team,
                "offensive_player": offensive_player,
                "matchup_type": row.get("matchup_type"),
                "expected_impact": row.get("expected_impact"),
            }
        )


def _active_players(
    rotations: pd.DataFrame,
    injuries: pd.DataFrame,
    player_matchups: pd.DataFrame,
    teams: list[str],
) -> dict[str, list[dict[str, Any]]]:
    active_players = _players_from_rotations(rotations)
    _attach_injury_context(active_players, injuries)
    _attach_matchup_context(active_players, player_matchups)

    for team in teams:
        active_players.setdefault(team, {}).setdefault("players", {})

    return {
        team: list(team_record.get("players", {}).values())
        for team, team_record in active_players.items()
    }


def _expected_players(
    rotations: pd.DataFrame,
    flag_column: str,
) -> dict[str, list[str]]:
    if rotations.empty or flag_column not in rotations.columns:
        return {}

    flagged = rotations[rotations[flag_column] == True]
    return {
        str(team): [str(player) for player in team_frame["player"].tolist()]
        for team, team_frame in flagged.groupby("team", sort=False)
    }


def _pre_series_rest(
    series_context: pd.DataFrame,
    schedule_records: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute each team's rest days before Game 1 from series_context.csv."""
    if series_context.empty:
        return {}
    game_1 = next(
        (g for g in schedule_records if int(g.get("game_number", 0)) == 1), None
    )
    if not game_1 or not game_1.get("date"):
        return {}
    try:
        g1_date = pd.Timestamp(game_1["date"]).date()
    except (ValueError, TypeError):
        return {}
    rest: dict[str, float] = {}
    for row in _records(series_context):
        team = str(row.get("team") or "")
        last_game = row.get("last_game_before_finals")
        if not team or last_game is None:
            continue
        try:
            last_date = pd.Timestamp(str(last_game)).date()
            rest[team] = float(max((g1_date - last_date).days, 0))
        except (ValueError, TypeError):
            pass
    return rest


def _uncertain_minutes(
    rotations: pd.DataFrame,
    injuries: pd.DataFrame,
    teams: list[str],
) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {team: [] for team in teams}

    if not rotations.empty:
        for row in _records(rotations):
            minute_range = (row.get("minutes_ceiling") or 0) - (row.get("minutes_floor") or 0)
            confidence = str(row.get("rotation_confidence") or "").lower()
            if confidence != "high" or minute_range >= 10:
                candidates.setdefault(str(row.get("team")), []).append(
                    {
                        "player": row.get("player"),
                        "reason": "rotation_range",
                        "rotation_confidence": row.get("rotation_confidence"),
                        "minutes_floor": row.get("minutes_floor"),
                        "projected_minutes": row.get("projected_minutes"),
                        "minutes_ceiling": row.get("minutes_ceiling"),
                    }
                )

    if not injuries.empty:
        for row in _records(injuries):
            status = str(row.get("status") or "").lower()
            minutes_adjustment = row.get("expected_minutes_adjustment") or 0
            if status in QUESTIONABLE_STATUSES or minutes_adjustment != 0:
                candidates.setdefault(str(row.get("team")), []).append(
                    {
                        "player": row.get("player"),
                        "reason": "injury_status",
                        "status": row.get("status"),
                        "injury": row.get("injury"),
                        "expected_minutes_adjustment": row.get("expected_minutes_adjustment"),
                    }
                )

    return {team: records for team, records in candidates.items() if records}


def _ensure_team_keys(
    grouped_records: dict[str, list[Any]],
    teams: list[str],
) -> dict[str, list[Any]]:
    for team in teams:
        grouped_records.setdefault(team, [])
    return grouped_records


def _load_team_stats_csv(path: Path, teams: list[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {team: {} for team in teams}
    frame = pd.read_csv(path)
    team_column = next(
        (
            column for column in ("TEAM_ABBREVIATION", "TEAM_ABBREVIATION_x", "team")
            if column in frame.columns
        ),
        None,
    )
    if not team_column:
        return {team: {} for team in teams}
    records = {}
    for team in teams:
        rows = frame[frame[team_column] == team]
        records[team] = _records(rows.head(1))[0] if not rows.empty else {}
    return records


def _load_player_stats_csv(path: Path, teams: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {team: [] for team in teams}
    frame = pd.read_csv(path)
    team_column = next(
        (
            column for column in ("TEAM_ABBREVIATION", "TEAM_ABBREVIATION_x", "team")
            if column in frame.columns
        ),
        None,
    )
    if not team_column:
        return {team: [] for team in teams}
    return {
        team: _records(frame[frame[team_column] == team])
        for team in teams
    }


def _has_team_data(data: dict[str, Any]) -> bool:
    return any(bool(value) for value in data.values())


def _load_live_stats(
    teams: list[str],
    season: str,
) -> dict[str, Any]:
    """Load current playoff and regular-season inputs.

    Playoff data is the current-form signal. Regular-season data is retained
    as the larger-sample prior used by the player and team projection layers.
    """
    try:
        from src.data.fetch_current_stats import (
            fetch_team_stats,
            fetch_player_stats,
            fetch_all_lineup_stats,
        )
        playoff_team_stats = fetch_team_stats(
            season=season,
            season_type="Playoffs",
        )
        playoff_player_stats = fetch_player_stats(
            season=season,
            season_type="Playoffs",
        )
        regular_season_team_stats = fetch_team_stats(
            season=season,
            season_type="Regular Season",
        )
        regular_season_player_stats = fetch_player_stats(
            season=season,
            season_type="Regular Season",
        )
        lineup_stats = fetch_all_lineup_stats(
            season=season,
            season_type="Playoffs",
        )
        if not _has_team_data(playoff_team_stats):
            playoff_team_stats = _load_team_stats_csv(
                PROJECT_ROOT / "data" / "processed" / "current_playoffs" / "team_stats.csv",
                teams,
            )
        if not _has_team_data(playoff_player_stats):
            playoff_player_stats = _load_player_stats_csv(
                PROJECT_ROOT / "data" / "processed" / "current_playoffs" / "player_stats.csv",
                teams,
            )
        if not _has_team_data(regular_season_team_stats):
            regular_season_team_stats = _load_team_stats_csv(
                PROJECT_ROOT / "data" / "processed" / "current_regular_season" / "team_stats.csv",
                teams,
            )
        if not _has_team_data(regular_season_player_stats):
            regular_season_player_stats = _load_player_stats_csv(
                PROJECT_ROOT / "data" / "processed" / "current_regular_season" / "player_stats.csv",
                teams,
            )
        return {
            "team_stats": playoff_team_stats,
            "player_stats": playoff_player_stats,
            "playoff_team_stats": playoff_team_stats,
            "playoff_player_stats": playoff_player_stats,
            "regular_season_team_stats": regular_season_team_stats,
            "regular_season_player_stats": regular_season_player_stats,
            "lineup_stats": lineup_stats,
        }
    except Exception:
        empty_teams = {t: {} for t in teams}
        empty_players = {t: [] for t in teams}
        return {
            "team_stats": empty_teams,
            "player_stats": empty_players,
            "playoff_team_stats": empty_teams,
            "playoff_player_stats": empty_players,
            "regular_season_team_stats": empty_teams,
            "regular_season_player_stats": empty_players,
            "lineup_stats": {t: [] for t in teams},
        }


def build_finals_context(
    settings_path: str | Path = DEFAULT_SETTINGS_PATH,
    manual_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create the Finals context object used by downstream engines and the app."""
    settings = load_settings(settings_path)
    finals = settings["finals"]
    manual_data = load_all_manual_data(manual_dir)

    schedule = manual_data["finals_schedule"]
    rotations = manual_data["rotations"]
    injuries = manual_data["injuries"]
    player_matchups = manual_data["player_matchups"]
    series_context = manual_data.get("series_context", pd.DataFrame())
    teams = [finals["team_a_abbr"], finals["team_b_abbr"]]

    schedule_records = _schedule_records(schedule)
    live = _load_live_stats(teams, str(settings["project"]["season"]))

    return {
        "series": finals["series_name"],
        "team_a": finals["team_a_abbr"],
        "team_b": finals["team_b_abbr"],
        "schedule": schedule_records,
        "active_players": _active_players(rotations, injuries, player_matchups, teams),
        "rotations": _ensure_team_keys(_group_records(rotations, "team"), teams),
        "injuries": _ensure_team_keys(_group_records(injuries, "team"), teams),
        "expected_starters": _ensure_team_keys(
            _expected_players(rotations, "is_starter"),
            teams,
        ),
        "expected_closers": _ensure_team_keys(
            _expected_players(rotations, "is_closer"),
            teams,
        ),
        "uncertain_minutes": _uncertain_minutes(rotations, injuries, teams),
        "pre_series_rest": _pre_series_rest(series_context, schedule_records),
        "team_stats": live["team_stats"],
        "player_stats": live["player_stats"],
        "playoff_team_stats": live["playoff_team_stats"],
        "playoff_player_stats": live["playoff_player_stats"],
        "regular_season_team_stats": live["regular_season_team_stats"],
        "regular_season_player_stats": live["regular_season_player_stats"],
        "lineup_stats": live["lineup_stats"],
    }


if __name__ == "__main__":
    print(json.dumps(build_finals_context(), indent=2))
