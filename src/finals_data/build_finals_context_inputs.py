"""Merge all data sources into the Finals context JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "finals_context"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "predictions"


def _json_safe(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {k: _json_safe(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _group_by_team(
    df: pd.DataFrame, teams: list[str],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {team: [] for team in teams}
    if df.empty or "team" not in df.columns:
        return result
    for team, group in df.groupby("team", sort=False):
        if team in result:
            result[str(team)] = _to_records(group)
    return result


def build_and_save(
    settings: dict[str, Any],
    schedule: pd.DataFrame,
    rotations: pd.DataFrame,
    injuries: pd.DataFrame,
    player_matchups: pd.DataFrame,
    coaching_notes: pd.DataFrame,
    regular_season_available: bool,
    playoffs_available: bool,
    validation_warnings: list[str],
) -> Path:
    """Build finals_context_inputs.json and write processed CSVs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    finals = settings["finals"]
    team_a = finals["team_a_abbr"]
    team_b = finals["team_b_abbr"]
    teams = [team_a, team_b]

    matchups = (
        _to_records(player_matchups) if not player_matchups.empty else []
    )
    notes = (
        _to_records(coaching_notes) if not coaching_notes.empty else []
    )

    context: dict[str, Any] = {
        "series": finals["series_name"],
        "team_a": team_a,
        "team_b": team_b,
        "schedule": _to_records(schedule) if not schedule.empty else [],
        "rotations": _group_by_team(rotations, teams),
        "injuries": _group_by_team(injuries, teams),
        "player_matchups": matchups,
        "coaching_notes": notes,
        "data_sources": {
            "regular_season": regular_season_available,
            "playoffs": playoffs_available,
            "manual_schedule": not schedule.empty,
            "manual_rotations": not rotations.empty,
            "manual_injuries": not injuries.empty,
        },
        "validation_warnings": validation_warnings,
    }

    context_path = PROCESSED_DIR / "finals_context_inputs.json"
    with context_path.open("w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, default=str)

    if not rotations.empty:
        rotations.to_csv(
            PROCESSED_DIR / "projected_rotations.csv", index=False,
        )
    if not injuries.empty:
        injuries.to_csv(
            PROCESSED_DIR / "injuries_processed.csv", index=False,
        )

    return context_path
