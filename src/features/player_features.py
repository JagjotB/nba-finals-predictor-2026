"""Player rotation and minutes projection features."""

from __future__ import annotations

from collections import defaultdict
from math import isnan
from typing import Any


MAX_REGULATION_MINUTES = 48.0
TEAM_REGULATION_MINUTES = 240.0

STATUS_DEFAULT_ADJUSTMENTS = {
    "available": 0.0,
    "probable": 0.0,
    "questionable": 0.0,
    "doubtful": -16.0,
    "out": 0.0,
}

STATUS_VOLATILITY = {
    "available": 0.0,
    "probable": 1.0,
    "questionable": 3.0,
    "doubtful": 6.0,
    "out": 0.0,
}

CONFIDENCE_VOLATILITY = {
    "high": 0.00,
    "medium": 2.00,
    "low": 4.00,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
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


def _clamp_minutes(value: Any) -> float:
    return round(max(0.0, min(MAX_REGULATION_MINUTES, _as_float(value))), 1)


def _normalized_status(status: Any) -> str:
    if status is None or status == "":
        return "available"
    return str(status).strip().lower()


def _normalized_confidence(confidence: Any) -> str:
    normalized = str(confidence or "medium").strip().lower()
    if normalized not in CONFIDENCE_VOLATILITY:
        return "medium"
    return normalized


def _injury_lookup(injuries: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for team, team_injuries in injuries.items():
        for injury in team_injuries:
            player = str(injury.get("player", "")).strip()
            if player:
                lookup[(str(team), player)] = injury
    return lookup


def _ensure_projection_order(floor: float, projected: float, ceiling: float) -> tuple[float, float, float]:
    projected = _clamp_minutes(projected)
    floor = _clamp_minutes(min(floor, projected))
    ceiling = _clamp_minutes(max(ceiling, projected))
    return floor, projected, ceiling


def project_player_minutes(
    rotation: dict[str, Any],
    injury: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one player's Finals rotation minutes from manual context.

    Manual projected minutes are treated as the anchor. Injury status and
    expected minute adjustment move that anchor, while low confidence widens
    the floor/ceiling range.
    """
    injury = injury or {}
    team = str(rotation.get("team", "")).strip()
    player = str(rotation.get("player", "")).strip()
    confidence = _normalized_confidence(rotation.get("rotation_confidence"))
    status = _normalized_status(injury.get("status"))
    raw_injury_adjustment = injury.get("expected_minutes_adjustment")
    injury_adjustment = _as_float(
        raw_injury_adjustment,
        STATUS_DEFAULT_ADJUSTMENTS.get(status, 0.0),
    )

    base_projected = _as_float(rotation.get("projected_minutes"))
    base_floor = _as_float(rotation.get("minutes_floor"), base_projected)
    base_ceiling = _as_float(rotation.get("minutes_ceiling"), base_projected)

    volatility = CONFIDENCE_VOLATILITY[confidence] + STATUS_VOLATILITY.get(status, 2.0)

    if status == "out":
        adjusted_projected = adjusted_floor = adjusted_ceiling = 0.0
    else:
        adjusted_projected = base_projected + injury_adjustment
        adjusted_floor = base_floor + injury_adjustment - volatility
        adjusted_ceiling = base_ceiling + injury_adjustment + volatility

    minutes_floor, projected_minutes, minutes_ceiling = _ensure_projection_order(
        adjusted_floor,
        adjusted_projected,
        adjusted_ceiling,
    )

    minutes_range = round(minutes_ceiling - minutes_floor, 1)

    return {
        "player_key": f"{team}:{player}:{rotation.get('role', '')}".strip(":"),
        "team": team,
        "player": player,
        "role": rotation.get("role"),
        "projected_minutes": projected_minutes,
        "minutes_floor": minutes_floor,
        "minutes_ceiling": minutes_ceiling,
        "minutes_range": minutes_range,
        "is_starter": _as_bool(rotation.get("is_starter")),
        "is_closer": _as_bool(rotation.get("is_closer")),
        "injury_status": injury.get("status", "Available"),
        "injury": injury.get("injury"),
        "injury_adjustment": injury_adjustment,
        "rotation_confidence": confidence,
        "minutes_confidence": confidence,
        "uncertain_minutes": confidence != "high" or minutes_range >= 10 or status not in {"available", "probable"},
        "notes": rotation.get("notes"),
    }


def project_team_minutes(
    team: str,
    rotations: list[dict[str, Any]],
    injuries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Project all player minutes for one team."""
    injury_by_player = {
        str(injury.get("player", "")).strip(): injury
        for injury in injuries or []
        if str(injury.get("player", "")).strip()
    }

    projections = []
    for rotation in rotations:
        enriched_rotation = {**rotation, "team": team}
        player = str(enriched_rotation.get("player", "")).strip()
        projections.append(project_player_minutes(enriched_rotation, injury_by_player.get(player)))

    return _normalize_team_minutes(projections)


def _normalize_team_minutes(
    projections: list[dict[str, Any]],
    target_minutes: float = TEAM_REGULATION_MINUTES,
) -> list[dict[str, Any]]:
    """Reconcile independent player estimates to one regulation game.

    Manual rotation rows remain proportional anchors, but the downstream
    player, lineup, and team models must all operate on a coherent 240-minute
    allocation.
    """
    total = sum(row["projected_minutes"] for row in projections)
    if total <= 0:
        return sorted(projections, key=lambda row: row["projected_minutes"], reverse=True)

    factor = target_minutes / total
    normalized = []
    for projection in projections:
        row = dict(projection)
        projected = _clamp_minutes(row["projected_minutes"] * factor)
        floor = _clamp_minutes(row["minutes_floor"] * factor)
        ceiling = _clamp_minutes(row["minutes_ceiling"] * factor)
        floor, projected, ceiling = _ensure_projection_order(floor, projected, ceiling)
        row.update(
            {
                "projected_minutes": projected,
                "minutes_floor": floor,
                "minutes_ceiling": ceiling,
                "minutes_range": round(ceiling - floor, 1),
                "minutes_normalization_factor": round(factor, 4),
            }
        )
        normalized.append(row)

    residual = round(target_minutes - sum(row["projected_minutes"] for row in normalized), 1)
    for row in sorted(normalized, key=lambda item: item["projected_minutes"], reverse=True):
        if abs(residual) < 0.05:
            break
        capacity = (
            MAX_REGULATION_MINUTES - row["projected_minutes"]
            if residual > 0
            else row["projected_minutes"]
        )
        adjustment = min(abs(residual), capacity)
        if adjustment <= 0:
            continue
        row["projected_minutes"] = round(
            row["projected_minutes"] + adjustment * (1 if residual > 0 else -1),
            1,
        )
        row["minutes_floor"] = min(row["minutes_floor"], row["projected_minutes"])
        row["minutes_ceiling"] = max(row["minutes_ceiling"], row["projected_minutes"])
        row["minutes_range"] = round(row["minutes_ceiling"] - row["minutes_floor"], 1)
        residual = round(
            target_minutes - sum(item["projected_minutes"] for item in normalized),
            1,
        )

    return sorted(normalized, key=lambda row: row["projected_minutes"], reverse=True)


def project_rotation_minutes(finals_context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Build player minute projections for every Finals team."""
    rotations = finals_context.get("rotations", {})
    injuries = finals_context.get("injuries", {})
    teams = [finals_context.get("team_a"), finals_context.get("team_b")]

    projections: dict[str, list[dict[str, Any]]] = {}
    for team in teams:
        if not team:
            continue
        projections[str(team)] = project_team_minutes(
            str(team),
            rotations.get(str(team), []),
            injuries.get(str(team), []),
        )
    return projections


def flatten_player_projections(
    player_projections: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten team-grouped player projections into one sorted list."""
    rows = [
        projection
        for team_projections in player_projections.values()
        for projection in team_projections
    ]
    return sorted(rows, key=lambda row: (row["team"], -row["projected_minutes"], row["player"]))


def summarize_rotation_strength(
    player_projections: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Summarize the minutes distribution each team is expected to rely on."""
    summaries: dict[str, dict[str, Any]] = {}
    for team, projections in player_projections.items():
        top_eight = sorted(projections, key=lambda row: row["projected_minutes"], reverse=True)[:8]
        summaries[team] = {
            "projected_rotation_minutes": round(
                sum(row["projected_minutes"] for row in projections),
                1,
            ),
            "starter_minutes": round(
                sum(row["projected_minutes"] for row in projections if row["is_starter"]),
                1,
            ),
            "closer_minutes": round(
                sum(row["projected_minutes"] for row in projections if row["is_closer"]),
                1,
            ),
            "top_eight_minutes": round(sum(row["projected_minutes"] for row in top_eight), 1),
            "uncertain_players": [
                row["player"] for row in projections if row["uncertain_minutes"]
            ],
        }
    return summaries


def group_players_by_role(
    player_projections: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group projected players by team and rotation role."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for team, projections in player_projections.items():
        role_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for projection in projections:
            role_groups[str(projection.get("role") or "Unknown")].append(projection)
        grouped[team] = dict(role_groups)
    return grouped
