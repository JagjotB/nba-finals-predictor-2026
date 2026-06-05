"""Build the canonical leakage-safe historical playoff game table.

Every feature is either an end-of-regular-season prior or is calculated from
playoff games completed before the row's game date. Historical sources that
are not available are represented by availability flags, never fabricated
values.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

from src.data.fetch_current_stats import (
    fetch_historical_playoff_logs,
    fetch_historical_team_ratings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "historical" / "canonical_pregame_games.csv"
)
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_PATH.with_suffix(".metadata.json")

METRIC_PRIOR_POSSESSIONS = {
    "net_rating": 900.0,
    "efg_pct": 1800.0,
    "tov_pct": 700.0,
    "oreb_pct": 700.0,
    "fta_rate": 1200.0,
    "pace": 400.0,
}

# Arena-city coordinates are sufficient for a transparent travel-load feature.
TEAM_COORDINATES = {
    "ATL": (33.7573, -84.3963), "BOS": (42.3662, -71.0621),
    "BKN": (40.6826, -73.9754), "CHA": (35.2251, -80.8392),
    "CHI": (41.8807, -87.6742), "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8103), "DEN": (39.7487, -105.0077),
    "DET": (42.3410, -83.0550), "GSW": (37.7680, -122.3877),
    "HOU": (29.7508, -95.3621), "IND": (39.7640, -86.1555),
    "LAC": (34.0430, -118.2673), "LAL": (34.0430, -118.2673),
    "MEM": (35.1382, -90.0505), "MIA": (25.7814, -80.1870),
    "MIL": (43.0451, -87.9172), "MIN": (44.9795, -93.2760),
    "NOP": (29.9490, -90.0821), "NYK": (40.7505, -73.9934),
    "OKC": (35.4634, -97.5151), "ORL": (28.5392, -81.3839),
    "PHI": (39.9012, -75.1720), "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6668), "SAC": (38.5802, -121.4997),
    "SAS": (29.4270, -98.4375), "TOR": (43.6435, -79.3791),
    "UTA": (40.7683, -111.9011), "WAS": (38.8981, -77.0209),
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _game_date(row: dict[str, Any]) -> date:
    return date.fromisoformat(str(row.get("GAME_DATE", ""))[:10])


def _team_abbr(row: dict[str, Any]) -> str:
    return str(row.get("TEAM_ABBREVIATION") or "").strip()


def _possessions(row: dict[str, Any]) -> float:
    return max(
        _as_float(row.get("FGA"))
        - _as_float(row.get("OREB"))
        + _as_float(row.get("TOV"))
        + 0.44 * _as_float(row.get("FTA")),
        1.0,
    )


def _haversine_miles(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    lat1, lon1 = map(radians, origin)
    lat2, lon2 = map(radians, destination)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 3958.8 * 2 * asin(sqrt(a))


def _travel_miles(previous_venue: str | None, current_venue: str) -> tuple[float, int]:
    if not previous_venue:
        return 0.0, 0
    origin = TEAM_COORDINATES.get(previous_venue)
    destination = TEAM_COORDINATES.get(current_venue)
    if not origin or not destination:
        return 0.0, 0
    return round(_haversine_miles(origin, destination), 1), 1


def _regular_metric(row: dict[str, Any], metric: str) -> float:
    aliases = {
        "net_rating": ("NET_RATING",),
        "efg_pct": ("EFG_PCT",),
        "tov_pct": ("TM_TOV_PCT",),
        "oreb_pct": ("OREB_PCT",),
        "fta_rate": ("FTA_RATE",),
        "pace": ("PACE",),
    }
    defaults = {
        "net_rating": 0.0,
        "efg_pct": 0.54,
        "tov_pct": 0.14,
        "oreb_pct": 0.27,
        "fta_rate": 0.25,
        "pace": 98.0,
    }
    for key in aliases[metric]:
        if row.get(key) not in (None, ""):
            return _as_float(row[key], defaults[metric])
    return defaults[metric]


def _playoff_snapshot(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {
            "games": 0.0, "possessions": 0.0, "net_rating": 0.0,
            "efg_pct": 0.54, "tov_pct": 0.14, "oreb_pct": 0.27,
            "fta_rate": 0.25, "pace": 98.0, "recent_net_rating": 0.0,
        }
    possessions = sum(row["possessions"] for row in history)
    points = sum(row["points"] for row in history)
    opponent_points = sum(row["opponent_points"] for row in history)
    fga = sum(row["fga"] for row in history)
    fgm = sum(row["fgm"] for row in history)
    fg3m = sum(row["fg3m"] for row in history)
    turnovers = sum(row["turnovers"] for row in history)
    oreb = sum(row["oreb"] for row in history)
    opponent_dreb = sum(row["opponent_dreb"] for row in history)
    fta = sum(row["fta"] for row in history)
    recent = history[-5:]
    recent_possessions = sum(row["possessions"] for row in recent)
    recent_margin = sum(row["points"] - row["opponent_points"] for row in recent)
    return {
        "games": float(len(history)),
        "possessions": possessions,
        "net_rating": 100.0 * (points - opponent_points) / max(possessions, 1.0),
        "efg_pct": (fgm + 0.5 * fg3m) / max(fga, 1.0),
        "tov_pct": turnovers / max(possessions, 1.0),
        "oreb_pct": oreb / max(oreb + opponent_dreb, 1.0),
        "fta_rate": fta / max(fga, 1.0),
        "pace": possessions / len(history),
        "recent_net_rating": 100.0 * recent_margin / max(recent_possessions, 1.0),
    }


def blend_regular_and_playoff(
    regular_value: float,
    playoff_value: float,
    playoff_possessions: float,
    metric: str,
) -> tuple[float, float]:
    """Shrink a playoff rate toward its regular-season prior."""
    prior = METRIC_PRIOR_POSSESSIONS[metric]
    weight = playoff_possessions / (playoff_possessions + prior)
    return (
        weight * playoff_value + (1.0 - weight) * regular_value,
        weight,
    )


def _pair_games(game_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    games: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in game_logs:
        key = (str(row.get("SEASON_YEAR", "")), str(row.get("GAME_ID", "")))
        if all(key):
            games.setdefault(key, []).append(row)
    paired = []
    for (season, game_id), teams in games.items():
        if len(teams) != 2:
            continue
        home = next((row for row in teams if " vs. " in str(row.get("MATCHUP", ""))), None)
        away = next((row for row in teams if " @ " in str(row.get("MATCHUP", ""))), None)
        if home and away:
            paired.append({
                "season": season, "game_id": game_id, "date": _game_date(home),
                "home": home, "away": away,
            })
    return sorted(paired, key=lambda game: (game["season"], game["date"], game["game_id"]))


def _history_record(team_row: dict[str, Any], opponent_row: dict[str, Any]) -> dict[str, float]:
    return {
        "possessions": _possessions(team_row),
        "points": _as_float(team_row.get("PTS")),
        "opponent_points": _as_float(opponent_row.get("PTS")),
        "fga": _as_float(team_row.get("FGA")),
        "fgm": _as_float(team_row.get("FGM")),
        "fg3m": _as_float(team_row.get("FG3M")),
        "turnovers": _as_float(team_row.get("TOV")),
        "oreb": _as_float(team_row.get("OREB")),
        "opponent_dreb": _as_float(opponent_row.get("DREB")),
        "fta": _as_float(team_row.get("FTA")),
    }


def build_canonical_pregame_rows(
    game_logs: list[dict[str, Any]],
    team_ratings: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create two symmetric, strictly pregame rows for every paired game."""
    histories: dict[tuple[str, str], list[dict[str, float]]] = {}
    previous_dates: dict[tuple[str, str], date] = {}
    previous_venues: dict[tuple[str, str], str] = {}
    rows: list[dict[str, Any]] = []

    for game in _pair_games(game_logs):
        season = game["season"]
        home, away = game["home"], game["away"]
        home_id, away_id = str(int(home["TEAM_ID"])), str(int(away["TEAM_ID"]))
        home_abbr, away_abbr = _team_abbr(home), _team_abbr(away)
        venue = home_abbr
        ratings = team_ratings.get(season, {})
        regular = {home_id: ratings.get(home_id, {}), away_id: ratings.get(away_id, {})}
        snapshots = {
            home_id: _playoff_snapshot(histories.get((season, home_id), [])),
            away_id: _playoff_snapshot(histories.get((season, away_id), [])),
        }
        rest = {}
        travel = {}
        for team_id in (home_id, away_id):
            previous_date = previous_dates.get((season, team_id))
            rest[team_id] = (
                max((game["date"] - previous_date).days, 1) if previous_date else 3
            )
            travel[team_id] = _travel_miles(
                previous_venues.get((season, team_id)),
                venue,
            )

        def make_row(
            team_row: dict[str, Any],
            opponent_row: dict[str, Any],
            team_id: str,
            opponent_id: str,
            home_context: float,
            perspective: str,
        ) -> dict[str, Any]:
            team_regular, opponent_regular = regular[team_id], regular[opponent_id]
            team_playoff, opponent_playoff = snapshots[team_id], snapshots[opponent_id]
            row: dict[str, Any] = {
                "season": season,
                "game_id": game["game_id"],
                "series_id": f"{season}:{min(home_id, away_id)}:{max(home_id, away_id)}",
                "game_date": game["date"].isoformat(),
                "feature_cutoff_date": (
                    max(
                        previous_dates.get((season, team_id), date.min),
                        previous_dates.get((season, opponent_id), date.min),
                    ).isoformat()
                    if previous_dates.get((season, team_id))
                    or previous_dates.get((season, opponent_id))
                    else ""
                ),
                "team_a_id": team_id,
                "team_b_id": opponent_id,
                "team_a": _team_abbr(team_row),
                "team_b": _team_abbr(opponent_row),
                "actual_home_team_id": home_id,
                "actual_away_team_id": away_id,
                "perspective": perspective,
                "home_court": home_context,
                "rest_diff": float(rest[team_id] - rest[opponent_id]),
                "travel_miles_diff": travel[team_id][0] - travel[opponent_id][0],
                "travel_data_available": min(travel[team_id][1], travel[opponent_id][1]),
                "playoff_games_team_a": int(team_playoff["games"]),
                "playoff_games_team_b": int(opponent_playoff["games"]),
                "playoff_possessions_team_a": round(team_playoff["possessions"], 2),
                "playoff_possessions_team_b": round(opponent_playoff["possessions"], 2),
                "recent_net_rating_diff": (
                    team_playoff["recent_net_rating"] - opponent_playoff["recent_net_rating"]
                ),
                "injury_data_available": 0,
                "rotation_data_available": 0,
                "lineup_data_available": 0,
                "injury_strength_diff": None,
                "rotation_strength_diff": None,
                "lineup_strength_diff": None,
                "team_score": _as_float(team_row.get("PTS")),
                "opponent_score": _as_float(opponent_row.get("PTS")),
                "won": int(str(team_row.get("WL", "")).upper() == "W"),
            }
            for metric in METRIC_PRIOR_POSSESSIONS:
                regular_a = _regular_metric(team_regular, metric)
                regular_b = _regular_metric(opponent_regular, metric)
                blended_a, weight_a = blend_regular_and_playoff(
                    regular_a, team_playoff[metric], team_playoff["possessions"], metric,
                )
                blended_b, weight_b = blend_regular_and_playoff(
                    regular_b, opponent_playoff[metric], opponent_playoff["possessions"], metric,
                )
                row[f"regular_{metric}_diff"] = regular_a - regular_b
                row[f"blended_{metric}_diff"] = blended_a - blended_b
                row[f"{metric}_diff"] = regular_a - regular_b
                row[f"{metric}_playoff_weight_a"] = round(weight_a, 4)
                row[f"{metric}_playoff_weight_b"] = round(weight_b, 4)
            return row

        rows.append(make_row(home, away, home_id, away_id, 1.0, "home"))
        rows.append(make_row(away, home, away_id, home_id, -1.0, "away"))

        histories.setdefault((season, home_id), []).append(_history_record(home, away))
        histories.setdefault((season, away_id), []).append(_history_record(away, home))
        for team_id in (home_id, away_id):
            previous_dates[(season, team_id)] = game["date"]
            previous_venues[(season, team_id)] = venue
    return rows


def validate_no_future_leakage(rows: list[dict[str, Any]]) -> list[str]:
    """Return leakage/schema errors for canonical rows."""
    errors = []
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        identity = (str(row.get("season")), str(row.get("game_id")), str(row.get("perspective")))
        if identity in seen:
            errors.append(f"row {index}: duplicate identity {identity}")
        seen.add(identity)
        game_date = str(row.get("game_date") or "")
        cutoff = str(row.get("feature_cutoff_date") or "")
        if cutoff and cutoff >= game_date:
            errors.append(f"row {index}: feature cutoff {cutoff} is not before {game_date}")
        if row.get("perspective") not in {"home", "away"}:
            errors.append(f"row {index}: invalid perspective")
        if float(row.get("home_court", 0.0)) not in {-1.0, 1.0}:
            errors.append(f"row {index}: invalid home context")
    return errors


def _dataset_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_canonical_dataset(
    rows: list[dict[str, Any]],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    errors = validate_no_future_leakage(rows)
    if errors:
        raise ValueError("Canonical dataset failed validation:\n" + "\n".join(errors[:20]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": "canonical-pregame-v1",
        "rows": len(rows),
        "games": len(rows) // 2,
        "seasons": sorted({str(row["season"]) for row in rows}),
        "sha256": _dataset_hash(rows),
        "source_policy": "regular-season priors plus playoff games strictly before tipoff",
        "unavailable_historical_sources": ["injuries", "expected_rotations", "lineups"],
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_and_save_historical_dataset(force_refresh: bool = False) -> dict[str, Any]:
    rows = build_canonical_pregame_rows(
        fetch_historical_playoff_logs(force_refresh=force_refresh),
        fetch_historical_team_ratings(force_refresh=force_refresh),
    )
    return save_canonical_dataset(rows)


if __name__ == "__main__":
    result = build_and_save_historical_dataset()
    print(json.dumps(result, indent=2))
