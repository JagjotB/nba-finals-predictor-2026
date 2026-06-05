"""Clutch and closing-lineup feature engineering."""

from __future__ import annotations

from math import isnan
from typing import Any

from src.features.archetype_features import classify_players
from src.features.lineup_features import build_lineup_features
from src.models.train_player_model import project_finals_players


CLUTCH_FEATURES = [
    "late_game_shot_creation",
    "isolation_scoring",
    "free_throw_reliability",
    "turnover_risk",
    "rim_protection",
    "switchability",
    "rebounding",
    "star_creation",
    "foul_risk",
]

TEAM_KEYS = ["team", "TEAM", "TEAM_ABBREVIATION", "TEAM_NAME"]
PLAYER_KEYS = ["player", "PLAYER_NAME", "name", "Name"]

PLAYER_ALIASES = {
    "minutes": ["minutes", "MIN", "MP", "projected_minutes"],
    "points": ["points", "PTS"],
    "assists": ["assists", "AST"],
    "rebounds": ["rebounds", "REB", "TRB"],
    "turnovers": ["turnovers", "TOV", "TO"],
    "steals": ["steals", "STL"],
    "blocks": ["blocks", "BLK"],
    "threes_attempted": ["threes_attempted", "FG3A", "3PA"],
    "free_throw_attempts": ["free_throw_attempts", "FTA"],
    "free_throw_percentage": ["free_throw_percentage", "FT_PCT", "FT%"],
    "usage_rate": ["usage_rate", "USG_PCT", "USG%"],
    "true_shooting_proxy": ["true_shooting_proxy", "TS_PCT", "TS%"],
    "foul_risk": ["foul_risk"],
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


def _team(row: dict[str, Any]) -> str:
    return str(_first_value(row, TEAM_KEYS) or "").strip()


def _player(row: dict[str, Any]) -> str:
    return str(_first_value(row, PLAYER_KEYS) or row.get("player") or "").strip()


def _display_player_names(players: list[dict[str, Any]]) -> list[str]:
    names = [_player(player) for player in players]
    duplicate_names = {name for name in names if names.count(name) > 1}
    display_names = []
    for player, name in zip(players, names):
        if name in duplicate_names and player.get("role"):
            display_names.append(f"{name} ({player['role']})")
        else:
            display_names.append(name)
    return display_names


def _metric(row: dict[str, Any], metric: str, default: float = 0.0) -> float:
    return _as_float(_first_value(row, PLAYER_ALIASES[metric]), default)


def _metric_percentage(row: dict[str, Any], metric: str, default: float = 0.0) -> float:
    return _percentage(_first_value(row, PLAYER_ALIASES[metric]), default)


def _player_minutes(player: dict[str, Any]) -> float:
    return _metric(player, "minutes", 0.0)


def _lineup_by_type(lineups: list[dict[str, Any]], lineup_type: str) -> dict[str, Any] | None:
    for lineup in lineups:
        if lineup.get("lineup_type") == lineup_type:
            return lineup
    return None


def _player_index(player_rows: Any) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _iter_records(player_rows):
        team = _team(row)
        player = _player(row)
        if not team or not player:
            continue
        index.setdefault((team, player), []).append(row)
    for key in index:
        index[key] = sorted(index[key], key=_player_minutes, reverse=True)
    return index


def _row_identity(row: dict[str, Any]) -> str:
    return str(row.get("player_key") or f"{_team(row)}:{_player(row)}:{row.get('role', '')}")


def _archetype_index(player_archetypes: Any) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_records(player_archetypes):
        team = _team(row)
        player = _player(row)
        if team and player:
            index[(team, player)] = row
    return index


def _closing_players(
    team: str,
    closing_lineup: dict[str, Any] | None,
    player_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not closing_lineup:
        return sorted(player_rows, key=_player_minutes, reverse=True)[:5]

    wanted_names = [str(name) for name in closing_lineup.get("players", [])]
    index = _player_index(player_rows)
    selected = []
    used_ids = set()
    for name in wanted_names:
        candidates = index.get((team, name), [])
        for candidate in candidates:
            identity = _row_identity(candidate)
            if identity in used_ids:
                continue
            selected.append(candidate)
            used_ids.add(identity)
            break

    for row in sorted(player_rows, key=_player_minutes, reverse=True):
        identity = _row_identity(row)
        if identity in used_ids:
            continue
        selected.append(row)
        used_ids.add(identity)
        if len(selected) >= 5:
            break
    return selected[:5]


def _weighted_average(
    rows: list[dict[str, Any]],
    values: list[float],
    default: float = 50.0,
) -> float:
    if not rows or not values:
        return default
    weights = [max(_player_minutes(row), 1.0) for row in rows]
    total_weight = sum(weights)
    if total_weight <= 0:
        return default
    return round(sum(value * weight for value, weight in zip(values, weights)) / total_weight, 1)


def _archetype_score(archetype: dict[str, Any] | None, key: str) -> float:
    return _as_float((archetype or {}).get("scores", {}).get(key), 0.0)


def _player_clutch_traits(
    player: dict[str, Any],
    archetype: dict[str, Any] | None,
) -> dict[str, float]:
    minutes = max(_player_minutes(player), 1.0)
    points_per_minute = _metric(player, "points") / minutes
    assists_per_minute = _metric(player, "assists") / minutes
    rebounds_per_minute = _metric(player, "rebounds") / minutes
    turnovers_per_minute = _metric(player, "turnovers") / minutes
    blocks_per_minute = _metric(player, "blocks") / minutes
    stocks_per_minute = (_metric(player, "steals") + _metric(player, "blocks")) / minutes
    fta_per_minute = _metric(player, "free_throw_attempts") / minutes
    usage = _metric_percentage(player, "usage_rate", 20.0)
    true_shooting = _metric_percentage(player, "true_shooting_proxy", 56.0)
    ft_pct = _metric_percentage(player, "free_throw_percentage", 78.0)
    foul_risk = _metric_percentage(player, "foul_risk", 25.0)

    isolation_score = max(
        _archetype_score(archetype, "isolation_scorer"),
        _archetype_score(archetype, "pull_up_shooting_guard") * 0.85,
        _score(points_per_minute, 0.35, 0.85) * 0.55 + _score(usage, 18.0, 34.0) * 0.45,
    )
    star_creation = _clip(
        _score(points_per_minute, 0.35, 0.85) * 0.35
        + _score(assists_per_minute, 0.03, 0.22) * 0.25
        + _score(usage, 18.0, 34.0) * 0.25
        + isolation_score * 0.15,
    )
    late_shot_creation = _clip(
        star_creation * 0.45
        + isolation_score * 0.30
        + _score(true_shooting, 52.0, 64.0) * 0.15
        + _score(turnovers_per_minute, 0.10, 0.03) * 0.10,
    )
    free_throw_reliability = _clip(
        _score(ft_pct, 68.0, 90.0) * 0.60
        + _score(fta_per_minute, 0.04, 0.20) * 0.25
        + _score(true_shooting, 52.0, 64.0) * 0.15,
    )
    rim_protection = max(
        _archetype_score(archetype, "rim_protector"),
        _score(blocks_per_minute, 0.015, 0.085),
    )
    switchability = max(
        _archetype_score(archetype, "switchable_wing"),
        _archetype_score(archetype, "point_of_attack_defender") * 0.85,
        _score(stocks_per_minute, 0.03, 0.095),
    )

    return {
        "late_game_shot_creation": round(late_shot_creation, 1),
        "isolation_scoring": round(isolation_score, 1),
        "free_throw_reliability": round(free_throw_reliability, 1),
        "turnover_risk": _score(turnovers_per_minute, 0.03, 0.11),
        "rim_protection": round(rim_protection, 1),
        "switchability": round(switchability, 1),
        "rebounding": _score(rebounds_per_minute, 0.08, 0.34),
        "star_creation": round(star_creation, 1),
        "foul_risk": round(_clip(foul_risk), 1),
    }


def _feature_label(feature: str, score: float) -> str:
    labels = {
        "late_game_shot_creation": ("limited late-clock creation", "solid late-clock creation", "strong late-clock shot creation"),
        "isolation_scoring": ("limited isolation scoring", "solid isolation scoring", "strong isolation scoring"),
        "free_throw_reliability": ("shaky free throw reliability", "solid free throw reliability", "strong free throw reliability"),
        "turnover_risk": ("low turnover risk", "moderate turnover risk", "high turnover risk"),
        "rim_protection": ("limited rim protection", "solid rim protection", "strong rim protection"),
        "switchability": ("limited defensive versatility", "solid defensive versatility", "strong defensive versatility"),
        "rebounding": ("weaker rebounding", "solid rebounding", "strong rebounding"),
        "star_creation": ("limited star creation", "solid star creation", "strong star creation"),
        "foul_risk": ("low foul risk", "moderate foul risk", "high foul risk"),
    }[feature]
    if score < 35.0:
        return labels[0]
    if score > 65.0:
        return labels[2]
    return labels[1]


def _metric_record(feature: str, score: float) -> dict[str, Any]:
    return {
        "score": round(_clip(score), 1),
        "label": _feature_label(feature, score),
    }


def _team_clutch_feature_profile(
    team: str,
    player_rows: list[dict[str, Any]],
    player_archetypes: Any,
    closing_lineup: dict[str, Any] | None,
) -> dict[str, Any]:
    archetypes = _archetype_index(player_archetypes)
    closing_players = _closing_players(team, closing_lineup, player_rows)
    traits = [
        _player_clutch_traits(player, archetypes.get((team, _player(player))))
        for player in closing_players
    ]

    top_creators = sorted(
        list(zip(closing_players, traits)),
        key=lambda item: item[1]["star_creation"],
        reverse=True,
    )[:2]
    top_creator_rows = [row for row, _ in top_creators]
    top_creator_traits = [trait for _, trait in top_creators]

    feature_scores = {
        "late_game_shot_creation": _weighted_average(
            top_creator_rows,
            [trait["late_game_shot_creation"] for trait in top_creator_traits],
        ),
        "isolation_scoring": _weighted_average(
            top_creator_rows,
            [trait["isolation_scoring"] for trait in top_creator_traits],
        ),
        "free_throw_reliability": _weighted_average(
            closing_players,
            [trait["free_throw_reliability"] for trait in traits],
        ),
        "turnover_risk": _weighted_average(
            closing_players,
            [trait["turnover_risk"] for trait in traits],
            default=35.0,
        ),
        "rim_protection": max([trait["rim_protection"] for trait in traits] or [50.0]),
        "switchability": _weighted_average(
            closing_players,
            [trait["switchability"] for trait in traits],
        ),
        "rebounding": _weighted_average(
            closing_players,
            [trait["rebounding"] for trait in traits],
        ),
        "star_creation": _weighted_average(
            top_creator_rows,
            [trait["star_creation"] for trait in top_creator_traits],
        ),
        "foul_risk": _weighted_average(
            closing_players,
            [trait["foul_risk"] for trait in traits],
            default=25.0,
        ),
    }

    if closing_lineup:
        feature_scores["rebounding"] = round(
            (feature_scores["rebounding"] + _as_float(closing_lineup.get("rebounding_strength"), 50.0)) / 2.0,
            1,
        )
        feature_scores["switchability"] = round(
            (feature_scores["switchability"] + _as_float(closing_lineup.get("defensive_switchability"), 50.0)) / 2.0,
            1,
        )
        feature_scores["foul_risk"] = round(
            (feature_scores["foul_risk"] + _as_float(closing_lineup.get("foul_risk"), 25.0)) / 2.0,
            1,
        )

    return {
        "team": team,
        "closing_five": _display_player_names(closing_players),
        "projected_closing_minutes": _as_float((closing_lineup or {}).get("projected_minutes"), 0.0),
        "closing_lineup_adjusted_net_rating": _as_float(
            (closing_lineup or {}).get("adjusted_lineup_net_rating"),
            0.0,
        ),
        "features": {
            feature: _metric_record(feature, feature_scores[feature])
            for feature in CLUTCH_FEATURES
        },
        "player_traits": [
            {
                "player": _player(player),
                "traits": trait,
            }
            for player, trait in zip(closing_players, traits)
        ],
    }


def build_clutch_features(
    finals_context: dict[str, Any],
    player_projections: Any = None,
    player_archetypes: Any = None,
    lineup_features: dict[str, list[dict[str, Any]]] | None = None,
    lineup_stats: Any = None,
    team_ratings: Any = None,
) -> dict[str, dict[str, Any]]:
    """Build clutch feature profiles for both Finals teams."""
    if player_projections is None:
        player_projections = project_finals_players(finals_context)
    if player_archetypes is None:
        player_archetypes = classify_players(player_projections)
    if lineup_features is None:
        lineup_features = build_lineup_features(
            finals_context,
            player_projections=player_projections,
            player_archetypes=player_archetypes,
            lineup_stats=lineup_stats,
            team_ratings=team_ratings,
        )

    player_rows = _iter_records(player_projections)
    profiles = {}
    for team in [
        str(finals_context.get("team_a") or "").strip(),
        str(finals_context.get("team_b") or "").strip(),
    ]:
        if not team:
            continue
        team_players = [row for row in player_rows if _team(row) == team]
        closing_lineup = _lineup_by_type(lineup_features.get(team, []), "closing_lineup")
        profiles[team] = _team_clutch_feature_profile(
            team,
            team_players,
            player_archetypes,
            closing_lineup,
        )
    return profiles


def clutch_feature_vector(clutch_features: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Flatten clutch profiles into numeric model features."""
    features: dict[str, float] = {}
    for team, profile in clutch_features.items():
        for feature, record in profile["features"].items():
            features[f"{team}_{feature}"] = float(record["score"])
        features[f"{team}_closing_lineup_adjusted_net_rating"] = float(
            profile["closing_lineup_adjusted_net_rating"]
        )
    return features


def summarize_clutch_features(clutch_features: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Return readable clutch trait summaries by team."""
    summaries = {}
    for team, profile in clutch_features.items():
        ranked = sorted(
            profile["features"].items(),
            key=lambda item: abs(item[1]["score"] - 50.0),
            reverse=True,
        )
        summaries[team] = [record["label"] for _, record in ranked[:5]]
    return summaries


if __name__ == "__main__":
    from src.data.build_dataset import build_finals_context

    context = build_finals_context()
    features = build_clutch_features(context)
    for team, profile in features.items():
        print(team, profile["closing_five"])
        for feature, record in profile["features"].items():
            print(f"  {feature}: {record['score']} ({record['label']})")
