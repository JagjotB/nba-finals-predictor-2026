"""Lineup construction and strength features."""

from __future__ import annotations

from math import isnan
from typing import Any

from src.features.archetype_features import classify_players
from src.models.train_player_model import project_finals_players


SHRINKAGE_K = 250.0
DEFAULT_OFF_RATING = 115.0
DEFAULT_DEF_RATING = 115.0

LINEUP_TYPES = [
    "starting_lineup",
    "closing_lineup",
    "bench_lineup",
    "small_ball_lineup",
    "big_lineup",
    "non_star_minutes",
    "staggered_star_minutes",
]

TEAM_KEYS = ["team", "TEAM", "TEAM_ABBREVIATION", "TEAM_NAME"]
PLAYER_KEYS = ["player", "PLAYER_NAME", "name", "Name"]

LINEUP_ALIASES = {
    "lineup_type": ["lineup_type", "type", "lineup_name"],
    "players": ["players", "lineup", "GROUP_NAME", "group_name", "players_display"],
    "minutes": ["minutes", "MIN", "min", "sample_minutes"],
    "offensive_rating": ["offensive_rating", "OFF_RATING", "off_rating"],
    "defensive_rating": ["defensive_rating", "DEF_RATING", "def_rating"],
    "net_rating": ["net_rating", "NET_RATING"],
    "rebounding_strength": ["rebounding_strength", "REB_PCT", "reb_pct"],
    "spacing": ["spacing", "FG3A_RATE", "three_point_attempt_rate"],
    "defensive_switchability": ["defensive_switchability", "switchability"],
    "foul_risk": ["foul_risk", "PF", "personal_fouls"],
}

TEAM_RATING_ALIASES = {
    "offensive_rating": ["offensive_rating", "OFF_RATING", "off_rating"],
    "defensive_rating": ["defensive_rating", "DEF_RATING", "def_rating"],
    "net_rating": ["net_rating", "NET_RATING"],
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


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score(value: float, low: float, high: float, invert: bool = False) -> float:
    if high == low:
        return 50.0
    scaled = (value - low) / (high - low) * 100.0
    if invert:
        scaled = 100.0 - scaled
    return round(_clip(scaled, 0.0, 100.0), 1)


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


def _metric(row: dict[str, Any], aliases: dict[str, list[str]], metric: str, default: float = 0.0) -> float:
    return _as_float(_first_value(row, aliases[metric]), default)


def _player_minutes(player: dict[str, Any]) -> float:
    return _as_float(player.get("minutes") or player.get("projected_minutes"), 0.0)


def _player_key(team: str, player: str, role: Any = None) -> str:
    role_text = str(role or "").strip()
    return f"{team}:{player}:{role_text}".strip(":")


def _flatten_player_projections(player_projections: Any) -> list[dict[str, Any]]:
    rows = _iter_records(player_projections)
    return sorted(rows, key=lambda row: (_team(row), -_player_minutes(row), _player(row)))


def _rotation_lookup(finals_context: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for team, rotations in finals_context.get("rotations", {}).items():
        for rotation in rotations:
            player = str(rotation.get("player") or "").strip()
            role = str(rotation.get("role") or "").strip()
            if not player:
                continue
            lookup[(team, player, role)] = rotation
            lookup.setdefault((team, player, ""), rotation)
    return lookup


def _enrich_players_with_rotation_flags(
    players: list[dict[str, Any]],
    finals_context: dict[str, Any],
) -> list[dict[str, Any]]:
    lookup = _rotation_lookup(finals_context)
    enriched = []
    for player in players:
        team = _team(player)
        name = _player(player)
        role = str(player.get("role") or "").strip()
        rotation = lookup.get((team, name, role)) or lookup.get((team, name, "")) or {}
        row = dict(player)
        row.setdefault("team", team)
        row.setdefault("player", name)
        row.setdefault("role", role or rotation.get("role"))
        row["is_starter"] = bool(row.get("is_starter", rotation.get("is_starter", False)))
        row["is_closer"] = bool(row.get("is_closer", rotation.get("is_closer", False)))
        row.setdefault("player_key", _player_key(team, name, row.get("role")))
        enriched.append(row)
    return enriched


def _archetype_index(player_archetypes: Any) -> dict[str, dict[str, Any]]:
    rows = _iter_records(player_archetypes)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        team = _team(row)
        player = _player(row)
        player_key = str(row.get("player_key") or _player_key(team, player, row.get("role")))
        if player_key:
            index[player_key] = row
        if team and player:
            index.setdefault(_player_key(team, player), row)
    return index


def _archetype_scores(player: dict[str, Any], index: dict[str, dict[str, Any]]) -> dict[str, float]:
    key = str(player.get("player_key") or _player_key(_team(player), _player(player), player.get("role")))
    fallback_key = _player_key(_team(player), _player(player))
    archetype = index.get(key) or index.get(fallback_key) or {}
    return {
        score_key: _as_float(score)
        for score_key, score in archetype.get("scores", {}).items()
    }


def _percentage(value: Any, default: float = 0.0) -> float:
    number = _as_float(value, default)
    if 0.0 <= number <= 1.5:
        return number * 100.0
    return number


def _player_trait_scores(
    player: dict[str, Any],
    archetype_scores: dict[str, float],
) -> dict[str, float]:
    minutes = max(_player_minutes(player), 1.0)
    points_per_minute = _as_float(player.get("points"), 0.0) / minutes
    rebounds_per_minute = _as_float(player.get("rebounds"), 0.0) / minutes
    stocks_per_minute = (
        _as_float(player.get("steals"), 0.0)
        + _as_float(player.get("blocks"), 0.0)
    ) / minutes
    threes_attempted_per_minute = _as_float(player.get("threes_attempted"), 0.0) / minutes
    assists_per_minute = _as_float(player.get("assists"), 0.0) / minutes
    turnovers_per_minute = _as_float(player.get("turnovers"), 0.0) / minutes
    usage = _percentage(player.get("usage_rate"), 20.0)
    true_shooting = _percentage(player.get("true_shooting_proxy"), 56.0)
    foul_risk = _percentage(player.get("foul_risk"), 0.25)

    spacing = max(
        _score(threes_attempted_per_minute, 0.05, 0.32),
        archetype_scores.get("spot_up_shooter", 0.0),
        archetype_scores.get("stretch_big", 0.0),
    )
    rebounding = max(
        _score(rebounds_per_minute, 0.08, 0.34),
        archetype_scores.get("post_big", 0.0) * 0.75,
        archetype_scores.get("rim_protector", 0.0) * 0.65,
    )
    switchability = max(
        archetype_scores.get("switchable_wing", 0.0),
        archetype_scores.get("point_of_attack_defender", 0.0) * 0.80,
        archetype_scores.get("rim_protector", 0.0) * 0.50,
        _score(stocks_per_minute, 0.03, 0.10),
    )
    offense = _clip(
        _score(points_per_minute, 0.30, 0.85) * 0.40
        + _score(assists_per_minute, 0.03, 0.22) * 0.20
        + _score(true_shooting, 51.0, 64.0) * 0.20
        + _score(usage, 12.0, 32.0) * 0.10
        + spacing * 0.10,
        0.0,
        100.0,
    )
    defense = _clip(
        switchability * 0.45
        + rebounding * 0.25
        + _score(stocks_per_minute, 0.025, 0.09) * 0.20
        + _score(foul_risk, 15.0, 75.0, invert=True) * 0.10,
        0.0,
        100.0,
    )
    ball_security = _score(turnovers_per_minute, 0.10, 0.03)
    star_score = _clip(
        _score(minutes, 14.0, 40.0) * 0.40
        + _score(usage, 16.0, 32.0) * 0.25
        + _score(points_per_minute, 0.35, 0.85) * 0.25
        + _score(assists_per_minute, 0.03, 0.22) * 0.10,
        0.0,
        100.0,
    )

    return {
        "offense": round(offense, 1),
        "defense": round(defense, 1),
        "rebounding": round(rebounding, 1),
        "spacing": round(spacing, 1),
        "switchability": round(switchability, 1),
        "foul_risk": round(_clip(foul_risk, 0.0, 100.0), 1),
        "ball_security": round(ball_security, 1),
        "star_score": round(star_score, 1),
        "big_score": round(
            max(archetype_scores.get("post_big", 0.0), archetype_scores.get("rim_protector", 0.0), rebounding),
            1,
        ),
        "small_ball_score": round(
            _clip(spacing * 0.45 + switchability * 0.35 + offense * 0.20, 0.0, 100.0),
            1,
        ),
    }


def _prepare_team_players(
    team: str,
    player_projections: Any,
    player_archetypes: Any,
    finals_context: dict[str, Any],
) -> list[dict[str, Any]]:
    players = [
        row
        for row in _enrich_players_with_rotation_flags(
            _flatten_player_projections(player_projections),
            finals_context,
        )
        if _team(row) == team
    ]

    if not player_archetypes:
        player_archetypes = classify_players(players)
    archetype_lookup = _archetype_index(player_archetypes)

    prepared = []
    for player in players:
        scores = _archetype_scores(player, archetype_lookup)
        row = dict(player)
        row["lineup_traits"] = _player_trait_scores(row, scores)
        row["archetype_scores"] = scores
        prepared.append(row)
    return sorted(prepared, key=lambda row: _player_minutes(row), reverse=True)


def _fill_to_five(selected: list[dict[str, Any]], pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {str(row.get("player_key")) for row in selected}
    filled = list(selected)
    for player in pool:
        key = str(player.get("player_key"))
        if key in seen:
            continue
        filled.append(player)
        seen.add(key)
        if len(filled) >= 5:
            break
    return filled[:5]


def _lineup_player_names(players: list[dict[str, Any]]) -> list[str]:
    return [_player(player) for player in players]


def _lineup_key(team: str, lineup_type: str, players: list[dict[str, Any]]) -> str:
    names = "|".join(sorted(_lineup_player_names(players)))
    return f"{team}:{lineup_type}:{names}"


def _lineup_minutes(lineup_type: str, players: list[dict[str, Any]]) -> float:
    if not players:
        return 0.0

    min_player_minutes = min(_player_minutes(player) for player in players)
    avg_player_minutes = sum(_player_minutes(player) for player in players) / len(players)
    type_factor = {
        "starting_lineup": 0.52,
        "closing_lineup": 0.32,
        "bench_lineup": 0.45,
        "small_ball_lineup": 0.28,
        "big_lineup": 0.24,
        "non_star_minutes": 0.38,
        "staggered_star_minutes": 0.42,
    }[lineup_type]
    cap = {
        "starting_lineup": 18.0,
        "closing_lineup": 10.0,
        "bench_lineup": 12.0,
        "small_ball_lineup": 9.0,
        "big_lineup": 8.0,
        "non_star_minutes": 14.0,
        "staggered_star_minutes": 16.0,
    }[lineup_type]
    estimated = (min_player_minutes * 0.70 + avg_player_minutes * 0.30) * type_factor
    if len(players) < 5:
        estimated *= len(players) / 5.0
    return round(_clip(estimated, 0.0, cap), 1)


def build_lineup_candidates(team: str, team_players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create the standard Finals lineup groups for one team."""
    sorted_by_minutes = sorted(team_players, key=lambda row: _player_minutes(row), reverse=True)
    starters = [player for player in sorted_by_minutes if player.get("is_starter")]
    closers = [player for player in sorted_by_minutes if player.get("is_closer")]
    bench = [player for player in sorted_by_minutes if not player.get("is_starter")]
    stars = sorted(
        sorted_by_minutes,
        key=lambda row: row["lineup_traits"]["star_score"],
        reverse=True,
    )[:2]
    star_keys = {str(player.get("player_key")) for player in stars}
    non_stars = [
        player for player in sorted_by_minutes if str(player.get("player_key")) not in star_keys
    ]
    small_ball = sorted(
        sorted_by_minutes,
        key=lambda row: row["lineup_traits"]["small_ball_score"],
        reverse=True,
    )
    big = sorted(
        sorted_by_minutes,
        key=lambda row: row["lineup_traits"]["big_score"],
        reverse=True,
    )

    candidates = [
        ("starting_lineup", _fill_to_five(starters, sorted_by_minutes)),
        ("closing_lineup", _fill_to_five(closers, sorted_by_minutes)),
        ("bench_lineup", _fill_to_five(bench, sorted_by_minutes)),
        ("small_ball_lineup", _fill_to_five(small_ball[:5], sorted_by_minutes)),
        ("big_lineup", _fill_to_five(big[:5], sorted_by_minutes)),
        ("non_star_minutes", _fill_to_five(non_stars, sorted_by_minutes)),
        ("staggered_star_minutes", _fill_to_five(stars[:1] + non_stars, sorted_by_minutes)),
    ]

    return [
        {
            "team": team,
            "lineup_type": lineup_type,
            "players": players,
            "player_names": _lineup_player_names(players),
            "lineup_key": _lineup_key(team, lineup_type, players),
            "projected_minutes": _lineup_minutes(lineup_type, players),
        }
        for lineup_type, players in candidates
    ]


def _parse_lineup_players(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    for delimiter in (" - ", "|", ";", ","):
        if delimiter in text:
            return [part.strip() for part in text.split(delimiter) if part.strip()]
    return [text]


def _normalized_player_set(players: list[str]) -> tuple[str, ...]:
    return tuple(sorted(str(player).strip().lower() for player in players if str(player).strip()))


def _build_name_map(team_players: list[dict[str, Any]]) -> dict[str, str]:
    """Map NBA API abbreviated names to full rotation names.

    NBA API returns "J. Brunson"; rotations have "Jalen Brunson".
    Builds: {"j. brunson": "jalen brunson", "j. hart": "josh hart", ...}
    """
    name_map: dict[str, str] = {}
    for player in team_players:
        full = str(player.get("player") or "").strip()
        if not full or " " not in full:
            continue
        parts = full.split()
        # Standard abbreviation: first initial + ". " + rest of name
        abbreviated = f"{parts[0][0].lower()}. {' '.join(parts[1:]).lower()}"
        name_map[abbreviated] = full.lower()
        # Also map last-name-only for edge cases
        name_map[parts[-1].lower()] = full.lower()
    return name_map


def _resolve_player_names(
    players: list[str],
    name_map: dict[str, str],
) -> list[str]:
    """Convert abbreviated API names to full names using the map."""
    resolved = []
    for p in players:
        key = p.strip().lower()
        resolved.append(name_map.get(key, key))
    return resolved


_MIN_LINEUP_MINUTES = 5.0  # ignore garbage-time lineups with <5 min


def _lineup_stats_index(
    lineup_stats: Any,
    name_map: dict[str, str] | None = None,
) -> dict[tuple[str, str, tuple[str, ...]], dict[str, Any]]:
    index: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for row in _iter_records(lineup_stats):
        team = _team(row)
        lineup_type = str(
            _first_value(row, LINEUP_ALIASES["lineup_type"]) or ""
        ).strip()
        players = _parse_lineup_players(
            _first_value(row, LINEUP_ALIASES["players"])
        )
        if name_map:
            players = _resolve_player_names(players, name_map)
        player_set = _normalized_player_set(players)
        if not team:
            continue
        # Skip lineups with almost no sample — their ratings are noise
        minutes = _as_float(_first_value(row, LINEUP_ALIASES["minutes"]), 0.0)
        if minutes < _MIN_LINEUP_MINUTES:
            continue
        if lineup_type:
            index[(team, lineup_type, tuple())] = row
        if player_set:
            # Prefer higher-minute entry when multiple lineups match same set
            existing = index.get((team, "", player_set))
            existing_min = _as_float(
                _first_value(existing or {}, LINEUP_ALIASES["minutes"]), 0.0
            )
            if existing is None or minutes > existing_min:
                index[(team, "", player_set)] = row
                if lineup_type:
                    index[(team, lineup_type, player_set)] = row
    return index


def _find_lineup_stat(
    index: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]],
    team: str,
    lineup_type: str,
    player_names: list[str],
) -> dict[str, Any] | None:
    player_set = _normalized_player_set(player_names)
    return (
        index.get((team, lineup_type, player_set))
        or index.get((team, "", player_set))
        or index.get((team, lineup_type, tuple()))
    )


def _team_rating_index(team_ratings: Any) -> dict[str, dict[str, float]]:
    index: dict[str, dict[str, float]] = {}
    for row in _iter_records(team_ratings):
        team = _team(row)
        if not team:
            continue
        off_rating = _metric(row, TEAM_RATING_ALIASES, "offensive_rating", DEFAULT_OFF_RATING)
        def_rating = _metric(row, TEAM_RATING_ALIASES, "defensive_rating", DEFAULT_DEF_RATING)
        net_rating = _metric(row, TEAM_RATING_ALIASES, "net_rating", off_rating - def_rating)
        index[team] = {
            "offensive_rating": off_rating,
            "defensive_rating": def_rating,
            "net_rating": net_rating,
        }
    return index


def _team_rating_from_players(team_players: list[dict[str, Any]]) -> dict[str, float]:
    if not team_players:
        return {
            "offensive_rating": DEFAULT_OFF_RATING,
            "defensive_rating": DEFAULT_DEF_RATING,
            "net_rating": 0.0,
        }

    total_minutes = sum(_player_minutes(player) for player in team_players) or 1.0
    offense = sum(
        player["lineup_traits"]["offense"] * _player_minutes(player)
        for player in team_players
    ) / total_minutes
    defense = sum(
        player["lineup_traits"]["defense"] * _player_minutes(player)
        for player in team_players
    ) / total_minutes
    off_rating = DEFAULT_OFF_RATING + (offense - 50.0) * 0.12
    def_rating = DEFAULT_DEF_RATING - (defense - 50.0) * 0.10
    return {
        "offensive_rating": round(off_rating, 1),
        "defensive_rating": round(def_rating, 1),
        "net_rating": round(off_rating - def_rating, 1),
    }


def apply_lineup_shrinkage(
    lineup_net_rating: float,
    team_net_rating: float,
    minutes: float,
    k: float = SHRINKAGE_K,
) -> float:
    """Shrink noisy lineup net rating toward team net rating."""
    minutes = max(_as_float(minutes), 0.0)
    k = max(_as_float(k, SHRINKAGE_K), 0.0)
    if minutes + k == 0.0:
        return round(team_net_rating, 2)
    return round(
        (minutes / (minutes + k)) * lineup_net_rating
        + (k / (minutes + k)) * team_net_rating,
        2,
    )


def _lineup_average(
    players: list[dict[str, Any]],
    trait: str,
    default: float = 50.0,
) -> float:
    if not players:
        return default
    total_minutes = sum(_player_minutes(player) for player in players) or len(players)
    return round(
        sum(player["lineup_traits"][trait] * max(_player_minutes(player), 1.0) for player in players)
        / total_minutes,
        1,
    )


def _lineup_ratings_from_players(
    lineup: dict[str, Any],
    team_rating: dict[str, float],
) -> dict[str, float]:
    players = lineup["players"]
    offense = _lineup_average(players, "offense")
    defense = _lineup_average(players, "defense")
    rebounding = _lineup_average(players, "rebounding")
    spacing = _lineup_average(players, "spacing")
    switchability = _lineup_average(players, "switchability")
    foul_risk = _lineup_average(players, "foul_risk", 25.0)
    ball_security = _lineup_average(players, "ball_security")

    offensive_rating = (
        team_rating["offensive_rating"]
        + (offense - 50.0) * 0.12
        + (spacing - 50.0) * 0.04
        + (ball_security - 50.0) * 0.03
    )
    defensive_rating = (
        team_rating["defensive_rating"]
        - (defense - 50.0) * 0.10
        - (switchability - 50.0) * 0.03
        - (rebounding - 50.0) * 0.02
        + (foul_risk - 25.0) * 0.04
    )

    return {
        "offensive_rating": round(offensive_rating, 1),
        "defensive_rating": round(defensive_rating, 1),
        "net_rating": round(offensive_rating - defensive_rating, 1),
        "rebounding_strength": rebounding,
        "spacing": spacing,
        "defensive_switchability": switchability,
        "foul_risk": foul_risk,
    }


def _apply_lineup_stat_overrides(
    metrics: dict[str, float],
    lineup_stat: dict[str, Any] | None,
) -> tuple[dict[str, float], float | None]:
    if not lineup_stat:
        return metrics, None

    updated = dict(metrics)
    sample_minutes = _metric(lineup_stat, LINEUP_ALIASES, "minutes", 0.0)
    has_net_override = False
    for metric in ["offensive_rating", "defensive_rating", "net_rating"]:
        value = _first_value(lineup_stat, LINEUP_ALIASES[metric])
        if value is not None:
            updated[metric] = round(_as_float(value), 1)
            if metric == "net_rating":
                has_net_override = True

    for metric in ["rebounding_strength", "spacing", "defensive_switchability", "foul_risk"]:
        value = _first_value(lineup_stat, LINEUP_ALIASES[metric])
        if value is not None:
            updated[metric] = round(_clip(_percentage(value), 0.0, 100.0), 1)

    if not has_net_override:
        updated["net_rating"] = round(updated["offensive_rating"] - updated["defensive_rating"], 1)

    return updated, sample_minutes if sample_minutes > 0 else None


def analyze_lineup(
    lineup: dict[str, Any],
    team_rating: dict[str, float],
    lineup_stat: dict[str, Any] | None = None,
    shrinkage_k: float = SHRINKAGE_K,
) -> dict[str, Any]:
    """Analyze one lineup's projected strength and adjusted net rating."""
    metrics = _lineup_ratings_from_players(lineup, team_rating)
    metrics, sample_minutes = _apply_lineup_stat_overrides(metrics, lineup_stat)
    shrinkage_minutes = sample_minutes if sample_minutes is not None else lineup["projected_minutes"]

    # Shrink net, OFF, and DEF individually — raw lineup stats in small samples
    # can produce extreme values (e.g. DEF=91 in 184 min, OFF=90 in 13 min).
    adjusted_net_rating = apply_lineup_shrinkage(
        metrics["net_rating"],
        team_rating["net_rating"],
        shrinkage_minutes,
        shrinkage_k,
    )
    adjusted_off = apply_lineup_shrinkage(
        metrics["offensive_rating"],
        team_rating["offensive_rating"],
        shrinkage_minutes,
        shrinkage_k,
    )
    adjusted_def = apply_lineup_shrinkage(
        metrics["defensive_rating"],
        team_rating["defensive_rating"],
        shrinkage_minutes,
        shrinkage_k,
    )

    return {
        "team": lineup["team"],
        "lineup_type": lineup["lineup_type"],
        "lineup_key": lineup["lineup_key"],
        "players": lineup["player_names"],
        "projected_minutes": lineup["projected_minutes"],
        "historical_sample_minutes": round(sample_minutes or 0.0, 1),
        "shrinkage_minutes": round(shrinkage_minutes, 1),
        "offensive_rating": round(adjusted_off, 1),
        "defensive_rating": round(adjusted_def, 1),
        "net_rating": round(adjusted_net_rating, 1),
        "adjusted_lineup_net_rating": adjusted_net_rating,
        "team_net_rating_anchor": round(team_rating["net_rating"], 1),
        "rebounding_strength": metrics["rebounding_strength"],
        "spacing": metrics["spacing"],
        "defensive_switchability": metrics["defensive_switchability"],
        "foul_risk": metrics["foul_risk"],
        "shrinkage_k": shrinkage_k,
    }


def analyze_team_lineups(
    team: str,
    player_projections: Any,
    finals_context: dict[str, Any],
    player_archetypes: Any = None,
    lineup_stats: Any = None,
    team_ratings: Any = None,
    shrinkage_k: float = SHRINKAGE_K,
) -> list[dict[str, Any]]:
    """Analyze standard lineup groups for one Finals team."""
    team_players = _prepare_team_players(team, player_projections, player_archetypes, finals_context)
    if not team_players:
        return []

    team_rating = _team_rating_index(team_ratings).get(team) or _team_rating_from_players(team_players)
    name_map = _build_name_map(team_players)
    lineup_index = _lineup_stats_index(lineup_stats, name_map=name_map)
    candidates = build_lineup_candidates(team, team_players)
    return [
        analyze_lineup(
            lineup,
            team_rating,
            _find_lineup_stat(lineup_index, team, lineup["lineup_type"], lineup["player_names"]),
            shrinkage_k,
        )
        for lineup in candidates
    ]


def build_lineup_features(
    finals_context: dict[str, Any],
    player_projections: Any = None,
    player_archetypes: Any = None,
    lineup_stats: Any = None,
    team_ratings: Any = None,
    shrinkage_k: float = SHRINKAGE_K,
) -> dict[str, list[dict[str, Any]]]:
    """Build lineup features for both Finals teams."""
    if player_projections is None:
        player_projections = project_finals_players(finals_context)

    # Prefer live lineup stats from context over caller-supplied
    if lineup_stats is None:
        lineup_stats = finals_context.get("lineup_stats")

    # Build team ratings from live team stats if not supplied
    if team_ratings is None:
        live_team_stats = finals_context.get("team_stats") or {}
        if live_team_stats:
            team_ratings = live_team_stats

    teams = [
        str(finals_context.get("team_a") or "").strip(),
        str(finals_context.get("team_b") or "").strip(),
    ]
    return {
        team: analyze_team_lineups(
            team,
            player_projections,
            finals_context,
            player_archetypes=player_archetypes,
            lineup_stats=lineup_stats,
            team_ratings=team_ratings,
            shrinkage_k=shrinkage_k,
        )
        for team in teams
        if team
    }


def summarize_lineup_features(
    lineup_features: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Summarize lineup strengths by team."""
    summaries: dict[str, dict[str, Any]] = {}
    for team, lineups in lineup_features.items():
        if not lineups:
            summaries[team] = {}
            continue
        best_lineup = max(lineups, key=lambda row: row["adjusted_lineup_net_rating"])
        riskiest_lineup = max(lineups, key=lambda row: row["foul_risk"])
        summaries[team] = {
            "best_lineup_type": best_lineup["lineup_type"],
            "best_adjusted_net_rating": best_lineup["adjusted_lineup_net_rating"],
            "closing_adjusted_net_rating": next(
                (
                    row["adjusted_lineup_net_rating"]
                    for row in lineups
                    if row["lineup_type"] == "closing_lineup"
                ),
                None,
            ),
            "strongest_spacing_lineup": max(lineups, key=lambda row: row["spacing"])["lineup_type"],
            "strongest_rebounding_lineup": max(lineups, key=lambda row: row["rebounding_strength"])["lineup_type"],
            "riskiest_foul_lineup": riskiest_lineup["lineup_type"],
        }
    return summaries


def lineup_feature_vector(lineup_features: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Flatten lineup features into numeric model inputs."""
    features: dict[str, float] = {}
    for team, lineups in lineup_features.items():
        for lineup in lineups:
            prefix = f"{team}_{lineup['lineup_type']}"
            for metric in [
                "projected_minutes",
                "offensive_rating",
                "defensive_rating",
                "net_rating",
                "adjusted_lineup_net_rating",
                "rebounding_strength",
                "spacing",
                "defensive_switchability",
                "foul_risk",
            ]:
                features[f"{prefix}_{metric}"] = float(lineup[metric])
    return features


if __name__ == "__main__":
    from src.data.build_dataset import build_finals_context

    context = build_finals_context()
    features = build_lineup_features(context)
    for team, lineups in features.items():
        print(team)
        for lineup in lineups:
            print(
                f"  {lineup['lineup_type']}: "
                f"{lineup['projected_minutes']} min, "
                f"{lineup['offensive_rating']} ORtg, "
                f"{lineup['defensive_rating']} DRtg, "
                f"{lineup['net_rating']} net, "
                f"{lineup['adjusted_lineup_net_rating']} adjusted"
            )
