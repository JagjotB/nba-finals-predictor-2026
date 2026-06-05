"""Foul-trouble scenario simulator for key Finals players."""

from __future__ import annotations

from math import exp, isnan, log
from typing import Any

from src.features.playstyle_features import build_playstyle_profiles
from src.models.train_player_model import project_finals_players


TEAM_KEYS = ["team", "TEAM", "TEAM_ABBREVIATION", "TEAM_NAME"]
PLAYER_KEYS = ["player", "PLAYER_NAME", "name", "Name"]

PLAYER_ALIASES = {
    "minutes": ["minutes", "projected_minutes", "MIN", "MP"],
    "minutes_floor": ["minutes_floor", "floor_minutes"],
    "points": ["points", "PTS"],
    "rebounds": ["rebounds", "REB", "TRB"],
    "assists": ["assists", "AST"],
    "turnovers": ["turnovers", "TOV", "TO"],
    "steals": ["steals", "STL"],
    "blocks": ["blocks", "BLK"],
    "personal_fouls": ["personal_fouls", "PF", "fouls"],
    "personal_fouls_per_minute": [
        "personal_fouls_per_minute",
        "fouls_per_minute",
        "pf_per_minute",
    ],
    "historical_foul_rate": ["historical_foul_rate", "career_pf_per_minute"],
    "foul_risk": ["foul_risk"],
    "usage_rate": ["usage_rate", "USG_PCT", "USG%"],
    "defensive_role": ["defensive_role", "role", "defense_role"],
    "matchup_physicality": ["matchup_physicality", "physicality"],
    "primary_rim_protector": ["primary_rim_protector", "is_primary_rim_protector"],
}

MATCHUP_TYPE_PHYSICALITY = {
    "post_up": 0.18,
    "post": 0.18,
    "rim": 0.16,
    "drive": 0.14,
    "pick_and_roll": 0.10,
    "isolation": 0.06,
    "spot_up": -0.04,
}

DEFENSIVE_ROLE_MULTIPLIERS = {
    "primary rim protector": 1.24,
    "rim protector": 1.20,
    "center": 1.18,
    "big": 1.13,
    "point-of-attack defender": 1.12,
    "primary defender": 1.10,
    "wing stopper": 1.06,
    "switch defender": 1.04,
    "guard": 1.00,
}

DEFAULT_BASE_WIN_PROBABILITY = 0.50


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


def _percentage(value: Any, default: float = 0.0) -> float:
    number = _as_float(value, default)
    if 0.0 <= number <= 1.5:
        return number * 100.0
    return number


def _logit(probability: float) -> float:
    probability = _clip(probability, 0.01, 0.99)
    return log(probability / (1.0 - probability))


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def _first_value(row: dict[str, Any] | None, keys: list[str]) -> Any:
    if not row:
        return None
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _metric(row: dict[str, Any], metric: str, default: float = 0.0) -> float:
    return _as_float(_first_value(row, PLAYER_ALIASES[metric]), default)


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


def _opponent(finals_context: dict[str, Any], team: str) -> str:
    team_a = str(finals_context.get("team_a") or "")
    team_b = str(finals_context.get("team_b") or "")
    return team_b if team == team_a else team_a


def _profile_metric(
    playstyle_profiles: dict[str, dict[str, Any]] | None,
    team: str,
    side: str,
    metric: str,
    default: float = 50.0,
) -> float:
    if not playstyle_profiles:
        return default
    return _as_float(
        playstyle_profiles.get(team, {})
        .get(side, {})
        .get("metrics", {})
        .get(metric, {})
        .get("score"),
        default,
    )


def _player_key(row: dict[str, Any]) -> str:
    return str(row.get("player_key") or f"{_team(row)}:{_player(row)}:{row.get('role', '')}").strip(":")


def _rotation_index(finals_context: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for team, rotations in finals_context.get("rotations", {}).items():
        for rotation in rotations:
            player = str(rotation.get("player") or "").strip()
            role = str(rotation.get("role") or "").strip()
            if player:
                index[(team, player)] = {**rotation, "role": role}
    return index


def _load_actual_foul_stats() -> dict[str, dict[str, float]]:
    """Load per-game PF and MIN from the playoff stats cache keyed by lowercase player name."""
    import pathlib, json
    cache_path = pathlib.Path(__file__).parent.parent.parent / "data" / "processed" / "stats_cache" / "player_stats_2025-26_Playoffs.json"
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text())
    except Exception:
        return {}
    lookup: dict[str, dict[str, float]] = {}
    for players in raw.values():
        for p in players:
            name = str(p.get("PLAYER_NAME") or "").strip().lower()
            pf = p.get("PF")
            minutes = p.get("MIN")
            if name and pf is not None and minutes:
                lookup[name] = {"personal_fouls": float(pf), "minutes": float(minutes)}
    return lookup


def _enrich_players_with_rotation(player_rows: Any, finals_context: dict[str, Any]) -> list[dict[str, Any]]:
    rotations = _rotation_index(finals_context)
    actual_fouls = _load_actual_foul_stats()
    enriched = []
    for row in _iter_records(player_rows):
        team = _team(row)
        player = _player(row)
        rotation = rotations.get((team, player), {})
        merged = {**rotation, **row}
        merged.setdefault("team", team)
        merged.setdefault("player", player)
        merged.setdefault("role", row.get("role") or rotation.get("role"))
        # Inject actual playoff foul rate so _foul_rate uses real data, not the
        # generic foul_risk formula (which doesn't capture KAT-level foul trouble).
        foul_data = actual_fouls.get(player.strip().lower(), {})
        if foul_data and not merged.get("personal_fouls"):
            merged.update(foul_data)
        enriched.append(merged)
    return enriched


def _foul_rate(player: dict[str, Any]) -> float:
    direct = _metric(player, "personal_fouls_per_minute", 0.0)
    if direct > 0:
        return direct

    historical = _metric(player, "historical_foul_rate", 0.0)
    if historical > 0:
        return historical

    minutes = max(_metric(player, "minutes", 0.0), 1.0)
    personal_fouls = _metric(player, "personal_fouls", 0.0)
    if personal_fouls > 0:
        return personal_fouls / minutes

    foul_risk = _percentage(_first_value(player, PLAYER_ALIASES["foul_risk"]), 0.0)
    if foul_risk > 0:
        return 0.045 + (foul_risk / 100.0) * 0.075

    role = str(player.get("role") or "").lower()
    if "center" in role or "big" in role:
        return 0.085
    return 0.065


def _defensive_role(player: dict[str, Any]) -> str:
    direct = str(_first_value(player, PLAYER_ALIASES["defensive_role"]) or "").strip().lower()
    if direct:
        return direct
    role = str(player.get("role") or "").strip().lower()
    if "rim" in role or "center" in role or "big" in role:
        return "rim protector"
    if "starter" in role:
        return "primary defender"
    return role or "rotation defender"


def _role_multiplier(defensive_role: str) -> float:
    for key, multiplier in DEFENSIVE_ROLE_MULTIPLIERS.items():
        if key in defensive_role:
            return multiplier
    return 1.0


def _manual_matchup_physicality(
    finals_context: dict[str, Any],
    player: dict[str, Any],
) -> float:
    team = _team(player)
    name = _player(player)
    opponent = _opponent(finals_context, team)
    physicality = 0.0
    for active_player in finals_context.get("active_players", {}).get(team, []):
        if str(active_player.get("player") or "") != name:
            continue
        for role in active_player.get("matchup_roles", []):
            # Foul trouble comes from defensive assignments, not offensive ones.
            # Offensive fouls (charging) are rare; defensive fouls (blocking, reach-ins)
            # are what put players on the bench. Reading the offensive side inflates
            # foul risk for players like Wembanyama who attack the rim.
            if role.get("side") != "defense":
                continue
            if role.get("opponent") != opponent:
                continue
            matchup_type = str(role.get("matchup_type") or "")
            for key, value in MATCHUP_TYPE_PHYSICALITY.items():
                if key in matchup_type:
                    physicality += value
                    break
    return physicality


def _matchup_physicality(
    player: dict[str, Any],
    matchup_physicality: Any = None,
) -> float:
    direct = _metric(player, "matchup_physicality", 0.0)
    if direct:
        return _clip(direct, -0.20, 0.40)

    player_team = _team(player)
    player_name = _player(player)
    for row in _iter_records(matchup_physicality):
        team = _team(row)
        name = _player(row)
        if team and team != player_team:
            continue
        if name and name != player_name:
            continue
        value = (
            row.get("matchup_physicality")
            or row.get("physicality")
            or row.get("expected_impact")
            or 0.0
        )
        numeric_value = _as_float(value, 0.0)
        if abs(numeric_value) > 1.0:
            numeric_value *= 0.04
        return _clip(numeric_value, -0.20, 0.40)
    return 0.0


def _is_primary_rim_protector(player: dict[str, Any], team_players: list[dict[str, Any]]) -> bool:
    direct = _first_value(player, PLAYER_ALIASES["primary_rim_protector"])
    if direct is not None:
        return _as_bool(direct)

    blocks = _metric(player, "blocks", 0.0)
    rebounds = _metric(player, "rebounds", 0.0)
    role = str(player.get("role") or "").lower()
    rim_score = blocks * 3.0 + rebounds * 0.35
    player_name = _player(player)
    best_name = None
    best_score = -1.0
    for candidate in team_players:
        candidate_score = _metric(candidate, "blocks", 0.0) * 3.0 + _metric(candidate, "rebounds", 0.0) * 0.35
        if candidate_score > best_score:
            best_score = candidate_score
            best_name = _player(candidate)
    return player_name == best_name or "rim" in role or "center" in role


def _foul_trouble_probability(expected_fouls: float, player: dict[str, Any]) -> float:
    minutes = _metric(player, "minutes", 0.0)
    high_minutes_pressure = _clip((minutes - 28.0) / 14.0, 0.0, 1.0)
    # Smooth approximation for probability of meaningful foul trouble.
    raw = 1.0 / (1.0 + exp(-(expected_fouls - 3.2) * 1.45))
    return round(_clip(raw + high_minutes_pressure * 0.06, 0.02, 0.85), 3)


def estimate_player_foul_risk(
    player: dict[str, Any],
    finals_context: dict[str, Any],
    team_players: list[dict[str, Any]] | None = None,
    playstyle_profiles: dict[str, dict[str, Any]] | None = None,
    matchup_physicality: Any = None,
) -> dict[str, Any]:
    """Estimate foul-trouble risk for one player."""
    team_players = team_players or [player]
    team = _team(player)
    opponent = _opponent(finals_context, team)
    expected_minutes = _metric(player, "minutes", 0.0)
    defensive_role = _defensive_role(player)
    rim_protector = _is_primary_rim_protector(player, team_players)
    base_rate = _foul_rate(player)
    opponent_rim_pressure = _profile_metric(
        playstyle_profiles,
        opponent,
        "offense",
        "rim_pressure",
        50.0,
    )
    opponent_free_throw_pressure = _profile_metric(
        playstyle_profiles,
        opponent,
        "offense",
        "free_throw_pressure",
        50.0,
    )
    physicality = _matchup_physicality(player, matchup_physicality) + _manual_matchup_physicality(
        finals_context,
        player,
    )

    rim_pressure_multiplier = 1.0 + (opponent_rim_pressure - 50.0) / 220.0
    free_throw_multiplier = 1.0 + (opponent_free_throw_pressure - 50.0) / 260.0
    role_multiplier = _role_multiplier(defensive_role)
    # primary_rim_multiplier only fires when the role tag doesn't already encode
    # rim-protector status (e.g. generic "center" or "big" roles). If role_multiplier
    # already applied a rim-protector premium (≥1.20), skip it to avoid double-counting.
    primary_rim_multiplier = 1.18 if (rim_protector and role_multiplier < 1.20) else 1.0
    physicality_multiplier = 1.0 + physicality
    adjusted_rate = base_rate * rim_pressure_multiplier * free_throw_multiplier * role_multiplier
    adjusted_rate *= primary_rim_multiplier * physicality_multiplier
    expected_fouls = adjusted_rate * expected_minutes
    foul_trouble_probability = _foul_trouble_probability(expected_fouls, player)
    risk_score = _score(foul_trouble_probability, 0.12, 0.55)

    return {
        "team": team,
        "player": _player(player),
        "player_key": _player_key(player),
        "expected_minutes": round(expected_minutes, 1),
        "foul_trouble_minutes": round(_clip(expected_minutes - max(6.0, expected_minutes * 0.28), 0.0, expected_minutes), 1),
        "personal_fouls_per_minute": round(base_rate, 4),
        "adjusted_foul_rate": round(adjusted_rate, 4),
        "expected_fouls": round(expected_fouls, 2),
        "foul_trouble_probability": foul_trouble_probability,
        "risk_score": risk_score,
        "risk_label": _risk_label(risk_score),
        "defensive_role": defensive_role,
        "opponent": opponent,
        "opponent_rim_pressure": round(opponent_rim_pressure, 1),
        "opponent_free_throw_pressure": round(opponent_free_throw_pressure, 1),
        "matchup_physicality": round(physicality, 3),
        "primary_rim_protector": rim_protector,
    }


def _risk_label(risk_score: float) -> str:
    if risk_score >= 70.0:
        return "high foul-trouble risk"
    if risk_score >= 40.0:
        return "moderate foul-trouble risk"
    return "low foul-trouble risk"


def build_foul_risk_profiles(
    finals_context: dict[str, Any],
    player_projections: Any = None,
    playstyle_profiles: dict[str, dict[str, Any]] | None = None,
    matchup_physicality: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Estimate foul risk for all Finals rotation players."""
    if player_projections is None:
        player_projections = project_finals_players(finals_context)
    if playstyle_profiles is None:
        playstyle_profiles = build_playstyle_profiles(finals_context)

    players = _enrich_players_with_rotation(player_projections, finals_context)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for team in {row["team"] for row in players}:
        team_players = [row for row in players if row["team"] == team]
        grouped[team] = sorted(
            [
                estimate_player_foul_risk(
                    player,
                    finals_context,
                    team_players,
                    playstyle_profiles,
                    matchup_physicality,
                )
                for player in team_players
            ],
            key=lambda row: (row["risk_score"], row["expected_minutes"]),
            reverse=True,
        )
    return grouped


def _team_probability(
    base_win_probability: float | dict[str, float],
    team: str,
) -> float:
    if isinstance(base_win_probability, dict):
        return _clip(_as_float(base_win_probability.get(team), DEFAULT_BASE_WIN_PROBABILITY), 0.01, 0.99)
    return _clip(_as_float(base_win_probability, DEFAULT_BASE_WIN_PROBABILITY), 0.01, 0.99)


def _player_value_per_minute(player: dict[str, Any], risk_profile: dict[str, Any]) -> float:
    minutes = max(_metric(player, "minutes", risk_profile.get("expected_minutes", 0.0)), 1.0)
    points = _metric(player, "points", 0.0)
    assists = _metric(player, "assists", 0.0)
    rebounds = _metric(player, "rebounds", 0.0)
    steals = _metric(player, "steals", 0.0)
    blocks = _metric(player, "blocks", 0.0)
    turnovers = _metric(player, "turnovers", 0.0)
    usage = _percentage(_first_value(player, PLAYER_ALIASES["usage_rate"]), 20.0)

    box_value = (
        points * 0.08
        + assists * 0.07
        + rebounds * 0.065
        + steals * 0.11
        + blocks * 0.20
        - turnovers * 0.055
    ) / minutes
    role_bonus = 0.055 if risk_profile.get("primary_rim_protector") else 0.0
    rim_pressure_leverage = (
        _clip((risk_profile.get("opponent_rim_pressure", 50.0) - 50.0) / 1000.0, 0.0, 0.035)
        if risk_profile.get("primary_rim_protector")
        else 0.0
    )
    creation_bonus = _clip((usage - 20.0) / 1000.0, -0.01, 0.025)
    starter_bonus = 0.010 if player.get("is_starter") else 0.0
    closer_bonus = 0.010 if player.get("is_closer") else 0.0
    return round(
        _clip(
            box_value + role_bonus + rim_pressure_leverage + creation_bonus + starter_bonus + closer_bonus,
            0.025,
            0.24,
        ),
        4,
    )


def _minute_scenario_probability(
    team_base_probability: float,
    value_per_minute: float,
    baseline_minutes: float,
    scenario_minutes: float,
) -> float:
    minute_delta = scenario_minutes - baseline_minutes
    margin_delta = minute_delta * value_per_minute
    # Convert approximate point/100 impact into probability movement.
    probability = _logistic(_logit(team_base_probability) + margin_delta / 5.5)
    return round(_clip(probability, 0.01, 0.99), 3)


def _find_player_row(players: list[dict[str, Any]], risk_profile: dict[str, Any]) -> dict[str, Any]:
    target_key = risk_profile.get("player_key")
    for player in players:
        if _player_key(player) == target_key:
            return player
    for player in players:
        if _team(player) == risk_profile["team"] and _player(player) == risk_profile["player"]:
            return player
    return {"team": risk_profile["team"], "player": risk_profile["player"], "minutes": risk_profile["expected_minutes"]}


def simulate_player_foul_scenario(
    player: dict[str, Any],
    risk_profile: dict[str, Any],
    base_win_probability: float | dict[str, float] = DEFAULT_BASE_WIN_PROBABILITY,
    normal_minutes: float | None = None,
    foul_trouble_minutes: float | None = None,
) -> dict[str, Any]:
    """Simulate one player's normal-minutes and foul-trouble outcomes."""
    team = risk_profile["team"]
    normal = _as_float(normal_minutes, risk_profile["expected_minutes"])
    trouble = _as_float(foul_trouble_minutes, risk_profile["foul_trouble_minutes"])
    team_base_probability = _team_probability(base_win_probability, team)
    value_per_minute = _player_value_per_minute(player, risk_profile)
    normal_probability = _minute_scenario_probability(
        team_base_probability,
        value_per_minute,
        risk_profile["expected_minutes"],
        normal,
    )
    foul_trouble_probability = _minute_scenario_probability(
        team_base_probability,
        value_per_minute,
        risk_profile["expected_minutes"],
        trouble,
    )
    expected_probability = round(
        normal_probability * (1.0 - risk_profile["foul_trouble_probability"])
        + foul_trouble_probability * risk_profile["foul_trouble_probability"],
        3,
    )

    return {
        "team": team,
        "player": risk_profile["player"],
        "player_key": risk_profile["player_key"],
        "risk_label": risk_profile["risk_label"],
        "foul_trouble_probability": risk_profile["foul_trouble_probability"],
        "value_per_minute": value_per_minute,
        "normal_minutes": round(normal, 1),
        "foul_trouble_minutes": round(trouble, 1),
        "normal_win_probability": normal_probability,
        "foul_trouble_win_probability": foul_trouble_probability,
        "expected_win_probability": expected_probability,
        "win_probability_swing": round(normal_probability - foul_trouble_probability, 3),
        "reason": _scenario_reason(risk_profile, normal, trouble, normal_probability, foul_trouble_probability),
    }


def _scenario_reason(
    risk_profile: dict[str, Any],
    normal_minutes: float,
    foul_trouble_minutes: float,
    normal_probability: float,
    foul_trouble_probability: float,
) -> str:
    role_text = " as the primary rim protector" if risk_profile.get("primary_rim_protector") else ""
    return (
        f"If {risk_profile['player']} reaches {normal_minutes:.1f} minutes, "
        f"{risk_profile['team']} projects around {normal_probability:.0%} win probability. "
        f"If foul trouble cuts that to {foul_trouble_minutes:.1f} minutes, "
        f"the projection falls to {foul_trouble_probability:.0%}{role_text}."
    )


def simulate_foul_trouble_scenarios(
    finals_context: dict[str, Any],
    player_projections: Any = None,
    playstyle_profiles: dict[str, dict[str, Any]] | None = None,
    matchup_physicality: Any = None,
    base_win_probability: float | dict[str, float] = DEFAULT_BASE_WIN_PROBABILITY,
    top_n: int = 6,
) -> dict[str, Any]:
    """Simulate foul-trouble win-probability scenarios for key players."""
    if player_projections is None:
        player_projections = project_finals_players(finals_context)

    players = _enrich_players_with_rotation(player_projections, finals_context)
    risk_profiles = build_foul_risk_profiles(
        finals_context,
        players,
        playstyle_profiles=playstyle_profiles,
        matchup_physicality=matchup_physicality,
    )
    flat_risks = [
        risk
        for team_risks in risk_profiles.values()
        for risk in team_risks
    ]
    ranked_risks = sorted(
        flat_risks,
        key=lambda risk: (
            risk["risk_score"],
            risk["expected_minutes"],
            10.0 if risk["primary_rim_protector"] else 0.0,
        ),
        reverse=True,
    )[:top_n]
    scenarios = [
        simulate_player_foul_scenario(
            _find_player_row(players, risk),
            risk,
            base_win_probability,
        )
        for risk in ranked_risks
    ]

    return {
        "risk_profiles": risk_profiles,
        "scenarios": scenarios,
        "top_risks": ranked_risks,
        "summary": summarize_foul_trouble_scenarios(scenarios),
    }


def summarize_foul_trouble_scenarios(scenarios: list[dict[str, Any]]) -> list[str]:
    """Create readable foul-trouble scenario summaries."""
    return [
        (
            f"{scenario['player']} ({scenario['team']}): "
            f"{scenario['normal_minutes']} minutes = {scenario['normal_win_probability']:.0%}; "
            f"{scenario['foul_trouble_minutes']} foul-trouble minutes = "
            f"{scenario['foul_trouble_win_probability']:.0%} "
            f"({scenario['win_probability_swing']:.1%} swing)."
        )
        for scenario in scenarios
    ]


def foul_trouble_feature_vector(simulation: dict[str, Any]) -> dict[str, float]:
    """Flatten foul-trouble scenarios into numeric model inputs."""
    features: dict[str, float] = {}
    for risk in simulation["top_risks"]:
        prefix = f"{risk['team']}_{risk['player']}".replace(" ", "_")
        features[f"{prefix}_foul_trouble_probability"] = float(risk["foul_trouble_probability"])
        features[f"{prefix}_risk_score"] = float(risk["risk_score"])
        features[f"{prefix}_expected_fouls"] = float(risk["expected_fouls"])
        features[f"{prefix}_adjusted_foul_rate"] = float(risk["adjusted_foul_rate"])
    for scenario in simulation["scenarios"]:
        prefix = f"{scenario['team']}_{scenario['player']}".replace(" ", "_")
        features[f"{prefix}_normal_win_probability"] = float(scenario["normal_win_probability"])
        features[f"{prefix}_foul_trouble_win_probability"] = float(scenario["foul_trouble_win_probability"])
        features[f"{prefix}_win_probability_swing"] = float(scenario["win_probability_swing"])
    return features


if __name__ == "__main__":
    from src.data.build_dataset import build_finals_context

    context = build_finals_context()
    simulation = simulate_foul_trouble_scenarios(context, base_win_probability={"NYK": 0.52, "SAS": 0.48})
    for line in simulation["summary"]:
        print(line)
