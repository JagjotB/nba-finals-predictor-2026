"""Matchup edge features for Finals team style comparisons."""

from __future__ import annotations

from typing import Any

from src.features.playstyle_features import build_playstyle_profiles


EDGE_LABELS = {
    -2: "strong disadvantage",
    -1: "slight disadvantage",
    0: "neutral",
    1: "slight advantage",
    2: "strong advantage",
}

EDGE_DEFINITIONS = [
    {
        "key": "rim_pressure_vs_rim_protection",
        "name": "rim pressure vs rim protection",
        "offense_metric": "rim_pressure",
        "defense_metrics": {"rim_protection": 1.0},
        "offense_trait": "rim pressure",
        "defense_trait": "rim protection",
        "weight": 1.20,
    },
    {
        "key": "offensive_rebounding_vs_defensive_rebounding",
        "name": "offensive rebounding vs defensive rebounding",
        "offense_metric": "offensive_rebounding",
        "defense_metrics": {"defensive_rebounding": 1.0},
        "offense_trait": "offensive rebounding",
        "defense_trait": "defensive rebounding",
        "weight": 1.15,
    },
    {
        "key": "transition_offense_vs_transition_defense",
        "name": "transition offense vs transition defense",
        "offense_metric": "transition_frequency",
        "defense_metrics": {"transition_defense": 1.0},
        "offense_trait": "transition frequency",
        "defense_trait": "transition defense",
        "weight": 1.00,
    },
    {
        "key": "pick_and_roll_vs_screen_defense",
        "name": "pick-and-roll offense vs screen defense",
        "offense_metric": "pick_and_roll_usage",
        "defense_metrics": {
            "point_of_attack_defense": 0.40,
            "drop_coverage_strength": 0.35,
            "switchability": 0.25,
        },
        "offense_trait": "pick-and-roll usage",
        "defense_trait": "screen defense",
        "weight": 1.15,
    },
    {
        "key": "isolation_vs_perimeter_defense",
        "name": "isolation creation vs perimeter defense",
        "offense_metric": "isolation_usage",
        "defense_metrics": {
            "point_of_attack_defense": 0.60,
            "switchability": 0.40,
        },
        "offense_trait": "isolation creation",
        "defense_trait": "perimeter defense",
        "weight": 1.00,
    },
    {
        "key": "post_up_vs_interior_defense",
        "name": "post-up offense vs interior defense",
        "offense_metric": "post_up_usage",
        "defense_metrics": {
            "rim_protection": 0.55,
            "drop_coverage_strength": 0.45,
        },
        "offense_trait": "post-up usage",
        "defense_trait": "interior defense",
        "weight": 0.80,
    },
    {
        "key": "corner_3_vs_corner_3_prevention",
        "name": "corner three volume vs corner three prevention",
        "offense_metric": "corner_3_frequency",
        "defense_metrics": {"corner_3_prevention": 1.0},
        "offense_trait": "corner three volume",
        "defense_trait": "corner three prevention",
        "weight": 0.95,
    },
    {
        "key": "above_break_3_vs_perimeter_defense",
        "name": "above-the-break threes vs perimeter defense",
        "offense_metric": "above_the_break_3_frequency",
        "defense_metrics": {
            "point_of_attack_defense": 0.50,
            "switchability": 0.50,
        },
        "offense_trait": "above-the-break three volume",
        "defense_trait": "perimeter defense",
        "weight": 0.85,
    },
    {
        "key": "free_throw_pressure_vs_foul_discipline",
        "name": "free throw pressure vs foul discipline",
        "offense_metric": "free_throw_pressure",
        "defense_metrics": {"foul_discipline": 1.0},
        "offense_trait": "free throw pressure",
        "defense_trait": "foul discipline",
        "weight": 1.00,
    },
    {
        "key": "turnover_risk_vs_forced_turnovers",
        "name": "ball security vs forced turnovers",
        "offense_metric": "turnover_risk",
        "defense_metrics": {"forced_turnovers": 1.0},
        "offense_trait": "ball security",
        "defense_trait": "forced turnovers",
        "weight": 1.05,
        "invert_offense": True,
    },
    {
        "key": "half_court_offense_vs_set_defense",
        "name": "half-court offense vs set defense",
        "offense_metric": "half_court_reliance",
        "defense_metrics": {
            "point_of_attack_defense": 0.35,
            "switchability": 0.30,
            "rim_protection": 0.35,
        },
        "offense_trait": "half-court creation",
        "defense_trait": "set defense",
        "weight": 0.90,
    },
    {
        "key": "pace_pressure_vs_transition_defense",
        "name": "pace pressure vs transition defense",
        "offense_metric": "pace",
        "defense_metrics": {"transition_defense": 1.0},
        "offense_trait": "pace pressure",
        "defense_trait": "transition defense",
        "weight": 0.70,
    },
]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _metric(profile: dict[str, Any], side: str, metric: str) -> dict[str, Any]:
    return profile.get(side, {}).get("metrics", {}).get(
        metric,
        {"score": 50.0, "label": metric.replace("_", " "), "value": 50.0},
    )


def _score(profile: dict[str, Any], side: str, metric: str) -> float:
    return float(_metric(profile, side, metric).get("score", 50.0))


def _label(profile: dict[str, Any], side: str, metric: str) -> str:
    return str(_metric(profile, side, metric).get("label", metric.replace("_", " ")))


def _weighted_defensive_score(
    defensive_profile: dict[str, Any],
    defensive_metrics: dict[str, float],
) -> float:
    total_weight = sum(defensive_metrics.values())
    if total_weight <= 0:
        return 50.0
    weighted = sum(
        _score(defensive_profile, "defense", metric) * weight
        for metric, weight in defensive_metrics.items()
    )
    return weighted / total_weight


def _weighted_defensive_label(
    defensive_profile: dict[str, Any],
    defensive_metrics: dict[str, float],
    defense_trait: str,
    defense_score: float,
) -> str:
    if len(defensive_metrics) > 1:
        if defense_score >= 65.0:
            return f"strong {defense_trait}"
        if defense_score <= 35.0:
            return f"weaker {defense_trait}"
        return f"solid {defense_trait}"

    strongest_metric = max(defensive_metrics, key=defensive_metrics.get)
    return _label(defensive_profile, "defense", strongest_metric)


def _edge_from_raw(raw_edge: float) -> int:
    if raw_edge >= 1.25:
        return 2
    if raw_edge >= 0.40:
        return 1
    if raw_edge <= -1.25:
        return -2
    if raw_edge <= -0.40:
        return -1
    return 0


def _advantage_phrase(score: int) -> str:
    return EDGE_LABELS.get(score, "neutral")


def _explain_edge(
    offensive_team: str,
    defensive_team: str,
    definition: dict[str, Any],
    edge_score: int,
    offense_score: float,
    defense_score: float,
    offense_label: str,
    defense_label: str,
) -> str:
    name = definition["name"]
    offense_trait = definition["offense_trait"]
    defense_trait = definition["defense_trait"]
    edge_text = _advantage_phrase(edge_score)

    if edge_score > 0:
        return (
            f"{offensive_team} has a {edge_text} in {name} because its "
            f"{offense_trait} rates as {offense_label} ({offense_score:.1f}/100), "
            f"while {defensive_team}'s {defense_trait} grades as {defense_label} "
            f"({defense_score:.1f}/100)."
        )
    if edge_score < 0:
        return (
            f"{offensive_team} faces a {edge_text} in {name} because its "
            f"{offense_trait} rates as {offense_label} ({offense_score:.1f}/100), "
            f"and {defensive_team}'s {defense_trait} counters it with "
            f"{defense_label} ({defense_score:.1f}/100)."
        )
    return (
        f"{offensive_team} vs {defensive_team} looks neutral in {name}: "
        f"{offensive_team}'s {offense_trait} ({offense_score:.1f}/100) is closely "
        f"matched by {defensive_team}'s {defense_trait} ({defense_score:.1f}/100)."
    )


def _manual_matchup_notes(
    finals_context: dict[str, Any],
    offensive_team: str,
    defensive_team: str,
    edge_key: str,
) -> list[str]:
    matchup_type_by_edge = {
        "pick_and_roll_vs_screen_defense": "pick_and_roll",
        "isolation_vs_perimeter_defense": "isolation",
        "rim_pressure_vs_rim_protection": "rim",
        "post_up_vs_interior_defense": "post",
    }
    target_type = matchup_type_by_edge.get(edge_key)
    if not target_type:
        return []

    notes: list[str] = []
    for player in finals_context.get("active_players", {}).get(offensive_team, []):
        for role in player.get("matchup_roles", []):
            if role.get("side") != "offense":
                continue
            if role.get("opponent") != defensive_team:
                continue
            matchup_type = str(role.get("matchup_type") or "")
            if target_type in matchup_type:
                primary = role.get("primary_defender")
                if primary:
                    notes.append(
                        f"Manual matchup context: {player.get('player')} is expected to see {primary}."
                    )
    return notes


def evaluate_matchup_edge(
    offensive_team: str,
    defensive_team: str,
    offensive_profile: dict[str, Any],
    defensive_profile: dict[str, Any],
    definition: dict[str, Any],
    finals_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one offensive pressure point against the defensive answer."""
    offense_metric = definition["offense_metric"]
    offense_score = _score(offensive_profile, "offense", offense_metric)
    if definition.get("invert_offense"):
        offense_score = 100.0 - offense_score

    defense_score = _weighted_defensive_score(defensive_profile, definition["defense_metrics"])
    raw_edge = ((offense_score - defense_score) / 25.0) * float(definition.get("weight", 1.0))
    edge_score = _edge_from_raw(raw_edge)

    offense_label = _label(offensive_profile, "offense", offense_metric)
    if definition.get("invert_offense"):
        offense_label = "low turnover risk" if offense_score >= 65 else "shaky ball security"
    defense_label = _weighted_defensive_label(
        defensive_profile,
        definition["defense_metrics"],
        definition["defense_trait"],
        defense_score,
    )
    explanation = _explain_edge(
        offensive_team,
        defensive_team,
        definition,
        edge_score,
        offense_score,
        defense_score,
        offense_label,
        defense_label,
    )

    manual_notes = _manual_matchup_notes(
        finals_context or {},
        offensive_team,
        defensive_team,
        definition["key"],
    )
    if manual_notes:
        explanation = f"{explanation} {' '.join(manual_notes)}"

    return {
        "key": definition["key"],
        "name": definition["name"],
        "offensive_team": offensive_team,
        "defensive_team": defensive_team,
        "offense_metric": offense_metric,
        "defense_metrics": dict(definition["defense_metrics"]),
        "offense_score": round(offense_score, 1),
        "defense_score": round(defense_score, 1),
        "raw_edge": round(raw_edge, 3),
        "matchup_score": edge_score,
        "advantage": EDGE_LABELS[edge_score],
        "explanation": explanation,
    }


def compare_offense_to_defense(
    offensive_team: str,
    defensive_team: str,
    playstyle_profiles: dict[str, dict[str, Any]],
    finals_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare one team's offense against the opponent's defense."""
    offensive_profile = playstyle_profiles[offensive_team]
    defensive_profile = playstyle_profiles[defensive_team]
    edges = [
        evaluate_matchup_edge(
            offensive_team,
            defensive_team,
            offensive_profile,
            defensive_profile,
            definition,
            finals_context,
        )
        for definition in EDGE_DEFINITIONS
    ]

    total_weight = sum(float(definition.get("weight", 1.0)) for definition in EDGE_DEFINITIONS)
    weighted_edge = sum(
        edge["matchup_score"] * float(definition.get("weight", 1.0))
        for edge, definition in zip(edges, EDGE_DEFINITIONS)
    ) / total_weight
    overall_edge = _edge_from_raw(weighted_edge)

    return {
        "offensive_team": offensive_team,
        "defensive_team": defensive_team,
        "overall_edge": overall_edge,
        "overall_edge_label": EDGE_LABELS[overall_edge],
        "average_matchup_score": round(weighted_edge, 3),
        "edges": edges,
        "top_edges": top_matchup_edges(edges),
    }


def top_matchup_edges(edges: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Return the highest-signal non-neutral matchup edges."""
    ranked = sorted(
        edges,
        key=lambda edge: (abs(edge["matchup_score"]), abs(edge["raw_edge"])),
        reverse=True,
    )
    non_neutral = [edge for edge in ranked if edge["matchup_score"] != 0]
    return (non_neutral or ranked)[:limit]


def build_matchup_edges(
    finals_context: dict[str, Any],
    playstyle_profiles: dict[str, dict[str, Any]] | None = None,
    team_stats: Any = None,
    offensive_stats: Any = None,
    defensive_stats: Any = None,
    manual_overrides: Any = None,
) -> dict[str, Any]:
    """Build both Finals team-vs-team matchup comparisons."""
    if playstyle_profiles is None:
        playstyle_profiles = build_playstyle_profiles(
            finals_context,
            team_stats=team_stats,
            offensive_stats=offensive_stats,
            defensive_stats=defensive_stats,
            manual_overrides=manual_overrides,
        )

    team_a = str(finals_context.get("team_a") or "").strip()
    team_b = str(finals_context.get("team_b") or "").strip()
    if not team_a or not team_b:
        raise ValueError("finals_context must include team_a and team_b.")

    comparisons = {
        f"{team_a}_offense_vs_{team_b}_defense": compare_offense_to_defense(
            team_a,
            team_b,
            playstyle_profiles,
            finals_context,
        ),
        f"{team_b}_offense_vs_{team_a}_defense": compare_offense_to_defense(
            team_b,
            team_a,
            playstyle_profiles,
            finals_context,
        ),
    }

    team_edges = {
        team_a: comparisons[f"{team_a}_offense_vs_{team_b}_defense"]["overall_edge"],
        team_b: comparisons[f"{team_b}_offense_vs_{team_a}_defense"]["overall_edge"],
    }

    return {
        "team_a": team_a,
        "team_b": team_b,
        "comparisons": comparisons,
        "team_edges": team_edges,
        "top_series_edges": top_series_edges(comparisons),
    }


def top_series_edges(
    comparisons: dict[str, dict[str, Any]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return the highest-signal matchup edges across both teams."""
    all_edges = [
        edge
        for comparison in comparisons.values()
        for edge in comparison["edges"]
    ]
    return top_matchup_edges(all_edges, limit)


def matchup_feature_vector(matchup_edges: dict[str, Any]) -> dict[str, float]:
    """Flatten matchup edges into numeric model features."""
    features: dict[str, float] = {}
    for comparison_key, comparison in matchup_edges["comparisons"].items():
        features[f"{comparison_key}_overall_edge"] = float(comparison["overall_edge"])
        features[f"{comparison_key}_average_matchup_score"] = float(comparison["average_matchup_score"])
        for edge in comparison["edges"]:
            features[f"{comparison_key}_{edge['key']}"] = float(edge["matchup_score"])
            features[f"{comparison_key}_{edge['key']}_raw"] = float(edge["raw_edge"])
    return features


def summarize_matchup_edges(
    matchup_edges: dict[str, Any],
    limit: int = 5,
) -> dict[str, list[str]]:
    """Return readable matchup explanations grouped by comparison."""
    return {
        comparison_key: [
            edge["explanation"]
            for edge in comparison["top_edges"][:limit]
        ]
        for comparison_key, comparison in matchup_edges["comparisons"].items()
    }


if __name__ == "__main__":
    from src.data.build_dataset import build_finals_context

    context = build_finals_context()
    matchup_edges = build_matchup_edges(context)
    for comparison_key, explanations in summarize_matchup_edges(matchup_edges).items():
        print(comparison_key)
        for explanation in explanations:
            print(f"  {explanation}")
