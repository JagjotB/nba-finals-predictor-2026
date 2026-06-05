"""Close-game edge model for Finals closing lineups."""

from __future__ import annotations

from typing import Any

from src.features.clutch_features import CLUTCH_FEATURES, build_clutch_features


POSITIVE_WEIGHTS = {
    "late_game_shot_creation": 0.22,
    "isolation_scoring": 0.10,
    "free_throw_reliability": 0.13,
    "rim_protection": 0.13,
    "switchability": 0.11,
    "rebounding": 0.10,
    "star_creation": 0.16,
}

NEGATIVE_WEIGHTS = {
    "turnover_risk": 0.09,
    "foul_risk": 0.06,
}

CLOSE_GAME_EDGE_SCALE = 8.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _feature_score(profile: dict[str, Any], feature: str) -> float:
    return float(profile.get("features", {}).get(feature, {}).get("score", 50.0))


def score_closing_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Score one team's close-game closing profile."""
    positive = sum(_feature_score(profile, feature) * weight for feature, weight in POSITIVE_WEIGHTS.items())
    negative = sum(_feature_score(profile, feature) * weight for feature, weight in NEGATIVE_WEIGHTS.items())
    total_weight = sum(POSITIVE_WEIGHTS.values()) + sum(NEGATIVE_WEIGHTS.values())
    raw_score = (positive - negative + 50.0 * sum(NEGATIVE_WEIGHTS.values())) / total_weight

    lineup_net = float(profile.get("closing_lineup_adjusted_net_rating", 0.0))
    lineup_bonus = _clip(lineup_net, -8.0, 8.0) * 0.75
    close_game_score = _clip(raw_score + lineup_bonus, 0.0, 100.0)

    return {
        "team": profile["team"],
        "closing_five": profile["closing_five"],
        "close_game_score": round(close_game_score, 2),
        "lineup_net_bonus": round(lineup_bonus, 2),
        "feature_scores": {
            feature: _feature_score(profile, feature)
            for feature in CLUTCH_FEATURES
        },
    }


def _strengths_and_concerns(profile: dict[str, Any]) -> tuple[list[str], list[str]]:
    feature_scores = profile["feature_scores"]
    strengths = [
        feature for feature, score in feature_scores.items()
        if score >= 62.0 and feature not in NEGATIVE_WEIGHTS
    ]
    concerns = [
        feature for feature, score in feature_scores.items()
        if (score <= 38.0 and feature not in NEGATIVE_WEIGHTS)
        or (score >= 62.0 and feature in NEGATIVE_WEIGHTS)
    ]
    strengths = sorted(strengths, key=lambda feature: feature_scores[feature], reverse=True)
    concerns = sorted(
        concerns,
        key=lambda feature: abs(feature_scores[feature] - 50.0),
        reverse=True,
    )
    return strengths[:3], concerns[:2]


def _feature_phrase(feature: str) -> str:
    return {
        "late_game_shot_creation": "late-clock shot creation",
        "isolation_scoring": "isolation scoring",
        "free_throw_reliability": "free throw reliability",
        "turnover_risk": "turnover risk",
        "rim_protection": "rim protection",
        "switchability": "defensive versatility",
        "rebounding": "rebounding",
        "star_creation": "star creation",
        "foul_risk": "foul risk",
    }[feature]


def _join_phrases(items: list[str]) -> str:
    phrases = [_feature_phrase(item) for item in items]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"


def explain_close_game_edge(
    leading_profile: dict[str, Any],
    trailing_profile: dict[str, Any],
    edge: float,
) -> str:
    """Generate a concise basketball explanation for a close-game edge."""
    leading_strengths, leading_concerns = _strengths_and_concerns(leading_profile)
    trailing_strengths, _ = _strengths_and_concerns(trailing_profile)

    if abs(edge) < 0.25:
        return (
            f"The close-game edge is essentially neutral: {leading_profile['team']} and "
            f"{trailing_profile['team']} have similar closing-lineup profiles."
        )

    leading_text = _join_phrases(leading_strengths) or "a cleaner closing profile"
    trailing_text = _join_phrases(trailing_strengths) or "enough counters to keep it close"
    concern_text = _join_phrases(leading_concerns)

    if concern_text:
        return (
            f"{leading_profile['team']} has the close-game edge behind {leading_text}, "
            f"but {trailing_profile['team']} can answer with {trailing_text}. "
            f"The main risk for {leading_profile['team']} is {concern_text}."
        )
    return (
        f"{leading_profile['team']} has the close-game edge behind {leading_text}, "
        f"while {trailing_profile['team']}'s best counter is {trailing_text}."
    )


def predict_close_game_edge(
    finals_context: dict[str, Any],
    clutch_features: dict[str, dict[str, Any]] | None = None,
    player_projections: Any = None,
    player_archetypes: Any = None,
    lineup_features: dict[str, list[dict[str, Any]]] | None = None,
    lineup_stats: Any = None,
    team_ratings: Any = None,
) -> dict[str, Any]:
    """Predict close-game edge per 100 possessions for the Finals matchup."""
    if clutch_features is None:
        clutch_features = build_clutch_features(
            finals_context,
            player_projections=player_projections,
            player_archetypes=player_archetypes,
            lineup_features=lineup_features,
            lineup_stats=lineup_stats,
            team_ratings=team_ratings,
        )

    team_a = str(finals_context.get("team_a") or "").strip()
    team_b = str(finals_context.get("team_b") or "").strip()
    scores = {
        team: score_closing_profile(profile)
        for team, profile in clutch_features.items()
    }
    score_delta = scores[team_a]["close_game_score"] - scores[team_b]["close_game_score"]
    team_a_edge = round(score_delta / 100.0 * CLOSE_GAME_EDGE_SCALE, 2)
    team_b_edge = round(-team_a_edge, 2)

    leading_team = team_a if team_a_edge >= 0 else team_b
    trailing_team = team_b if leading_team == team_a else team_a
    leading_profile = scores[leading_team]
    trailing_profile = scores[trailing_team]
    leading_edge = team_a_edge if leading_team == team_a else team_b_edge

    return {
        "team_a": team_a,
        "team_b": team_b,
        "close_game_edges_per_100": {
            team_a: team_a_edge,
            team_b: team_b_edge,
        },
        "favorite": leading_team if abs(leading_edge) >= 0.25 else "Even",
        "favorite_edge_per_100": round(abs(leading_edge), 2),
        "team_scores": scores,
        "reason": explain_close_game_edge(leading_profile, trailing_profile, leading_edge),
    }


def closing_lineup_feature_vector(prediction: dict[str, Any]) -> dict[str, float]:
    """Flatten close-game model output into numeric features."""
    features = {}
    for team, score in prediction["team_scores"].items():
        features[f"{team}_close_game_score"] = float(score["close_game_score"])
        features[f"{team}_lineup_net_bonus"] = float(score["lineup_net_bonus"])
        for feature, value in score["feature_scores"].items():
            features[f"{team}_{feature}"] = float(value)
    for team, edge in prediction["close_game_edges_per_100"].items():
        features[f"{team}_close_game_edge_per_100"] = float(edge)
    return features


if __name__ == "__main__":
    from src.data.build_dataset import build_finals_context

    context = build_finals_context()
    prediction = predict_close_game_edge(context)
    for team, edge in prediction["close_game_edges_per_100"].items():
        print(f"{team}: {edge:+.2f} per 100 possessions")
    print(prediction["reason"])
