"""Post-game update engine for Finals predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isnan
from pathlib import Path
from random import Random
from typing import Any

from src.data.build_dataset import build_finals_context, load_settings
from src.features.adjustment_features import build_adjustment_features
from src.models.predict_game import predict_finals_games
from src.models.simulate_series import (
    _foul_trouble_scenarios,
    build_series_probability_table,
    format_series_result,
    sample_game_probability,
    simulate_series,
)
from src.models.uncertainty import add_game_uncertainty


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


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


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _percentage_label(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def _load_defaults(settings_path: str | Path = DEFAULT_SETTINGS_PATH) -> dict[str, int]:
    try:
        settings = load_settings(settings_path)
    except (FileNotFoundError, ValueError):
        return {
            "simulations": 100000,
            "random_seed": 42,
            "best_of": 7,
        }

    return {
        "simulations": int(settings.get("finals", {}).get("simulations", 100000)),
        "random_seed": int(settings.get("project", {}).get("random_seed", 42)),
        "best_of": int(settings.get("finals", {}).get("best_of", 7)),
    }


def _with_uncertainty(game_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        prediction if "team_a_win_probability_range" in prediction else add_game_uncertainty(prediction)
        for prediction in sorted(game_predictions, key=lambda row: int(row.get("game_number", 0)))
    ]


def _game_number(actual_game: dict[str, Any], predicted_game: dict[str, Any] | None = None) -> int:
    return int(actual_game.get("game_number") or (predicted_game or {}).get("game_number") or 0)


def _find_prediction(
    game_predictions: list[dict[str, Any]],
    game_number: int,
) -> dict[str, Any]:
    for prediction in game_predictions:
        if int(prediction.get("game_number", 0)) == game_number:
            return prediction
    raise ValueError(f"No prediction found for Game {game_number}.")


def _actual_winner_from_features(adjustment_features: dict[str, Any]) -> str:
    winner = adjustment_features.get("result", {}).get("actual_winner")
    if not winner:
        raise ValueError("Actual game data must include enough scoring data to identify a winner.")
    return str(winner)


def _normalize_completed_results(
    previous_results: list[dict[str, Any]] | None,
    actual_game: dict[str, Any],
    adjustment_features: dict[str, Any],
) -> list[dict[str, Any]]:
    completed = []
    for row in previous_results or []:
        if not row:
            continue
        completed.append(
            {
                "game_number": int(row.get("game_number", len(completed) + 1)),
                "winner": str(row.get("winner") or row.get("actual_winner")),
                "actual_scores": row.get("actual_scores") or row.get("score"),
            }
        )

    current_game_number = _game_number(actual_game)
    current_winner = _actual_winner_from_features(adjustment_features)
    completed = [row for row in completed if int(row["game_number"]) != current_game_number]
    completed.append(
        {
            "game_number": current_game_number,
            "winner": current_winner,
            "actual_scores": adjustment_features["result"]["actual_scores"],
        }
    )
    return sorted(completed, key=lambda row: row["game_number"])


def _adjust_probability_range(
    game: dict[str, Any],
    probability: float,
    team_a: str,
    team_b: str,
) -> dict[str, Any]:
    center_before = _clip(_as_float(game.get("team_a_win_probability"), 0.5), 0.01, 0.99)
    range_before = game.get("team_a_win_probability_range") or {}
    low_width = max(center_before - _as_float(range_before.get("low"), center_before - 0.04), 0.01)
    high_width = max(_as_float(range_before.get("high"), center_before + 0.04) - center_before, 0.01)
    low = _clip(probability - low_width, 0.01, probability)
    high = _clip(probability + high_width, probability, 0.99)
    team_a_range = {
        "center": round(probability, 3),
        "low": round(low, 3),
        "high": round(high, 3),
        "label": f"{round(low * 100):.0f}%-{round(high * 100):.0f}%",
    }
    team_b_range = {
        "center": round(1.0 - probability, 3),
        "low": round(1.0 - high, 3),
        "high": round(1.0 - low, 3),
        "label": f"{round((1.0 - high) * 100):.0f}%-{round((1.0 - low) * 100):.0f}%",
    }
    return {
        **game,
        "team_a_win_probability": round(probability, 4),
        "team_b_win_probability": round(1.0 - probability, 4),
        "team_a_win_probability_range": team_a_range,
        "team_b_win_probability_range": team_b_range,
        "realistic_win_probability_range": {
            team_a: team_a_range,
            team_b: team_b_range,
        },
    }


def _adjust_expected_score(
    game: dict[str, Any],
    applied_shift: float,
) -> dict[str, Any]:
    point_delta = int(round(applied_shift * 32.0))
    if point_delta == 0:
        return game
    pace = max(_as_float(game.get("projected_pace"), 98.0), 80.0)
    score_a = max(int(game.get("expected_score_team_a", 0)) + point_delta, 70)
    score_b = max(int(game.get("expected_score_team_b", 0)) - point_delta, 70)
    return {
        **game,
        "expected_score_team_a": score_a,
        "expected_score_team_b": score_b,
        "projected_team_a_off_rating": round(score_a * 100.0 / pace, 1),
        "projected_team_b_off_rating": round(score_b * 100.0 / pace, 1),
    }


def apply_postgame_update_to_predictions(
    game_predictions: list[dict[str, Any]],
    adjustment_features: dict[str, Any],
    completed_results: list[dict[str, Any]] | None = None,
    decay: float = 0.88,
) -> list[dict[str, Any]]:
    """Apply the post-game learning move to future game predictions."""
    result = adjustment_features["result"]
    model_update = adjustment_features["model_update"]
    team_a = result["team_a"]
    team_b = result["team_b"]
    completed_by_game = {
        int(row["game_number"]): row
        for row in completed_results or []
    }
    completed_game_number = int(result["game_number"])
    base_shift = _as_float(model_update.get("team_a_probability_shift"), 0.0)
    updated_predictions = []

    for game in _with_uncertainty(game_predictions):
        game_number = int(game.get("game_number", 0))
        updated = dict(game)
        if game_number in completed_by_game:
            completed = completed_by_game[game_number]
            actual_winner = completed["winner"]
            updated.update(
                {
                    "status": "completed",
                    "actual_winner": actual_winner,
                    "actual_scores": completed.get("actual_scores"),
                    "team_a_actual_result": 1.0 if actual_winner == team_a else 0.0,
                }
            )
        elif game_number > completed_game_number:
            applied_shift = base_shift * (decay ** max(game_number - completed_game_number - 1, 0))
            probability = _clip(_as_float(updated.get("team_a_win_probability"), 0.5) + applied_shift, 0.01, 0.99)
            updated = _adjust_probability_range(updated, probability, team_a, team_b)
            updated = _adjust_expected_score(updated, applied_shift)
            updated["postgame_probability_shift_team_a"] = round(applied_shift, 4)
            updated["postgame_update_reason"] = model_update["recommendation"]
        updated_predictions.append(updated)

    return updated_predictions


def _series_length_distribution(length_counts: Counter[int], simulations: int) -> list[dict[str, Any]]:
    rows = []
    for games in range(4, 8):
        count = length_counts.get(games, 0)
        probability = count / max(simulations, 1)
        rows.append(
            {
                "games": games,
                "count": count,
                "probability": round(probability, 4),
                "percentage": _percentage_label(probability),
            }
        )
    return rows


def simulate_series_after_results(
    game_predictions: list[dict[str, Any]],
    completed_results: list[dict[str, Any]],
    finals_context: dict[str, Any],
    scenario_settings: dict[str, Any] | None = None,
    settings_path: str | Path = DEFAULT_SETTINGS_PATH,
    foul_trouble_simulation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate the rest of a series while honoring completed games exactly."""
    defaults = _load_defaults(settings_path)
    settings = {
        "simulations": defaults["simulations"],
        "random_seed": defaults["random_seed"],
        "best_of": defaults["best_of"],
        **(scenario_settings or {}),
    }
    simulations = int(settings["simulations"])
    best_of = int(settings["best_of"])
    wins_needed = best_of // 2 + 1
    team_a = str(finals_context["team_a"])
    team_b = str(finals_context["team_b"])
    completed_by_game = {int(row["game_number"]): row for row in completed_results}
    completed_wins = {team_a: 0, team_b: 0}
    for row in completed_results:
        winner = str(row["winner"])
        if winner in completed_wins:
            completed_wins[winner] += 1

    remaining_games = [
        game
        for game in _with_uncertainty(game_predictions)
        if int(game.get("game_number", 0)) not in completed_by_game
    ]
    rng = Random(int(settings["random_seed"]))
    if foul_trouble_simulation is None:
        from src.models.foul_trouble_simulator import simulate_foul_trouble_scenarios

        average_probability = sum(
            _as_float(game.get("team_a_win_probability"), 0.5)
            for game in remaining_games
        ) / max(len(remaining_games), 1)
        foul_trouble_simulation = simulate_foul_trouble_scenarios(
            finals_context,
            base_win_probability={
                team_a: average_probability,
                team_b: 1.0 - average_probability,
            },
        )
    foul_scenarios = _foul_trouble_scenarios(foul_trouble_simulation)
    team_series_wins = Counter()
    result_counts = Counter()
    series_length_counts = Counter()
    game_play_counts: dict[int, int] = defaultdict(int)
    game_team_a_wins: dict[int, int] = defaultdict(int)

    for _ in range(simulations):
        wins = dict(completed_wins)
        games_played = len(completed_results)
        for game in remaining_games:
            if max(wins.values()) >= wins_needed:
                break
            sampled = sample_game_probability(
                game,
                team_a,
                team_b,
                rng,
                settings,
                foul_scenarios,
            )
            winner = team_a if rng.random() < sampled["team_a_win_probability"] else team_b
            wins[winner] += 1
            games_played += 1
            game_number = int(game.get("game_number", 0))
            game_play_counts[game_number] += 1
            if winner == team_a:
                game_team_a_wins[game_number] += 1

        winner = team_a if wins[team_a] >= wins_needed or wins[team_a] > wins[team_b] else team_b
        team_series_wins[winner] += 1
        result_counts[format_series_result(winner, games_played)] += 1
        series_length_counts[games_played] += 1

    team_a_probability = team_series_wins[team_a] / max(simulations, 1)
    team_b_probability = team_series_wins[team_b] / max(simulations, 1)
    result_distribution = build_series_probability_table(result_counts, simulations, team_a, team_b)
    return {
        "team_a": team_a,
        "team_b": team_b,
        "best_of": best_of,
        "simulations": simulations,
        "completed_results": completed_results,
        "series_score": {team_a: completed_wins[team_a], team_b: completed_wins[team_b]},
        "team_a_series_win_probability": round(team_a_probability, 4),
        "team_b_series_win_probability": round(team_b_probability, 4),
        "team_a_series_win_percentage": _percentage_label(team_a_probability),
        "team_b_series_win_percentage": _percentage_label(team_b_probability),
        "most_likely_result": max(result_distribution, key=lambda row: row["probability"])["result"],
        "result_distribution": result_distribution,
        "series_length_distribution": _series_length_distribution(series_length_counts, simulations),
        "average_games": round(
            sum(games * count for games, count in series_length_counts.items()) / max(simulations, 1),
            2,
        ),
        "played_game_summary": _played_game_summary(game_play_counts, game_team_a_wins, simulations, team_a, team_b),
    }


def _played_game_summary(
    game_play_counts: dict[int, int],
    game_team_a_wins: dict[int, int],
    simulations: int,
    team_a: str,
    team_b: str,
) -> list[dict[str, Any]]:
    rows = []
    for game_number in sorted(game_play_counts):
        played = game_play_counts[game_number]
        team_a_rate = game_team_a_wins.get(game_number, 0) / max(played, 1)
        rows.append(
            {
                "game_number": game_number,
                "play_probability": round(played / max(simulations, 1), 4),
                f"{team_a}_win_rate_when_played": round(team_a_rate, 4),
                f"{team_b}_win_rate_when_played": round(1.0 - team_a_rate, 4),
            }
        )
    return rows


def _run_bayesian_update(
    context: dict[str, Any],
    base_series: dict[str, Any],
    completed_results: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Run Bayesian series update. Returns None on any failure."""
    try:
        from src.models.bayesian_updater import (
            bayesian_series_win_probability,
            build_game_context,
            prior_mean_from_game_predictions,
        )
        team_a = str(context["team_a"])

        schedule = context.get("schedule", [])
        completed_game_numbers = {int(r["game_number"]) for r in completed_results}
        prior_contexts = []
        for prediction in predictions:
            gn = int(prediction.get("game_number", 0))
            game_sched = next(
                (g for g in schedule if int(g.get("game_number", 0)) == gn),
                prediction,
            )
            prior_contexts.append(build_game_context(game_sched, team_a, context))
        prior_mean = prior_mean_from_game_predictions(predictions, prior_contexts)

        game_results_ctx = []
        for r in completed_results:
            gn = int(r["game_number"])
            game_sched = next(
                (g for g in schedule if int(g.get("game_number", 0)) == gn), {}
            )
            ctx_g = build_game_context(game_sched, team_a, context)
            ctx_g["won"] = str(r["winner"]) == team_a
            game_results_ctx.append(ctx_g)

        remaining_ctx = []
        for game in sorted(predictions, key=lambda g: int(g.get("game_number", 0))):
            gn = int(game.get("game_number", 0))
            if gn in completed_game_numbers:
                continue
            game_sched = next(
                (g for g in schedule if int(g.get("game_number", 0)) == gn), game
            )
            remaining_ctx.append(build_game_context(game_sched, team_a, context))

        bayes_prob = bayesian_series_win_probability(
            prior_mean, game_results_ctx, remaining_ctx
        )
        return {
            "team_a_series_win_probability": bayes_prob,
            "team_b_series_win_probability": round(1.0 - bayes_prob, 4),
            "team_a_series_win_percentage": f"{bayes_prob * 100:.1f}%",
            "team_b_series_win_percentage": f"{(1-bayes_prob) * 100:.1f}%",
            "method": "bayesian_laplace",
        }
    except Exception:
        return None


def update_after_game(
    actual_game: dict[str, Any],
    game_predictions: list[dict[str, Any]] | None = None,
    finals_context: dict[str, Any] | None = None,
    previous_results: list[dict[str, Any]] | None = None,
    scenario_settings: dict[str, Any] | None = None,
    settings_path: str | Path = DEFAULT_SETTINGS_PATH,
    foul_trouble_simulation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update future game and series projections after a completed Finals game."""
    context = finals_context or build_finals_context()
    game_number = int(actual_game.get("game_number") or 1)
    if game_predictions is None:
        from src.models.prediction_snapshot import load_latest_prediction_snapshot
        snapshot = load_latest_prediction_snapshot(game_number)
        game_predictions = snapshot.get("predictions") if snapshot else None
    predictions = _with_uncertainty(game_predictions or predict_finals_games(context))
    predicted_game = _find_prediction(predictions, game_number)
    adjustment_features = build_adjustment_features(predicted_game, actual_game, context)
    completed_results = _normalize_completed_results(previous_results, actual_game, adjustment_features)

    base_series_before_game = simulate_series(
        predictions,
        context,
        scenario_settings=scenario_settings,
        settings_path=settings_path,
        foul_trouble_simulation=foul_trouble_simulation,
    )
    no_learning_after_result = simulate_series_after_results(
        predictions,
        completed_results,
        context,
        scenario_settings=scenario_settings,
        settings_path=settings_path,
        foul_trouble_simulation=foul_trouble_simulation,
    )
    updated_predictions = apply_postgame_update_to_predictions(
        predictions,
        adjustment_features,
        completed_results,
    )
    updated_series = simulate_series_after_results(
        updated_predictions,
        completed_results,
        context,
        scenario_settings=scenario_settings,
        settings_path=settings_path,
        foul_trouble_simulation=foul_trouble_simulation,
    )

    # Bayesian update (runs alongside heuristic; used for reporting).
    # Uses the original pre-update predictions so the prior is derived from
    # the pre-game model output, not the heuristically shifted values.
    bayesian_series = _run_bayesian_update(
        context, base_series_before_game, completed_results, predictions,
    )

    team_a = context["team_a"]
    update_delta = (
        updated_series["team_a_series_win_probability"]
        - no_learning_after_result["team_a_series_win_probability"]
    )

    return {
        "game_number": game_number,
        "team_a": context["team_a"],
        "team_b": context["team_b"],
        "completed_results": completed_results,
        "adjustment_features": adjustment_features,
        "cause_classification": adjustment_features["cause_classification"],
        "primary_cause": adjustment_features["primary_cause"],
        "model_update": adjustment_features["model_update"],
        "updated_game_predictions": updated_predictions,
        "base_series_before_game": base_series_before_game,
        "series_after_result_no_learning": no_learning_after_result,
        "updated_series": updated_series,
        "bayesian_series": bayesian_series,
        "learning_delta_team_a": round(update_delta, 4),
        "learning_delta_team_a_label": f"{update_delta * 100:+.1f} pts",
        "summary": {
            "actual_winner": adjustment_features["result"]["actual_winner"],
            "primary_cause": adjustment_features["model_update"]["primary_cause_label"],
            "update_strength": adjustment_features["model_update"]["update_strength"],
            "recommendation": adjustment_features["model_update"]["recommendation"],
            f"{team_a}_series_before_game": base_series_before_game["team_a_series_win_percentage"],
            f"{team_a}_series_after_result": no_learning_after_result["team_a_series_win_percentage"],
            f"{team_a}_series_after_update": updated_series["team_a_series_win_percentage"],
            f"{team_a}_learning_delta": f"{update_delta * 100:+.1f} pts",
        },
    }


def postgame_update_feature_vector(update_report: dict[str, Any]) -> dict[str, float]:
    """Flatten the post-game update into numeric fields."""
    team_a = update_report["team_a"]
    features = dict(update_report["adjustment_features"]["feature_vector"])
    features[f"{team_a}_series_before_game"] = float(
        update_report["base_series_before_game"]["team_a_series_win_probability"]
    )
    features[f"{team_a}_series_after_result_no_learning"] = float(
        update_report["series_after_result_no_learning"]["team_a_series_win_probability"]
    )
    features[f"{team_a}_series_after_update"] = float(
        update_report["updated_series"]["team_a_series_win_probability"]
    )
    features[f"{team_a}_learning_delta"] = float(update_report["learning_delta_team_a"])
    return features


def _load_game_actuals(game_number: int) -> dict[str, Any]:
    """Build actual_game dict from saved CSV files for game N."""
    import argparse as _argparse
    import pandas as _pd
    from pathlib import Path as _Path

    games_dir = DEFAULT_SETTINGS_PATH.parent.parent / "data" / "processed" / "finals_games"
    trad_path = games_dir / f"game_{game_number}_team_traditional.csv"
    player_path = games_dir / f"game_{game_number}_player_traditional.csv"
    adv_team_path = games_dir / f"game_{game_number}_team_actuals.csv"

    if not trad_path.exists():
        raise FileNotFoundError(
            f"No traditional team stats found at {trad_path}.\n"
            "Run: python -m src.finals_data.update_after_finals_game"
            f" --game {game_number}"
        )

    trad_df = _pd.read_csv(trad_path)
    agg = (
        trad_df.groupby("teamTricode")[
            ["fieldGoalsMade", "fieldGoalsAttempted",
             "threePointersMade", "threePointersAttempted",
             "freeThrowsMade", "freeThrowsAttempted",
             "reboundsOffensive", "reboundsDefensive", "reboundsTotal",
             "assists", "steals", "blocks", "turnovers", "points"]
        ]
        .sum()
        .reset_index()
    )

    team_stats = []
    for _, row in agg.iterrows():
        team_stats.append({
            "team": row["teamTricode"],
            "PTS": int(row["points"]),
            "FGM": int(row["fieldGoalsMade"]),
            "FGA": int(row["fieldGoalsAttempted"]),
            "FG3M": int(row["threePointersMade"]),
            "FG3A": int(row["threePointersAttempted"]),
            "FTM": int(row["freeThrowsMade"]),
            "FTA": int(row["freeThrowsAttempted"]),
            "OREB": int(row["reboundsOffensive"]),
            "DREB": int(row["reboundsDefensive"]),
            "REB": int(row["reboundsTotal"]),
            "AST": int(row["assists"]),
            "STL": int(row["steals"]),
            "BLK": int(row["blocks"]),
            "TOV": int(row["turnovers"]),
        })

    actual_scores = {
        row["team"]: row["PTS"] for row in team_stats
    }

    actual_pace = 99.5
    if adv_team_path.exists():
        adv_df = _pd.read_csv(adv_team_path)
        if "pace" in adv_df.columns and not adv_df.empty:
            actual_pace = float(adv_df.iloc[0]["pace"])

    player_minutes = []
    if player_path.exists():
        players_df = _pd.read_csv(player_path)
        for _, r in players_df.iterrows():
            if str(r.get("minutes", "")).startswith("0:"):
                continue
            full_name = f"{r['firstName']} {r['familyName']}"
            player_minutes.append({
                "team": r["teamTricode"],
                "player": full_name,
                "MIN": str(r["minutes"]),
            })

    return {
        "game_number": game_number,
        "actual_scores": actual_scores,
        "team_stats": team_stats,
        "player_minutes": player_minutes,
        "actual_pace": actual_pace,
    }


if __name__ == "__main__":
    import argparse as _ap
    import json as _json

    parser = _ap.ArgumentParser(description="Post-game model update")
    parser.add_argument(
        "--game", type=int, required=True,
        choices=range(1, 8), help="Game number (1-7)",
    )
    parser.add_argument(
        "--previous-games", type=int, nargs="*",
        help="Prior completed game numbers (for series context)",
    )
    args = parser.parse_args()

    context = build_finals_context()
    actual_game = _load_game_actuals(args.game)

    previous_results: list[dict[str, Any]] | None = None
    if args.previous_games:
        previous_results = []
        for prev_n in args.previous_games:
            prev = _load_game_actuals(prev_n)
            winner = max(prev["actual_scores"], key=prev["actual_scores"].get)
            previous_results.append({
                "game_number": prev_n,
                "winner": winner,
                "actual_scores": prev["actual_scores"],
            })

    report = update_after_game(
        actual_game,
        finals_context=context,
        previous_results=previous_results,
        scenario_settings={"simulations": 100000, "random_seed": 42},
    )
    summary = report["summary"]
    team_a = context["team_a"]
    team_b = context["team_b"]

    print(f"\n{'='*60}")
    print(f"Game {report['game_number']} Post-Game Update")
    print(f"{'='*60}")
    scores = actual_game["actual_scores"]
    score_str = "  ".join(f"{t}: {s}" for t, s in scores.items())
    print(f"Final score : {score_str}")
    print(f"Winner      : {summary['actual_winner']}")
    print(f"Primary cause : {summary['primary_cause']}")
    print(f"Update strength : {summary['update_strength']}")
    print(f"Recommendation : {summary['recommendation']}")

    print(f"\n--- {team_a} series win probability ---")
    print(f"  Before game  : {summary[f'{team_a}_series_before_game']}")
    print(f"  After result : {summary[f'{team_a}_series_after_result']}")
    print(f"  After update : {summary[f'{team_a}_series_after_update']}")
    print(f"  Learning delta : {report['learning_delta_team_a_label']}")

    series_score = report["updated_series"]["series_score"]
    print(f"\n--- Series score ---")
    for team, wins in series_score.items():
        print(f"  {team}: {wins}")

    dist = report["updated_series"]["result_distribution"]
    print(f"\n--- Updated result distribution ---")
    for row in dist:
        print(f"  {row['result']:12s}  {row['percentage']}")

    print(f"\n{'='*60}")
    print("Next steps:")
    print("  python -m src.models.predict_game")
    print("  python -m src.models.simulate_series")
