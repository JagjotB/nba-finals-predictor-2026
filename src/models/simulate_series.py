"""Best-of-7 NBA Finals series simulator.

Runs Monte Carlo simulations from game-level win probabilities, uncertainty
ranges, schedule context, foul-trouble risk, and optional scenario settings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import exp, isnan, log
from pathlib import Path
from random import Random
from typing import Any

from src.data.build_dataset import build_finals_context, load_settings
from src.models.predict_game import predict_finals_games
from src.models.uncertainty import add_game_uncertainty


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

DEFAULT_SCENARIO_SETTINGS = {
    "use_uncertainty": True,
    "uncertainty_scale": 1.0,
    "use_foul_trouble": True,
    "foul_trouble_scale": 0.50,
    "foul_trouble_probability_scale": 1.0,
    "team_a_probability_shift": 0.0,
    "team_b_probability_shift": 0.0,
    "game_probability_overrides": {},
    "shared_strength_sigma_logit": 0.12,
    "series_availability_scenarios": [],
}


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


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _percentage_label(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def _load_simulation_defaults(settings_path: str | Path = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    try:
        settings = load_settings(settings_path)
    except (FileNotFoundError, ValueError):
        return {
            "best_of": 7,
            "simulations": 100000,
            "random_seed": 42,
        }

    finals = settings.get("finals", {})
    project = settings.get("project", {})
    return {
        "best_of": int(finals.get("best_of", 7)),
        "simulations": int(finals.get("simulations", 100000)),
        "random_seed": int(project.get("random_seed", 42)),
    }


def _scenario_settings(scenario_settings: dict[str, Any] | None) -> dict[str, Any]:
    return {**DEFAULT_SCENARIO_SETTINGS, **(scenario_settings or {})}


def _team_names(
    game_predictions: list[dict[str, Any]],
    finals_context: dict[str, Any] | None,
) -> tuple[str, str]:
    if game_predictions:
        first_game = game_predictions[0]
        return str(first_game["team_a"]), str(first_game["team_b"])
    if finals_context:
        return str(finals_context["team_a"]), str(finals_context["team_b"])
    return "Team A", "Team B"


def _wins_needed(best_of: int) -> int:
    return best_of // 2 + 1


def _game_sort_key(game_prediction: dict[str, Any]) -> int:
    return int(game_prediction.get("game_number", 0))


def _prepare_game_predictions(
    game_predictions: list[dict[str, Any]] | None,
    finals_context: dict[str, Any] | None,
    use_uncertainty: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = finals_context or build_finals_context()
    predictions = game_predictions or predict_finals_games(context)
    prepared = []
    for prediction in sorted(predictions, key=_game_sort_key):
        if use_uncertainty and "team_a_win_probability_range" not in prediction:
            prepared.append(add_game_uncertainty(prediction))
        else:
            prepared.append(dict(prediction))
    return context, prepared


def _range_for_team_a(game_prediction: dict[str, Any]) -> tuple[float, float, float]:
    center = _clip(_as_float(game_prediction.get("team_a_win_probability"), 0.5), 0.01, 0.99)
    probability_range = game_prediction.get("team_a_win_probability_range") or {}
    low = _clip(_as_float(probability_range.get("low"), center), 0.01, 0.99)
    high = _clip(_as_float(probability_range.get("high"), center), 0.01, 0.99)
    return min(low, center), center, max(high, center)


def _override_probability(
    game_prediction: dict[str, Any],
    scenario_settings: dict[str, Any],
    current_probability: float,
) -> float:
    overrides = scenario_settings.get("game_probability_overrides") or {}
    game_number = str(game_prediction.get("game_number"))
    override = overrides.get(game_number, overrides.get(int(game_prediction.get("game_number", 0)), None))
    if override is None:
        return current_probability
    if isinstance(override, dict):
        return _clip(_as_float(override.get("team_a_win_probability"), current_probability), 0.01, 0.99)
    return _clip(_as_float(override, current_probability), 0.01, 0.99)


def _sample_base_probability(
    game_prediction: dict[str, Any],
    rng: Random,
    scenario_settings: dict[str, Any],
    shared_logit_shift: float = 0.0,
) -> float:
    low, center, high = _range_for_team_a(game_prediction)
    uncertainty_scale = _clip(_as_float(scenario_settings.get("uncertainty_scale"), 1.0), 0.0, 3.0)
    if _as_bool(scenario_settings.get("use_uncertainty"), True) and uncertainty_scale > 0:
        scaled_low = _clip(center - (center - low) * uncertainty_scale, 0.01, 0.99)
        scaled_high = _clip(center + (high - center) * uncertainty_scale, 0.01, 0.99)
        probability = rng.triangular(scaled_low, scaled_high, center)
    else:
        probability = center

    probability = _override_probability(game_prediction, scenario_settings, probability)
    probability += _as_float(scenario_settings.get("team_a_probability_shift"), 0.0)
    probability -= _as_float(scenario_settings.get("team_b_probability_shift"), 0.0)
    probability = _clip(probability, 0.01, 0.99)
    probability = 1.0 / (
        1.0 + exp(-(log(probability / (1.0 - probability)) + shared_logit_shift))
    )
    return _clip(probability, 0.01, 0.99)


def _series_availability_shift(
    rng: Random,
    scenario_settings: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    shift = 0.0
    triggered = []
    for scenario in scenario_settings.get("series_availability_scenarios") or []:
        probability = _clip(_as_float(scenario.get("probability"), 0.0), 0.0, 1.0)
        if rng.random() >= probability:
            continue
        scenario_shift = _as_float(scenario.get("team_a_probability_shift"), 0.0)
        shift += scenario_shift
        triggered.append({
            "player": scenario.get("player"),
            "team": scenario.get("team"),
            "team_a_probability_shift": round(scenario_shift, 4),
        })
    return shift, triggered


def _foul_trouble_scenarios(foul_trouble_simulation: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not foul_trouble_simulation:
        return []
    return [
        dict(scenario)
        for scenario in foul_trouble_simulation.get("scenarios", [])
        if "team" in scenario and "foul_trouble_probability" in scenario
    ]


def _apply_foul_trouble_events(
    probability: float,
    team_a: str,
    team_b: str,
    scenarios: list[dict[str, Any]],
    rng: Random,
    scenario_settings: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    if not _as_bool(scenario_settings.get("use_foul_trouble"), True):
        return probability, []

    impact_scale = _clip(_as_float(scenario_settings.get("foul_trouble_scale"), 0.50), 0.0, 2.0)
    probability_scale = _clip(_as_float(scenario_settings.get("foul_trouble_probability_scale"), 1.0), 0.0, 2.0)
    if impact_scale <= 0 or probability_scale <= 0:
        return probability, []

    triggered = []
    adjusted_probability = probability
    for scenario in scenarios:
        event_probability = _clip(
            _as_float(scenario.get("foul_trouble_probability"), 0.0) * probability_scale,
            0.0,
            0.95,
        )
        if rng.random() >= event_probability:
            continue

        swing = abs(_as_float(scenario.get("win_probability_swing"), 0.0)) * impact_scale
        team = str(scenario.get("team") or "")
        if team == team_a:
            adjusted_probability -= swing
        elif team == team_b:
            adjusted_probability += swing
        else:
            continue

        triggered.append(
            {
                "team": team,
                "player": scenario.get("player"),
                "probability_swing": round(swing, 4),
            }
        )

    return _clip(adjusted_probability, 0.01, 0.99), triggered


def sample_game_probability(
    game_prediction: dict[str, Any],
    team_a: str,
    team_b: str,
    rng: Random,
    scenario_settings: dict[str, Any] | None = None,
    foul_trouble_scenarios: list[dict[str, Any]] | None = None,
    shared_logit_shift: float = 0.0,
) -> dict[str, Any]:
    """Sample one game's Team A win probability for a series simulation."""
    settings = _scenario_settings(scenario_settings)
    probability = _sample_base_probability(
        game_prediction,
        rng,
        settings,
        shared_logit_shift=shared_logit_shift,
    )
    probability, triggered_foul_events = _apply_foul_trouble_events(
        probability,
        team_a,
        team_b,
        foul_trouble_scenarios or [],
        rng,
        settings,
    )
    return {
        "game_number": int(game_prediction.get("game_number", 0)),
        "team_a_win_probability": round(probability, 4),
        "triggered_foul_events": triggered_foul_events,
    }


def simulate_one_series(
    game_predictions: list[dict[str, Any]],
    team_a: str,
    team_b: str,
    rng: Random,
    best_of: int = 7,
    scenario_settings: dict[str, Any] | None = None,
    foul_trouble_scenarios: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Simulate one best-of series and stop when a team reaches four wins."""
    needed = _wins_needed(best_of)
    wins = {team_a: 0, team_b: 0}
    games_played = []
    settings = _scenario_settings(scenario_settings)
    shared_sigma = _clip(
        _as_float(settings.get("shared_strength_sigma_logit"), 0.12),
        0.0,
        0.75,
    )
    shared_logit_shift = rng.gauss(0.0, shared_sigma) if shared_sigma > 0 else 0.0
    availability_probability_shift, availability_events = _series_availability_shift(
        rng,
        settings,
    )
    if availability_probability_shift:
        center = 0.5 + availability_probability_shift
        center = _clip(center, 0.01, 0.99)
        shared_logit_shift += log(center / (1.0 - center))

    for game_prediction in sorted(game_predictions, key=_game_sort_key):
        sampled = sample_game_probability(
            game_prediction,
            team_a,
            team_b,
            rng,
            scenario_settings,
            foul_trouble_scenarios,
            shared_logit_shift,
        )
        winner = team_a if rng.random() < sampled["team_a_win_probability"] else team_b
        wins[winner] += 1
        games_played.append(
            {
                **sampled,
                "winner": winner,
                "home_team": game_prediction.get("home_team"),
                "away_team": game_prediction.get("away_team"),
            }
        )
        if wins[winner] >= needed:
            break

    if wins[team_a] == wins[team_b]:
        winner = team_a if rng.random() < 0.5 else team_b
    else:
        winner = team_a if wins[team_a] > wins[team_b] else team_b

    return {
        "winner": winner,
        "games": len(games_played),
        "wins": wins,
        "result": format_series_result(winner, len(games_played)),
        "games_played": games_played,
        "shared_strength_logit_shift": round(shared_logit_shift, 4),
        "availability_events": availability_events,
    }


def format_series_result(team: str, games: int) -> str:
    """Format a result bucket like `NYK in 6`."""
    return f"{team} in {games}"


def _empty_result_distribution(team_a: str, team_b: str) -> list[dict[str, Any]]:
    rows = []
    for team in (team_a, team_b):
        for games in range(4, 8):
            rows.append(
                {
                    "result": format_series_result(team, games),
                    "team": team,
                    "games": games,
                    "count": 0,
                    "probability": 0.0,
                    "percentage": "0.0%",
                }
            )
    return rows


def build_series_probability_table(
    result_counts: Counter[str],
    simulations: int,
    team_a: str,
    team_b: str,
) -> list[dict[str, Any]]:
    """Create the ordered result-probability table for the dashboard/report."""
    rows = _empty_result_distribution(team_a, team_b)
    for row in rows:
        count = result_counts.get(row["result"], 0)
        probability = count / max(simulations, 1)
        row["count"] = count
        row["probability"] = round(probability, 4)
        row["percentage"] = _percentage_label(probability)
    return rows


def _game_probability_summary(game_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for game in sorted(game_predictions, key=_game_sort_key):
        low, center, high = _range_for_team_a(game)
        summary.append(
            {
                "game_number": int(game.get("game_number", 0)),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "team_a_win_probability": round(center, 4),
                "team_a_win_probability_low": round(low, 4),
                "team_a_win_probability_high": round(high, 4),
                "team_b_win_probability": round(1.0 - center, 4),
            }
        )
    return summary


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
        team_a_wins = game_team_a_wins.get(game_number, 0)
        win_rate_when_played = team_a_wins / max(played, 1)
        rows.append(
            {
                "game_number": game_number,
                "play_probability": round(played / max(simulations, 1), 4),
                f"{team_a}_win_rate_when_played": round(win_rate_when_played, 4),
                f"{team_b}_win_rate_when_played": round(1.0 - win_rate_when_played, 4),
            }
        )
    return rows


def _most_likely_result(result_distribution: list[dict[str, Any]]) -> str:
    if not result_distribution:
        return ""
    return max(result_distribution, key=lambda row: row["probability"])["result"]


def simulate_series(
    game_predictions: list[dict[str, Any]] | None = None,
    finals_context: dict[str, Any] | None = None,
    foul_trouble_simulation: dict[str, Any] | None = None,
    scenario_settings: dict[str, Any] | None = None,
    settings_path: str | Path = DEFAULT_SETTINGS_PATH,
) -> dict[str, Any]:
    """Run a Monte Carlo Finals series simulation."""
    defaults = _load_simulation_defaults(settings_path)
    settings = _scenario_settings(scenario_settings)
    simulations = int(settings.get("simulations") or defaults["simulations"])
    random_seed = int(settings.get("random_seed") or defaults["random_seed"])
    best_of = int(settings.get("best_of") or defaults["best_of"])
    context, prepared_predictions = _prepare_game_predictions(
        game_predictions,
        finals_context,
        _as_bool(settings.get("use_uncertainty"), True),
    )
    team_a, team_b = _team_names(prepared_predictions, context)
    rng = Random(random_seed)
    if foul_trouble_simulation is None and _as_bool(settings.get("use_foul_trouble"), True):
        from src.models.foul_trouble_simulator import simulate_foul_trouble_scenarios

        average_probability = sum(
            _as_float(game.get("team_a_win_probability"), 0.5)
            for game in prepared_predictions
        ) / max(len(prepared_predictions), 1)
        foul_trouble_simulation = simulate_foul_trouble_scenarios(
            context,
            base_win_probability={
                team_a: average_probability,
                team_b: 1.0 - average_probability,
            },
        )
    scenarios = _foul_trouble_scenarios(foul_trouble_simulation)

    team_series_wins = Counter()
    result_counts = Counter()
    series_length_counts = Counter()
    game_play_counts: dict[int, int] = defaultdict(int)
    game_team_a_wins: dict[int, int] = defaultdict(int)

    for _ in range(simulations):
        simulated = simulate_one_series(
            prepared_predictions,
            team_a,
            team_b,
            rng,
            best_of=best_of,
            scenario_settings=settings,
            foul_trouble_scenarios=scenarios,
        )
        winner = simulated["winner"]
        team_series_wins[winner] += 1
        result_counts[simulated["result"]] += 1
        series_length_counts[simulated["games"]] += 1
        for game in simulated["games_played"]:
            game_number = int(game["game_number"])
            game_play_counts[game_number] += 1
            if game["winner"] == team_a:
                game_team_a_wins[game_number] += 1

    team_a_probability = team_series_wins[team_a] / max(simulations, 1)
    team_b_probability = team_series_wins[team_b] / max(simulations, 1)
    result_distribution = build_series_probability_table(result_counts, simulations, team_a, team_b)
    series_length_distribution = [
        {
            "games": games,
            "count": series_length_counts.get(games, 0),
            "probability": round(series_length_counts.get(games, 0) / max(simulations, 1), 4),
            "percentage": _percentage_label(series_length_counts.get(games, 0) / max(simulations, 1)),
        }
        for games in range(4, 8)
    ]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "best_of": best_of,
        "simulations": simulations,
        "random_seed": random_seed,
        "team_a_series_win_probability": round(team_a_probability, 4),
        "team_b_series_win_probability": round(team_b_probability, 4),
        "team_a_series_win_percentage": _percentage_label(team_a_probability),
        "team_b_series_win_percentage": _percentage_label(team_b_probability),
        "most_likely_result": _most_likely_result(result_distribution),
        "result_distribution": result_distribution,
        "series_length_distribution": series_length_distribution,
        "average_games": round(
            sum(games * count for games, count in series_length_counts.items()) / max(simulations, 1),
            2,
        ),
        "game_probability_summary": _game_probability_summary(prepared_predictions),
        "played_game_summary": _played_game_summary(
            game_play_counts,
            game_team_a_wins,
            simulations,
            team_a,
            team_b,
        ),
        "scenario_settings": settings,
        "shared_strength_uncertainty": {
            "sigma_logit": _as_float(settings.get("shared_strength_sigma_logit"), 0.12),
            "description": "One latent team-strength draw is shared across every game in a simulated series.",
        },
    }


def series_feature_vector(series_simulation: dict[str, Any]) -> dict[str, float]:
    """Flatten a series simulation into model/dashboard-friendly features."""
    team_a = series_simulation["team_a"]
    team_b = series_simulation["team_b"]
    features = {
        f"{team_a}_series_win_probability": float(series_simulation["team_a_series_win_probability"]),
        f"{team_b}_series_win_probability": float(series_simulation["team_b_series_win_probability"]),
        "average_games": float(series_simulation["average_games"]),
    }
    for row in series_simulation.get("result_distribution", []):
        key = row["result"].replace(" ", "_")
        features[f"result_{key}_probability"] = float(row["probability"])
    return features


def _load_completed_results(
    game_numbers: list[int],
) -> list[dict[str, Any]]:
    """Load actual results from saved CSVs for a list of completed games."""
    import pandas as _pd

    games_dir = DEFAULT_SETTINGS_PATH.parent.parent / "data" / "processed" / "finals_games"
    results = []
    for gn in sorted(game_numbers):
        trad_path = games_dir / f"game_{gn}_team_traditional.csv"
        if not trad_path.exists():
            raise FileNotFoundError(
                f"No traditional team stats found at {trad_path}.\n"
                "Run: python -m src.finals_data.update_after_finals_game"
                f" --game {gn}"
            )
        df = _pd.read_csv(trad_path)
        agg = (
            df.groupby("teamTricode")["points"].sum().reset_index()
        )
        scores = dict(zip(agg["teamTricode"], agg["points"]))
        winner = max(scores, key=scores.get)
        results.append({
            "game_number": gn,
            "winner": winner,
            "actual_scores": scores,
        })
    return results


if __name__ == "__main__":
    import argparse as _ap
    from src.models.update_after_game import (
        apply_postgame_update_to_predictions,
        simulate_series_after_results,
    )

    parser = _ap.ArgumentParser(description="NBA Finals series simulation")
    parser.add_argument(
        "--completed", type=int, nargs="*", default=[],
        metavar="N",
        help="Completed game numbers (e.g. --completed 1 2 3)",
    )
    args = parser.parse_args()

    context = build_finals_context()
    predictions = [
        add_game_uncertainty(p)
        for p in predict_finals_games(context)
    ]

    if args.completed:
        completed_results = _load_completed_results(args.completed)
        series_score = {context["team_a"]: 0, context["team_b"]: 0}
        for r in completed_results:
            series_score[r["winner"]] += 1

        simulation = simulate_series_after_results(
            predictions,
            completed_results,
            context,
            scenario_settings={"simulations": 100000, "random_seed": 42},
        )

        team_a = context["team_a"]
        team_b = context["team_b"]
        print(f"\nSeries state  : {team_a} {series_score[team_a]}  -  {series_score[team_b]} {team_b}")
        print(f"Games played  : {len(completed_results)}")
        completed_str = ", ".join(
            f"G{r['game_number']} {r['winner']}" for r in completed_results
        )
        print(f"Completed     : {completed_str}")
    else:
        simulation = simulate_series(predictions, context)

    team_a = context["team_a"]
    team_b = context["team_b"]
    print(f"\n{team_a} wins series: {simulation['team_a_series_win_percentage']}")
    print(f"{team_b} wins series: {simulation['team_b_series_win_percentage']}")
    print(f"Most likely result: {simulation['most_likely_result']}")
    print("")
    print("Result\tProbability")
    for row in simulation["result_distribution"]:
        print(f"{row['result']}\t{row['percentage']}")
