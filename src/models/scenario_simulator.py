"""Scenario simulator for Finals what-if analysis.

Scenarios translate basketball stories into controlled probability, pace,
score, and uncertainty adjustments, then rerun the series simulator.
"""

from __future__ import annotations

import re
from math import isnan
from pathlib import Path
from typing import Any

from src.data.build_dataset import build_finals_context, load_settings
from src.models.predict_game import predict_finals_games
from src.models.simulate_series import simulate_series
from src.models.uncertainty import add_game_uncertainty


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

DEFAULT_DASHBOARD_SCENARIOS = [
    "slow_half_court_series",
    "team_a_wins_rebounding",
    "team_b_shoots_39_from_three",
    "team_a_center_foul_trouble",
    "team_a_three_point_surge",
]

SCENARIO_LIBRARY: dict[str, dict[str, Any]] = {
    "team_a_three_point_surge": {
        "name": "Team A shoots above expectation from three",
        "category": "shot_making",
        "benefiting_team": "team_a",
        "probability_shift": 0.035,
        "score_delta_team_a": 5,
        "uncertainty_delta": 0.006,
        "explanation": "A shooting heater lifts spacing and late-clock margin for Team A.",
    },
    "team_b_offensive_rebounding": {
        "name": "Team B dominates offensive rebounds",
        "category": "rebounding",
        "benefiting_team": "team_b",
        "probability_shift": 0.030,
        "score_delta_team_b": 4,
        "explanation": "Extra possessions swing the possession math toward Team B.",
    },
    "team_a_star_doubled": {
        "name": "Team A star gets doubled",
        "category": "coverage",
        "affected_team": "team_a",
        "impact_on_affected_team": -0.025,
        "score_delta_team_a": -3,
        "uncertainty_delta": 0.010,
        "explanation": "Aggressive help forces the primary creator into earlier passes and tougher counters.",
    },
    "team_b_big_foul_trouble": {
        "name": "Team B big gets in foul trouble",
        "category": "foul_trouble",
        "affected_team": "team_b",
        "impact_on_affected_team": -0.035,
        "score_delta_team_b": -3,
        "scenario_settings": {
            "foul_trouble_probability_scale": 1.20,
        },
        "explanation": "Team B loses rim protection and rotation stability when its big sits early.",
    },
    "slow_half_court_series": {
        "name": "Slow half-court series",
        "category": "pace",
        "team_a_probability_shift": -0.030,
        "pace_delta": -5.0,
        "score_delta_team_a": -4,
        "score_delta_team_b": -2,
        "uncertainty_delta": 0.004,
        "explanation": "A slower series reduces easy points and puts more weight on set-defense execution.",
    },
    "fast_transition_series": {
        "name": "Fast transition series",
        "category": "pace",
        "team_a_probability_shift": -0.015,
        "pace_delta": 5.5,
        "score_delta_team_a": 2,
        "score_delta_team_b": 4,
        "uncertainty_delta": 0.006,
        "explanation": "More transition chances increase volatility and reward the faster open-floor team.",
    },
    "team_a_bench_exposed": {
        "name": "Team A bench unit gets exposed",
        "category": "lineup",
        "affected_team": "team_a",
        "impact_on_affected_team": -0.035,
        "score_delta_team_a": -3,
        "uncertainty_delta": 0.008,
        "explanation": "Non-star minutes become expensive when the bench loses its defensive or spacing role.",
    },
    "team_a_closing_lineup_dominates": {
        "name": "Team A closing lineup dominates",
        "category": "clutch",
        "benefiting_team": "team_a",
        "probability_shift": 0.030,
        "score_delta_team_a": 2,
        "uncertainty_delta": -0.003,
        "explanation": "Late-game shot creation and two-way lineup balance move close games toward Team A.",
    },
    "team_a_wins_rebounding": {
        "name": "Team A wins rebounding",
        "category": "rebounding",
        "benefiting_team": "team_a",
        "probability_shift": 0.045,
        "score_delta_team_a": 4,
        "score_delta_team_b": -1,
        "explanation": "Team A turns the glass into a repeatable possession edge.",
    },
    "team_b_shoots_39_from_three": {
        "name": "Team B shoots 39% from three",
        "category": "shot_making",
        "benefiting_team": "team_b",
        "probability_shift": 0.055,
        "score_delta_team_b": 6,
        "uncertainty_delta": 0.010,
        "explanation": "Team B's three-point shooting outpaces the base shot-quality expectation.",
    },
    "team_a_center_foul_trouble": {
        "name": "Team A center foul trouble",
        "category": "foul_trouble",
        "affected_team": "team_a",
        "impact_on_affected_team": -0.045,
        "score_delta_team_a": -3,
        "score_delta_team_b": 3,
        "scenario_settings": {
            "foul_trouble_probability_scale": 1.25,
        },
        "explanation": "Team A loses rim protection, rebounding, and rotation certainty when its center sits.",
    },
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


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _percentage_label(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _load_defaults(settings_path: str | Path = DEFAULT_SETTINGS_PATH) -> dict[str, int]:
    try:
        settings = load_settings(settings_path)
    except (FileNotFoundError, ValueError):
        return {"simulations": 100000, "random_seed": 42}

    return {
        "simulations": int(settings.get("finals", {}).get("simulations", 100000)),
        "random_seed": int(settings.get("project", {}).get("random_seed", 42)),
    }


def _resolve_team(team_ref: Any, team_a: str, team_b: str) -> str:
    value = str(team_ref or "").strip()
    normalized = value.lower().replace("_", " ")
    if normalized in {"team a", "a", "home team a"}:
        return team_a
    if normalized in {"team b", "b", "home team b"}:
        return team_b
    if value == team_a:
        return team_a
    if value == team_b:
        return team_b
    return value


def _resolved_text(text: str, team_a: str, team_b: str) -> str:
    return text.replace("Team A", team_a).replace("Team B", team_b)


def _template_lookup() -> dict[str, dict[str, Any]]:
    lookup = {}
    for scenario_id, scenario in SCENARIO_LIBRARY.items():
        lookup[_slug(scenario_id)] = {"scenario_id": scenario_id, **scenario}
        lookup[_slug(str(scenario["name"]))] = {"scenario_id": scenario_id, **scenario}
    return lookup


def resolve_scenario_definition(
    scenario: str | dict[str, Any],
    team_a: str,
    team_b: str,
) -> dict[str, Any]:
    """Resolve a library scenario id/name or normalize a custom scenario dict."""
    if isinstance(scenario, str):
        lookup = _template_lookup()
        key = _slug(scenario)
        if key not in lookup:
            raise ValueError(f"Unknown scenario: {scenario}")
        resolved = dict(lookup[key])
    else:
        resolved = dict(scenario)
        scenario_id = str(resolved.get("scenario_id") or resolved.get("id") or _slug(str(resolved.get("name", "custom_scenario"))))
        resolved["scenario_id"] = scenario_id
        resolved.setdefault("name", scenario_id.replace("_", " ").title())

    resolved["resolved_name"] = _resolved_text(str(resolved["name"]), team_a, team_b)
    resolved["resolved_explanation"] = _resolved_text(str(resolved.get("explanation", "")), team_a, team_b)
    return resolved


def default_scenarios(finals_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the full built-in scenario library with resolved labels."""
    context = finals_context or build_finals_context()
    team_a = str(context["team_a"])
    team_b = str(context["team_b"])
    return [
        resolve_scenario_definition({"scenario_id": scenario_id, **scenario}, team_a, team_b)
        for scenario_id, scenario in SCENARIO_LIBRARY.items()
    ]


def dashboard_scenarios(finals_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the smaller scenario set used by the default report view."""
    context = finals_context or build_finals_context()
    team_a = str(context["team_a"])
    team_b = str(context["team_b"])
    return [
        resolve_scenario_definition(scenario_id, team_a, team_b)
        for scenario_id in DEFAULT_DASHBOARD_SCENARIOS
    ]


def _scenario_team_a_shift(scenario: dict[str, Any], team_a: str, team_b: str) -> float:
    shift = _as_float(scenario.get("team_a_probability_shift"), 0.0)
    shift -= _as_float(scenario.get("team_b_probability_shift"), 0.0)

    benefiting_team = _resolve_team(scenario.get("benefiting_team"), team_a, team_b)
    if benefiting_team:
        amount = _as_float(scenario.get("probability_shift"), 0.0)
        if benefiting_team == team_a:
            shift += amount
        elif benefiting_team == team_b:
            shift -= amount

    affected_team = _resolve_team(scenario.get("affected_team"), team_a, team_b)
    if affected_team:
        amount = _as_float(scenario.get("impact_on_affected_team"), _as_float(scenario.get("probability_shift"), 0.0))
        if affected_team == team_a:
            shift += amount
        elif affected_team == team_b:
            shift -= amount

    return _clip(shift, -0.18, 0.18)


def _game_extra_shift(scenario: dict[str, Any], game_number: int) -> float:
    game_shifts = scenario.get("game_probability_shifts") or {}
    return _as_float(game_shifts.get(str(game_number), game_shifts.get(game_number, 0.0)), 0.0)


def _applies_to_game(scenario: dict[str, Any], game_number: int) -> bool:
    applies_to = scenario.get("applies_to_games")
    if not applies_to:
        return True
    return game_number in {int(game) for game in applies_to}


def _probability_range_label(low: float, high: float) -> str:
    return f"{round(low * 100):.0f}%-{round(high * 100):.0f}%"


def _adjust_probability_fields(
    game: dict[str, Any],
    team_a: str,
    team_b: str,
    shift: float,
    uncertainty_delta: float,
) -> dict[str, Any]:
    original_center = _clip(_as_float(game.get("team_a_win_probability"), 0.5), 0.01, 0.99)
    original_range = game.get("team_a_win_probability_range") or {}
    original_low = _clip(_as_float(original_range.get("low"), original_center), 0.01, 0.99)
    original_high = _clip(_as_float(original_range.get("high"), original_center), 0.01, 0.99)

    center = _clip(original_center + shift, 0.01, 0.99)
    low_width = max(original_center - original_low + uncertainty_delta, 0.01)
    high_width = max(original_high - original_center + uncertainty_delta, 0.01)
    low = _clip(center - low_width, 0.01, center)
    high = _clip(center + high_width, center, 0.99)

    team_a_range = {
        "center": round(center, 3),
        "low": round(low, 3),
        "high": round(high, 3),
        "label": _probability_range_label(low, high),
    }
    team_b_range = {
        "center": round(1.0 - center, 3),
        "low": round(1.0 - high, 3),
        "high": round(1.0 - low, 3),
        "label": _probability_range_label(1.0 - high, 1.0 - low),
    }

    adjusted = {
        **game,
        "team_a_win_probability": round(center, 4),
        "team_b_win_probability": round(1.0 - center, 4),
        "team_a_win_probability_range": team_a_range,
        "team_b_win_probability_range": team_b_range,
        "realistic_win_probability_range": {
            team_a: team_a_range,
            team_b: team_b_range,
        },
    }
    return adjusted


def _team_score_delta(scenario: dict[str, Any], team_key: str, team: str, team_a: str, team_b: str) -> int:
    direct_key = f"score_delta_{team_key}"
    if direct_key in scenario:
        return int(round(_as_float(scenario.get(direct_key), 0.0)))

    benefiting_team = _resolve_team(scenario.get("benefiting_team"), team_a, team_b)
    affected_team = _resolve_team(scenario.get("affected_team"), team_a, team_b)
    if benefiting_team == team:
        return int(round(_as_float(scenario.get("score_delta_benefiting_team"), 0.0)))
    if affected_team == team:
        return int(round(_as_float(scenario.get("score_delta_affected_team"), 0.0)))
    return 0


def _adjust_score_fields(
    game: dict[str, Any],
    scenario: dict[str, Any],
    team_a: str,
    team_b: str,
) -> dict[str, Any]:
    pace_delta = _as_float(scenario.get("pace_delta"), 0.0)
    projected_pace = max(_as_float(game.get("projected_pace"), 98.0) + pace_delta, 80.0)
    score_a = int(game.get("expected_score_team_a", 0)) + _team_score_delta(scenario, "team_a", team_a, team_a, team_b)
    score_b = int(game.get("expected_score_team_b", 0)) + _team_score_delta(scenario, "team_b", team_b, team_a, team_b)

    adjusted = {
        **game,
        "projected_pace": round(projected_pace, 1),
        "expected_score_team_a": max(score_a, 70),
        "expected_score_team_b": max(score_b, 70),
    }
    adjusted["projected_team_a_off_rating"] = round(adjusted["expected_score_team_a"] * 100.0 / projected_pace, 1)
    adjusted["projected_team_b_off_rating"] = round(adjusted["expected_score_team_b"] * 100.0 / projected_pace, 1)
    return adjusted


def apply_scenario_to_game_predictions(
    game_predictions: list[dict[str, Any]],
    scenario: dict[str, Any],
    team_a: str,
    team_b: str,
) -> list[dict[str, Any]]:
    """Adjust game predictions according to one scenario definition."""
    adjusted_games = []
    base_shift = _scenario_team_a_shift(scenario, team_a, team_b)
    uncertainty_delta = _as_float(scenario.get("uncertainty_delta"), 0.0)
    explanation = str(scenario.get("resolved_explanation") or scenario.get("explanation") or "")
    scenario_name = str(scenario.get("resolved_name") or scenario.get("name") or "Scenario")

    for game in game_predictions:
        game_number = int(game.get("game_number", 0))
        adjusted = dict(game)
        if _applies_to_game(scenario, game_number):
            shift = base_shift + _game_extra_shift(scenario, game_number)
            adjusted = _adjust_probability_fields(adjusted, team_a, team_b, shift, uncertainty_delta)
            adjusted = _adjust_score_fields(adjusted, scenario, team_a, team_b)
            adjusted["scenario_probability_shift_team_a"] = round(shift, 4)
            adjusted["top_edges"] = [*adjusted.get("top_edges", []), scenario_name][:7]
            if explanation:
                adjusted["x_factors"] = [*adjusted.get("x_factors", []), explanation][:8]
        adjusted_games.append(adjusted)
    return adjusted_games


def _merge_scenario_settings(
    base_settings: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    return {
        **base_settings,
        **(scenario.get("scenario_settings") or {}),
    }


def _case_row(
    scenario: dict[str, Any],
    simulation: dict[str, Any],
    base_probability: float | None = None,
    is_base_case: bool = False,
) -> dict[str, Any]:
    team_a_probability = _as_float(simulation.get("team_a_series_win_probability"), 0.0)
    delta = 0.0 if base_probability is None else team_a_probability - base_probability
    return {
        "scenario_id": scenario.get("scenario_id", "base_case"),
        "scenario": scenario.get("name", "Base case"),
        "resolved_scenario": scenario.get("resolved_name", scenario.get("name", "Base case")),
        "category": scenario.get("category", "base"),
        "description": scenario.get("resolved_explanation", scenario.get("explanation", "")),
        "is_base_case": is_base_case,
        "team_a": simulation["team_a"],
        "team_b": simulation["team_b"],
        "team_a_series_win_probability": simulation["team_a_series_win_probability"],
        "team_b_series_win_probability": simulation["team_b_series_win_probability"],
        "team_a_series_win_percentage": simulation["team_a_series_win_percentage"],
        "team_b_series_win_percentage": simulation["team_b_series_win_percentage"],
        "team_a_delta_from_base": round(delta, 4),
        "team_a_delta_from_base_label": f"{delta * 100:+.1f} pts",
        "most_likely_result": simulation["most_likely_result"],
        "average_games": simulation["average_games"],
        "result_distribution": simulation["result_distribution"],
        "series_simulation": simulation,
    }


def simulate_scenario(
    scenario: str | dict[str, Any],
    game_predictions: list[dict[str, Any]] | None = None,
    finals_context: dict[str, Any] | None = None,
    base_series_probability: float | None = None,
    scenario_settings: dict[str, Any] | None = None,
    foul_trouble_simulation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one named or custom scenario and return a dashboard-ready row."""
    context = finals_context or build_finals_context()
    team_a = str(context["team_a"])
    team_b = str(context["team_b"])
    predictions = game_predictions or predict_finals_games(context)
    predictions_with_uncertainty = [
        prediction if "team_a_win_probability_range" in prediction else add_game_uncertainty(prediction)
        for prediction in predictions
    ]
    definition = resolve_scenario_definition(scenario, team_a, team_b)
    adjusted_predictions = apply_scenario_to_game_predictions(predictions_with_uncertainty, definition, team_a, team_b)
    simulation = simulate_series(
        adjusted_predictions,
        context,
        foul_trouble_simulation=foul_trouble_simulation,
        scenario_settings=_merge_scenario_settings(scenario_settings or {}, definition),
    )
    return _case_row(definition, simulation, base_series_probability)


def run_scenario_suite(
    scenarios: list[str | dict[str, Any]] | None = None,
    game_predictions: list[dict[str, Any]] | None = None,
    finals_context: dict[str, Any] | None = None,
    scenario_settings: dict[str, Any] | None = None,
    foul_trouble_simulation: dict[str, Any] | None = None,
    settings_path: str | Path = DEFAULT_SETTINGS_PATH,
) -> dict[str, Any]:
    """Run base case plus a suite of what-if scenarios."""
    context = finals_context or build_finals_context()
    team_a = str(context["team_a"])
    team_b = str(context["team_b"])
    defaults = _load_defaults(settings_path)
    settings = {
        "simulations": defaults["simulations"],
        "random_seed": defaults["random_seed"],
        **(scenario_settings or {}),
    }
    predictions = game_predictions or predict_finals_games(context)
    predictions_with_uncertainty = [
        prediction if "team_a_win_probability_range" in prediction else add_game_uncertainty(prediction)
        for prediction in predictions
    ]
    selected_scenarios = scenarios or DEFAULT_DASHBOARD_SCENARIOS

    base_simulation = simulate_series(
        predictions_with_uncertainty,
        context,
        foul_trouble_simulation=foul_trouble_simulation,
        scenario_settings=settings,
    )
    base_case = _case_row(
        {
            "scenario_id": "base_case",
            "name": "Base case",
            "resolved_name": "Base case",
            "category": "base",
            "explanation": "Current model projection with no scenario adjustment.",
        },
        base_simulation,
        is_base_case=True,
    )
    base_probability = base_simulation["team_a_series_win_probability"]

    scenario_rows = [
        simulate_scenario(
            scenario,
            predictions_with_uncertainty,
            context,
            base_probability,
            settings,
            foul_trouble_simulation=foul_trouble_simulation,
        )
        for scenario in selected_scenarios
    ]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "simulations": settings["simulations"],
        "base_case": base_case,
        "scenarios": scenario_rows,
        "all_cases": [base_case, *scenario_rows],
        "summary": scenario_summary_table([base_case, *scenario_rows]),
    }


def scenario_summary_table(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact rows for the dashboard scenario table."""
    return [
        {
            "scenario": case["resolved_scenario"],
            "team_a": case["team_a"],
            "team_a_series_win_probability": case["team_a_series_win_probability"],
            "team_a_series_win_percentage": case["team_a_series_win_percentage"],
            "team_a_delta_from_base": case["team_a_delta_from_base"],
            "team_a_delta_from_base_label": case["team_a_delta_from_base_label"],
            "most_likely_result": case["most_likely_result"],
        }
        for case in cases
    ]


def scenario_feature_vector(scenario_report: dict[str, Any]) -> dict[str, float]:
    """Flatten scenario results into numeric report/model features."""
    team_a = scenario_report["team_a"]
    features = {}
    for case in scenario_report.get("all_cases", []):
        key = _slug(str(case["resolved_scenario"]))
        features[f"{key}_{team_a}_series_win_probability"] = float(case["team_a_series_win_probability"])
        features[f"{key}_{team_a}_delta_from_base"] = float(case["team_a_delta_from_base"])
    return features


if __name__ == "__main__":
    context = build_finals_context()
    report = run_scenario_suite(finals_context=context)
    team_a = report["team_a"]
    for row in report["summary"]:
        delta = "" if row["scenario"] == "Base case" else f" ({row['team_a_delta_from_base_label']})"
        print(f"{row['scenario']}: {team_a} {row['team_a_series_win_percentage']}{delta}")
