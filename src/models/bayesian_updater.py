"""Bayesian series updater.

Replaces the heuristic probability-shift approach with a principled
Bayesian update. Each game result is evidence that updates the prior
on each team's true strength difference.

Model:
  - Prior: θ ~ N(μ₀, σ₀²)  where θ is team_a log-odds advantage
  - Each game: P(team_a wins | θ, home, rest) = sigmoid(θ + home_bonus + rest_bonus)
  - Posterior after N games: updated normal approximation via variational inference

The posterior mean and variance are propagated forward to produce
an updated series win probability.
"""

from __future__ import annotations

import math
from typing import Any


# Prior standard deviation on team log-odds advantage.
# σ = 0.40 corresponds to ~±10 percentage points of uncertainty,
# which is reasonable for two Finals-calibre teams.
PRIOR_SIGMA = 0.40
HOME_BONUS = 0.22   # log-odds boost for home team (~+5.5% win prob at 50%)
REST_BONUS_PER_DAY = 0.025  # log-odds per day of rest advantage


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1.0 - p))


def _game_log_odds(
    theta: float,
    is_home_a: bool,
    rest_diff: float,
) -> float:
    """Log-odds of team_a winning one game given latent strength θ."""
    return theta + (HOME_BONUS if is_home_a else -HOME_BONUS) + REST_BONUS_PER_DAY * rest_diff


def _laplace_update(
    prior_mean: float,
    prior_var: float,
    game_results: list[dict[str, Any]],
) -> tuple[float, float]:
    """Laplace (Gaussian) approximation to the posterior after observed games.

    Each game_result dict must have:
      - won: bool (True = team_a won)
      - is_home_a: bool
      - rest_diff: float (team_a rest - team_b rest)
    """
    mu = prior_mean

    for _ in range(20):  # Newton iterations
        gradient = -(mu - prior_mean) / prior_var
        hessian = -1.0 / prior_var  # prior curvature

        for result in game_results:
            log_odds = _game_log_odds(mu, result["is_home_a"], result["rest_diff"])
            p = _sigmoid(log_odds)
            y = 1.0 if result["won"] else 0.0
            gradient += y - p          # likelihood gradient
            hessian -= p * (1.0 - p)   # likelihood curvature (negative)

        if abs(gradient) < 1e-8:
            break
        mu = mu - gradient / hessian

    # Recompute hessian at the converged mu so posterior_var is always correct,
    # even when the loop exhausts all iterations without a clean convergence check.
    hessian = -1.0 / prior_var
    for result in game_results:
        log_odds = _game_log_odds(mu, result["is_home_a"], result["rest_diff"])
        p = _sigmoid(log_odds)
        hessian -= p * (1.0 - p)

    return mu, max(-1.0 / hessian, 0.001)


def bayesian_series_win_probability(
    prior_mean: float,
    game_results: list[dict[str, Any]],
    remaining_games: list[dict[str, Any]],
    simulations: int = 50000,
    random_seed: int = 42,
) -> float:
    """P(team_a wins series) using Bayesian posterior after observed games.

    Args:
        prior_mean: log-odds advantage for team_a before the series.
                    Derived from the base model's pre-series probability.
        game_results: list of completed game dicts with won/is_home_a/rest_diff.
        remaining_games: list of future game dicts with is_home_a/rest_diff.
        simulations: Monte Carlo draws from the posterior.
        random_seed: for reproducibility.

    Returns:
        P(team_a wins the series) as a float in [0, 1].
    """
    import random

    prior_var = PRIOR_SIGMA ** 2
    posterior_mean, posterior_var = _laplace_update(prior_mean, prior_var, game_results)
    posterior_std = math.sqrt(posterior_var)

    # Count wins already achieved
    completed_a_wins = sum(1 for r in game_results if r["won"])
    completed_b_wins = len(game_results) - completed_a_wins
    wins_needed = 4  # best of 7

    if completed_a_wins >= wins_needed:
        return 1.0
    if completed_b_wins >= wins_needed:
        return 0.0

    rng = random.Random(random_seed)
    team_a_series_wins = 0

    for _ in range(simulations):
        # Draw θ from posterior
        theta = rng.gauss(posterior_mean, posterior_std)
        a_wins = completed_a_wins
        b_wins = completed_b_wins

        for game in remaining_games:
            if a_wins >= wins_needed or b_wins >= wins_needed:
                break
            log_odds = _game_log_odds(theta, game["is_home_a"], game["rest_diff"])
            p_a = _sigmoid(log_odds)
            if rng.random() < p_a:
                a_wins += 1
            else:
                b_wins += 1

        if a_wins >= wins_needed:
            team_a_series_wins += 1

    return round(team_a_series_wins / simulations, 4)


def prior_mean_from_probability(p: float) -> float:
    """Convert a win probability to a log-odds prior mean."""
    return _logit(p)


def prior_mean_from_game_predictions(
    game_predictions: list[dict[str, Any]],
    game_contexts: list[dict[str, Any]],
) -> float:
    """Estimate neutral-court latent strength from game-level forecasts."""
    latent_estimates = []
    for prediction, context in zip(game_predictions, game_contexts):
        probability = float(prediction.get("team_a_win_probability", 0.5))
        context_offset = (
            (HOME_BONUS if context.get("is_home_a") else -HOME_BONUS)
            + REST_BONUS_PER_DAY * float(context.get("rest_diff", 0.0))
        )
        latent_estimates.append(_logit(probability) - context_offset)
    if not latent_estimates:
        return 0.0
    return sum(latent_estimates) / len(latent_estimates)


def build_game_context(
    game: dict[str, Any],
    team_a: str,
    finals_context: dict[str, Any],
) -> dict[str, Any]:
    """Build a game context dict for Bayesian update from a game dict."""
    home_team = str(game.get("home_team", ""))
    is_neutral = bool(game.get("neutral_site", False))
    is_home_a = (not is_neutral) and (home_team == team_a)

    # Rest diff: use pre_series_rest for game 1, 0 for subsequent games
    # (both teams have same rest between Finals games)
    game_number = int(game.get("game_number", 1))
    if game_number == 1:
        pre_rest = finals_context.get("pre_series_rest") or {}
        team_b = str(finals_context.get("team_b", ""))
        rest_a = float(pre_rest.get(team_a, 3.0))
        rest_b = float(pre_rest.get(team_b, 3.0))
        rest_diff = rest_a - rest_b
    else:
        rest_diff = 0.0

    return {
        "game_number": game_number,
        "is_home_a": is_home_a,
        "rest_diff": rest_diff,
    }
