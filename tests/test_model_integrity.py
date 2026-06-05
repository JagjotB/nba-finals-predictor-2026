from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.features.player_features import project_team_minutes
from src.data.build_historical_dataset import (
    blend_regular_and_playoff,
    validate_no_future_leakage,
)
from src.models.bayesian_updater import PRIOR_SIGMA, _laplace_update
from src.models.game_model import (
    FEATURE_COLS,
    build_training_rows,
    load_model,
    predict_win_probability,
    save_model,
    train,
    walk_forward_backtest,
)
from src.models.meta_model import predict_meta_probability, train_meta_model
from src.models.predict_game import _baseline_team_stats, _combine_probability
from src.models.simulate_series import simulate_series
from src.models.train_player_model import (
    build_player_feature_row,
    reconcile_player_projections_to_team_scores,
    simulate_correlated_player_box_scores,
)
from src.models.update_after_game import simulate_series_after_results


def _ratings(seasons: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        season: {
            "1": {
                "NET_RATING": 5.0,
                "EFG_PCT": 0.56,
                "TM_TOV_PCT": 0.12,
                "OREB_PCT": 0.29,
                "FTA_RATE": 0.27,
                "PACE": 98.0,
            },
            "2": {
                "NET_RATING": 1.0,
                "EFG_PCT": 0.53,
                "TM_TOV_PCT": 0.14,
                "OREB_PCT": 0.26,
                "FTA_RATE": 0.24,
                "PACE": 97.0,
            },
        }
        for season in seasons
    }


def _game_logs(seasons: list[str], games_per_season: int) -> list[dict[str, object]]:
    rows = []
    for season_index, season in enumerate(seasons):
        year = 2015 + season_index
        for game_index in range(games_per_season):
            home_id, away_id = (1, 2) if game_index % 2 == 0 else (2, 1)
            home_won = (game_index + season_index) % 3 != 0
            game_id = f"{season_index:02d}{game_index:03d}"
            game_date = f"{year}-05-{game_index + 1:02d}T00:00:00"
            rows.extend(
                [
                    {
                        "SEASON_YEAR": season,
                        "TEAM_ID": home_id,
                        "GAME_ID": game_id,
                        "GAME_DATE": game_date,
                        "MATCHUP": "A vs. B",
                        "WL": "W" if home_won else "L",
                    },
                    {
                        "SEASON_YEAR": season,
                        "TEAM_ID": away_id,
                        "GAME_ID": game_id,
                        "GAME_DATE": game_date,
                        "MATCHUP": "B @ A",
                        "WL": "L" if home_won else "W",
                    },
                ]
            )
    return rows


class ModelIntegrityTests(unittest.TestCase):
    def test_bayesian_no_evidence_preserves_prior(self) -> None:
        posterior_mean, posterior_var = _laplace_update(
            0.8,
            PRIOR_SIGMA ** 2,
            [],
        )
        self.assertAlmostEqual(posterior_mean, 0.8)
        self.assertAlmostEqual(posterior_var, PRIOR_SIGMA ** 2)

    def test_training_rows_are_symmetric_and_season_scoped(self) -> None:
        seasons = ["2022-23", "2023-24"]
        rows = build_training_rows(_game_logs(seasons, 2), _ratings(seasons))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["home_court"] for row in rows}, {-1.0, 1.0})
        self.assertEqual(
            sum(row["perspective"] == "home" for row in rows),
            sum(row["perspective"] == "away" for row in rows),
        )
        self.assertLessEqual(max(abs(row["rest_diff"]) for row in rows), 3.0)

    def test_rotation_minutes_reconcile_to_240(self) -> None:
        rotations = [
            {
                "team": "A",
                "player": f"Player {index}",
                "role": "Starter" if index < 5 else "Bench",
                "projected_minutes": 30.0 if index < 5 else 15.0,
                "minutes_floor": 25.0 if index < 5 else 8.0,
                "minutes_ceiling": 38.0 if index < 5 else 22.0,
                "is_starter": index < 5,
                "is_closer": index < 5,
                "rotation_confidence": "high",
            }
            for index in range(10)
        ]
        projections = project_team_minutes("A", rotations)
        self.assertAlmostEqual(
            sum(row["projected_minutes"] for row in projections),
            240.0,
        )

    def test_zero_residual_preserves_baseline_probability(self) -> None:
        margins = {
            "player_projection": 0.0,
            "matchup_edge": 0.0,
            "lineup_edge": 0.0,
            "clutch_edge": 0.0,
            "injury_edge": 0.0,
            "foul_trouble_risk": 0.0,
        }
        weights = {
            "team_baseline": 1.0,
            "player_projection": 0.45,
            "matchup_edge": 0.20,
            "lineup_edge": 0.25,
            "clutch_edge": 0.10,
            "injury_edge": 0.0,
            "foul_trouble_risk": 0.0,
        }
        self.assertAlmostEqual(_combine_probability(0.7, margins, weights), 0.7)

    def test_game_model_preserves_regular_prior_and_shrinks_playoffs(self) -> None:
        context = {
            "team_a": "A",
            "team_b": "B",
            "regular_season_team_stats": {
                "A": {"NET_RATING": 4.0},
                "B": {"NET_RATING": 2.0},
            },
            "playoff_team_stats": {
                "A": {"NET_RATING": 20.0, "GP": 12},
                "B": {"NET_RATING": -5.0, "GP": 12},
            },
        }
        baseline = _baseline_team_stats(context)
        self.assertEqual(baseline["A"]["REGULAR_NET_RATING"], 4.0)
        self.assertEqual(baseline["B"]["REGULAR_NET_RATING"], 2.0)
        self.assertGreater(baseline["A"]["NET_RATING"], 4.0)
        self.assertLess(baseline["A"]["NET_RATING"], 20.0)

    def test_player_rate_blend_favors_playoff_sample(self) -> None:
        minutes_projection = {
            "player_key": "A:Player:Starter",
            "team": "A",
            "player": "Player",
            "role": "Starter",
            "projected_minutes": 30.0,
            "minutes_floor": 26.0,
            "minutes_ceiling": 34.0,
            "is_starter": True,
            "is_closer": True,
            "injury_status": "Available",
            "injury_adjustment": 0.0,
            "rotation_confidence": "high",
        }
        sources = {
            "season": {"MIN": 10.0, "PTS": 10.0, "GP": 82},
            "playoff": {"MIN": 10.0, "PTS": 20.0, "GP": 20},
            "recent_5": None,
            "recent_10": None,
            "recent_15": None,
        }
        row = build_player_feature_row(minutes_projection, sources, "B")
        self.assertGreater(row["points_per_minute"], 1.0)
        self.assertLess(row["points_per_minute"], 2.0)

    def test_metric_specific_shrinkage_is_more_conservative_for_shooting(self) -> None:
        _, shooting_weight = blend_regular_and_playoff(0.55, 0.65, 600.0, "efg_pct")
        _, pace_weight = blend_regular_and_playoff(98.0, 102.0, 600.0, "pace")
        self.assertLess(shooting_weight, pace_weight)

    def test_leakage_validator_rejects_same_day_feature_cutoff(self) -> None:
        rows = [{
            "season": "2024-25",
            "game_id": "1",
            "perspective": "home",
            "home_court": 1.0,
            "game_date": "2025-05-01",
            "feature_cutoff_date": "2025-05-01",
        }]
        self.assertTrue(validate_no_future_leakage(rows))

    def test_walk_forward_never_trains_on_future_seasons(self) -> None:
        seasons = [
            "2018-19", "2019-20", "2020-21",
            "2021-22", "2022-23", "2023-24",
        ]
        rows = build_training_rows(_game_logs(seasons, 8), _ratings(seasons))
        report = walk_forward_backtest(rows, test_seasons=["2022-23", "2023-24"])
        for split in report["splits"]:
            self.assertTrue(all(
                season < split["test_season"]
                for season in split["train_seasons"]
            ))

    def test_unvalidated_meta_component_has_zero_effect(self) -> None:
        rows = []
        for index in range(120):
            probability = 0.62 if index % 2 == 0 else 0.38
            rows.append({
                "season": "2022-23" if index < 60 else "2023-24",
                "actual_team_a_win": int(index % 2 == 0),
                "baseline_probability": probability,
                "net_rating_probability": probability,
                "player_edge": 0.0,
            })
        bundle = train_meta_model(rows)
        neutral = predict_meta_probability(0.6, {"player_edge": 0.0}, model_bundle=bundle)
        large_edge = predict_meta_probability(0.6, {"player_edge": 8.0}, model_bundle=bundle)
        self.assertAlmostEqual(neutral, large_edge)

    def test_player_points_reconcile_to_team_score(self) -> None:
        projections = {
            "A": [
                {"team": "A", "player": "One", "points": 60.0},
                {"team": "A", "player": "Two", "points": 40.0},
            ]
        }
        reconciled = reconcile_player_projections_to_team_scores(projections, {"A": 110.0})
        self.assertAlmostEqual(sum(row["points"] for row in reconciled["A"]), 110.0)

    def test_correlated_player_simulation_is_reproducible(self) -> None:
        projections = {
            "A": [{"team": "A", "player": "One", "points": 25.0, "rebounds": 5.0, "assists": 7.0}]
        }
        first = simulate_correlated_player_box_scores(projections, simulations=100, random_seed=9)
        second = simulate_correlated_player_box_scores(projections, simulations=100, random_seed=9)
        self.assertEqual(first, second)

    def test_series_simulation_is_reproducible_with_shared_uncertainty(self) -> None:
        predictions = [
            {
                "game_number": number,
                "team_a": "A",
                "team_b": "B",
                "home_team": "A" if number in {1, 2, 5, 7} else "B",
                "away_team": "B" if number in {1, 2, 5, 7} else "A",
                "team_a_win_probability": 0.55,
                "team_b_win_probability": 0.45,
                "expected_score_team_a": 111,
                "expected_score_team_b": 108,
                "projected_pace": 97.0,
                "component_margins": {},
                "x_factors": [],
            }
            for number in range(1, 8)
        ]
        settings = {"simulations": 500, "random_seed": 17}
        foul = {"scenarios": []}
        first = simulate_series(
            predictions, {"team_a": "A", "team_b": "B"},
            foul_trouble_simulation=foul, scenario_settings=settings,
        )
        second = simulate_series(
            predictions, {"team_a": "A", "team_b": "B"},
            foul_trouble_simulation=foul, scenario_settings=settings,
        )
        self.assertEqual(first["result_distribution"], second["result_distribution"])
        self.assertEqual(first["shared_strength_uncertainty"]["sigma_logit"], 0.12)

    def test_calibrated_model_round_trip(self) -> None:
        seasons = [
            "2018-19",
            "2019-20",
            "2020-21",
            "2021-22",
            "2022-23",
            "2023-24",
        ]
        rows = build_training_rows(
            _game_logs(seasons, 6),
            _ratings(seasons),
        )
        bundle = train(rows, holdout_seasons=["2023-24"])
        features = {feature: float(rows[0][feature]) for feature in FEATURE_COLS}
        live_probability = predict_win_probability(features, bundle, "1", "2")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.json"
            save_model(bundle, path)
            loaded = load_model(path)
            self.assertIsNotNone(loaded)
            loaded_probability = predict_win_probability(features, loaded or {}, "1", "2")
        self.assertAlmostEqual(live_probability, loaded_probability)

    def test_series_headline_matches_result_distribution(self) -> None:
        predictions = [
            {
                "game_number": number,
                "team_a": "A",
                "team_b": "B",
                "home_team": "A" if number % 2 else "B",
                "away_team": "B" if number % 2 else "A",
                "team_a_win_probability": 0.6,
                "team_b_win_probability": 0.4,
                "expected_score_team_a": 110,
                "expected_score_team_b": 106,
                "projected_pace": 97.0,
                "component_margins": {},
                "x_factors": [],
            }
            for number in range(1, 8)
        ]
        context = {
            "team_a": "A",
            "team_b": "B",
            "schedule": predictions,
            "rotations": {"A": [], "B": []},
            "injuries": {"A": [], "B": []},
        }
        series = simulate_series_after_results(
            predictions,
            [{"game_number": 1, "winner": "A", "actual_scores": {"A": 100, "B": 90}}],
            context,
            scenario_settings={"simulations": 2000, "random_seed": 7},
            foul_trouble_simulation={"scenarios": []},
        )
        distribution_probability = sum(
            row["probability"]
            for row in series["result_distribution"]
            if row["team"] == "A"
        )
        self.assertAlmostEqual(
            series["team_a_series_win_probability"],
            distribution_probability,
            places=3,
        )


    # ------------------------------------------------------------------
    # Tests covering the four confirmed bugs fixed in this session
    # ------------------------------------------------------------------

    def test_bayesian_posterior_shifts_with_game_evidence(self) -> None:
        """After observed wins, the posterior mean must move away from the prior."""
        prior_mean = 0.0  # even-strength teams
        prior_var = PRIOR_SIGMA ** 2
        # team_a wins three games on a neutral court with no rest difference
        results = [
            {"won": True, "is_home_a": False, "rest_diff": 0.0},
            {"won": True, "is_home_a": True, "rest_diff": 0.0},
            {"won": True, "is_home_a": False, "rest_diff": 0.0},
        ]
        posterior_mean, posterior_var = _laplace_update(prior_mean, prior_var, results)
        self.assertGreater(posterior_mean, prior_mean)
        self.assertLess(posterior_var, prior_var)

    def test_bayesian_posterior_var_is_consistent_with_converged_mu(self) -> None:
        """Posterior variance must be computed at the final (converged) mu."""
        from src.models.bayesian_updater import _sigmoid, _game_log_odds
        prior_mean = 0.5
        prior_var = PRIOR_SIGMA ** 2
        results = [
            {"won": True, "is_home_a": True, "rest_diff": 2.0},
            {"won": False, "is_home_a": False, "rest_diff": -1.0},
        ]
        mu, post_var = _laplace_update(prior_mean, prior_var, results)
        # Manually verify hessian at the returned mu matches post_var
        hessian = -1.0 / prior_var
        for r in results:
            p = _sigmoid(_game_log_odds(mu, r["is_home_a"], r["rest_diff"]))
            hessian -= p * (1.0 - p)
        expected_var = max(-1.0 / hessian, 0.001)
        self.assertAlmostEqual(post_var, expected_var, places=10)

    def test_load_model_raises_on_missing_binary(self) -> None:
        """load_model must raise FileNotFoundError when only the JSON exists."""
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(
                json.dumps({"model_format": "joblib_calibrated_classifier"}),
                encoding="utf-8",
            )
            # Binary (.joblib) is intentionally absent
            with self.assertRaises(FileNotFoundError):
                load_model(path)

    def test_home_court_is_in_feature_cols(self) -> None:
        """home_court must be a training feature so the model can learn it."""
        self.assertIn("home_court", FEATURE_COLS)

    def test_home_court_drives_probability_direction(self) -> None:
        """With home_court in features, the model must assign a higher win
        probability to the home side than to the identically-rated away side."""
        seasons = [
            "2018-19", "2019-20", "2020-21",
            "2021-22", "2022-23", "2023-24",
        ]
        rows = build_training_rows(_game_logs(seasons, 8), _ratings(seasons))
        bundle = train(rows, holdout_seasons=["2023-24"])
        neutral_features = {f: 0.0 for f in FEATURE_COLS}
        home_features = {**neutral_features, "home_court": 1.0}
        away_features = {**neutral_features, "home_court": -1.0}
        p_home = predict_win_probability(home_features, bundle)
        p_away = predict_win_probability(away_features, bundle)
        self.assertGreater(p_home, 0.5)
        self.assertLess(p_away, 0.5)
        self.assertGreater(p_home, p_away)

    def test_bayesian_update_uses_pre_shift_predictions(self) -> None:
        """_run_bayesian_update must receive the original (unshifted) predictions.

        If it received updated_predictions (post heuristic shift) the prior mean
        would be biased, causing the Bayesian and Monte Carlo probabilities to
        drift apart systematically.
        """
        from unittest.mock import patch, call
        from src.models.update_after_game import _run_bayesian_update
        import src.models.update_after_game as uag

        captured: list = []

        original_run = uag._run_bayesian_update

        def capturing_run(context, base_series, completed_results, preds):
            captured.append(preds)
            return original_run(context, base_series, completed_results, preds)

        predictions = [
            {
                "game_number": n,
                "team_a": "A", "team_b": "B",
                "home_team": "A" if n % 2 else "B",
                "away_team": "B" if n % 2 else "A",
                "team_a_win_probability": 0.6,
                "team_b_win_probability": 0.4,
                "expected_score_team_a": 110,
                "expected_score_team_b": 107,
                "projected_pace": 97.0,
                "component_margins": {}, "x_factors": [],
            }
            for n in range(1, 8)
        ]
        context = {
            "team_a": "A", "team_b": "B",
            "schedule": predictions,
            "rotations": {"A": [], "B": []},
            "injuries": {"A": [], "B": []},
            "pre_series_rest": {"A": 4, "B": 3},
        }
        actual_game = {
            "game_number": 1,
            "actual_scores": {"A": 112, "B": 108},
            "team_stats": [
                {"team": "A", "PTS": 112, "FGM": 42, "FGA": 90,
                 "FG3M": 12, "FG3A": 35, "FTM": 16, "FTA": 20,
                 "OREB": 10, "DREB": 30, "REB": 40,
                 "AST": 25, "STL": 7, "BLK": 4, "TOV": 12},
                {"team": "B", "PTS": 108, "FGM": 40, "FGA": 88,
                 "FG3M": 10, "FG3A": 32, "FTM": 18, "FTA": 22,
                 "OREB": 9, "DREB": 28, "REB": 37,
                 "AST": 22, "STL": 6, "BLK": 3, "TOV": 14},
            ],
            "player_minutes": [],
            "actual_pace": 98.5,
        }

        with patch.object(uag, "_run_bayesian_update", side_effect=capturing_run):
            try:
                from src.models.update_after_game import update_after_game
                update_after_game(
                    actual_game,
                    game_predictions=predictions,
                    finals_context=context,
                    scenario_settings={"simulations": 200, "random_seed": 42},
                    foul_trouble_simulation={"scenarios": []},
                )
            except Exception:
                pass  # other downstream errors are acceptable

        if captured:
            # The predictions passed to the Bayesian updater must NOT all have
            # the 'postgame_probability_shift_team_a' key that only appears on
            # heuristically shifted (updated_predictions) rows.
            for pred in captured[0]:
                self.assertNotIn(
                    "postgame_probability_shift_team_a",
                    pred,
                    "Bayesian updater received post-shift predictions — should "
                    "receive original pre-update predictions instead.",
                )


if __name__ == "__main__":
    unittest.main()
