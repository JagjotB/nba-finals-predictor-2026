"""Streamlit dashboard for the 2026 NBA Finals prediction engine."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("The Streamlit dashboard requires pandas.") from exc

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import build_finals_context, load_settings
from src.features.lineup_features import build_lineup_features, summarize_lineup_features
from src.features.matchup_features import build_matchup_edges
from src.features.playstyle_features import build_playstyle_profiles
from src.models.closing_lineup_model import predict_close_game_edge
from src.models.foul_trouble_simulator import simulate_foul_trouble_scenarios
from src.models.predict_game import predict_finals_games
from src.models.prediction_snapshot import model_provenance
from src.models.scenario_simulator import run_scenario_suite
from src.models.simulate_series import simulate_series
from src.models.train_player_model import (
    project_finals_players,
    reconcile_player_projections_to_team_scores,
    simulate_correlated_player_box_scores,
)
from src.models.uncertainty import add_game_uncertainty
from src.models.update_after_game import (
    _load_game_actuals,
    _run_bayesian_update,
    simulate_series_after_results,
)


MODEL_VERSION = "lr-xgb-ensemble-v4"
DEFAULT_SIMULATIONS = 100000
DEFAULT_SCENARIO_SIMULATIONS = 25000

SCENARIO_TOGGLES = {
    "Hot shooting": "team_a_three_point_surge",
    "Foul trouble": "team_a_center_foul_trouble",
    "Slow pace": "slow_half_court_series",
    "Fast pace": "fast_transition_series",
    "Rebounding dominance": "team_a_wins_rebounding",
    "Star doubled": "team_a_star_doubled",
    "Bench exposed": "team_a_bench_exposed",
}

EDGE_CARD_KEYS = {
    "Rim pressure": "rim_pressure_vs_rim_protection",
    "Pick-and-roll": "pick_and_roll_vs_screen_defense",
    "Isolation": "isolation_vs_perimeter_defense",
    "Transition": "transition_offense_vs_transition_defense",
    "Corner threes": "corner_3_vs_corner_3_prevention",
    "Rebounding": "offensive_rebounding_vs_defensive_rebounding",
    "Foul pressure": "free_throw_pressure_vs_foul_discipline",
}

EDGE_EXPLANATIONS = {
    "Rim pressure": "How often a team attacks the basket vs how well the defense protects it",
    "Pick-and-roll": "The most common NBA play - ball-handler uses a screen to create space",
    "Isolation": "One-on-one matchups where a player attacks their defender solo",
    "Transition": "Fast break scoring - pushing the ball before the defense is set",
    "Corner threes": "Three-point shots from the corners of the court (shorter distance)",
    "Rebounding": "Grabbing missed shots to get extra possessions",
    "Foul pressure": "Drawing free throws by attacking the basket aggressively",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cache_data(**kwargs: Any) -> Any:
    if st is None:
        def decorator(function: Any) -> Any:
            return function
        return decorator
    return st.cache_data(**kwargs)


def _find_completed_game_numbers() -> list[int]:
    games_dir = PROJECT_ROOT / "data" / "processed" / "finals_games"
    if not games_dir.exists():
        return []
    completed = []
    for f in sorted(games_dir.glob("game_*_team_traditional.csv")):
        try:
            completed.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(completed)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def _score_label(score: Any) -> str:
    value = _as_float(score, 0.0)
    return f"+{value:.0f}" if value > 0 else f"{value:.0f}"


def _metric_record(profile: dict[str, Any], side: str, metric: str) -> dict[str, Any]:
    return profile.get(side, {}).get("metrics", {}).get(
        metric, {"value": 0.0, "score": 50.0, "label": metric.replace("_", " ")},
    )


def _lineup_by_type(
    lineups: list[dict[str, Any]], lineup_type: str
) -> dict[str, Any] | None:
    return next((l for l in lineups if l.get("lineup_type") == lineup_type), None)


def _team_pair(context: dict[str, Any]) -> tuple[str, str]:
    return str(context["team_a"]), str(context["team_b"])


def _ml_model_status() -> dict[str, Any]:
    """Check whether the trained ML model and live data are available."""
    status: dict[str, Any] = {
        "model_trained": False,
        "model_accuracy": None,
        "model_training_rows": None,
        "live_team_stats": False,
        "live_player_stats": False,
        "live_lineup_stats": False,
        "bayesian_active": True,
    }
    try:
        from src.models.game_model import load_model, MODEL_PATH
        bundle = load_model()
        if bundle:
            status["model_trained"] = True
            m = bundle.get("metrics", {})
            validation = bundle.get("validation_metrics", {})
            status["model_accuracy"] = validation.get("accuracy") or m.get("holdout_accuracy")
            status["model_brier"] = validation.get("brier_score") or m.get("holdout_brier")
            status["model_training_rows"] = m.get("train_size")
        report_path = PROJECT_ROOT / "outputs" / "reports" / "walk_forward_backtest.json"
        meta_path = PROJECT_ROOT / "data" / "processed" / "stats_cache" / "meta_model.json"
        if report_path.exists():
            import json
            report = json.loads(report_path.read_text(encoding="utf-8"))
            # Use ensemble metrics — that is the production model
            ens_metrics = report.get("overall", {}).get("ensemble", {})
            if ens_metrics.get("accuracy"):
                status["model_accuracy"] = ens_metrics["accuracy"]
                status["model_brier"] = ens_metrics.get("brier_score", status.get("model_brier"))
                status["model_ece"] = ens_metrics.get("expected_calibration_error")
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            status["validated_components"] = meta.get("validated_components", [])
    except Exception:
        pass

    cache_dir = PROJECT_ROOT / "data" / "processed" / "stats_cache"
    status["live_team_stats"] = (cache_dir / "team_stats_2025-26_Playoffs.json").exists()
    status["live_player_stats"] = (cache_dir / "player_stats_2025-26_Playoffs.json").exists()
    status["live_lineup_stats"] = (cache_dir / "lineup_stats_all_2025-26_Playoffs.json").exists()
    return status


# ---------------------------------------------------------------------------
# Data bundle
# ---------------------------------------------------------------------------

def _data_version() -> str:
    """Return a hash of key data file mtimes so cache busts when data changes."""
    import hashlib
    paths = [
        PROJECT_ROOT / "data" / "processed" / "finals_games",
        PROJECT_ROOT / "data" / "processed" / "finals_context" / "projected_rotations.csv",
        PROJECT_ROOT / "data" / "processed" / "stats_cache" / "team_stats_2025-26_Playoffs.json",
        PROJECT_ROOT / "config" / "model_weights.yaml",
    ]
    sig = ""
    for p in paths:
        if p.is_dir():
            sig += "".join(str(f.stat().st_mtime) for f in sorted(p.glob("*")) if f.is_file())
        elif p.exists():
            sig += str(p.stat().st_mtime)
    return hashlib.md5(sig.encode()).hexdigest()[:8]


@cache_data(show_spinner="Building predictions - this takes about 15 seconds...")
def load_dashboard_bundle(series_simulations: int, _data_ver: str = "") -> dict[str, Any]:
    settings = load_settings()
    context = build_finals_context()
    team_a, team_b = _team_pair(context)
    player_projections = project_finals_players(
        context, matchup_adjustments=context.get("active_players"),
    )
    playstyle_profiles = build_playstyle_profiles(context)
    matchup_edges = build_matchup_edges(context, playstyle_profiles=playstyle_profiles)
    lineup_features = build_lineup_features(context, player_projections=player_projections)
    clutch_prediction = predict_close_game_edge(
        context, player_projections=player_projections, lineup_features=lineup_features,
    )
    foul_trouble_simulation = simulate_foul_trouble_scenarios(
        context,
        player_projections=player_projections,
        playstyle_profiles=playstyle_profiles,
    )
    game_predictions = predict_finals_games(
        context, player_projections=player_projections,
        playstyle_profiles=playstyle_profiles, matchup_edges=matchup_edges,
        lineup_features=lineup_features, clutch_prediction=clutch_prediction,
        foul_trouble_simulation=foul_trouble_simulation,
    )
    game_predictions = [add_game_uncertainty(g) for g in game_predictions]
    expected_scores = {
        team_a: sum(game["expected_score_team_a"] for game in game_predictions) / max(len(game_predictions), 1),
        team_b: sum(game["expected_score_team_b"] for game in game_predictions) / max(len(game_predictions), 1),
    }
    player_projections = reconcile_player_projections_to_team_scores(
        player_projections,
        expected_scores,
    )
    player_outcome_simulation = simulate_correlated_player_box_scores(
        player_projections,
        simulations=1000,
        random_seed=int(settings.get("project", {}).get("random_seed", 42)),
    )

    sim_settings = {
        "simulations": int(series_simulations),
        "random_seed": int(settings.get("project", {}).get("random_seed", 42)),
    }

    completed_game_numbers = _find_completed_game_numbers()
    series_score: dict[str, int] | None = None
    bayesian_series: dict[str, Any] | None = None

    if completed_game_numbers:
        completed_results = []
        for game_num in completed_game_numbers:
            actual_game = _load_game_actuals(game_num)
            scores = actual_game.get("actual_scores") or {}
            if scores:
                completed_results.append(
                    {
                        "game_number": game_num,
                        "winner": max(scores, key=scores.get),
                        "actual_scores": scores,
                    }
                )
        series_simulation = simulate_series_after_results(
            game_predictions,
            completed_results,
            context,
            scenario_settings=sim_settings,
            foul_trouble_simulation=foul_trouble_simulation,
        )
        series_score = series_simulation.get("series_score")
        bayesian_series = _run_bayesian_update(
            context, series_simulation, completed_results, game_predictions
        )
    else:
        series_simulation = simulate_series(
            game_predictions,
            context,
            foul_trouble_simulation=foul_trouble_simulation,
            scenario_settings=sim_settings,
        )

    return {
        "settings": settings,
        "context": context,
        "team_a": team_a,
        "team_b": team_b,
        "player_projections": player_projections,
        "player_outcome_simulation": player_outcome_simulation,
        "playstyle_profiles": playstyle_profiles,
        "matchup_edges": matchup_edges,
        "lineup_features": lineup_features,
        "lineup_summary": summarize_lineup_features(lineup_features),
        "clutch_prediction": clutch_prediction,
        "foul_trouble_simulation": foul_trouble_simulation,
        "game_predictions": game_predictions,
        "series_simulation": series_simulation,
        "series_score": series_score,
        "bayesian_series": bayesian_series,
        "completed_games": completed_game_numbers,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ml_status": _ml_model_status(),
        "provenance": model_provenance(),
    }


# ---------------------------------------------------------------------------
# TAB 1 -The Pick (casual-friendly overview)
# ---------------------------------------------------------------------------

def _series_confidence_range(
    series: dict[str, Any], games: list[dict[str, Any]]
) -> tuple[float, float]:
    center = _as_float(series.get("team_a_series_win_probability"), 0.5)
    widths = []
    for game in games:
        rng = game.get("team_a_win_probability_range") or {}
        low = _as_float(rng.get("low"), game.get("team_a_win_probability", 0.5))
        high = _as_float(rng.get("high"), game.get("team_a_win_probability", 0.5))
        widths.append((high - low) / 2.0)
    width = min((sum(widths) / max(len(widths), 1)) * 1.20, 0.12)
    return max(center - width, 0.01), min(center + width, 0.99)


def render_the_pick(bundle: dict[str, Any]) -> None:
    team_a, team_b = bundle["team_a"], bundle["team_b"]
    series = bundle["series_simulation"]
    completed = bundle["completed_games"]
    series_score = bundle.get("series_score") or {}

    fav_prob = series["team_a_series_win_probability"]
    und_prob = series["team_b_series_win_probability"]
    favorite = team_a if fav_prob >= 0.5 else team_b
    underdog = team_b if favorite == team_a else team_a
    fav_pct = max(fav_prob, und_prob)
    low, high = _series_confidence_range(series, bundle["game_predictions"])

    # Hero banner
    st.markdown("## Who wins the 2026 NBA Finals?")

    nyk_w = series_score.get(team_a, 0)
    sas_w = series_score.get(team_b, 0)
    score_html = (
        f"<span style='background:#1a3a1a; color:#2ecc71; font-size:1.4rem; font-weight:800; "
        f"padding:4px 18px; border-radius:20px; margin:0 6px;'>{team_a} {nyk_w}</span>"
        f"<span style='color:#888; font-size:1.1rem;'>–</span>"
        f"<span style='background:#3a1a1a; color:#e74c3c; font-size:1.4rem; font-weight:800; "
        f"padding:4px 18px; border-radius:20px; margin:0 6px;'>{team_b} {sas_w}</span>"
    ) if completed else ""

    col_fav, col_vs, col_und = st.columns([2, 1, 2])
    with col_fav:
        st.markdown(
            f"<div style='text-align:center; font-size:3rem; font-weight:900; letter-spacing:-1px;'>{favorite}</div>"
            f"<div style='text-align:center; font-size:2rem; color:#2ecc71; font-weight:800; margin-top:4px;'>{_pct(fav_pct, 0)}</div>"
            f"<div style='text-align:center; font-size:0.85rem; color:#888; margin-top:2px;'>to win the series</div>",
            unsafe_allow_html=True,
        )
    with col_vs:
        st.markdown(
            f"<div style='text-align:center; padding-top:0.6rem;'>"
            f"<div style='font-size:1.8rem; color:#555;'>vs</div>"
            f"<div style='margin-top:10px;'>{score_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_und:
        st.markdown(
            f"<div style='text-align:center; font-size:3rem; font-weight:900; letter-spacing:-1px;'>{underdog}</div>"
            f"<div style='text-align:center; font-size:2rem; color:#e74c3c; font-weight:800; margin-top:4px;'>{_pct(min(fav_prob, und_prob), 0)}</div>"
            f"<div style='text-align:center; font-size:0.85rem; color:#888; margin-top:2px;'>to win the series</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # Key stats in plain English
    most_likely = series["most_likely_result"]
    dist = series.get("result_distribution", [])
    sorted_dist = sorted(dist, key=lambda r: float(r["probability"]), reverse=True)
    top_prob = float(sorted_dist[0]["probability"]) if sorted_dist else 0.0
    second = sorted_dist[1] if len(sorted_dist) > 1 else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Most likely outcome",
        most_likely,
        f"{top_prob:.0%} of simulations",
    )
    col2.metric(
        "2nd most likely",
        second["result"] if second else "—",
        f"{float(second['probability']):.0%} of simulations" if second else "",
    )
    col3.metric(
        "Confidence range",
        f"{_pct(low, 0)} – {_pct(high, 0)}",
        "Probability band across simulated outcomes",
    )
    col4.metric(
        "Simulations run",
        f"{series['simulations']:,}",
        "Series paths modelled to build these odds",
    )

    st.markdown("---")

    # Plain-English breakdown
    st.markdown("### Why does the model pick this?")

    fav_reasons = _favorite_reasons(bundle, favorite)
    st.markdown(
        f"**{favorite}** is favored because they have an edge in: "
        + ", ".join(f"**{r}**" for r in fav_reasons) + "."
    )

    # X-factor — use series data when games have been played
    if completed:
        series_xf = _series_xfactors_from_data(bundle)
        if series_xf:
            st.info(f"**Biggest wildcard:** {series_xf[0]}")
    else:
        x_factors = []
        for game in bundle["game_predictions"]:
            x_factors.extend(game.get("x_factors", []))
        if x_factors:
            st.info(f"**Biggest wildcard:** {x_factors[0]}")

    # What the underdog needs
    flip = _opponent_flip_paths(bundle, underdog)
    st.markdown(
        f"**{underdog}** can turn this around by: "
        + ", ".join(f"**{p}**" for p in flip) + "."
    )

    st.markdown("---")

    # Result distribution as a simple table
    st.markdown("### How does each outcome play out?")
    dist = sorted(
        [r for r in series.get("result_distribution", []) if float(r["probability"]) >= 0.01],
        key=lambda r: -float(r["probability"]),
    )
    for row in dist:
        prob = float(row["probability"])
        result = row["result"]
        winner = result.split(" in ")[0]
        bar_w = int(prob * 100)
        is_fav_win = winner == favorite
        bar_color = "#2ecc71" if is_fav_win else "#e74c3c"
        text_color = "#2ecc71" if is_fav_win else "#e74c3c"
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:8px;'>"
            f"<div style='width:120px; font-weight:700; color:{text_color};'>{result}</div>"
            f"<div style='flex:1; background:#222; border-radius:4px; height:22px; position:relative;'>"
            f"<div style='width:{bar_w}%; background:{bar_color}; height:100%; border-radius:4px; opacity:0.85;'></div>"
            f"<span style='position:absolute; left:8px; top:2px; font-size:0.8rem; font-weight:700; color:#fff;'>"
            f"{row['percentage']}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "Odds update automatically after each game. "
        "Hit 'Refresh data' in the sidebar after any game is played."
    )


# ---------------------------------------------------------------------------
# TAB 2 -Game by Game
# ---------------------------------------------------------------------------

def _render_game_reasoning(game: dict[str, Any], team_a: str, team_b: str, series_xf: list[str] | None = None) -> None:
    """Plain-English breakdown of what's driving the model's prediction for one game."""
    prob_a = _as_float(game.get("team_a_win_probability"), 0.5)
    baseline = _as_float(game.get("baseline_probability_team_a"), prob_a)
    net_rating_base = _as_float(game.get("net_rating_probability_team_a"), prob_a)
    home = str(game.get("home_team", ""))
    margins = game.get("component_margins") or {}

    # ── Row 1: neutral-site quality ──────────────────────────────────────────
    neutral_pct = round(baseline * 100)
    if neutral_pct > 52:
        quality_line = f"At a neutral site, the model gives **{team_a}** the edge - **{neutral_pct}%** - based on net rating, shooting efficiency, pace, and turnovers."
    elif neutral_pct < 48:
        quality_line = f"At a neutral site, the model gives **{team_b}** the edge - **{100 - neutral_pct}%** - based on net rating, shooting efficiency, pace, and turnovers."
    else:
        quality_line = f"At a neutral site this game is essentially a **coin flip** ({neutral_pct}%) based on team quality alone."
    st.markdown(quality_line)

    # ── Row 2: home court ────────────────────────────────────────────────────
    home_shift = round((net_rating_base - 0.5) * 100 - (baseline - 0.5) * 100)
    if home == team_a:
        st.markdown(f"🏠 **Home court** ({team_a} at home) pushes their probability up.")
    elif home == team_b:
        st.markdown(f"🏠 **Home court** ({team_b} at home) pulls {team_a}'s probability down.")

    # ── Row 3: component margins ─────────────────────────────────────────────
    MARGIN_LABELS = {
        "player_projection":  ("👤 Player matchup",   "How each team's projected player stats stack up against the other"),
        "matchup_edge":       ("🎯 Shooting matchup",  "Which team has the better shooting angles and defensive assignments"),
        "lineup_edge":        ("📋 Lineup combinations", "How well each team's key lineups perform when they're on the court together"),
        "clutch_edge":        ("⏱️ Late-game situations", "Which team performs better when the game is close in the final minutes"),
        "foul_trouble_risk":  ("⚠️ Foul trouble risk",  "Whether key players on either team are likely to pick up fouls and sit"),
        "injury_edge":        ("🩹 Injury/availability", "Impact of any players missing time or playing limited minutes"),
    }

    st.markdown("**What's moving the needle:**")
    has_any = False
    for key, (label, tooltip) in MARGIN_LABELS.items():
        val = _as_float(margins.get(key), 0.0)
        if abs(val) < 0.05:
            continue
        has_any = True
        beneficiary = team_a if val > 0 else team_b
        pts = abs(round(val, 1))
        arrow = "▲" if val > 0 else "▼"
        color = "#2ecc71" if val > 0 else "#e74c3c"
        st.markdown(
            f"{label} &nbsp; <span style='color:{color}; font-weight:700'>{arrow} {pts} pts → {beneficiary}</span><br>"
            f"<span style='color:#888; font-size:0.85rem'>{tooltip}</span>",
            unsafe_allow_html=True,
        )
    if not has_any:
        st.caption("No component has a meaningful edge - this game is driven almost entirely by team quality and home court.")

    # ── Row 4: x-factors ─────────────────────────────────────────────────────
    display_xf = series_xf if series_xf else (game.get("x_factors") or [])
    if display_xf:
        st.markdown("**Wildcards to watch:**")
        for xf in display_xf[:3]:
            st.caption(f"• {xf}")


def render_game_by_game(bundle: dict[str, Any]) -> None:
    team_a, team_b = bundle["team_a"], bundle["team_b"]
    completed = set(bundle["completed_games"])
    predictions = bundle["game_predictions"]

    st.markdown("## Game-by-Game Predictions")
    st.caption(
        "Each game has a projected winner and score. "
        "Completed games show the actual result. "
        "Future games show what the model expects."
    )

    # Pre-load actual results for completed games
    _actuals: dict[int, dict] = {}
    for _gn in completed:
        try:
            from src.models.update_after_game import _load_game_actuals
            _actuals[_gn] = _load_game_actuals(_gn)
        except Exception:
            pass

    series_xf = _series_xfactors_from_data(bundle)

    for game in predictions:
        gn = int(game["game_number"])
        home = str(game.get("home_team", ""))
        away = str(game.get("away_team", ""))
        prob_a = _as_float(game.get("team_a_win_probability"), 0.5)
        prob_b = 1.0 - prob_a
        fav = team_a if prob_a >= 0.5 else team_b
        fav_pct = max(prob_a, prob_b)
        score_a = game.get("expected_score_team_a", 0)
        score_b = game.get("expected_score_team_b", 0)
        date = str(game.get("date", ""))[:10]
        is_done = gn in completed

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
            with c1:
                st.markdown(f"### G{gn}")
                if date:
                    st.caption(date)

            if is_done:
                actual = _actuals.get(gn, {})
                scores = actual.get("actual_scores", {})
                winner = max(scores, key=scores.get) if scores else None
                model_correct = winner == fav if winner else None
                sa_actual = scores.get(team_a, "?")
                sb_actual = scores.get(team_b, "?")

                with c2:
                    st.markdown(f"**{away}** @ **{home}**")
                    if winner:
                        badge_color = "#2ecc71" if model_correct else "#e74c3c"
                        badge_text = "MODEL CORRECT" if model_correct else "MODEL WRONG"
                        st.markdown(
                            f"<span style='background:{badge_color}22; color:{badge_color}; "
                            f"font-size:0.75rem; font-weight:700; padding:2px 10px; border-radius:10px; "
                            f"border:1px solid {badge_color}55;'>{badge_text}</span>",
                            unsafe_allow_html=True,
                        )
                with c3:
                    if scores:
                        st.markdown(
                            f"<div style='font-size:1.5rem; font-weight:800; margin-top:4px;'>"
                            f"{team_a} {sa_actual} – {team_b} {sb_actual}</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Winner: **{winner}**")
                    else:
                        st.caption("Result not available")
                with c4:
                    st.caption(f"Model had picked: **{fav}** at {_pct(fav_pct)}")
                    rng = game.get("team_a_win_probability_range") or {}
                    low_p = _as_float(rng.get("low"), prob_a)
                    high_p = _as_float(rng.get("high"), prob_a)
                    st.caption(f"Range: {_pct(low_p, 0)}–{_pct(high_p, 0)} for {team_a}")
            else:
                with c2:
                    st.markdown(f"**{away}** @ **{home}**")
                    proj_home = score_a if home == team_a else score_b
                    proj_away = score_b if home == team_a else score_a
                    st.caption(f"Projected: {home} {proj_home} – {away} {proj_away}")
                with c3:
                    st.markdown(f"Model: **{fav}** wins")
                    st.progress(fav_pct, text=f"{_pct(fav_pct)} confidence")
                with c4:
                    edges = game.get("top_edges") or []
                    if edges:
                        st.caption(edges[0][:90])
                    rng = game.get("team_a_win_probability_range") or {}
                    low_p = _as_float(rng.get("low"), prob_a)
                    high_p = _as_float(rng.get("high"), prob_a)
                    st.caption(f"Range: {_pct(low_p, 0)}–{_pct(high_p, 0)} for {team_a}")
                with st.expander("Why does the model say this?"):
                    _render_game_reasoning(game, team_a, team_b, series_xf=series_xf)

    st.markdown("---")

    # Matchup edges simplified
    st.markdown("## Who Has the Advantage Where?")
    st.caption(
        "These cards show which team has an edge in each style of play. "
        "+1 or +2 = advantage, −1 or −2 = disadvantage, 0 = even."
    )

    rows = _build_matchup_card_rows(bundle)
    for i in range(0, len(rows), 3):
        cols = st.columns(3)
        for col, row in zip(cols, rows[i:i + 3]):
            with col:
                with st.container(border=True):
                    score_val = _as_float(row["raw_score"], 0.0)
                    color = "#2ecc71" if score_val > 0 else "#e74c3c" if score_val < 0 else "#888"
                    arrow = "▲" if score_val > 0 else "▼" if score_val < 0 else "-"
                    st.markdown(
                        f"**{row['label']}** &nbsp; "
                        f"<span style='color:{color}; font-size:1.2rem; font-weight:700;'>{arrow} {row['value']}</span>  "
                        f"<span style='color:#aaa; font-size:0.85rem;'>({row['caption']})</span>",
                        unsafe_allow_html=True,
                    )
                    # Plain English explanation of what this means
                    st.caption(EDGE_EXPLANATIONS.get(row["label"], ""))
                    if row.get("detail"):
                        with st.expander("Details"):
                            st.write(row["detail"])


def _build_matchup_card_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, key in EDGE_CARD_KEYS.items():
        edge = _best_edge_for_key(bundle["matchup_edges"], key)
        if edge:
            rows.append({
                "label": label,
                "value": _score_label(edge.get("matchup_score")),
                "raw_score": edge.get("matchup_score", 0),
                "caption": f"{edge.get('offensive_team')} offense",
                "detail": edge.get("explanation"),
            })

    non_star_a = _lineup_by_type(bundle["lineup_features"].get(bundle["team_a"], []), "non_star_minutes") or {}
    non_star_b = _lineup_by_type(bundle["lineup_features"].get(bundle["team_b"], []), "non_star_minutes") or {}
    bench_edge = _as_float(non_star_a.get("adjusted_lineup_net_rating")) - _as_float(non_star_b.get("adjusted_lineup_net_rating"))
    rows.append({
        "label": "Bench depth",
        "value": f"{bundle['team_a'] if bench_edge >= 0 else bundle['team_b']} {_score_label(abs(bench_edge))}",
        "raw_score": bench_edge,
        "caption": "non-starter lineup edge",
        "detail": "How much better the bench players perform when the stars rest.",
    })

    clutch = bundle["clutch_prediction"]
    fav_c = clutch.get("favorite", "Even")
    edge_c = _as_float(clutch.get("favorite_edge_per_100"), 0.0)
    rows.append({
        "label": "Closing time",
        "value": f"{fav_c} +{edge_c:.1f}" if fav_c != "Even" else "Even",
        "raw_score": edge_c if fav_c == bundle["team_a"] else -edge_c,
        "caption": "close-game edge",
        "detail": "Which team performs better in games that come down to the wire in the final 5 minutes.",
    })
    return rows


def _best_edge_for_key(matchup_edges: dict[str, Any], key: str) -> dict[str, Any] | None:
    candidates = [
        edge
        for comparison in matchup_edges.get("comparisons", {}).values()
        for edge in comparison.get("edges", [])
        if edge.get("key") == key
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (abs(r.get("matchup_score", 0)), abs(r.get("raw_edge", 0))))


# ---------------------------------------------------------------------------
# TAB 3 -Player Breakdown
# ---------------------------------------------------------------------------

def render_player_breakdown(bundle: dict[str, Any]) -> None:
    team_a, team_b = bundle["team_a"], bundle["team_b"]
    st.markdown("## Player Projections")
    st.caption(
        "Stats are projected per game based on each player's recent performance, "
        "their matchup, and how many minutes we expect them to play. "
        "PIE (Player Impact Estimate) shows overall impact - league average is 10%."
    )

    tabs = st.tabs([team_a, team_b])
    for tab, team in zip(tabs, [team_a, team_b]):
        with tab:
            players = bundle["player_projections"].get(team, [])
            rows = []
            for p in players:
                if _as_float(p.get("minutes")) < 3:
                    continue
                pie = _as_float(p.get("pie"), 0.0)
                pie_label = "Elite" if pie > 0.20 else "Above avg" if pie > 0.13 else "Average" if pie > 0.08 else "Below avg"
                eff_mult = _as_float(p.get("efficiency_multiplier"), 1.0)
                adj = f"+{(eff_mult-1)*100:.0f}%" if eff_mult > 1.01 else f"{(eff_mult-1)*100:.0f}%" if eff_mult < 0.99 else "neutral"
                rows.append({
                    "Player": p.get("player"),
                    "Role": str(p.get("role", "")).title(),
                    "Minutes": round(_as_float(p.get("minutes")), 1),
                    "Points": round(_as_float(p.get("points")), 1),
                    "Rebounds": round(_as_float(p.get("rebounds")), 1),
                    "Assists": round(_as_float(p.get("assists")), 1),
                    "Turnovers": round(_as_float(p.get("turnovers")), 1),
                    "PIE": f"{pie:.1%}" if pie else "N/A",
                    "Impact level": pie_label,
                    "Matchup adj": adj,
                    "Confidence": str(p.get("rotation_confidence", "medium")).title(),
                })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.caption(
                "**How to read this:** Minutes = projected playing time. "
                "PIE = Player Impact Estimate (10% is league average). "
                "Matchup adj = how much the specific opponent affects this player's production."
            )


# ---------------------------------------------------------------------------
# TAB 4 -ML & Model Info
# ---------------------------------------------------------------------------

def render_ml_panel(bundle: dict[str, Any]) -> None:
    import json as _json

    ml = bundle.get("ml_status", {})
    team_a = bundle["team_a"]
    team_b = bundle["team_b"]

    st.markdown("## How the Model Works")
    st.caption(
        "Architecture, training results, signal breakdown, and live data status "
        "for every component powering these predictions."
    )

    # ── 1. Key performance metrics ────────────────────────────────────────────
    acc = ml.get("model_accuracy")
    brier = ml.get("model_brier")
    ece = ml.get("model_ece")
    rows_trained = ml.get("model_training_rows", 0)
    games_trained = (rows_trained // 2) if rows_trained else 915

    st.markdown("### Model Performance")

    if not ml.get("model_trained"):
        st.warning(
            "ML model not loaded — using statistical fallback. "
            "Run `python scripts/validate_models.py` to train."
        )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.metric(
                "Walk-forward accuracy",
                f"{acc:.1%}" if acc else "—",
                delta=f"+{(acc - 0.5) * 100:.1f}% vs coin flip" if acc else None,
            )
    with m2:
        with st.container(border=True):
            st.metric(
                "Brier score",
                f"{brier:.4f}" if brier else "—",
                delta="lower = better  (random = 0.25)",
                delta_color="off",
            )
    with m3:
        with st.container(border=True):
            st.metric(
                "Calibration (ECE)",
                f"{ece:.4f}" if ece else "—",
                delta="avg probability error",
                delta_color="off",
            )
    with m4:
        with st.container(border=True):
            st.metric(
                "Training games",
                f"{games_trained:,}",
                delta="11 playoff seasons",
                delta_color="off",
            )

    st.caption(
        "Walk-forward validation: model is trained on older seasons and tested on seasons it has never seen. "
        "This prevents the accuracy number from being inflated by overfitting."
    )

    # Performance comparison bar chart
    st.markdown("**How it compares to simpler approaches** (674 held-out games, 4 seasons)")
    comparison_df = pd.DataFrame({
        "Model": [
            "ELO Baseline",
            "Net Rating Only",
            "Logistic Regression",
            "XGBoost",
            "LR + XGBoost Ensemble",
        ],
        "Accuracy": [0.5579, 0.5964, 0.6113, 0.6157, 0.6172],
    }).set_index("Model")
    st.bar_chart(comparison_df, y="Accuracy", use_container_width=True)

    st.markdown("---")

    # ── 2. Architecture ───────────────────────────────────────────────────────
    st.markdown("### Architecture")
    st.caption(
        "Two ML models produce the baseline probability. "
        "Five signal adjustments then shift it up or down."
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        with st.container(border=True):
            st.markdown("**Logistic Regression**")
            st.metric("Accuracy", "61.1%", delta="16 features", delta_color="off")
            st.caption(
                "Trained on blended regular-season + playoff stats. "
                "Net rating, eFG%, TOV%, OReb%, FTA rate, pace, home court, travel."
            )
    with a2:
        with st.container(border=True):
            st.markdown("**XGBoost Classifier**")
            st.metric("Accuracy", "61.6%", delta="24 features", delta_color="off")
            st.caption(
                "All LR features plus injury signal, offensive/defensive rating split, "
                "rest advantage, and playoff experience."
            )
    with a3:
        with st.container(border=True):
            st.markdown("**Ensemble (50 / 50 blend)**")
            st.metric("Accuracy", "61.7%", delta="production model", delta_color="off")
            st.caption(
                "LR and XGBoost probabilities averaged, then five component "
                "adjustments shift the final number."
            )

    st.markdown("---")

    # ── 3. Signal breakdown ───────────────────────────────────────────────────
    st.markdown("### The Seven Signals")
    st.caption(
        "Each signal is computed from live NBA data and either baked into the ML baseline "
        "or applied as a weighted shift on top of it."
    )

    signals = [
        ("ML Baseline", "Anchor", "Team quality across 24 stats", "#1f77b4",
         "LR + XGBoost blend trained on 915 playoff games. Net rating, shooting, turnovers, pace, injury."),
        ("Player Projections", "0.45×", "Projected scoring edge", "#2ca02c",
         "Regular-season / playoff rate blend × projected minutes × PIE efficiency multiplier."),
        ("Lineup Strength", "0.25×", "Depth and closing lineup edge", "#9467bd",
         "2-man and 5-man lineup net ratings. Accounts for rotation quality and late-game units."),
        ("Matchup Edges", "0.20×", "Style-specific advantages", "#8c564b",
         "Pick-and-roll, transition, corner 3s. Flags structural mismatches between play styles."),
        ("Clutch Edge", "0.10×", "Close-game performance", "#e377c2",
         "Applied only when game is expected to be close (30% of games). Uses closing lineup ratings."),
        ("Injury / Availability", "Baseline + upstream", "Key player absences", "#d62728",
         "Baked into XGBoost via pts-share differential. Also adjusts upstream player projections."),
        ("Foul Trouble", "0.10×", "Star player availability risk", "#ff7f0e",
         "Monte Carlo simulation of 100,000 games. Penalises teams whose key players foul out often."),
    ]

    col_l, col_r = st.columns(2)
    for i, (name, weight, what, color, detail) in enumerate(signals):
        col = col_l if i % 2 == 0 else col_r
        with col:
            st.markdown(
                f"<div style='border-left:4px solid {color}; padding:10px 14px; "
                f"margin-bottom:10px; background:#1a1a1a; border-radius:0 6px 6px 0;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                f"<span style='font-weight:700; font-size:0.95rem;'>{name}</span>"
                f"<span style='font-size:0.8rem; color:#aaa; background:#2a2a2a; "
                f"padding:2px 8px; border-radius:12px;'>{weight}</span>"
                f"</div>"
                f"<div style='font-size:0.8rem; color:#ccc; margin-top:2px;'>{what}</div>"
                f"<div style='font-size:0.75rem; color:#888; margin-top:4px;'>{detail}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── 4. Spread / margin model ──────────────────────────────────────────────
    st.markdown("### Spread Model")
    st.caption(
        "A separate XGBoost regression model trained to predict point margin directly, "
        "independent of the win-probability model. "
        "Used to show the projected spread and a cross-check probability on each game."
    )

    margin_path = PROJECT_ROOT / "data" / "processed" / "stats_cache" / "xgb_margin_model.json"
    if margin_path.exists():
        mm = _json.loads(margin_path.read_text(encoding="utf-8"))
        sm1, sm2, sm3 = st.columns(3)
        with sm1:
            with st.container(border=True):
                st.metric("Margin MAE", f"{mm.get('margin_mae', 0):.1f} pts",
                          delta="avg prediction error", delta_color="off")
        with sm2:
            with st.container(border=True):
                st.metric("Residual std", f"{mm.get('residual_std', 0):.1f} pts",
                          delta="68% CI width", delta_color="off")
        with sm3:
            with st.container(border=True):
                st.metric("Raw margin std", f"{mm.get('margin_std_raw', 0):.1f} pts",
                          delta="how noisy NBA games are", delta_color="off")
        st.caption(
            "Even Vegas narrows margin std to ~12 pts. An MAE of 11.8 pts is realistic — "
            "individual game outcomes have genuine irreducible randomness. "
            "The spread model's value is in direction and divergence, not in an exact number."
        )
    else:
        st.info("Spread model not trained. Run `python scripts/validate_models.py`.")

    st.markdown("---")

    # ── 5. Live data status ───────────────────────────────────────────────────
    st.markdown("### Live Data Status")
    st.caption("Green = using real NBA API data for this prediction. Red = using a statistical fallback.")

    d1, d2, d3, d4 = st.columns(4)

    def _status_card(ok: bool, label: str, detail: str) -> None:
        color = "#2ecc71" if ok else "#e74c3c"
        bg = "#0d2b1a" if ok else "#2b0d0d"
        icon = "✅" if ok else "⚠️"
        st.markdown(
            f"<div style='border:1px solid {color}; background:{bg}; border-radius:8px; "
            f"padding:12px; text-align:center;'>"
            f"<div style='font-size:1.4rem;'>{icon}</div>"
            f"<div style='font-weight:700; font-size:0.9rem; margin-top:4px;'>{label}</div>"
            f"<div style='font-size:0.75rem; color:#aaa; margin-top:2px;'>{detail}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with d1:
        _status_card(ml.get("live_team_stats", False), "Team Stats", "Off/def ratings · pace · eFG%")
    with d2:
        _status_card(ml.get("live_player_stats", False), "Player Stats", "Per-game stats · PIE · minutes")
    with d3:
        _status_card(ml.get("live_lineup_stats", False), "Lineup Stats", "2-man & 5-man net ratings")
    with d4:
        _status_card(ml.get("model_trained", False), "ML Model", "LR + XGBoost trained & loaded")

    st.markdown("---")

    # ── 6. Bayesian series updater ────────────────────────────────────────────
    st.markdown("### Bayesian Series Cross-Check")
    st.caption(
        "After games are played, a Bayesian model updates its estimate of each team's true strength "
        "based on actual results. An upset shifts beliefs more than an expected win. "
        "This runs independently of the main model — agreement confirms the call."
    )

    bayesian = bundle.get("bayesian_series")
    completed = bundle["completed_games"]
    series = bundle["series_simulation"]

    if bayesian and completed:
        mc_p = float(series["team_a_series_win_probability"])
        bt_p = float(bayesian.get("team_a_series_win_probability", 0.5))
        gap = abs(mc_p - bt_p) * 100

        col_mc, col_bt, col_gap = st.columns(3)
        with col_mc:
            with st.container(border=True):
                st.metric(f"{team_a} series win — MC simulation", _pct(mc_p))
                st.caption("Monte Carlo: 100,000 simulated series from current state")
        with col_bt:
            with st.container(border=True):
                st.metric(f"{team_a} series win — Bayesian", _pct(bt_p))
                st.caption("Posterior after updating on actual game results")
        with col_gap:
            with st.container(border=True):
                st.metric("Gap between methods", f"{gap:.1f} pts")
                if gap < 5:
                    st.caption("Models agree — strengthens the forecast.")
                else:
                    st.caption("Meaningful divergence — treat series estimate with more uncertainty.")

        agreement = "agree" if gap < 5 else "diverge"
        st.info(
            f"**After {len(completed)} game(s):** Monte Carlo says **{_pct(mc_p)}** for {team_a}, "
            f"Bayesian cross-check says **{_pct(bt_p)}**. "
            f"The two methods {agreement} ({gap:.1f} pt gap)."
        )
    elif not completed:
        st.info(
            "Bayesian updates activate after Game 1. Before any games, both methods use the pre-series model."
        )
    else:
        st.info(
            "Bayesian cross-check requires pre-game prediction snapshots — "
            "withheld rather than reconstructed from postgame data."
        )

    st.markdown("---")

    # ── 7. Series simulation ──────────────────────────────────────────────────
    st.markdown("### Series Simulation")
    st.caption(
        "100,000 simulated series from the current game state. "
        "Each game uses that game's win probability plus randomness — "
        "because a 60% favorite still loses 40% of the time."
    )

    series_data = bundle["series_simulation"]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Series win probability**")
        st.bar_chart(
            pd.DataFrame({
                "Team": [team_a, team_b],
                "Probability": [
                    series_data["team_a_series_win_probability"],
                    series_data["team_b_series_win_probability"],
                ],
            }).set_index("Team"),
            y="Probability",
        )
    with col_b:
        st.markdown("**Series length**")
        st.bar_chart(
            pd.DataFrame({
                "Games": [str(r["games"]) for r in series_data.get("series_length_distribution", [])],
                "Probability": [r["probability"] for r in series_data.get("series_length_distribution", [])],
            }).set_index("Games"),
            y="Probability",
        )
    with col_c:
        st.markdown("**Outcome distribution**")
        dist_rows = [
            {"Result": r["result"], "Probability": r["percentage"]}
            for r in series_data.get("result_distribution", [])
            if float(r["probability"]) > 0.01
        ]
        if dist_rows:
            st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.info(
        "**What-if scenarios** are in the **Deep Stats** tab — "
        "toggle foul trouble, pace, shooting, and other conditions to see how the series odds shift."
    )


# ---------------------------------------------------------------------------
# TAB 5 -Deep Stats
# ---------------------------------------------------------------------------

def render_deep_stats(bundle: dict[str, Any], selected_scenarios: tuple[str, ...], scenario_sims: int) -> None:
    team_a, team_b = bundle["team_a"], bundle["team_b"]

    st.markdown("## Team Playstyle Profiles")
    st.caption("How each team plays - based on real stats from their 2025-26 playoff run.")

    cols = st.columns(2)
    for col, team in zip(cols, [team_a, team_b]):
        profile = bundle["playstyle_profiles"].get(team, {})
        with col:
            with st.container(border=True):
                st.markdown(f"### {team}")
                off = profile.get("offense", {})
                defe = profile.get("defense", {})
                st.markdown("**Offense**")
                for trait in (off.get("summary") or []):
                    st.caption(f"• {trait}")
                st.markdown("**Defense**")
                for trait in (defe.get("summary") or []):
                    st.caption(f"• {trait}")

                # Key real stats
                pace_val = off.get("metrics", {}).get("pace", {}).get("value")
                oreb_val = off.get("metrics", {}).get("offensive_rebounding", {}).get("value")
                if pace_val:
                    st.metric("Pace (possessions/game)", f"{pace_val:.1f}", "League avg ~98")
                if oreb_val:
                    st.metric("Offensive rebounding %", f"{oreb_val:.1f}%")

    st.markdown("---")
    st.markdown("## Lineup Analysis")
    st.caption(
        "Net rating = points scored minus points allowed per 100 possessions. "
        "+10 means outscoring opponents by 10 per 100 possessions - excellent. "
        "These are blended with real 2-man and 5-man lineup data from the NBA API."
    )

    cols = st.columns(2)
    for col, team in zip(cols, [team_a, team_b]):
        lineups = bundle["lineup_features"].get(team, [])
        with col:
            with st.container(border=True):
                st.markdown(f"### {team}")
                for lineup in lineups:
                    ltype = lineup.get("lineup_type", "").replace("_", " ").title()
                    net = lineup.get("adjusted_lineup_net_rating", 0)
                    off = lineup.get("offensive_rating", 0)
                    defe = lineup.get("defensive_rating", 0)
                    players = lineup.get("players", [])
                    color = "#2ecc71" if net > 12 else "#f39c12" if net > 5 else "#e74c3c"
                    st.markdown(
                        f"**{ltype}** -"
                        f"<span style='color:{color};'>Net {net:+.1f}</span> "
                        f"(Off {off:.0f} / Def {defe:.0f})",
                        unsafe_allow_html=True,
                    )
                    if players:
                        st.caption(", ".join(players[:5]))

    st.markdown("---")
    st.markdown("## Scenario Simulator")
    st.caption("Toggle scenarios below to see how the series odds shift under different conditions.")

    _selected: list[str] = []
    sc_cols = st.columns(3)
    for i, (label, sid) in enumerate(SCENARIO_TOGGLES.items()):
        default = label in {"Hot shooting", "Foul trouble", "Slow pace", "Rebounding dominance"}
        if sc_cols[i % 3].checkbox(label, value=default, key=f"sc_{sid}"):
            _selected.append(sid)
    selected_scenarios = tuple(_selected)

    if not selected_scenarios:
        st.info("Select at least one scenario above to run simulations.")
        return

    with st.spinner("Running simulations..."):
        report = run_scenario_suite(
            scenarios=list(selected_scenarios),
            game_predictions=bundle["game_predictions"],
            finals_context=bundle["context"],
            scenario_settings={
                "simulations": scenario_sims,
                "random_seed": int(bundle["settings"].get("project", {}).get("random_seed", 42)),
            },
        )

    baseline_prob = float(
        bundle["series_simulation"].get("team_a_series_win_probability", 0.5)
    )
    st.markdown(
        f"**Baseline:** {team_a} wins series at **{_pct(baseline_prob)}** · "
        f"Each scenario below shows how that changes if the condition holds all series."
    )
    st.markdown("")

    sc_cols = st.columns(2)
    for i, row in enumerate(report.get("summary", [])):
        scenario_label = str(row.get("scenario", "")).replace("_", " ").title()
        prob_str = str(row.get("team_a_series_win_percentage", "")).replace("%", "")
        try:
            prob_val = float(prob_str) / 100
        except ValueError:
            prob_val = baseline_prob
        delta = prob_val - baseline_prob
        most_likely = str(row.get("most_likely_result", ""))
        bar_color = "#2ecc71" if delta > 0.01 else "#e74c3c" if delta < -0.01 else "#888"
        delta_str = f"{delta*100:+.1f} pts" if abs(delta) > 0.005 else "No change"
        arrow = "▲" if delta > 0.01 else "▼" if delta < -0.01 else "—"

        with sc_cols[i % 2]:
            with st.container(border=True):
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                    f"<span style='font-weight:700; font-size:0.95rem;'>{scenario_label}</span>"
                    f"<span style='font-size:1.3rem; font-weight:900; color:{bar_color};'>"
                    f"{_pct(prob_val, 0)}</span>"
                    f"</div>"
                    f"<div style='display:flex; justify-content:space-between; margin-top:4px;'>"
                    f"<span style='font-size:0.8rem; color:{bar_color}; font-weight:700;'>"
                    f"{arrow} {delta_str} vs baseline</span>"
                    f"<span style='font-size:0.75rem; color:#888;'>Most likely: {most_likely}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# TAB 0 -Next Game deep-dive
# ---------------------------------------------------------------------------

def _next_game(bundle: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first upcoming (not yet completed) game prediction."""
    completed = set(bundle.get("completed_games", []))
    for g in bundle["game_predictions"]:
        if int(g.get("game_number", 0)) not in completed:
            return g
    return None


def _component_explanation(key: str, value: float, team_a: str, team_b: str) -> str:
    edge_team = team_a if value >= 0 else team_b
    abs_val = abs(value)
    if abs_val < 0.3:
        return "Essentially neutral - neither team has a meaningful edge here."
    strength = "large" if abs_val >= 3 else "moderate" if abs_val >= 1 else "slight"
    labels = {
        "player_projection": f"{edge_team} projects to outperform on paper - "
            "better players in their minutes, weighted by PIE efficiency.",
        "matchup_edge": f"{edge_team} has a {strength} structural advantage in key "
            "play-type matchups (rim pressure, PnR, transition, rebounding).",
        "lineup_edge": f"{edge_team}'s lineup groups grade out stronger - "
            "real 2-man and 5-man net ratings from this season.",
        "clutch_edge": f"{edge_team} has the slight edge when games are close late - "
            "driven by closing lineup composition and star creation ability.",
        "injury_edge": f"Minute adjustments from injury/availability tilt slightly toward "
            f"{edge_team}.",
        "foul_trouble_risk": f"Foul-trouble scenarios tilt toward {edge_team} - "
            "the opponent's key player is more foul-prone in this matchup.",
    }
    return labels.get(key, f"{edge_team} has an edge here ({value:+.2f}).")


def render_next_game(bundle: dict[str, Any]) -> None:
    game = _next_game(bundle)
    team_a = bundle["team_a"]
    team_b = bundle["team_b"]
    series = bundle["series_simulation"]
    series_score = bundle.get("series_score") or {team_a: 0, team_b: 0}

    if game is None:
        st.info("The series is over. No upcoming games.")
        return

    gnum = int(game["game_number"])
    home = str(game.get("home_team", ""))
    away = str(game.get("away_team", ""))
    date_str = str(game.get("date", "TBD"))
    prob_a = _as_float(game.get("team_a_win_probability"), 0.5)
    prob_b = _as_float(game.get("team_b_win_probability"), 0.5)
    score_a = game.get("expected_score_team_a", "N/A")
    score_b = game.get("expected_score_team_b", "N/A")
    pace = game.get("projected_pace")
    margins = game.get("component_margins", {})
    prob_range = game.get("team_a_win_probability_range") or {}
    x_factors = game.get("x_factors", [])
    top_edges = game.get("top_edges", [])
    baseline = _as_float(game.get("baseline_probability_team_a"), prob_a)
    post_shift = game.get("postgame_probability_shift_team_a")

    sa = series_score.get(team_a, 0)
    sb = series_score.get(team_b, 0)
    fav_game = team_a if prob_a >= 0.5 else team_b
    dog_game = team_b if fav_game == team_a else team_a
    fav_prob_game = max(prob_a, prob_b)
    dog_prob_game = 1.0 - fav_prob_game
    home_label = f"{'home' if home == fav_game else 'away'}"

    # ── Hero pre-game card ────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0d1f2d 0%,#1a2a3a 100%); "
        f"border:1px solid #2a4a6a; border-radius:12px; padding:24px 28px; margin-bottom:18px;'>"
        f"<div style='text-align:center; color:#8ab4cc; font-size:0.85rem; margin-bottom:8px; letter-spacing:2px;'>"
        f"GAME {gnum} &nbsp;·&nbsp; {date_str} &nbsp;·&nbsp; {away} @ {home}</div>"
        f"<div style='display:flex; justify-content:space-around; align-items:center; margin:12px 0;'>"
        # Team A
        f"<div style='text-align:center;'>"
        f"<div style='font-size:2.6rem; font-weight:900; letter-spacing:-1px;'>{team_a}</div>"
        f"<div style='font-size:1.8rem; font-weight:800; color:{'#2ecc71' if prob_a >= 0.5 else '#e74c3c'}; margin-top:4px;'>{_pct(prob_a, 0)}</div>"
        f"<div style='font-size:0.8rem; color:#888; margin-top:2px;'>{'HOME' if home == team_a else 'AWAY'}</div>"
        f"</div>"
        # Score
        f"<div style='text-align:center;'>"
        f"<div style='font-size:2.2rem; font-weight:900; color:#aaa;'>{score_a}–{score_b}</div>"
        f"<div style='font-size:0.75rem; color:#666; margin-top:4px;'>Projected</div>"
        f"<div style='font-size:0.75rem; color:#888; margin-top:8px;'>"
        f"{'Series: ' + team_a + ' leads ' + str(sa) + '–' + str(sb) if sa > sb else 'Series: ' + team_b + ' leads ' + str(sb) + '–' + str(sa) if sb > sa else 'Series tied ' + str(sa) + '–' + str(sb)}"
        f"</div>"
        f"</div>"
        # Team B
        f"<div style='text-align:center;'>"
        f"<div style='font-size:2.6rem; font-weight:900; letter-spacing:-1px;'>{team_b}</div>"
        f"<div style='font-size:1.8rem; font-weight:800; color:{'#2ecc71' if prob_b >= 0.5 else '#e74c3c'}; margin-top:4px;'>{_pct(prob_b, 0)}</div>"
        f"<div style='font-size:0.8rem; color:#888; margin-top:2px;'>{'HOME' if home == team_b else 'AWAY'}</div>"
        f"</div>"
        f"</div>"
        f"<div style='text-align:center; color:#8ab4cc; font-size:0.8rem; margin-top:8px;'>"
        f"Model: <strong>{fav_game}</strong> favored at {_pct(fav_prob_game, 0)} · {home_label} court advantage"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # --- Win probability ---
    st.markdown("### Win Probability")
    c1, c2, c3 = st.columns(3)
    fav = team_a if prob_a >= prob_b else team_b
    fav_pct = max(prob_a, prob_b)
    with c1:
        delta_color = "normal" if prob_a >= 0.5 else "inverse"
        st.metric(
            f"{team_a} (Away)" if away == team_a else f"{team_a} (Home)",
            _pct(prob_a),
            delta=f"{'Favorite' if prob_a >= 0.5 else 'Underdog'}",
            delta_color=delta_color,
        )
    with c2:
        st.metric(
            f"{team_b} (Away)" if away == team_b else f"{team_b} (Home)",
            _pct(prob_b),
            delta=f"{'Favorite' if prob_b >= 0.5 else 'Underdog'}",
            delta_color="normal" if prob_b >= 0.5 else "inverse",
        )
    with c3:
        low = _as_float(prob_range.get("low"), prob_a - 0.08)
        high = _as_float(prob_range.get("high"), prob_a + 0.08)
        st.metric(
            "Confidence range",
            f"{_pct(low, 0)} – {_pct(high, 0)}",
            delta=f"{team_a} range",
            delta_color="off",
        )

    if abs(prob_a - 0.5) < 0.07:
        st.caption(
            "This game is essentially a coin flip. Both teams are nearly even on paper - "
            "execution, adjustments, and single-player performances will decide it."
        )
    elif fav_pct > 0.62:
        st.caption(
            f"{fav} has a meaningful structural advantage for this game. "
            "Home court and team quality both point the same direction."
        )

    st.markdown("---")

    # --- Projected score ---
    st.markdown("### Projected Score")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric(f"{team_a}", str(score_a))
    with sc2:
        st.metric(f"{team_b}", str(score_b))
    with sc3:
        if pace:
            st.metric("Projected pace", f"{pace:.1f}", delta="possessions / 48 min", delta_color="off")
    total = _as_float(score_a, 0) + _as_float(score_b, 0)
    if total > 0:
        pace_label = "slow-paced, defense-first" if total < 205 else "up-tempo" if total > 215 else "balanced"
        st.caption(
            f"Total projected: **{int(total)} points** - a {pace_label} game. "
            f"Game 1 actual was NYK 105, SAS 95 (200 total, slower than average)."
        )

    # --- Spread / margin model ---
    spread_margin = game.get("spread_margin")
    _spread_implied = game.get("spread_implied_probability")
    _spread_gap = abs(prob_a - _spread_implied) if _spread_implied else 0.0

    if spread_margin is not None and _spread_gap <= 0.03:
        fav_dir = team_a if spread_margin >= 0 else team_b
        st.caption(
            f"Spread model: **{fav_dir} by {abs(spread_margin):.1f} pts** · "
            f"Implied {_pct(_spread_implied)} — models agree (gap: {_spread_gap*100:.1f} pts)."
        )
    elif spread_margin is not None:
        st.markdown("---")
        st.markdown("### Projected Spread")
        spread_std = _as_float(game.get("spread_std"), 14.9)
        spread_implied = game.get("spread_implied_probability")

        # Express as a standard market spread: negative = team is favored
        if spread_margin >= 0:
            spread_label = f"{team_a} -{abs(spread_margin):.1f}"
            dog_label = f"{team_b} +{abs(spread_margin):.1f}"
        else:
            spread_label = f"{team_b} -{abs(spread_margin):.1f}"
            dog_label = f"{team_a} +{abs(spread_margin):.1f}"

        ci_low = round(spread_margin - spread_std, 1)
        ci_high = round(spread_margin + spread_std, 1)
        ci_str = (
            f"{team_a} by {ci_low:.1f} to {ci_high:.1f}"
            if ci_low >= 0
            else f"{team_a} by {ci_low:.1f} to {team_b} by {abs(ci_low):.1f}"
            if ci_high >= 0
            else f"{team_b} by {abs(ci_high):.1f} to {abs(ci_low):.1f}"
        )

        spr1, spr2, spr3 = st.columns(3)
        with spr1:
            st.metric("Model line", spread_label, delta=dog_label, delta_color="off")
        with spr2:
            st.metric("68% range", ci_str, delta="±1 std dev", delta_color="off")
        with spr3:
            if spread_implied is not None:
                ensemble_prob = prob_a
                gap = round((ensemble_prob - spread_implied) * 100, 1)
                gap_str = f"{gap:+.1f}% vs spread model"
                st.metric(
                    "Spread-implied prob",
                    _pct(spread_implied),
                    delta=gap_str,
                    delta_color="off",
                )
        st.caption(
            f"Spread model: independently trained XGBoost regression on historical playoff margins "
            f"(MAE ≈ 11.8 pts, residual std ≈ {spread_std:.1f} pts). "
            f"Win-prob model says **{_pct(prob_a)} {team_a}**; spread model implies **{_pct(spread_implied) if spread_implied else '?'} {team_a}**. "
            "Agreement between them strengthens the call; divergence flags uncertainty."
        )

    st.markdown("---")

    # --- How it was predicted: component breakdown ---
    st.markdown("### How This Prediction Was Built")
    st.caption(
        "Each row below is one ingredient. The final win probability combines all of "
        "them using weights learned from 10 seasons of playoff data."
    )

    # Baseline row
    comp_rows = [
        {
            "Component": "ML Baseline (structural)",
            "Raw value": f"{baseline:.1%} NYK",
            "Direction": f"{'NYK' if baseline >= 0.5 else 'SAS'} +{abs(baseline - 0.5) * 100:.1f}%",
            "Weight": "Anchor",
            "What it captures": (
                "Logistic regression trained on 832 playoff games. Uses net rating, "
                "shooting efficiency, turnovers, pace, rest, home court."
            ),
        }
    ]

    component_names = {
        "player_projection": ("Player projection margin", "25%"),
        "matchup_edge": ("Matchup edge", "20%"),
        "lineup_edge": ("Lineup strength edge", "15%"),
        "clutch_edge": ("Clutch / closing-lineup edge", "10%"),
        "injury_edge": ("Injury adjustment", "5%"),
        "foul_trouble_risk": ("Foul-trouble risk", "8%"),
    }
    for key, (label, weight) in component_names.items():
        val = _as_float(margins.get(key), 0.0)
        direction = f"{'NYK' if val >= 0 else 'SAS'} +{abs(val):.2f}" if abs(val) >= 0.15 else "Neutral"
        comp_rows.append({
            "Component": label,
            "Raw value": f"{val:+.3f}",
            "Direction": direction,
            "Weight": weight,
            "What it captures": _component_explanation(key, val, team_a, team_b),
        })

    if post_shift is not None:
        comp_rows.append({
            "Component": "Post-game learning shift",
            "Raw value": f"{post_shift:+.4f}",
            "Direction": f"{'NYK' if post_shift >= 0 else 'SAS'} adjusted from G1 result",
            "Weight": "Applied",
            "What it captures": (
                "After Game 1, the model re-weights future games based on what actually "
                "happened - rotation changes, efficiency gaps, and result surprise."
            ),
        })

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Key matchups for this game ---
    st.markdown("### Key Matchups to Watch")
    from src.data.load_manual_data import load_player_matchups
    from src.data.load_manual_data import load_coaching_notes
    matchups_df = load_player_matchups()
    coaching_df = load_coaching_notes()

    m_cols = st.columns(2)
    matchups = matchups_df.to_dict(orient="records")
    highlighted = [m for m in matchups if abs(_as_float(m.get("expected_impact"))) >= 0.5][:6]
    for i, m in enumerate(highlighted):
        with m_cols[i % 2]:
            impact = _as_float(m.get("expected_impact"))
            off_team = str(m.get("offensive_team", ""))
            off_player = str(m.get("offensive_player", ""))
            def_player = str(m.get("primary_defender", ""))
            mtype = str(m.get("matchup_type", "")).replace("_", " ").title()
            color = "#2ecc71" if impact > 0 else "#e74c3c"
            label = "Advantage offense" if impact > 0 else "Advantage defense"
            with st.container(border=True):
                st.markdown(
                    f"**{off_player}** ({off_team}) vs **{def_player}**  \n"
                    f"*{mtype}* - "
                    f"<span style='color:{color};'>{label} ({impact:+.1f})</span>",
                    unsafe_allow_html=True,
                )
                notes = str(m.get("notes", ""))
                if notes:
                    # Show only the first sentence before " - G1"
                    short = notes.split(" - G1")[0].split("-")[0].strip()
                    st.caption(short)

    # Coaching adjustments for this game
    game_notes = coaching_df[coaching_df["game_number"] == gnum].to_dict(orient="records") if not coaching_df.empty else []
    if game_notes:
        st.markdown("#### Coaching adjustments expected for this game")
        for note in game_notes:
            team = str(note.get("team", ""))
            desc = str(note.get("description", ""))
            impact = str(note.get("expected_impact", ""))
            icon = "↑" if "+" in impact else "↓" if "-" in impact else "→"
            st.caption(f"**{team}** {icon} {desc}")

    st.markdown("---")

    # --- Rotation expectations ---
    st.markdown("### Rotation Expectations")
    st.caption("Projected minutes for this game based on Game 1 actuals and current rotation confidence.")

    rot_cols = st.columns(2)
    for col, team in zip(rot_cols, [team_a, team_b]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{team}**")
                team_players = [
                    p for p in bundle["player_projections"].get(team, [])
                    if _as_float(p.get("minutes")) >= 6
                ]
                for p in team_players[:8]:
                    mins = _as_float(p.get("minutes"))
                    conf = str(p.get("rotation_confidence", "medium")).lower()
                    pts = _as_float(p.get("points"))
                    conf_icon = "✅" if conf == "high" else "⚠️" if conf == "medium" else "❓"
                    st.caption(
                        f"{conf_icon} **{p.get('player')}** - "
                        f"{mins:.0f} min, {pts:.1f} pts projected"
                    )

    st.markdown("---")

    # --- X-factors ---
    st.markdown("### Biggest X-Factors for This Game")
    st.caption("Derived from actual game data — variables with the most series impact so far.")

    completed = bundle.get("completed_games", [])
    display_xf = _series_xfactors_from_data(bundle) if completed else x_factors
    if display_xf:
        for i, xf in enumerate(display_xf[:4], 1):
            st.markdown(f"**{i}.** {xf}")
    else:
        st.caption("No significant X-factors flagged beyond normal game variance.")

    # Foul scenarios
    foul_sim = bundle.get("context", {})
    major_fouls = [
        s for s in bundle.get("context", {}).get("foul_scenarios", [])
        if abs(_as_float(s.get("win_probability_swing"))) >= 0.04
    ]
    if major_fouls:
        st.markdown("**Foul trouble scenarios to watch:**")
        for s in major_fouls[:3]:
            swing = _as_float(s.get("win_probability_swing"))
            st.caption(
                f"• {s.get('team')} {s.get('player')}: "
                f"foul trouble swings win prob by {swing:+.1%}"
            )

    st.markdown("---")

    # --- Series stakes for this game ---
    st.markdown("### Series Stakes")
    c_a, c_b = st.columns(2)
    sa_curr = series_score.get(team_a, 0)
    sb_curr = series_score.get(team_b, 0)
    with c_a:
        new_sa = sa_curr + 1
        sim_a_wins = series.get("team_a_series_win_probability", 0.5)
        st.metric(
            f"If {team_a} wins Game {gnum}",
            f"Series {new_sa}–{sb_curr}",
            delta=f"Leads {team_a}",
            delta_color="normal",
        )
        st.caption(
            f"Series win probability after this result would increase from "
            f"{_pct(sim_a_wins)} - the team ahead after Game {gnum} "
            f"has historically won the Finals ~75% of the time."
        )
    with c_b:
        new_sb = sb_curr + 1
        sim_b_wins = series.get("team_b_series_win_probability", 0.5)
        st.metric(
            f"If {team_b} wins Game {gnum}",
            f"Series {sa_curr}–{new_sb}",
            delta=f"{'Ties' if sa_curr == sb_curr + 1 else 'Leads'} {team_b}",
            delta_color="normal",
        )
        st.caption(
            f"Series win probability shifts to ~{_pct(sim_b_wins)} for {team_b}. "
            "This game has enormous series-swing value."
        )


# ---------------------------------------------------------------------------
# Game Findings tab
# ---------------------------------------------------------------------------

def _parse_minutes(minutes_str: Any) -> float:
    """Convert 'MM:SS' or numeric to float minutes."""
    try:
        s = str(minutes_str)
        if ":" in s:
            mm, ss = s.split(":", 1)
            return int(mm) + int(ss) / 60
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _series_xfactors_from_data(bundle: dict[str, Any]) -> list[str]:
    """Derive X-factors from actual game box scores — replaces pre-series structural flags."""
    completed = bundle.get("completed_games", [])
    if not completed:
        return []

    games_dir = PROJECT_ROOT / "data" / "processed" / "finals_games"
    player_series: dict[str, dict[str, Any]] = {}

    for gn in sorted(completed):
        path = games_dir / f"game_{gn}_player_traditional.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["player"] = df["firstName"].astype(str) + " " + df["familyName"].astype(str)
        df["min_float"] = df["minutes"].apply(_parse_minutes)

        for _, r in df.iterrows():
            p = r["player"]
            if p not in player_series:
                player_series[p] = {
                    "team": str(r["teamTricode"]),
                    "games": 0,
                    "total_min": 0.0,
                    "total_pts": 0,
                    "total_to": 0,
                    "total_fouls": 0,
                    "total_fgm": 0,
                    "total_fga": 0,
                    "pts_by_game": [],
                    "min_by_game": [],
                }
            d = player_series[p]
            d["games"] += 1
            d["total_min"] += r["min_float"]
            d["total_pts"] += int(r.get("points", 0))
            d["total_to"] += int(r.get("turnovers", 0))
            d["total_fouls"] += int(r.get("foulsPersonal", 0))
            d["total_fgm"] += int(r.get("fieldGoalsMade", 0))
            d["total_fga"] += int(r.get("fieldGoalsAttempted", 0))
            d["pts_by_game"].append(int(r.get("points", 0)))
            d["min_by_game"].append(r["min_float"])

    candidates: list[tuple[float, str, str]] = []

    for player, d in player_series.items():
        g = d["games"]
        avg_min = d["total_min"] / g
        avg_pts = d["total_pts"] / g
        avg_to = d["total_to"] / g
        avg_fouls = d["total_fouls"] / g
        fg_pct = d["total_fgm"] / max(d["total_fga"], 1)
        team = d["team"]

        # High-minute star with dangerous turnover rate
        if avg_min >= 28 and avg_to >= 3.0:
            score = avg_to * avg_min / 10
            candidates.append((score, player, f"{player} ({team}) — {avg_to:.1f} TOs/game on {avg_min:.0f} min, gifting possessions at a critical rate"))

        # Foul trouble depleting a key rotation piece
        if avg_min >= 15 and avg_fouls >= 3.8:
            score = avg_fouls * avg_min / 12
            candidates.append((score, player, f"{player} ({team}) — {d['total_fouls']} fouls in {g} game(s), averaging {avg_fouls:.1f}/game and running out of margin"))

        # Star shooting poorly on heavy volume
        if avg_min >= 28 and d["total_fga"] >= 18 and fg_pct < 0.40:
            score = d["total_fga"] * max(0.42 - fg_pct, 0)
            candidates.append((score, player, f"{player} ({team}) — {fg_pct:.1%} FG on {d['total_fga']/g:.0f} attempts/game; efficiency is the ceiling on this team"))

        # Role player expected to contribute but going quiet
        if avg_min >= 22 and avg_pts < 7 and d["total_fga"] >= 5:
            score = (8 - avg_pts) * avg_min / 18
            candidates.append((score, player, f"{player} ({team}) — averaging {avg_pts:.0f} pts in {avg_min:.0f} min, {team} needs more from this role"))

        # Emerging co-star (big scoring jump)
        if g >= 2 and (d["pts_by_game"][-1] - d["pts_by_game"][0]) >= 9 and d["min_by_game"][-1] >= 22:
            jump = d["pts_by_game"][-1] - d["pts_by_game"][0]
            score = jump * 0.85
            candidates.append((score, player, f"{player} ({team}) — stepped up from {d['pts_by_game'][0]} pts G1 to {d['pts_by_game'][-1]} pts G{g}, becoming a genuine third option"))

    candidates.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    results: list[str] = []
    for _, player, text in candidates:
        if player not in seen:
            seen.add(player)
            results.append(text)
        if len(results) >= 5:
            break
    return results


def _strategy_audit(
    pred_source: dict[str, Any],
    box: dict[str, Any],
    team_a: str,
    team_b: str,
) -> list[tuple[bool, str, str]]:
    """Compare pre-game predicted edges to what actually happened.

    Returns list of (correct, label, detail) tuples.
    """
    items: list[tuple[bool, str, str]] = []
    team_stats = box.get("team_stats")
    player_df = box.get("player_stats")
    adv_df = box.get("team_advanced")

    if team_stats is None or team_stats.empty:
        return items

    actual: dict[str, Any] = {}
    for _, row in team_stats.iterrows():
        actual[str(row["teamTricode"])] = row
    aa = actual.get(team_a, {})
    ab = actual.get(team_b, {})

    pred_a = int(pred_source.get("expected_score_team_a") or 0)
    pred_b = int(pred_source.get("expected_score_team_b") or 0)
    act_a = int(aa.get("points", 0))
    act_b = int(ab.get("points", 0))
    pred_margin = abs(pred_a - pred_b)
    act_margin = abs(act_a - act_b)

    # 1. Game closeness
    if pred_margin <= 8 and act_margin <= 8:
        items.append((True, "Close-game call", f"Predicted {pred_margin}-pt margin — actual was {act_margin} pts"))
    elif pred_margin <= 8 and act_margin >= 15:
        items.append((False, "Expected closer", f"Predicted {pred_margin}-pt game — actual blowout by {act_margin}"))
    elif pred_margin >= 12 and act_margin >= 12:
        items.append((True, "Dominant-win call", f"Predicted {pred_margin}-pt margin — actual was {act_margin}"))

    # 2. Score line accuracy (each team's projected total)
    if pred_a > 0 and abs(act_a - pred_a) <= 3:
        items.append((True, f"{team_a} scoring", f"Projected {pred_a}, actual {act_a} (off by {abs(act_a - pred_a)})"))
    if pred_b > 0 and abs(act_b - pred_b) <= 3:
        items.append((True, f"{team_b} scoring", f"Projected {pred_b}, actual {act_b} (off by {abs(act_b - pred_b)})"))

    # 3. Pace accuracy — only check if adv_df has a valid in-range pace value
    pred_pace = pred_source.get("projected_pace")
    if pred_pace and adv_df is not None and not adv_df.empty:
        pace_col = next((c for c in adv_df.columns if "pace" in c.lower()), None)
        if pace_col:
            raw_pace = float(adv_df[pace_col].mean())
            if 85 <= raw_pace <= 115:  # sanity-check the value is real NBA pace
                pace_err = abs(float(pred_pace) - raw_pace)
                if pace_err <= 3:
                    items.append((True, "Pace prediction", f"Projected {float(pred_pace):.0f} possessions — actual {raw_pace:.0f}"))
                else:
                    items.append((False, "Pace off", f"Projected {float(pred_pace):.0f} — actual {raw_pace:.0f} ({pace_err:.0f} diff)"))

    # 4. Edge materialization — parse top_edges text
    oreb_a = int(aa.get("reboundsOffensive", 0))
    oreb_b = int(ab.get("reboundsOffensive", 0))
    tov_a = int(aa.get("turnovers", 0))
    tov_b = int(ab.get("turnovers", 0))
    fta_a = int(aa.get("freeThrowsAttempted", 0))
    fta_b = int(ab.get("freeThrowsAttempted", 0))

    _checked_edge_types: set[str] = set()
    for edge in (pred_source.get("top_edges") or [])[:4]:
        e = edge.lower()
        ta_pos = e.find(team_a.lower())
        tb_pos = e.find(team_b.lower())
        # Which team is named first? That team has the edge (advantage) or the problem (disadvantage).
        first_team = team_a if (ta_pos >= 0 and (tb_pos < 0 or ta_pos < tb_pos)) else team_b
        other_team = team_b if first_team == team_a else team_a
        has_adv = "advantage" in e
        has_dis = "disadvantage" in e

        if "offensive rebound" in e and "oreb" not in _checked_edge_types:
            _checked_edge_types.add("oreb")
            a_wins = oreb_a > oreb_b
            pred_a_wins = (first_team == team_a and has_adv) or (first_team == team_b and has_dis)
            correct = (pred_a_wins and a_wins) or (not pred_a_wins and not a_wins)
            winner = team_a if a_wins else team_b
            items.append((correct, "Offensive rebounding", f"{team_a} {oreb_a} vs {team_b} {oreb_b} — {winner} won the glass"))

        elif ("ball security" in e or ("turnover" in e and "forced" in e)) and "tov" not in _checked_edge_types:
            _checked_edge_types.add("tov")
            pred_team_has_risk = first_team if has_dis else other_team
            actual_more_tovs = team_a if tov_a > tov_b else team_b
            correct = pred_team_has_risk == actual_more_tovs
            items.append((correct, "Turnover risk", f"{team_a} {tov_a} TOV vs {team_b} {tov_b} TOV — {'matched prediction' if correct else 'opposite of prediction'}"))

        elif "free throw pressure" in e and "fta" not in _checked_edge_types:
            _checked_edge_types.add("fta")
            pred_fta_winner = first_team if has_adv else other_team
            actual_fta_winner = team_a if fta_a > fta_b else team_b
            correct = pred_fta_winner == actual_fta_winner
            items.append((correct, "Free throw pressure", f"{team_a} {fta_a} FTA vs {team_b} {fta_b} FTA — {actual_fta_winner} got to the line more"))

    # 5. Foul trouble X-factors
    if player_df is not None and not player_df.empty:
        for xf in (pred_source.get("x_factors") or [])[:4]:
            if "foul trouble" not in xf.lower():
                continue
            for _, prow in player_df.iterrows():
                player_name = str(prow.get("player", ""))
                last = player_name.split()[-1].lower() if player_name else ""
                if len(last) > 3 and last in xf.lower():
                    pf = int(prow.get("foulsPersonal", 0))
                    if pf >= 4:
                        items.append((True, f"{player_name} foul trouble", f"Flagged pre-game — picked up {pf} fouls"))
                    break

    return items


def _load_pregame_snapshot(game_number: int) -> dict[str, Any] | None:
    """Return the pre-game prediction dict for game_number from the newest matching snapshot."""
    import json
    snaps_dir = PROJECT_ROOT / "outputs" / "predictions" / "snapshots"
    if not snaps_dir.exists():
        return None
    candidates = sorted(snaps_dir.glob(f"*_game_{game_number}.json"), reverse=True)
    for snap_path in candidates:
        try:
            data = json.loads(snap_path.read_text(encoding="utf-8"))
            for pred in data.get("predictions", []):
                if isinstance(pred, dict) and int(pred.get("game_number", -1)) == game_number:
                    return pred
        except Exception:
            pass
    return None


def _load_game_box_score(game_number: int) -> dict[str, Any]:
    """Load traditional + advanced box scores for a completed game from CSVs."""
    games_dir = PROJECT_ROOT / "data" / "processed" / "finals_games"
    result: dict[str, Any] = {"game_number": game_number}

    trad_path = games_dir / f"game_{game_number}_team_traditional.csv"
    player_path = games_dir / f"game_{game_number}_player_traditional.csv"
    adv_path = games_dir / f"game_{game_number}_team_actuals.csv"

    if trad_path.exists():
        trad_df = pd.read_csv(trad_path)
        stat_cols = [
            "fieldGoalsMade", "fieldGoalsAttempted",
            "threePointersMade", "threePointersAttempted",
            "freeThrowsMade", "freeThrowsAttempted",
            "reboundsOffensive", "reboundsDefensive", "reboundsTotal",
            "assists", "steals", "blocks", "turnovers", "foulsPersonal", "points",
        ]
        existing = [c for c in stat_cols if c in trad_df.columns]
        agg = trad_df.groupby("teamTricode")[existing].sum().reset_index()
        result["team_stats"] = agg

    if player_path.exists():
        pf = pd.read_csv(player_path)
        pf["player"] = pf["firstName"].astype(str) + " " + pf["familyName"].astype(str)
        pf["min_float"] = pf["minutes"].apply(_parse_minutes)
        result["player_stats"] = pf

    if adv_path.exists():
        result["team_advanced"] = pd.read_csv(adv_path)

    return result


def render_game_findings(bundle: dict[str, Any]) -> None:
    """Per-game analysis: model vs actual, key stats, player trends, adjustment flags."""
    completed = bundle.get("completed_games", [])

    if not completed:
        st.info("Game-by-game findings will appear here after each game is played.")
        return

    team_a = bundle["team_a"]
    team_b = bundle["team_b"]
    series_score = bundle.get("series_score") or {}
    pred_by_game = {int(g["game_number"]): g for g in bundle.get("game_predictions", [])}

    # Series state header
    sa = series_score.get(team_a, 0)
    sb = series_score.get(team_b, 0)
    if sa > sb:
        st.markdown(f"## {team_a} leads {sa}–{sb}")
    elif sb > sa:
        st.markdown(f"## {team_b} leads {sb}–{sa}")
    else:
        st.markdown(f"## Series tied {sa}–{sb}")

    series_sim = bundle.get("series_simulation") or {}
    prob_a = series_sim.get("team_a_series_win_probability", 0.5)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(f"{team_a} to win series", _pct(prob_a))
    with col_b:
        st.metric(f"{team_b} to win series", _pct(1 - prob_a))
    st.divider()

    all_player_stats: dict[int, pd.DataFrame] = {}

    for game_num in sorted(completed):
        box = _load_game_box_score(game_num)
        pred = pred_by_game.get(game_num, {})
        team_stats_df: pd.DataFrame | None = box.get("team_stats")
        player_df: pd.DataFrame | None = box.get("player_stats")
        adv_df: pd.DataFrame | None = box.get("team_advanced")

        if team_stats_df is None or team_stats_df.empty:
            continue

        scores = {str(r["teamTricode"]): int(r["points"]) for _, r in team_stats_df.iterrows()}
        actual_winner = max(scores, key=scores.get)
        score_a = scores.get(team_a, 0)
        score_b = scores.get(team_b, 0)

        # Use pre-game snapshot if available — current model predictions would be
        # retroactively recalculated and don't reflect what was actually predicted.
        snapshot = _load_pregame_snapshot(game_num)
        has_snapshot = snapshot is not None
        pred_source = snapshot if has_snapshot else pred

        pred_prob_a = float(pred_source.get("team_a_win_probability", 0.5))
        pred_score_a = int(pred_source.get("expected_score_team_a", 0))
        pred_score_b = int(pred_source.get("expected_score_team_b", 0))
        predicted_winner = team_a if pred_prob_a >= 0.5 else team_b
        model_correct = predicted_winner == actual_winner

        if has_snapshot:
            accuracy_label = "✅ Model correct" if model_correct else "❌ Model missed"
        else:
            accuracy_label = "📋 No pre-game snapshot"

        expander_label = (
            f"Game {game_num}  —  {accuracy_label}  —  "
            f"{team_a} {score_a}, {team_b} {score_b}"
        )
        with st.expander(expander_label, expanded=True):

            # Model vs actual summary
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Pre-game model prediction**" if has_snapshot else "**Pre-game prediction**")
                if has_snapshot:
                    fav = team_a if pred_prob_a >= 0.5 else team_b
                    fav_conf = _pct(max(pred_prob_a, 1 - pred_prob_a))
                    st.markdown(f"Winner: **{fav}** ({fav_conf})")
                    st.markdown(f"Score: {pred_score_a} – {pred_score_b}")
                    pred_pace = pred_source.get("projected_pace")
                    if pred_pace:
                        st.caption(f"Projected pace: {float(pred_pace):.1f}")
                else:
                    st.caption("No pre-game snapshot was saved for this game.")
            with c2:
                margin = abs(score_a - score_b)
                st.markdown("**Actual result**")
                st.markdown(f"Winner: **{actual_winner}** by {margin}")
                st.markdown(f"Score: {score_a} – {score_b}")
                if adv_df is not None and not adv_df.empty and "pace" in adv_df.columns:
                    actual_pace = float(adv_df.iloc[0]["pace"])
                    st.caption(f"Actual pace: {actual_pace:.1f}")
            with c3:
                st.markdown("**Accuracy**")
                if has_snapshot:
                    if model_correct:
                        st.success("✅ Correct pick")
                    else:
                        st.error("❌ Wrong pick")
                    err_a = score_a - pred_score_a
                    err_b = score_b - pred_score_b
                    st.caption(
                        f"{team_a} score error: {'+' if err_a >= 0 else ''}{err_a}  \n"
                        f"{team_b} score error: {'+' if err_b >= 0 else ''}{err_b}"
                    )
                else:
                    st.info("Snapshot needed for accuracy tracking. Future games are saved automatically.")

            # Strategy audit — what the model identified correctly
            audit = _strategy_audit(pred_source, box, team_a, team_b)
            if audit:
                st.markdown("---")
                st.markdown("**What the model got right (and wrong) beyond the pick**")
                col_hit, col_miss = st.columns(2)
                hits = [(l, d) for ok, l, d in audit if ok]
                misses = [(l, d) for ok, l, d in audit if not ok]
                with col_hit:
                    if hits:
                        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#2ecc71; margin-bottom:6px;'>IDENTIFIED CORRECTLY</div>", unsafe_allow_html=True)
                    for label, detail in hits:
                        st.markdown(
                            f"<div style='display:flex; gap:8px; margin-bottom:6px; align-items:flex-start;'>"
                            f"<span style='color:#2ecc71; font-weight:900; font-size:1rem;'>✓</span>"
                            f"<div><strong style='color:#ddd;'>{label}</strong><br>"
                            f"<span style='color:#888; font-size:0.8rem;'>{detail}</span></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                with col_miss:
                    if misses:
                        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#e74c3c; margin-bottom:6px;'>MISSED / WRONG</div>", unsafe_allow_html=True)
                    for label, detail in misses:
                        st.markdown(
                            f"<div style='display:flex; gap:8px; margin-bottom:6px; align-items:flex-start;'>"
                            f"<span style='color:#e74c3c; font-weight:900; font-size:1rem;'>✗</span>"
                            f"<div><strong style='color:#ddd;'>{label}</strong><br>"
                            f"<span style='color:#888; font-size:0.8rem;'>{detail}</span></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown("---")

            # Team box score
            st.markdown("**Team box score**")
            team_rows = []
            for _, row in team_stats_df.iterrows():
                fgm = int(row["fieldGoalsMade"])
                fga = int(row["fieldGoalsAttempted"])
                fg3m = int(row["threePointersMade"])
                fg3a = int(row["threePointersAttempted"])
                ftm = int(row["freeThrowsMade"])
                fta = int(row["freeThrowsAttempted"])
                efg = (fgm + 0.5 * fg3m) / max(fga, 1)
                team_rows.append({
                    "Team": str(row["teamTricode"]),
                    "PTS": int(row["points"]),
                    "FG": f"{fgm}/{fga}",
                    "eFG%": f"{efg:.1%}",
                    "3P": f"{fg3m}/{fg3a}",
                    "FT": f"{ftm}/{fta}",
                    "REB": int(row["reboundsTotal"]),
                    "AST": int(row["assists"]),
                    "TOV": int(row["turnovers"]),
                })
            st.dataframe(pd.DataFrame(team_rows), hide_index=True, use_container_width=True)

            if adv_df is not None and not adv_df.empty:
                adv_rows = []
                for _, row in adv_df.iterrows():
                    adv_rows.append({
                        "Team": str(row.get("teamTricode", "")),
                        "OffRtg": f"{row['offensiveRating']:.1f}",
                        "DefRtg": f"{row['defensiveRating']:.1f}",
                        "NetRtg": f"{row['netRating']:+.1f}",
                        "eFG%": f"{row['effectiveFieldGoalPercentage']:.1%}",
                        "TS%": f"{row['trueShootingPercentage']:.1%}",
                    })
                st.dataframe(pd.DataFrame(adv_rows), hide_index=True, use_container_width=True)

            st.markdown("---")

            # Player performances
            if player_df is not None and not player_df.empty:
                st.markdown("**Player performances**")
                top_p = player_df[player_df["min_float"] >= 12].sort_values(
                    ["teamTricode", "min_float"], ascending=[True, False]
                )
                perf_rows = []
                for _, r in top_p.iterrows():
                    fgm_p = int(r.get("fieldGoalsMade", 0))
                    fga_p = int(r.get("fieldGoalsAttempted", 0))
                    fg3m_p = int(r.get("threePointersMade", 0))
                    fg3a_p = int(r.get("threePointersAttempted", 0))
                    perf_rows.append({
                        "Team": str(r["teamTricode"]),
                        "Player": r["player"],
                        "MIN": f"{r['min_float']:.0f}",
                        "PTS": int(r.get("points", 0)),
                        "FG": f"{fgm_p}/{fga_p}",
                        "3P": f"{fg3m_p}/{fg3a_p}",
                        "REB": int(r.get("reboundsTotal", 0)),
                        "AST": int(r.get("assists", 0)),
                        "TOV": int(r.get("turnovers", 0)),
                        "PF": int(r.get("foulsPersonal", 0)),
                    })
                st.dataframe(pd.DataFrame(perf_rows), hide_index=True, use_container_width=True)

                foul_alerts = [
                    f"{r['player']} ({r['teamTricode']}) — {int(r['foulsPersonal'])} PF in {r['min_float']:.0f} min"
                    for _, r in player_df.iterrows()
                    if r.get("foulsPersonal", 0) >= 4 and r["min_float"] >= 10
                ]
                if foul_alerts:
                    st.warning("**Foul trouble:**  " + "  ·  ".join(foul_alerts))

                all_player_stats[game_num] = player_df

    # Cross-game player trends (only appears after 2+ games)
    if len(completed) >= 2 and len(all_player_stats) >= 2:
        st.markdown("## Player Trends")
        st.caption("How key players have tracked across completed games (↑ up 5+ pts, ↓ down 5+ pts)")
        game_nums = sorted(all_player_stats.keys())
        trend_rows: list[dict[str, Any]] = []
        seen_players: set[str] = set()

        for team in [team_a, team_b]:
            team_frames = {
                gn: all_player_stats[gn][all_player_stats[gn]["teamTricode"] == team]
                for gn in game_nums
            }
            all_players_team: set[str] = set()
            for df in team_frames.values():
                all_players_team |= set(df["player"].tolist())

            for player in sorted(all_players_team):
                if player in seen_players:
                    continue
                game_data = {
                    gn: (team_frames[gn][team_frames[gn]["player"] == player].iloc[0]
                         if not team_frames[gn][team_frames[gn]["player"] == player].empty
                         else None)
                    for gn in game_nums
                }
                max_min = max(
                    (row["min_float"] for row in game_data.values() if row is not None),
                    default=0,
                )
                if max_min < 10:
                    continue
                seen_players.add(player)
                row_dict: dict[str, Any] = {"Team": team, "Player": player}
                pts_list: list[int] = []
                for gn in game_nums:
                    r = game_data[gn]
                    if r is not None:
                        pts = int(r.get("points", 0))
                        row_dict[f"G{gn} MIN"] = f"{r['min_float']:.0f}"
                        row_dict[f"G{gn} PTS"] = pts
                        row_dict[f"G{gn} PF"] = int(r.get("foulsPersonal", 0))
                        pts_list.append(pts)
                    else:
                        row_dict[f"G{gn} MIN"] = "—"
                        row_dict[f"G{gn} PTS"] = "—"
                        row_dict[f"G{gn} PF"] = "—"
                if len(pts_list) >= 2:
                    diff = pts_list[-1] - pts_list[-2]
                    row_dict["Trend"] = "↑" if diff >= 5 else ("↓" if diff <= -5 else "→")
                else:
                    row_dict["Trend"] = "—"
                trend_rows.append(row_dict)

        if trend_rows:
            st.dataframe(pd.DataFrame(trend_rows), hide_index=True, use_container_width=True)

    # Series X-factors from actual game data
    if all_player_stats:
        st.markdown("## Series X-Factors")
        st.caption("Derived from actual box score data across all completed games.")
        xfactors = _series_xfactors_from_data(bundle)
        if xfactors:
            for xf in xfactors:
                st.markdown(f"- {xf}")
        else:
            st.info("X-factors will populate as game data is imported.")
        st.divider()

    # Adjustment flags heading into next game
    if completed:
        latest_game = max(completed)
        latest_box = _load_game_box_score(latest_game)
        lp_df = latest_box.get("player_stats")
        lt_df = latest_box.get("team_stats")

        st.markdown(f"## Entering Game {latest_game + 1}: Key Adjustments")
        col1, col2 = st.columns(2)
        for col, team in zip([col1, col2], [team_a, team_b]):
            with col:
                st.markdown(f"### {team}")
                flags: list[str] = []
                if lt_df is not None and not lt_df.empty:
                    ts_row = lt_df[lt_df["teamTricode"] == team]
                    if not ts_row.empty:
                        t = ts_row.iloc[0]
                        fgm = int(t["fieldGoalsMade"])
                        fga = int(t["fieldGoalsAttempted"])
                        fg3m = int(t["threePointersMade"])
                        fg3a = int(t["threePointersAttempted"])
                        tov = int(t["turnovers"])
                        efg = (fgm + 0.5 * fg3m) / max(fga, 1)
                        fg3_pct = fg3m / max(fg3a, 1)
                        if efg < 0.47:
                            flags.append(
                                f"🎯 Shooting efficiency — eFG% {efg:.1%} last game, "
                                "needs better shot quality"
                            )
                        if tov >= 13:
                            flags.append(
                                f"🔄 Ball security — {tov} turnovers last game, "
                                "must be more careful with possessions"
                            )
                        elif tov >= 10:
                            flags.append(f"⚠️ {tov} turnovers last game — watch closely")
                        if fg3a >= 20 and fg3_pct < 0.32:
                            flags.append(
                                f"📉 3-point shot selection — {fg3m}/{fg3a} ({fg3_pct:.1%}), "
                                "consider better looks"
                            )
                if lp_df is not None and not lp_df.empty:
                    team_p = lp_df[lp_df["teamTricode"] == team]
                    foul_risk = team_p[
                        (team_p["foulsPersonal"] >= 4) & (team_p["min_float"] >= 10)
                    ]
                    for _, fp in foul_risk.iterrows():
                        flags.append(
                            f"🚨 {fp['player']} — {int(fp['foulsPersonal'])} fouls last game, "
                            "foul discipline is critical"
                        )
                    heavy_min = team_p[team_p["min_float"] >= 38]
                    if not heavy_min.empty:
                        names = ", ".join(heavy_min["player"].tolist())
                        avg_m = heavy_min["min_float"].mean()
                        flags.append(
                            f"⏱️ Load management: {names} averaged {avg_m:.0f}+ min — "
                            "rest-and-recovery matters"
                        )
                if not flags:
                    flags.append("✅ Clean performance — maintain execution")
                for flag in flags:
                    st.markdown(f"- {flag}")


# ---------------------------------------------------------------------------
# Helpers used across tabs
# ---------------------------------------------------------------------------

def _favorite_and_opponent(bundle: dict[str, Any]) -> tuple[str, str]:
    series = bundle["series_simulation"]
    team_a, team_b = bundle["team_a"], bundle["team_b"]
    return (team_a, team_b) if series["team_a_series_win_probability"] >= 0.5 else (team_b, team_a)


def _favorite_reasons(bundle: dict[str, Any], favorite: str) -> list[str]:
    reasons = []
    for edge in bundle["matchup_edges"].get("top_series_edges", []):
        if edge.get("offensive_team") == favorite and _as_float(edge.get("matchup_score"), 0.0) > 0:
            reasons.append(edge.get("name", "").replace(" vs ", " against "))
    if bundle["clutch_prediction"].get("favorite") == favorite:
        reasons.append("clutch/late-game situations")
    summary = bundle["lineup_summary"].get(favorite, {})
    if summary.get("closing_adjusted_net_rating") is not None:
        reasons.append("closing lineup strength")
    return (reasons[:3] if reasons
            else ["overall team strength", "rotation stability", "scoring consistency"])


def _opponent_flip_paths(bundle: dict[str, Any], opponent: str) -> list[str]:
    paths = [
        edge.get("name", "").replace(" vs ", " against ")
        for edge in bundle["matchup_edges"].get("top_series_edges", [])
        if edge.get("offensive_team") == opponent and _as_float(edge.get("matchup_score"), 0.0) > 0
    ]
    paths.extend(["protect the rim without fouling", "win the non-star minutes"])
    return paths[:3]


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_dashboard() -> None:
    st.set_page_config(
        page_title="NBA Finals Predictor 2026",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🏀",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.45rem;}
        h1, h2, h3 {letter-spacing: 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    settings = load_settings()
    configured_simulations = int(settings.get("finals", {}).get("simulations", DEFAULT_SIMULATIONS))

    # Sidebar
    st.sidebar.title("🏀 NBA Finals 2026")
    st.sidebar.caption("Predictions powered by ML + real NBA data")
    st.sidebar.divider()

    series_simulations = configured_simulations
    scenario_simulations = min(DEFAULT_SCENARIO_SIMULATIONS, configured_simulations)
    selected_scenarios: tuple[str, ...] = ()  # scenarios now live in Deep Stats tab

    if st.sidebar.button("Refresh data", type="primary", use_container_width=True,
                         help="Reload predictions after a game is played"):
        load_dashboard_bundle.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("What-if scenarios and deep analysis are available in the **Deep Stats** tab.")

    st.title("🏀 NBA Finals Predictor 2026")
    bundle = load_dashboard_bundle(int(series_simulations), _data_ver=_data_version())

    st.caption(
        f"Last updated: {bundle['last_updated']} · "
        f"Model: {MODEL_VERSION} · "
        f"{len(bundle['completed_games'])} game(s) played"
    )
    provenance = bundle.get("provenance", {})
    dataset_hash = str(provenance.get("canonical_dataset_sha256") or "unavailable")[:12]
    st.caption(
        f"Canonical data: {provenance.get('canonical_schema_version') or 'unavailable'} "
        f"({dataset_hash}) · Manual assumptions are separate from validated probability inputs."
    )

    tab1, tab0, tab2, tab6, tab3, tab4, tab5 = st.tabs([
        "The Pick",
        "Next Game",
        "Game by Game",
        "Game Findings",
        "Player Breakdown",
        "How the Model Works",
        "Deep Stats",
    ])

    with tab1:
        render_the_pick(bundle)
    with tab0:
        render_next_game(bundle)
    with tab2:
        render_game_by_game(bundle)
    with tab6:
        render_game_findings(bundle)
    with tab3:
        render_player_breakdown(bundle)
    with tab4:
        render_ml_panel(bundle)
    with tab5:
        render_deep_stats(bundle, selected_scenarios, int(scenario_simulations))


def main() -> None:
    if st is None:
        print("Streamlit not installed. Run: streamlit run src/app/streamlit_app.py")
        return
    render_dashboard()


if __name__ == "__main__":
    main()
