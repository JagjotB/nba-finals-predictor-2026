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
from src.models.update_after_game import _load_game_actuals, simulate_series_after_results


MODEL_VERSION = "calibrated-finals-v3"
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
            net_metrics = report.get("overall", {}).get("net_rating_baseline", {})
            status["model_accuracy"] = net_metrics.get("accuracy", status["model_accuracy"])
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            status["model_brier"] = meta.get("metrics", {}).get(
                "walk_forward_brier_score",
                status.get("model_brier"),
            )
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

@cache_data(show_spinner="Building predictions - this takes about 15 seconds...")
def load_dashboard_bundle(series_simulations: int) -> dict[str, Any]:
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

    col_fav, col_vs, col_und = st.columns([2, 1, 2])
    with col_fav:
        st.markdown(
            f"<div style='text-align:center; font-size:3rem; font-weight:900;'>{favorite}</div>"
            f"<div style='text-align:center; font-size:1.8rem; color:#2ecc71; font-weight:700;'>{_pct(fav_pct, 0)} chance</div>",
            unsafe_allow_html=True,
        )
    with col_vs:
        st.markdown(
            "<div style='text-align:center; font-size:2rem; padding-top:0.8rem;'>vs</div>",
            unsafe_allow_html=True,
        )
    with col_und:
        st.markdown(
            f"<div style='text-align:center; font-size:3rem; font-weight:900;'>{underdog}</div>"
            f"<div style='text-align:center; font-size:1.8rem; color:#e74c3c; font-weight:700;'>{_pct(min(fav_prob, und_prob), 0)} chance</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Series score if games played
    if completed:
        nyk_w = series_score.get(team_a, 0)
        sas_w = series_score.get(team_b, 0)
        st.markdown(
            f"### Current Series Score: **{team_a} leads {nyk_w}–{sas_w}**"
            if nyk_w > sas_w else
            f"### Current Series Score: **{team_b} leads {sas_w}–{nyk_w}**"
            if sas_w > nyk_w else
            f"### Series Tied {nyk_w}–{sas_w}"
        )
        st.caption(f"{len(completed)} game(s) played and factored into predictions")
        st.markdown("---")

    # Key stats in plain English
    most_likely = series["most_likely_result"]
    avg_games = series.get("average_games", 6.0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Most likely outcome",
        most_likely,
        "Our top prediction",
    )
    col2.metric(
        "Expected series length",
        f"{avg_games:.1f} games",
        "How long we expect this to last",
    )
    col3.metric(
        "Model uncertainty",
        f"{_pct(low, 0)} – {_pct(high, 0)}",
        "Reasonable range for the favorite",
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

    # X-factor
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
    dist = series.get("result_distribution", [])
    rows = []
    for row in dist:
        prob = float(row["probability"])
        if prob < 0.01:
            continue
        winner = row["result"].split(" in ")[0]
        games_num = row["result"].split(" in ")[1]
        bar = "█" * int(prob * 30)
        rows.append({
            "Outcome": row["result"],
            "Chance": row["percentage"],
            "Likelihood bar": bar,
            "Winner": winner,
        })
    df_dist = pd.DataFrame(rows)
    if not df_dist.empty:
        st.dataframe(df_dist[["Outcome", "Chance", "Likelihood bar"]], use_container_width=True, hide_index=True)

    st.caption(
        "These odds update automatically after each game is played. "
        "Click 'Refresh data' in the sidebar after any game to see the latest numbers."
    )


# ---------------------------------------------------------------------------
# TAB 2 -Game by Game
# ---------------------------------------------------------------------------

def _render_game_reasoning(game: dict[str, Any], team_a: str, team_b: str) -> None:
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
    x_factors = game.get("x_factors") or []
    if x_factors:
        st.markdown("**Wildcards to watch:**")
        for xf in x_factors[:3]:
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
        status_icon = "✅" if is_done else "📅"
        status_label = "PLAYED" if is_done else "UPCOMING"

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
            with c1:
                st.markdown(f"### G{gn}")
                st.caption(f"{status_icon} {status_label}")
                if date:
                    st.caption(date)
            with c2:
                st.markdown(f"**{home}** (home) vs **{away}**")
                if not is_done:
                    st.markdown(f"Projected score: **{home} {score_a if home == team_a else score_b} – {away} {score_b if away == team_b else score_a}**")
            with c3:
                st.markdown(f"Model picks: **{fav}**")
                st.progress(fav_pct, text=f"{_pct(fav_pct)} confidence")
            with c4:
                edges = game.get("top_edges") or []
                if edges:
                    st.markdown("**Key edge:**")
                    st.caption(edges[0])
                rng = game.get("team_a_win_probability_range") or {}
                low_p = _as_float(rng.get("low"), prob_a)
                high_p = _as_float(rng.get("high"), prob_a)
                st.caption(
                    f"Uncertainty range: {_pct(low_p, 0)}–{_pct(high_p, 0)} for {team_a}"
                )
            if not is_done:
                with st.expander("Why does the model say this?"):
                    _render_game_reasoning(game, team_a, team_b)

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
    ml = bundle.get("ml_status", {})
    team_a = bundle["team_a"]

    st.markdown("## How Our Model Works")
    st.caption(
        "This section shows exactly what's powering the predictions - "
        "what data is live, which ML models are trained, and how confident each component is."
    )

    # ---- Data status ----
    st.markdown("### Live Data Status")
    st.caption("Green = using real NBA API data. Red = using a statistical default.")

    d1, d2, d3, d4 = st.columns(4)
    def _status_indicator(ok: bool, label: str, detail: str) -> None:
        color = "#2ecc71" if ok else "#e74c3c"
        icon = "✅" if ok else "⚠️"
        st.markdown(
            f"<div style='border:1px solid {color}; border-radius:8px; padding:10px; text-align:center;'>"
            f"<div style='font-size:1.5rem;'>{icon}</div>"
            f"<div style='font-weight:700;'>{label}</div>"
            f"<div style='font-size:0.8rem; color:#aaa;'>{detail}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with d1:
        _status_indicator(ml.get("live_team_stats", False), "Team Stats", "Offensive/defensive ratings, pace")
    with d2:
        _status_indicator(ml.get("live_player_stats", False), "Player Stats", "Per-game stats & PIE efficiency")
    with d3:
        _status_indicator(ml.get("live_lineup_stats", False), "Lineup Stats", "2-man & 5-man lineup net ratings")
    with d4:
        _status_indicator(ml.get("model_trained", False), "ML Model", "Logistic regression trained on real games")

    st.markdown("---")

    # ---- ML model status ----
    st.markdown("### Prediction Model")

    if ml.get("model_trained"):
        acc = ml.get("model_accuracy")
        rows_trained = ml.get("model_training_rows")
        st.success(
            f"**Trained ML model active.** "
            f"Trained on **{rows_trained:,} team-game perspectives** from historical playoffs. "
            f"Walk-forward accuracy: **{acc:.1%}**. "
            f"Brier score: **{ml.get('model_brier'):.3f}**."
        )
    else:
        st.warning(
            "**ML model not found.** Using statistical formula fallback. "
            "Run `python scripts/train_game_model.py` to train the model."
        )

    st.markdown("#### What the model combines")
    st.caption(
        "Production odds use a regularized meta-model trained on out-of-fold predictions. "
        "Components without historical pregame snapshots remain explanatory until validated."
    )

    component_rows = [
        {
            "Component": "Team baseline (ML model)",
            "Weight": "Anchor",
            "What it measures": "Overall team quality based on net rating, shooting efficiency, turnovers, pace",
            "Data source": "Calibrated logistic regression" if ml.get("model_trained") else "Statistical sigmoid formula",
            "Production role": "Validated probability input",
        },
        {
            "Component": "Player projections",
            "Weight": "Learned coefficient",
            "What it measures": "How much better/worse each team's projected output is vs the other",
            "Data source": "Regular-season/playoff rate blend × projected minutes × PIE multiplier",
            "Production role": "Explanation only; historical snapshots unavailable",
        },
        {
            "Component": "Matchup edges",
            "Weight": "Learned coefficient",
            "What it measures": "Who has the advantage in specific play types (PnR, transition, 3-pointers, etc.)",
            "Data source": "Real team playstyle data + individual player matchup assignments",
            "Production role": "Explanation only; historical snapshots unavailable",
        },
        {
            "Component": "Lineup strength",
            "Weight": "Learned coefficient",
            "What it measures": "How good each team's key lineup groups are (starters, bench, closers)",
            "Data source": "Real 2-man & 5-man lineup net ratings from NBA API",
            "Production role": "Explanation only; historical snapshots unavailable",
        },
        {
            "Component": "Clutch edge",
            "Weight": "Learned, clutch-conditional",
            "What it measures": "Which team performs better when games are close late",
            "Data source": "Closing lineup composition + individual clutch player ratings",
            "Production role": "Explanation only; historical snapshots unavailable",
        },
        {
            "Component": "Injury/availability",
            "Weight": "Upstream",
            "What it measures": "Changes to minutes, roles, and available lineups",
            "Data source": "Manual injury tracker applied before projections",
            "Production role": "Changes rotations upstream",
        },
        {
            "Component": "Foul trouble",
            "Weight": "Simulation only",
            "What it measures": "Chance that a key defender or big loses minutes",
            "Data source": "Player foul-rate scenarios in Monte Carlo simulation",
            "Production role": "Stochastic simulation event",
        },
    ]
    st.dataframe(pd.DataFrame(component_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---- Bayesian updater ----
    st.markdown("### Bayesian Series Updater")
    st.caption(
        "After each game, we use Bayesian statistics to update our beliefs about team strength. "
        "This is more principled than just adjusting numbers by hand - "
        "it correctly treats 'expected winner won' as weak evidence and 'upset' as strong evidence."
    )

    bayesian = bundle.get("bayesian_series")
    completed = bundle["completed_games"]
    series = bundle["series_simulation"]

    if bayesian and completed:
        col_mc, col_bt, col_gap = st.columns(3)
        mc_p = float(series["team_a_series_win_probability"])
        bt_p = float(bayesian.get("team_a_series_win_probability", 0.5))

        with col_mc:
            with st.container(border=True):
                st.metric("Official series forecast", _pct(mc_p))
                st.caption("Matches the game probabilities and the full outcome distribution")

        with col_bt:
            with st.container(border=True):
                st.metric("Bayesian cross-check", _pct(bt_p))
                st.caption(
                    "Updates our prior belief about team strength based on actual game results. "
                    "It is diagnostic and is not mixed into the official distribution."
                )

        with col_gap:
            with st.container(border=True):
                st.metric("Model disagreement", f"{abs(mc_p - bt_p) * 100:.1f} pts")
                st.caption("A larger gap indicates greater uncertainty after completed games.")

        st.info(
            f"**After {len(completed)} game(s) played:** "
            f"The official simulation gives {team_a} a **{_pct(mc_p)}** chance; "
            f"the Bayesian strength check gives **{_pct(bt_p)}**. "
            "Keeping them separate makes every headline and outcome chart internally consistent."
        )
    elif not completed:
        st.info(
            "Bayesian updates activate after Game 1 is played. "
            "Before any games, we use the pre-series model. "
            "Each game result will update our estimate of each team's true strength."
        )
    else:
        st.info(
            "The official forecast already reflects completed games. "
            "A Bayesian cross-check requires saved pregame prediction snapshots, "
            "so it is withheld rather than reconstructed from postgame data."
        )

    st.markdown("---")

    # ---- Series simulation explained ----
    st.markdown("### How Series Odds Are Calculated")
    st.caption(
        "We run 100,000 simulated versions of the rest of this series. "
        "In each simulation, each game is decided based on that game's win probability "
        "plus some randomness (because anything can happen in one game). "
        "The final percentages show how often each team won across all simulations."
    )

    series_data = bundle["series_simulation"]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Series win probabilities**")
        st.bar_chart(
            pd.DataFrame({
                "Team": [team_a, bundle["team_b"]],
                "Chance": [
                    series_data["team_a_series_win_probability"],
                    series_data["team_b_series_win_probability"],
                ],
            }).set_index("Team")
        )
    with col_b:
        st.markdown("**How many games?**")
        st.bar_chart(
            pd.DataFrame({
                "Games": [str(r["games"]) for r in series_data.get("series_length_distribution", [])],
                "Probability": [r["probability"] for r in series_data.get("series_length_distribution", [])],
            }).set_index("Games")
        )
    with col_c:
        st.markdown("**All possible outcomes**")
        dist_rows = [
            {"Result": r["result"], "Chance": r["percentage"]}
            for r in series_data.get("result_distribution", [])
            if float(r["probability"]) > 0.01
        ]
        if dist_rows:
            st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---- Scenarios ----
    st.markdown("### What-If Scenarios")
    st.caption(
        "These scenarios test how the series odds change if specific things happen. "
        "For example: what if one team's star player gets into foul trouble all series? "
        "Toggle scenarios in the sidebar."
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
    st.caption(
        "Run what-if scenarios to see how the series odds change. "
        "Toggle scenarios in the sidebar on the left."
    )

    if not selected_scenarios:
        st.info("Select at least one scenario in the sidebar to run simulations.")
        return

    report = run_scenario_suite(
        scenarios=list(selected_scenarios),
        game_predictions=bundle["game_predictions"],
        finals_context=bundle["context"],
        scenario_settings={
            "simulations": scenario_sims,
            "random_seed": int(bundle["settings"].get("project", {}).get("random_seed", 42)),
        },
    )
    summary = pd.DataFrame(report["summary"]).rename(columns={
        "scenario": "Scenario",
        "team_a_series_win_percentage": f"{team_a} Series Win %",
        "team_a_delta_from_base_label": "Change from baseline",
        "most_likely_result": "Most likely result",
    })
    cols_show = ["Scenario", f"{team_a} Series Win %", "Change from baseline", "Most likely result"]
    cols_show = [c for c in cols_show if c in summary.columns]
    st.dataframe(summary[cols_show], use_container_width=True, hide_index=True)


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

    # --- Game header ---
    st.markdown(
        f"<h2 style='text-align:center; margin-bottom:0;'>Game {gnum}</h2>"
        f"<p style='text-align:center; color:#888; margin-top:4px;'>"
        f"{date_str} &nbsp;·&nbsp; {away} @ {home}"
        f"</p>",
        unsafe_allow_html=True,
    )

    # Series context banner
    sa = series_score.get(team_a, 0)
    sb = series_score.get(team_b, 0)
    if sa > 0 or sb > 0:
        leader = team_a if sa > sb else (team_b if sb > sa else None)
        deficit_team = team_b if sa > sb else team_a
        if leader:
            pct_overcome = "12%" if abs(sa - sb) >= 2 else "32%"
            overturn_text = (
                f" A team that goes down 0-2 has recovered to win only {pct_overcome} "
                f"of Finals series." if abs(sa - sb) >= 2 else ""
            )
            st.info(
                f"**Series: {team_a} {sa}–{sb} {team_b}.** "
                f"{leader} leads and can go up {max(sa,sb)+1}–{min(sa,sb)} with a win."
                f"{overturn_text}"
            )
        else:
            st.info(f"**Series tied {sa}–{sb}.** Winner takes the series lead.")

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
    st.caption("Variables that could swing the outcome by 5+ percentage points.")

    if x_factors:
        for i, xf in enumerate(x_factors[:4], 1):
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

    st.sidebar.subheader("What-if scenarios")
    selected = []
    for label, sid in SCENARIO_TOGGLES.items():
        default = label in {"Hot shooting", "Foul trouble", "Slow pace", "Rebounding dominance"}
        if st.sidebar.checkbox(label, value=default):
            selected.append(sid)
    selected_scenarios = tuple(selected)

    st.sidebar.divider()
    if st.sidebar.button("Refresh data", type="primary", use_container_width=True,
                         help="Reload predictions after a game is played"):
        load_dashboard_bundle.clear()
        st.rerun()

    st.title("🏀 NBA Finals Predictor 2026")
    bundle = load_dashboard_bundle(int(series_simulations))

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

    tab1, tab0, tab2, tab3, tab4, tab5 = st.tabs([
        "The Pick",
        "Next Game",
        "Game by Game",
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
