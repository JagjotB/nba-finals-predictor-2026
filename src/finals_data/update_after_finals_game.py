"""
Post-game update for a completed Finals game.

Usage:
    python -m src.finals_data.update_after_finals_game --game 1

Requires a 'game_id' column in data/manual/finals_schedule.csv with the
NBA game ID (e.g. 0042500401) to fetch the live box score.
If game_id is missing, the script saves a stub report and prints instructions.

After running, call:
    python -m src.models.update_after_game --game <N>
    python -m src.models.predict_game
    python -m src.models.simulate_series
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
MANUAL_DIR = PROJECT_ROOT / "data" / "manual"
FINALS_GAMES_DIR = PROJECT_ROOT / "data" / "processed" / "finals_games"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CONTEXT_DIR = PROJECT_ROOT / "data" / "processed" / "finals_context"


def _load_settings() -> dict[str, Any]:
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _game_id_from_schedule(game_number: int) -> str | None:
    path = MANUAL_DIR / "finals_schedule.csv"
    if not path.exists():
        return None
    schedule = pd.read_csv(path, dtype={"game_id": str})
    if "game_id" not in schedule.columns:
        return None
    row = schedule[schedule["game_number"] == game_number]
    if row.empty or pd.isna(row.iloc[0]["game_id"]):
        return None
    return str(row.iloc[0]["game_id"])


def _fetch_box_score(game_id: str) -> dict[str, pd.DataFrame] | None:
    try:
        from src.data.fetch_nba_api import fetch_boxscore_advanced, fetch_boxscore_traditional
        advanced = fetch_boxscore_advanced(game_id)
        try:
            traditional = fetch_boxscore_traditional(game_id)
            advanced["player_traditional"] = traditional.get("player_stats")
            advanced["team_traditional"] = traditional.get("team_stats")
        except Exception as exc:
            print(f"  [WARN] Traditional box score fetch failed: {exc}")
        return advanced
    except Exception as exc:
        print(
            f"  [WARN] Box score fetch failed for game_id={game_id}: {exc}"
        )
        return None


def _load_projected_rotations() -> pd.DataFrame:
    path = CONTEXT_DIR / "projected_rotations.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _compare_projection_vs_actual(
    projected: pd.DataFrame, actual_players: pd.DataFrame,
) -> pd.DataFrame:
    if projected.empty or actual_players.empty:
        return pd.DataFrame()

    player_col = next(
        (
            c for c in ["PLAYER_NAME", "PLAYER", "player"]
            if c in actual_players.columns
        ),
        None,
    )
    min_col = next(
        (
            c for c in ["MIN", "MINUTES", "minutes"]
            if c in actual_players.columns
        ),
        None,
    )
    if not player_col or not min_col or "player" not in projected.columns:
        return pd.DataFrame()

    actual_renamed = actual_players[[player_col, min_col]].rename(
        columns={player_col: "player", min_col: "actual_minutes"}
    )
    merged = projected[["team", "player", "projected_minutes"]].merge(
        actual_renamed, on="player", how="outer",
    )
    merged["actual_minutes"] = pd.to_numeric(
        merged["actual_minutes"], errors="coerce"
    )
    merged["projected_minutes"] = pd.to_numeric(
        merged["projected_minutes"], errors="coerce"
    )
    merged["minutes_diff"] = (
        merged["actual_minutes"] - merged["projected_minutes"]
    )
    return merged.sort_values("actual_minutes", ascending=False)


def _write_report(
    game_number: int,
    game_id: str | None,
    box_score: dict | None,
    comparison: pd.DataFrame,
    settings: dict[str, Any],
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"game_{game_number}_update_report.md"
    finals = settings["finals"]

    lines = [
        f"# Game {game_number} Post-Game Update -- {finals['series_name']}",
        "",
        "## Data Status",
        f"- Game ID used : `{game_id or 'not available'}`",
        f"- Box score fetched : {'Yes' if box_score else 'No'}",
        f"- Projection comparison rows : {len(comparison)}",
        "",
    ]

    if not comparison.empty:
        lines.append("## Projection vs Actual Minutes")
        lines.append("")
        try:
            lines.append(comparison.to_markdown(index=False))
        except Exception:
            lines.append(comparison.to_string(index=False))
        lines.append("")

    if not box_score:
        lines += [
            "## How to add box score data manually",
            "",
            "1. Find the NBA game ID at stats.nba.com (e.g. `0042500401`).",
            "2. Add a `game_id` column to `data/manual/finals_schedule.csv`.",
            (
                "3. Re-run: "
                "`python -m src.finals_data.update_after_finals_game"
                f" --game {game_number}`"
            ),
        ]

    lines += [
        "",
        "## Next steps",
        "",
        "```",
        f"python -m src.models.update_after_game --game {game_number}",
        "python -m src.models.predict_game",
        "python -m src.models.simulate_series",
        "```",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(game_number: int) -> None:
    print(f"Post-game update: Game {game_number}")
    FINALS_GAMES_DIR.mkdir(parents=True, exist_ok=True)

    settings = _load_settings()
    game_id = _game_id_from_schedule(game_number)
    box_score: dict | None = None

    if game_id:
        print(f"  Fetching box score (game_id={game_id})...")
        box_score = _fetch_box_score(game_id)
    else:
        print(
            f"  [INFO] No game_id in finals_schedule.csv for game"
            f" {game_number}.\n"
            "  Add a 'game_id' column with the NBA game ID to enable"
            " box score fetch."
        )

    if box_score:
        player_stats: pd.DataFrame | None = box_score.get("player_stats")
        team_stats: pd.DataFrame | None = box_score.get("team_stats")

        if player_stats is not None and not player_stats.empty:
            p_path = (
                FINALS_GAMES_DIR / f"game_{game_number}_player_actuals.csv"
            )
            player_stats.to_csv(p_path, index=False)
            print(f"  Player actuals -> {p_path}")

        if team_stats is not None and not team_stats.empty:
            t_path = (
                FINALS_GAMES_DIR / f"game_{game_number}_team_actuals.csv"
            )
            team_stats.to_csv(t_path, index=False)
            print(f"  Team actuals   -> {t_path}")

        player_trad: pd.DataFrame | None = box_score.get("player_traditional")
        team_trad: pd.DataFrame | None = box_score.get("team_traditional")

        if player_trad is not None and not player_trad.empty:
            pt_path = (
                FINALS_GAMES_DIR / f"game_{game_number}_player_traditional.csv"
            )
            player_trad.to_csv(pt_path, index=False)
            print(f"  Player traditional -> {pt_path}")

        if team_trad is not None and not team_trad.empty:
            tt_path = (
                FINALS_GAMES_DIR / f"game_{game_number}_team_traditional.csv"
            )
            team_trad.to_csv(tt_path, index=False)
            print(f"  Team traditional   -> {tt_path}")

        projected = _load_projected_rotations()
        actual_df = box_score.get("player_stats", pd.DataFrame())
        comparison = _compare_projection_vs_actual(projected, actual_df)

        if not comparison.empty:
            c_path = (
                FINALS_GAMES_DIR
                / f"game_{game_number}_projection_vs_actual.csv"
            )
            comparison.to_csv(c_path, index=False)
            print(f"  Comparison     -> {c_path}")

            big_misses = comparison[
                comparison["minutes_diff"].abs() > 8
            ].dropna(subset=["minutes_diff"])
            if not big_misses.empty:
                print("\n  Large minutes deviations (>8 min):")
                for _, r in big_misses.iterrows():
                    diff_str = f"{r['minutes_diff']:+.1f}"
                    print(
                        f"    {r['player']:30s}"
                        f" projected={r['projected_minutes']:.1f}"
                        f"  actual={r['actual_minutes']:.1f}"
                        f"  diff={diff_str}"
                    )
    else:
        comparison = pd.DataFrame()

    report_path = _write_report(
        game_number, game_id, box_score, comparison, settings,
    )
    print(f"\n  Report -> {report_path}")

    print("\nNext steps:")
    print(f"  python -m src.models.update_after_game --game {game_number}")
    print("  python -m src.models.predict_game")
    print("  python -m src.models.simulate_series")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-game Finals update")
    parser.add_argument(
        "--game", type=int, required=True,
        choices=range(1, 8), help="Game number (1-7)",
    )
    args = parser.parse_args()
    main(args.game)
