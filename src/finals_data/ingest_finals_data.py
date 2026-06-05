"""
2026 NBA Finals Data Ingestion Module — main entry point.

Usage:
    python -m src.finals_data.ingest_finals_data

Steps:
  1. Load Finals config from settings.yaml
  2. Fetch current regular-season stats from NBA API
  3. Fetch current playoff stats from NBA API (all lineup sizes)
  4. Auto-populate rotations.csv from playoff minutes if placeholders detected
  5. Load all manual data files
  6. Validate all inputs
  7. Build and save data/processed/finals_context/finals_context_inputs.json
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUAL_DIR = PROJECT_ROOT / "data" / "manual"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

REGULAR_SEASON_DIR = PROCESSED_DIR / "current_regular_season"
PLAYOFFS_DIR = PROCESSED_DIR / "current_playoffs"

API_DELAY = 0.6  # seconds between NBA API calls to avoid rate-limit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_settings() -> dict[str, Any]:
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_nba_api() -> bool:
    try:
        import nba_api  # noqa: F401
        return True
    except ImportError:
        return False


def _load_manual_csv(filename: str) -> pd.DataFrame:
    path = MANUAL_DIR / filename
    if not path.exists():
        print(f"  [WARN] {filename} not found — returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Loaded {filename} ({len(df)} rows).")
    return df


def _has_placeholder_data(rotations: pd.DataFrame) -> bool:
    if rotations.empty:
        return True
    return bool((rotations["player"] == "Player Name").any())


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _filter_teams(
    df: pd.DataFrame | None, col: str, teams: list[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if col not in df.columns:
        return df
    return df[df[col].isin(teams)].reset_index(drop=True)


def _call_api(label: str, fn, **kwargs) -> pd.DataFrame | None:
    """Call an NBA API function, sleep after, and return None on failure."""
    print(f"  Fetching {label}...")
    try:
        result = fn(**kwargs)
        time.sleep(API_DELAY)
        return result
    except Exception as exc:
        print(f"  [WARN] {label} failed: {exc}")
        time.sleep(API_DELAY)
        return None


# ---------------------------------------------------------------------------
# API fetchers
# ---------------------------------------------------------------------------

def _fetch_regular_season(season: str, teams: list[str]) -> bool:
    from src.data.fetch_nba_api import fetch_team_stats, fetch_player_stats

    REGULAR_SEASON_DIR.mkdir(parents=True, exist_ok=True)

    team_df = _call_api(
        "regular-season team stats", fetch_team_stats,
        season=season, season_type="Regular Season",
    )
    if team_df is not None:
        _save_csv(_filter_teams(team_df, "TEAM_ABBREVIATION", teams),
                  REGULAR_SEASON_DIR / "team_stats.csv")

    player_df = _call_api(
        "regular-season player stats", fetch_player_stats,
        season=season, season_type="Regular Season",
    )
    if player_df is not None:
        _save_csv(_filter_teams(player_df, "TEAM_ABBREVIATION", teams),
                  REGULAR_SEASON_DIR / "player_stats.csv")

    return team_df is not None


def _fetch_playoffs(season: str, teams: list[str]) -> bool:
    from src.data.fetch_nba_api import (
        fetch_team_stats, fetch_player_stats, fetch_lineup_stats,
    )

    PLAYOFFS_DIR.mkdir(parents=True, exist_ok=True)

    team_df = _call_api(
        "playoff team stats", fetch_team_stats,
        season=season, season_type="Playoffs",
    )
    if team_df is not None:
        _save_csv(_filter_teams(team_df, "TEAM_ABBREVIATION", teams),
                  PLAYOFFS_DIR / "team_stats.csv")

    player_df = _call_api(
        "playoff player stats", fetch_player_stats,
        season=season, season_type="Playoffs",
    )
    if player_df is not None:
        _save_csv(_filter_teams(player_df, "TEAM_ABBREVIATION", teams),
                  PLAYOFFS_DIR / "player_stats.csv")

    for size in [5, 4, 3, 2]:
        lineup_df = _call_api(
            f"playoff {size}-man lineups", fetch_lineup_stats,
            season=season, season_type="Playoffs", group_quantity=size,
        )
        if lineup_df is not None:
            _save_csv(_filter_teams(lineup_df, "TEAM_ABBREVIATION", teams),
                      PLAYOFFS_DIR / f"lineups_{size}man.csv")

    return team_df is not None


# ---------------------------------------------------------------------------
# Auto-roster population
# ---------------------------------------------------------------------------

def _build_rotation_row(
    player: str, team: str, gp: float, mpg: float,
) -> dict[str, Any]:
    # MIN from LeagueDashPlayerStats is a season total; divide by GP for per-game
    is_starter = mpg >= 25.0

    role = "Starter" if is_starter else ("Bench" if mpg >= 14 else "Reserve")
    confidence = "high" if mpg >= 22 else ("medium" if mpg >= 10 else "low")
    floor = round(max(0.0, mpg - 5), 1)
    ceiling = round(min(48.0, mpg + 5), 1)

    return {
        "team": team,
        "player": player,
        "role": role,
        "projected_minutes": round(mpg, 1),
        "minutes_floor": floor,
        "minutes_ceiling": ceiling,
        "is_starter": is_starter,
        "is_closer": is_starter,
        "rotation_confidence": confidence,
        "notes": (
            f"Auto-populated — {gp:.0f} GP in 2025-26 playoffs. "
            "Review before predicting."
        ),
    }


def _auto_populate_rotations(teams: list[str]) -> pd.DataFrame | None:
    player_csv = PLAYOFFS_DIR / "player_stats.csv"
    if not player_csv.exists():
        print(
            "  [WARN] Playoff player_stats.csv not found"
            " — cannot auto-populate."
        )
        return None

    df = pd.read_csv(player_csv)
    # GS is not returned by LeagueDashPlayerStats; MIN is a season total
    required = {"PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"  [WARN] Missing columns for auto-populate: {missing}")
        return None

    rows: list[dict[str, Any]] = []
    for team in teams:
        team_df = (
            df[df["TEAM_ABBREVIATION"] == team]
            .copy()
        )
        if team_df.empty:
            print(f"  [WARN] No playoff player data for {team}.")
            continue

        gp_safe = team_df["GP"].replace(0, float("nan"))
        team_df["_mpg"] = team_df["MIN"] / gp_safe
        team_df = team_df.sort_values("_mpg", ascending=False).head(13)

        for _, row in team_df.iterrows():
            mpg = float(row["_mpg"]) if not pd.isna(row["_mpg"]) else 0.0
            if mpg < 3.0:
                continue
            rows.append(_build_rotation_row(
                player=str(row["PLAYER_NAME"]),
                team=team,
                gp=float(row["GP"]),
                mpg=mpg,
            ))

    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Status printer
# ---------------------------------------------------------------------------

def _status(label: str, ok: bool) -> None:
    print(f"  {'[OK]' if ok else '[--]'} {label}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("2026 NBA Finals — Data Ingestion Module")
    print("=" * 60)

    settings = _load_settings()
    finals = settings["finals"]
    season = settings["project"]["season"]
    team_a = finals["team_a_abbr"]
    team_b = finals["team_b_abbr"]
    teams = [team_a, team_b]

    print(f"\nSeries : {finals['series_name']}")
    print(f"Teams  : {team_a} vs {team_b}")
    print(f"Season : {season}")

    # --- Check nba_api ---
    has_api = _check_nba_api()
    if not has_api:
        print(
            "\n[ERROR] nba_api is not installed.\n"
            "  Run: pip install nba_api\n"
            "  Continuing with manual data only.\n"
        )

    # --- Fetch API data ---
    regular_season_ok = False
    playoffs_ok = False

    if has_api:
        print(f"\n[1/4] Fetching regular-season data ({season})...")
        try:
            regular_season_ok = _fetch_regular_season(season, teams)
        except Exception as exc:
            print(f"  [ERROR] {exc}")

        print(f"\n[2/4] Fetching playoff data ({season})...")
        try:
            playoffs_ok = _fetch_playoffs(season, teams)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
    else:
        print("\n[1/4] Skipped — nba_api not installed.")
        print("[2/4] Skipped — nba_api not installed.")

    # --- Load manual files ---
    print("\n[3/4] Loading manual data files...")
    schedule = _load_manual_csv("finals_schedule.csv")
    rotations = _load_manual_csv("rotations.csv")
    injuries = _load_manual_csv("injuries.csv")
    player_matchups = _load_manual_csv("player_matchups.csv")
    coaching_notes = _load_manual_csv("coaching_notes.csv")

    # --- Auto-populate rotations if placeholders detected ---
    if _has_placeholder_data(rotations):
        print("\n  [INFO] Placeholder data in rotations.csv detected.")
        if playoffs_ok:
            print("  Auto-populating from 2025-26 playoff minutes...")
            auto_df = _auto_populate_rotations(teams)
            if auto_df is not None and not auto_df.empty:
                backup = MANUAL_DIR / "rotations.csv.bak"
                shutil.copy(MANUAL_DIR / "rotations.csv", backup)
                print(f"  Backup written to: {backup}")
                auto_df.to_csv(MANUAL_DIR / "rotations.csv", index=False)
                rotations = auto_df
                print(
                    f"  rotations.csv updated with {len(auto_df)} players.\n"
                    "  [ACTION] Review data/manual/rotations.csv"
                    " before running predictions."
                )
            else:
                print(
                    "  [WARN] Auto-population produced no rows."
                    " Fill rotations.csv manually."
                )
        else:
            print(
                "  [WARN] No playoff data available for auto-population.\n"
                "  Fill data/manual/rotations.csv with real player data."
            )

    # --- Validate ---
    print("\n[4/4] Validating inputs...")
    from src.finals_data.validate_finals_inputs import validate_all

    warnings = validate_all(
        schedule, rotations, injuries, player_matchups, coaching_notes,
        team_a, team_b,
    )

    if warnings:
        print(f"  {len(warnings)} validation warning(s):")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  All inputs valid.")

    # --- Build context JSON ---
    from src.finals_data.build_finals_context_inputs import build_and_save

    context_path = build_and_save(
        settings=settings,
        schedule=schedule,
        rotations=rotations,
        injuries=injuries,
        player_matchups=player_matchups,
        coaching_notes=coaching_notes,
        regular_season_available=regular_season_ok,
        playoffs_available=playoffs_ok,
        validation_warnings=warnings,
    )
    print(f"\n  Context JSON saved: {context_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Finals Data Status")
    print("=" * 60)
    _status("Schedule loaded", not schedule.empty)
    _status(
        "Rotations loaded (real players)",
        not rotations.empty and not _has_placeholder_data(rotations),
    )
    _status("Injuries loaded", not injuries.empty)
    _status("Player matchups loaded", not player_matchups.empty)
    _status(f"Regular-season data ({season})", regular_season_ok)
    _status(f"Playoff data ({season})", playoffs_ok)
    _status("Context JSON built", context_path.exists())
    print("=" * 60)

    ready = not rotations.empty and not _has_placeholder_data(rotations)
    if ready:
        print("\nReady to predict. Run:")
        print("  python -m src.models.predict_game")
        print("  python -m src.models.simulate_series")
        print("  streamlit run src/app/streamlit_app.py")
    else:
        print(
            "\n[ACTION REQUIRED] data/manual/rotations.csv"
            " still needs real player data.\n"
            "  Edit the file, then re-run:\n"
            "  python -m src.finals_data.ingest_finals_data"
        )


if __name__ == "__main__":
    main()
