"""Compute top-5 regular-season pts/g sum per team per season.

Used as a leakage-free player projection proxy for the meta-model training:
regular season ends before the playoffs start, so no future data leaks in.

Output cached at data/processed/stats_cache/player_projections_cache.json:
  {season: {team_abbr: top5_reg_pts_sum}}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stats_cache" / "player_projections_cache.json"
)
TOP_N = 5


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(data: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_player_projections(
    seasons: list[str],
    force_refresh: bool = False,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Return {season: {team_abbr: top5_reg_season_pts_sum}} for all seasons."""
    from src.data.fetch_nba_api import fetch_player_stats

    cache = _load_cache() if not force_refresh else {}
    seasons_needed = [s for s in seasons if s not in cache]

    if not seasons_needed:
        if verbose:
            print(f"  Player projections: all {len(seasons)} seasons cached.")
        return {k: v for k, v in cache.items() if k in set(seasons)}

    for season in seasons_needed:
        if verbose:
            print(f"  Fetching regular season player stats for {season}...")
        try:
            df = fetch_player_stats(season, "Regular Season")
            time.sleep(0.8)
        except Exception as exc:
            if verbose:
                print(f"    [WARN] Could not fetch {season}: {exc}")
            continue

        if df is None or df.empty:
            continue

        gp_col = "GP" if "GP" in df.columns else None
        pts_col = "PTS" if "PTS" in df.columns else None
        if not gp_col or not pts_col:
            continue

        df = df.copy()
        df["avg_pts"] = df[pts_col] / df[gp_col].clip(lower=1)

        team_proj: dict[str, float] = {}
        for team_abbr, group in df.groupby("TEAM_ABBREVIATION"):
            top5 = group.nlargest(TOP_N, "avg_pts")["avg_pts"].sum()
            team_proj[str(team_abbr)] = round(float(top5), 3)

        cache[season] = team_proj
        if verbose:
            print(f"    {season}: {len(team_proj)} teams")

    _save_cache(cache)
    return {k: v for k, v in cache.items() if k in set(seasons)}


def get_player_projection_edge(
    season: str,
    team_a_abbr: str,
    team_b_abbr: str,
    projections: dict[str, dict[str, float]],
) -> tuple[float, int]:
    """Return (player_edge_pts, data_available) for a training row.

    player_edge_pts > 0 means team_a has more star scoring than team_b.
    Divided by 2 so the scale is comparable to a realistic per-game margin.
    """
    season_proj = projections.get(season)
    if not season_proj:
        return 0.0, 0
    pts_a = season_proj.get(str(team_a_abbr), 0.0)
    pts_b = season_proj.get(str(team_b_abbr), 0.0)
    if pts_a == 0.0 and pts_b == 0.0:
        return 0.0, 0
    edge = round((pts_a - pts_b) / 2.0, 3)
    return edge, 1


if __name__ == "__main__":
    import sys
    seasons = sys.argv[1:] if len(sys.argv) > 1 else [
        "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
        "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    ]
    print(f"Building player projections for {len(seasons)} seasons...")
    result = build_player_projections(seasons, verbose=True)
    print(f"Done. {len(result)} seasons cached.")
    # Sample output
    for season in list(result.keys())[-2:]:
        teams = sorted(result[season].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  {season} top teams: {teams}")
