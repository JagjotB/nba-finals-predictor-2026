"""Fetch regular-season clutch net rating per team per season.

Clutch = last 5 minutes of the second half, game within 5 points.
Using regular-season clutch stats as a leakage-free prior for playoff games.

Output cached at data/processed/stats_cache/clutch_stats_cache.json:
  {season: {team_abbr: clutch_net_rating}}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stats_cache" / "clutch_stats_cache.json"
)


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(data: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_clutch_stats(
    seasons: list[str],
    force_refresh: bool = False,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Return {season: {team_abbr: clutch_net_rating}} for all seasons."""
    from src.data.fetch_nba_api import fetch_team_stats_clutch

    cache = _load_cache() if not force_refresh else {}
    seasons_needed = [s for s in seasons if s not in cache]

    if not seasons_needed:
        if verbose:
            print(f"  Clutch stats: all {len(seasons)} seasons cached.")
        return {k: v for k, v in cache.items() if k in set(seasons)}

    for season in seasons_needed:
        if verbose:
            print(f"  Fetching clutch stats for {season}...")
        try:
            df = fetch_team_stats_clutch(season, "Regular Season")
            time.sleep(1.0)
        except Exception as exc:
            if verbose:
                print(f"    [WARN] Could not fetch {season}: {exc}")
            continue

        if df is None or df.empty:
            continue

        season_clutch: dict[str, float] = {}
        for _, row in df.iterrows():
            team_id = str(int(row.get("TEAM_ID", 0)))
            net = row.get("NET_RATING") or row.get("E_NET_RATING")
            if team_id != "0" and net is not None:
                season_clutch[team_id] = round(float(net), 3)

        cache[season] = season_clutch
        if verbose:
            print(f"    {season}: {len(season_clutch)} teams")

    _save_cache(cache)
    return {k: v for k, v in cache.items() if k in set(seasons)}


def get_clutch_diff(
    season: str,
    team_a_id: str,
    team_b_id: str,
    clutch_cache: dict[str, dict[str, float]],
) -> tuple[float, int]:
    """Return (clutch_net_rating_diff, data_available).

    Keyed by numeric team_id string (e.g. '1610612752').
    Positive = team_a better in clutch situations.
    """
    season_data = clutch_cache.get(season, {})
    a = season_data.get(str(team_a_id))
    b = season_data.get(str(team_b_id))
    if a is None or b is None:
        return 0.0, 0
    return round(a - b, 3), 1


if __name__ == "__main__":
    import sys
    seasons = sys.argv[1:] if len(sys.argv) > 1 else [
        "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
        "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    ]
    print(f"Building clutch stats for {len(seasons)} seasons...")
    result = build_clutch_stats(seasons, verbose=True)
    print(f"Done. {len(result)} seasons cached.")
    for season in list(result.keys())[-2:]:
        top = sorted(result[season].items(), key=lambda x: -x[1])[:3]
        print(f"  {season} top clutch teams: {top}")
