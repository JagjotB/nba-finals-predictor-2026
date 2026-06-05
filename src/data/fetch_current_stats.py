"""Fetch and cache current-season team and player stats for Finals teams."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "stats_cache"
NYK_ID = 1610612752
SAS_ID = 1610612759
TEAM_IDS = {"NYK": NYK_ID, "SAS": SAS_ID}


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.json"


def _save_cache(name: str, data: Any) -> None:
    with _cache_path(name).open("w", encoding="utf-8") as f:
        json.dump(data, f)


def _load_cache(name: str) -> Any | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _endpoint_with_timeout(endpoint_cls: Any, **params: Any) -> Any:
    try:
        return endpoint_cls(**params, timeout=60)
    except TypeError:
        return endpoint_cls(**params)


def _coerce(val: Any) -> Any:
    """Convert numpy/pandas types to plain Python for JSON serialization."""
    import pandas as pd
    if pd.isna(val) if not isinstance(val, (list, dict)) else False:
        return None
    try:
        import numpy as np
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return round(float(val), 4)
        if isinstance(val, (np.bool_,)):
            return bool(val)
    except ImportError:
        pass
    if isinstance(val, float):
        return round(val, 4)
    return val


def _df_to_team_dict(df: Any, team_id: int) -> dict[str, Any]:
    rows = df[df["TEAM_ID"] == team_id]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {col: _coerce(val) for col, val in row.items()}


def fetch_team_stats(
    season: str = "2025-26",
    season_type: str = "Playoffs",
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fetch advanced + four-factor + scoring stats for NYK and SAS.

    Returns {team_abbr: merged_stat_dict}.
    Results are cached to disk; use force_refresh=True to re-pull.
    """
    cache_key = f"team_stats_{season}_{season_type.replace(' ', '_')}"
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached:
            return cached

    try:
        from nba_api.stats.endpoints import leaguedashteamstats
    except ImportError:
        return {}

    merged: dict[str, dict[str, Any]] = {abbr: {} for abbr in TEAM_IDS}

    for measure in ("Advanced", "Four Factors", "Scoring"):
        try:
            ep = _endpoint_with_timeout(
                leaguedashteamstats.LeagueDashTeamStats,
                season=season,
                season_type_all_star=season_type,
                measure_type_detailed_defense=measure,
                per_mode_detailed="PerGame",
            )
            df = ep.get_data_frames()[0]
            for abbr, team_id in TEAM_IDS.items():
                row = _df_to_team_dict(df, team_id)
                merged[abbr].update(row)
            time.sleep(0.6)
        except Exception:
            pass

    _save_cache(cache_key, merged)
    return merged


def fetch_player_stats(
    season: str = "2025-26",
    season_type: str = "Playoffs",
    force_refresh: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch advanced + traditional per-game stats for all Finals players.

    Returns {team_abbr: [player_stat_dicts]}.
    """
    cache_key = f"player_stats_{season}_{season_type.replace(' ', '_')}"
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached:
            return cached

    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        import pandas as pd
    except ImportError:
        return {}

    result: dict[str, list[dict[str, Any]]] = {"NYK": [], "SAS": []}
    frames = []

    for measure in ("Base", "Advanced"):
        try:
            ep = _endpoint_with_timeout(
                leaguedashplayerstats.LeagueDashPlayerStats,
                season=season,
                season_type_all_star=season_type,
                measure_type_detailed_defense=measure,
                per_mode_detailed="PerGame",
            )
            frames.append(ep.get_data_frames()[0])
            time.sleep(0.6)
        except Exception:
            pass

    if not frames:
        return result

    # Merge on PLAYER_ID
    merged_df = frames[0]
    for extra in frames[1:]:
        overlap = [
            c for c in extra.columns
            if c in merged_df.columns and c != "PLAYER_ID"
        ]
        extra = extra.drop(columns=overlap, errors="ignore")
        merged_df = merged_df.merge(extra, on="PLAYER_ID", how="left")

    for abbr, team_id in TEAM_IDS.items():
        team_rows = merged_df[merged_df["TEAM_ID"] == team_id]
        players = []
        for _, row in team_rows.iterrows():
            d = {
                col: (None if pd.isna(val) else
                      round(float(val), 4) if isinstance(val, float) else val)
                for col, val in row.items()
            }
            players.append(d)
        result[abbr] = players

    _save_cache(cache_key, result)
    return result


def fetch_lineup_stats(
    season: str = "2025-26",
    season_type: str = "Playoffs",
    group_quantity: int = 2,
    force_refresh: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch lineup net ratings for both Finals teams (one group size).

    Returns {team_abbr: [lineup_dicts]}.
    """
    cache_key = (
        f"lineup_stats_{group_quantity}man_"
        f"{season}_{season_type.replace(' ', '_')}"
    )
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached:
            return cached

    try:
        from nba_api.stats.endpoints import leaguedashlineups
        import pandas as pd
    except ImportError:
        return {}

    result: dict[str, list[dict[str, Any]]] = {"NYK": [], "SAS": []}
    try:
        ep = _endpoint_with_timeout(
            leaguedashlineups.LeagueDashLineups,
            season=season,
            season_type_all_star=season_type,
            group_quantity=group_quantity,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
        )
        df = ep.get_data_frames()[0]
        for abbr, team_id in TEAM_IDS.items():
            team_df = df[df["TEAM_ID"] == team_id]
            lineups = []
            for _, row in team_df.iterrows():
                d = {col: _coerce(val) for col, val in row.items()}
                lineups.append(d)
            result[abbr] = lineups
    except Exception:
        pass

    _save_cache(cache_key, result)
    return result


def fetch_all_lineup_stats(
    season: str = "2025-26",
    season_type: str = "Playoffs",
    force_refresh: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch and merge 2-man and 5-man lineup stats for both Finals teams.

    Combining both gives the index both pair-level and full-lineup-level data.
    Returns {team_abbr: [lineup_dicts]}.
    """
    cache_key = f"lineup_stats_all_{season}_{season_type.replace(' ', '_')}"
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached:
            return cached

    merged: dict[str, list[dict[str, Any]]] = {"NYK": [], "SAS": []}
    for gq in (2, 5):
        stats = fetch_lineup_stats(
            season=season,
            season_type=season_type,
            group_quantity=gq,
            force_refresh=force_refresh,
        )
        for abbr in ("NYK", "SAS"):
            merged[abbr].extend(stats.get(abbr, []))
        time.sleep(0.6)

    _save_cache(cache_key, merged)
    return merged


def fetch_historical_playoff_logs(
    seasons: list[str] | None = None,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch game-by-game playoff results for model training.

    Returns a flat list of game dicts with MATCHUP, WL, PTS, etc.
    """
    if seasons is None:
        seasons = [
            "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
            "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
            "2024-25",
        ]

    cache_key = "historical_playoff_logs"
    if not force_refresh:
        cached = _load_cache(cache_key)
        cached_seasons = {
            str(row.get("SEASON_YEAR"))
            for row in (cached or [])
            if row.get("SEASON_YEAR")
        }
        if cached and set(seasons).issubset(cached_seasons):
            return cached

    try:
        from nba_api.stats.endpoints import teamgamelogs
        import pandas as pd
    except ImportError:
        return []

    all_rows: list[dict[str, Any]] = []
    for season in seasons:
        try:
            ep = _endpoint_with_timeout(
                teamgamelogs.TeamGameLogs,
                season_nullable=season,
                season_type_nullable="Playoffs",
            )
            df = ep.get_data_frames()[0]
            for _, row in df.iterrows():
                d = {
                    col: (None if pd.isna(val) else
                          str(val) if col in ("GAME_DATE", "MATCHUP", "WL",
                                              "TEAM_ABBREVIATION",
                                              "SEASON_YEAR") else
                          int(val) if isinstance(val, float) and val == int(val)
                          else round(float(val), 4) if isinstance(val, float)
                          else val)
                    for col, val in row.items()
                }
                all_rows.append(d)
            time.sleep(0.8)
        except Exception:
            continue

    _save_cache(cache_key, all_rows)
    return all_rows


def fetch_historical_team_ratings(
    seasons: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fetch end-of-regular-season advanced ratings for all teams, all seasons.

    Returns {season: {team_abbr: stat_dict}}.
    Used as features in the logistic regression training pipeline.
    """
    if seasons is None:
        seasons = [
            "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
            "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
            "2024-25",
        ]

    cache_key = "historical_team_ratings"
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached and set(seasons).issubset(set(cached)):
            return cached

    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        import pandas as pd
    except ImportError:
        return {}

    all_ratings: dict[str, dict[str, Any]] = {}
    for season in seasons:
        try:
            ep = _endpoint_with_timeout(
                leaguedashteamstats.LeagueDashTeamStats,
                season=season,
                season_type_all_star="Regular Season",
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
            )
            df = ep.get_data_frames()[0]
            season_ratings: dict[str, Any] = {}
            for _, row in df.iterrows():
                abbr = str(row.get("TEAM_ABBREVIATION", "")).strip()
                if not abbr:
                    # Fall back to TEAM_NAME
                    abbr = str(row.get("TEAM_NAME", "")).strip()
                season_ratings[str(int(row["TEAM_ID"]))] = {
                    col: (None if pd.isna(val) else
                          round(float(val), 4) if isinstance(val, float)
                          else val)
                    for col, val in row.items()
                }
            all_ratings[season] = season_ratings
            time.sleep(0.8)
        except Exception:
            continue

    _save_cache(cache_key, all_ratings)
    return all_ratings
