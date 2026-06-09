"""Data collection helpers for stats.nba.com via nba_api."""

from __future__ import annotations

from typing import Any


DEFAULT_TIMEOUT = 60


def _endpoint_with_timeout(endpoint_cls: Any, **params: Any) -> Any:
    params_with_timeout = {**params, "timeout": DEFAULT_TIMEOUT}
    try:
        return endpoint_cls(**params_with_timeout)
    except TypeError as exc:
        if "timeout" not in str(exc):
            raise
        return endpoint_cls(**params)


def _first_data_frame(endpoint: Any) -> Any:
    frames = endpoint.get_data_frames()
    if not frames:
        raise ValueError("nba_api returned no data frames for this request.")
    return frames[0]


def _nba_api_import_error(exc: ImportError) -> ImportError:
    return ImportError(
        "nba_api is required for primary NBA data collection. "
        "Install it with `pip install nba_api`."
    ).with_traceback(exc.__traceback__)


def fetch_team_stats(season: str, season_type: str) -> Any:
    """Fetch league-level team stats for a season and season type."""
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
    except ImportError as exc:
        raise _nba_api_import_error(exc) from exc

    endpoint = _endpoint_with_timeout(
        leaguedashteamstats.LeagueDashTeamStats,
        season=season,
        season_type_all_star=season_type,
    )
    return _first_data_frame(endpoint)


def fetch_team_stats_clutch(season: str, season_type: str = "Regular Season") -> Any:
    """Fetch clutch-situation team stats (last 5 min, within 5 pts)."""
    try:
        from nba_api.stats.endpoints import leaguedashteamclutch
    except ImportError as exc:
        raise _nba_api_import_error(exc) from exc

    endpoint = _endpoint_with_timeout(
        leaguedashteamclutch.LeagueDashTeamClutch,
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense="Advanced",
        clutch_time="Last 5 Minutes",
        ahead_behind="Ahead or Behind",
        point_diff="5",
    )
    return _first_data_frame(endpoint)


def fetch_player_stats(season: str, season_type: str) -> Any:
    """Fetch league-level player stats for a season and season type."""
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
    except ImportError as exc:
        raise _nba_api_import_error(exc) from exc

    endpoint = _endpoint_with_timeout(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=season,
        season_type_all_star=season_type,
    )
    return _first_data_frame(endpoint)


def fetch_team_game_logs(season: str, season_type: str) -> Any:
    """Fetch team game logs for a season and season type."""
    try:
        from nba_api.stats.endpoints import teamgamelogs
    except ImportError as exc:
        raise _nba_api_import_error(exc) from exc

    endpoint = _endpoint_with_timeout(
        teamgamelogs.TeamGameLogs,
        season_nullable=season,
        season_type_nullable=season_type,
    )
    return _first_data_frame(endpoint)


def fetch_player_game_logs(player_id: int | str, season: str, season_type: str) -> Any:
    """Fetch game logs for one player."""
    try:
        from nba_api.stats.endpoints import playergamelog
    except ImportError as exc:
        raise _nba_api_import_error(exc) from exc

    endpoint = _endpoint_with_timeout(
        playergamelog.PlayerGameLog,
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
    )
    return _first_data_frame(endpoint)


def fetch_lineup_stats(season: str, season_type: str, group_quantity: int) -> Any:
    """Fetch lineup stats for two-, three-, four-, or five-player groups."""
    if int(group_quantity) not in {2, 3, 4, 5}:
        raise ValueError("group_quantity must be one of 2, 3, 4, or 5.")

    try:
        from nba_api.stats.endpoints import leaguedashlineups
    except ImportError as exc:
        raise _nba_api_import_error(exc) from exc

    endpoint = _endpoint_with_timeout(
        leaguedashlineups.LeagueDashLineups,
        season=season,
        season_type_all_star=season_type,
        group_quantity=group_quantity,
    )
    return _first_data_frame(endpoint)


def fetch_boxscore_traditional(game_id: str) -> dict[str, Any]:
    """Fetch traditional box score (PTS, FGM, FGA, FG3M, FG3A, REB, AST, TOV, etc.)."""
    try:
        from nba_api.stats.endpoints import boxscoretraditionalv3

        endpoint = _endpoint_with_timeout(
            boxscoretraditionalv3.BoxScoreTraditionalV3,
            game_id=game_id,
        )
    except ImportError as first_exc:
        try:
            from nba_api.stats.endpoints import boxscoretraditionalv2
        except ImportError as second_exc:
            raise _nba_api_import_error(first_exc) from second_exc

        endpoint = _endpoint_with_timeout(
            boxscoretraditionalv2.BoxScoreTraditionalV2,
            game_id=game_id,
        )

    frames = endpoint.get_data_frames()
    if len(frames) < 2:
        raise ValueError(
            "nba_api returned incomplete traditional box score data."
        )

    return {
        "player_stats": frames[0],
        "team_stats": frames[1],
    }


def fetch_boxscore_advanced(game_id: str) -> dict[str, Any]:
    """Fetch advanced box score data for one game.

    Returns separate player and team tables because the NBA endpoint exposes both.
    """
    try:
        from nba_api.stats.endpoints import boxscoreadvancedv3

        endpoint = _endpoint_with_timeout(
            boxscoreadvancedv3.BoxScoreAdvancedV3,
            game_id=game_id,
        )
    except ImportError as first_exc:
        try:
            from nba_api.stats.endpoints import boxscoreadvancedv2
        except ImportError as second_exc:
            raise _nba_api_import_error(first_exc) from second_exc

        endpoint = _endpoint_with_timeout(
            boxscoreadvancedv2.BoxScoreAdvancedV2,
            game_id=game_id,
        )

    frames = endpoint.get_data_frames()
    if len(frames) < 2:
        raise ValueError("nba_api returned incomplete advanced box score data.")

    return {
        "player_stats": frames[0],
        "team_stats": frames[1],
    }
