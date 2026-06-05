"""Optional Basketball-Reference playoff data collection helpers."""

from __future__ import annotations

from typing import Any


BASE_PLAYOFFS_URL = "https://www.basketball-reference.com/playoffs/NBA_{year}_{slug}.html"

STAT_TABLES = {
    "per_game": ("per_game", "per_game_stats"),
    "advanced": ("advanced", "advanced_stats"),
    "per_100": ("per_poss", "per_poss_stats"),
}


def _empty_frame() -> Any:
    try:
        import pandas as pd
    except ImportError:
        return []
    return pd.DataFrame()


def _clean_bref_table(frame: Any, season_end_year: int) -> Any:
    if hasattr(frame.columns, "to_flat_index"):
        frame.columns = [
            "_".join(str(part) for part in column if str(part) != "nan").strip("_")
            if isinstance(column, tuple)
            else str(column)
            for column in frame.columns.to_flat_index()
        ]

    for column in ("Rk", "Player", "Team"):
        if column in frame.columns:
            frame = frame[frame[column].astype(str) != column]

    frame = frame.reset_index(drop=True)
    frame["season_end_year"] = season_end_year
    return frame


def fetch_playoff_table(season_end_year: int, stat_type: str) -> Any:
    """Fetch one Basketball-Reference playoff table.

    This source is intentionally optional: any import, network, parsing, or
    site-layout failure returns an empty DataFrame instead of stopping the app.
    """
    if stat_type not in STAT_TABLES:
        raise ValueError(f"Unknown stat_type: {stat_type}")

    slug, table_id = STAT_TABLES[stat_type]
    url = BASE_PLAYOFFS_URL.format(year=season_end_year, slug=slug)

    try:
        import pandas as pd

        tables = pd.read_html(url, attrs={"id": table_id})
        if not tables:
            return pd.DataFrame()
        return _clean_bref_table(tables[0], int(season_end_year))
    except Exception:
        return _empty_frame()


def fetch_playoff_per_game_stats(season_end_year: int) -> Any:
    """Fetch playoff per-game player stats."""
    return fetch_playoff_table(season_end_year, "per_game")


def fetch_playoff_advanced_stats(season_end_year: int) -> Any:
    """Fetch playoff advanced player stats."""
    return fetch_playoff_table(season_end_year, "advanced")


def fetch_playoff_per_100_poss_stats(season_end_year: int) -> Any:
    """Fetch playoff per-100-possession player stats."""
    return fetch_playoff_table(season_end_year, "per_100")


def fetch_historical_playoff_data(
    start_season_end_year: int,
    end_season_end_year: int,
    stat_type: str = "per_game",
) -> Any:
    """Fetch and combine Basketball-Reference playoff data across seasons."""
    try:
        import pandas as pd
    except ImportError:
        return []

    frames = [
        fetch_playoff_table(year, stat_type)
        for year in range(int(start_season_end_year), int(end_season_end_year) + 1)
    ]
    frames = [frame for frame in frames if hasattr(frame, "empty") and not frame.empty]

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
