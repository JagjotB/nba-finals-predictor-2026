"""Build a game-level injury proxy from player absence in NBA game logs.

For each historical playoff game we compute an injury_strength_diff:
  - Identify each team's top-6 players by season minutes
  - A player is 'absent' if they have no entry in their game log for a game
    their team played (playoff stars almost never rest — absence = injury)
  - injury_strength_score = sum of (absent_player_pts_share) for absent players
  - injury_strength_diff = team_a_score - team_b_score (positive = team_b more hurt)

Output: {game_id: {'team_a_abbr': str, 'team_b_abbr': str,
                    'injury_strength_diff': float, 'team_a_injury': float,
                    'team_b_injury': float}}
Cached at data/processed/stats_cache/injury_proxy_cache.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "stats_cache" / "injury_proxy_cache.json"
TOP_PLAYERS_PER_TEAM = 6
ABSENT_MINUTES_THRESHOLD = 5.0  # played < 5 min → treat as absent


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(data: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _fetch_with_retry(fn: Any, *args: Any, retries: int = 3, delay: float = 2.0) -> Any:
    for attempt in range(retries):
        try:
            return fn(*args)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))
    return None


def build_injury_proxy(
    seasons: list[str],
    force_refresh: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Return {game_id: injury_proxy_dict} for all games in the given seasons.

    Results are cached so re-running is cheap after the first fetch.
    """
    from src.data.fetch_nba_api import fetch_player_game_logs, fetch_player_stats, fetch_team_game_logs

    cache = _load_cache() if not force_refresh else {}
    seasons_done = set(cache.get("_seasons_fetched", []))
    seasons_needed = [s for s in seasons if s not in seasons_done]

    if not seasons_needed:
        if verbose:
            print(f"  Injury proxy: all {len(seasons)} seasons already cached.")
        return cache

    game_injury: dict[str, Any] = {k: v for k, v in cache.items() if not k.startswith("_")}

    for season in seasons_needed:
        if verbose:
            print(f"  Building injury proxy for {season}...")

        # --- Step 1: Season player stats → top players per team ---
        try:
            player_stats = _fetch_with_retry(fetch_player_stats, season, "Playoffs")
            time.sleep(0.6)
        except Exception as exc:
            if verbose:
                print(f"    [WARN] Could not fetch player stats for {season}: {exc}")
            continue

        if player_stats is None or player_stats.empty:
            continue

        # Map team → top-N players with their avg pts share
        team_totals = player_stats.groupby("TEAM_ABBREVIATION")["PTS"].sum().to_dict()
        top_players: dict[str, list[dict[str, Any]]] = {}

        for team_abbr, group in player_stats.groupby("TEAM_ABBREVIATION"):
            gp_col = "GP" if "GP" in group.columns else None
            min_col = "MIN" if "MIN" in group.columns else None
            pts_col = "PTS" if "PTS" in group.columns else None

            if not all([gp_col, min_col, pts_col]):
                continue

            group = group.copy()
            group["avg_min"] = group[min_col] / group[gp_col].clip(lower=1)
            group["avg_pts"] = group[pts_col] / group[gp_col].clip(lower=1)
            group["pts_share"] = group[pts_col] / max(team_totals.get(team_abbr, 1), 1)

            top = group.nlargest(TOP_PLAYERS_PER_TEAM, "avg_min")
            top_players[str(team_abbr)] = top[
                ["PLAYER_ID", "PLAYER_NAME", "avg_pts", "avg_min", "pts_share"]
            ].to_dict("records")

        # --- Step 2: Team game log → which games did each team play? ---
        try:
            team_logs = _fetch_with_retry(fetch_team_game_logs, season, "Playoffs")
            time.sleep(0.6)
        except Exception as exc:
            if verbose:
                print(f"    [WARN] Could not fetch team logs for {season}: {exc}")
            continue

        if team_logs is None or team_logs.empty:
            continue

        team_logs["GAME_ID"] = team_logs["GAME_ID"].astype(str)
        # Build {team_abbr: set of game_ids}
        team_game_ids: dict[str, set[str]] = {}
        for team_abbr, group in team_logs.groupby("TEAM_ABBREVIATION"):
            team_game_ids[str(team_abbr)] = set(group["GAME_ID"].tolist())

        # Build game → (team_a, team_b) mapping
        game_teams: dict[str, list[str]] = {}
        for gid, group in team_logs.groupby("GAME_ID"):
            game_teams[str(gid)] = group["TEAM_ABBREVIATION"].tolist()

        # --- Step 3: Player game logs → detect absences ---
        player_game_participation: dict[str, set[str]] = {}  # {player_id: {game_ids}}

        total_players = sum(len(v) for v in top_players.values())
        fetched = 0
        for team_abbr, players in top_players.items():
            for player in players:
                player_id = str(player["PLAYER_ID"])
                try:
                    log = _fetch_with_retry(
                        fetch_player_game_logs, player_id, season, "Playoffs"
                    )
                    time.sleep(0.5)
                    gid_col = next((c for c in (log.columns if log is not None else []) if c.lower() == "game_id"), None)
                    if log is not None and not log.empty and gid_col:
                        player_game_participation[player_id] = set(log[gid_col].astype(str).tolist())
                    else:
                        player_game_participation[player_id] = set()
                except Exception:
                    player_game_participation[player_id] = set()
                fetched += 1
                if verbose and fetched % 20 == 0:
                    print(f"    {fetched}/{total_players} player logs fetched...")

        # --- Step 4: For each game, compute injury strength per team ---
        for game_id, teams in game_teams.items():
            team_injury: dict[str, float] = {}
            for team_abbr in teams:
                players = top_players.get(str(team_abbr), [])
                injury_score = 0.0
                for player in players:
                    player_id = str(player["PLAYER_ID"])
                    participated = player_game_participation.get(player_id, set())
                    if game_id not in participated:
                        # Player absent — add their pts_share as injury impact
                        injury_score += float(player.get("pts_share", 0.0))
                team_injury[str(team_abbr)] = round(injury_score, 4)

            if len(teams) == 2:
                team_a_abbr, team_b_abbr = teams[0], teams[1]
                diff = team_injury.get(team_a_abbr, 0.0) - team_injury.get(team_b_abbr, 0.0)
                game_injury[game_id] = {
                    "team_a": team_a_abbr,
                    "team_b": team_b_abbr,
                    "team_a_injury_score": team_injury.get(team_a_abbr, 0.0),
                    "team_b_injury_score": team_injury.get(team_b_abbr, 0.0),
                    "injury_strength_diff": round(diff, 4),
                    "data_available": 1,
                }

        seasons_done.add(season)
        if verbose:
            print(f"    {season}: {len([g for g in game_injury if game_injury[g].get('data_available')])} games with injury data")

    result = dict(game_injury)
    result["_seasons_fetched"] = sorted(seasons_done)
    _save_cache(result)
    return result


def get_injury_diff_for_row(
    game_id: str,
    team_a_abbr: str,
    injury_cache: dict[str, Any],
) -> tuple[float, int]:
    """Return (injury_strength_diff, data_available) for a training row.

    injury_strength_diff > 0 means team_a is healthier (team_b has more injuries).
    Returns (0.0, 0) if no data.
    """
    entry = injury_cache.get(str(game_id))
    if not entry or not entry.get("data_available"):
        return 0.0, 0

    stored_team_a = str(entry.get("team_a", ""))
    diff = float(entry.get("injury_strength_diff", 0.0))

    # Flip sign if team_a in this row is team_b in the stored entry
    if stored_team_a != str(team_a_abbr):
        diff = -diff

    return diff, 1


if __name__ == "__main__":
    import sys
    seasons = sys.argv[1:] if len(sys.argv) > 1 else [
        "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
        "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    ]
    print(f"Building injury proxy for {len(seasons)} seasons...")
    result = build_injury_proxy(seasons, verbose=True)
    games_with_data = sum(1 for k, v in result.items() if not k.startswith("_") and isinstance(v, dict))
    print(f"Done. {games_with_data} games with injury data cached.")
