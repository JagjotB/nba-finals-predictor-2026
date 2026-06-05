"""Load manual Finals context CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUAL_DIR = PROJECT_ROOT / "data" / "manual"

MANUAL_FILES = {
    "finals_schedule": "finals_schedule.csv",
    "rotations": "rotations.csv",
    "injuries": "injuries.csv",
    "player_matchups": "player_matchups.csv",
    "coaching_notes": "coaching_notes.csv",
    "series_context": "series_context.csv",
}


def _manual_dir(manual_dir: str | Path | None = None) -> Path:
    return Path(manual_dir) if manual_dir is not None else DEFAULT_MANUAL_DIR


def _read_manual_csv(
    dataset_name: str,
    manual_dir: str | Path | None = None,
    parse_dates: list[str] | None = None,
) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to load manual CSV data.") from exc

    if dataset_name not in MANUAL_FILES:
        raise ValueError(f"Unknown manual dataset: {dataset_name}")

    path = _manual_dir(manual_dir) / MANUAL_FILES[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing manual data file: {path}")

    return pd.read_csv(path, parse_dates=parse_dates)


def _coerce_bool_columns(frame: Any, columns: list[str]) -> Any:
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}

    def parse_bool(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in true_values:
            return True
        if normalized in false_values:
            return False
        return value

    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].map(parse_bool)
    return frame


def load_finals_schedule(manual_dir: str | Path | None = None) -> Any:
    """Load the manual Finals schedule."""
    frame = _read_manual_csv("finals_schedule", manual_dir, parse_dates=["date"])
    return _coerce_bool_columns(frame, ["neutral_site"])


def load_rotations(manual_dir: str | Path | None = None) -> Any:
    """Load projected rotations and minute ranges."""
    frame = _read_manual_csv("rotations", manual_dir)
    return _coerce_bool_columns(frame, ["is_starter", "is_closer"])


def load_injuries(manual_dir: str | Path | None = None) -> Any:
    """Load manual injury statuses and expected minute adjustments."""
    return _read_manual_csv("injuries", manual_dir, parse_dates=["date"])


def load_player_matchups(manual_dir: str | Path | None = None) -> Any:
    """Load manually curated player matchup notes."""
    return _read_manual_csv("player_matchups", manual_dir)


def load_coaching_notes(manual_dir: str | Path | None = None) -> Any:
    """Load game-by-game coaching adjustments and notes."""
    return _read_manual_csv("coaching_notes", manual_dir)


def load_series_context(manual_dir: str | Path | None = None) -> Any:
    """Load pre-series context: last game dates, rest days per team."""
    try:
        return _read_manual_csv("series_context", manual_dir)
    except FileNotFoundError:
        import pandas as pd
        return pd.DataFrame()


def load_all_manual_data(manual_dir: str | Path | None = None) -> dict[str, Any]:
    """Load every manual Finals context table."""
    return {
        "finals_schedule": load_finals_schedule(manual_dir),
        "rotations": load_rotations(manual_dir),
        "injuries": load_injuries(manual_dir),
        "player_matchups": load_player_matchups(manual_dir),
        "coaching_notes": load_coaching_notes(manual_dir),
        "series_context": load_series_context(manual_dir),
    }
