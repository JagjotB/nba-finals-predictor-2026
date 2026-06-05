"""Persist immutable pregame prediction snapshots and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.build_dataset import build_finals_context
from src.models.predict_game import predict_finals_games


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = PROJECT_ROOT / "outputs" / "predictions" / "snapshots"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_provenance() -> dict[str, Any]:
    historical_metadata = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "historical"
        / "canonical_pregame_games.metadata.json"
    )
    metadata = (
        json.loads(historical_metadata.read_text(encoding="utf-8"))
        if historical_metadata.exists() else {}
    )
    return {
        "model_version": "calibrated-finals-v3",
        "game_model_sha256": _sha256(
            PROJECT_ROOT / "data" / "processed" / "stats_cache" / "game_model.joblib"
        ),
        "meta_model_sha256": _sha256(
            PROJECT_ROOT / "data" / "processed" / "stats_cache" / "meta_model.joblib"
        ),
        "canonical_dataset_sha256": metadata.get("sha256"),
        "canonical_schema_version": metadata.get("schema_version"),
        "validation_report": str(
            PROJECT_ROOT / "outputs" / "reports" / "walk_forward_backtest.json"
        ),
    }


def create_prediction_snapshot(
    game_number: int | None = None,
    context: dict[str, Any] | None = None,
    predictions: list[dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    context = context or build_finals_context()
    predictions = predictions or predict_finals_games(context)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    selected = (
        [row for row in predictions if int(row["game_number"]) == game_number]
        if game_number is not None else predictions
    )
    payload = {
        "generated_at_utc": generated_at,
        "game_number": game_number,
        "series": context.get("series"),
        "teams": [context.get("team_a"), context.get("team_b")],
        "provenance": model_provenance(),
        "manual_assumptions": {
            "rotations": context.get("rotations"),
            "injuries": context.get("injuries"),
            "player_matchups": context.get("player_matchups"),
            "coaching_notes": context.get("coaching_notes"),
        },
        "predictions": predictions,
        "focus_game_predictions": selected,
    }
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"game_{game_number}" if game_number is not None else "series"
    path = SNAPSHOT_DIR / f"{timestamp}_{suffix}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path, payload


def load_latest_prediction_snapshot(game_number: int) -> dict[str, Any] | None:
    candidates = sorted(SNAPSHOT_DIR.glob(f"*_game_{game_number}.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save an immutable pregame prediction snapshot")
    parser.add_argument("--game", type=int, default=None)
    args = parser.parse_args()
    output, _ = create_prediction_snapshot(args.game)
    print(output)
