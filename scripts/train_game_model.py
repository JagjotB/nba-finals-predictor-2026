"""Train and save the game win-probability model.

Run once before using the predictor, then re-run after each new season.

Usage:
    python scripts/train_game_model.py
    python scripts/train_game_model.py --refresh   # force re-fetch from API
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    parser = argparse.ArgumentParser(description="Train NBA Finals game model")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch from NBA API")
    args = parser.parse_args()
    if args.refresh:
        from src.data.fetch_current_stats import (
            fetch_historical_playoff_logs,
            fetch_historical_team_ratings,
        )
        fetch_historical_playoff_logs(force_refresh=True)
        fetch_historical_team_ratings(force_refresh=True)
    from scripts.validate_models import main as validate_and_train
    validate_and_train()


if __name__ == "__main__":
    main()
