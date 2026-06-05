# nba-finals-predictor-2026

An ML-powered NBA Finals prediction engine for the 2026 Finals matchup between the New York Knicks (`NYK`) and San Antonio Spurs (`SAS`).

## Quick Start

**Requirements:** Python 3.11+

```bash
git clone https://github.com/JagjotB/nba-finals-predictor-2026.git
cd nba-finals-predictor-2026
pip install -r requirements.txt

# Train the game model (fetches 10 seasons of playoff data from NBA API)
python scripts/train_game_model.py

# Run the dashboard
streamlit run src/app/streamlit_app.py
```

After each Finals game, import the box score and update predictions:

```bash
python -m src.finals_data.update_after_finals_game --game 1
python -m src.models.update_after_game --game 1
```

## What The Project Does

The goal is not just to predict the series winner. The project builds a full basketball explanation layer:

- player projections
- rotation and minutes context
- team play-style profiles
- matchup edges
- lineup strength
- clutch and closing lineup edge
- foul trouble risk
- game-by-game win probabilities
- best-of-7 Monte Carlo series simulation
- scenario analysis
- post-game model updates
- backtesting and calibration
- Streamlit dashboard

## Why Games First, Then Series

The model predicts each Finals game first, then simulates the series.

That matters because a Finals series is not one prediction. Home court changes by game, rest changes by game, foul trouble and injuries can affect individual games, and a team can win the series through many different paths. The series simulator takes game-level probabilities and runs repeated best-of-7 simulations to estimate:

- series win probability
- most likely result
- probability of each result, such as `NYK in 6` or `SAS in 7`
- series length distribution

## Why Player Projections Matter

Bad minutes projections create bad player projections, and bad player projections create bad team predictions.

The player engine estimates:

- minutes
- points
- rebounds
- assists
- turnovers
- steals
- blocks
- threes
- free throw attempts
- usage
- true shooting proxy
- foul risk

This lets the model reason about the players who actually decide playoff games rather than relying only on team-level season averages.

## Why Matchup Analysis Matters

Playoff series are matchup-specific. A team can be excellent overall and still struggle against a particular defensive scheme, rim protector, rebounding profile, or transition team.

The matchup engine compares each team's offensive style against the opponent's defensive style, including:

- rim pressure vs rim protection
- pick-and-roll offense vs screen defense
- isolation creation vs perimeter defense
- transition offense vs transition defense
- corner threes vs corner three prevention
- offensive rebounding vs defensive rebounding
- free throw pressure vs foul discipline

This is the main layer that makes the model feel like it understands basketball, not just spreadsheets.

## Why Lineup Analysis Matters

NBA playoff outcomes are often decided by short lineup stretches:

- starting lineup minutes
- closing lineup minutes
- bench minutes
- small-ball groups
- big lineups
- non-star minutes
- staggered star minutes

The lineup engine estimates offensive rating, defensive rating, net rating, spacing, rebounding, switchability, and foul risk for each lineup. It also uses shrinkage so low-minute lineup samples do not overwhelm the model.

## Project Structure

```text
config/
  settings.yaml
  model_weights.yaml

data/
  manual/
    finals_schedule.csv
    injuries.csv
    rotations.csv
    player_matchups.csv
    coaching_notes.csv

src/
  data/
  features/
  models/
  evaluation/
  app/
```

## Manual Data Updates

Some Finals context should be manually curated because it changes quickly and is not always available cleanly through APIs.

### Update Injuries

Edit:

```text
data/manual/injuries.csv
```

Important columns:

- `date`
- `team`
- `player`
- `status`
- `injury`
- `notes`
- `expected_minutes_adjustment`

Use `expected_minutes_adjustment` to tell the model how availability affects rotation minutes. For example, `-8` means the player is expected to lose about eight minutes.

### Update Rotations

Edit:

```text
data/manual/rotations.csv
```

Important columns:

- `team`
- `player`
- `role`
- `projected_minutes`
- `minutes_floor`
- `minutes_ceiling`
- `is_starter`
- `is_closer`
- `rotation_confidence`
- `notes`

Use realistic floors and ceilings. Wide ranges tell the uncertainty and scenario layers that the player's role is unstable.

### Update Schedule

Edit:

```text
data/manual/finals_schedule.csv
```

This controls game number, date, home team, away team, and neutral-site status.

### Update Matchups

Edit:

```text
data/manual/player_matchups.csv
```

Use this for manually curated defensive assignments and matchup notes. These feed player projection and matchup explanation layers.

## Data Collection

Primary source:

```text
src/data/fetch_nba_api.py
```

The NBA API helpers support:

- team stats
- player stats
- team game logs
- player game logs
- lineup stats
- advanced box scores

Example:

```bash
python -c "from src.data.fetch_nba_api import fetch_team_stats; print(fetch_team_stats('2025-26', 'Playoffs').head())"
```

Optional Basketball-Reference helpers:

```text
src/data/fetch_basketball_reference.py
```

These are optional backups for historical playoff tables. If Basketball-Reference fetching fails, the project is designed to keep running.

## Build Finals Context

The Finals context loader combines config and manual data into the object used by the rest of the project.

```bash
python -m src.data.build_dataset
```

It includes:

- series name
- teams
- schedule
- active players
- rotations
- injuries
- starters
- closers
- uncertain minutes

## Train Models

Build the canonical historical table, run walk-forward validation, compare
against Elo and net-rating baselines, then fit the production game/meta models:

```bash
python scripts/validate_models.py
```

Inspect current player projections:

```bash
python -m src.models.train_player_model
```

Inspect current lineup projections:

```bash
python -m src.features.lineup_features
```

## Run Predictions

Game predictions:

```bash
python -m src.models.predict_game
```

Series simulation:

```bash
python -m src.models.simulate_series
```

Scenario simulation:

```bash
python -m src.models.scenario_simulator
```

Post-game update engine:

```bash
python -m src.models.update_after_game
```

## Run Dashboard

Start the Streamlit app:

```bash
streamlit run src/app/streamlit_app.py
```

The dashboard includes:

- overview
- game predictions
- player projections
- matchup edges
- team play styles
- lineup analysis
- clutch edge
- scenario simulator
- series simulation charts
- explanation report

## Run Backtests

Backtesting and calibration live in:

```text
src/evaluation/
```

Build the canonical pregame dataset, run chronological walk-forward validation,
fit the learned meta-model, and save production artifacts:

```bash
python scripts/validate_models.py
```

To print the real walk-forward comparison report without retraining artifacts:

```bash
python -m src.evaluation.backtest
```

The default split pattern is:

- train 2014-2021, test 2022 playoffs
- train 2014-2022, test 2023 playoffs
- train 2014-2023, test 2024 playoffs
- train 2014-2024, test 2025 playoffs

Metrics include:

- accuracy
- log loss
- Brier score
- ROC AUC
- game winner accuracy
- series winner accuracy
- upset detection
- confidence bucket performance
- calibration tables

The key calibration question is:

> When the model says 60%, does that team win close to 60% historically?

That is how the project checks whether it is actually calibrated rather than confidently wrong.

## Core Commands

```bash
python -m src.data.build_dataset
python scripts/validate_models.py
python -m src.models.train_player_model
python -m src.features.lineup_features
python -m src.models.predict_game
python -m src.models.simulate_series
streamlit run src/app/streamlit_app.py
```

Production model training:

```bash
python scripts/validate_models.py
```

Save an immutable pregame snapshot before a Finals game:

```bash
python -m src.models.prediction_snapshot --game 2
```

Current lineup-analysis equivalent:

```bash
python -m src.features.lineup_features
```

Additional useful commands:

```bash
python -m src.models.scenario_simulator
python -m src.models.update_after_game
python -m src.evaluation.backtest
python -m src.evaluation.calibration
python -m src.evaluation.metrics
```

## Limitations

- Manual rotations and injuries must be updated before every game; the current files include Game 1-informed assumptions.
- NBA API availability can be inconsistent because stats.nba.com sometimes rate limits or blocks requests.
- Basketball-Reference is optional and can fail because page structure or access rules can change.
- Player and lineup projections are only as good as the rotation and injury inputs.
- Lineup samples are noisy, so the lineup engine uses shrinkage instead of trusting raw net ratings.
- Scenario outputs are directional what-if estimates, not guarantees.
- Walk-forward validation uses 915 real playoff games from 2014-15 through 2024-25.
- Historical injury, expected-rotation, and lineup snapshots are unavailable in the current archive. Those components are therefore labeled as assumptions and receive zero learned meta-model coefficient until genuine pregame histories are added.
- The validated net-rating baseline currently outperforms the larger structural model, so production stacking favors the simpler model.
- The dashboard is an explanation and decision-support tool, not a betting system.
