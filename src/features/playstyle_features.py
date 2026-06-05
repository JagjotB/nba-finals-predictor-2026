"""Team offensive and defensive play-style features."""

from __future__ import annotations

from math import isnan
from typing import Any


OFFENSIVE_METRICS = [
    "pace",
    "half_court_reliance",
    "transition_frequency",
    "pick_and_roll_usage",
    "isolation_usage",
    "post_up_usage",
    "rim_pressure",
    "midrange_frequency",
    "corner_3_frequency",
    "above_the_break_3_frequency",
    "offensive_rebounding",
    "free_throw_pressure",
    "turnover_risk",
]

DEFENSIVE_METRICS = [
    "rim_protection",
    "point_of_attack_defense",
    "switchability",
    "drop_coverage_strength",
    "corner_3_prevention",
    "defensive_rebounding",
    "forced_turnovers",
    "foul_discipline",
    "transition_defense",
]

TEAM_KEYS = ["team", "TEAM", "TEAM_ABBREVIATION", "TEAM_NAME", "opponent"]

ALIASES = {
    "pace": ["pace", "PACE"],
    "half_court_reliance": ["half_court_reliance", "halfcourt_reliance", "HALF_COURT_RELIANCE"],
    "transition_frequency": [
        "transition_frequency",
        "transition_freq",
        "fast_break_frequency",
        "PCT_PTS_FB",
        "FB_PTS",
        "fast_break_points",
    ],
    "pick_and_roll_usage": ["pick_and_roll_usage", "pnr_usage", "PICK_ROLL_USAGE"],
    "isolation_usage": ["isolation_usage", "iso_usage", "ISO_USAGE"],
    "post_up_usage": ["post_up_usage", "postup_usage", "POST_UP_USAGE"],
    "rim_pressure": ["rim_pressure", "rim_frequency", "restricted_area_frequency", "PCT_FGA_RA"],
    "midrange_frequency": ["midrange_frequency", "mid_range_frequency", "PCT_FGA_MID"],
    "corner_3_frequency": ["corner_3_frequency", "corner_three_frequency", "PCT_FGA_CORNER3"],
    "above_the_break_3_frequency": [
        "above_the_break_3_frequency",
        "above_break_3_frequency",
        "PCT_FGA_ABOVE_BREAK3",
    ],
    "offensive_rebounding": ["offensive_rebounding", "OREB_PCT", "OREB%", "oreb_pct"],
    "free_throw_pressure": ["free_throw_pressure", "FTA_RATE", "FTR", "fta_rate"],
    "turnover_risk": ["turnover_risk", "TOV_PCT", "TOV%", "tov_pct", "TM_TOV_PCT"],
    "rim_protection": ["rim_protection", "rim_protection_score", "rim_protection_proxy"],
    "point_of_attack_defense": [
        "point_of_attack_defense",
        "poa_defense",
        "drive_defense",
    ],
    "switchability": ["switchability", "switchability_score", "switch_rate"],
    "drop_coverage_strength": ["drop_coverage_strength", "drop_strength"],
    "corner_3_prevention": ["corner_3_prevention", "corner_three_prevention"],
    "defensive_rebounding": ["defensive_rebounding", "DREB_PCT", "DREB%", "dreb_pct"],
    "forced_turnovers": ["forced_turnovers", "OPP_TOV_PCT", "opp_tov_pct", "STL_PCT", "opp_tov_pct_proxy"],
    "foul_discipline": ["foul_discipline", "OPP_FTA_RATE", "opp_fta_rate", "PF", "opp_fta_rate_proxy"],
    "transition_defense": [
        "transition_defense",
        "opp_fast_break_frequency",
        "OPP_PCT_PTS_FB",
        "OPP_FB_PTS",
    ],
    "points_in_paint": ["points_in_paint", "PITP", "PTS_PAINT"],
    "drives": ["drives", "DRIVES"],
    "opp_rim_fg_pct": ["opp_rim_fg_pct", "OPP_RA_FG_PCT", "OPP_RESTRICTED_AREA_FG_PCT"],
    "opp_drive_fg_pct": ["opp_drive_fg_pct", "OPP_DRIVE_FG_PCT"],
    "opp_corner_3_frequency": ["opp_corner_3_frequency", "OPP_CORNER3_FREQ"],
    "block_rate": ["block_rate", "BLK_PCT", "block_pct"],
    "steal_rate": ["steal_rate", "STL_PCT", "steal_pct"],
    "defensive_rating": ["defensive_rating", "DEF_RATING"],
}

METRIC_SPECS = {
    "pace": (94.0, 103.0, False),
    "transition_frequency": (7.0, 18.0, False),
    "pick_and_roll_usage": (16.0, 30.0, False),
    "isolation_usage": (4.0, 13.0, False),
    "post_up_usage": (1.0, 8.0, False),
    "midrange_frequency": (7.0, 20.0, False),
    "corner_3_frequency": (5.0, 15.0, False),
    "above_the_break_3_frequency": (20.0, 38.0, False),
    "offensive_rebounding": (20.0, 34.0, False),
    "free_throw_pressure": (18.0, 34.0, False),
    "turnover_risk": (10.0, 17.0, False),
    "rim_protection": (0.0, 100.0, False),
    "point_of_attack_defense": (0.0, 100.0, False),
    "switchability": (0.0, 100.0, False),
    "drop_coverage_strength": (0.0, 100.0, False),
    "corner_3_prevention": (0.0, 100.0, False),
    "defensive_rebounding": (68.0, 78.0, False),
    "forced_turnovers": (11.0, 17.0, False),
    "foul_discipline": (0.0, 100.0, False),
    "transition_defense": (0.0, 100.0, False),
}

OFFENSE_LABELS = {
    "pace": ("slower pace", "average pace", "faster pace"),
    "half_court_reliance": (
        "lower half-court reliance",
        "balanced half-court reliance",
        "high half-court reliance",
    ),
    "transition_frequency": (
        "low transition frequency",
        "moderate transition frequency",
        "high transition frequency",
    ),
    "pick_and_roll_usage": (
        "light pick-and-roll usage",
        "moderate pick-and-roll usage",
        "heavy pick-and-roll usage",
    ),
    "isolation_usage": (
        "low isolation creation",
        "moderate isolation creation",
        "strong isolation creation",
    ),
    "post_up_usage": ("low post-up volume", "moderate post-up volume", "high post-up volume"),
    "rim_pressure": ("low rim pressure", "moderate rim pressure", "strong rim pressure"),
    "midrange_frequency": (
        "low midrange volume",
        "moderate midrange volume",
        "high midrange volume",
    ),
    "corner_3_frequency": (
        "low corner three volume",
        "moderate corner three volume",
        "high corner three volume",
    ),
    "above_the_break_3_frequency": (
        "low above-the-break three volume",
        "moderate above-the-break three volume",
        "high above-the-break three volume",
    ),
    "offensive_rebounding": (
        "weaker offensive rebounding",
        "moderate offensive rebounding",
        "strong offensive rebounding",
    ),
    "free_throw_pressure": (
        "low free throw pressure",
        "moderate free throw pressure",
        "high free throw pressure",
    ),
    "turnover_risk": ("low turnover risk", "moderate turnover risk", "high turnover risk"),
}

DEFENSE_LABELS = {
    "rim_protection": ("weaker rim protection", "solid rim protection", "elite rim protection"),
    "point_of_attack_defense": (
        "weaker point-of-attack defense",
        "solid point-of-attack defense",
        "strong point-of-attack defense",
    ),
    "switchability": ("limited switchability", "moderate switchability", "high switchability"),
    "drop_coverage_strength": (
        "vulnerable drop coverage",
        "solid drop coverage",
        "strong drop coverage",
    ),
    "corner_3_prevention": (
        "weaker corner three prevention",
        "solid corner three prevention",
        "strong corner three prevention",
    ),
    "defensive_rebounding": (
        "weaker defensive rebounding",
        "moderate defensive rebounding",
        "strong defensive rebounding",
    ),
    "forced_turnovers": (
        "low forced turnovers",
        "moderate forced turnovers",
        "high forced turnovers",
    ),
    "foul_discipline": ("high foul risk", "solid foul discipline", "strong foul discipline"),
    "transition_defense": (
        "weaker transition defense",
        "solid transition defense",
        "strong transition defense",
    ),
}

DEFAULT_OFFENSE = {
    "pace": 96.5,
    "transition_frequency": 12.0,
    "pick_and_roll_usage": 22.0,
    "isolation_usage": 8.0,
    "post_up_usage": 4.0,
    "midrange_frequency": 12.0,
    "corner_3_frequency": 9.0,
    "above_the_break_3_frequency": 29.0,
    "offensive_rebounding": 27.0,
    "free_throw_pressure": 25.0,
    "turnover_risk": 13.5,
}

DEFAULT_DEFENSE = {
    "rim_protection": 55.0,
    "point_of_attack_defense": 50.0,
    "switchability": 50.0,
    "drop_coverage_strength": 52.0,
    "corner_3_prevention": 50.0,
    "defensive_rebounding": 73.0,
    "forced_turnovers": 13.5,
    "foul_discipline": 50.0,
    "transition_defense": 50.0,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if isnan(number):
        return default
    return number


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score(value: float, low: float, high: float, invert: bool = False) -> float:
    if high == low:
        return 50.0
    scaled = (value - low) / (high - low) * 100.0
    if invert:
        scaled = 100.0 - scaled
    return round(_clip(scaled, 0.0, 100.0), 1)


def _percentage_value(value: Any, default: float) -> float:
    number = _as_float(value, default)
    if 0.0 <= number <= 1.5:
        return number * 100.0
    return number


def _first_value(row: dict[str, Any] | None, keys: list[str]) -> Any:
    if not row:
        return None
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _metric_value(row: dict[str, Any] | None, metric: str, default: float) -> float:
    return _percentage_value(_first_value(row, ALIASES[metric]), default)


def _iter_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return [dict(row) for row in data.to_dict(orient="records")]
    if isinstance(data, dict):
        if any(key in data for key in TEAM_KEYS):
            return [dict(data)]
        records = []
        for key, value in data.items():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        record = dict(row)
                        record.setdefault("team", key)
                        records.append(record)
            elif isinstance(value, dict):
                record = dict(value)
                record.setdefault("team", key)
                records.append(record)
        return records
    return [dict(row) for row in data]


def _team_name(row: dict[str, Any]) -> str:
    return str(_first_value(row, TEAM_KEYS) or "").strip()


def _team_index(data: Any) -> dict[str, dict[str, Any]]:
    index = {}
    for row in _iter_records(data):
        team = _team_name(row)
        if team:
            index[team] = row
    return index


def _team_overrides(manual_overrides: Any) -> dict[str, dict[str, Any]]:
    return _team_index(manual_overrides)


def _label(score: float, labels: tuple[str, str, str]) -> str:
    if score < 35.0:
        return labels[0]
    if score > 65.0:
        return labels[2]
    return labels[1]


def _metric_record(
    value: float,
    score: float,
    label: str,
    source: str,
) -> dict[str, Any]:
    return {
        "value": round(value, 3),
        "score": round(score, 1),
        "label": label,
        "source": source,
    }


def _direct_or_composite_score(
    row: dict[str, Any] | None,
    metric: str,
    default: float,
) -> tuple[float, str]:
    raw_value = _first_value(row, ALIASES[metric])
    if raw_value is not None:
        value = _percentage_value(raw_value, default)
        low, high, invert = METRIC_SPECS[metric]
        return _score(value, low, high, invert), "direct"
    return default, "default"


def _rim_pressure_score(row: dict[str, Any] | None) -> tuple[float, float, str]:
    raw_value = _first_value(row, ALIASES["rim_pressure"])
    if raw_value is not None:
        value = _percentage_value(raw_value, 34.0)
        return value, _score(value, 26.0, 42.0), "direct"

    points_in_paint = _metric_value(row, "points_in_paint", 48.0)
    free_throw_pressure = _metric_value(row, "free_throw_pressure", 25.0)
    paint_score = _score(points_in_paint, 38.0, 58.0)
    free_throw_score = _score(free_throw_pressure, 18.0, 34.0)
    composite = round((paint_score + free_throw_score) / 2.0, 1)
    return composite, composite, "composite"


def _rim_protection_score(row: dict[str, Any] | None) -> tuple[float, str]:
    direct, source = _direct_or_composite_score(row, "rim_protection", DEFAULT_DEFENSE["rim_protection"])
    if source == "direct":
        return direct, source

    opp_rim_fg = _metric_value(row, "opp_rim_fg_pct", 66.0)
    block_rate = _metric_value(row, "block_rate", 4.8)
    rim_fg_score = _score(opp_rim_fg, 72.0, 58.0)
    block_score = _score(block_rate, 3.0, 7.5)
    return round((rim_fg_score * 0.65) + (block_score * 0.35), 1), "composite"


def _point_of_attack_score(row: dict[str, Any] | None) -> tuple[float, str]:
    direct, source = _direct_or_composite_score(
        row,
        "point_of_attack_defense",
        DEFAULT_DEFENSE["point_of_attack_defense"],
    )
    if source == "direct":
        return direct, source

    opp_drive_fg = _metric_value(row, "opp_drive_fg_pct", 48.0)
    steal_rate = _metric_value(row, "steal_rate", 7.5)
    drive_score = _score(opp_drive_fg, 54.0, 42.0)
    steal_score = _score(steal_rate, 5.5, 9.5)
    return round((drive_score * 0.70) + (steal_score * 0.30), 1), "composite"


def _corner_3_prevention_score(row: dict[str, Any] | None) -> tuple[float, str]:
    direct, source = _direct_or_composite_score(
        row,
        "corner_3_prevention",
        DEFAULT_DEFENSE["corner_3_prevention"],
    )
    if source == "direct":
        return direct, source

    opp_corner_freq = _metric_value(row, "opp_corner_3_frequency", 9.0)
    return _score(opp_corner_freq, 15.0, 5.0), "composite"


def _foul_discipline_score(row: dict[str, Any] | None) -> tuple[float, str]:
    raw_value = _first_value(row, ALIASES["foul_discipline"])
    if raw_value is not None:
        value = _percentage_value(raw_value, 25.0)
        if value <= 34.0:
            return _score(value, 34.0, 18.0), "direct"
        return _score(value, 24.0, 16.0), "direct"
    return DEFAULT_DEFENSE["foul_discipline"], "default"


def _transition_defense_score(row: dict[str, Any] | None) -> tuple[float, str]:
    raw_value = _first_value(row, ALIASES["transition_defense"])
    if raw_value is not None:
        value = _percentage_value(raw_value, 12.0)
        return _score(value, 18.0, 7.0), "direct"
    return DEFAULT_DEFENSE["transition_defense"], "default"


def _apply_nested_override(
    profile: dict[str, dict[str, Any]],
    overrides: dict[str, Any] | None,
    side: str,
    labels: dict[str, tuple[str, str, str]],
) -> None:
    if not overrides:
        return

    side_overrides = overrides.get(side, overrides)
    if not isinstance(side_overrides, dict):
        return

    for metric, value in side_overrides.items():
        if metric not in profile:
            continue
        if isinstance(value, dict):
            override_value = value.get("value", profile[metric]["value"])
            override_score = value.get("score")
        else:
            override_value = value
            override_score = None

        if override_score is None:
            if metric in METRIC_SPECS:
                low, high, invert = METRIC_SPECS[metric]
                override_score = _score(_percentage_value(override_value, profile[metric]["value"]), low, high, invert)
            else:
                override_score = _percentage_value(override_value, profile[metric]["score"])

        profile[metric] = _metric_record(
            _percentage_value(override_value, profile[metric]["value"]),
            _as_float(override_score, profile[metric]["score"]),
            _label(_as_float(override_score, profile[metric]["score"]), labels[metric]),
            "override",
        )


def calculate_offensive_profile(
    team: str,
    team_stats: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate offensive play-style metrics for one team."""
    metrics: dict[str, dict[str, Any]] = {}

    pace = _metric_value(team_stats, "pace", DEFAULT_OFFENSE["pace"])
    pace_score = _score(pace, *METRIC_SPECS["pace"])
    metrics["pace"] = _metric_record(pace, pace_score, _label(pace_score, OFFENSE_LABELS["pace"]), "direct")

    transition = _metric_value(team_stats, "transition_frequency", DEFAULT_OFFENSE["transition_frequency"])
    transition_score = _score(transition, *METRIC_SPECS["transition_frequency"])
    metrics["transition_frequency"] = _metric_record(
        transition,
        transition_score,
        _label(transition_score, OFFENSE_LABELS["transition_frequency"]),
        "direct",
    )

    half_court_raw = _first_value(team_stats, ALIASES["half_court_reliance"])
    if half_court_raw is not None:
        half_court = _percentage_value(half_court_raw, 100.0 - transition_score)
        half_court_score = _score(half_court, 45.0, 75.0)
        source = "direct"
    else:
        half_court_score = round(100.0 - transition_score, 1)
        half_court = half_court_score
        source = "derived"
    metrics["half_court_reliance"] = _metric_record(
        half_court,
        half_court_score,
        _label(half_court_score, OFFENSE_LABELS["half_court_reliance"]),
        source,
    )

    for metric in [
        "pick_and_roll_usage",
        "isolation_usage",
        "post_up_usage",
        "midrange_frequency",
        "corner_3_frequency",
        "above_the_break_3_frequency",
        "offensive_rebounding",
        "free_throw_pressure",
        "turnover_risk",
    ]:
        value = _metric_value(team_stats, metric, DEFAULT_OFFENSE[metric])
        metric_score = _score(value, *METRIC_SPECS[metric])
        metrics[metric] = _metric_record(
            value,
            metric_score,
            _label(metric_score, OFFENSE_LABELS[metric]),
            "direct",
        )

    rim_value, rim_score, rim_source = _rim_pressure_score(team_stats)
    metrics["rim_pressure"] = _metric_record(
        rim_value,
        rim_score,
        _label(rim_score, OFFENSE_LABELS["rim_pressure"]),
        rim_source,
    )

    _apply_nested_override(metrics, overrides, "offense", OFFENSE_LABELS)
    return {
        "team": team,
        "metrics": metrics,
        "summary": describe_playstyle_metrics(metrics, OFFENSE_LABELS),
    }


def calculate_defensive_profile(
    team: str,
    team_stats: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate defensive play-style metrics for one team."""
    metrics: dict[str, dict[str, Any]] = {}

    rim_score, rim_source = _rim_protection_score(team_stats)
    metrics["rim_protection"] = _metric_record(
        rim_score,
        rim_score,
        _label(rim_score, DEFENSE_LABELS["rim_protection"]),
        rim_source,
    )

    poa_score, poa_source = _point_of_attack_score(team_stats)
    metrics["point_of_attack_defense"] = _metric_record(
        poa_score,
        poa_score,
        _label(poa_score, DEFENSE_LABELS["point_of_attack_defense"]),
        poa_source,
    )

    for metric in ["switchability", "defensive_rebounding", "forced_turnovers"]:
        value = _metric_value(team_stats, metric, DEFAULT_DEFENSE[metric])
        metric_score = _score(value, *METRIC_SPECS[metric])
        metrics[metric] = _metric_record(
            value,
            metric_score,
            _label(metric_score, DEFENSE_LABELS[metric]),
            "direct",
        )

    drop_direct, drop_source = _direct_or_composite_score(
        team_stats,
        "drop_coverage_strength",
        DEFAULT_DEFENSE["drop_coverage_strength"],
    )
    if drop_source == "default":
        drop_score = round((metrics["rim_protection"]["score"] * 0.70) + (metrics["defensive_rebounding"]["score"] * 0.30), 1)
        drop_source = "composite"
    else:
        drop_score = drop_direct
    metrics["drop_coverage_strength"] = _metric_record(
        drop_score,
        drop_score,
        _label(drop_score, DEFENSE_LABELS["drop_coverage_strength"]),
        drop_source,
    )

    corner_score, corner_source = _corner_3_prevention_score(team_stats)
    metrics["corner_3_prevention"] = _metric_record(
        corner_score,
        corner_score,
        _label(corner_score, DEFENSE_LABELS["corner_3_prevention"]),
        corner_source,
    )

    foul_score, foul_source = _foul_discipline_score(team_stats)
    metrics["foul_discipline"] = _metric_record(
        foul_score,
        foul_score,
        _label(foul_score, DEFENSE_LABELS["foul_discipline"]),
        foul_source,
    )

    transition_score, transition_source = _transition_defense_score(team_stats)
    metrics["transition_defense"] = _metric_record(
        transition_score,
        transition_score,
        _label(transition_score, DEFENSE_LABELS["transition_defense"]),
        transition_source,
    )

    _apply_nested_override(metrics, overrides, "defense", DEFENSE_LABELS)
    return {
        "team": team,
        "metrics": metrics,
        "summary": describe_playstyle_metrics(metrics, DEFENSE_LABELS),
    }


def describe_playstyle_metrics(
    metrics: dict[str, dict[str, Any]],
    labels: dict[str, tuple[str, str, str]],
    max_traits: int = 5,
) -> list[str]:
    """Convert metric scores into a compact play-style summary."""
    strong_traits = [
        (name, record["score"], record["label"])
        for name, record in metrics.items()
        if record["score"] >= 65.0
    ]
    weak_traits = [
        (name, 100.0 - record["score"], record["label"])
        for name, record in metrics.items()
        if record["score"] <= 35.0
    ]

    ranked = sorted(strong_traits + weak_traits, key=lambda item: item[1], reverse=True)
    if not ranked:
        ranked = sorted(
            [(name, abs(record["score"] - 50.0), record["label"]) for name, record in metrics.items()],
            key=lambda item: item[1],
            reverse=True,
        )

    return [label for _, _, label in ranked[:max_traits]]


def build_team_playstyle_profile(
    team: str,
    offensive_stats: dict[str, Any] | None = None,
    defensive_stats: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build offensive and defensive style profiles for one team."""
    return {
        "team": team,
        "offense": calculate_offensive_profile(team, offensive_stats, overrides),
        "defense": calculate_defensive_profile(team, defensive_stats, overrides),
    }


def _enrich_team_stats(raw: dict[str, Any]) -> dict[str, Any]:
    """Derive playstyle aliases from NBA API columns not in ALIASES directly."""
    if not raw:
        return raw
    enriched = dict(raw)

    # OPP_EFG_PCT → rim_protection_proxy (inverted: lower opp EFG = better)
    opp_efg = raw.get("OPP_EFG_PCT")
    if opp_efg is not None:
        # Scale: 0.44 (elite) → 100, 0.58 (poor) → 0
        enriched["rim_protection_proxy"] = round(
            max(0.0, min(100.0, (0.58 - float(opp_efg)) / 0.14 * 100.0)), 1
        )

    # OPP_TOV_PCT → forced_turnovers proxy (convert decimal to %)
    opp_tov = raw.get("OPP_TOV_PCT")
    if opp_tov is not None and "opp_tov_pct_proxy" not in raw:
        enriched["opp_tov_pct_proxy"] = float(opp_tov)

    # OPP_FTA_RATE → foul_discipline proxy
    opp_ftar = raw.get("OPP_FTA_RATE")
    if opp_ftar is not None and "opp_fta_rate_proxy" not in raw:
        # Lower OPP_FTA_RATE = better foul discipline; convert to 0-100 score
        enriched["opp_fta_rate_proxy"] = round(
            max(0.0, min(100.0, (0.35 - float(opp_ftar)) / 0.20 * 100.0)), 1
        )

    # PCT_FGA_3PT → above_the_break_3_frequency proxy (~70% of 3s are above break)
    pct_3 = raw.get("PCT_FGA_3PT")
    if pct_3 is not None and "above_the_break_3_frequency" not in raw:
        enriched["above_the_break_3_frequency"] = round(float(pct_3) * 100.0 * 0.70, 1)

    # PCT_PTS_2PT_MR → midrange_frequency proxy
    pct_mr = raw.get("PCT_PTS_2PT_MR")
    if pct_mr is not None and "midrange_frequency" not in raw:
        enriched["midrange_frequency"] = round(float(pct_mr) * 100.0, 1)

    return enriched


def build_playstyle_profiles(
    finals_context: dict[str, Any],
    team_stats: Any = None,
    offensive_stats: Any = None,
    defensive_stats: Any = None,
    manual_overrides: Any = None,
) -> dict[str, dict[str, Any]]:
    """Build play-style profiles for both Finals teams."""
    teams = [
        str(finals_context.get("team_a") or "").strip(),
        str(finals_context.get("team_b") or "").strip(),
    ]
    teams = [team for team in teams if team]

    # Prefer live stats from context over caller-supplied team_stats
    ctx_team_stats = finals_context.get("team_stats") or {}
    if ctx_team_stats and not team_stats:
        # Enrich each team's stats dict with derived aliases
        enriched = {t: _enrich_team_stats(ctx_team_stats.get(t, {})) for t in teams}
        team_stats = enriched

    combined_index = _team_index(team_stats)
    offense_index = _team_index(offensive_stats) or combined_index
    defense_index = _team_index(defensive_stats) or combined_index
    overrides_index = _team_overrides(manual_overrides)

    return {
        team: build_team_playstyle_profile(
            team,
            offense_index.get(team),
            defense_index.get(team),
            overrides_index.get(team),
        )
        for team in teams
    }


def summarize_playstyle_profiles(
    playstyle_profiles: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """Return only the human-readable offensive and defensive summaries."""
    return {
        team: {
            "offense": profile["offense"]["summary"],
            "defense": profile["defense"]["summary"],
        }
        for team, profile in playstyle_profiles.items()
    }


def playstyle_feature_vector(profile: dict[str, Any]) -> dict[str, float]:
    """Flatten a team profile into numeric model features."""
    features: dict[str, float] = {}
    for side in ("offense", "defense"):
        for metric, record in profile[side]["metrics"].items():
            features[f"{side}_{metric}_score"] = float(record["score"])
            features[f"{side}_{metric}_value"] = float(record["value"])
    return features


def playstyle_feature_vectors(
    playstyle_profiles: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Flatten all team play-style profiles into numeric model features."""
    return {
        team: playstyle_feature_vector(profile)
        for team, profile in playstyle_profiles.items()
    }


if __name__ == "__main__":
    from src.data.build_dataset import build_finals_context

    context = build_finals_context()
    profiles = build_playstyle_profiles(context)
    for team, summary in summarize_playstyle_profiles(profiles).items():
        print(team)
        print("  offense:", ", ".join(summary["offense"]))
        print("  defense:", ", ".join(summary["defense"]))
