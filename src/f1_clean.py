"""Cleaning rules for the Formula 1 podium-prediction milestone.

Raw CSVs are read and never written. Sentinels are replaced on copies, types are
applied, the broken lap-time key is repaired, and a separate driver-race table
is built with one row per (raceId, driverId).

This module does not build model features. Columns that are only known after
the race are labeled as such in DRIVER_RACE_DICTIONARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SENTINEL = r"\N"
ASSIGNMENT_LAST_YEAR = 2024

TABLES = (
    "circuits",
    "constructors",
    "constructor_results",
    "constructor_standings",
    "drivers",
    "driver_standings",
    "races",
    "results",
    "qualifying",
    "lap_times",
    "pit_stops",
    "sprint_results",
    "seasons",
    "status",
)

# Primary keys of the Ergast extracts. lap_times is the only one that fails
# on the raw files; cleaning makes it unique.
PRIMARY_KEYS: dict[str, list[str]] = {
    "circuits": ["circuitId"],
    "constructors": ["constructorId"],
    "constructor_results": ["constructorResultsId"],
    "constructor_standings": ["constructorStandingsId"],
    "drivers": ["driverId"],
    "driver_standings": ["driverStandingsId"],
    "races": ["raceId"],
    "results": ["resultId"],
    "qualifying": ["qualifyId"],
    "lap_times": ["raceId", "driverId", "lap"],
    "pit_stops": ["raceId", "driverId", "stop"],
    "sprint_results": ["resultId"],
    "seasons": ["year"],
    "status": ["statusId"],
}

# Grain checks that are not the single-column primary key.
GRAIN_KEYS: dict[str, list[str]] = {
    "results_driver_race": ["raceId", "driverId"],
    "qualifying_driver_race": ["raceId", "driverId"],
    "sprint_driver_race": ["raceId", "driverId"],
    "driver_standings_driver_race": ["raceId", "driverId"],
    "constructor_standings_race": ["raceId", "constructorId"],
    "constructor_results_race": ["raceId", "constructorId"],
}

GRAIN_TABLE = {
    "results_driver_race": "results",
    "qualifying_driver_race": "qualifying",
    "sprint_driver_race": "sprint_results",
    "driver_standings_driver_race": "driver_standings",
    "constructor_standings_race": "constructor_standings",
    "constructor_results_race": "constructor_results",
}

# child table, child column, parent table, parent column
FOREIGN_KEYS = (
    ("races", "circuitId", "circuits", "circuitId"),
    ("results", "raceId", "races", "raceId"),
    ("results", "driverId", "drivers", "driverId"),
    ("results", "constructorId", "constructors", "constructorId"),
    ("results", "statusId", "status", "statusId"),
    ("qualifying", "raceId", "races", "raceId"),
    ("qualifying", "driverId", "drivers", "driverId"),
    ("qualifying", "constructorId", "constructors", "constructorId"),
    ("driver_standings", "raceId", "races", "raceId"),
    ("driver_standings", "driverId", "drivers", "driverId"),
    ("constructor_standings", "raceId", "races", "raceId"),
    ("constructor_standings", "constructorId", "constructors", "constructorId"),
    ("constructor_results", "raceId", "races", "raceId"),
    ("constructor_results", "constructorId", "constructors", "constructorId"),
    ("lap_times", "raceId", "races", "raceId"),
    ("lap_times", "driverId", "drivers", "driverId"),
    ("pit_stops", "raceId", "races", "raceId"),
    ("pit_stops", "driverId", "drivers", "driverId"),
    ("sprint_results", "raceId", "races", "raceId"),
    ("sprint_results", "driverId", "drivers", "driverId"),
    ("sprint_results", "constructorId", "constructors", "constructorId"),
    ("sprint_results", "statusId", "status", "statusId"),
)

INTEGER_COLUMNS: dict[str, list[str]] = {
    "circuits": ["circuitId"],
    "constructors": ["constructorId"],
    "constructor_results": ["constructorResultsId", "raceId", "constructorId"],
    "constructor_standings": ["constructorStandingsId", "raceId", "constructorId", "position", "wins"],
    "drivers": ["driverId", "number"],
    "driver_standings": ["driverStandingsId", "raceId", "driverId", "position", "wins"],
    "races": ["raceId", "year", "round", "circuitId"],
    "results": [
        "resultId", "raceId", "driverId", "constructorId", "number", "grid",
        "position", "positionOrder", "laps", "milliseconds", "fastestLap", "rank", "statusId",
    ],
    "qualifying": ["qualifyId", "raceId", "driverId", "constructorId", "number", "position"],
    "lap_times": ["raceId", "driverId", "lap", "position", "milliseconds"],
    "pit_stops": ["raceId", "driverId", "stop", "lap", "milliseconds"],
    "sprint_results": [
        "resultId", "raceId", "driverId", "constructorId", "number", "grid",
        "position", "positionOrder", "laps", "milliseconds", "fastestLap", "statusId", "rank",
    ],
    "seasons": ["year"],
    "status": ["statusId"],
}

FLOAT_COLUMNS: dict[str, list[str]] = {
    "circuits": ["lat", "lng", "alt"],
    "constructor_results": ["points"],
    "constructor_standings": ["points"],
    "driver_standings": ["points"],
    "results": ["points", "fastestLapSpeed"],
    "sprint_results": ["points"],
}

DATE_COLUMNS: dict[str, list[str]] = {
    "drivers": ["dob"],
    "races": [
        "date", "fp1_date", "fp2_date", "fp3_date", "quali_date", "sprint_date",
    ],
}

# Clock strings such as 06:00:00. Kept as text and also combined with the date.
CLOCK_COLUMNS: dict[str, list[str]] = {
    "races": ["time", "fp1_time", "fp2_time", "fp3_time", "quali_time", "sprint_time"],
}

DURATION_COLUMNS: dict[str, list[str]] = {
    "qualifying": ["q1", "q2", "q3"],
    "lap_times": ["time"],
    "pit_stops": ["duration"],
    "results": ["fastestLapTime"],
    "sprint_results": ["fastestLapTime"],
}

DRIVER_RACE_DICTIONARY = (
    ("raceId", "identifier", "Race key. Already known when the race is scheduled."),
    ("driverId", "identifier", "Driver key."),
    ("resultId", "identifier", "resultId of the row kept by the collapse rule."),
    ("year", "pre_race_context", "Season. Known before the race."),
    ("round", "pre_race_context", "Round number. Known before the race."),
    ("race_date", "pre_race_context", "Scheduled race date."),
    ("circuitId", "pre_race_context", "Circuit. Known before the race."),
    ("race_name", "pre_race_context", "Grand Prix name."),
    ("constructorId", "pre_race_context", "Constructor on the kept row. When n_constructors > 1 the driver drove more than one car, so this is the car of the better classified stint, not necessarily the car they qualified."),
    ("car_number", "pre_race_context", "Car number on the kept row."),
    ("grid", "pre_race_context", "Starting grid. 0 means no grid slot (pit lane or did not start). Null means the source file omitted the grid for that row."),
    ("n_result_rows", "audit", "How many results rows this driver had in this race before the collapse."),
    ("n_constructors", "audit", "How many constructors those rows used."),
    ("shared_entry", "audit", "True when n_result_rows > 1. These are shared drives or double entries, almost all from 1950-1964, plus one 1978 case."),
    ("position", "post_race_label", "Official classified position. Null means not classified. Not a legal model feature: it is known only after the race."),
    ("positionText", "post_race_label", "Official position text, including R, D, W, F, N. Not a legal model feature."),
    ("positionOrder", "post_race_label", "Dense rank of every starter, including retirements. The brief defines the label from this column. Not a legal model feature."),
    ("points", "post_race_label", "Points scored in this race. Not a legal model feature."),
    ("laps", "post_race_label", "Laps completed. Not a legal model feature."),
    ("statusId", "post_race_label", "Finishing status. Not a legal model feature."),
    ("podium", "post_race_label", "Brief's label: positionOrder <= 3."),
    ("official_podium", "post_race_label", "positionText is 1, 2, or 3. Kept so the one known disagreement with podium can be audited."),
    ("podium_definition_conflict", "audit", "True when podium and official_podium disagree."),
    ("in_assignment_window", "audit", "True when year <= 2024, the window named in the brief. Later completed races are retained and flagged, not deleted."),
    ("collapse_rule", "audit", "Rule that chose resultId. The same rule is used on every row."),
)

COLLAPSE_RULE = "best_classification"


@dataclass
class CleaningResult:
    cleaned: dict[str, pd.DataFrame]
    sentinel_summary: pd.DataFrame
    keys_before: pd.DataFrame
    keys_after: pd.DataFrame
    coercion_failures: pd.DataFrame
    lap_conflict_summary: pd.DataFrame
    lap_conflicts: pd.DataFrame
    duplicate_rows: pd.DataFrame
    rule_comparison: pd.DataFrame
    driver_race: pd.DataFrame
    races_without_results: pd.DataFrame
    time_checks: pd.DataFrame
    missing_policy: pd.DataFrame


def project_paths(root: Path | None = None) -> dict[str, Path]:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "raw": root / "data" / "raw",
        "cleaned": root / "data" / "cleaned",
        "processed": root / "data" / "processed",
        "audit": root / "data" / "audit",
        "figures": root / "figures",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_hashes(raw_dir: Path) -> dict[str, str]:
    return {name: file_sha256(raw_dir / f"{name}.csv") for name in TABLES}


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the 14 assignment tables as strings, with no sentinel conversion."""
    frames = {}
    for name in TABLES:
        path = raw_dir / f"{name}.csv"
        frames[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frames


def summarize_sentinels(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in raw.items():
        for column in frame.columns:
            sentinel_n = int((frame[column] == SENTINEL).sum())
            if sentinel_n == 0:
                continue
            rows.append({
                "table": name,
                "column": column,
                "sentinel_n": sentinel_n,
                "rows": len(frame),
                "sentinel_share": sentinel_n / len(frame),
            })
    summary = pd.DataFrame(rows)
    return summary.sort_values(["sentinel_share", "table", "column"], ascending=[False, True, True]).reset_index(drop=True)


# What to do with each column that contains the \N marker.
# mean, median, and mode are not used: a blank here means "not recorded" or "did not happen".
MISSING_POLICY: dict[tuple[str, str], tuple[str, str]] = {
    ("constructor_results", "status"): (
        "code as not disqualified",
        "\\N means the constructor was not disqualified. Stored as disqualified=False. D stays True.",
    ),
    ("constructor_standings", "position"): (
        "leave missing",
        "One standings row is unranked. A median rank would invent a place.",
    ),
    ("drivers", "number"): (
        "leave missing",
        "Permanent numbers did not exist for most of the history. The mode would copy one modern number onto old drivers.",
    ),
    ("drivers", "code"): (
        "leave missing",
        "Three-letter codes did not exist for most of the history. The mode would copy one code onto old drivers.",
    ),
    ("driver_standings", "position"): (
        "leave missing",
        "The text is already '-': unranked. The mode would assign a championship place.",
    ),
    ("races", "time"): (
        "leave missing",
        "The start clock was not stored. An average of clock times is not a start time.",
    ),
    ("races", "fp1_date"): ("leave missing", "This session was not stored for that race."),
    ("races", "fp1_time"): ("leave missing", "This session was not stored for that race."),
    ("races", "fp2_date"): ("leave missing", "This session was not stored for that race."),
    ("races", "fp2_time"): ("leave missing", "This session was not stored for that race."),
    ("races", "fp3_date"): ("leave missing", "This session was not stored for that race."),
    ("races", "fp3_time"): ("leave missing", "This session was not stored for that race."),
    ("races", "quali_date"): ("leave missing", "This session was not stored for that race."),
    ("races", "quali_time"): ("leave missing", "This session was not stored for that race."),
    ("races", "sprint_date"): ("leave missing", "This race had no sprint, or the sprint date was not stored."),
    ("races", "sprint_time"): ("leave missing", "This race had no sprint, or the sprint time was not stored."),
    ("results", "number"): (
        "leave missing",
        "A few cars have no number. The mode would give them a number that already belongs to another car.",
    ),
    ("results", "grid"): (
        "leave missing",
        "The file omitted the grid for two whole races. The median grid would pretend those drivers started from midfield. Grid 0 is a real value and is not filled.",
    ),
    ("results", "position"): (
        "leave missing",
        "Empty means not classified. positionOrder is already filled. A median position would create a finish.",
    ),
    ("results", "time"): (
        "leave missing",
        "No finish clock was stored. A median time would give a retired car a finish time. Known only after the race.",
    ),
    ("results", "milliseconds"): (
        "leave missing",
        "No elapsed time was stored. Same column as the finish clock, in milliseconds.",
    ),
    ("results", "fastestLap"): (
        "leave missing",
        "No fastest lap was stored. Known only after the race.",
    ),
    ("results", "rank"): (
        "leave missing",
        "No fastest-lap rank was stored. Known only after the race.",
    ),
    ("results", "fastestLapTime"): (
        "leave missing",
        "No fastest-lap time was stored. A median lap time would invent one.",
    ),
    ("results", "fastestLapSpeed"): (
        "leave missing",
        "No fastest-lap speed was stored. The distribution is skewed, and a median would still invent a speed.",
    ),
    ("qualifying", "q1"): (
        "leave missing",
        "The driver set no Q1 time. A median lap time would invent one.",
    ),
    ("qualifying", "q2"): (
        "leave missing",
        "The driver did not reach Q2. A median would say they set a Q2 time.",
    ),
    ("qualifying", "q3"): (
        "leave missing",
        "The driver did not reach Q3. A median would say they set a Q3 time.",
    ),
    ("pit_stops", "duration"): (
        "leave missing",
        "Three stops have no duration. A median would hide a broken measurement. Pit stops happen during the race.",
    ),
    ("pit_stops", "milliseconds"): (
        "leave missing",
        "The same three stops have no duration in milliseconds.",
    ),
    ("sprint_results", "position"): (
        "leave missing",
        "The driver was not classified in the sprint.",
    ),
    ("sprint_results", "time"): (
        "leave missing",
        "No sprint finish clock was stored.",
    ),
    ("sprint_results", "milliseconds"): (
        "leave missing",
        "No sprint elapsed time was stored.",
    ),
    ("sprint_results", "fastestLap"): (
        "leave missing",
        "No sprint fastest lap was stored.",
    ),
    ("sprint_results", "fastestLapTime"): (
        "leave missing",
        "No sprint fastest-lap time was stored.",
    ),
    ("sprint_results", "statusId"): (
        "leave missing",
        "Two sprint rows have no status. The mode would mark them as the most common status.",
    ),
    ("sprint_results", "rank"): (
        "leave missing",
        "No sprint fastest-lap rank was stored.",
    ),
}

DEFAULT_MISSING_POLICY = (
    "leave missing",
    "The marker is an absent value. Mean, median, and mode would invent one.",
)


def missing_value_report(raw: dict[str, pd.DataFrame], cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Record the method applied to every column that contained \\N. Nothing is imputed."""
    rows = []
    for name, frame in raw.items():
        for column in frame.columns:
            sentinel_n = int((frame[column] == SENTINEL).sum())
            if sentinel_n == 0:
                continue
            method, reason = MISSING_POLICY.get((name, column), DEFAULT_MISSING_POLICY)
            missing_after = int(cleaned[name][column].isna().sum())
            rows.append({
                "table": name,
                "column": column,
                "blank_markers": sentinel_n,
                "method": method,
                "values_filled": sentinel_n - missing_after,
                "still_missing": missing_after,
                "reason": reason,
            })
    status = raw["constructor_results"]["status"]
    disqualified = cleaned["constructor_results"]["disqualified"]
    rows.append({
        "table": "constructor_results",
        "column": "disqualified",
        "blank_markers": int((status == SENTINEL).sum()),
        "method": "filled with False",
        "values_filled": int((~disqualified).sum()),
        "still_missing": int(disqualified.isna().sum()),
        "reason": "Each \\N in status was written here as False. Every row now has True or False.",
    })
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values(["table", "column"]).reset_index(drop=True)


def replace_sentinels(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn the Ergast null marker into missing values. Does not change row count."""
    cleaned = frame.copy()
    return cleaned.mask(cleaned == SENTINEL, pd.NA)


def parse_duration_to_seconds(series: pd.Series) -> tuple[pd.Series, int]:
    """Parse ss.sss, m:ss.sss, or h:mm:ss.sss. A leading + is removed by the caller when needed.

    Returns the seconds and the number of non-missing values that did not parse.
    """
    text = series.astype("string").str.strip()
    missing = text.isna() | (text == "")
    n_colons = text.str.count(":").fillna(0)
    parts = text.str.split(":", n=2, expand=True)
    while parts.shape[1] < 3:
        parts[parts.shape[1]] = pd.NA
    numbers = [pd.to_numeric(parts[i], errors="coerce") for i in range(3)]
    seconds = pd.Series(np.nan, index=series.index, dtype="float64")
    one = (~missing) & (n_colons == 0)
    two = (~missing) & (n_colons == 1)
    three = (~missing) & (n_colons == 2)
    seconds.loc[one] = numbers[0].loc[one]
    seconds.loc[two] = numbers[0].loc[two] * 60 + numbers[1].loc[two]
    seconds.loc[three] = numbers[0].loc[three] * 3600 + numbers[1].loc[three] * 60 + numbers[2].loc[three]
    failed = int((~missing & seconds.isna()).sum())
    return seconds, failed


def _cast_numeric(series: pd.Series, dtype: str) -> tuple[pd.Series, int]:
    missing = series.isna()
    numeric = pd.to_numeric(series, errors="coerce")
    failed = ~missing & numeric.isna()
    if dtype == "Int64":
        fractional = numeric.notna() & ((numeric % 1) != 0)
        failed = failed | fractional
        numeric = numeric.mask(fractional)
        numeric = numeric.astype("Int64")
    else:
        numeric = numeric.astype("float64")
    return numeric, int(failed.sum())


def _cast_dates(series: pd.Series) -> tuple[pd.Series, int]:
    missing = series.isna()
    parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    failed = int((~missing & parsed.isna()).sum())
    return parsed, failed


def _record_failure(rows: list[dict], table: str, column: str, failed: int) -> None:
    if failed:
        rows.append({"table": table, "column": column, "failed_n": failed})


def _apply_declared_types(name: str, frame: pd.DataFrame, failures: list[dict]) -> pd.DataFrame:
    """Cast the columns this file actually has. A missing optional column is skipped.

    The Kaggle extract does not always include `rank` (fastest-lap rank). That
    field is only known after the race, so its absence does not block cleaning.
    """
    cleaned = frame.copy()
    for column in INTEGER_COLUMNS.get(name, []):
        if column not in cleaned.columns:
            continue
        cleaned[column], failed = _cast_numeric(cleaned[column], "Int64")
        _record_failure(failures, name, column, failed)
    for column in FLOAT_COLUMNS.get(name, []):
        if column not in cleaned.columns:
            continue
        cleaned[column], failed = _cast_numeric(cleaned[column], "float64")
        _record_failure(failures, name, column, failed)
    for column in DATE_COLUMNS.get(name, []):
        if column not in cleaned.columns:
            continue
        cleaned[column], failed = _cast_dates(cleaned[column])
        _record_failure(failures, name, column, failed)
    for column in DURATION_COLUMNS.get(name, []):
        if column not in cleaned.columns:
            continue
        seconds, failed = parse_duration_to_seconds(cleaned[column])
        cleaned[f"{column}_seconds"] = seconds
        _record_failure(failures, name, f"{column}_seconds", failed)
    return cleaned


def _parse_results_time(frame: pd.DataFrame, table: str, failures: list[dict]) -> pd.DataFrame:
    """`time` mixes two different measurements. Keep them in separate columns.

    A value with no leading + is the winner's elapsed race time.
    A value with a leading + is the gap to the winner.
    `milliseconds`, when present, is elapsed time for that car, including cars
    whose `time` string is only a gap. It is not the gap itself.
    """
    cleaned = frame.copy()
    if "time" not in cleaned.columns:
        return cleaned
    text = cleaned["time"].astype("string")
    is_gap = text.str.startswith("+", na=False)
    gap_seconds, gap_failed = parse_duration_to_seconds(text.where(is_gap).str.replace(r"^\+", "", regex=True))
    winner_seconds, winner_failed = parse_duration_to_seconds(text.where(~is_gap))
    cleaned["gap_seconds"] = gap_seconds
    cleaned["winner_time_seconds"] = winner_seconds
    _record_failure(failures, table, "gap_seconds", gap_failed)
    _record_failure(failures, table, "winner_time_seconds", winner_failed)
    if "milliseconds" in cleaned.columns:
        cleaned["elapsed_seconds"] = cleaned["milliseconds"].astype("Float64") / 1000
    return cleaned


def _add_race_start(frame: pd.DataFrame, failures: list[dict]) -> pd.DataFrame:
    cleaned = frame.copy()
    date_text = cleaned["date"].dt.strftime("%Y-%m-%d")
    clock = cleaned["time"].astype("string")
    both = date_text.notna() & clock.notna()
    stamp = pd.to_datetime(date_text.where(both) + " " + clock.where(both), errors="coerce")
    failed = int((both & stamp.isna()).sum())
    _record_failure(failures, "races", "race_start", failed)
    cleaned["race_start"] = stamp
    return cleaned


def _key_rows_for(name: str, frame: pd.DataFrame, keys: list[str], check: str) -> dict:
    missing = frame[keys].isna().any(axis=1)
    problem_rows = int(frame.duplicated(keys).sum())
    return {
        "table": name,
        "check": check,
        "columns": ", ".join(keys),
        "rows": len(frame),
        "null_key_rows": int(missing.sum()),
        "problem_rows": problem_rows,
        "passed": bool(missing.sum() == 0 and problem_rows == 0),
    }


def audit_keys(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, keys in PRIMARY_KEYS.items():
        rows.append(_key_rows_for(name, frames[name], keys, "primary_key"))
    for grain_name, keys in GRAIN_KEYS.items():
        table = GRAIN_TABLE[grain_name]
        rows.append(_key_rows_for(grain_name, frames[table], keys, "grain"))
    parent_ids = {name: set(frame[PRIMARY_KEYS[name][0]].dropna()) for name, frame in frames.items() if len(PRIMARY_KEYS[name]) == 1}
    for child, column, parent, parent_column in FOREIGN_KEYS:
        values = frames[child][column]
        missing = values.isna()
        orphans = ~missing & ~values.isin(parent_ids[parent])
        rows.append({
            "table": child,
            "check": "foreign_key",
            "columns": f"{column} -> {parent}.{parent_column}",
            "rows": len(frames[child]),
            "null_key_rows": int(missing.sum()),
            "problem_rows": int(orphans.sum()),
            "passed": bool(orphans.sum() == 0),
        })
    return pd.DataFrame(rows)


def deduplicate_lap_times(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make (raceId, driverId, lap) unique.

    Exact copies are dropped. When two copies disagree, the faster time is kept
    and both measurements are written to the conflict audit. Lap times happen
    during the race, so this repair is for key integrity, not for model features.
    """
    keys = ["raceId", "driverId", "lap"]
    ordered = frame.sort_values(keys + ["milliseconds"], kind="mergesort")
    grouped = ordered.groupby(keys, dropna=False)["milliseconds"]
    spread = grouped.agg(n="size", min_ms="min", max_ms="max").reset_index()
    summary = pd.DataFrame([
        {"kind": "unique_already", "groups": int((spread["n"] == 1).sum())},
        {"kind": "exact_duplicate", "groups": int(((spread["n"] > 1) & (spread["min_ms"] == spread["max_ms"])).sum())},
        {"kind": "one_millisecond", "groups": int(((spread["n"] > 1) & ((spread["max_ms"] - spread["min_ms"]) == 1)).sum())},
        {"kind": "larger_conflict", "groups": int(((spread["n"] > 1) & ((spread["max_ms"] - spread["min_ms"]) > 1)).sum())},
    ])
    conflict_keys = spread.loc[spread["n"] > 1, keys]
    conflict_rows = ordered.merge(conflict_keys, on=keys, how="inner")
    conflict_rows = conflict_rows.copy()
    conflict_rows["delta_ms"] = conflict_rows.groupby(keys)["milliseconds"].transform(lambda s: s.max() - s.min())
    kept = ordered.drop_duplicates(keys, keep="first").reset_index(drop=True)
    return kept, summary, conflict_rows.reset_index(drop=True)


def collapse_entries(results: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Keep one results row per driver per race.

    best_classification: lowest positionOrder, then most laps, then smallest resultId.
    most_laps: most laps, then lowest positionOrder, then smallest resultId.
    """
    if rule == "best_classification":
        sort_cols = ["raceId", "driverId", "positionOrder", "laps", "resultId"]
        ascending = [True, True, True, False, True]
    elif rule == "most_laps":
        sort_cols = ["raceId", "driverId", "laps", "positionOrder", "resultId"]
        ascending = [True, True, False, True, True]
    else:
        raise ValueError(f"Unknown collapse rule: {rule}")
    ordered = results.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return ordered.drop_duplicates(["raceId", "driverId"], keep="first")


def compare_collapse_rules(results: pd.DataFrame) -> pd.DataFrame:
    """Run both candidate rules on duplicate pairs only."""
    duplicate = results["n_result_rows"] > 1 if "n_result_rows" in results.columns else results.duplicated(["raceId", "driverId"], keep=False)
    subset = results.loc[duplicate].copy()
    if "n_result_rows" not in subset.columns:
        subset["n_result_rows"] = subset.groupby(["raceId", "driverId"])["resultId"].transform("size")
    best = collapse_entries(subset, "best_classification")
    laps = collapse_entries(subset, "most_laps")
    compared = best[["raceId", "driverId", "resultId", "positionOrder", "laps", "constructorId", "grid"]].merge(
        laps[["raceId", "driverId", "resultId", "positionOrder", "laps", "constructorId", "grid"]],
        on=["raceId", "driverId"],
        suffixes=("_best", "_laps"),
    )
    compared["same_row"] = compared["resultId_best"] == compared["resultId_laps"]
    compared["same_position"] = compared["positionOrder_best"] == compared["positionOrder_laps"]
    compared["same_podium"] = (compared["positionOrder_best"] <= 3) == (compared["positionOrder_laps"] <= 3)
    return compared


def build_driver_race(results: pd.DataFrame, races: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One row per driver per race.

    Rule, in order: better classified result (lower positionOrder), then more
    laps, then the smaller resultId. positionOrder is first because the label
    is the driver's finishing result. The most-laps rule is computed as a check,
    not as the rule that is stored.
    """
    working = results.copy()
    working["n_result_rows"] = working.groupby(["raceId", "driverId"])["resultId"].transform("size")
    working["n_constructors"] = working.groupby(["raceId", "driverId"])["constructorId"].transform("nunique")
    comparison = compare_collapse_rules(working)
    kept = collapse_entries(working, COLLAPSE_RULE).copy()
    kept["shared_entry"] = kept["n_result_rows"] > 1
    kept["podium"] = kept["positionOrder"] <= 3
    kept["official_podium"] = kept["positionText"].isin(["1", "2", "3"])
    kept["podium_definition_conflict"] = kept["podium"] != kept["official_podium"]
    kept["collapse_rule"] = COLLAPSE_RULE

    context = races[["raceId", "year", "round", "date", "circuitId", "name"]].rename(
        columns={"date": "race_date", "name": "race_name"}
    )
    driver_race = kept.merge(context, on="raceId", how="left", validate="many_to_one")
    driver_race["in_assignment_window"] = driver_race["year"] <= ASSIGNMENT_LAST_YEAR
    driver_race = driver_race.rename(columns={"number": "car_number"})
    columns = [name for name, _, _ in DRIVER_RACE_DICTIONARY]
    driver_race = driver_race[columns].sort_values(["year", "round", "positionOrder", "driverId"]).reset_index(drop=True)

    duplicate_rows = working.loc[working["n_result_rows"] > 1].copy()
    duplicate_rows["kept"] = duplicate_rows["resultId"].isin(set(driver_race["resultId"]))
    return driver_race, duplicate_rows, comparison


def time_consistency(cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Parsed strings should match the millisecond columns already in the file."""
    rows = []

    results = cleaned["results"]
    absolute = results["winner_time_seconds"].notna() & results["elapsed_seconds"].notna()
    if absolute.any():
        delta = (results.loc[absolute, "winner_time_seconds"] - results.loc[absolute, "elapsed_seconds"]).abs()
        rows.append({"check": "results winner time vs milliseconds", "rows": int(absolute.sum()), "max_abs_seconds": float(delta.max())})

    laps = cleaned["lap_times"]
    lap_delta = (laps["time_seconds"] - laps["milliseconds"].astype("float64") / 1000).abs()
    rows.append({"check": "lap time string vs milliseconds", "rows": int(lap_delta.notna().sum()), "max_abs_seconds": float(lap_delta.max())})

    pits = cleaned["pit_stops"]
    known = pits["duration_seconds"].notna() & pits["milliseconds"].notna()
    pit_delta = (pits.loc[known, "duration_seconds"] - pits.loc[known, "milliseconds"].astype("float64") / 1000).abs()
    rows.append({
        "check": "pit duration string vs milliseconds",
        "rows": int(known.sum()),
        "max_abs_seconds": float(pit_delta.max()) if known.any() else 0.0,
    })
    return pd.DataFrame(rows)


def clean_tables(raw: dict[str, pd.DataFrame]) -> CleaningResult:
    keys_before = audit_keys(raw)
    sentinel_summary = summarize_sentinels(raw)
    failures: list[dict] = []
    cleaned: dict[str, pd.DataFrame] = {}
    for name, frame in raw.items():
        typed = _apply_declared_types(name, replace_sentinels(frame), failures)
        if name in {"results", "sprint_results"}:
            typed = _parse_results_time(typed, name, failures)
        if name == "races":
            typed = _add_race_start(typed, failures)
        if name == "constructor_results":
            # In this column \N means "not disqualified", not "status unknown".
            typed["disqualified"] = typed["status"].eq("D")
        if name == "results":
            typed["classified"] = typed["position"].notna()
        cleaned[name] = typed

    cleaned["lap_times"], lap_summary, lap_conflicts = deduplicate_lap_times(cleaned["lap_times"])
    driver_race, duplicate_rows, comparison = build_driver_race(cleaned["results"], cleaned["races"])
    keys_after = audit_keys(cleaned)
    # The modeling grain is a new table, so add its key check beside the source-table audit.
    grain = _key_rows_for("driver_race", driver_race, ["raceId", "driverId"], "grain")
    keys_after = pd.concat([keys_after, pd.DataFrame([grain])], ignore_index=True)

    result_races = set(cleaned["results"]["raceId"])
    missing_results = cleaned["races"].loc[~cleaned["races"]["raceId"].isin(result_races), ["raceId", "year", "round", "name", "date"]].copy()
    missing_results = missing_results.sort_values(["year", "round"]).reset_index(drop=True)

    return CleaningResult(
        cleaned=cleaned,
        sentinel_summary=sentinel_summary,
        keys_before=keys_before,
        keys_after=keys_after,
        coercion_failures=pd.DataFrame(failures, columns=["table", "column", "failed_n"]),
        lap_conflict_summary=lap_summary,
        lap_conflicts=lap_conflicts,
        duplicate_rows=duplicate_rows,
        rule_comparison=comparison,
        driver_race=driver_race,
        races_without_results=missing_results,
        time_checks=time_consistency(cleaned),
        missing_policy=missing_value_report(raw, cleaned),
    )


def save_outputs(result: CleaningResult, paths: dict[str, Path]) -> None:
    for folder in ("cleaned", "processed", "audit"):
        paths[folder].mkdir(parents=True, exist_ok=True)
    for name, frame in result.cleaned.items():
        frame.to_csv(paths["cleaned"] / f"{name}.csv", index=False)
    result.driver_race.to_csv(paths["processed"] / "driver_race.csv", index=False)
    dictionary = pd.DataFrame(DRIVER_RACE_DICTIONARY, columns=["column", "role", "meaning"])
    dictionary.to_csv(paths["processed"] / "driver_race_dictionary.csv", index=False)
    result.sentinel_summary.to_csv(paths["audit"] / "sentinel_summary.csv", index=False)
    result.keys_before.to_csv(paths["audit"] / "keys_before.csv", index=False)
    result.keys_after.to_csv(paths["audit"] / "keys_after.csv", index=False)
    result.coercion_failures.to_csv(paths["audit"] / "coercion_failures.csv", index=False)
    result.lap_conflict_summary.to_csv(paths["audit"] / "lap_time_conflict_summary.csv", index=False)
    result.lap_conflicts.to_csv(paths["audit"] / "lap_time_conflicts.csv", index=False)
    result.duplicate_rows.to_csv(paths["audit"] / "duplicate_result_rows.csv", index=False)
    result.rule_comparison.to_csv(paths["audit"] / "collapse_rule_comparison.csv", index=False)
    result.races_without_results.to_csv(paths["audit"] / "races_without_results.csv", index=False)
    result.time_checks.to_csv(paths["audit"] / "time_checks.csv", index=False)
    result.missing_policy.to_csv(paths["audit"] / "missing_value_policy.csv", index=False)
