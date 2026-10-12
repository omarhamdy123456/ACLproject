"""Checks the cleaning rules against the raw Ergast extract."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.f1_clean import (  # noqa: E402
    COLLAPSE_RULE,
    SENTINEL,
    clean_tables,
    load_raw,
    parse_duration_to_seconds,
    project_paths,
    raw_hashes,
)


def main() -> None:
    paths = project_paths(ROOT)
    before = raw_hashes(paths["raw"])
    raw = load_raw(paths["raw"])
    result = clean_tables(raw)
    after = raw_hashes(paths["raw"])
    assert before == after, "raw CSVs changed"

    assert result.coercion_failures.empty, result.coercion_failures
    assert (result.time_checks["max_abs_seconds"] < 0.01).all(), result.time_checks

    seconds, failed = parse_duration_to_seconds(pd.Series(["1:26.572", "58.2", "3:22:11.0", pd.NA]))
    assert failed == 0
    assert abs(seconds.iloc[0] - 86.572) < 1e-9
    assert abs(seconds.iloc[1] - 58.2) < 1e-9
    assert abs(seconds.iloc[2] - 12131.0) < 1e-9
    assert pd.isna(seconds.iloc[3])

    results = result.cleaned["results"]
    assert len(results) == len(raw["results"])
    assert not (results.astype(str) == SENTINEL).any().any()
    laps = result.cleaned["lap_times"]
    assert not laps.duplicated(["raceId", "driverId", "lap"]).any()

    keys = result.keys_after
    primary = keys[(keys["check"] == "primary_key")]
    foreign = keys[(keys["check"] == "foreign_key")]
    assert primary["passed"].all(), primary.loc[~primary["passed"]]
    assert foreign["passed"].all(), foreign.loc[~foreign["passed"]]

    raw_grain = result.keys_before.query("table == 'results_driver_race'").iloc[0]
    assert not raw_grain["passed"]
    assert raw_grain["problem_rows"] == 91

    driver_race = result.driver_race
    assert not driver_race.duplicated(["raceId", "driverId"]).any()
    assert len(driver_race) == len(raw["results"]) - 91
    assert driver_race["collapse_rule"].eq(COLLAPSE_RULE).all()
    assert driver_race["podium"].eq(driver_race["positionOrder"] <= 3).all()
    assert driver_race["podium_definition_conflict"].sum() == 1
    assert result.rule_comparison["same_row"].all()
    assert result.rule_comparison["same_podium"].all()
    assert len(result.races_without_results) == 7
    assert result.driver_race["year"].max() >= 2025
    assert set(result.duplicate_rows.loc[result.duplicate_rows["kept"], "resultId"]) <= set(driver_race["resultId"])
    kept_n = int(result.duplicate_rows["kept"].sum())
    assert kept_n == result.duplicate_rows.groupby(["raceId", "driverId"]).ngroups

    # Synthetic rule check: a short podium stint beats a long retirement.
    synthetic = pd.DataFrame([
        {"raceId": 1, "driverId": 1, "resultId": 10, "positionOrder": 5, "laps": 80, "constructorId": 1, "grid": 2, "positionText": "R", "n_result_rows": 2},
        {"raceId": 1, "driverId": 1, "resultId": 11, "positionOrder": 1, "laps": 10, "constructorId": 2, "grid": 4, "positionText": "1", "n_result_rows": 2},
    ])
    from src.f1_clean import collapse_entries
    chosen = collapse_entries(synthetic, "best_classification")
    assert int(chosen.iloc[0]["resultId"]) == 11

    policy = result.missing_policy
    left = policy.loc[policy["method"] == "leave missing", "values_filled"]
    assert left.eq(0).all()
    coded = policy.loc[policy["column"] == "disqualified"].iloc[0]
    assert coded["method"] == "filled with False"
    assert int(coded["still_missing"]) == 0
    assert int(coded["values_filled"]) == int((raw["constructor_results"]["status"] == r"\N").sum())

    print("PASSED")
    print(result.time_checks.to_string(index=False))
    print(result.lap_conflict_summary.to_string(index=False))
    print("driver_race", len(driver_race), "conflicts", int(driver_race["podium_definition_conflict"].sum()))
    print("shared", int(driver_race["shared_entry"].sum()), "multi", int((driver_race["n_constructors"] > 1).sum()))
    print(result.races_without_results.to_string(index=False))


if __name__ == "__main__":
    main()
