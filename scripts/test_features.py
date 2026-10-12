"""Leakage checks for the modeling table and coverage of the status rules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.f1_clean import load_raw, project_paths, clean_tables
from src.f1_features import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    build_model_frame,
    modeling_sample,
)
from src.f1_questions import (
    front_row_conversion,
    home_advantage,
    mechanical_retirements,
    status_audit,
)


def _manual_last5(driver_rows: pd.DataFrame) -> None:
    ordered = driver_rows.sort_values(["race_date", "season_round", "raceId"])
    points = ordered["points"].to_numpy() if "points" in ordered.columns else None
    # The model frame no longer carries current-race points. Rebuild the check
    # from the feature definition stored on the row: career_starts is the prior count.
    assert ordered["career_starts"].iloc[0] == 0
    assert ordered["form_missing"].iloc[0] == 1
    assert ordered["career_starts"].tolist() == list(range(len(ordered)))


def main() -> None:
    raw = load_raw(project_paths(ROOT)["raw"])
    result = clean_tables(raw)
    model = build_model_frame(result.driver_race, result.cleaned)
    leaked = [column for column in LEAKAGE_COLUMNS if column in model.columns]
    assert not leaked, leaked
    assert model.duplicated(["raceId", "driverId"]).sum() == 0
    assert set(FEATURE_COLUMNS).issubset(model.columns)

    sample = modeling_sample(model)
    assert sample["year"].min() == 2003
    assert sample["year"].max() == 2024
    assert sample.loc[sample["split"] == "train", "year"].max() == 2018
    assert sample.loc[sample["split"] == "test", "year"].min() == 2022
    assert (sample["year"] > 2024).sum() == 0

    # Entering standings must be the previous round, not the standings published for this race.
    standings = result.cleaned["driver_standings"].merge(
        result.cleaned["races"][["raceId", "year", "round"]], on="raceId", how="left"
    )
    standings["year"] = standings["year"].astype("Int64")
    standings["round"] = standings["round"].astype("Int64")
    check = sample.loc[sample["season_round"] > 1, ["year", "season_round", "driverId", "driver_standing_entering"]].copy()
    check["year"] = check["year"].astype("Int64")
    previous = standings.rename(columns={"round": "season_round", "position": "previous_position"})
    previous["season_round"] = previous["season_round"] + 1
    merged = check.merge(
        previous[["year", "season_round", "driverId", "previous_position"]],
        on=["year", "season_round", "driverId"],
        how="left",
    )
    both = merged["driver_standing_entering"].notna() & merged["previous_position"].notna()
    assert both.sum() > 1000
    assert (
        merged.loc[both, "driver_standing_entering"].astype(float).to_numpy()
        == merged.loc[both, "previous_position"].astype(float).to_numpy()
    ).all()

    current = standings.rename(columns={"round": "season_round", "position": "current_position"})
    same_round = check.merge(
        current[["year", "season_round", "driverId", "current_position"]],
        on=["year", "season_round", "driverId"],
        how="inner",
    )
    compared = same_round["driver_standing_entering"].notna()
    disagreements = (
        same_round.loc[compared, "driver_standing_entering"].astype(float).to_numpy()
        != same_round.loc[compared, "current_position"].astype(float).to_numpy()
    ).sum()
    assert disagreements > 0

    one_driver = model.loc[model["driverId"] == model["driverId"].iloc[0]].copy()
    # career_starts is defined without the current race, so it has to be 0, 1, 2, ...
    # Reattach points from driver_race for an independent last-5 check.
    points = result.driver_race[["raceId", "driverId", "points", "podium"]]
    detailed = one_driver.merge(points, on=["raceId", "driverId"], how="left", suffixes=("", "_label"))
    detailed = detailed.sort_values(["race_date", "season_round", "raceId"]).reset_index(drop=True)
    _manual_last5(detailed)
    for index in range(1, len(detailed)):
        prior = detailed.iloc[max(0, index - 5):index]
        expected_points = float(prior["points"].sum())
        expected_rate = float(prior["podium_label"].astype(float).mean()) if "podium_label" in detailed.columns else float(prior["podium"].astype(float).mean())
        # podium from driver_race merge may be named podium_label if model already has podium
        got_points = detailed.loc[index, "driver_points_last5"]
        got_rate = detailed.loc[index, "driver_podium_rate_last5"]
        assert abs(got_points - expected_points) < 1e-6, (index, got_points, expected_points)
        assert abs(got_rate - expected_rate) < 1e-6, (index, got_rate, expected_rate)

    audit = status_audit(result.cleaned["status"])
    assert audit["category"].isna().sum() == 0
    assert set(audit["category"]) >= {"mechanical", "crash", "driver", "regulation", "classified", "did_not_start"}
    named = audit.set_index("status")["category"]
    assert named["Engine"] == "mechanical"
    assert named["Accident"] == "crash"
    assert named["Disqualified"] == "regulation"
    assert named["Finished"] == "classified"
    assert named["+1 Lap"] == "classified"
    assert named["Did not qualify"] == "did_not_start"

    british = result.cleaned["drivers"]
    british = british.loc[british["nationality"].str.strip() == "British", "driverId"].iloc[0]
    silverstone = result.cleaned["circuits"].loc[result.cleaned["circuits"]["country"] == "UK", "circuitId"].iloc[0]
    home_row = model.loc[(model["driverId"] == british) & (model["circuitId"] == silverstone)]
    assert len(home_row) > 0
    assert home_row["home_race"].eq(1).all()

    ranked = front_row_conversion(result.driver_race, result.cleaned["circuits"], result.cleaned["qualifying"])
    assert ranked["pre_2014"]["rank"].min() == 1
    assert len(ranked["rank_change"]) > 5
    home = home_advantage(result.driver_race, result.cleaned["drivers"], result.cleaned["circuits"])
    assert set(home["where"]) == {"home", "away"}
    mechanical = mechanical_retirements(
        result.driver_race, result.cleaned["status"], result.cleaned["constructors"]
    )
    assert {"constructor", "rate_2014-2021", "rate_2022-2024"}.issubset(mechanical.columns)
    print("feature rows", len(model), "model rows", len(sample))
    print(sample["split"].value_counts().to_string())
    print("status categories")
    print(audit["category"].value_counts().to_string())
    print("PASSED")


if __name__ == "__main__":
    main()
