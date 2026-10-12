"""Leakage-safe features for one driver in one race.

The prediction cutoff is after qualifying and before the race starts.
Current-race results, points, laps, status, pit stops, and the standings
published for this race are not features. Rolling form and entering
standings use only earlier races.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# First season in which every completed race in this extract has qualifying.
QUALIFYING_COMPLETE_FROM = 2003
TRAIN_LAST_YEAR = 2018
VALIDATION_LAST_YEAR = 2021
TEST_LAST_YEAR = 2024

FEATURE_DICTIONARY = (
    ("grid", "grid", "Starting grid after penalties. Known before the race. 0 means no grid slot, and it is not filled in."),
    ("front_row", "grid", "1 when the grid is 1 or 2, otherwise 0. Missing when the grid itself is missing."),
    ("quali_position", "qualifying", "Official qualifying position for this race. Known after qualifying. Missing when that session is not in the file."),
    ("reached_q3", "qualifying", "1 when the driver set a Q3 time. Missing for a whole race that did not use Q3, so a blank is not treated as an elimination."),
    ("quali_position_missing", "qualifying", "1 when quali_position is missing, so an imputed position is not mistaken for a real one."),
    ("driver_podium_rate_last5", "form", "Share of this driver's previous 5 starts that were podiums. This race is excluded."),
    ("driver_points_last5", "form", "Points from this driver's previous 5 starts. This race's points are excluded."),
    ("constructor_podium_rate_last5", "form", "Mean podium share of this constructor's cars over its previous 5 races. This race is excluded."),
    ("career_starts", "form", "Number of race starts this driver had before this one."),
    ("form_missing", "form", "1 on a driver's first start in the file, when the previous-5 rates do not exist."),
    ("driver_standing_entering", "standings", "Driver championship position after the previous round. The standings row for this race is not used."),
    ("constructor_standing_entering", "standings", "Constructor championship position after the previous round."),
    ("standing_missing", "standings", "1 when there is no driver standings row for the previous round. That includes every round 1."),
    ("season_round", "standings", "Round number in the season. It is known when the calendar is published."),
    ("home_race", "home", "1 when the circuit's country matches the driver's nationality. A dual national matches either country."),
)

FEATURE_COLUMNS = [name for name, _, _ in FEATURE_DICTIONARY]
FEATURE_GROUPS: dict[str, list[str]] = {}
for _name, _group, _meaning in FEATURE_DICTIONARY:
    FEATURE_GROUPS.setdefault(_group, []).append(_name)

# Post-race fields. They may be used to build a shifted feature, and they must
# not appear as columns the model is allowed to see.
LEAKAGE_COLUMNS = (
    "position",
    "positionText",
    "positionOrder",
    "points",
    "laps",
    "statusId",
    "fastestLap",
    "rank",
    "milliseconds",
    "official_podium",
)

# Nationality strings in drivers.csv, after stripping whitespace.
# Circuit countries keep the file's own names, including both "USA" and "United States".
NATIONALITY_COUNTRIES = {
    "American": ("USA", "United States"),
    "American-Italian": ("USA", "United States", "Italy"),
    "Argentine": ("Argentina",),
    "Argentine-Italian": ("Argentina", "Italy"),
    "Argentinian": ("Argentina",),
    "Australian": ("Australia",),
    "Austrian": ("Austria",),
    "Belgian": ("Belgium",),
    "Brazilian": ("Brazil",),
    "British": ("UK",),
    "Canadian": ("Canada",),
    "Chilean": ("Chile",),
    "Chinese": ("China",),
    "Colombian": ("Colombia",),
    "Czech": ("Czech Republic",),
    "Danish": ("Denmark",),
    "Dutch": ("Netherlands",),
    "East German": ("Germany",),
    "Finnish": ("Finland",),
    "French": ("France",),
    "German": ("Germany",),
    "Hungarian": ("Hungary",),
    "Indian": ("India",),
    "Indonesian": ("Indonesia",),
    "Irish": ("Ireland",),
    "Italian": ("Italy",),
    "Japanese": ("Japan",),
    "Liechtensteiner": ("Liechtenstein",),
    "Malaysian": ("Malaysia",),
    "Mexican": ("Mexico",),
    "Monegasque": ("Monaco",),
    "New Zealander": ("New Zealand",),
    "Polish": ("Poland",),
    "Portuguese": ("Portugal",),
    "Rhodesian": ("Rhodesia",),
    "Russian": ("Russia",),
    "South African": ("South Africa",),
    "Spanish": ("Spain",),
    "Swedish": ("Sweden",),
    "Swiss": ("Switzerland",),
    "Thai": ("Thailand",),
    "Uruguayan": ("Uruguay",),
    "Venezuelan": ("Venezuela",),
}


def feature_dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_DICTIONARY, columns=["column", "group", "meaning"])


def assign_split(year: pd.Series) -> pd.Series:
    """Chronological split. The test years are never used to choose a model."""
    split = pd.Series("excluded", index=year.index, dtype="object")
    split.loc[year.between(QUALIFYING_COMPLETE_FROM, TRAIN_LAST_YEAR)] = "train"
    split.loc[year.between(TRAIN_LAST_YEAR + 1, VALIDATION_LAST_YEAR)] = "validation"
    split.loc[year.between(VALIDATION_LAST_YEAR + 1, TEST_LAST_YEAR)] = "test"
    return split


def _countries_for(nationality: str) -> set[str]:
    return set(NATIONALITY_COUNTRIES.get(str(nationality).strip(), ()))


def add_home_race(frame: pd.DataFrame, drivers: pd.DataFrame, circuits: pd.DataFrame) -> pd.DataFrame:
    """Mark a home race from driver nationality and circuit country."""
    out = frame.copy()
    driver_nation = drivers[["driverId", "nationality"]].copy()
    driver_nation["nationality"] = driver_nation["nationality"].astype("string").str.strip()
    circuit_country = circuits[["circuitId", "country"]].copy()
    circuit_country["country"] = circuit_country["country"].astype("string").str.strip()
    out = out.merge(driver_nation, on="driverId", how="left", validate="many_to_one")
    out = out.merge(circuit_country, on="circuitId", how="left", validate="many_to_one")
    out["home_race"] = [
        int(country in _countries_for(nation))
        for nation, country in zip(out["nationality"], out["country"])
    ]
    return out


def _prior_driver_form(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["driverId", "race_date", "season_round", "raceId"]).copy()
    podium = ordered["podium"].astype(float)
    points = ordered["points"].astype(float)
    ordered["prior_points"] = points.groupby(ordered["driverId"], sort=False).shift(1)
    ordered["prior_podium"] = podium.groupby(ordered["driverId"], sort=False).shift(1)
    by_driver = ordered.groupby("driverId", sort=False)
    ordered["driver_podium_rate_last5"] = by_driver["prior_podium"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    ordered["driver_points_last5"] = by_driver["prior_points"].transform(
        lambda s: s.rolling(5, min_periods=1).sum()
    )
    ordered["career_starts"] = by_driver.cumcount()
    ordered["form_missing"] = ordered["driver_podium_rate_last5"].isna().astype(int)
    return ordered


def _prior_constructor_form(frame: pd.DataFrame) -> pd.DataFrame:
    constructor_race = (
        frame.groupby(["constructorId", "raceId"], as_index=False)
        .agg(
            podium_mean=("podium", "mean"),
            race_date=("race_date", "min"),
            season_round=("season_round", "min"),
        )
        .sort_values(["constructorId", "race_date", "season_round", "raceId"])
    )
    prior = constructor_race.groupby("constructorId", sort=False)["podium_mean"].shift(1)
    constructor_race["constructor_podium_rate_last5"] = prior.groupby(
        constructor_race["constructorId"], sort=False
    ).transform(lambda s: s.rolling(5, min_periods=1).mean())
    return frame.merge(
        constructor_race[["constructorId", "raceId", "constructor_podium_rate_last5"]],
        on=["constructorId", "raceId"],
        how="left",
        validate="many_to_one",
    )


def _entering_standings(frame: pd.DataFrame, standings: pd.DataFrame, races: pd.DataFrame, column: str, key: str) -> pd.Series:
    """Championship position after the previous round, aligned to `frame`."""
    context = races[["raceId", "year", "round"]].rename(columns={"round": "standings_round"})
    table = standings[["raceId", key, "position"]].merge(context, on="raceId", how="left")
    table = table.rename(columns={"position": column, "standings_round": "previous_round"})
    table["year"] = table["year"].astype("Int64")
    table["previous_round"] = table["previous_round"].astype("Int64")
    lookup = table[["year", "previous_round", key, column]].drop_duplicates(["year", "previous_round", key])
    merged = frame[["year", "season_round", key]].copy()
    merged["year"] = merged["year"].astype("Int64")
    merged["previous_round"] = (merged["season_round"] - 1).astype("Int64")
    merged = merged.merge(lookup, on=["year", "previous_round", key], how="left")
    return pd.Series(merged[column].to_numpy(), index=frame.index)


def _add_qualifying(frame: pd.DataFrame, qualifying: pd.DataFrame) -> pd.DataFrame:
    quali = qualifying[["raceId", "driverId", "position", "q3"]].rename(columns={"position": "quali_position"})
    out = frame.merge(quali, on=["raceId", "driverId"], how="left", validate="one_to_one")
    race_has_q3 = out.groupby("raceId")["q3"].transform(lambda s: bool(s.notna().any()))
    reached = pd.Series(np.nan, index=out.index, dtype="float64")
    reached.loc[race_has_q3] = out.loc[race_has_q3, "q3"].notna().astype(float)
    out["reached_q3"] = reached
    out["quali_position_missing"] = out["quali_position"].isna().astype(int)
    return out.drop(columns=["q3"])


def build_model_frame(driver_race: pd.DataFrame, cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per driver per race, with pre-race features and the podium label."""
    frame = driver_race.copy()
    frame["season_round"] = frame["round"].astype("Int64")
    frame = _prior_driver_form(frame)
    frame = _prior_constructor_form(frame)
    frame["driver_standing_entering"] = _entering_standings(
        frame, cleaned["driver_standings"], cleaned["races"], "driver_standing_entering", "driverId"
    )
    frame["constructor_standing_entering"] = _entering_standings(
        frame, cleaned["constructor_standings"], cleaned["races"], "constructor_standing_entering", "constructorId"
    )
    frame["standing_missing"] = frame["driver_standing_entering"].isna().astype(int)
    frame = _add_qualifying(frame, cleaned["qualifying"])
    known_grid = frame["grid"].notna()
    frame["front_row"] = pd.Series(np.nan, index=frame.index, dtype="float64")
    frame.loc[known_grid, "front_row"] = frame.loc[known_grid, "grid"].isin([1, 2]).astype(float)
    frame = add_home_race(frame, cleaned["drivers"], cleaned["circuits"])
    frame["split"] = assign_split(frame["year"])
    keep = [
        "raceId", "driverId", "year", "race_date", "race_name",
        "circuitId", "constructorId", "podium", "split", "nationality", "country",
        *FEATURE_COLUMNS,
    ]
    model = frame[keep].sort_values(["year", "season_round", "raceId", "driverId"]).reset_index(drop=True)
    leaked = [column for column in LEAKAGE_COLUMNS if column in model.columns]
    if leaked:
        raise RuntimeError(f"Model frame still has post-race columns: {leaked}")
    return model


def modeling_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows where the cutoff is observable: qualifying exists and the year is in the brief."""
    return frame.loc[frame["split"].isin(["train", "validation", "test"])].copy()


def feature_evidence(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Podium rates that motivate the feature groups. Computed on the training years only."""
    sample = frame.loc[frame["split"] == "train"].copy()
    sample["podium"] = sample["podium"].astype(int)

    def _rate(table: pd.DataFrame, key: str) -> pd.DataFrame:
        grouped = table.groupby(key, dropna=False, observed=True)["podium"].agg(starts="size", podium_rate="mean")
        return grouped.reset_index()

    grid = sample.loc[sample["grid"].notna() & (sample["grid"] > 0)].copy()
    grid["grid"] = grid["grid"].astype(int)
    grid_rate = _rate(grid, "grid")
    grid_rate = grid_rate.loc[grid_rate["starts"] >= 30]

    quali = sample.loc[sample["quali_position"].notna()].copy()
    quali["quali_position"] = quali["quali_position"].astype(int)
    quali_rate = _rate(quali, "quali_position")
    quali_rate = quali_rate.loc[quali_rate["starts"] >= 30]

    standing = sample.loc[sample["driver_standing_entering"].notna()].copy()
    standing["standing_bin"] = pd.cut(
        standing["driver_standing_entering"].astype(float),
        bins=[0, 1, 3, 10, 40],
        labels=["1", "2-3", "4-10", "11+"],
    )
    standing_rate = _rate(standing, "standing_bin")

    form = sample.loc[sample["driver_podium_rate_last5"].notna()].copy()
    form["form_bin"] = pd.cut(
        form["driver_podium_rate_last5"],
        bins=[-0.01, 0, 0.2, 0.4, 1],
        labels=["0", "0-20%", "20-40%", "40%+"],
    )
    form_rate = _rate(form, "form_bin")
    return {
        "grid": grid_rate,
        "qualifying": quali_rate,
        "standings": standing_rate,
        "form": form_rate,
    }
