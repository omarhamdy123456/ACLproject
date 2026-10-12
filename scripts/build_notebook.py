"""Generate the cleaning notebook. Re-run this only if the notebook source is rebuilt."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_data_cleaning.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip("\n")))


md("""
# Formula 1 podium prediction: data cleaning

This notebook is the cleaning and audit section of Milestone 1. The prediction, later in the project, is whether a driver finishes on the podium, using only what is known after qualifying and before the race starts. This notebook stops once that unit of analysis exists: **one row per driver per race**, with the label stored separately from anything a model is allowed to see.

It does not build form or standings features, and it does not train a model.
""")

md("""
## 1. Setup

Raw CSVs are read and never written. Every cleaned table is a new file. The rules themselves live in `src/f1_clean.py`, and `scripts/test_cleaning.py` checks them.

The 14 tables named in the brief are `circuits`, `constructors`, `constructor_results`, `constructor_standings`, `drivers`, `driver_standings`, `races`, `results`, `qualifying`, `lap_times`, `pit_stops`, `sprint_results`, `seasons`, and `status`. They are the Ergast extracts distributed as the Kaggle dataset *Formula 1 World Championship (1950–2024)*. This copy was downloaded on 6 October 2026 from the TracingInsights RaceData mirror of that dataset (CC0):

`https://github.com/TracingInsights/RaceData/releases/latest/download/data.zip`

The zip also contains safety-car and red-flag files. Those are outside the brief, so they were moved to `data/not_in_brief` and are not loaded. The extract on disk includes completed races after 2024. Those races are kept and flagged. They are not deleted.
""")

code("""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if not (ROOT / "src" / "f1_clean.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.f1_clean import (
    ASSIGNMENT_LAST_YEAR,
    SENTINEL,
    clean_tables,
    collapse_entries,
    load_raw,
    project_paths,
    raw_hashes,
    replace_sentinels,
)

PATHS = project_paths(ROOT)
PATHS["figures"].mkdir(parents=True, exist_ok=True)
HASHES_AT_START = raw_hashes(PATHS["raw"])

pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 40)
pd.set_option("display.width", 120)

NAVY = "#1f4e79"
CLAY = "#c46b2c"
TEAL = "#2f6f6a"


def styled(ax, title, xlabel, ylabel, grid="y"):
    ax.set_title(title, loc="left", fontsize=13, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid, color="#e6e6e6")
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PATHS["figures"] / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.show()


raw = load_raw(PATHS["raw"])
print(f"Loaded {len(raw)} tables from {PATHS['raw']}")
""")

md("""
## 2. What is in the files

The unit the model will eventually score is a driver entering a race. `results` is that spine: one recorded result per car a driver drove. `races`, `drivers`, `constructors`, `circuits`, and `status` are the lookup tables. `qualifying` is the pre-race session. Lap times, pit stops, sprint results, and both standings tables are part of the relational extract, so they are audited here too. Whether any of them is legal as a feature is decided by the pre-race cutoff, not by the fact that the column exists.
""")

code("""
inventory = pd.DataFrame([
    {"table": name, "rows": len(frame), "columns": len(frame.columns)}
    for name, frame in raw.items()
])
display(inventory)

races_raw = replace_sentinels(raw["races"]).copy()
races_raw["year"] = pd.to_numeric(races_raw["year"])
print(
    f"Seasons in races: {int(races_raw['year'].min())}–{int(races_raw['year'].max())}. "
    f"Distinct seasons: {races_raw['year'].nunique()}."
)
print(f"Assignment window named in the brief: 1950–{ASSIGNMENT_LAST_YEAR}.")
""")

md("""
The printed year range is the range in this download. The brief names 1950–2024. Seasons after 2024 stay in the cleaned tables, and each driver-race row carries `in_assignment_window` so a later split can follow the brief without rebuilding the file.
""")

code("""
results_year = replace_sentinels(raw["results"])[["raceId"]].merge(
    races_raw[["raceId", "year"]], on="raceId", how="left"
)
volume = pd.DataFrame({
    "races": races_raw.groupby("year").size(),
    "result_rows": results_year.groupby("year").size(),
}).fillna(0)

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(volume.index, volume["races"], color=TEAL, linewidth=2, label="Races")
ax.plot(volume.index, volume["result_rows"], color=NAVY, linewidth=2, label="Result rows")
styled(ax, "Race calendar and result rows by season", "Season", "Count")
ax.legend(frameon=False)
save(fig, "01_season_volume.png")
""")

md("""
The calendar is not a fixed sample. The number of races and the size of the field both change by season, and the early championship includes the Indianapolis 500, which had a much larger grid than a modern Grand Prix. A single podium rate for 1950–2024 mixes those eras. The decade chart after cleaning separates them.
""")

md("""
## 3. Sentinel values, before they are converted

Ergast writes a missing value as the two characters `\\N`. Pandas does not treat that as missing, so a numeric column full of `\\N` still looks populated. The first audit counts those markers on the raw strings.

One column does not fit the usual reading. In `constructor_results.status`, `\\N` means the constructor was not disqualified. The only other value is `D`. Converting `\\N` to “unknown” would make almost every constructor result look incomplete. That column is mapped to `disqualified`, which is true only for `D`.
""")

code("""
from src.f1_clean import summarize_sentinels

sentinels = summarize_sentinels(raw)
display(sentinels)

status_counts = raw["constructor_results"]["status"].value_counts()
print("constructor_results.status raw values:")
display(status_counts.to_frame("rows"))

plot_df = sentinels.sort_values("sentinel_share")
fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(plot_df))))
ax.barh(
    plot_df["table"] + "." + plot_df["column"],
    plot_df["sentinel_share"] * 100,
    color=NAVY,
)
styled(
    ax,
    "Share of cells holding the \\\\N sentinel",
    "Percent of rows",
    "",
    grid="x",
)
save(fig, "02_sentinel_share.png")
""")

md("""
The tall bars are structural, not scattered holes. Qualifying Q2 and Q3 are empty when a driver is eliminated in the previous session. Session dates and times on `races` are empty for most of history because the file only records them for recent seasons. `results.position` is empty when the driver was not classified; `positionOrder` is filled for every row and is the column the brief uses for the label. Driver numbers and three-letter codes are empty for older drivers because those identifiers did not exist yet. That last pattern is checked on its own below, by decade of debut.
""")

code("""
results_status = replace_sentinels(raw["results"]).merge(
    replace_sentinels(raw["status"]), on="statusId", how="left"
)
top_status = results_status["status"].value_counts().head(12).sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top_status.index, top_status.values, color=NAVY)
styled(ax, "Twelve most common finishing statuses", "Result rows", "", grid="x")
save(fig, "03_status_counts.png")
print(f"Distinct statuses used in results: {results_status['status'].nunique()}")
""")

md("""
A large share of result rows are not `Finished`. For those rows the official position is empty and the status says why the car stopped. `positionOrder` still ranks them behind the classified finishers. That is a property of the label column. It is also a reminder that status, laps, and position are known only after the race, so they stay out of the feature set.
""")

md("""
## 4. Keys before cleaning

A primary key must be unique and non-null. A foreign key may be null, and every non-null value must exist in the parent table. Grain checks ask a separate question: does one driver appear once per race?
""")

code("""
from src.f1_clean import audit_keys

keys_before = audit_keys(raw)
failed_before = keys_before.loc[~keys_before["passed"]].copy()
print(f"{int(keys_before['passed'].sum())} of {len(keys_before)} checks passed before cleaning.")
display(failed_before)
""")

md("""
Three checks fail, and they fail for different reasons.

- The lap-time key `(raceId, driverId, lap)` is not unique. Some laps are stored twice. That is a broken key, so the cleaned lap-time table keeps one row and writes the discarded copy to an audit file. Lap times are measured during the race. Repairing the key does not make them legal features.
- `(raceId, driverId)` is not unique in `results`. Section 5 shows that these rows are real repeated entries, mostly shared drives, so they stay in `results`. The modeling table is a new table that keeps one of them.
- Two sprint rows have `\\N` in `statusId`. That is a sentinel, not a status that is missing from `status`. After conversion they are null foreign keys, which is allowed. No real status id points at a missing parent.
""")

md("""
## 5. Repeated driver-race rows

The brief asks for one row per `(raceId, driverId)`, and it says the early duplicates come from shared drives. The count by season is the check on that claim.
""")

code("""
results_view = replace_sentinels(raw["results"]).copy()
for column in ["grid", "positionOrder", "laps", "points"]:
    results_view[column] = pd.to_numeric(results_view[column])

drivers_view = replace_sentinels(raw["drivers"]).copy()
drivers_view["driver"] = drivers_view["forename"] + " " + drivers_view["surname"]

repeated = results_view.loc[results_view.duplicated(["raceId", "driverId"], keep=False)].copy()
repeated = repeated.merge(races_raw[["raceId", "year", "name"]], on="raceId", how="left")
repeated = repeated.merge(drivers_view[["driverId", "driver"]], on="driverId", how="left")
pairs = repeated.drop_duplicates(["raceId", "driverId"])

print(f"Repeated result rows: {len(repeated)}")
print(f"Driver-race pairs involved: {len(pairs)}")
print(f"Seasons involved: {int(pairs['year'].min())}–{int(pairs['year'].max())}")

year_counts = pairs.groupby("year").size().reindex(range(1950, 1979), fill_value=0)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(year_counts.index, year_counts.values, color=NAVY, width=0.8)
styled(ax, "Driver-race pairs that appear more than once", "Season", "Pairs")
save(fig, "04_duplicate_pairs_by_year.png")
""")

md("""
The repeated keys are concentrated in the 1950s. The chart stops at 1978 because that is the last season with any repeated pair. Modern seasons already have one row per driver per race.

The 1951 French Grand Prix is a shared drive: the same car number is attached to two drivers, and each of those drivers also has a second row. Both rows are real. They are two stints, not a corrupted copy of one stint.
""")

code("""
french_id = races_raw.loc[
    (races_raw["year"] == 1951) & (races_raw["name"] == "French Grand Prix"),
    "raceId",
].iloc[0]
french = repeated.loc[repeated["raceId"] == french_id].copy()
shared_numbers = french.groupby("number")["driverId"].nunique()
shared_numbers = shared_numbers[shared_numbers > 1].index
french_shared = french.loc[french["number"].isin(shared_numbers), [
    "driver", "number", "grid", "positionText", "positionOrder", "laps", "points",
]].sort_values(["number", "positionOrder", "driver"])
display(french_shared)
""")

md("""
Take car 8. Two drivers are both recorded on that car, with the same grid, the same classified position, and the same lap count, while the points are split. Each of those drivers also has another row on a different car. Keeping both rows would train the model on two outcomes for one driver in one race. The modeling table has to choose one row per driver. The rule is in Section 8. Both of the rules considered there keep the classified result for each of these drivers, so the podium label is the same either way.
""")

code("""
late = repeated.loc[repeated["year"] >= 1965, [
    "year", "name", "driver", "constructorId", "grid", "positionText", "positionOrder", "laps",
]].sort_values(["year", "driver", "positionOrder"])
print(f"Repeated rows after 1964: {len(late)}")
display(late)

status_view = replace_sentinels(raw["status"])
late_status = repeated.loc[repeated["year"] >= 1965, ["statusId"]].merge(status_view, on="statusId")
print("Statuses on those rows:")
display(late_status["status"].value_counts().to_frame("rows"))
""")

md("""
The only repeated pair after 1964 is the 1978 Italian Grand Prix. It is not a shared drive that completed laps. The same driver has two entries, both on grid 0 with zero laps, and the statuses are “Did not qualify” and “Did not prequalify”. The collapse still has to keep one row. Neither row is a podium, so the label does not depend on which zero-lap row is kept.
""")

md("""
## 6. What “podium” means in this file

The brief defines the label as `positionOrder <= 3`. `positionOrder` is a dense rank of every car in the result, including cars that did not finish. `positionText` is the official classification, and it uses letters (`R`, `D`, `W`, `F`) when the driver was not classified. The check below asks whether those two columns tell the same podium story.
""")

code("""
results_view["brief_podium"] = results_view["positionOrder"] <= 3
results_view["official_podium"] = results_view["positionText"].isin(["1", "2", "3"])
brief_only = results_view["brief_podium"] & ~results_view["official_podium"]
official_only = results_view["official_podium"] & ~results_view["brief_podium"]
print(f"Brief podium rows: {int(results_view['brief_podium'].sum())}")
print(f"Official podium rows: {int(results_view['official_podium'].sum())}")
print(f"In the brief label only: {int(brief_only.sum())}")
print(f"In the official label only: {int(official_only.sum())}")

conflict_ids = results_view.loc[brief_only, "raceId"].unique()
conflict_race = results_view.loc[results_view["raceId"].isin(conflict_ids)].copy()
conflict_race = conflict_race.merge(drivers_view[["driverId", "driver"]], on="driverId", how="left")
conflict_race = conflict_race.merge(races_raw[["raceId", "year", "name"]], on="raceId", how="left")
conflict_race = conflict_race.merge(status_view, on="statusId", how="left")
show_cols = ["year", "name", "driver", "grid", "positionText", "positionOrder", "points", "status"]
display(
    conflict_race.sort_values(["raceId", "positionOrder"])[show_cols]
    .head(8)
)
""")

md("""
On this extract the two definitions disagree on one row, in the 1983 Brazilian Grand Prix. The official position text goes 1, then 3, then 4. There is no official 2nd place in the file. `positionOrder` has no gap, so the driver whose official text is 4 receives `positionOrder` 3. The points move with the official text: 9, then 4, then 3, skipping the step that the missing place would have scored. Those cars are classified as `Finished`. This is not a retirement that was ranked third.

The stored label follows the brief: `podium` is `positionOrder <= 3`. `official_podium` keeps the text-based version, and `podium_definition_conflict` marks the disagreement. One row is not a reason to replace the brief’s definition. It is a reason to be able to point at it.
""")

md("""
## 7. Coverage the later feature work has to respect

Three coverage facts are properties of the extract. Cleaning does not impute them.
""")

code("""
result_race_ids = set(results_view["raceId"])
unraced = races_raw.loc[~races_raw["raceId"].isin(result_race_ids), ["year", "round", "name", "date"]]
unraced = unraced.sort_values(["year", "round"])
print(f"Scheduled races with no result rows: {len(unraced)}")
display(unraced)

quali_ids = set(replace_sentinels(raw["qualifying"])["raceId"])
completed = races_raw.loc[races_raw["raceId"].isin(result_race_ids)].copy()
completed["has_qualifying"] = completed["raceId"].isin(quali_ids)
coverage = completed.groupby("year")["has_qualifying"].mean()
print(f"First season present in qualifying: {int(coverage[coverage > 0].index.min())}")

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(coverage.index, coverage.values * 100, color=NAVY, linewidth=2)
ax.axvspan(1994, 2002.5, color="#f3e3d3", zorder=0)
styled(ax, "Completed races that have a qualifying table", "Season", "Percent of races")
ax.set_ylim(-5, 105)
save(fig, "05_qualifying_coverage.png")
""")

md("""
Unstarted races have a row in `races` and no rows in `results`. They never enter the driver-race table, because there is no entry to label. On this download those rows are the remaining 2026 rounds.

Qualifying coverage has three regimes, and the shaded band is the middle one.

- Before 1994 the qualifying table has no races. A missing qualifying row in this era means the source has not started yet.
- From 1994 through 2002 the table exists and many races are absent. A missing row in this window is a hole in the extract, not a driver who failed to set a time.
- From 2003 onward, every completed race in this file has qualifying rows. The unstarted 2026 races are excluded from the share, so a race that has not been held is not counted as missing qualifying.

The null grids are a separate hole. Grid 0 is a real recorded value: the driver had no grid slot. A null grid means the file omitted the number. The races below have a null grid for the whole field. Those nulls are left as null. They are not filled from qualifying, because qualifying position and the starting grid come apart when a penalty is applied, and that distinction belongs to a later question.
""")

code("""
grid_null = results_view.loc[results_view["grid"].isna(), ["raceId"]].merge(
    races_raw[["raceId", "year", "name", "date"]], on="raceId"
)
print("Rows with a null grid, by race:")
display(grid_null.groupby(["year", "name", "date"]).size().rename("null_grid_rows").reset_index())
print(f"Rows with grid 0: {int((results_view['grid'] == 0).sum())}")

debut = results_view.merge(races_raw[["raceId", "year"]], on="raceId").groupby("driverId")["year"].min()
driver_codes = drivers_view.merge(debut.rename("debut"), left_on="driverId", right_index=True, how="inner")
driver_codes["decade"] = (driver_codes["debut"] // 10 * 10).astype(int)
driver_codes["code_missing"] = driver_codes["code"].isna()
driver_codes["number_missing"] = driver_codes["number"].isna()
code_by_decade = driver_codes.groupby("decade").agg(
    drivers=("driverId", "size"),
    code_missing=("code_missing", "mean"),
    number_missing=("number_missing", "mean"),
)
code_by_decade[["code_missing", "number_missing"]] = code_by_decade[["code_missing", "number_missing"]] * 100
display(code_by_decade.round(1))

fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(len(code_by_decade))
width = 0.38
ax.bar(x - width / 2, code_by_decade["code_missing"], width=width, color=NAVY, label="3-letter code missing")
ax.bar(x + width / 2, code_by_decade["number_missing"], width=width, color=CLAY, label="Permanent number missing")
styled(ax, "Missing modern driver identifiers by decade of debut", "Decade of first result", "Percent of drivers")
ax.set_xticks(x, [str(int(year)) + "s" for year in code_by_decade.index], rotation=0)
ax.set_ylim(0, 105)
ax.legend(frameon=False)
save(fig, "06_driver_identifier_missingness.png")
""")

md("""
Permanent numbers and three-letter codes are modern identifiers. Their absence on a driver who debuted in the 1950s is the era of the record, not a failed join. Those drivers stay in the table. The fields stay null.

The same reading applies to the standings tables. A handful of rows have no numeric position and `positionText` equal to `-`, with zero points, at the opening races of recent seasons. That marker means unranked. It is left as null. Standings of the current race are post-race information anyway, so this notebook does not turn them into features.
""")

code("""
standings = replace_sentinels(raw["driver_standings"]).copy()
standings["points"] = pd.to_numeric(standings["points"])
unranked = standings.loc[standings["position"].isna()].merge(
    races_raw[["raceId", "year", "name"]], on="raceId"
)
print(f"Driver-standings rows with no position: {len(unranked)}")
display(
    unranked.groupby(["year", "name", "positionText"])
    .size()
    .rename("rows")
    .reset_index()
)
""")

md("""
## 8. The cleaning rules

Applied to copies, in this order:

1. Replace `\\N` with missing values. Row counts of the source tables stay the same.
2. Cast identifiers and counts to integers, points and coordinates to floats, and calendar dates to dates. A value that is present but cannot be cast is recorded as a coercion failure. The standard for this extract is zero failures.
3. Parse duration strings (`ss.sss`, `m:ss.sss`, `h:mm:ss.sss`) into seconds. Qualifying segments, lap times, pit durations, and fastest laps use this parser.
4. Split `results.time` into two measurements. A string with no leading `+` is the winner’s elapsed time. A string with a leading `+` is the gap to the winner. `milliseconds` is elapsed time for that car when the source has it, including cars whose `time` string is only a gap. Those three numbers are stored in separate columns.
5. Where `races.date` and `races.time` are both present, combine them into `race_start`.
6. Map `constructor_results.status` to `disqualified`.
7. Mark `results.classified` when the official position is present.
8. Repair lap times so `(raceId, driverId, lap)` is unique. Exact copies are dropped. When the two copies disagree, the faster time is kept and both copies are written to the audit file.
9. Build `driver_race` with one row per `(raceId, driverId)`. The source `results` table keeps every stint.

The collapse rule is `best_classification`: lowest `positionOrder`, then most laps, then the smallest `resultId`. The label is the driver’s finishing result, so the classified result wins over a longer retirement. `most_laps` is computed on the same pairs as a check. It is not the stored rule.
""")

code("""
result = clean_tables(raw)

print("Coercion failures (empty means every present value cast successfully):")
display(result.coercion_failures)

print("Parsed strings against the millisecond columns already in the file:")
display(result.time_checks)

print(f"Disqualified constructor results: {int(result.cleaned['constructor_results']['disqualified'].sum())}")

print("Missing-value method for every column that had a blank marker:")
display(result.missing_policy)
left_blank = result.missing_policy.loc[result.missing_policy["method"] == "leave missing", "values_filled"]
assert left_blank.eq(0).all()
""")

md("""
Zero coercion failures means the type conversion did not silently turn bad strings into missing numbers. The time check is the other half of that claim. The winner’s clock, the lap-time string, and the pit-stop string match the millisecond columns to the nearest millisecond. The parsed seconds are the same measurement as the integer the file already stored.

`disqualified` is true for 17 constructor results, which is the number of raw `D` values. Every former `\\N` in that column is false, because that marker meant “not disqualified”.

The table above is the missing-value decision for every column that contained `\\N`. Mean, median, and mode are not used. `leave missing` keeps the blank, and `values_filled` is 0 on those rows, so no average was written in. The one coded column is `constructor_results.status`: the blank marker is stored as `disqualified = False`.
""")

code("""
q1 = result.cleaned["qualifying"][["raceId", "driverId", "q1", "q1_seconds"]].dropna(subset=["q1_seconds"])
ordinary = q1.loc[q1["q1_seconds"] <= 180, "q1_seconds"]
odd = q1.loc[q1["q1_seconds"] > 180].merge(
    result.cleaned["races"][["raceId", "year", "name"]], on="raceId"
).merge(
    result.cleaned["drivers"][["driverId", "forename", "surname"]], on="driverId"
)
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.hist(ordinary, bins=40, color=NAVY)
styled(
    ax,
    f"Parsed Q1 times, excluding {len(odd)} value above 3 minutes. Median {ordinary.median():.1f}s",
    "Q1 (seconds)",
    "Qualifying rows",
)
save(fig, "07_q1_seconds.png")
print("Qualifying times longer than 3 minutes, left as parsed from the source string:")
display(odd[["year", "name", "forename", "surname", "q1", "q1_seconds"]])

sample_times = result.cleaned["results"].loc[
    result.cleaned["results"]["winner_time_seconds"].notna()
    | result.cleaned["results"]["gap_seconds"].notna(),
    ["time", "winner_time_seconds", "gap_seconds", "elapsed_seconds"],
].head(6)
display(sample_times)
""")

md("""
A Q1 string such as `1:26.572` is now about 86.6 seconds. The mass of the histogram sits on real lap times: the median is 87 seconds. One source string does not. Mark Blundell’s 1995 Japanese Grand Prix Q1 is stored as `16:42.640`, which is 1002.6 seconds. The parser is right about what the characters say, and the value is left unchanged, because replacing it with a guessed lap time would be inventing data. It is excluded from the histogram only so the other times stay readable. The sample of `results.time` shows the split: a clock with no `+` fills `winner_time_seconds`, and a clock with `+` fills `gap_seconds`. `elapsed_seconds` comes from `milliseconds` and is populated for both kinds of row when the source has it.
""")

code("""
keys_after = result.keys_after
failed_after = keys_after.loc[~keys_after["passed"]].copy()
print(f"{int(keys_after['passed'].sum())} of {len(keys_after)} checks passed after cleaning.")
display(failed_after)
print("Lap-time key repair:")
display(result.lap_conflict_summary)
""")

md("""
The only check that still fails is the `(raceId, driverId)` grain of the source `results` table. That failure is deliberate. Shared-drive stints remain in `results` so the choice of row can be audited. The new `driver_race` table is the unique grain, and its own key check passes.

The lap-time repair is summarized above. Most repeated laps were exact copies or a one-millisecond disagreement. A smaller set differs by more than one millisecond. The cleaned table keeps the faster time. Every conflicting copy is in `data/audit/lap_time_conflicts.csv`.
""")

md("""
## 9. Which row is kept

`best_classification` and `most_laps` were both run on every repeated pair. The question is whether the stored label, and the stored grid and constructor, depend on that choice.
""")

code("""
comparison = result.rule_comparison
print(f"Repeated pairs compared: {len(comparison)}")
print(f"Same resultId: {int(comparison['same_row'].sum())}")
print(f"Same positionOrder: {int(comparison['same_position'].sum())}")
print(f"Same podium label: {int(comparison['same_podium'].sum())}")
print(f"Disagreements: {int((~comparison['same_row']).sum())}")

toy = pd.DataFrame([
    {"raceId": 1, "driverId": 7, "resultId": 10, "positionOrder": 8, "laps": 70},
    {"raceId": 1, "driverId": 7, "resultId": 11, "positionOrder": 2, "laps": 12},
])
toy_compare = pd.DataFrame({
    "rule": ["best_classification", "most_laps"],
    "kept_resultId": [
        int(collapse_entries(toy, "best_classification").iloc[0]["resultId"]),
        int(collapse_entries(toy, "most_laps").iloc[0]["resultId"]),
    ],
})
display(toy)
display(toy_compare)
""")

md("""
On this extract the two rules keep the same `resultId` for all 85 repeated pairs, so they also keep the same podium label, the same grid, and the same constructor. The agreement is a property of these rows. It is not a property of the rules.

The toy pair is a short podium stint and a long retirement. `best_classification` keeps the podium. `most_laps` keeps the retirement. The label this project predicts is the podium, so the stored rule is `best_classification` even though the two rules happen to agree on the real pairs. If a later refresh of the extract contains a pair like the toy, the label will follow the classified result.
""")

md("""
## 10. The driver-race table, and what cleaning changed

`driver_race` has one row per driver per race. `shared_entry` marks the  pairs that were collapsed. `n_constructors` is greater than 1 when the stints used different cars. For those rows, `constructorId` is the constructor on the kept stint, which is the better classified car. It is not safe to read it as “the car they qualified” without looking at the flag.
""")

code("""
driver_race = result.driver_race
print(f"Result rows: {len(result.cleaned['results'])}")
print(f"Driver-race rows: {len(driver_race)}")
print(f"Rows removed by the collapse: {len(result.cleaned['results']) - len(driver_race)}")
print(f"Unique (raceId, driverId): {not driver_race.duplicated(['raceId', 'driverId']).any()}")
print(f"Shared entries: {int(driver_race['shared_entry'].sum())}")
print(f"Shared entries with more than one constructor: {int((driver_race['n_constructors'] > 1).sum())}")
print(f"Podium definition conflicts: {int(driver_race['podium_definition_conflict'].sum())}")
print(
    f"Rows inside the brief's 1950–{ASSIGNMENT_LAST_YEAR} window: "
    f"{int(driver_race['in_assignment_window'].sum())} of {len(driver_race)}"
)

before = results_view[["raceId", "positionOrder"]].merge(
    races_raw[["raceId", "year"]], on="raceId", how="left"
)
before["podium"] = before["positionOrder"] <= 3
before["decade"] = (before["year"] // 10 * 10).astype(int)
after = driver_race[["raceId", "year", "podium"]].copy()
after["decade"] = (after["year"] // 10 * 10).astype(int)

rate = pd.DataFrame({
    "before": before.groupby("decade")["podium"].mean() * 100,
    "after": after.groupby("decade")["podium"].mean() * 100,
})
field = (
    after.groupby(["decade", "raceId"]).size()
    .groupby("decade").median()
    .rename("median_starters")
)
display(rate.round(2).join(field))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
x = np.arange(len(rate))
width = 0.38
axes[0].bar(x - width / 2, rate["before"], width=width, color="#9bb4c8", label="Before collapse")
axes[0].bar(x + width / 2, rate["after"], width=width, color=NAVY, label="After collapse")
styled(axes[0], "Podium rate by decade", "Decade", "Percent of rows")
axes[0].set_xticks(x, [str(int(year)) + "s" for year in rate.index], rotation=45)
axes[0].legend(frameon=False)

axes[1].bar(x, field.values, color=TEAL)
styled(axes[1], "Median starters per race", "Decade", "Drivers")
axes[1].set_xticks(x, [str(int(year)) + "s" for year in field.index], rotation=45)
save(fig, "08_podium_rate_before_after.png")
""")

md("""
The collapse removes 91 result rows, all of them from the repeated pairs. In the 1950s the podium rate moves from 13.6% to 14.1%, because a driver who had both a classified stint and a retirement stint is now only the classified row. From the 1970s onward the rates match at the precision in the table. Modern seasons were already one row per driver.

The rate that remains still depends on the era, and it tracks field size rather than “early versus modern”. The 1980s have the largest median field, 28 starters, and the lowest podium rate, 10.4%. The 2020s have a median field of 20 and the highest rate, 14.8%. Three podium places are a larger share of a shorter field.
""")

code("""
balance = driver_race["podium"].value_counts().rename({False: "Not podium", True: "Podium"})
balance_share = driver_race["podium"].mean()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(balance.index, balance.values, color=[NAVY, CLAY])
styled(ax, f"Class balance on driver_race ({balance_share * 100:.1f}% podium)", "", "Rows")
save(fig, "09_class_balance.png")

multi_podium = driver_race.groupby("raceId")["podium"].sum()
print("Races by how many drivers carry the podium label:")
display(multi_podium.value_counts().sort_index().rename("races").to_frame())
""")

md("""
Podiums are 12.8% of the cleaned driver-race rows. That share is higher in shorter fields and lower in longer ones, so the overall percentage is not a constant the model can lean on. A few races have more than three drivers with the podium label. That is what a shared classified result looks like after the collapse: two drivers can both be credited with the same finishing position, and both rows are kept because they are different drivers.
""")

code("""
grid = driver_race.dropna(subset=["grid"]).copy()
grid["grid"] = grid["grid"].astype(int)
grid_rates = grid.groupby("grid")["podium"].agg(rate="mean", starts="size")
plotted = grid_rates.loc[grid_rates["starts"] >= 50].copy()
omitted = int(grid_rates.loc[grid_rates["starts"] < 50, "starts"].sum())

fig, ax = plt.subplots(figsize=(11, 4.4))
colors = [CLAY if grid_pos == 0 else NAVY for grid_pos in plotted.index]
ax.bar(plotted.index.astype(str), plotted["rate"] * 100, color=colors)
styled(ax, "Podium rate by starting grid", "Grid", "Percent of starts")
save(fig, "10_podium_rate_by_grid.png")
grid_zero = driver_race.loc[driver_race["grid"] == 0, "podium"]
print(f"Grid 0 podium rate: {grid_zero.mean():.1%} on {len(grid_zero)} starts")
print(f"Starts on grids with fewer than 50 starts, omitted from the chart: {omitted}")
print(f"Starts with a null grid, also omitted: {int(driver_race['grid'].isna().sum())}")
""")

md("""
Grid 0 is drawn in clay, and on this extract its bar has no height: 1,635 starts with no grid slot produced no podiums. It is not the back of the grid. It is a recorded “no grid slot”, and those entries did not finish on the podium. From grid 1, the rate falls as the starting position gets worse. Grids with fewer than 50 starts, mostly the long Indianapolis fields of the 1950s, are omitted because a rate from a handful of starts is not a stable pattern. Null grids are omitted too; they were listed with the two races whose whole field is missing a grid. This chart is the overall association only. It is not the circuit question, which has to separate qualifying position from the starting grid and to compare the seasons before and after 2014.
""")

md("""
## 11. Handoff

Every column on `driver_race` has a role.

- `identifier` and `pre_race_context` are known when the race is scheduled, or they are on the entry that was kept. Grid, car number, and constructor are in this group, with the shared-entry caveat above.
- `post_race_label` is known only after the race. These columns exist so the label can be audited. They are not features.
- `audit` records the collapse and the two label definitions.

Current-race lap times, pit stops, fastest laps, race time, points, status, finishing position, and the standings published after the race are post-race information as well. They are cleaned in the source tables so the keys are trustworthy. They are not on the feature side of this handoff. Qualifying is pre-race where a qualifying row exists, and it is intentionally not joined yet.
""")

code("""
from src.f1_clean import DRIVER_RACE_DICTIONARY, save_outputs

dictionary = pd.DataFrame(DRIVER_RACE_DICTIONARY, columns=["column", "role", "meaning"])
display(dictionary)

kept_example = result.duplicate_rows.loc[result.duplicate_rows["raceId"] == int(french_id)].copy()
name_map = (
    drivers_view.assign(driverId=pd.to_numeric(drivers_view["driverId"]))
    .set_index("driverId")["driver"]
)
kept_example["driver"] = kept_example["driverId"].map(name_map)
shared_number_ids = pd.to_numeric(pd.Index(shared_numbers))
display(
    kept_example.loc[kept_example["number"].isin(shared_number_ids), [
        "driver", "number", "grid", "positionText", "positionOrder", "laps", "kept",
    ]].sort_values(["number", "positionOrder", "driver"])
)

assert result.coercion_failures.empty
time_gap = result.time_checks.loc[~(result.time_checks["max_abs_seconds"] < 0.01)]
if time_gap.empty:
    print("Parsed times match the millisecond columns within 0.01 seconds.")
else:
    print("These source columns disagree with the millisecond values by 0.01 seconds or more. Both numbers are kept:")
    display(time_gap)
assert not driver_race.duplicated(["raceId", "driverId"]).any()
assert not result.cleaned["lap_times"].duplicated(["raceId", "driverId", "lap"]).any()
assert HASHES_AT_START == raw_hashes(PATHS["raw"])

save_outputs(result, PATHS)
print("Wrote cleaned tables, driver_race, and audit files.")
print(f"Raw files unchanged: {HASHES_AT_START == raw_hashes(PATHS['raw'])}")
""")

md("""
The `kept` flag on the 1951 French Grand Prix rows is the rule applied to the shared cars from Section 5. For each driver, the classified stint is kept and the other stint stays in `results` and in the duplicate audit, with `kept` false.

Re-run this notebook from the project folder with Run All. It reads `data/raw`, rewrites `data/cleaned`, `data/processed`, `data/audit`, and `figures`, and leaves the raw CSVs byte-for-byte as they were.
""")

nb.cells = cells
from strip_descriptions import keep_headings
keep_headings(nb)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
