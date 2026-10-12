"""Append the three questions and the pre-race feature table to the cleaning notebook."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "01_data_cleaning.ipynb"
OUT = ROOT / "notebooks" / "milestone1.ipynb"

nb = nbf.read(SOURCE, as_version=4)
nb.cells[0].source = "# Formula 1 podium prediction"


def md(source: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(source.strip("\n")))


md("""
## 12. Features that exist before the race

The modeling table keeps the cleaning grain: one row per `(raceId, driverId)`. The years in the model are 2003–2024. From 2003 onward every completed race in this file has qualifying, so the cutoff is actually in the data. Earlier seasons stay in the cleaned tables and are marked `excluded` on this frame.

Each feature is built so that the current race cannot leak in.

- Grid and front row are the starting slot after penalties. Grid 0 stays 0. A blank grid stays blank.
- Qualifying position is this race's qualifying result. `reached_q3` is blank for a whole race that did not run Q3, so those seasons are not coded as eliminations.
- Form uses the previous five starts. The points and the podium of the row being predicted are not in that window.
- Entering standings are the championship table after the previous round. The standings row published for this race is not used. Round 1 is blank.
- Home race matches the driver's nationality to the circuit country. A dual national matches either country. East German nationality is matched to German circuits because the file has no separate country.

Same-weekend sprint points are left out. In the current format they are known before the race, but they live in a results table for the same event. Leaving them out is the reading that cannot be accused of using the race itself.
""")

code("""
from src.f1_features import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    build_model_frame,
    feature_dictionary_frame,
    feature_evidence,
    modeling_sample,
)
from src.f1_questions import plot_feature_evidence

model = build_model_frame(result.driver_race, result.cleaned)
sample = modeling_sample(model)
display(feature_dictionary_frame())
print("Rows in the modeling window:", len(sample))
print(sample["split"].value_counts().to_string())
assert not any(column in model.columns for column in LEAKAGE_COLUMNS)
assert not sample.duplicated(["raceId", "driverId"]).any()

evidence = feature_evidence(model)
for name, table in evidence.items():
    print(name)
    display(table)
save(plot_feature_evidence(evidence), "11_feature_evidence.png")
model.to_csv(PATHS["processed"] / "model_frame.csv", index=False)
feature_dictionary_frame().to_csv(PATHS["processed"] / "model_feature_dictionary.csv", index=False)
print("Wrote model_frame.csv")
""")

md("""
The training-year plots are the reason these groups are in the model. The podium rate falls as the grid number gets worse, and it falls the same way as qualifying position gets worse. Drivers who arrive in the lead of the championship, and drivers whose previous five starts were often podiums, finish on the podium more often. Those are associations at the cutoff. They are not causes.
""")

md("""
## 13. Which circuits convert a front-row start into a podium?

Front row means grid 1 or 2, the slot the car actually starts from. Qualifying position is kept beside it because a penalty can move a car back after qualifying. A circuit needs at least 20 front-row starts in an era. The join is the driver-race table, then `circuits`, then `qualifying`.
""")

code("""
from src.f1_questions import (
    front_row_conversion,
    home_advantage,
    mechanical_retirements,
    narrate_front_row,
    narrate_home,
    narrate_mechanical,
    plot_front_row,
    plot_home,
    plot_mechanical,
    status_audit,
)

ranked = front_row_conversion(result.driver_race, result.cleaned["circuits"], result.cleaned["qualifying"])
print(narrate_front_row(ranked))
display(ranked["pre_2014"].head(8))
display(ranked["from_2014"].head(8))
display(ranked["rank_change"].head(8))
display(ranked["qualifying_versus_grid"])
save(plot_front_row(ranked), "12_front_row_conversion.png")
""")

md("""
The ranking uses the grid, not the qualifying position. The table under the plot counts drivers who qualified on the front row and started further back. Their podium rate is much lower than the drivers who qualified there and started there. That gap is why a circuit ranking built from qualifying would answer a different question.
""")

md("""
## 14. Do drivers podium more often at home once the start slot is held fixed?

The raw home rate would be misleading if home drivers also start further forward. The comparison is therefore inside a grid band: pole, rows 2–3, grid 4–10, and grid 11 or worse. Blank grids and grid 0 are excluded because there is no starting slot to control for. The join is the driver-race table, `drivers`, and `circuits`.
""")

code("""
home = home_advantage(result.driver_race, result.cleaned["drivers"], result.cleaned["circuits"])
print(narrate_home(home))
display(home)
save(plot_home(home), "13_home_by_grid.png")
""")

md("""
Inside a grid band the home rate is not higher in a way that survives the sample size. From pole, the home rate is lower, on only about ninety home poles. From the rest of the grid the two rates sit within a few points. `home_race` stays in the model so the ablation can measure it. The plot is not a claim that the crowd causes the finish.
""")

md("""
## 15. Which constructors retire for mechanical reasons?

A mechanical retirement is a start whose status names a failed component. The denominator is a race start. Cars that did not start are removed. Classified finishes, crashes, driver causes, regulation decisions, and a generic `Retired` are not in the numerator. Tyres, punctures, and wings count only when the status names that part rather than a collision. The join is the driver-race table, `status`, and `constructors`. Constructors need at least 30 starts in an era, and the comparison keeps those that clear the bar in both 2014–2021 and 2022–2024.
""")

code("""
audit = status_audit(result.cleaned["status"])
print(audit["category"].value_counts().to_string())
mechanical = mechanical_retirements(
    result.driver_race, result.cleaned["status"], result.cleaned["constructors"]
)
print(narrate_mechanical(mechanical))
display(mechanical)
save(plot_mechanical(mechanical), "14_mechanical_retirements.png")
audit.to_csv(PATHS["audit"] / "status_categories.csv", index=False)
""")

md("""
McLaren's high rate in 2014–2021 is the Honda years, and the rate falls once that period is over. Alfa Romeo moves the other way in 2022–2024. The status list is a judgment, and the category table is the place to defend a status that was left out. A generic `Retired` is left out because the file does not say a component failed.
""")

from strip_descriptions import keep_headings
keep_headings(nb)
nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(nb.cells)} cells")
