"""The three data-engineering questions in the brief.

Each answer joins more than one table. The grain, the missing rows, and the
pre-race cutoff are part of the result, not only the chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.f1_features import TEST_LAST_YEAR, add_home_race

NAVY = "#1f4e79"
CLAY = "#c46b2c"
TEAL = "#2f6f6a"
MIN_FRONT_ROW_STARTS = 20
MIN_CONSTRUCTOR_STARTS = 30

# A start that did not happen. These rows are not in the retirement denominator.
DID_NOT_START = {
    "Did not qualify",
    "Did not prequalify",
    "Withdrew",
    "107% Rule",
}

# The file attributed the end of the race to a crash or to contact.
CRASH = {
    "Accident",
    "Collision",
    "Collision damage",
    "Spun off",
    "Fatal accident",
    "Debris",
    "Damage",
}

# The file attributed the end of the race to the driver, not to a part.
DRIVER_CAUSE = {
    "Injured",
    "Injury",
    "Physical",
    "Driver unwell",
    "Eye injury",
    "Illness",
    "Stalled",
}

# A regulation decision, not a broken component.
REGULATION = {
    "Disqualified",
    "Excluded",
    "Underweight",
}

# Not a classified finish, and not evidence of a specific failed component.
NOT_ATTRIBUTED = {
    "Retired",
    "Not classified",
    "Not restarted",
    "Handling",
    "Out of fuel",
    "Fuel",
    "Safety",
    "Safety concerns",
    "Refuelling",
    "Fuel rig",
}


def status_category(name: str) -> str:
    """Put every finishing status into one bucket used by question 3."""
    if name == "Finished" or str(name).startswith("+"):
        return "classified"
    if name in DID_NOT_START:
        return "did_not_start"
    if name in CRASH:
        return "crash"
    if name in DRIVER_CAUSE:
        return "driver"
    if name in REGULATION:
        return "regulation"
    if name in NOT_ATTRIBUTED:
        return "not_attributed"
    return "mechanical"


def status_audit(status: pd.DataFrame) -> pd.DataFrame:
    table = status[["statusId", "status"]].copy()
    table["category"] = table["status"].map(status_category)
    return table.sort_values("statusId").reset_index(drop=True)


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6")
    ax.set_axisbelow(True)


def front_row_conversion(
    driver_race: pd.DataFrame,
    circuits: pd.DataFrame,
    qualifying: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Question 1. Front row means grid 1 or 2. Qualifying position is kept separate."""
    entries = driver_race.loc[driver_race["year"] <= TEST_LAST_YEAR, [
        "raceId", "driverId", "year", "circuitId", "grid", "podium", "race_name",
    ]].copy()
    entries = entries.merge(
        circuits[["circuitId", "name", "country"]].rename(columns={"name": "circuit_name"}),
        on="circuitId",
        how="left",
        validate="many_to_one",
    )
    quali = qualifying[["raceId", "driverId", "position"]].rename(columns={"position": "quali_position"})
    entries = entries.merge(quali, on=["raceId", "driverId"], how="left", validate="one_to_one")
    entries["podium"] = entries["podium"].astype(int)
    entries["era"] = np.where(entries["year"] >= 2014, "2014-2024", "1950-2013")
    entries["grid_front"] = entries["grid"].isin([1, 2])
    entries["quali_front"] = entries["quali_position"].isin([1, 2])

    def _rank(front_rows: pd.DataFrame, era_name: str) -> pd.DataFrame:
        grouped = (
            front_rows.groupby(["circuit_name", "country"], as_index=False)
            .agg(front_row_starts=("podium", "size"), podiums=("podium", "sum"))
        )
        grouped = grouped.loc[grouped["front_row_starts"] >= MIN_FRONT_ROW_STARTS].copy()
        grouped["podium_rate"] = grouped["podiums"] / grouped["front_row_starts"]
        grouped["era"] = era_name
        grouped["rank"] = grouped["podium_rate"].rank(ascending=False, method="min").astype(int)
        return grouped.sort_values(["rank", "circuit_name"]).reset_index(drop=True)

    grid_known = entries.loc[entries["grid"].notna()]
    pre = _rank(grid_known.loc[(grid_known["era"] == "1950-2013") & grid_known["grid_front"]], "1950-2013")
    post = _rank(grid_known.loc[(grid_known["era"] == "2014-2024") & grid_known["grid_front"]], "2014-2024")
    both = pre[["circuit_name", "country", "podium_rate", "rank", "front_row_starts"]].merge(
        post[["circuit_name", "podium_rate", "rank", "front_row_starts"]],
        on="circuit_name",
        suffixes=("_pre", "_post"),
    )
    both["ranks_gained"] = both["rank_pre"] - both["rank_post"]
    both = both.sort_values("ranks_gained", ascending=False).reset_index(drop=True)

    both_known = entries.loc[
        entries["year"].between(2003, TEST_LAST_YEAR)
        & entries["grid"].notna()
        & entries["quali_position"].notna()
    ].copy()
    quali_front = both_known.loc[both_known["quali_front"]]
    moved_back = quali_front.loc[~quali_front["grid_front"]]
    stayed = quali_front.loc[quali_front["grid_front"]]
    disagreement = pd.DataFrame([
        {
            "group": "Qualified front row and started there",
            "starts": int(len(stayed)),
            "podium_rate": float(stayed["podium"].mean()) if len(stayed) else np.nan,
        },
        {
            "group": "Qualified front row but started further back",
            "starts": int(len(moved_back)),
            "podium_rate": float(moved_back["podium"].mean()) if len(moved_back) else np.nan,
        },
    ])
    return {"pre_2014": pre, "from_2014": post, "rank_change": both, "qualifying_versus_grid": disagreement}


def plot_front_row(ranked: dict[str, pd.DataFrame]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    panels = (("pre_2014", "1950-2013"), ("from_2014", "2014-2024"))
    for ax, (key, title) in zip(axes, panels):
        table = ranked[key].nsmallest(12, "rank").sort_values("podium_rate")
        ax.barh(table["circuit_name"], table["podium_rate"] * 100, color=NAVY)
        _style(ax, f"Front-row podium rate, {title}", "Percent of front-row starts", "")
    fig.tight_layout()
    return fig


def narrate_front_row(ranked: dict[str, pd.DataFrame]) -> str:
    pre = ranked["pre_2014"].iloc[0]
    post = ranked["from_2014"].iloc[0]
    change = ranked["rank_change"]
    gained = change.iloc[0]
    lost = change.sort_values("ranks_gained").iloc[0]
    gap = ranked["qualifying_versus_grid"].set_index("group")
    stayed = gap.loc["Qualified front row and started there"]
    moved = gap.loc["Qualified front row but started further back"]
    return (
        f"Using the starting grid, the highest front-row conversion before 2014 is "
        f"{pre['circuit_name']} at {pre['podium_rate'] * 100:.1f}% "
        f"({int(pre['podiums'])} podiums from {int(pre['front_row_starts'])} front-row starts). "
        f"From 2014 onward it is {post['circuit_name']} at {post['podium_rate'] * 100:.1f}% "
        f"({int(post['podiums'])} from {int(post['front_row_starts'])}). "
        f"Circuits need at least {MIN_FRONT_ROW_STARTS} front-row starts in an era, so a handful of races cannot top the list. "
        f"Among circuits that clear that bar in both eras, {gained['circuit_name']} gains the most ranks "
        f"({int(gained['rank_pre'])} to {int(gained['rank_post'])}), and {lost['circuit_name']} loses the most "
        f"({int(lost['rank_pre'])} to {int(lost['rank_post'])}). "
        f"Qualifying position is not the same list. In 2003-2024, {int(moved['starts'])} drivers qualified on the front row "
        f"and started further back. They podium {moved['podium_rate'] * 100:.1f}% of the time, against "
        f"{stayed['podium_rate'] * 100:.1f}% for the {int(stayed['starts'])} who qualified on the front row and started there. "
        f"A penalty is why those two columns come apart, so the ranking above uses the grid."
    )


def home_advantage(driver_race: pd.DataFrame, drivers: pd.DataFrame, circuits: pd.DataFrame) -> pd.DataFrame:
    """Question 2. Home versus away inside the same starting-grid band."""
    entries = driver_race.loc[
        (driver_race["year"] <= TEST_LAST_YEAR) & driver_race["grid"].notna() & (driver_race["grid"] >= 1),
        ["raceId", "driverId", "circuitId", "grid", "podium", "year"],
    ].copy()
    entries = add_home_race(entries, drivers, circuits)
    bins = [0, 1, 3, 10, 100]
    labels = ["1", "2-3", "4-10", "11+"]
    entries["grid_band"] = pd.cut(entries["grid"].astype(float), bins=bins, labels=labels)
    entries["podium"] = entries["podium"].astype(int)
    entries["where"] = np.where(entries["home_race"] == 1, "home", "away")
    table = (
        entries.groupby(["grid_band", "where"], observed=True)["podium"]
        .agg(starts="size", podium_rate="mean")
        .reset_index()
    )
    return table


def plot_home(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bands = ["1", "2-3", "4-10", "11+"]
    x = np.arange(len(bands))
    width = 0.36
    for offset, where, color in ((-width / 2, "home", CLAY), (width / 2, "away", NAVY)):
        rates = []
        for band in bands:
            row = table.loc[(table["grid_band"].astype(str) == band) & (table["where"] == where)]
            rates.append(float(row["podium_rate"].iloc[0]) * 100 if len(row) else 0)
        ax.bar(x + offset, rates, width=width, color=color, label=where)
    ax.set_xticks(x, bands)
    _style(ax, "Podium rate at home and away, by starting grid", "Starting grid", "Percent of starts")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e6e6e6")
    fig.tight_layout()
    return fig


def narrate_home(table: pd.DataFrame) -> str:
    lines = []
    for band in ["1", "2-3", "4-10", "11+"]:
        home = table.loc[(table["grid_band"].astype(str) == band) & (table["where"] == "home")].iloc[0]
        away = table.loc[(table["grid_band"].astype(str) == band) & (table["where"] == "away")].iloc[0]
        gap = (home["podium_rate"] - away["podium_rate"]) * 100
        lines.append(
            f"grid {band}: home {home['podium_rate'] * 100:.1f}% ({int(home['starts'])} starts), "
            f"away {away['podium_rate'] * 100:.1f}% ({int(away['starts'])} starts), gap {gap:+.1f} points"
        )
    return (
        "The comparison holds the starting position roughly fixed, so a home driver who also qualified at the front is not the whole result. "
        + " | ".join(lines)
        + ". Blank grids and grid 0 are left out, because those rows have no starting slot to control for. "
        "East German nationality is matched to German circuits, and a dual national is at home in either country. "
        "This is a difference inside a grid band, not a claim that the crowd causes the result."
    )


def mechanical_retirements(
    driver_race: pd.DataFrame,
    status: pd.DataFrame,
    constructors: pd.DataFrame,
) -> pd.DataFrame:
    """Question 3. Mechanical retirements divided by race starts, in two eras."""
    audit = status_audit(status)
    entries = driver_race.loc[
        driver_race["year"].between(2014, TEST_LAST_YEAR),
        ["raceId", "driverId", "year", "constructorId", "statusId"],
    ].copy()
    entries = entries.merge(audit, on="statusId", how="left", validate="many_to_one")
    entries = entries.merge(
        constructors[["constructorId", "name"]].rename(columns={"name": "constructor"}),
        on="constructorId",
        how="left",
        validate="many_to_one",
    )
    entries = entries.loc[entries["category"] != "did_not_start"].copy()
    entries["mechanical"] = (entries["category"] == "mechanical").astype(int)
    entries["era"] = np.where(entries["year"] <= 2021, "2014-2021", "2022-2024")
    grouped = (
        entries.groupby(["constructor", "era"], as_index=False)
        .agg(starts=("mechanical", "size"), mechanical_retirements=("mechanical", "sum"))
    )
    grouped["rate"] = grouped["mechanical_retirements"] / grouped["starts"]
    grouped = grouped.loc[grouped["starts"] >= MIN_CONSTRUCTOR_STARTS].copy()
    wide = grouped.pivot(index="constructor", columns="era", values=["starts", "mechanical_retirements", "rate"])
    wide.columns = [f"{value}_{era}" for value, era in wide.columns]
    wide = wide.reset_index()
    for column in ("rate_2014-2021", "rate_2022-2024"):
        if column not in wide.columns:
            wide[column] = np.nan
    wide = wide.dropna(subset=["rate_2014-2021", "rate_2022-2024"])
    wide["rate_change"] = wide["rate_2022-2024"] - wide["rate_2014-2021"]
    return wide.sort_values("rate_2022-2024", ascending=False).reset_index(drop=True)


def plot_mechanical(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    ordered = table.sort_values("rate_2022-2024")
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(ordered))
    ax.barh(y - 0.18, ordered["rate_2014-2021"] * 100, height=0.36, color="#9bb4c8", label="2014-2021")
    ax.barh(y + 0.18, ordered["rate_2022-2024"] * 100, height=0.36, color=CLAY, label="2022-2024")
    ax.set_yticks(y, ordered["constructor"])
    _style(ax, "Mechanical retirements per start", "Percent of starts", "")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_feature_evidence(evidence: dict):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        (axes[0, 0], evidence["grid"], "grid", "Podium rate by grid, training years"),
        (axes[0, 1], evidence["qualifying"], "quali_position", "Podium rate by qualifying position"),
        (axes[1, 0], evidence["standings"], "standing_bin", "Podium rate by standings entering the race"),
        (axes[1, 1], evidence["form"], "form_bin", "Podium rate by previous-5 podium rate"),
    )
    for ax, table, key, title in panels:
        labels = table[key].astype(str)
        ax.bar(labels, table["podium_rate"] * 100, color=NAVY)
        ax.set_title(title, loc="left", fontsize=11)
        ax.tick_params(axis="x", labelrotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def narrate_mechanical(table: pd.DataFrame) -> str:
    early = table.sort_values("rate_2014-2021", ascending=False).iloc[0]
    late = table.sort_values("rate_2022-2024", ascending=False).iloc[0]
    rose = table.sort_values("rate_change", ascending=False).iloc[0]
    fell = table.sort_values("rate_change").iloc[0]
    return (
        f"A mechanical retirement is a start whose status names a failed component. "
        f"Crashes, driver causes, regulation decisions, and unclassified 'Retired' rows are not counted. "
        f"Cars that did not start are out of the denominator. Constructors with fewer than {MIN_CONSTRUCTOR_STARTS} starts in an era are left out of that era, "
        f"and the comparison below keeps constructors that clear the bar in both. "
        f"Highest rate in 2014-2021: {early['constructor']} at {early['rate_2014-2021'] * 100:.1f}%. "
        f"Highest rate in 2022-2024: {late['constructor']} at {late['rate_2022-2024'] * 100:.1f}%. "
        f"Largest rise: {rose['constructor']} ({rose['rate_change'] * 100:+.1f} points). "
        f"Largest fall: {fell['constructor']} ({fell['rate_change'] * 100:+.1f} points)."
    )
