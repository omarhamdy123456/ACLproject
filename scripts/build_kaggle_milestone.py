"""Build the Kaggle notebook: cleaning, the three questions, and the pre-race feature table."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.read(ROOT / "notebooks" / "milestone1.ipynb", as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

modules = {
    "CLEAN_SOURCE": ROOT / "src" / "f1_clean.py",
    "FEATURES_SOURCE": ROOT / "src" / "f1_features.py",
    "QUESTIONS_SOURCE": ROOT / "src" / "f1_questions.py",
}
embedded = "\n".join(f"{name} = {path.read_text(encoding='utf-8')!r}" for name, path in modules.items())

setup = embedded + """

import os
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def _install(alias, source):
    module = types.ModuleType(alias)
    module.__dict__["__name__"] = alias
    sys.modules[alias] = module
    sys.modules["src." + alias] = module
    setattr(src_pkg, alias, module)
    exec(compile(source, alias + ".py", "exec"), module.__dict__)


src_pkg = types.ModuleType("src")
src_pkg.__path__ = []
sys.modules["src"] = src_pkg
_install("f1_clean", CLEAN_SOURCE)
_install("f1_features", FEATURES_SOURCE)
_install("f1_questions", QUESTIONS_SOURCE)

from src.f1_clean import (
    ASSIGNMENT_LAST_YEAR,
    SENTINEL,
    TABLES,
    clean_tables,
    collapse_entries,
    load_raw,
    raw_hashes,
    replace_sentinels,
)


def find_raw_dir() -> Path:
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        search_roots = [kaggle_input, kaggle_input / "formula-1-world-championship-1950-2020"]
    else:
        cwd = Path.cwd().resolve()
        search_roots = [cwd]
        if (cwd.parent / "data" / "raw" / "results.csv").exists():
            search_roots.append(cwd.parent)

    folders = []
    seen = set()

    def add_folder(folder: Path) -> None:
        try:
            resolved = folder.resolve()
        except OSError:
            return
        if not resolved.is_dir():
            return
        key = str(resolved).lower()
        if key in seen or resolved.name.lower() in {"cleaned", "processed", "audit", "not_in_brief", "working"}:
            return
        seen.add(key)
        folders.append(resolved)

    for root in search_roots:
        if not root.exists():
            continue
        add_folder(root)
        try:
            children = list(root.iterdir())
        except OSError as error:
            print("Could not list", root, error)
            children = []
        for child in children:
            add_folder(child)
            try:
                for grandchild in child.resolve().iterdir():
                    add_folder(grandchild)
            except OSError:
                continue
        for dirpath, _, _ in os.walk(root, followlinks=True):
            add_folder(Path(dirpath))

    required = ("results.csv", "races.csv", "drivers.csv", "status.csv")
    found = []
    for folder in folders:
        names = {path.name.lower(): path for path in folder.glob("*.csv")}
        if all(name in names for name in required):
            found.append(folder)
    if not found:
        listing = []
        for root in search_roots:
            if not root.exists():
                listing.append(f"{root} does not exist")
                continue
            try:
                entries = list(root.iterdir())
            except OSError as error:
                listing.append(f"{root} could not be listed: {error}")
                continue
            if not entries:
                listing.append(f"{root} is empty")
            for entry in entries:
                listing.append(f"{entry} -> {entry.resolve()}")
        detail = " | ".join(listing) if listing else "no input folder was visible"
        raise FileNotFoundError(
            "No Formula 1 CSVs were found in the notebook input. "
            "On the right, open Input, click Add Data, and add "
            "Formula 1 World Championship (1950 - 2024) by Vopani. "
            "Wait until that name appears under Input, then Run All again. "
            "What was visible: " + detail
        )

    def score(folder: Path) -> tuple:
        present = sum((folder / f"{name}.csv").exists() for name in TABLES)
        return (present, -len(folder.parts))

    chosen = max(found, key=score)
    print("Dataset folder:", chosen)
    for path in sorted(chosen.glob("*.csv")):
        print(" ", path.name)
    return chosen


RAW_DIR = find_raw_dir()
missing = [name for name in TABLES if not (RAW_DIR / f"{name}.csv").exists()]
if missing:
    raise FileNotFoundError(f"Missing tables in {RAW_DIR}: {missing}")

if Path("/kaggle/working").exists():
    WORK_DIR = Path("/kaggle/working")
else:
    WORK_DIR = Path.cwd().resolve()
    if not (WORK_DIR / "data").exists() and (WORK_DIR.parent / "data").exists():
        WORK_DIR = WORK_DIR.parent
    WORK_DIR = WORK_DIR / "kaggle_output"

PATHS = {
    "root": WORK_DIR,
    "raw": RAW_DIR,
    "cleaned": WORK_DIR / "cleaned",
    "processed": WORK_DIR / "processed",
    "audit": WORK_DIR / "audit",
    "figures": WORK_DIR / "figures",
}
for folder in ("cleaned", "processed", "audit", "figures"):
    PATHS[folder].mkdir(parents=True, exist_ok=True)
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
print(f"Writing outputs to {WORK_DIR}")
"""

for index, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and "load_raw" in cell.source and "find_raw_dir" not in cell.source:
        cell.source = setup
        setup_index = index
        break
else:
    raise RuntimeError("Could not find the setup cell")

nb.cells[0].source = "# Formula 1 podium prediction"

from strip_descriptions import keep_headings
keep_headings(nb)
out = ROOT / "notebooks" / "milestone1_kaggle.ipynb"
nbf.write(nb, out)
print(f"Wrote {out} with {len(nb.cells)} cells; setup cell {setup_index}")
