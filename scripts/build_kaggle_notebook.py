"""Build a single notebook that runs on Kaggle without the local src package."""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "notebooks" / "01_data_cleaning.ipynb").read_text(encoding="utf-8")
MODULE = (ROOT / "src" / "f1_clean.py").read_text(encoding="utf-8")
nb = json.loads(SOURCE)

for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    else:
        cell.pop("outputs", None)
        cell.pop("execution_count", None)

nb["cells"][0]["source"] = "# Formula 1 podium prediction: data cleaning\n"
nb["cells"][1]["source"] = "## 1. Setup\n"

setup = f'''import os
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE_SOURCE = {MODULE!r}

module = types.ModuleType("f1_clean")
module.__dict__["__name__"] = "f1_clean"
sys.modules["f1_clean"] = module
exec(compile(MODULE_SOURCE, "f1_clean.py", "exec"), module.__dict__)
src_pkg = types.ModuleType("src")
src_pkg.f1_clean = module
sys.modules["src"] = src_pkg
sys.modules["src.f1_clean"] = module

from src.f1_clean import (
    ASSIGNMENT_LAST_YEAR,
    TABLES,
    clean_tables,
    collapse_entries,
    load_raw,
    raw_hashes,
    replace_sentinels,
)


def find_raw_dir() -> Path:
    """Find the Formula 1 CSVs Kaggle attached under /kaggle/input.

    Kaggle mounts each dataset as a symlink. A normal recursive search does not
    enter that link, so the folders are resolved first.
    """
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        search_roots = [kaggle_input]
        # The usual mount for the Vopani dataset, checked only if it is present.
        search_roots.append(kaggle_input / "formula-1-world-championship-1950-2020")
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
        if key in seen or resolved.name.lower() in {{"cleaned", "processed", "audit", "not_in_brief", "working"}}:
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
        names = {{path.name.lower(): path for path in folder.glob("*.csv")}}
        if all(name in names for name in required):
            found.append(folder)
    if not found:
        listing = []
        for root in search_roots:
            if not root.exists():
                listing.append(f"{{root}} does not exist")
                continue
            try:
                entries = list(root.iterdir())
            except OSError as error:
                listing.append(f"{{root}} could not be listed: {{error}}")
                continue
            if not entries:
                listing.append(f"{{root}} is empty")
            for entry in entries:
                listing.append(f"{{entry}} -> {{entry.resolve()}}")
        detail = " | ".join(listing) if listing else "no input folder was visible"
        raise FileNotFoundError(
            "No Formula 1 CSVs were found in the notebook input. "
            "The ipynb does not contain the data. On the right, open Input, "
            "click Add Data, and add Formula 1 World Championship (1950 - 2024) by Vopani. "
            "Wait until that name appears under Input, then Run All again. "
            "What was visible: " + detail
        )

    def score(folder: Path) -> tuple:
        present = sum((folder / f"{{name}}.csv").exists() for name in TABLES)
        return (present, -len(folder.parts))

    chosen = max(found, key=score)
    print("Dataset folder:", chosen)
    for path in sorted(chosen.glob("*.csv")):
        print(" ", path.name)
    return chosen


RAW_DIR = find_raw_dir()
missing = [name for name in TABLES if not (RAW_DIR / f"{{name}}.csv").exists()]
if missing:
    raise FileNotFoundError(f"Missing tables in {{RAW_DIR}}: {{missing}}")

if Path("/kaggle/working").exists():
    WORK_DIR = Path("/kaggle/working")
else:
    WORK_DIR = Path.cwd().resolve()
    if not (WORK_DIR / "data").exists() and (WORK_DIR.parent / "data").exists():
        WORK_DIR = WORK_DIR.parent
    WORK_DIR = WORK_DIR / "kaggle_output"

PATHS = {{
    "root": WORK_DIR,
    "raw": RAW_DIR,
    "cleaned": WORK_DIR / "cleaned",
    "processed": WORK_DIR / "processed",
    "audit": WORK_DIR / "audit",
    "figures": WORK_DIR / "figures",
}}
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
print(f"Loaded {{len(raw)}} tables from {{PATHS['raw']}}")
print(f"Writing outputs to {{WORK_DIR}}")
'''

nb["cells"][2]["source"] = setup.splitlines(keepends=True)
# nbformat wants the notebook object, not a raw dict, for writer niceties.
notebook = nbf.from_dict(nb)
from strip_descriptions import keep_headings
keep_headings(notebook)
out = ROOT / "notebooks" / "01_data_cleaning_kaggle.ipynb"
nbf.write(notebook, out)
print(f"Wrote {out}")
