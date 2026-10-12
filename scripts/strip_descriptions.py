"""Keep the section title in each markdown cell and drop the written description."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "notebooks" / "01_data_cleaning.ipynb",
    ROOT / "notebooks" / "01_data_cleaning_kaggle.ipynb",
    ROOT / "notebooks" / "milestone1.ipynb",
    ROOT / "notebooks" / "milestone1_kaggle.ipynb",
)


def keep_headings(notebook):
    kept = []
    removed = 0
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            kept.append(cell)
            continue
        source = cell.source
        if isinstance(source, list):
            source = "".join(source)
        heading = ""
        for line in source.splitlines():
            if line.startswith("#"):
                heading = line
                break
        if not heading:
            removed += 1
            continue
        cell.source = heading
        kept.append(cell)
    notebook.cells = kept
    return removed


def main() -> None:
    for path in NOTEBOOKS:
        notebook = nbf.read(path, as_version=4)
        removed = keep_headings(notebook)
        nbf.write(notebook, path)
        print(f"{path.name}: removed {removed} description cells, {len(notebook.cells)} cells left")


if __name__ == "__main__":
    main()
