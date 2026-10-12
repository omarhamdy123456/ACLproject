# ACL project

omar hamdy

Formula 1 podium prediction for one driver entering one race. The label is a podium finish (`positionOrder <= 3`). The cutoff is after qualifying and before the race starts, so the race result, laps, pit stops, status, points, and the standings published for that same race are not used as features.

## Dataset

Formula 1 World Championship (1950–2024) by Vopani:

https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

The 14 source CSVs are in `data/raw/`. Cleaning does not overwrite them.

## What is in this repository

- `src/f1_clean.py` — replace the Ergast `\N` placeholder, type the columns, and collapse shared drives to one driver–race row.
- `src/f1_questions.py` — front-row conversion by circuit, home versus away inside a grid band, and mechanical retirements by constructor.
- `src/f1_features.py` — leakage-safe features for 2003–2024. The split is train 2003–2018, validation 2019–2021, test 2022–2024.
- `src/f1_models.py` — logistic regression, random forest, gradient boosting, and a small neural network, plus ablation, preprocessing order, and explanations.
- `src/f1_report.py` — writes `report/milestone1_report.pdf`.
- `notebooks/milestone1.ipynb` — local notebook. It reads `data/raw` and imports `src`.
- `notebooks/milestone1_kaggle.ipynb` — the file to upload on Kaggle. The cleaning, feature, and question code is embedded in the notebook, so Kaggle does not need the `src` folder.
- `figures/` — plots from the audit, the three questions, and the model checks.
- `report/` — the written report.

## Run it locally

```powershell
pip install -r requirements.txt
python scripts/test_cleaning.py
python scripts/test_features.py
python scripts/run_milestone.py
```

`scripts/run_milestone.py` rebuilds the cleaned tables, the three question answers, the model comparison, the figures, and the PDF report.

## Run it on Kaggle

1. Open the Vopani dataset and choose **New Notebook**, so the CSVs are attached.
2. Use **File → Import notebook** and select `notebooks/milestone1_kaggle.ipynb`.
3. Run all cells from the top.

The setup cell finds the CSVs under `/kaggle/input`. Do not import `notebooks/milestone1.ipynb` on Kaggle. That copy expects this repository’s `src` folder.

## Rules that matter for the result

- `\N` is a placeholder, not a number. It is turned into a real blank. It is not filled with a mean, median, or mode.
- In `constructor_results`, `\N` means the constructor was not disqualified. That is stored as `disqualified = False`.
- Grid `0` means the driver had no grid slot. A blank grid is left blank.
- Two drivers can share one car in early seasons. The kept row is the better classification: lowest `positionOrder`, then most laps, then smallest `resultId`.
- Form uses only the previous five starts. Entering standings are the table after the previous round.
- The model chosen for the report is the one with the highest validation PR-AUC. A higher test score is reported and does not change that choice.
