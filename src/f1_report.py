"""Analytical report for Milestone 1. The numbers come from the pipeline, not from a hand-edited table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fpdf import FPDF


def _metrics_view(metrics: pd.DataFrame) -> pd.DataFrame:
    view = metrics.copy()
    for column in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        view[column] = view[column].map(lambda value: f"{value:.3f}")
    view["threshold"] = view["threshold"].map(lambda value: f"{value:.2f}")
    return view


def build_sections(parts: dict) -> list[tuple[str, str]]:
    metrics = _metrics_view(parts["metrics"])
    chosen = parts["chosen_model"]
    chosen_val = parts["metrics"].loc[
        (parts["metrics"]["model"] == chosen) & (parts["metrics"]["split"] == "validation")
    ].iloc[0]
    chosen_test = parts["metrics"].loc[
        (parts["metrics"]["model"] == chosen) & (parts["metrics"]["split"] == "test")
    ].iloc[0]
    baseline_test = parts["metrics"].loc[
        (parts["metrics"]["model"] == "logistic_regression") & (parts["metrics"]["split"] == "test")
    ].iloc[0]
    test_leader = parts["metrics"].loc[parts["metrics"]["split"] == "test"].sort_values("pr_auc", ascending=False).iloc[0]
    held_out_note = ""
    if test_leader["model"] != chosen:
        held_out_note = (
            f" On the test years, {test_leader['model']} has a higher PR-AUC ({float(test_leader['pr_auc']):.3f}) "
            f"than the selected model ({float(chosen_test['pr_auc']):.3f}). That fact is reported and is not used to change the choice."
        )
    ablation = parts["ablation"]
    biggest_drop = ablation.loc[ablation["removed_group"] != "(none)"].sort_values("drop", ascending=False).iloc[0]
    unused = ablation.loc[ablation["drop"] < 0, "removed_group"].tolist()
    unused_note = ""
    if unused:
        unused_note = (
            " Removing " + ", ".join(unused) + " raises validation PR-AUC, so that group is not adding information once the other groups are present."
        )
    prep = parts["preprocessing"]["metrics"]
    prep_val = prep.loc[prep["split"] == "validation"].sort_values("pr_auc", ascending=False)
    sections = [
        ("What this report answers",
         "The prediction is whether a driver finishes on the podium. The cutoff is after qualifying and before the race starts. "
         "Current-race results, points, laps, status, pit stops, and the standings published for that race are not features. "
         "Primary metric: PR-AUC, because a podium is the minority outcome. Secondary metric: F1 at a threshold chosen on the validation years."),
        ("Question 1. Front-row conversion by circuit", parts["q1"]),
        ("Question 2. Home races, controlling for the grid", parts["q2"]),
        ("Question 3. Mechanical retirements", parts["q3"]),
        ("Features that are legal at the cutoff",
         "Grid and front row are known once penalties are applied. Qualifying position and reaching Q3 are known when qualifying ends. "
         "Form uses the previous five starts only. Entering standings are the championship table after the previous round, so round 1 is blank on purpose. "
         "Home race compares the driver's nationality with the circuit country. Same-weekend sprint points are left out: they are known before the race in the current format, "
         "but they sit in a results table from the same event, and the safer reading of the brief is to exclude them. "
         "The model years are 2003-2024. Before 2003 the qualifying table does not cover every completed race, so the cutoff is not in the file."),
        ("Split",
         "Train 2003-2018, validation 2019-2021, test 2022-2024. The split is by season so a later race cannot be used to fit a model that is then scored on an earlier race. "
         "The 2022 regulation change sits inside the test years. Thresholds and hyperparameters are chosen on validation only."),
        ("Model comparison",
         f"The selected model is {chosen}, with parameters {parts['chosen_params']}. "
         f"It has the highest validation PR-AUC, {float(chosen_val['pr_auc']):.3f}. "
         f"Its F1 threshold, {float(chosen_test['threshold']):.2f}, was chosen on the validation years and then frozen. "
         f"On the test years its PR-AUC is {float(chosen_test['pr_auc']):.3f} and its F1 is {float(chosen_test['f1']):.3f}. "
         f"Logistic regression is the baseline. On the test years its PR-AUC is {float(baseline_test['pr_auc']):.3f} and its F1 is {float(baseline_test['f1']):.3f}. "
         "The neural network is a shallow feed-forward network with one or two hidden layers."
         + held_out_note
         + "\n\n"
         + metrics.to_string(index=False)),
        ("Preprocessing order",
         "Both experiments use logistic regression with C=1. Only the order of imputation and scaling changes. "
         "The distribution table is the driver's entering championship position on the training years.\n\n"
         + prep.to_string(index=False)
         + "\n\n"
         + parts["preprocessing"]["distribution"].to_string(index=False)
         + f"\n\nHigher validation PR-AUC: {prep_val.iloc[0]['order']} at {prep_val.iloc[0]['pr_auc']:.3f}."),
        ("Ablation",
         f"Each row removes one feature group and retrains {chosen} with the same hyperparameters. "
         f"The largest validation drop is {biggest_drop['removed_group']} ({float(biggest_drop['drop']):.3f}). "
         + unused_note
         + " The test column is a check. It is not used to pick the group.\n\n"
         + ablation.to_string(index=False)),
        ("Explanations are not causes",
         f"Permutation importance shuffles one feature of the selected model and measures the drop in validation PR-AUC. "
         f"SHAP is computed for {parts['shap_model']}, because that model is a tree model and the values stay in the original feature units. "
         f"The local LIME explanation is one test row: race {parts['lime_case']['raceId']}, driver {parts['lime_case']['driverId']}, "
         f"predicted probability {parts['lime_case']['probability']:.3f}, actual podium {parts['lime_case']['actual_podium']}. "
         "A large weight means the model used that feature for this row. It does not mean the feature caused the podium."),
        ("Inference on a complete row and a partial row",
         parts["inference_demo"].to_string(index=False)
         + "\n\nThe second row is the same driver and race with qualifying position and entering standings removed. The pipeline fills those blanks from the training median. The probability changes, and the function still returns a score."
         + (
             " On this run the yes-or-no decision also changes, because the probability crosses the frozen threshold."
             if bool(parts["inference_demo"].iloc[0]["predicted_podium"]) != bool(parts["inference_demo"].iloc[1]["predicted_podium"])
             else ""
         )),
        ("Limitations",
         "Qualifying is missing or partial before 2003, so those seasons are not in the model. "
         "Q3 did not exist in the early seasons that are in the training window, and reached_q3 is left blank for races that had no Q3. "
         "Entering standings ignore sprint points earned earlier in the same weekend. "
         "The mechanical-retirement list follows the status text: a tyre or a wing counts only when the file names that part, and a generic Retired does not count. "
         "Home country is nationality, not birthplace. East German drivers are matched to German circuits, and a dual national matches either country. "
         "One 1983 row disagrees between positionOrder and the official position text; the label follows positionOrder, as the brief defines it. "
         "The test seasons are a different regulation era from most of the training seasons, so a drop from validation to test is part of the result, not a broken split."),
    ]
    return sections


def write_report(directory: Path, parts: dict) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    sections = build_sections(parts)
    markdown_path = directory / "milestone1_report.md"
    lines = ["# Milestone 1: Formula 1 podium prediction", ""]
    for title, body in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body)
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, "Milestone 1: Formula 1 podium prediction")
    pdf.ln(2)
    for title, body in sections:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        safe = body.encode("latin-1", errors="replace").decode("latin-1")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 4.5, safe, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    pdf_path = directory / "milestone1_report.pdf"
    pdf.output(str(pdf_path))
    return markdown_path, pdf_path
