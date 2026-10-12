"""Run questions, models, figures, and the report. Used to check the pipeline locally."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from src.f1_clean import clean_tables, load_raw, project_paths
from src.f1_features import build_model_frame, feature_dictionary_frame, feature_evidence
from src.f1_models import (
    plot_ablation,
    plot_feature_evidence,
    plot_importance,
    plot_lime,
    plot_model_comparison,
    plot_pr_curves,
    plot_preprocessing,
    plot_shap,
    run_models,
)
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
from src.f1_report import write_report


def save(fig, figures: Path, name: str) -> None:
    fig.savefig(figures / name, dpi=140, bbox_inches="tight", facecolor="white")
    fig.clf()


def main() -> None:
    paths = project_paths(ROOT)
    paths["figures"].mkdir(parents=True, exist_ok=True)
    paths["processed"].mkdir(parents=True, exist_ok=True)
    raw = load_raw(paths["raw"])
    result = clean_tables(raw)
    model = build_model_frame(result.driver_race, result.cleaned)
    model.to_csv(paths["processed"] / "model_frame.csv", index=False)
    feature_dictionary_frame().to_csv(paths["processed"] / "model_feature_dictionary.csv", index=False)

    evidence = feature_evidence(model)
    save(plot_feature_evidence(evidence), paths["figures"], "11_feature_evidence.png")

    ranked = front_row_conversion(result.driver_race, result.cleaned["circuits"], result.cleaned["qualifying"])
    home = home_advantage(result.driver_race, result.cleaned["drivers"], result.cleaned["circuits"])
    mechanical = mechanical_retirements(result.driver_race, result.cleaned["status"], result.cleaned["constructors"])
    status_audit(result.cleaned["status"]).to_csv(paths["audit"] / "status_categories.csv", index=False)
    ranked["rank_change"].to_csv(paths["audit"] / "front_row_rank_change.csv", index=False)
    mechanical.to_csv(paths["audit"] / "mechanical_retirements.csv", index=False)
    save(plot_front_row(ranked), paths["figures"], "12_front_row_conversion.png")
    save(plot_home(home), paths["figures"], "13_home_by_grid.png")
    save(plot_mechanical(mechanical), paths["figures"], "14_mechanical_retirements.png")

    q1 = narrate_front_row(ranked)
    q2 = narrate_home(home)
    q3 = narrate_mechanical(mechanical)
    print(q1)
    print(q2)
    print(q3)

    trained = run_models(model)
    trained["metrics"].to_csv(paths["audit"] / "model_metrics.csv", index=False)
    trained["ablation"].to_csv(paths["audit"] / "ablation.csv", index=False)
    trained["search"].to_csv(paths["audit"] / "model_search.csv", index=False)
    save(plot_model_comparison(trained["metrics"]), paths["figures"], "15_model_comparison.png")
    save(plot_pr_curves(trained["curves"]), paths["figures"], "16_pr_curves.png")
    save(plot_ablation(trained["ablation"]), paths["figures"], "17_ablation.png")
    save(plot_preprocessing(trained["preprocessing"]["transformed"]), paths["figures"], "18_preprocessing_order.png")
    save(plot_importance(trained["permutation"]), paths["figures"], "19_permutation_importance.png")
    save(plot_shap(trained["shap_values"], trained["shap_sample"]), paths["figures"], "20_shap_summary.png")
    save(plot_lime(trained["lime"]), paths["figures"], "21_lime_local.png")

    report_parts = {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "metrics": trained["metrics"],
        "chosen_model": trained["chosen_model"],
        "chosen_params": trained["chosen_params"],
        "ablation": trained["ablation"],
        "preprocessing": trained["preprocessing"],
        "shap_model": trained["shap_model"],
        "lime_case": trained["lime_case"],
        "inference_demo": trained["inference_demo"],
    }
    markdown_path, pdf_path = write_report(paths["root"] / "report", report_parts)
    print(trained["metrics"].to_string(index=False))
    print("chosen", trained["chosen_model"], trained["chosen_params"], "threshold", trained["threshold"])
    print(trained["ablation"].to_string(index=False))
    print(trained["inference_demo"].to_string(index=False))
    print("report", markdown_path, pdf_path)


if __name__ == "__main__":
    main()
