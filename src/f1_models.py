"""Models, ablation, preprocessing order, explanations, and inference.

Hyperparameters and the decision threshold are chosen on the validation years only.
The test years are scored after those choices are frozen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.f1_features import FEATURE_COLUMNS, FEATURE_GROUPS

PRIMARY_METRIC = "pr_auc"
SECONDARY_METRIC = "f1"
METRIC_NAMES = ("precision", "recall", "f1", "roc_auc", "pr_auc")


class ScaleObservedThenImpute(BaseEstimator, TransformerMixin):
    """Standardize the observed values, then fill the remaining blanks with the median."""

    @staticmethod
    def _matrix(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.astype(float).to_numpy()
        return np.asarray(X, dtype=float)

    def fit(self, X, y=None):
        matrix = self._matrix(X)
        self.mean_ = np.nanmean(matrix, axis=0)
        self.scale_ = np.nanstd(matrix, axis=0)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ == 0)] = 1.0
        self.mean_[~np.isfinite(self.mean_)] = 0.0
        scaled = (matrix - self.mean_) / self.scale_
        self.median_ = np.nanmedian(scaled, axis=0)
        self.median_[~np.isfinite(self.median_)] = 0.0
        return self

    def transform(self, X):
        matrix = self._matrix(X)
        scaled = (matrix - self.mean_) / self.scale_
        missing = np.isnan(scaled)
        scaled[missing] = np.take(self.median_, np.where(missing)[1])
        return scaled


def _weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    count = np.bincount(y.astype(int))
    return np.where(y == 1, len(y) / (2 * count[1]), len(y) / (2 * count[0])).astype(float)


def _linear(model) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def _tree(model) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def _candidates() -> list[tuple[str, Pipeline, dict, bool]]:
    """Family name, pipeline, parameter label, and whether fit needs sample weights."""
    rows = []
    for c_value in (0.2, 1.0, 5.0):
        model = LogisticRegression(C=c_value, max_iter=400, class_weight="balanced", random_state=42)
        rows.append(("logistic_regression", _linear(model), {"C": c_value}, False))
    for depth in (6, 12):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=depth,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rows.append(("random_forest", _tree(model), {"max_depth": depth}, False))
    for depth in (2, 3):
        model = GradientBoostingClassifier(
            n_estimators=80, max_depth=depth, learning_rate=0.1, random_state=42,
        )
        rows.append(("gradient_boosting", _tree(model), {"max_depth": depth}, True))
    for layers in ((32,), (64, 32)):
        model = MLPClassifier(
            hidden_layer_sizes=layers,
            alpha=1e-4,
            max_iter=40,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
        rows.append(("neural_network", _linear(model), {"hidden_layers": layers}, True))
    return rows


def _fit(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, weighted: bool) -> Pipeline:
    if weighted:
        pipeline.fit(X, y, model__sample_weight=_weights(y))
    else:
        pipeline.fit(X, y)
    return pipeline


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 37):
        score = f1_score(y_true, probability >= threshold, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def score_split(model: str, split: str, y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = probability >= threshold
    return {
        "model": model,
        "split": split,
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "threshold": threshold,
    }


def _matrices(frame: pd.DataFrame, columns: list[str]):
    parts = {}
    for split in ("train", "validation", "test"):
        rows = frame.loc[frame["split"] == split]
        parts[split] = (rows[columns], rows["podium"].astype(int).to_numpy(), rows)
    return parts


def run_models(frame: pd.DataFrame) -> dict:
    """Fit the candidates, choose on validation, and score the frozen test years."""
    data = _matrices(frame, FEATURE_COLUMNS)
    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["validation"]
    x_test, y_test, test_rows = data["test"]

    search_rows = []
    fitted = {}
    for name, pipeline, params, weighted in _candidates():
        _fit(pipeline, x_train, y_train, weighted)
        val_probability = pipeline.predict_proba(x_val)[:, 1]
        val_pr = float(average_precision_score(y_val, val_probability))
        search_rows.append({
            "model": name,
            "params": str(params),
            "validation_pr_auc": val_pr,
            "weighted": weighted,
        })
        previous = fitted.get(name)
        if previous is None or val_pr > previous["validation_pr_auc"]:
            fitted[name] = {
                "pipeline": pipeline,
                "params": params,
                "weighted": weighted,
                "validation_pr_auc": val_pr,
            }

    chosen_name = max(fitted, key=lambda name: fitted[name]["validation_pr_auc"])
    metric_rows = []
    curves = {}
    for name, item in fitted.items():
        pipeline = item["pipeline"]
        val_probability = pipeline.predict_proba(x_val)[:, 1]
        threshold = best_f1_threshold(y_val, val_probability)
        item["threshold"] = threshold
        for split, (features, target, _) in data.items():
            probability = pipeline.predict_proba(features)[:, 1]
            metric_rows.append(score_split(name, split, target, probability, threshold))
            if split == "test":
                precision, recall, _ = precision_recall_curve(target, probability)
                curves[name] = (recall, precision)

    ablation = _ablation(fitted[chosen_name], data, chosen_name)
    preprocessing = _preprocessing_orders(x_train, y_train, x_val, y_val, x_test, y_test)
    explanations = _explanations(fitted, chosen_name, x_train, y_train, x_val, y_val, x_test, y_test, test_rows)
    return {
        "search": pd.DataFrame(search_rows).sort_values("validation_pr_auc", ascending=False),
        "metrics": pd.DataFrame(metric_rows),
        "chosen_model": chosen_name,
        "chosen_params": fitted[chosen_name]["params"],
        "threshold": fitted[chosen_name]["threshold"],
        "fitted": fitted,
        "ablation": ablation,
        "preprocessing": preprocessing,
        "curves": curves,
        **explanations,
    }


def _ablation(chosen: dict, data: dict, chosen_name: str) -> pd.DataFrame:
    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["validation"]
    x_test, y_test, _ = data["test"]
    full_val = float(average_precision_score(y_val, chosen["pipeline"].predict_proba(x_val)[:, 1]))
    rows = [{
        "removed_group": "(none)",
        "validation_pr_auc": full_val,
        "drop": 0.0,
        "test_pr_auc": float(average_precision_score(y_test, chosen["pipeline"].predict_proba(x_test)[:, 1])),
    }]
    for group, columns in FEATURE_GROUPS.items():
        kept = [column for column in FEATURE_COLUMNS if column not in columns]
        pipeline = clone(chosen["pipeline"])
        _fit(pipeline, x_train[kept], y_train, chosen["weighted"])
        val_pr = float(average_precision_score(y_val, pipeline.predict_proba(x_val[kept])[:, 1]))
        test_pr = float(average_precision_score(y_test, pipeline.predict_proba(x_test[kept])[:, 1]))
        rows.append({
            "removed_group": group,
            "validation_pr_auc": val_pr,
            "drop": full_val - val_pr,
            "test_pr_auc": test_pr,
        })
    table = pd.DataFrame(rows)
    table.insert(0, "model", chosen_name)
    return table


def _preprocessing_orders(x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    """Two orders of scaling and imputation, with the rest of the model held fixed."""
    orders = {
        "impute then scale": Pipeline([
            ("prepare", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])),
            ("model", LogisticRegression(C=1.0, max_iter=400, class_weight="balanced", random_state=42)),
        ]),
        "scale observed values then impute": Pipeline([
            ("prepare", ScaleObservedThenImpute()),
            ("model", LogisticRegression(C=1.0, max_iter=400, class_weight="balanced", random_state=42)),
        ]),
    }
    rows = []
    transformed = {}
    standing_index = FEATURE_COLUMNS.index("driver_standing_entering")
    for name, pipeline in orders.items():
        pipeline.fit(x_train, y_train)
        for split, features, target in (
            ("validation", x_val, y_val),
            ("test", x_test, y_test),
        ):
            probability = pipeline.predict_proba(features)[:, 1]
            rows.append({
                "order": name,
                "split": split,
                "pr_auc": float(average_precision_score(target, probability)),
                "roc_auc": float(roc_auc_score(target, probability)),
            })
        prepared = pipeline.named_steps["prepare"].transform(x_train)
        transformed[name] = prepared[:, standing_index]
    before = x_train["driver_standing_entering"].astype(float)
    summary = pd.DataFrame([
        {
            "version": "raw, blanks left empty",
            "missing": int(before.isna().sum()),
            "median": float(before.median()),
            "min": float(before.min()),
            "max": float(before.max()),
        },
        {
            "version": "impute then scale",
            "missing": int(np.isnan(transformed["impute then scale"]).sum()),
            "median": float(np.median(transformed["impute then scale"])),
            "min": float(np.min(transformed["impute then scale"])),
            "max": float(np.max(transformed["impute then scale"])),
        },
        {
            "version": "scale observed values then impute",
            "missing": int(np.isnan(transformed["scale observed values then impute"]).sum()),
            "median": float(np.median(transformed["scale observed values then impute"])),
            "min": float(np.min(transformed["scale observed values then impute"])),
            "max": float(np.max(transformed["scale observed values then impute"])),
        },
    ])
    return {"metrics": pd.DataFrame(rows), "distribution": summary, "transformed": transformed}


def predict_podium(record: dict, pipeline: Pipeline, threshold: float) -> dict:
    """Score one driver-race. Missing fields stay missing and the pipeline fills them."""
    row = {column: record.get(column, np.nan) for column in FEATURE_COLUMNS}
    features = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    probability = float(pipeline.predict_proba(features)[0, 1])
    return {
        "podium_probability": probability,
        "predicted_podium": probability >= threshold,
        "threshold": threshold,
    }


def _inference_demo(pipeline: Pipeline, threshold: float, test_rows: pd.DataFrame) -> pd.DataFrame:
    complete_mask = test_rows[FEATURE_COLUMNS].notna().all(axis=1)
    candidates = test_rows.loc[complete_mask]
    scores = pipeline.predict_proba(candidates[FEATURE_COLUMNS])[:, 1]
    source = candidates.iloc[int(np.argmax(scores))]
    complete = predict_podium(source.to_dict(), pipeline, threshold)
    partial_record = source.to_dict()
    partial_record["quali_position"] = np.nan
    partial_record["driver_standing_entering"] = np.nan
    partial_record["quali_position_missing"] = 1
    partial_record["standing_missing"] = 1
    partial = predict_podium(partial_record, pipeline, threshold)
    return pd.DataFrame([
        {
            "record": "complete test row",
            "driverId": int(source["driverId"]),
            "raceId": int(source["raceId"]),
            "actual_podium": bool(source["podium"]),
            **complete,
        },
        {
            "record": "same row with qualifying and standings removed",
            "driverId": int(source["driverId"]),
            "raceId": int(source["raceId"]),
            "actual_podium": bool(source["podium"]),
            **partial,
        },
    ])


def _explanations(fitted, chosen_name, x_train, y_train, x_val, y_val, x_test, y_test, test_rows) -> dict:
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    chosen = fitted[chosen_name]
    permutation = permutation_importance(
        chosen["pipeline"],
        x_val,
        y_val,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": permutation.importances_mean,
    }).sort_values("importance", ascending=False)

    tree_name = chosen_name if chosen_name in {"random_forest", "gradient_boosting"} else "gradient_boosting"
    tree_pipeline = fitted[tree_name]["pipeline"]
    sample = x_val.sample(n=min(400, len(x_val)), random_state=42)
    imputed = tree_pipeline.named_steps["impute"].transform(sample)
    explainer = shap.TreeExplainer(tree_pipeline.named_steps["model"])
    shap_values = explainer.shap_values(imputed)
    if isinstance(shap_values, list):
        shap_values = np.asarray(shap_values[1])
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    grid_index = FEATURE_COLUMNS.index("grid")
    # A larger grid number is a worse start. For the podium class that should
    # push the score down, so the correlation has to be negative.
    if np.corrcoef(imputed[:, grid_index], shap_values[:, grid_index])[0, 1] > 0:
        shap_values = -shap_values
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_table = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs})
    shap_table = shap_table.sort_values("mean_abs_shap", ascending=False)

    background = x_train.copy()
    background = background.fillna(background.median(numeric_only=True))
    lime_explainer = LimeTabularExplainer(
        background.to_numpy(dtype=float),
        feature_names=FEATURE_COLUMNS,
        class_names=["not_podium", "podium"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    probability = chosen["pipeline"].predict_proba(x_test)[:, 1]
    complete = x_test.notna().all(axis=1).to_numpy()
    if complete.any():
        local_index = int(np.argmax(np.where(complete, probability, -1)))
    else:
        local_index = int(np.argmax(probability))
    local_row = x_test.iloc[local_index].fillna(background.median(numeric_only=True))
    def _lime_predict(rows):
        features = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        return chosen["pipeline"].predict_proba(features)

    explanation = lime_explainer.explain_instance(
        local_row.to_numpy(dtype=float),
        _lime_predict,
        num_features=8,
    )
    lime_table = pd.DataFrame(explanation.as_list(), columns=["condition", "weight"])
    local_meta = test_rows.iloc[local_index]
    demo = _inference_demo(chosen["pipeline"], chosen["threshold"], test_rows)
    return {
        "permutation": importance,
        "shap_model": tree_name,
        "shap_values": shap_values,
        "shap_sample": imputed,
        "shap_table": shap_table,
        "lime": lime_table,
        "lime_case": {
            "raceId": int(local_meta["raceId"]),
            "driverId": int(local_meta["driverId"]),
            "year": int(local_meta["year"]),
            "actual_podium": int(y_test[local_index]),
            "probability": float(probability[local_index]),
        },
        "inference_demo": demo,
    }


def plot_model_comparison(metrics: pd.DataFrame):
    import matplotlib.pyplot as plt

    test = metrics.loc[metrics["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    x = np.arange(len(test))
    width = 0.36
    ax.bar(x - width / 2, test["pr_auc"], width=width, color="#1f4e79", label="PR-AUC")
    ax.bar(x + width / 2, test["f1"], width=width, color="#c46b2c", label="F1")
    ax.set_xticks(x, test["model"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Test scores after validation chose the threshold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_pr_curves(curves: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5))
    for name, (recall, precision) in curves.items():
        ax.plot(recall, precision, label=name, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on the test years", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_ablation(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    plotted = table.loc[table["removed_group"] != "(none)"].sort_values("drop")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(plotted["removed_group"], plotted["drop"], color="#1f4e79")
    ax.set_xlabel("Drop in validation PR-AUC when the group is removed")
    ax.set_title("Feature-group ablation", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_importance(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    ordered = table.sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(ordered["feature"], ordered["importance"], color="#2f6f6a")
    ax.set_xlabel("Drop in validation PR-AUC when the feature is shuffled")
    ax.set_title("Permutation importance of the selected model", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_shap(shap_values: np.ndarray, sample: np.ndarray):
    import matplotlib.pyplot as plt
    import shap

    plt.figure(figsize=(8, 5.2))
    shap.summary_plot(shap_values, sample, feature_names=FEATURE_COLUMNS, show=False, max_display=15)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_lime(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    ordered = table.iloc[::-1]
    colors = ["#c46b2c" if weight > 0 else "#1f4e79" for weight in ordered["weight"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(ordered["condition"], ordered["weight"], color=colors)
    ax.set_xlabel("LIME weight for the podium class")
    ax.set_title("One local explanation", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_preprocessing(transformed: dict):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, values) in zip(axes, transformed.items()):
        ax.hist(values, bins=30, color="#1f4e79")
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_xlabel("Transformed entering championship position")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
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
        ax.bar(labels, table["podium_rate"] * 100, color="#1f4e79")
        ax.set_title(title, loc="left", fontsize=11)
        ax.tick_params(axis="x", labelrotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
