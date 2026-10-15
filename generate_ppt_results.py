from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_DIR = Path("ppt_screenshots")


def main() -> None:
    csv_files = sorted(Path.cwd().glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found in the workspace. Place the dataset beside this script."
        )
    if len(csv_files) > 1:
        raise RuntimeError(
            "Multiple CSV files found; refusing to combine or select unrelated data: "
            + ", ".join(path.name for path in csv_files)
        )

    dataset_path = csv_files[0]
    raw = pd.read_csv(dataset_path)
    blank_rows = int(raw.isna().all(axis=1).sum())
    data = raw.dropna(how="all").copy()
    if "Potability" not in data.columns:
        raise ValueError(
            "Could not identify the target column: expected a Potability column. "
            f"Found columns: {list(data.columns)}"
        )

    target = pd.to_numeric(data["Potability"], errors="coerce")
    labeled_mask = target.notna()
    unlabeled_rows = int((~labeled_mask).sum())
    data = data.loc[labeled_mask].copy()
    data["Potability"] = target.loc[labeled_mask].astype(int)
    labels = sorted(data["Potability"].unique().tolist())
    if len(labels) != 2:
        raise ValueError(
            f"Expected a binary classification target; found labels: {labels}"
        )
    feature_columns = [column for column in data.columns if column != "Potability"]
    features = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    if "pH" in features:
        features.loc[(features["pH"] < 0) | (features["pH"] > 14), "pH"] = np.nan
    for column in ("Hardness", "Fluoride"):
        if column in features:
            features.loc[features[column] < 0, column] = np.nan
    y = data["Potability"]

    duplicates_labeled = int(data.duplicated().sum())
    duplicates_nonempty = int(raw.dropna(how="all").duplicated().sum())
    missing_nonempty = raw.dropna(how="all").isna().sum()
    class_counts = y.value_counts().sort_index()

    if int(class_counts.min()) < 2:
        raise ValueError(
            "Stratified train/test split requires at least two samples per class; "
            f"observed counts: {class_counts.to_dict()}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_leaf_nodes=512,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.named_steps["randomforestclassifier"].classes_
    positive_label = 1 if 1 in classes else classes[-1]
    positive_index = list(classes).index(positive_label)
    positive_scores = probabilities[:, positive_index]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label=positive_label, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label=positive_label, zero_division=0
    )
    f1 = f1_score(y_test, predictions, pos_label=positive_label, zero_division=0)
    roc_auc = roc_auc_score(y_test, positive_scores)
    average_precision = average_precision_score(
        y_test == positive_label, positive_scores
    )
    cm = confusion_matrix(y_test, predictions, labels=labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(
        context="talk",
        style="whitegrid",
        palette="deep",
        font="DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    # 01 is intentionally reserved as a title/summary image slot.
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1,
        linecolor="white",
        xticklabels=[f"Class {label}" for label in labels],
        yticklabels=[f"Class {label}" for label in labels],
        annot_kws={"size": 18, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Random Forest — Confusion Matrix", pad=18)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.savefig(OUTPUT_DIR / "02_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    forest = model.named_steps["randomforestclassifier"]
    importance = pd.Series(
        forest.feature_importances_, index=feature_columns
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(
        figsize=(11, max(7, len(feature_columns) * 0.42)),
        constrained_layout=True,
    )
    sns.barplot(
        x=importance.values,
        y=importance.index,
        color="#168C8C",
        ax=ax,
    )
    ax.set_title("Random Forest — Feature Importance", pad=16)
    ax.set_xlabel("Mean decrease in impurity (relative importance)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUTPUT_DIR / "03_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test == positive_label, positive_scores
    )
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color="#167DAD",
        linewidth=3,
        label=f"Random Forest (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.5, label="Chance")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(OUTPUT_DIR / "04_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    curve_precision, curve_recall, _ = precision_recall_curve(
        y_test == positive_label, positive_scores
    )
    pr_auc = auc(curve_recall, curve_precision)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(
        curve_recall,
        curve_precision,
        color="#C45A24",
        linewidth=3,
        label=f"Random Forest (AP = {average_precision:.3f}; PR-AUC = {pr_auc:.3f})",
    )
    ax.axhline(
        (y_test == positive_label).mean(),
        linestyle="--",
        color="#777777",
        linewidth=1.5,
        label="Positive-class prevalence",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall Curve",
    )
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    fig.savefig(OUTPUT_DIR / "05_precision_recall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    correlation_data = features.copy()
    correlation_data["Potability"] = y
    corr = correlation_data.corr(method="pearson")
    fig, ax = plt.subplots(
        figsize=(max(12, len(corr.columns) * 0.72), max(10, len(corr.columns) * 0.62)),
        constrained_layout=True,
    )
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.25,
        cbar_kws={"label": "Pearson correlation"},
        annot=len(corr.columns) <= 12,
        fmt=".2f",
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("Water-Quality Feature Correlation", pad=16)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    fig.savefig(OUTPUT_DIR / "06_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sns.barplot(
        x=[str(label) for label in class_counts.index],
        y=class_counts.values,
        hue=[str(label) for label in class_counts.index],
        palette=["#168C8C", "#E28A36"][: len(class_counts)],
        legend=False,
        ax=ax,
    )
    ax.set_title("Target Class Distribution", pad=16)
    ax.set_xlabel("Potability class (dataset labels)")
    ax.set_ylabel("Number of samples")
    for bar, count in zip(ax.patches, class_counts.values):
        ax.annotate(
            f"{count:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            xytext=(0, 5),
            textcoords="offset points",
        )
    fig.savefig(
        OUTPUT_DIR / "07_target_class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # One subplot per feature preserves original units and allows inspection by class.
    box_data = features.copy()
    box_data["Potability"] = y.astype(str)
    ncols = 3
    nrows = int(np.ceil(len(feature_columns) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, nrows * 3.15),
        constrained_layout=True,
        squeeze=False,
    )
    class_palette = {str(label): color for label, color in zip(labels, ["#168C8C", "#E28A36"])}
    for index, column in enumerate(feature_columns):
        ax = axes.flat[index]
        sns.boxplot(
            data=box_data,
            x="Potability",
            y=column,
            hue="Potability",
            order=[str(label) for label in labels],
            hue_order=[str(label) for label in labels],
            palette=class_palette,
            showfliers=False,
            legend=False,
            width=0.62,
            ax=ax,
        )
        ax.set_title(column, fontsize=12, pad=8)
        ax.set_xlabel("Class")
        ax.set_ylabel("Recorded value")
        ax.tick_params(axis="both", labelsize=9)
    for index in range(len(feature_columns), nrows * ncols):
        axes.flat[index].set_visible(False)
    fig.suptitle(
        "Feature Distributions by Potability Class\n(whisker plots omit individual outlier points for readability)",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(
        OUTPUT_DIR / "08_feature_boxplots_by_class.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Regression baseline for pH; both pH and Potability are excluded from predictors.
    regression_data = raw.dropna(how="all").copy()
    regression_data["pH"] = pd.to_numeric(regression_data["pH"], errors="coerce")
    regression_data = regression_data.loc[
        regression_data["pH"].between(0, 14)
    ].copy()
    regression_features = [
        column
        for column in regression_data.columns
        if column not in {"pH", "Potability"}
    ]
    X_reg = regression_data[regression_features].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    for column in ("Hardness", "Fluoride"):
        if column in X_reg:
            X_reg.loc[X_reg[column] < 0, column] = np.nan
    y_reg = regression_data["pH"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg,
        y_reg,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    pH_regressor = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_leaf_nodes=512,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    )
    pH_regressor.fit(Xr_train, yr_train)
    pH_predictions = pH_regressor.predict(Xr_test)
    pH_mae = mean_absolute_error(yr_test, pH_predictions)
    pH_rmse = mean_squared_error(yr_test, pH_predictions) ** 0.5
    pH_r2 = r2_score(yr_test, pH_predictions)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.scatter(
        yr_test,
        pH_predictions,
        s=26,
        alpha=0.42,
        color="#167DAD",
        edgecolors="white",
        linewidths=0.25,
        rasterized=True,
    )
    plot_min = min(float(yr_test.min()), float(pH_predictions.min()))
    plot_max = max(float(yr_test.max()), float(pH_predictions.max()))
    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        linestyle="--",
        linewidth=2,
        color="#C45A24",
        label="Ideal prediction (y = x)",
    )
    ax.set(
        xlabel="Actual pH",
        ylabel="Predicted pH",
        title="pH Regression — Actual vs Predicted",
        xlim=(plot_min, plot_max),
        ylim=(plot_min, plot_max),
    )
    ax.text(
        0.04,
        0.96,
        f"MAE  = {pH_mae:.4f}\nRMSE = {pH_rmse:.4f}\nR²     = {pH_r2:.4f}\nTest samples = {len(yr_test):,}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.92},
    )
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(
        OUTPUT_DIR / "09_ph_actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    class_names = ", ".join(str(label) for label in labels)
    metrics_text = f"""AQUA-EPOCH — RANDOM FOREST CLASSIFICATION RESULTS
=================================================
Dataset file: {dataset_path.resolve()}
Dataset shape as read: {raw.shape[0]} rows x {raw.shape[1]} columns
Target column: Potability
Observed target labels: {class_names}
Label interpretation: retained as dataset codes; not assumed BIS-certified
Feature columns ({len(feature_columns)}): {", ".join(feature_columns)}
Completely blank rows excluded: {blank_rows}
Rows without a usable target excluded: {unlabeled_rows}
Exact duplicate non-empty rows: {duplicates_nonempty}
Exact duplicate labeled rows: {duplicates_labeled}
Split: stratified 80% train / 20% test; random_state={RANDOM_STATE}
Training samples: {len(X_train)}
Testing samples: {len(X_test)}
Positive class used for ROC/PR: {positive_label}
Positive-class prevalence in test set: {(y_test == positive_label).mean():.6f}

Missing values by column in non-empty source rows:
{missing_nonempty.to_string()}

Target class distribution among labeled rows:
{class_counts.to_string()}

HELD-OUT TEST METRICS
---------------------
Accuracy:  {accuracy:.6f}
Precision: {precision:.6f}
Recall:    {recall:.6f}
F1-score:  {f1:.6f}
ROC-AUC:   {roc_auc:.6f}
Average Precision: {average_precision:.6f}
Precision-Recall AUC: {pr_auc:.6f}

Confusion matrix (rows=actual, columns=predicted), label order={labels}:
{cm}

Classification report:
{classification_report(y_test, predictions, labels=labels, zero_division=0)}

Feature importance (descending; impurity based, not causal):
{importance.sort_values(ascending=False).to_string()}

TASK B — pH REGRESSION (HELD-OUT TEST SET)
-----------------------------------------
Target: pH; rows outside 0–14 excluded for this regression task.
Predictors: all numeric features except pH and Potability.
Split: 80% train / 20% test; random_state={RANDOM_STATE}
Training samples: {len(Xr_train)}
Testing samples: {len(Xr_test)}
Model: RandomForestRegressor(n_estimators=300, max_leaf_nodes=512,
                             min_samples_leaf=2, random_state=42, n_jobs=-1)
Missing-value handling: median imputation, fitted on training subset only.
MAE:  {pH_mae:.10f}
RMSE: {pH_rmse:.10f}
R²:   {pH_r2:.10f}

Preprocessing: median imputation, fitted only on the training subset.
Model: RandomForestClassifier(n_estimators=400, class_weight='balanced',
                              max_leaf_nodes=512, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
Limitations: These metrics predict the supplied dataset labels only. They do not
establish real-world generalization or BIS compliance. Verify label provenance,
units, data quality, and external validation before operational use.
"""
    metrics_path = OUTPUT_DIR / "model_metrics.txt"
    metrics_path.write_text(metrics_text, encoding="utf-8")

    print(f"Dataset inspected: {dataset_path.resolve()}")
    print(f"Format: CSV; raw shape: {raw.shape}")
    print(f"Features ({len(feature_columns)}): {feature_columns}")
    print(f"Target: Potability; observed classes: {labels}")
    print(f"Completely blank rows: {blank_rows}")
    print(f"Rows without usable target: {unlabeled_rows}")
    print(f"Exact duplicate non-empty rows: {duplicates_nonempty}")
    print(f"Exact duplicate labeled rows: {duplicates_labeled}")
    print("Missing values by column (non-empty rows):")
    print(missing_nonempty.to_string())
    print("Class distribution:")
    print(class_counts.to_string())
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print("\nHeld-out test metrics:")
    print(f"Accuracy:             {accuracy:.6f}")
    print(f"Precision (class {positive_label}):    {precision:.6f}")
    print(f"Recall (class {positive_label}):       {recall:.6f}")
    print(f"F1-score (class {positive_label}):     {f1:.6f}")
    print(f"ROC-AUC:              {roc_auc:.6f}")
    print(f"Average precision:    {average_precision:.6f}")
    print(f"Precision-Recall AUC: {pr_auc:.6f}")
    print(f"Confusion matrix (labels={labels}):\n{cm}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print("\nFeature importance (highest first):")
    print(importance.sort_values(ascending=False).to_string())
    print("\nTask B — pH regression held-out test results:")
    print(f"Training samples: {len(Xr_train)}")
    print(f"Testing samples:  {len(Xr_test)}")
    print(f"MAE:  {pH_mae:.10f}")
    print(f"RMSE: {pH_rmse:.10f}")
    print(f"R²:   {pH_r2:.10f}")
    print(f"\nMetrics saved: {metrics_path.resolve()}")
    print("Generated image files:")
    for image_path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"{image_path.resolve()} ({image_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
