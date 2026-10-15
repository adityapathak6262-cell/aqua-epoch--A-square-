from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parent
DATASET_PATH = (
    ROOT
    / "Aqua_Epoch_BIS_Water_Quality_Dataset.xls - "
    "Aqua_Epoch_BIS_Water_Quality_Dataset.csv"
)
MODEL_PATH = ROOT / "model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "pH",
    "Hardness",
    "TDS",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
    "Calcium",
    "Magnesium",
    "Sodium",
    "Potassium",
    "Chloride",
    "Nitrate",
    "Fluoride",
    "Iron",
    "Alkalinity",
]


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    frame = pd.read_csv(DATASET_PATH).dropna(how="all").copy()
    missing_columns = set(FEATURE_COLUMNS + ["Potability"]) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    return frame


def clean_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = frame[columns].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).copy()
    if "pH" in cleaned:
        cleaned.loc[(cleaned["pH"] < 0) | (cleaned["pH"] > 14), "pH"] = np.nan
    for column in ("Hardness", "Fluoride"):
        if column in cleaned:
            cleaned.loc[cleaned[column] < 0, column] = np.nan
    return cleaned


def classification_pipeline() -> object:
    return make_pipeline(
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


def regression_pipeline() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_leaf_nodes=512,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    )


def main() -> None:
    frame = load_dataset()
    print(f"Dataset: {DATASET_PATH}")
    print(f"Usable non-empty rows: {len(frame)}")

    labels = pd.to_numeric(frame["Potability"], errors="coerce")
    labeled = labels.notna()
    classification_frame = frame.loc[labeled].copy()
    y_class = labels.loc[labeled].astype(int)
    observed_classes = sorted(y_class.unique().tolist())
    if observed_classes != [0, 1]:
        raise ValueError(
            f"Expected Potability labels [0, 1], found {observed_classes}"
        )

    X_class = clean_features(classification_frame, FEATURE_COLUMNS)
    X_train, X_test, y_train, y_test = train_test_split(
        X_class,
        y_class,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )
    evaluation_classifier = classification_pipeline()
    evaluation_classifier.fit(X_train, y_train)
    class_predictions = evaluation_classifier.predict(X_test)
    class_probabilities = evaluation_classifier.predict_proba(X_test)
    positive_index = list(
        evaluation_classifier.named_steps["randomforestclassifier"].classes_
    ).index(1)

    print("\nClassification held-out evaluation (positive class=1):")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    print(f"Accuracy:  {accuracy_score(y_test, class_predictions):.6f}")
    print(
        f"Precision: {precision_score(y_test, class_predictions, pos_label=1):.6f}"
    )
    print(f"Recall:    {recall_score(y_test, class_predictions, pos_label=1):.6f}")
    print(f"F1:        {f1_score(y_test, class_predictions, pos_label=1):.6f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, class_probabilities[:, positive_index]):.6f}")
    print(classification_report(y_test, class_predictions, zero_division=0))

    # Fit the deployment classifier on all labeled records after evaluating the holdout.
    classifier = classification_pipeline()
    classifier.fit(X_class, y_class)

    regressors: dict[str, object] = {}
    regression_features: dict[str, list[str]] = {}
    regression_evaluation: dict[str, dict[str, float | int]] = {}
    for target in ("pH", "TDS"):
        target_values = pd.to_numeric(frame[target], errors="coerce")
        valid_target = target_values.notna() & np.isfinite(target_values)
        if target == "pH":
            valid_target &= target_values.between(0, 14)
        regression_frame = frame.loc[valid_target].copy()
        y_reg = pd.to_numeric(regression_frame[target], errors="coerce")
        predictors = [
            column
            for column in FEATURE_COLUMNS
            if column not in {target, "Potability"}
        ]
        X_reg = clean_features(regression_frame, predictors)

        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            X_reg,
            y_reg,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        evaluation_regressor = regression_pipeline()
        evaluation_regressor.fit(Xr_train, yr_train)
        predictions = evaluation_regressor.predict(Xr_test)
        regression_evaluation[target] = {
            "train_samples": int(len(Xr_train)),
            "test_samples": int(len(Xr_test)),
            "mae": float(mean_absolute_error(yr_test, predictions)),
            "rmse": float(mean_squared_error(yr_test, predictions) ** 0.5),
            "r2": float(r2_score(yr_test, predictions)),
        }

        # Refit the final deployment regressor on all valid target rows.
        final_regressor = regression_pipeline()
        final_regressor.fit(X_reg, y_reg)
        regressors[target] = final_regressor
        regression_features[target] = predictors

    print("\nRegression held-out evaluations:")
    for target, values in regression_evaluation.items():
        print(
            f"{target}: train={values['train_samples']}, "
            f"test={values['test_samples']}, "
            f"MAE={values['mae']:.6f}, RMSE={values['rmse']:.6f}, "
            f"R2={values['r2']:.6f}"
        )

    regression_training_rows = {
        "pH": int(
            pd.to_numeric(frame["pH"], errors="coerce").between(0, 14).sum()
        ),
        "TDS": int(
            (
                pd.to_numeric(frame["TDS"], errors="coerce").notna()
                & np.isfinite(pd.to_numeric(frame["TDS"], errors="coerce"))
            ).sum()
        ),
    }
    bundle = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": [0, 1],
        "label_mapping": {"0": "Unsafe", "1": "Safe"},
        "classifier": classifier,
        "regressors": regressors,
        "regression_features": regression_features,
        "training_rows": {
            "classification": int(len(X_class)),
            **regression_training_rows,
        },
        "evaluation": {
            "classification": {
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "accuracy": float(accuracy_score(y_test, class_predictions)),
            },
            "regression": regression_evaluation,
        },
        "notes": [
            "Class names follow the project brief and must be verified against the source dataset.",
            "Predictions are not BIS compliance determinations.",
            "Regression predictions use the other measured features and do not use Potability.",
        ],
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"\nSaved deployment model bundle: {MODEL_PATH}")
    print(
        "Bundle includes the classifier plus pH and TDS regressors. "
        "Only load this trusted local joblib artifact."
    )


if __name__ == "__main__":
    main()
