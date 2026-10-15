from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sqlite3
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("AQUAEPOCH_MODEL_PATH", str(ROOT / "model.pkl")))
DATABASE_PATH = Path(
    os.getenv("AQUAEPOCH_DATABASE_PATH", str(ROOT / "prediction_history.sqlite3"))
)

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

# These are broad application input sanity bounds, not drinking-water limits.
# Confirm measurement units and replace/configure bounds using authoritative
# laboratory and instrument documentation before real-world deployment.
INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "pH": (0.0, 14.0),
    "Hardness": (0.0, 10_000.0),
    "TDS": (0.0, 1_000_000.0),
    "Chloramines": (0.0, 1_000.0),
    "Sulfate": (0.0, 100_000.0),
    "Conductivity": (0.0, 1_000_000.0),
    "Organic_carbon": (0.0, 1_000.0),
    "Trihalomethanes": (0.0, 100_000.0),
    "Turbidity": (0.0, 10_000.0),
    "Calcium": (0.0, 100_000.0),
    "Magnesium": (0.0, 100_000.0),
    "Sodium": (0.0, 100_000.0),
    "Potassium": (0.0, 100_000.0),
    "Chloride": (0.0, 100_000.0),
    "Nitrate": (0.0, 100_000.0),
    "Fluoride": (0.0, 10_000.0),
    "Iron": (0.0, 10_000.0),
    "Alkalinity": (0.0, 100_000.0),
}

UNIT_LABELS = {
    "pH": "pH units",
    "Hardness": "mg/L as CaCO3 (verify source units)",
    "TDS": "mg/L (verify source units)",
    "Chloramines": "mg/L (verify source units)",
    "Sulfate": "mg/L (verify source units)",
    "Conductivity": "µS/cm (verify source units)",
    "Organic_carbon": "mg/L (verify source units)",
    "Trihalomethanes": "µg/L (verify source units)",
    "Turbidity": "NTU (verify source units)",
    "Calcium": "mg/L (verify source units)",
    "Magnesium": "mg/L (verify source units)",
    "Sodium": "mg/L (verify source units)",
    "Potassium": "mg/L (verify source units)",
    "Chloride": "mg/L (verify source units)",
    "Nitrate": "mg/L (verify source units)",
    "Fluoride": "mg/L (verify source units)",
    "Iron": "mg/L (verify source units)",
    "Alkalinity": "mg/L as CaCO3 (verify source units)",
}


class WaterMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    pH: FiniteFloat | None = Field(description="pH units; hard input range 0–14.")
    Hardness: FiniteFloat | None
    TDS: FiniteFloat | None = Field(
        validation_alias=AliasChoices("TDS", "Solids"),
        serialization_alias="TDS",
        description="Total dissolved solids; Solids is accepted as an input alias.",
    )
    Chloramines: FiniteFloat | None
    Sulfate: FiniteFloat | None
    Conductivity: FiniteFloat | None
    Organic_carbon: FiniteFloat | None
    Trihalomethanes: FiniteFloat | None
    Turbidity: FiniteFloat | None
    Calcium: FiniteFloat | None
    Magnesium: FiniteFloat | None
    Sodium: FiniteFloat | None
    Potassium: FiniteFloat | None
    Chloride: FiniteFloat | None
    Nitrate: FiniteFloat | None
    Fluoride: FiniteFloat | None
    Iron: FiniteFloat | None
    Alkalinity: FiniteFloat | None

    @field_validator("*", mode="before")
    @classmethod
    def require_json_number(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("All measurements must be JSON numbers, not strings or booleans.")
        return value

    @model_validator(mode="after")
    def validate_physical_input_bounds(self) -> "WaterMetrics":
        values = self.model_dump()
        if all(value is None for value in values.values()):
            raise ValueError("At least one measurement must be provided per sample.")
        for feature_name, value in values.items():
            if value is None:
                continue
            lower, upper = INPUT_BOUNDS[feature_name]
            numeric_value = float(value)
            if not lower <= numeric_value <= upper:
                raise ValueError(
                    f"{feature_name} must be between {lower:g} and {upper:g} "
                    f"({UNIT_LABELS[feature_name]}). These are input sanity bounds, "
                    "not BIS safe limits."
                )
        return self


def get_model_bundle() -> dict[str, object]:
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise RuntimeError("Unsupported or invalid model bundle schema.")
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise RuntimeError("Model feature schema does not match the API feature schema.")
    for key in ("classifier", "regressors", "regression_features"):
        if key not in bundle:
            raise RuntimeError(f"Model bundle is missing required key: {key}")
    return bundle


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH, timeout=10) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = get_model_bundle()
    initialize_database()
    yield


app = FastAPI(
    title="Aqua-Epoch Water Quality API",
    description=(
        "Prototype predictions for the supplied dataset labels. "
        "Predictions are not BIS compliance determinations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model artifact is unavailable.")
    return {"status": "ok", "model_artifact": MODEL_PATH.name}


def run_prediction(
    values: dict[str, float | None], bundle: dict[str, object]
) -> dict[str, object]:
    input_frame = pd.DataFrame([values], columns=FEATURE_COLUMNS).apply(
        pd.to_numeric, errors="coerce"
    )
    classifier_input = input_frame.copy()

    try:
        classifier = bundle["classifier"]
        predicted_class = int(classifier.predict(classifier_input)[0])
        class_estimator = classifier.named_steps["randomforestclassifier"]
        class_probabilities = classifier.predict_proba(classifier_input)[0]
        class_indexes = {
            int(label): index for index, label in enumerate(class_estimator.classes_)
        }
        if set(class_indexes) != {0, 1}:
            raise RuntimeError("Classifier does not expose expected classes 0 and 1.")

        regression_predictions: dict[str, float] = {}
        regressors = bundle["regressors"]
        regression_feature_map = bundle["regression_features"]
        for target in ("pH", "TDS"):
            regression_input = input_frame[regression_feature_map[target]].copy()
            prediction_value = float(regressors[target].predict(regression_input)[0])
            if not np.isfinite(prediction_value):
                raise RuntimeError(f"{target} regressor returned a non-finite value.")
            regression_predictions[target] = prediction_value

        result: dict[str, object] = {
            "potability": {
                "predicted_class": predicted_class,
                "predicted_label": "Safe" if predicted_class == 1 else "Unsafe",
                "probability_safe": float(class_probabilities[class_indexes[1]]),
                "probability_unsafe": float(class_probabilities[class_indexes[0]]),
                "label_semantics": "Dataset class mapping: 1=Safe, 0=Unsafe; verify source.",
            },
            "estimates": {
                "pH": regression_predictions["pH"],
                "TDS": regression_predictions["TDS"],
            },
            "compliance_status": "not_assessed",
            "warnings": [
                "Model predictions are based on supplied dataset labels and are not BIS certification.",
                "BIS thresholds and source units have not been verified/configured.",
                "Predicted pH/TDS are estimates; use validated laboratory measurements for decisions.",
                "Input sanity bounds are not drinking-water safety limits.",
            ],
            "model_created_at_utc": bundle.get("created_at_utc"),
        }
        return result
    except Exception as exc:
        raise RuntimeError(f"Prediction failed: {type(exc).__name__}: {exc}") from exc


def log_prediction(values: dict[str, float | None], result: dict[str, object]) -> None:
    try:
        with sqlite3.connect(DATABASE_PATH, timeout=10) as connection:
            connection.execute(
                "INSERT INTO predictions (created_at, input_json, result_json) "
                "VALUES (datetime('now'), ?, ?)",
                (
                    json.dumps(values, allow_nan=False),
                    json.dumps(result, allow_nan=False),
                ),
            )
            connection.commit()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Prediction history logging failed: {exc}") from exc


@app.post("/predict")
def predict(metrics: WaterMetrics) -> dict[str, object]:
    bundle = getattr(app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Start the API after training the model.",
        )
    values = metrics.model_dump(by_alias=False)
    try:
        result = run_prediction(values, bundle)
        log_prediction(values, result)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.post("/predict/batch")
def predict_batch(
    samples: Annotated[list[WaterMetrics], Body(min_length=1, max_length=500)],
) -> list[dict[str, object]]:
    bundle = getattr(app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Start the API after training the model.",
        )

    predictions: list[dict[str, object]] = []
    try:
        for row_number, sample in enumerate(samples, start=1):
            values = sample.model_dump(by_alias=False)
            result = run_prediction(values, bundle)
            predictions.append(
                {
                    "row_number": row_number,
                    "input": values,
                    "result": result,
                }
            )
        with sqlite3.connect(DATABASE_PATH, timeout=10) as connection:
            connection.executemany(
                "INSERT INTO predictions (created_at, input_json, result_json) "
                "VALUES (datetime('now'), ?, ?)",
                [
                    (
                        json.dumps(record["input"], allow_nan=False),
                        json.dumps(record["result"], allow_nan=False),
                    )
                    for record in predictions
                ],
            )
            connection.commit()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Batch predictions computed but history logging failed: {exc}",
        ) from exc

    return predictions

    return result


@app.get("/history")
def history(limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[dict[str, object]]:
    try:
        with sqlite3.connect(DATABASE_PATH, timeout=10) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, created_at, input_json, result_json
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail=f"Prediction history unavailable: {exc}"
        ) from exc

    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "input": json.loads(row["input_json"]),
            "result": json.loads(row["result_json"]),
        }
        for row in rows
    ]
