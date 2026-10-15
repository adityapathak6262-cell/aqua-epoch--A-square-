from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
from upload_processing import (
    prepare_uploaded_samples,
    read_csv_upload,
    read_pdf_upload,
    row_input_error,
    sanitize_download_frame,
)


API_BASE_URL = os.getenv("AQUAEPOCH_API_URL", "http://127.0.0.1:8000").rstrip("/")

FEATURES = [
    ("pH", "pH units", "0–14 physical pH scale. This is not a verified compliance limit."),
    ("Hardness", "mg/L (verify units)", "Hardness reading. Verify dataset/laboratory units; no BIS limit configured."),
    ("TDS", "mg/L (verify units)", "Total dissolved solids; accepted API alias: Solids. Verify units; no BIS limit configured."),
    ("Chloramines", "mg/L (verify units)", "Chloramines concentration. Verify units and applicable standard before judging safety."),
    ("Sulfate", "mg/L (verify units)", "Sulfate concentration. Verify units and applicable standard before judging safety."),
    ("Conductivity", "µS/cm (verify units)", "Electrical conductivity. Verify measurement units and temperature compensation."),
    ("Organic_carbon", "mg/L (verify units)", "Organic carbon measurement. Confirm the method and units."),
    ("Trihalomethanes", "µg/L (verify units)", "Trihalomethanes measurement. Verify whether source data uses µg/L or another unit."),
    ("Turbidity", "NTU (verify units)", "Turbidity measurement. Verify the reporting unit and applicable standard."),
    ("Calcium", "mg/L (verify units)", "Calcium concentration. Confirm laboratory units."),
    ("Magnesium", "mg/L (verify units)", "Magnesium concentration. Confirm laboratory units."),
    ("Sodium", "mg/L (verify units)", "Sodium concentration. Confirm laboratory units."),
    ("Potassium", "mg/L (verify units)", "Potassium concentration. Confirm laboratory units."),
    ("Chloride", "mg/L (verify units)", "Chloride concentration. Confirm laboratory units."),
    ("Nitrate", "mg/L (verify units)", "Nitrate concentration. Confirm whether reported as nitrate or nitrate-nitrogen."),
    ("Fluoride", "mg/L (verify units)", "Fluoride concentration. Confirm laboratory units."),
    ("Iron", "mg/L (verify units)", "Iron concentration. Confirm laboratory units."),
    ("Alkalinity", "mg/L as CaCO3 (verify units)", "Alkalinity reading. Confirm laboratory units and reporting basis."),
]

INPUT_BOUNDS = {
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

DEFAULTS = {
    "pH": 7.1,
    "Hardness": 196.8,
    "TDS": 20963.9,
    "Chloramines": 7.1,
    "Sulfate": 332.2,
    "Conductivity": 423.4,
    "Organic_carbon": 16.7,
    "Trihalomethanes": 66.2,
    "Turbidity": 4.0,
    "Calcium": 70.0,
    "Magnesium": 30.0,
    "Sodium": 80.0,
    "Potassium": 7.0,
    "Chloride": 150.0,
    "Nitrate": 25.0,
    "Fluoride": 1.0,
    "Iron": 0.17,
    "Alkalinity": 147.0,
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 30_000
MAX_PDF_PAGES = 20

st.set_page_config(
    page_title="Aqua-Epoch | Water Quality",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("💧 Aqua-Epoch Water Quality Prototype")

with st.expander("Upload sample data (CSV or text-based PDF)", expanded=False):
    st.caption(
        "Upload a table with headers for all 18 trained model features. "
        "CSV and PDFs containing selectable text/tables are supported; scanned PDFs "
        f"are not. Maximum file size: 10 MB; maximum rows: {MAX_UPLOAD_ROWS:,}. "
        "Blank cells are median-imputed by the trained pipeline."
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV or PDF",
        type=["csv", "pdf"],
        key="water_sample_upload",
    )

    if uploaded_file is not None:
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            st.error("Maximum upload size is 10 MB.")
        else:
            try:
                file_bytes = uploaded_file.getvalue()
                if uploaded_file.name.lower().endswith(".csv"):
                    uploaded_data = read_csv_upload(file_bytes)
                else:
                    uploaded_data = read_pdf_upload(
                        file_bytes,
                        required_columns=set(INPUT_BOUNDS),
                        max_pages=MAX_PDF_PAGES,
                    )
                sample_frame, parsing_errors = prepare_uploaded_samples(
                    uploaded_data,
                    input_bounds=INPUT_BOUNDS,
                    max_rows=MAX_UPLOAD_ROWS,
                )
                st.write(f"Detected **{len(sample_frame)}** sample row(s).")
                if parsing_errors:
                    st.error("\n".join(parsing_errors[:20]))
                    if len(parsing_errors) > 20:
                        st.caption(f"{len(parsing_errors) - 20} additional parse errors omitted.")
                st.dataframe(sample_frame.head(10), use_container_width=True)

                if len(sample_frame) > MAX_UPLOAD_ROWS:
                    st.error(f"Upload exceeds the {MAX_UPLOAD_ROWS}-row limit.")
                elif st.button("Analyze uploaded samples", type="primary"):
                    result_rows: list[dict[str, object]] = []
                    valid_rows: list[tuple[int, dict[str, float | None]]] = []
                    for row_offset, (_, row) in enumerate(
                        sample_frame.iterrows(), start=1
                    ):
                        error = row_input_error(
                            row_offset,
                            row,
                            sample_frame.attrs.get("malformed_rows", set()),
                            INPUT_BOUNDS,
                        )
                        row_result: dict[str, object] = {
                            "Sample row": row_offset,
                            **row.to_dict(),
                            "Status": "Prediction pending",
                        }
                        if error:
                            row_result["Status"] = error
                            result_rows.append(row_result)
                            continue
                        valid_rows.append(
                            (
                                row_offset,
                                {
                                    feature: (
                                        None if pd.isna(value) else float(value)
                                    )
                                    for feature, value in row.items()
                                },
                            )
                        )
                        result_rows.append(row_result)

                    predictions_by_row: dict[int, dict[str, object]] = {}
                    for start in range(0, len(valid_rows), 100):
                        chunk = valid_rows[start : start + 100]
                        try:
                            batch_response = requests.post(
                                f"{API_BASE_URL}/predict/batch",
                                json=[sample for _, sample in chunk],
                                timeout=120,
                            )
                            batch_response.raise_for_status()
                            for prediction in batch_response.json():
                                original_row = chunk[prediction["row_number"] - 1][0]
                                predictions_by_row[original_row] = prediction["result"]
                        except requests.RequestException as exc:
                            st.error(
                                f"Batch prediction request failed: {exc}. "
                                "Check that the FastAPI backend is running."
                            )
                            break
                        except (ValueError, KeyError, IndexError) as exc:
                            st.error(f"Could not read backend prediction response: {exc}")
                            break

                    output_rows = []
                    for row_result in result_rows:
                        row_number = int(row_result["Sample row"])
                        prediction = predictions_by_row.get(row_number)
                        if prediction is not None:
                            potability = prediction["potability"]
                            row_result.update(
                                {
                                    "Status": "Predicted",
                                    "Predicted class": potability["predicted_class"],
                                    "Predicted label": potability["predicted_label"],
                                    "Probability safe": potability["probability_safe"],
                                    "Probability unsafe": potability["probability_unsafe"],
                                    "Estimated pH": prediction["estimates"]["pH"],
                                    "Estimated TDS": prediction["estimates"]["TDS"],
                                }
                            )
                        elif row_result["Status"] == "Prediction pending":
                            row_result["Status"] = (
                                "Prediction unavailable; check the backend/API response."
                            )
                        output_rows.append(row_result)

                    results_frame = pd.DataFrame(output_rows)
                    st.subheader("Uploaded sample predictions")
                    st.dataframe(results_frame, use_container_width=True, hide_index=True)
                    predicted_frame = results_frame[
                        results_frame["Status"] == "Predicted"
                    ]
                    if not predicted_frame.empty:
                        if len(predicted_frame) <= 200:
                            st.bar_chart(
                                predicted_frame.set_index("Sample row")[
                                    ["Probability safe", "Probability unsafe"]
                                ]
                            )
                        else:
                            chart_columns = st.columns(2)
                            class_counts = predicted_frame[
                                "Predicted label"
                            ].value_counts()
                            chart_columns[0].caption("Predicted class counts")
                            chart_columns[0].bar_chart(class_counts)
                            histogram, edges = np.histogram(
                                predicted_frame["Probability safe"], bins=10, range=(0, 1)
                            )
                            probability_distribution = pd.DataFrame(
                                {"Samples": histogram},
                                index=[
                                    f"{edges[i]:.1f}–{edges[i + 1]:.1f}"
                                    for i in range(len(histogram))
                                ],
                            )
                            chart_columns[1].caption("Safe-class probability distribution")
                            chart_columns[1].bar_chart(probability_distribution)
                    csv_bytes = sanitize_download_frame(results_frame).to_csv(
                        index=False
                    ).encode("utf-8-sig")
                    st.download_button(
                        "Download results as CSV",
                        data=csv_bytes,
                        file_name="aqua_epoch_predictions.csv",
                        mime="text/csv",
                    )
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(str(exc))

st.subheader("Enter water-quality parameters")

with st.form("water_metrics_form"):
    columns = st.columns(3)
    submitted_values: dict[str, float] = {}
    for index, (name, unit, tooltip) in enumerate(FEATURES):
        with columns[index % 3]:
            lower_bound, upper_bound = INPUT_BOUNDS[name]
            submitted_values[name] = st.number_input(
                label=f"{name} ({unit})",
                min_value=lower_bound,
                max_value=upper_bound,
                value=float(DEFAULTS[name]),
                step=0.1,
                format="%.4f",
                help=tooltip,
            )
    submitted = st.form_submit_button(
        "Analyze sample",
        type="primary",
        use_container_width=True,
    )

if submitted:
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=submitted_values,
            timeout=30,
        )
        if response.status_code == 422:
            st.error(f"Input validation failed: {response.text}")
            st.stop()
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
        st.stop()
    except ValueError as exc:
        st.error(f"Backend returned invalid JSON: {exc}")
        st.stop()

    potability = result["potability"]
    label = potability["predicted_label"]
    if label == "Safe":
        st.success(f"Model prediction: {label}")
    else:
        st.error(f"Model prediction: {label}")

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Probability: class 1 (project mapping: Safe)",
        f"{potability['probability_safe']:.1%}",
    )
    metric_columns[1].metric(
        "Probability: class 0 (project mapping: Unsafe)",
        f"{potability['probability_unsafe']:.1%}",
    )
    metric_columns[2].metric("Estimated pH", f"{result['estimates']['pH']:.3f}")
    metric_columns[3].metric("Estimated TDS", f"{result['estimates']['TDS']:.2f}")

    probability_frame = pd.DataFrame(
        {
            "Dataset class": ["Unsafe (0)", "Safe (1)"],
            "Model probability": [
                potability["probability_unsafe"],
                potability["probability_safe"],
            ],
        }
    ).set_index("Dataset class")
    st.subheader("Classification probability")
    st.bar_chart(probability_frame)

st.divider()
st.subheader("Recent prediction history")
try:
    history_response = requests.get(f"{API_BASE_URL}/history?limit=100", timeout=5)
    history_response.raise_for_status()
    history_records = history_response.json()
    if history_records:
        history_rows = []
        for item in reversed(history_records):
            result = item.get("result")
            result = result if isinstance(result, dict) else {}
            potability = result.get("potability")
            potability = potability if isinstance(potability, dict) else {}
            estimates = result.get("estimates")
            estimates = estimates if isinstance(estimates, dict) else {}
            history_rows.append(
                {
                    "Prediction ID": item["id"],
                    "Created at": item["created_at"],
                    "Prediction status": (
                        "Available"
                        if potability.get("predicted_label") is not None
                        else "Unavailable"
                    ),
                    "Predicted label": potability.get("predicted_label"),
                    "pH estimate": estimates.get("pH"),
                    "TDS estimate": estimates.get("TDS"),
                }
            )
        history_frame = pd.DataFrame(history_rows)
        st.dataframe(history_frame, use_container_width=True, hide_index=True)
        trend = history_frame.set_index("Prediction ID")[
            ["pH estimate", "TDS estimate"]
        ].dropna(how="all")
        if not trend.empty:
            st.line_chart(trend)
    else:
        st.caption("No prediction records yet. Submit a sample to start local history.")
except requests.RequestException as exc:
    st.warning(f"Prediction history is currently unavailable: {exc}")
