from __future__ import annotations

from io import BytesIO
import re

import numpy as np
import pandas as pd
import pdfplumber


HEADER_ALIASES = {
    "ph": "pH",
    "hardness": "Hardness",
    "tds": "TDS",
    "solids": "TDS",
    "totaldissolvedsolids": "TDS",
    "chloramines": "Chloramines",
    "sulfate": "Sulfate",
    "sulphate": "Sulfate",
    "conductivity": "Conductivity",
    "organiccarbon": "Organic_carbon",
    "trihalomethanes": "Trihalomethanes",
    "turbidity": "Turbidity",
    "calcium": "Calcium",
    "magnesium": "Magnesium",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "chloride": "Chloride",
    "nitrate": "Nitrate",
    "fluoride": "Fluoride",
    "iron": "Iron",
    "alkalinity": "Alkalinity",
}


def normalize_header(value: object) -> str:
    compact = re.sub(r"[^a-z0-9]", "", str(value).strip().lower())
    for alias in sorted(HEADER_ALIASES, key=len, reverse=True):
        if compact.startswith(alias):
            return HEADER_ALIASES[alias]
    return str(value).strip()


def read_csv_upload(file_bytes: bytes) -> pd.DataFrame:
    try:
        uploaded = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python")
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Could not read this CSV file: {exc}") from exc
    uploaded.columns = [normalize_header(column) for column in uploaded.columns]
    return uploaded


def read_pdf_upload(
    file_bytes: bytes,
    required_columns: set[str],
    max_pages: int,
) -> pd.DataFrame:
    extracted_tables: list[pd.DataFrame] = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            if len(pdf.pages) > max_pages:
                raise ValueError(f"PDF is over the {max_pages}-page limit.")
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    if not table:
                        continue
                    normalized_rows = [
                        [normalize_header(cell or "") for cell in row]
                        for row in table
                    ]
                    header_position = next(
                        (
                            index
                            for index, row in enumerate(normalized_rows[:8])
                            if required_columns.issubset(set(row))
                        ),
                        None,
                    )
                    if header_position is None:
                        continue
                    header = normalized_rows[header_position]
                    rows = [
                        (row + [""] * len(header))[: len(header)]
                        for row in normalized_rows[header_position + 1 :]
                    ]
                    if rows:
                        extracted_tables.append(pd.DataFrame(rows, columns=header))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse this PDF: {exc}") from exc

    if not extracted_tables:
        raise ValueError(
            "No selectable-text PDF table with all required model feature headers "
            "was found. Scanned/image-only PDFs and free-form reports are not supported."
        )
    return pd.concat(extracted_tables, ignore_index=True)


def prepare_uploaded_samples(
    uploaded: pd.DataFrame,
    input_bounds: dict[str, tuple[float, float]],
    max_rows: int,
) -> tuple[pd.DataFrame, list[str]]:
    if uploaded.empty:
        raise ValueError("The uploaded file contains no sample rows.")
    if uploaded.columns.duplicated().any():
        duplicates = uploaded.columns[uploaded.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate metric headers found: {duplicates}")

    missing_columns = sorted(set(input_bounds) - set(uploaded.columns))
    if missing_columns:
        raise ValueError(
            "Required model feature columns are missing: "
            + ", ".join(missing_columns)
            + f". Include all {len(input_bounds)} training features as column headers."
        )
    if len(uploaded) > max_rows:
        raise ValueError(
            f"Upload has {len(uploaded)} rows; maximum supported is {max_rows}."
        )

    samples = uploaded[list(input_bounds)].copy()
    errors: list[str] = []
    malformed_rows: set[int] = set()
    for column in input_bounds:
        original = samples[column]
        numeric = pd.to_numeric(original, errors="coerce")
        malformed = original.notna() & numeric.isna()
        for row_index in samples.index[malformed]:
            malformed_rows.add(int(row_index) + 1)
            errors.append(
                f"Row {row_index + 1}: {column} is not a valid numeric measurement."
            )
        samples[column] = numeric.replace([np.inf, -np.inf], np.nan)

    samples.attrs["malformed_rows"] = malformed_rows
    return samples, errors


def row_input_error(
    row_number: int,
    row: pd.Series,
    malformed_rows: set[int],
    input_bounds: dict[str, tuple[float, float]],
) -> str | None:
    if row_number in malformed_rows:
        return f"Row {row_number}: one or more measurements are not numeric."
    if row.isna().all():
        return f"Row {row_number}: all measurements are blank."
    for feature, value in row.items():
        if pd.isna(value):
            continue
        lower, upper = input_bounds[feature]
        if not np.isfinite(value):
            return f"Row {row_number}: {feature} must be finite."
        if not lower <= value <= upper:
            return (
                f"Row {row_number}: {feature}={value:g} is outside the accepted "
                f"input range [{lower:g}, {upper:g}]."
            )
    return None


def sanitize_download_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.select_dtypes(include="object").columns:
        safe[column] = safe[column].map(
            lambda value: (
                "'" + value
                if isinstance(value, str)
                and value.lstrip().startswith(("=", "+", "-", "@"))
                else value
            )
        )
    return safe
