# Aqua-Epoch Water Quality Prototype

AI-powered water-quality analysis prototype for the SLIET techFEST '26 Aqua-Epoch challenge. It includes a trained Random Forest classification model, pH and TDS regression models, a FastAPI backend, and a Streamlit web application.

## Team

- **Team name:** [Enter team name]
- **Team members:** [Enter member names]
- **Contact:** team@aquaepoch.com

Replace the placeholders above with your registered team details before presenting.

## What's included

- `model.pkl` — serialized classifier and pH/TDS regressors used by the application.
- `main.py` — FastAPI prediction, batch prediction, health, and SQLite history endpoints.
- `app.py` — Streamlit interface for manual measurements and CSV/text-based PDF uploads.
- `upload_processing.py` — upload parsing, feature/header mapping, and row validation.
- `train_model.py` — reproducible model training and evaluation script.
- `generate_ppt_results.py` — generates evaluation metrics and presentation plots.
- `requirements.txt` — Python dependencies.

The source dataset, local prediction history, virtual environment, and generated presentation images are not included in this repository.

## Requirements

- Python 3.10 or later
- pip

## Run the web application

Open a terminal in the project directory and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the API in the first terminal:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Start Streamlit in a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open **http://localhost:8501** in a browser. The API health endpoint is **http://127.0.0.1:8000/health**, and interactive API documentation is at **http://127.0.0.1:8000/docs**.

The app supports individual measurements and batch CSV or selectable-text PDF input. Uploads need headers for all 18 model features; scanned PDFs and OCR are not supported. Prediction history is stored locally in `prediction_history.sqlite3`.

## Train the models again

The trained bundle is included, so retraining is not needed to run the app. To train a fresh bundle, place the challenge CSV in the project root with this exact filename:

```text
Aqua_Epoch_BIS_Water_Quality_Dataset.xls - Aqua_Epoch_BIS_Water_Quality_Dataset.csv
```

Then run:

```powershell
python train_model.py
```

This evaluates on a stratified 80/20 classification split with `random_state=42`, trains the deployment models, and writes `model.pkl`. The dataset itself is not committed; obtain it from the official challenge/source and confirm its license and measurement units before use.

## Prediction scope

The classifier predicts the supplied dataset's `Potability` label (mapped in the project to `0 = Unsafe`, `1 = Safe`). Verify that mapping against the dataset source before relying on it. The pH and TDS regressors estimate those values from the other measured features.

This is a prototype, not a BIS compliance assessment or certification. Configured input bounds are broad sanity checks, not safety thresholds. Verify source units, labels, and model performance with qualified laboratory measurements before any real-world decision.
