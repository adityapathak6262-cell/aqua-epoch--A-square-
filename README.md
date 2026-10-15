# The Aqua-Epoch — Water Quality Intelligence Prototype

[![techFEST'26](https://img.shields.io/badge/techFEST'26-SLIET%20Longowal-blue.svg)](http://techfest26.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning solution and interactive web application developed for **The Aqua-Epoch** challenge at **techFEST'26**, Sant Longowal Institute of Engineering and Technology (SLIET Longowal).

The system analyzes physicochemical water parameters, predicts potability safety with high precision (minimizing false positives), continuously estimates pH levels, and provides real-time visualization and historical trend analysis.

---

## 👥 Team Information

- **Team Name:** A-Square
- **Team Members:**
  - **Aditya Pathak** 
  — **Aaditya Dubey **
- **Contact:** adityapathak6262@gmail.com
- **Repository:** [https://github.com/adityapathak6262-cell/aqua-epoch--A-square-](https://github.com/adityapathak6262-cell/aqua-epoch--A-square-)

---

## 🎯 Problem Statement & Predictive Tasks

Access to safe drinking water is a critical public health and environmental challenge. Modern sensor networks capture multidimensional physical and chemical water parameters, but automated interpretation is essential for rapid contamination detection.

### Task A — Classification: Water Safety / Potability
- **Objective:** Classify water samples as **Safe (1)** or **Unsafe (0)** based on Bureau of Indian Standards (BIS IS 10500:2012) metrics.
- **Handling Challenges:** Missing data imputed using robust median pipelines; class imbalance handled via cost-sensitive learning (`class_weight='balanced'`).
- **Critical Focus:** **Minimizing False Positives** (preventing contaminated water from being erroneously classified as safe).
- **Evaluation Metrics:** **F1-Score: 0.9912**, **ROC-AUC: 0.9997**, **Precision: 0.9876**, **Recall: 0.9949**.

### Task B — Regression: pH Estimation
- **Objective:** Predict continuous water pH from inter-related chemical features (Turbidity, Hardness, TDS, Sulfate, Chloramines, etc.).
- **Evaluation Metrics:** **MAE: 0.4941**, **RMSE: 0.6772**.

---

## 🏗️ System Architecture & Workflow

The solution follows a decoupled, low-latency client-server architecture:

```
┌────────────────────────────────────────────────────────┐
│               Streamlit Web Interface                  │
│  - Core metrics form (Turbidity, Hardness, TDS)        │
│  - BIS IS 10500 safe limit tooltips                    │
│  - Client-side input validation & physical ranges      │
│  - Real-time probability & parameter visualizers       │
│  - Batch upload support (CSV & PDF tables)             │
│  - Historical trend visualization                      │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP REST API (JSON)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│  - POST /predict (real-time sample inference)          │
│  - POST /predict/batch (bulk sample processing)        │
│  - GET  /history (trend retrieval)                     │
│  - GET  /health  (model health check)                  │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌────────────────────────┐
│  Trained ML Pipeline  │       │  SQLite Database       │
│  - model.pkl          │       │  - prediction_history  │
│  - Median Imputation  │       │  - Bonus logging       │
│  - Random Forest      │       │  - Trend tracking      │
└───────────────────────┘       └────────────────────────┘
```

---

## 📊 Model Performance Summary

Evaluated on an 80/20 held-out stratified test set (4,523 test samples):

| Task | Metric | Value |
| :--- | :--- | :--- |
| **Task A: Classification** | **F1-Score** | **0.9912** |
| | **ROC-AUC** | **0.9997** |
| | **Accuracy** | **0.9931 (99.31%)** |
| | **Precision** | **0.9876** |
| | **Recall** | **0.9949** |
| | **False Positives** | **22 out of 2,768 unsafe samples (0.79%)** |
| **Task B: pH Regression** | **MAE** | **0.4941** |
| | **RMSE** | **0.6772** |

---

## 📁 Repository Structure

```
aquaepoch/
│
├── app.py                     # Streamlit frontend application
├── main.py                    # FastAPI REST backend service
├── train_model.py             # Model training & serialization script
├── upload_processing.py       # Batch upload parsing (CSV & PDF tables)
├── generate_ppt_results.py    # Metric calculation & visualization generator
├── Aqua_Epoch_Phase4_Analysis.ipynb # Jupyter notebook with EDA & experiments
├── model.pkl                  # Serialized Scikit-Learn model bundle
├── requirements.txt           # Python library dependencies
├── README.md                  # Project overview & instructions
└── .gitignore                 # Git ignore rules
```

---

## ⚡ How to Run Locally

### 1. Clone the Repository & Setup Environment

```bash
git clone https://github.com/adityapathak6262-cell/aqua-epoch--A-square-.git
cd aqua-epoch--A-square-

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend Server

In your first terminal, launch the REST API:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 3. Start the Streamlit Frontend Web App

In a second terminal (with the virtual environment activated):

```bash
streamlit run app.py
```
- Web Application: [http://localhost:8501](http://localhost:8501)

---

## 🔄 Retraining the Models (Optional)

The pre-trained deployment model is serialized in `model.pkl`. To retrain from scratch:

1. Place the dataset `Aqua_Epoch_BIS_Water_Quality_Dataset.xls - Aqua_Epoch_BIS_Water_Quality_Dataset.csv` in the root folder.
2. Run:
   ```bash
   python train_model.py
   ```

---

## 📜 Regulatory Standards Reference

Safe guidelines in the tooltips reflect **Bureau of Indian Standards (BIS IS 10500:2012 Drinking Water Specification)**:
