# Q-Diagnose REST API Specification

> **Platform:** Q-Diagnose: Patient Lab Report Review and ML Research Platform  
> **Base URL:** `http://127.0.0.1:8000` (or `http://localhost:8000`)  
> **Interactive Documentation:** Disabled in this build  
> **Compliance:** RFC-8259 JSON, Zero-Leakage Pipeline Architecture  

---

## 1. System & Platform Endpoints

### `GET /health`
Returns service status and separates research-cohort models from patient-report capabilities. `supported_models` is retained for existing clients; `model_scope` clarifies that these models are for research datasets and are not connected to uploaded patient reports.

#### Request
```bash
curl -X GET http://127.0.0.1:8000/health
```

#### Response (HTTP 200)
```json
{
  "platform": "Q-Diagnose",
  "tagline": "Patient Lab Report Review and ML Research Platform",
  "status": "operational",
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device.",
  "supported_models": [
    "logistic_regression",
    "svm",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "qsvc",
    "vqc",
    "qnn"
  ],
  "model_scope": "research_cohorts_only",
  "capabilities": {
    "research_experiments": {
      "status": "available",
      "models": ["logistic_regression", "svm", "random_forest", "extra_trees", "hist_gradient_boosting", "xgboost", "qsvc", "vqc", "qnn"],
      "description": "Run and compare models on labeled research datasets."
    },
    "patient_lab_reports": {
      "status": "biomarker_review_only",
      "supported_file_types": ["PDF", "DOCX", "CSV", "TXT", "JSON"],
      "disease_prediction": "unavailable",
      "description": "Extract recognizable lab values and compare them with general reference intervals. No validated disease-specific ML or quantum model is connected to this workflow."
    }
  }
}
```

### `GET /backends` (or `GET /api/backends`)
Lists available classical compute resources and quantum simulation/QPU backends.

#### Request
```bash
curl -X GET http://127.0.0.1:8000/api/backends
```

#### Real Captured Response (HTTP 200)
```json
{
  "backends": [
    {
      "id": "ideal_simulator",
      "name": "Qiskit Statevector Simulator (Ideal / Noiseless)",
      "type": "quantum_simulator",
      "qubits_supported": 10,
      "noise_model": null,
      "description": "Full statevector linear algebra simulation representing ideal coherent unitary evolution."
    },
    {
      "id": "noisy_simulator",
      "name": "Qiskit Aer Depolarizing Simulator",
      "type": "quantum_noisy_simulator",
      "qubits_supported": 8,
      "noise_model": "DepolarizingChannel(p=0.02) + ReadoutError",
      "description": "Empirical NISQ hardware simulation with thermal relaxation and two-qubit gate noise."
    },
    {
      "id": "classical_cpu",
      "name": "Host Multicore Classical Processing Unit",
      "type": "classical_hardware",
      "threads": 8,
      "description": "OpenMP/BLAS classical hardware acceleration for classical baselines."
    }
  ],
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
}
```

### `GET /comparisons` (or `GET /api/comparisons`)
Serves the empirical model tuning and paired $t$-test statistical significance study matrix across classical and quantum models.

#### Request
```bash
curl -X GET http://127.0.0.1:8000/api/comparisons
```

#### Real Captured Response (HTTP 200)
```json
{
  "status": "success",
  "comparison_summary": [
    {
      "Model / Intervention": "SVM (RBF)",
      "Metric": "Accuracy",
      "Before": 0.9579,
      "After": 0.9526,
      "Change": "-0.53%",
      "p-value": 0.2073,
      "Statistically Significant?": "No (p = 0.207)"
    },
    {
      "Model / Intervention": "SelectKBest vs PCA (SVM)",
      "Metric": "Accuracy",
      "Before": 0.9579,
      "After": 0.9385,
      "Change": "-1.93%",
      "p-value": 0.0108,
      "Statistically Significant?": "Yes (p < 0.05)"
    }
  ],
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
}
```

---

## 2. Dataset Management Endpoints

### `GET /datasets`
Lists all registered datasets stored in the local SQLite metadata database with schema and class distributions.

#### Request
```bash
curl -X GET http://127.0.0.1:8000/datasets
```

#### Real Captured Response (HTTP 200)
```json
{
  "datasets": [
    {
      "id": "breast_cancer_wdbc",
      "name": "UCI Breast Cancer Wisconsin (Diagnostic)",
      "filename": "wdbc.csv",
      "filepath": "uploads/wdbc.csv",
      "n_rows": 569,
      "n_cols": 31,
      "target_col": "diagnosis",
      "missing_count": 0,
      "duplicate_count": 0,
      "class_distribution": {
        "Malignant (0)": 212,
        "Benign (1)": 357
      },
      "created_at": "2026-09-25T09:22:10.010279"
    }
  ],
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
}
```

### `POST /datasets/upload`
Uploads and validates a clinical CSV dataset. Validates binary classification target and checks missing and duplicate values.

#### Request
```bash
curl -X POST http://127.0.0.1:8000/datasets/upload \
  -F "file=@patient_cohort.csv" \
  -F "name=Cohort 2026 Trial" \
  -F "target_col=diagnosis"
```

---

## 3. Threshold Tuning & ROC Sweep (Gap 1)

### `GET /experiments/{id}/metrics-at-threshold?threshold={value}`
Computes the confusion matrix, sensitivity, specificity, precision, and F1 score at an exact operating threshold $\tau \in [0.01, 0.99]$ from stored held-out test predictions without retraining.

#### Request
```bash
curl -X GET "http://127.0.0.1:8000/experiments/exp_qsvc_calibrated/metrics-at-threshold?threshold=0.35"
```

#### Real Captured Response (HTTP 200)
```json
{
  "experiment_id": "exp_qsvc_calibrated",
  "model_type": "qsvc",
  "threshold": 0.35,
  "confusion_matrix": {
    "tn": 23,
    "fp": 48,
    "fn": 2,
    "tp": 117
  },
  "metrics": {
    "accuracy": 0.7368421052631579,
    "sensitivity": 0.9831932773109243,
    "specificity": 0.323943661971831,
    "precision": 0.7090909090909091,
    "f1": 0.8239436619718309
  },
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
}
```

### `GET /experiments/{id}/roc-curve`
Returns a dense 100-point operating sweep across thresholds ($0.01$ to $0.99$) showing exact TPR, FPR, Sensitivity, Specificity, and confusion matrices per step.

#### Request
```bash
curl -X GET "http://127.0.0.1:8000/experiments/exp_qsvc_calibrated/roc-curve"
```

#### Real Captured Response (HTTP 200)
```json
{
  "experiment_id": "exp_qsvc_calibrated",
  "model_type": "qsvc",
  "count": 100,
  "threshold_sweep": [
    {
      "threshold": 0.01,
      "tpr": 1.0,
      "fpr": 1.0,
      "sensitivity": 1.0,
      "specificity": 0.0,
      "precision": 0.6263,
      "f1": 0.7702,
      "accuracy": 0.6263,
      "confusion_matrix": {
        "tn": 0,
        "fp": 71,
        "fn": 0,
        "tp": 119
      }
    }
  ],
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
}
```

---

## 4. Single-Sample Research Inference

### `POST /predict`
This legacy endpoint is disabled with HTTP 410. Research cohort models are not validated for individual patient records, and uploaded lab reports are not inputs to these models. Use the research experiment endpoints for labeled cohort evaluation only.

#### Request
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "breast_cancer_wdbc",
    "model_type": "qsvc",
    "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
    "threshold": 0.5
  }'
```

#### Current response
```json
{
  "detail": "Patient-specific disease scoring is disabled. Research cohort models and benchmark scores are not validated for individual patient records."
}
```

---

## 5. Experiment Execution & Robustness Endpoints

### `POST /experiments/run`
Executes Stratified K-Fold Cross-Validation over multiple seeds with zero data leakage.

```json
{
  "dataset_id": "breast_cancer_wdbc",
  "model_type": "qsvc",
  "n_features": 4,
  "backend": "noisy_simulator",
  "circuit_depth": 2,
  "cv_folds": 5,
  "seeds": [42, 123, 7]
}
```

### `GET /experiments/robustness-study`
Returns pre-computed depolarizing noise sweeps ($p = 0.00$ to $0.10$) and circuit depth sweeps ($d = 1$ to $4$) evaluating quantum kernel robustness under simulated decoherence.

---

## 6. Patient Details & Longitudinal Tracking Endpoints (Part A)

### `POST /patients`
Registers a new de-identified patient record.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/patients" \
  -H "Content-Type: application/json" \
  -d '{"name": "Subject-SIH-9921", "age": 54, "sex": "Female", "notes": "Post-screening follow-up"}'
```

**Response (200 OK):**
```json
{
  "status": "success",
  "patient_id": "25a9a6b1-4e40-4b89-b431-8018852c6154",
  "message": "Patient '25a9a6b1-4e40-4b89-b431-8018852c6154' created successfully.",
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
}
```

### `POST /patients/{patient_id}/records`
Appends a multi-feature clinical visit record.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/patients/25a9a6b1-4e40-4b89-b431-8018852c6154/records" \
  -H "Content-Type: application/json" \
  -d '{"feature_values": [17.99, 10.38, 122.8, ...], "source": "Outpatient Clinic Alpha"}'
```

### `POST /patients/{patient_id}/predict`
Returns HTTP 410. Patient-specific disease scoring is disabled; research cohort models are not validated for individual patient records.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/patients/25a9a6b1-4e40-4b89-b431-8018852c6154/predict" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "svm", "threshold": 0.5}'
```

### `GET /patients/{patient_id}/trend`
Returns record count only with an empty `trend` and `prediction_status: "unavailable_until_validated"`. It does not return historical risk scores.

**Response (200 OK):**
```json
{
  "trend_data": {
    "patient_id": "25a9a6b1-4e40-4b89-b431-8018852c6154",
    "total_records": 3,
    "trend": [],
    "prediction_status": "unavailable_until_validated"
  },
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
}
```

---

## 6. Patient Lab Report Extraction and Review Endpoints

### `GET /api/lab-reports/presets`
Returns the supported biomarker schema. Demonstration patient reports are not exposed by this build; the `presets` field remains as an empty array for compatibility.

#### Request
```bash
curl -X GET http://127.0.0.1:8000/api/lab-reports/presets
```

#### Response (HTTP 200)
```json
{
  "status": "success",
  "presets": [],
  "message": "No demonstration patient reports are provided in this build.",
  "biomarker_definitions": { ... },
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
}
```

---

### `POST /api/lab-reports/analyze`
Extracts recognizable biomarkers from report text or accepts a biomarker map, then returns parsed values and general reference-range flags. `disease_prediction_status` is `unavailable_until_validated`; the empty legacy disease fields must not be read as a negative diagnosis.

#### Request
```bash
curl -X POST http://127.0.0.1:8000/api/lab-reports/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "report_title": "<report title>",
    "biomarkers": {}
  }'
```

#### Response (HTTP 200)
```json
{
  "status": "success",
  "report_title": "<report title>",
  "biomarkers_extracted_count": 0,
  "abnormal_biomarkers_count": 0,
  "parsed_biomarkers": {},
  "flagged_biomarkers": [],
  "diseases": [],
  "primary_suspected_disease": "Disease scoring unavailable",
  "primary_chance_percentage": null,
  "primary_risk_tier": "insufficient_data",
  "primary_risk_category": "Insufficient Data",
  "shap_explanations": [],
  "disease_prediction_status": "unavailable_until_validated",
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
}
```

---

### `POST /api/lab-reports/upload`
Uploads a lab document (`.pdf`, `.docx`, `.csv`, `.txt`, or `.json`) for value extraction and reference review. It does not predict disease.

#### Request
```bash
curl -X POST http://127.0.0.1:8000/api/lab-reports/upload \
  -F "file=@sample_reports/cardiac_lipid_panel.txt" \
  -F "report_title=Inpatient Cardiac Panel"
```

---

### `POST /api/risk/predict` and `GET /api/risk/{report_id}/explanation`
Both legacy endpoints return HTTP 410. They do not produce a disease score or model explanation for an individual report.

### `POST /api/lab-reports/save-to-patient`
Saves extracted lab values to a local research patient record. It creates no disease prediction; non-null prediction fields are rejected with HTTP 422.

#### Request
```bash
curl -X POST http://127.0.0.1:8000/api/lab-reports/save-to-patient \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PT-001",
    "report_title": "<report title>",
    "biomarkers": {}
  }'
```

#### Response (HTTP 200)
```json
{
  "status": "success",
  "message": "Lab report 'Comprehensive Lipid Panel' successfully saved to patient 'PT-001'.",
  "record_id": "rec_8a49c2ef1240",
  "prediction_id": null,
  "disclaimer": "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
}
```

