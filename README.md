# Q-Diagnose

Q-Diagnose is a local research dashboard for comparing classical machine-learning and quantum-machine-learning approaches on **binary, tabular datasets**. It includes CSV intake, experiment tracking, cross-validation metrics, parameter sweeps, calibration, and model explanations.

> **Research prototype only.** This software is not a clinically validated medical device. It does not diagnose a patient, predict future illness from an individual lab report, or provide medical advice. Results are experimental and must not be used to guide patient care.

## What is included

- **Classical baselines:** logistic regression, support-vector machine, random forest, and XGBoost.
- **Quantum models:** QSVC, variational quantum classifier (VQC), and quantum neural network (QNN), using Qiskit simulators.
- **Backends:** ideal and noisy simulators. Real-hardware execution is not implemented; the application reports that it is unavailable rather than silently substituting a simulator.
- **Data workflows:** built-in breast-cancer and diabetes datasets, plus validated CSV uploads with a binary target.
- **Evaluation:** stratified folds and multiple random seeds, accuracy, precision, sensitivity, specificity, F1, ROC-AUC, PR-AUC, confusion matrices, and probability calibration.
- **Research tools:** parameter sweeps, cross-dataset comparisons, classical SHAP explanations, quantum sensitivity analysis, and circuit previews.
- **Persistence:** datasets and experiments are stored in a local SQLite database.

The diabetes example is derived from a regression dataset by splitting its target at the median. It is a demonstration benchmark, not a diabetes diagnosis dataset.

## Requirements

- Python 3.11 or newer
- Node.js 20 LTS or 22 LTS and npm

The Python and frontend dependencies are pinned in `requirements.txt` and `frontend/package-lock.json`.

## Run locally

From the repository root, create and activate a virtual environment, then install the Python packages:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Build the dashboard and start the API server from the repository root:

```bash
cd frontend
npm ci
npm run build
cd ..
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The API reference is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), and the health endpoint is [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health).

The API serves the built frontend from `frontend/dist`. Rebuild it after frontend changes. For frontend development with hot reload, keep the API running on port `8000` and, in another terminal, run:

```bash
cd frontend
npm ci
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite proxies `/api` requests to the local API server.

## Use the dashboard

1. Open the dashboard and add the built-in breast-cancer dataset, the diabetes demonstration dataset, or upload a CSV.
2. Choose a dataset and configure the model, backend, feature count, cross-validation folds, random seeds, and quantum circuit settings as applicable.
3. Run the experiment and review its saved metrics, curves, confusion matrix, calibration, and available explanations.
4. Use parameter sweeps and cross-dataset comparisons to explore how a model behaves under different research settings.

For uploaded CSVs, Q-Diagnose selects a target column named `target`, `label`, `diagnosis`, `class`, or `outcome` when present; otherwise it uses the last column. You may specify the target explicitly. The target must contain exactly two classes, and the upload limit is 10 MiB. Categorical predictors are encoded by the preprocessing pipeline.

Preprocessing is fitted on each training fold before transforming its validation fold to avoid fitting those transformations on validation data. Results still depend on the dataset, sample size, split, and selected settings; they are not evidence of clinical performance.

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service, dataset, experiment, and hardware status |
| `GET` | `/api/models` | Supported model and metric names |
| `GET` | `/api/backends` | Simulator and hardware availability |
| `GET` | `/api/datasets` | List registered datasets |
| `POST` | `/api/datasets/builtin` | Register the breast-cancer benchmark |
| `POST` | `/api/datasets/builtin/diabetes` | Register the median-split diabetes demonstration dataset |
| `POST` | `/api/datasets/upload` | Validate and register a CSV (`multipart/form-data`) |
| `POST` | `/api/experiments` | Run and save an experiment |
| `GET` | `/api/experiments` | List saved experiments |
| `GET` | `/api/experiments/{id}` | Read an experiment and its artifacts |
| `POST` | `/api/sweeps` | Run a parameter sweep |
| `POST` | `/api/phase5/validate` | Compare selected models across datasets |

The interactive OpenAPI documentation at `/docs` includes request schemas and the complete endpoint list.

## Reproducibility and configuration

Experiment records include their settings, pipeline revision, fold results, runtime, metrics, and available holdout artifacts. The default seed list is `42, 123, 2025`; the API requires at least three seeds. Quantum models support at most eight PCA features.

Optional settings are documented in `.env.example`. Copy it to `.env` if needed; never commit a real token or secret. Uploaded CSVs are stored under `data/uploads` by default. Set `QDIAGNOSE_UPLOAD_DIR` to choose a different directory. The SQLite database is `q_diagnose.db`; generated data, uploads, databases, and logs are excluded by `.gitignore`.

## Development checks

```bash
python -m pytest
```

Build the frontend with `cd frontend && npm ci && npm run build`.

## Repository layout

```text
.
├── main.py                 # FastAPI application and REST endpoints
├── datasets.py             # CSV validation and dataset registration
├── pipeline.py             # Fold-local preprocessing
├── model_registry.py       # Classical and quantum model construction
├── experiment_runner.py    # Experiment execution and artifact generation
├── evaluate.py             # Metrics and evaluation utilities
├── explanations.py         # Classical and quantum explanations
├── phase5.py               # Cross-dataset validation workflow
├── storage.py              # SQLite persistence
├── frontend/               # React, TypeScript, and Vite dashboard
└── test_*.py               # Python test suites
```

## Safety and deployment scope

The API currently has no authentication and permits cross-origin requests. Run it on a trusted local machine for research and do not expose it directly to the public internet. Remove patient identifiers before uploading data, and follow your organization’s data-protection requirements.
