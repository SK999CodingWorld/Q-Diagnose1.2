"""
Q-Diagnose FastAPI Backend
===========================
REST API for the Q-Diagnose hybrid quantum-classical healthcare platform.

Endpoints:
- POST /datasets/upload : Upload and validate healthcare CSV dataset
- GET /datasets         : List registered datasets
- GET /datasets/{id}    : Get dataset summary and class distribution
- POST /experiments/run : Run an experiment with any of 7 classical/quantum models
- GET /experiments      : List experiments (filterable by dataset_id)
- GET /experiments/{id} : Retrieve full experiment results and curve coordinates
- GET /explain/{id}     : Compute and retrieve dual model explanations (SHAP / Quantum)

Global Rule: Every metric comes from actual executed model runs.
Persistent Disclaimer: Research and decision-support prototype. Not a clinically validated medical diagnostic device.
"""

import os
import uuid
import json
import io
import zipfile
from collections import OrderedDict
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Any, Dict
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from qdiagnose.api.schemas import (
    ExperimentRunRequest,
    SweepRunRequest,
    PredictRequest,
    PatientCreateRequest,
    PatientRecordCreateRequest,
    LabReportAnalyzeRequest,
    SaveLabReportToPatientRequest,
    ReportConfirmRequest,
    RiskPredictRequest,
)

from qdiagnose.domain.pipeline import load_breast_cancer_data, run_pipeline_cv_fold, run_pipeline, fit_full_pipeline
from qdiagnose.domain.evaluate import evaluate_model, compute_metrics_at_threshold, generate_threshold_sweep
from qdiagnose.domain.models import get_model, MODEL_REGISTRY, assign_risk_tier, DEFAULT_RISK_TIERS
from qdiagnose.domain import database as db
from qdiagnose.domain import lab_analyzer
from qdiagnose.domain.dataset_catalog import DATASET_CATALOG, import_public_dataset, list_public_datasets
from qdiagnose.settings import PROJECT_ROOT, settings

UPLOAD_DIR = settings.upload_dir
REPORT_FILE_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".txt", ".csv", ".tsv", ".json", ".rtf"}


async def _read_report_upload(file: UploadFile) -> tuple[str, bytes]:
    """Read a bounded report upload after validating its extension and size."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if not filename or extension not in REPORT_FILE_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported report file type.")
    content = await file.read(settings.max_report_file_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded report is empty.")
    if len(content) > settings.max_report_file_bytes:
        raise HTTPException(status_code=413, detail="The report exceeds the per-file upload limit.")
    if extension in {".docx", ".xlsx"}:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                entries = archive.infolist()
                expanded_size = sum(item.file_size for item in entries)
                if len(entries) > 1000 or expanded_size > 100 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="The document expands beyond the allowed processing limit.")
                if any(
                    item.file_size > 1_000_000
                    and (item.compress_size == 0 or item.file_size / item.compress_size > 200)
                    for item in entries
                ):
                    raise HTTPException(status_code=413, detail="The document contains an unsafe compressed entry.")
        except zipfile.BadZipFile as error:
            raise HTTPException(status_code=400, detail="The uploaded Office document is invalid.") from error
    if extension == ".pdf":
        try:
            import pypdf
            if len(pypdf.PdfReader(io.BytesIO(content)).pages) > 50:
                raise HTTPException(status_code=413, detail="PDF reports are limited to 50 pages.")
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=400, detail="The uploaded PDF could not be validated.") from error
    if extension in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        try:
            from PIL import Image
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                if width < 1 or height < 1 or width * height > 40_000_000:
                    raise HTTPException(status_code=413, detail="Images are limited to 40 megapixels.")
                image.verify()
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=400, detail="The uploaded image is invalid.") from error
    return filename, content


def init_default_dataset():
    """Ensure database is initialized and Breast Cancer dataset is registered."""
    db.init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Register default UCI Breast Cancer dataset if not already present
    existing = db.get_dataset("breast_cancer_wdbc")
    if not existing:
        X, y, feature_names = load_breast_cancer_data()
        df = pd.DataFrame(X, columns=feature_names)
        df["diagnosis"] = y  # 0=malignant, 1=benign
        csv_path = UPLOAD_DIR / "wdbc.csv"
        df.to_csv(csv_path, index=False)

        class_dist = {
            "Malignant (0)": int(sum(y == 0)),
            "Benign (1)": int(sum(y == 1))
        }

        db.save_dataset({
            "id": "breast_cancer_wdbc",
            "name": "UCI Breast Cancer Wisconsin (Diagnostic)",
            "filename": "wdbc.csv",
            "filepath": str(csv_path),
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "target_col": "diagnosis",
            "missing_count": int(df.isna().sum().sum()),
            "duplicate_count": int(df.duplicated().sum()),
            "class_distribution": class_dist,
        })
        print("Default dataset 'breast_cancer_wdbc' registered successfully.")


router = APIRouter()

DISCLAIMER = "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
_DEFAULT_POSITIVE_LABELS = {
    "breast_cancer_wdbc": ("Malignant", "Benign"),
}
_QUANTUM_MODELS = {"qsvc", "vqc", "qnn"}


def _preprocessing_components(model_type: str, requested: int, feature_count: int) -> int:
    """Keep all scaled signal for classical models; cap quantum dimensions."""
    if model_type in _QUANTUM_MODELS:
        return min(requested, feature_count)
    return feature_count


def _load_binary_dataset_arrays(dataset_meta):
    """Load numeric predictors and normalize the positive outcome to label 1.

    The built-in sklearn breast cancer source encodes malignant as 0 and
    benign as 1; all other imported/uploaded cohorts use positive as 1.
    Keeping this convention consistent is essential for risk interpretation.
    """
    filepath = dataset_meta["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Dataset file is missing.")
    frame = pd.read_csv(filepath)
    target_col = dataset_meta["target_col"]
    if target_col not in frame.columns:
        raise HTTPException(status_code=422, detail="Dataset target column is missing.")
    features = frame.drop(columns=[target_col]).select_dtypes(include=[np.number])
    if features.empty:
        raise HTTPException(status_code=400, detail="Dataset contains no numeric feature columns.")
    target = pd.to_numeric(frame[target_col], errors="coerce")
    valid = target.notna()
    features = features.loc[valid]
    target = target.loc[valid]
    if not target.isin([0, 1]).all():
        raise HTTPException(status_code=422, detail="The dataset must contain binary labels coded 0 and 1.")
    target = target.astype(int)
    if dataset_meta["id"] == "breast_cancer_wdbc":
        target = 1 - target
    labels = set(target.unique().tolist())
    if labels != {0, 1}:
        raise HTTPException(status_code=422, detail="The dataset must contain binary labels coded 0 and 1.")
    return features.columns.tolist(), features.to_numpy(dtype=float), target.to_numpy(dtype=int)


from qdiagnose.domain.explain import explain_classical_model, explain_quantum_model


# --- Dataset Endpoints ---

@router.get("/api")
@router.get("/health")
@router.get("/api/health")
@router.get("/models")
@router.get("/api/models")
def api_status():
    research_models = list(MODEL_REGISTRY.keys())
    return {
        "platform": "Q-Diagnose",
        "tagline": "Patient Lab Report Review and ML Research Platform",
        "status": "operational",
        "disclaimer": DISCLAIMER,
        # Backward-compatible key: these models operate on research cohorts,
        # not on the uploaded patient lab report workflow.
        "supported_models": research_models,
        "model_scope": "research_cohorts_only",
        "capabilities": {
            "research_experiments": {
                "status": "available",
                "models": research_models,
                "description": "Run and compare models on labeled research datasets.",
            },
            "patient_lab_reports": {
                "status": "biomarker_review_only",
                "supported_file_types": ["PDF", "DOCX", "CSV", "TXT", "JSON"],
                "disease_prediction": "unavailable",
                "description": (
                    "Extract recognizable lab values and compare them with general "
                    "reference intervals. No validated disease-specific ML or quantum "
                    "model is connected to this workflow."
                ),
            },
        },
    }


@router.get("/backends")
@router.get("/api/backends")
def get_backends():
    """List available classical compute and quantum simulation backends."""
    return db.sanitize_json_obj({
        "backends": [
            {
                "id": "ideal_simulator",
                "name": "Qiskit Statevector Simulator (Ideal / Noiseless)",
                "type": "quantum_simulator",
                "qubits_supported": 10,
                "noise_model": None,
                "description": "Full statevector linear algebra simulation representing ideal coherent unitary evolution."
            },
            {
                "id": "noisy_simulator",
                "name": "Qiskit Aer Depolarizing Simulator",
                "type": "quantum_noisy_simulator",
                "qubits_supported": 8,
                "noise_model": "DepolarizingChannel(p=0.02) + ReadoutError",
                "description": "Qiskit simulation with configured gate-noise models; this is not execution on physical hardware."
            },
            {
                "id": "classical_cpu",
                "name": "Host Multicore Classical Processing Unit",
                "type": "classical_hardware",
                "threads": os.cpu_count() or 4,
                "description": "OpenMP/BLAS classical hardware acceleration for classical baselines."
            }
        ],
        "classical": [
            {
                "id": "cpu",
                "name": "Host CPU (x86_64)",
                "status": "online",
                "cores": os.cpu_count() or 4,
                "supported_models": ["logistic_regression", "svm", "random_forest", "xgboost"]
            }
        ],
        "quantum": [
            {
                "id": "aer_simulator_statevector",
                "name": "Qiskit Aer Simulator (Statevector)",
                "status": "online",
                "max_qubits": 32,
                "type": "ideal_simulator",
                "supported_models": ["qsvc", "vqc", "qnn"]
            },
            {
                "id": "aer_simulator_noisy",
                "name": "Qiskit Aer Simulator (Configured Gate Noise)",
                "status": "online",
                "max_qubits": 16,
                "type": "noisy_simulator",
                "supported_models": ["qsvc", "vqc", "qnn"]
            },
            {
                "id": "ibm_quantum_runtime",
                "name": "IBM Quantum Platform (Qiskit Runtime QPU)",
                "status": "configured",
                "type": "physical_qpu",
                "note": "Requires IBMQ_API_TOKEN environment variable for cloud dispatch"
            }
        ],
        "disclaimer": DISCLAIMER
    })


@router.get("/comparisons")
@router.get("/api/comparisons")
def get_comparisons():
    """Retrieve verified classical vs quantum benchmark comparison summary and tuning study matrix."""
    tuning_file = PROJECT_ROOT / "results" / "tuning_study_results.json"
    if tuning_file.exists():
        with open(tuning_file, "r") as f:
            study_data = json.load(f)
        comparison_list = study_data.get("comparison_table", [])
        return db.sanitize_json_obj({
            "status": "success",
            "comparison_summary": comparison_list,
            "comparisons": comparison_list,
            "baselines": study_data.get("baselines", {}),
            "disclaimer": DISCLAIMER
        })
    return db.sanitize_json_obj({
        "status": "empty",
        "comparison_summary": [],
        "comparisons": [],
        "message": "Tuning comparison study results loaded from database.",
        "disclaimer": DISCLAIMER
    })


@router.get("/datasets")
@router.get("/api/datasets")
def get_datasets():
    datasets = db.list_datasets()
    for dataset in datasets:
        try:
            schema_sample = pd.read_csv(dataset["filepath"], nrows=100)
            target_col = dataset["target_col"]
            dataset["feature_names"] = schema_sample.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number]).columns.tolist()
        except (OSError, ValueError, pd.errors.ParserError):
            dataset["feature_names"] = []
        metadata = dataset.get("metadata") or {}
        defaults = _DEFAULT_POSITIVE_LABELS.get(dataset["id"], ("Positive class", "Negative class"))
        dataset["positive_class_label"] = metadata.get("positive_class_label") or defaults[0]
        dataset["negative_class_label"] = metadata.get("negative_class_label") or defaults[1]
        if dataset["id"] == "breast_cancer_wdbc":
            old_distribution = dataset.get("class_distribution") or {}
            malignant_count = old_distribution.get("Malignant (0)", 0)
            benign_count = old_distribution.get("Benign (1)", 0)
            dataset["class_distribution"] = {
                "Benign (0)": int(benign_count),
                "Malignant (1)": int(malignant_count),
            }
    return {"datasets": datasets, "disclaimer": DISCLAIMER}


@router.get("/datasets/catalog")
@router.get("/api/datasets/catalog")
def get_public_dataset_catalog():
    """List curated public disease datasets available for reproducible import."""
    installed_ids = {row["id"] for row in db.list_datasets()}
    datasets = list_public_datasets()
    for dataset in datasets:
        dataset["imported"] = f"uci_{dataset['key']}" in installed_ids
    return {"datasets": datasets, "disclaimer": DISCLAIMER}


@router.get("/datasets/{dataset_id}")
@router.get("/api/datasets/{dataset_id}")
def get_dataset_by_id(dataset_id: str):
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return {"dataset": dataset, "disclaimer": DISCLAIMER}


@router.post("/datasets/catalog/{dataset_key}/import")
@router.post("/api/datasets/catalog/{dataset_key}/import")
def import_catalog_dataset(dataset_key: str):
    """Fetch and register one curated UCI dataset; no external data is fetched at app startup."""
    if dataset_key not in DATASET_CATALOG:
        raise HTTPException(status_code=404, detail=f"Public dataset '{dataset_key}' is not in the supported catalog.")
    try:
        dataset, created = import_public_dataset(dataset_key, UPLOAD_DIR)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "status": "imported" if created else "already_imported",
        "dataset": dataset,
        "disclaimer": DISCLAIMER,
    }


@router.post("/datasets/upload")
@router.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None)
):
    """
    Upload and validate a healthcare CSV dataset.
    Checks: CSV format, row/column count, missing values, duplicate rows,
    and presence of a valid binary target column.
    """
    # Security: Sanitize filename to prevent path traversal
    safe_filename = Path(file.filename or "upload.csv").name
    if not safe_filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    file_id = f"ds_{uuid.uuid4().hex[:8]}"
    save_path = UPLOAD_DIR / f"{file_id}_{safe_filename}"

    max_upload_bytes = 50 * 1024 * 1024
    content = await file.read(max_upload_bytes + 1)
    if len(content) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="CSV uploads are limited to 50 MB.")
    try:
        save_path.write_bytes(content)
    except OSError as error:
        raise HTTPException(status_code=500, detail="Unable to save the uploaded dataset.") from error

    try:
        df = pd.read_csv(save_path)
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {str(e)}")

    if len(df) < 20:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail="Dataset must have at least 20 rows.")

    # Target column validation
    target_col = target_column
    if not target_col or target_col not in df.columns:
        # Auto-detect target column: look for common names or last column
        candidates = ["target", "label", "diagnosis", "outcome", "class", "disease"]
        for c in candidates:
            match = next((column for column in df.columns if str(column).strip().casefold() == c), None)
            if match:
                target_col = match
                break
        if not target_col:
            target_col = df.columns[-1]

    df = df.loc[df[target_col].notna()].copy()
    if len(df) < 20:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail="Dataset must contain at least 20 rows with a known target label.")

    unique_vals = list(df[target_col].dropna().unique())
    if len(unique_vals) != 2:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' must be binary (found {len(unique_vals)} unique values: {list(unique_vals)})."
        )

    # Correctness: Preserve 0/1 semantics without accidental inversion
    if set(unique_vals) == {0, 1} or set(unique_vals) == {0.0, 1.0}:
        val_map = {0: 0, 1: 1, 0.0: 0, 1.0: 1}
        neg_val, pos_val = 0, 1
    else:
        pos_indicators = ["1", "m", "malignant", "positive", "pos", "yes", "true", "disease", "cancer"]
        if str(unique_vals[0]).lower().strip() in pos_indicators:
            val_map = {unique_vals[0]: 1, unique_vals[1]: 0}
            pos_val, neg_val = unique_vals[0], unique_vals[1]
        elif str(unique_vals[1]).lower().strip() in pos_indicators:
            val_map = {unique_vals[1]: 1, unique_vals[0]: 0}
            pos_val, neg_val = unique_vals[1], unique_vals[0]
        else:
            val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            neg_val, pos_val = unique_vals[0], unique_vals[1]

    df[target_col] = df[target_col].map(val_map)

    # Encode categorical predictors without using target labels; median imputation is
    # performed inside each training fold by the evaluation pipeline.
    mapped_target = df[target_col].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != target_col]
    feature_frame = df[feature_cols].copy()
    for column in feature_frame.columns:
        series = feature_frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        observed = series.notna().sum()
        numeric_ratio = numeric.notna().sum() / observed if observed else 0.0
        if pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.85:
            feature_frame[column] = numeric
        else:
            feature_frame[column] = series.astype("string").fillna("__missing__").str.strip()
    encoded_features = pd.get_dummies(feature_frame, dummy_na=False, dtype=np.uint8)
    if encoded_features.shape[1] < 2:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail="Dataset must contain at least 2 usable feature columns.")

    df = encoded_features.reset_index(drop=True)
    df[target_col] = mapped_target
    df.to_csv(save_path, index=False)

    class_dist = {
        f"Class 0 ({neg_val})": int(sum(df[target_col] == 0)),
        f"Class 1 ({pos_val})": int(sum(df[target_col] == 1))
    }

    dataset_record = {
        "id": file_id,
        "name": name or safe_filename.replace(".csv", ""),
        "filename": safe_filename,
        "filepath": str(save_path),
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "target_col": target_col,
        "missing_count": int(df.isna().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "class_distribution": class_dist,
        "metadata": {
            "source": "User uploaded CSV",
            "disease_area": "User supplied",
            "target_definition": f"Binary target mapping: 0={neg_val}, 1={pos_val}.",
            "positive_class_label": str(pos_val),
            "negative_class_label": str(neg_val),
            "clinical_use": "Research only; user-provided labels have not been independently validated.",
        },
    }

    db.save_dataset(dataset_record)

    return {
        "status": "success",
        "message": "Dataset uploaded and validated successfully.",
        "dataset": dataset_record,
        "disclaimer": DISCLAIMER,
    }


# --- Experiment Endpoints ---

@router.post("/experiments/run")
@router.post("/api/experiments/run")
def run_experiment(req: ExperimentRunRequest):
    """
    Run an end-to-end experiment using identical preprocessing and evaluation.
    Model types: logistic_regression | svm | random_forest | xgboost | qsvc | vqc | qnn
    """
    model_type = req.model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: '{req.model_type}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    dataset_meta = db.get_dataset(req.dataset_id)
    if not dataset_meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    # Load dataset
    filepath = dataset_meta["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Dataset file at {filepath} is missing.")

    _, X, y = _load_binary_dataset_arrays(dataset_meta)

    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise HTTPException(status_code=422, detail="Experiments require exactly two target classes.")
    if int(counts.min()) < req.cv_folds:
        raise HTTPException(status_code=422, detail=f"Each target class needs at least {req.cv_folds} rows for cross-validation; the smallest class has {int(counts.min())}.")

    if model_type in _QUANTUM_MODELS and req.n_features > X.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Requested n_features ({req.n_features}) exceeds available numeric features ({X.shape[1]})."
        )

    # Clean missing values if any
    exp_id = f"exp_{uuid.uuid4().hex[:10]}"

    # Model builder closure for cross-validation
    def current_model_builder():
        return get_model(
            model_type=model_type,
            n_features=req.n_features,
            circuit_depth=req.circuit_depth,
            backend=req.backend,
            random_state=42
        )

    # Run evaluation harness (exact CV with multiple seeds)
    eval_result = evaluate_model(
        model_builder=current_model_builder,
        model_name=model_type.upper(),
        X=X,
        y=y,
        n_components=_preprocessing_components(model_type, req.n_features, X.shape[1]),
        n_folds=req.cv_folds,
        seeds=req.seeds,
        verbose=False,
    )

    # Save to database
    exp_record = {
        "id": exp_id,
        "dataset_id": req.dataset_id,
        "model_type": model_type,
        "config": {
            "n_features": req.n_features,
            "preprocessing_components": _preprocessing_components(model_type, req.n_features, X.shape[1]),
            "target_label_semantics_version": 2,
            "backend": req.backend,
            "circuit_depth": req.circuit_depth,
            "cv_folds": req.cv_folds,
            "seeds": req.seeds,
        },
        "metrics": eval_result["aggregated_metrics"],
        "confusion_matrix": eval_result["representative_fold"]["confusion_matrix"] if eval_result.get("representative_fold") else {},
        "roc_curve": eval_result["representative_fold"]["roc_curve"] if eval_result.get("representative_fold") else {},
        "test_predictions": {
            "y_true": eval_result["representative_fold"]["y_true"],
            "y_prob": eval_result["representative_fold"]["y_prob"],
            "y_pred": eval_result["representative_fold"]["y_pred"],
        } if eval_result.get("representative_fold") and eval_result["representative_fold"].get("y_prob") is not None else {},
        "runtime_seconds": eval_result["total_runtime_seconds"],
        "status": "completed",
    }
    db.save_experiment(exp_record)

    # Automatically generate explanation for this experiment
    try:
        # Fit a single model on 80/20 train/test split for explanations
        pipe_res = run_pipeline(
            X, y,
            n_components=_preprocessing_components(model_type, req.n_features, X.shape[1]),
            random_state=42,
        )
        fit_model = current_model_builder()
        fit_model.fit(pipe_res.X_train, pipe_res.y_train)

        if model_type in ["qsvc", "vqc", "qnn"]:
            expl_data = explain_quantum_model(
                model=fit_model,
                X_test=pipe_res.X_test,
                n_features=req.n_features,
                circuit_depth=req.circuit_depth,
                model_type=model_type
            )
        else:
            expl_data = explain_classical_model(
                model=fit_model,
                X_train=pipe_res.X_train,
                X_test=pipe_res.X_test
            )

        db.save_explanation(exp_id, expl_data["type"], expl_data)
    except Exception as e:
        print(f"Explanation generation warning: {e}")

    return db.sanitize_json_obj({
        "status": "completed",
        "experiment_id": exp_id,
        "model_type": model_type,
        "metrics": eval_result["aggregated_metrics"],
        "runtime_seconds": eval_result["total_runtime_seconds"],
        "disclaimer": DISCLAIMER,
    })


@router.post("/experiments/sweep")
@router.post("/api/experiments/sweep")
def run_sweep(req: SweepRunRequest):
    """
    Run one model across a list of configs (e.g. circuit_depth = [1, 2, 3, 4])
    and return all results as an array.
    """
    model_type = req.model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: '{req.model_type}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    dataset_meta = db.get_dataset(req.dataset_id)
    if not dataset_meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    filepath = dataset_meta["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Dataset file at {filepath} is missing.")

    _, X, y = _load_binary_dataset_arrays(dataset_meta)

    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise HTTPException(status_code=422, detail="Sweeps require exactly two target classes.")
    if int(counts.min()) < req.cv_folds:
        raise HTTPException(status_code=422, detail=f"Each target class needs at least {req.cv_folds} rows for cross-validation; the smallest class has {int(counts.min())}.")

    sweep_results = []
    for val in req.sweep_values:
        curr_depth = req.circuit_depth
        curr_features = min(req.n_features, X.shape[1])
        curr_backend = req.backend

        if req.sweep_param == "circuit_depth":
            curr_depth = int(val)
        elif req.sweep_param == "n_features":
            curr_features = min(int(val), X.shape[1])
        elif req.sweep_param == "backend":
            curr_backend = str(val)

        def sweep_builder():
            return get_model(
                model_type=model_type,
                n_features=curr_features,
                circuit_depth=curr_depth,
                backend=curr_backend,
                random_state=42
            )

        eval_res = evaluate_model(
            model_builder=sweep_builder,
            model_name=f"{model_type.upper()}_{req.sweep_param}_{val}",
            X=X,
            y=y,
            n_components=_preprocessing_components(model_type, curr_features, X.shape[1]),
            n_folds=req.cv_folds,
            seeds=req.seeds,
            verbose=False,
        )

        sweep_results.append({
            "param": req.sweep_param,
            "param_value": val,
            "metrics": eval_res["aggregated_metrics"],
            "runtime_seconds": eval_res["total_runtime_seconds"],
        })

    return db.sanitize_json_obj({
        "status": "completed",
        "dataset_id": req.dataset_id,
        "model_type": model_type,
        "sweep_param": req.sweep_param,
        "sweep_values": req.sweep_values,
        "results": sweep_results,
        "disclaimer": DISCLAIMER,
    })


@router.get("/experiments")
@router.get("/api/experiments")
def get_experiments(dataset_id: Optional[str] = Query(None)):
    """List all experiments, optionally filtered by dataset."""
    experiments = db.list_experiments(dataset_id=dataset_id)
    for experiment in experiments:
        config = experiment.get("config") or {}
        experiment["requires_rerun_for_label_semantics"] = (
            experiment.get("dataset_id") == "breast_cancer_wdbc"
            and config.get("target_label_semantics_version") != 2
        )
    return db.sanitize_json_obj({
        "count": len(experiments),
        "experiments": experiments,
        "disclaimer": DISCLAIMER,
    })


@router.get("/experiments/robustness-study")
@router.get("/api/experiments/robustness-study")
def get_robustness_study():
    """Retrieve pre-computed or live noise and circuit depth sweep results."""
    study_file = PROJECT_ROOT / "results" / "noise_study_results.json"
    if study_file.exists():
        with open(study_file, "r") as f:
            data = json.load(f)
        return db.sanitize_json_obj({
            "status": "success",
            "study": data,
            "disclaimer": DISCLAIMER,
        })
    raise HTTPException(status_code=404, detail="Robustness study data not found.")


@router.get("/experiments/{experiment_id}/roc-curve")
@router.get("/api/experiments/{experiment_id}/roc-curve")
def get_experiment_roc_curve(experiment_id: str):
    """
    Return dense threshold-swept curve coordinates (0.01 to 0.99)
    from stored held-out test predictions without re-running models.
    """
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    test_preds = exp.get("test_predictions") or {}
    y_true = test_preds.get("y_true")
    y_prob = test_preds.get("y_prob")

    if not y_true or not y_prob:
        # Fall back to regenerating or pre-existing stored roc_curve
        roc = exp.get("roc_curve", {})
        return db.sanitize_json_obj({
            "experiment_id": experiment_id,
            "model_type": exp["model_type"],
            "threshold_sweep": [],
            "standard_roc_curve": roc,
            "disclaimer": DISCLAIMER,
        })

    sweep = generate_threshold_sweep(y_true, y_prob, steps=100)
    return db.sanitize_json_obj({
        "experiment_id": experiment_id,
        "model_type": exp["model_type"],
        "count": len(sweep),
        "threshold_sweep": sweep,
        "standard_roc_curve": exp.get("roc_curve", {}),
        "disclaimer": DISCLAIMER,
    })


@router.get("/experiments/{experiment_id}/metrics-at-threshold")
@router.get("/api/experiments/{experiment_id}/metrics-at-threshold")
def get_metrics_at_threshold(
    experiment_id: str,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Operating classification threshold")
):
    """
    Calculate confusion matrix and full metric set (sensitivity, specificity,
    precision, F1, accuracy) at a specific threshold on held-out test predictions.
    """
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    test_preds = exp.get("test_predictions") or {}
    y_true = test_preds.get("y_true")
    y_prob = test_preds.get("y_prob")

    if not y_true or not y_prob:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment '{experiment_id}' has no stored test probability predictions."
        )

    res = compute_metrics_at_threshold(y_true, y_prob, threshold=threshold)
    return db.sanitize_json_obj({
        "experiment_id": experiment_id,
        "model_type": exp["model_type"],
        "threshold": threshold,
        "confusion_matrix": res["confusion_matrix"],
        "metrics": res["metrics"],
        "disclaimer": DISCLAIMER,
    })


_MODEL_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_MODEL_CACHE_LIMIT = 8

def get_cached_fitted_pipeline_and_model(
    dataset_id: str,
    model_type: str,
    n_features: int,
    circuit_depth: int,
    backend: str,
    X: np.ndarray,
    y: np.ndarray
):
    """Retrieve pre-fitted pipeline and model from memory cache or train once."""
    key = (dataset_id, model_type, n_features, circuit_depth, backend)
    if key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(key)
        return _MODEL_CACHE[key]

    pipe_res = fit_full_pipeline(X, y, n_components=n_features)
    model = get_model(
        model_type=model_type,
        n_features=n_features,
        circuit_depth=circuit_depth,
        backend=backend,
        random_state=42
    )
    model.fit(pipe_res.X_train, pipe_res.y_train)
    _MODEL_CACHE[key] = (pipe_res, model)
    while len(_MODEL_CACHE) > _MODEL_CACHE_LIMIT:
        _MODEL_CACHE.popitem(last=False)
    return pipe_res, model


@router.post("/predict")
@router.post("/api/predict")
def predict_single_sample(req: PredictRequest):
    """
    Single-sample inference endpoint:
    Accepts raw clinical feature values, transforms them via the fitted pipeline
    (StandardScaler + PCA), infers disease probability from the chosen model,
    assigns an auditable risk tier, and returns the decision at the active threshold.
    """
    dataset_meta = db.get_dataset(req.dataset_id)
    if not dataset_meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    filepath = dataset_meta["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Dataset file at {filepath} is missing.")

    feature_names, X, y = _load_binary_dataset_arrays(dataset_meta)

    raw_features = np.asarray(
        [np.nan if value is None else float(value) for value in req.features],
        dtype=float,
    ).reshape(1, -1)
    if raw_features.shape[1] != X.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {X.shape[1]} features in this order: {', '.join(feature_names)}. Got {raw_features.shape[1]}."
        )
    if np.isinf(raw_features).any():
        raise HTTPException(status_code=422, detail="Feature values must be finite numbers or left blank.")
    if np.isnan(raw_features).all():
        raise HTTPException(status_code=422, detail="Enter at least one feature value before running inference.")

    n_features = min(req.n_features or 4, X.shape[1])
    preprocessing_components = _preprocessing_components(req.model_type, n_features, X.shape[1])
    circuit_depth = req.circuit_depth or 1
    backend = req.backend or "ideal_simulator"

    pipe_res, model = get_cached_fitted_pipeline_and_model(
        dataset_id=req.dataset_id,
        model_type=req.model_type,
        n_features=preprocessing_components,
        circuit_depth=circuit_depth,
        backend=backend,
        X=X,
        y=y
    )

    # Transform the single sample using the fitted scaler and PCA
    sample_imputed = pipe_res.imputer.transform(raw_features)
    sample_scaled = pipe_res.scaler.transform(sample_imputed)
    sample_pca = pipe_res.pca.transform(sample_scaled)

    # Score the normalized positive outcome. This is a model score, not a
    # clinically calibrated probability.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample_pca)
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        positive_index = int(np.flatnonzero(classes == 1)[0]) if np.any(classes == 1) else 1
        p1 = float(proba[0, positive_index]) if proba.ndim == 2 else float(proba[0])
    elif hasattr(model, "decision_function"):
        from scipy.special import expit
        df_val = model.decision_function(sample_pca)
        p1 = float(expit(df_val)[0])
    else:
        raise HTTPException(
            status_code=422,
            detail="The selected model does not provide a supported positive-class score.",
        )

    threshold = float(req.threshold if req.threshold is not None else 0.5)
    predicted_positive = p1 >= threshold

    return db.sanitize_json_obj({
        "status": "success",
        "dataset_id": req.dataset_id,
        "model_type": req.model_type,
        "preprocessing_components": preprocessing_components,
        "feature_names": feature_names,
        "positive_class_label": (dataset_meta.get("metadata") or {}).get("positive_class_label") or _DEFAULT_POSITIVE_LABELS.get(req.dataset_id, ("Positive class", "Negative class"))[0],
        "negative_class_label": (dataset_meta.get("metadata") or {}).get("negative_class_label") or _DEFAULT_POSITIVE_LABELS.get(req.dataset_id, ("Positive class", "Negative class"))[1],
        "probability": round(p1, 4),
        "positive_class_score": round(p1, 4),
        "score_kind": "uncalibrated_model_score",
        "predicted_class": int(predicted_positive),
        "threshold": threshold,
        "pca_features": [round(float(v), 4) for v in sample_pca[0]],
        "disclaimer": DISCLAIMER,
    })


@router.get("/experiments/{experiment_id}")
@router.get("/api/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    """Retrieve full experiment record including ROC curve data and confusion matrix."""
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return db.sanitize_json_obj({"experiment": exp, "disclaimer": DISCLAIMER})



@router.get("/explain/{experiment_id}")
@router.get("/api/explain/{experiment_id}")
def get_experiment_explanation(experiment_id: str):
    """
    Retrieve or compute model explanations:
    - Classical: SHAP values and top feature importances
    - Quantum: Circuit diagram, qubit count, depth, and perturbation sensitivities
    """
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    saved_expl = db.get_explanation(experiment_id)
    if saved_expl:
        return {
            "experiment_id": experiment_id,
            "model_type": exp["model_type"],
            "explanation": saved_expl["values"],
            "disclaimer": DISCLAIMER,
        }

    # If explanation was not saved during run, compute it on the fly
    dataset_meta = db.get_dataset(exp["dataset_id"])
    if not dataset_meta or not os.path.exists(dataset_meta["filepath"]):
        raise HTTPException(status_code=404, detail="Dataset file for this experiment is missing.")
    _, X, y = _load_binary_dataset_arrays(dataset_meta)

    config = exp["config"]
    n_features = config.get("n_features", 4)
    preprocessing_components = config.get("preprocessing_components") or _preprocessing_components(
        exp["model_type"], n_features, X.shape[1]
    )
    circuit_depth = config.get("circuit_depth", 1)

    pipe_res = run_pipeline(X, y, n_components=preprocessing_components, random_state=42)
    model = get_model(exp["model_type"], n_features=n_features, circuit_depth=circuit_depth, random_state=42)
    model.fit(pipe_res.X_train, pipe_res.y_train)

    if exp["model_type"] in ["qsvc", "vqc", "qnn"]:
        expl_data = explain_quantum_model(
            model=model,
            X_test=pipe_res.X_test,
            n_features=n_features,
            circuit_depth=circuit_depth,
            model_type=exp["model_type"]
        )
    else:
        expl_data = explain_classical_model(
            model=model,
            X_train=pipe_res.X_train,
            X_test=pipe_res.X_test
        )

    db.save_explanation(experiment_id, expl_data["type"], expl_data)

    return {
        "experiment_id": experiment_id,
        "model_type": exp["model_type"],
        "explanation": expl_data,
        "disclaimer": DISCLAIMER,
    }


# =====================================================
# PART A — PATIENT DETAILS MODULE ENDPOINTS
# =====================================================

@router.post("/patients")
@router.post("/api/patients")
def create_patient(req: PatientCreateRequest):
    """Register a new patient record (de-identified research subject by default)."""
    p_id = req.patient_id or str(uuid.uuid4())
    patient_data = {
        "patient_id": p_id,
        "name": req.name or "De-identified Subject",
        "age": req.age,
        "sex": req.sex or "Unknown",
        "notes": req.notes or "",
    }
    saved_id = db.save_patient(patient_data)
    return db.sanitize_json_obj({
        "status": "success",
        "patient_id": saved_id,
        "message": f"Patient '{saved_id}' created successfully.",
        "disclaimer": DISCLAIMER,
    })


@router.get("/patients")
@router.get("/api/patients")
def list_all_patients():
    """Retrieve all registered patient profiles and visit metadata."""
    pts = db.list_patients()
    return db.sanitize_json_obj({
        "total": len(pts),
        "patients": pts,
        "disclaimer": DISCLAIMER,
    })


@router.get("/patients/{patient_id}")
@router.get("/api/patients/{patient_id}")
def get_patient_profile(patient_id: str):
    """Retrieve profile and research records without legacy disease-score history."""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    patient["predictions"] = []
    patient["prediction_status"] = "unavailable_until_validated"
    return db.sanitize_json_obj({
        "patient": patient,
        "disclaimer": DISCLAIMER,
    })


@router.post("/patients/{patient_id}/records")
@router.post("/api/patients/{patient_id}/records")
def add_patient_record(patient_id: str, req: PatientRecordCreateRequest):
    """Add a research-only feature record; this does not generate a disease prediction."""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    if not req.feature_values:
        raise HTTPException(status_code=400, detail="feature_values cannot be empty.")
    try:
        clean_features = [float(x) for x in req.feature_values]
        if any(np.isnan(clean_features)) or any(np.isinf(clean_features)):
            raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="feature_values must contain valid finite numbers.")

    record_id = f"rec_{uuid.uuid4().hex[:12]}"
    record_data = {
        "record_id": record_id,
        "patient_id": patient_id,
        "recorded_at": req.recorded_at or datetime.utcnow().isoformat(),
        "feature_values": clean_features,
        "source": req.source or "manual_entry",
    }
    db.save_patient_record(record_data)

    return db.sanitize_json_obj({
        "status": "success",
        "patient_id": patient_id,
        "record_id": record_id,
        "recorded_at": record_data["recorded_at"],
        "message": "Research feature record added. No diagnosis or risk prediction was generated.",
        "disclaimer": DISCLAIMER,
    })


@router.post("/patients/{patient_id}/predict")
@router.post("/api/patients/{patient_id}/predict")
def predict_patient_record(patient_id: str):
    """Patient-level scoring stays unavailable until an intended-use model is validated."""
    raise HTTPException(
        status_code=410,
        detail=(
            "Patient-specific disease scoring is disabled. Research cohort models and "
            "benchmark scores are not validated for individual patient records."
        ),
    )


@router.get("/patients/{patient_id}/trend")
@router.get("/api/patients/{patient_id}/trend")
def get_patient_longitudinal_trend(patient_id: str):
    """Report record count while withholding unvalidated legacy risk trajectories."""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    trend_data = {
        "patient_id": patient_id,
        "total_records": len(patient.get("records", [])),
        "trend": [],
        "prediction_status": "unavailable_until_validated",
    }
    return db.sanitize_json_obj({
        "trend_data": trend_data,
        "disclaimer": DISCLAIMER,
    })


# --- Lab Report Extraction and Review Endpoints ---


@router.get("/lab-reports/presets")
@router.get("/api/lab-reports/presets")
def get_lab_report_presets():
    """Return the supported biomarker schema without synthetic patient reports."""
    return db.sanitize_json_obj({
        "status": "success",
        # Keep the field for API compatibility while withholding fabricated
        # patient cases from the public interface.
        "presets": [],
        "message": "No demonstration patient reports are provided in this build.",
        "biomarker_definitions": lab_analyzer.BIOMARKER_DEFINITIONS,
        "disclaimer": DISCLAIMER
    })


@router.post("/lab-reports/analyze")
@router.post("/api/lab-reports/analyze")
def analyze_lab_report_data(req: LabReportAnalyzeRequest):
    """Extract recognizable report values and return reference-range flags."""
    result = lab_analyzer.analyze_report(
        raw_text=req.report_text,
        direct_biomarkers=req.biomarkers,
        report_title=req.report_title
    )
    return db.sanitize_json_obj({
        "status": "success",
        **result,
        "disclaimer": DISCLAIMER
    })


@router.post("/lab-reports/upload")
@router.post("/api/lab-reports/upload")
async def upload_lab_report_file(
    file: UploadFile = File(...),
    report_title: Optional[str] = Form(None),
    direct_biomarkers_json: Optional[str] = Form(None)
):
    """Upload a supported lab report and return extracted values and range flags."""
    safe_name, content_bytes = await _read_report_upload(file)

    biomarkers = None
    if direct_biomarkers_json:
        try:
            biomarkers = json.loads(direct_biomarkers_json)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=422, detail="Biomarker data must be valid JSON.") from error
        if not isinstance(biomarkers, dict):
            raise HTTPException(status_code=422, detail="Biomarker data must be a JSON object.")

    result = lab_analyzer.analyze_report(
        file_bytes=content_bytes,
        filename=safe_name,
        direct_biomarkers=biomarkers,
        report_title=report_title or safe_name
    )
    return db.sanitize_json_obj({
        "status": "success",
        **result,
        "disclaimer": DISCLAIMER
    })


@router.post("/lab-reports/save-to-patient")
@router.post("/api/lab-reports/save-to-patient")
def save_lab_report_to_patient(req: SaveLabReportToPatientRequest):
    """Save extracted lab report values to a local research patient record."""
    if req.primary_disease is not None or req.chance_percentage is not None or req.risk_tier is not None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Disease predictions from uploaded lab reports are disabled. "
                "Save extracted biomarker values without prediction fields."
            ),
        )

    patient = db.get_patient(req.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{req.patient_id}' not found.")

    record_id = f"rec_{uuid.uuid4().hex[:12]}"
    feature_vals = list(req.biomarkers.values())

    record_data = {
        "record_id": record_id,
        "patient_id": req.patient_id,
        "recorded_at": datetime.utcnow().isoformat(),
        "feature_values": feature_vals,
        "source": f"Lab Report: {req.report_title}",
    }
    db.save_patient_record(record_data)

    # The report workflow persists biomarker values only; it creates no disease score.
    prediction_id = None

    return db.sanitize_json_obj({
        "status": "success",
        "message": f"Lab report '{req.report_title}' successfully saved to patient '{req.patient_id}'.",
        "record_id": record_id,
        "prediction_id": prediction_id,
        "disclaimer": DISCLAIMER,
    })


# ========================================================
# SECTION C: HEALTH REPORT REVIEW API
# ========================================================

@router.post("/api/reports/upload")
async def upload_patient_health_reports(
    files: List[UploadFile] = File(...),
    patient_profile_json: Optional[str] = Form(None)
):
    """
    Step 2: Upload one or multiple clinical health reports (PDF, CSV, XLSX, DOCX, TXT, or Scanned Image).
    Extracts text, recognizes test names and synonyms, normalizes units (mg/dL <-> mmol/L),
    flags out-of-range values, and stores a temporary review session.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one report file.")
    if len(files) > settings.max_report_files:
        raise HTTPException(status_code=413, detail=f"Upload no more than {settings.max_report_files} report files at once.")

    patient_profile = {}
    if patient_profile_json:
        try:
            patient_profile = json.loads(patient_profile_json)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=422, detail="Patient profile must be valid JSON.") from error
        if not isinstance(patient_profile, dict):
            raise HTTPException(status_code=422, detail="Patient profile must be a JSON object.")

    combined_text = []
    file_summaries = []
    total_bytes = 0

    for file in files:
        safe_name, content_bytes = await _read_report_upload(file)
        total_bytes += len(content_bytes)
        if total_bytes > settings.max_report_total_bytes:
            raise HTTPException(status_code=413, detail="The combined report upload exceeds the total size limit.")
        extracted_text = lab_analyzer.extract_text_from_file(content_bytes, safe_name)
        combined_text.append(f"--- File: {safe_name} ---\n{extracted_text}")
        file_summaries.append({
            "filename": safe_name,
            "bytes_received": len(content_bytes),
            "text_extracted": bool(extracted_text.strip())
        })

    merged_text = "\n\n".join(combined_text)
    parsed_biomarkers = lab_analyzer.parse_biomarkers_from_text(merged_text)

    # Flag all extracted biomarkers
    flagged_list = [lab_analyzer.flag_biomarker(k, v) for k, v in parsed_biomarkers.items()]

    # Generate session ID and store in database
    report_id = f"rep_{uuid.uuid4().hex[:12]}"
    session_data = {
        "report_id": report_id,
        "filename": ", ".join(f["filename"] for f in file_summaries),
        "patient_profile": patient_profile,
        "extracted_biomarkers": parsed_biomarkers,
        "confirmed_biomarkers": parsed_biomarkers,
        "prediction_results": {},
        "explanation": {},
        "status": "extracted",
        "created_at": datetime.utcnow().isoformat()
    }
    db.save_report_session(session_data)

    return db.sanitize_json_obj({
        "status": "success",
        "report_id": report_id,
        "files_processed": file_summaries,
        "extracted_biomarkers": flagged_list,
        "biomarkers_count": len(parsed_biomarkers),
        "patient_profile": patient_profile,
        "disclaimer": DISCLAIMER
    })


@router.get("/api/reports/{report_id}/extracted")
def get_extracted_report_biomarkers(report_id: str):
    """
    Step 3: Retrieve extracted biomarker table for user review, showing
    test names, values, units, reference intervals, and status flags.
    """
    session = db.get_report_session(report_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Report session '{report_id}' not found.")

    biomarkers = session.get("confirmed_biomarkers") or session.get("extracted_biomarkers") or {}
    flagged_list = [lab_analyzer.flag_biomarker(k, v) for k, v in biomarkers.items()]

    return db.sanitize_json_obj({
        "status": "success",
        "report_id": report_id,
        "filename": session.get("filename"),
        "extracted_biomarkers": flagged_list,
        "raw_biomarkers": biomarkers,
        "patient_profile": session.get("patient_profile", {}),
        "session_status": session.get("status"),
        "created_at": session.get("created_at"),
        "disclaimer": DISCLAIMER
    })


@router.put("/api/reports/{report_id}/confirm")
def confirm_report_biomarkers(report_id: str, req: ReportConfirmRequest):
    """
    Save user-verified or manually adjusted biomarker values and patient profile.
    """
    session = db.get_report_session(report_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Report session '{report_id}' not found.")

    updated_profile = req.patient_profile if req.patient_profile is not None else session.get("patient_profile", {})

    db.update_report_session(report_id, {
        "confirmed_biomarkers": req.biomarkers,
        "patient_profile": updated_profile,
        "status": "confirmed"
    })

    flagged_list = [lab_analyzer.flag_biomarker(k, v) for k, v in req.biomarkers.items()]

    return db.sanitize_json_obj({
        "status": "success",
        "report_id": report_id,
        "message": "Extracted biomarker panel successfully confirmed.",
        "confirmed_count": len(req.biomarkers),
        "flagged_biomarkers": flagged_list,
        "disclaimer": DISCLAIMER
    })


@router.post("/api/risk/predict")
def predict_multi_disease_risk(req: RiskPredictRequest):
    """Reject legacy patient disease scoring instead of returning an empty success."""
    raise HTTPException(
        status_code=410,
        detail=(
            "Disease prediction from uploaded lab reports is unavailable. "
            "Use the report review workflow to inspect extracted values and general reference flags."
        ),
    )


@router.get("/api/risk/{report_id}/explanation")
def get_risk_explanations(report_id: str):
    """Reject the legacy report risk explanation route; no validated model exists."""
    raise HTTPException(
        status_code=410,
        detail="Patient-report disease explanations are unavailable because no validated prediction model is configured.",
    )


from fastapi.responses import FileResponse, RedirectResponse, Response

@router.get("/api/reports/{report_id}/export-pdf")
@router.post("/api/reports/{report_id}/export-pdf")
def export_report_pdf(report_id: str):
    """
    Download an extracted-value and reference-range review PDF.
    """
    session = db.get_report_session(report_id)
    if not session and report_id in ["session", "latest"]:
        session = db.get_latest_report_session()
    if not session:
        raise HTTPException(status_code=404, detail=f"Report session '{report_id}' not found.")

    markers = session.get("confirmed_biomarkers") or session.get("extracted_biomarkers") or {}
    profile = session.get("patient_profile") or {}
    pred_res = session.get("prediction_results") or {}

    if not pred_res or not pred_res.get("diseases"):
        # Compute on the fly if not yet predicted
        diseases = lab_analyzer.evaluate_disease_risks(markers, profile)
        pred_res = {
            "diseases": diseases,
            "report_title": session.get("filename") or "Clinical Laboratory Evaluation"
        }

    analysis_data = {
        "report_title": session.get("filename") or "Clinical Laboratory Evaluation",
        "diseases": pred_res.get("diseases", []),
        "flagged_biomarkers": [lab_analyzer.flag_biomarker(k, v) for k, v in markers.items()]
    }

    pdf_bytes = lab_analyzer.generate_patient_report_pdf(
        patient_profile=profile,
        biomarkers=markers,
        analysis_result=analysis_data
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=qdiagnose_clinical_report_{report_id}.pdf"
        }
    )


@router.get("/dashboard")
def dashboard_redirect():
    """Redirect /dashboard to root frontend dashboard."""
    return RedirectResponse(url="/")



