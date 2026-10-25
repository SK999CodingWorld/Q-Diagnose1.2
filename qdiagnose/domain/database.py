"""
Q-Diagnose Database Module
===========================
SQLite database interface for storing datasets, experiment logs,
and model explanations. Ensures full reproducibility and audit trail.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from qdiagnose.settings import settings

DB_PATH = settings.database_path
JSON_REPORT_FIELDS = {
    "patient_profile", "extracted_biomarkers", "confirmed_biomarkers",
    "prediction_results", "explanation",
}


def get_connection() -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.database_path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connection():
    """Yield a transaction-scoped connection and always close it."""
    conn = get_connection()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


import math

def sanitize_json_obj(obj):
    """Recursively convert NaNs, Infs, and numpy types to JSON-compliant primitives."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {str(k): sanitize_json_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json_obj(v) for v in obj]
    elif hasattr(obj, "tolist"):
        return sanitize_json_obj(obj.tolist())
    elif hasattr(obj, "item"):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Datasets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            n_rows INTEGER NOT NULL,
            n_cols INTEGER NOT NULL,
            target_col TEXT NOT NULL,
            missing_count INTEGER NOT NULL,
            duplicate_count INTEGER NOT NULL,
            class_distribution TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """)

    dataset_columns = {row[1] for row in cursor.execute("PRAGMA table_info(datasets)").fetchall()}
    if "metadata_json" not in dataset_columns:
        cursor.execute("ALTER TABLE datasets ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")

    # Experiments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            model_type TEXT NOT NULL,
            config TEXT NOT NULL,
            metrics TEXT NOT NULL,
            confusion_matrix TEXT NOT NULL,
            roc_curve TEXT NOT NULL,
            runtime_seconds REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            created_at TEXT NOT NULL,
            test_predictions TEXT DEFAULT '{}',
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )
    """)

    # Safe migration: ensure test_predictions column exists if table was previously created
    cursor.execute("PRAGMA table_info(experiments)")
    columns = [col[1] for col in cursor.fetchall()]
    if "test_predictions" not in columns:
        cursor.execute("ALTER TABLE experiments ADD COLUMN test_predictions TEXT DEFAULT '{}'")

    # Explanations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS explanations (
            id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            method TEXT NOT NULL,
            values_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
    """)

    # Patients table (Part A - Patient Details Module)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            sex TEXT,
            date_registered TEXT NOT NULL,
            notes TEXT DEFAULT ''
        )
    """)

    # Patient clinical visit records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (
            record_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            feature_values TEXT NOT NULL,
            source TEXT DEFAULT 'manual_entry',
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    # Patient predictions / longitudinal audit trail
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_predictions (
            prediction_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            model_type TEXT NOT NULL,
            probability REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            threshold_used REAL NOT NULL,
            predicted_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (record_id) REFERENCES patient_records(record_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS report_sessions (
            report_id TEXT PRIMARY KEY,
            filename TEXT,
            patient_profile TEXT,
            extracted_biomarkers TEXT,
            confirmed_biomarkers TEXT,
            prediction_results TEXT,
            explanation TEXT,
            status TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_dataset(dataset_info: Dict[str, Any]) -> str:
    """Register a new dataset."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO datasets 
        (id, name, filename, filepath, n_rows, n_cols, target_col,
         missing_count, duplicate_count, class_distribution, metadata_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        dataset_info["id"],
        dataset_info["name"],
        dataset_info["filename"],
        dataset_info["filepath"],
        dataset_info["n_rows"],
        dataset_info["n_cols"],
        dataset_info["target_col"],
        dataset_info["missing_count"],
        dataset_info["duplicate_count"],
        json.dumps(dataset_info["class_distribution"]),
        json.dumps(sanitize_json_obj(dataset_info.get("metadata", {}))),
        dataset_info.get("created_at", datetime.utcnow().isoformat())
    ))
    conn.commit()
    conn.close()
    return dataset_info["id"]


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve dataset metadata by id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["class_distribution"] = json.loads(d["class_distribution"])
        d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
        return d
    return None


def list_datasets() -> List[Dict[str, Any]]:
    """List all registered datasets."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["class_distribution"] = json.loads(d["class_distribution"])
        d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
        result.append(d)
    return result


def save_experiment(exp_data: Dict[str, Any]) -> str:
    """Save experiment run to database with strict JSON sanitization."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO experiments
        (id, dataset_id, model_type, config, metrics, confusion_matrix, 
         roc_curve, runtime_seconds, status, created_at, test_predictions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        exp_data["id"],
        exp_data["dataset_id"],
        exp_data["model_type"],
        json.dumps(sanitize_json_obj(exp_data["config"])),
        json.dumps(sanitize_json_obj(exp_data["metrics"])),
        json.dumps(sanitize_json_obj(exp_data.get("confusion_matrix", {}))),
        json.dumps(sanitize_json_obj(exp_data.get("roc_curve", {}))),
        float(exp_data["runtime_seconds"]),
        exp_data.get("status", "completed"),
        exp_data.get("created_at", datetime.utcnow().isoformat()),
        json.dumps(sanitize_json_obj(exp_data.get("test_predictions", {})))
    ))
    conn.commit()
    conn.close()
    return exp_data["id"]


def get_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    """Get full details of a specific experiment."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["config"] = sanitize_json_obj(json.loads(d["config"]))
        d["metrics"] = sanitize_json_obj(json.loads(d["metrics"]))
        d["confusion_matrix"] = sanitize_json_obj(json.loads(d["confusion_matrix"]))
        d["roc_curve"] = sanitize_json_obj(json.loads(d["roc_curve"]))
        if "test_predictions" in d and d["test_predictions"]:
            try:
                d["test_predictions"] = sanitize_json_obj(json.loads(d["test_predictions"]))
            except Exception:
                d["test_predictions"] = {}
        else:
            d["test_predictions"] = {}
        return d
    return None


def list_experiments(dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List experiments, optionally filtered by dataset_id."""
    conn = get_connection()
    cursor = conn.cursor()
    if dataset_id:
        cursor.execute("SELECT * FROM experiments WHERE dataset_id = ? ORDER BY created_at DESC", (dataset_id,))
    else:
        cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["config"] = sanitize_json_obj(json.loads(d["config"]))
        d["metrics"] = sanitize_json_obj(json.loads(d["metrics"]))
        d["confusion_matrix"] = sanitize_json_obj(json.loads(d["confusion_matrix"]))
        d["roc_curve"] = sanitize_json_obj(json.loads(d["roc_curve"]))
        if "test_predictions" in d and d["test_predictions"]:
            try:
                d["test_predictions"] = sanitize_json_obj(json.loads(d["test_predictions"]))
            except Exception:
                d["test_predictions"] = {}
        else:
            d["test_predictions"] = {}
        result.append(d)
    return result


def save_explanation(exp_id: str, method: str, values: Dict[str, Any]) -> str:
    """Save model explanation with strict JSON sanitization."""
    conn = get_connection()
    cursor = conn.cursor()
    expl_id = f"expl_{exp_id}_{method}"
    cursor.execute("""
        INSERT OR REPLACE INTO explanations
        (id, experiment_id, method, values_json, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        expl_id,
        exp_id,
        method,
        json.dumps(sanitize_json_obj(values)),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()
    return expl_id


def get_explanation(exp_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve explanation for an experiment."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM explanations WHERE experiment_id = ? ORDER BY created_at DESC LIMIT 1", (exp_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["values"] = sanitize_json_obj(json.loads(d["values_json"]))
        return d
    return None


# --- Patient Module Helper Functions (Part A) ---

def save_patient(patient_data: Dict[str, Any]) -> str:
    """Register or update a patient record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO patients 
        (patient_id, name, age, sex, date_registered, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        patient_data["patient_id"],
        patient_data.get("name", "De-identified Subject"),
        patient_data.get("age"),
        patient_data.get("sex", "Unknown"),
        patient_data.get("date_registered", datetime.utcnow().isoformat()),
        patient_data.get("notes", "")
    ))
    conn.commit()
    conn.close()
    return patient_data["patient_id"]


def list_patients() -> List[Dict[str, Any]]:
    """List all registered patients with their visit and prediction counts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.*, 
               COUNT(DISTINCT r.record_id) as record_count,
               COUNT(DISTINCT pr.prediction_id) as prediction_count,
               MAX(pr.predicted_at) as latest_prediction_at
        FROM patients p
        LEFT JOIN patient_records r ON p.patient_id = r.patient_id
        LEFT JOIN patient_predictions pr ON p.patient_id = pr.patient_id
        GROUP BY p.patient_id
        ORDER BY p.date_registered DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """Get patient details along with their full clinical records and prediction history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    patient = dict(row)

    # Get records
    cursor.execute("""
        SELECT * FROM patient_records 
        WHERE patient_id = ? 
        ORDER BY recorded_at ASC
    """, (patient_id,))
    records = []
    for r in cursor.fetchall():
        rd = dict(r)
        rd["feature_values"] = sanitize_json_obj(json.loads(rd["feature_values"]))
        records.append(rd)
    patient["records"] = records

    # Get predictions
    cursor.execute("""
        SELECT * FROM patient_predictions 
        WHERE patient_id = ? 
        ORDER BY predicted_at ASC
    """, (patient_id,))
    predictions = [dict(r) for r in cursor.fetchall()]
    patient["predictions"] = predictions

    conn.close()
    return sanitize_json_obj(patient)


def save_patient_record(record_data: Dict[str, Any]) -> str:
    """Save a clinical visit record with raw biomedical feature values."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO patient_records 
        (record_id, patient_id, recorded_at, feature_values, source)
        VALUES (?, ?, ?, ?, ?)
    """, (
        record_data["record_id"],
        record_data["patient_id"],
        record_data.get("recorded_at", datetime.utcnow().isoformat()),
        json.dumps(sanitize_json_obj(record_data["feature_values"])),
        record_data.get("source", "manual_entry")
    ))
    conn.commit()
    conn.close()
    return record_data["record_id"]


def save_patient_prediction(pred_data: Dict[str, Any]) -> str:
    """Save a prediction result linked to a patient and specific visit record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO patient_predictions 
        (prediction_id, patient_id, record_id, model_type, probability, risk_tier, threshold_used, predicted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pred_data["prediction_id"],
        pred_data["patient_id"],
        pred_data["record_id"],
        pred_data["model_type"],
        float(pred_data["probability"]),
        pred_data["risk_tier"],
        float(pred_data.get("threshold_used", 0.5)),
        pred_data.get("predicted_at", datetime.utcnow().isoformat())
    ))
    conn.commit()
    conn.close()
    return pred_data["prediction_id"]


def get_patient_trend(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Get longitudinal risk trajectory across visits for a patient.
    Returns sorted chronological points with visit timestamps, risk scores, and tier transitions.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    patient_row = cursor.fetchone()
    if not patient_row:
        conn.close()
        return None

    patient = dict(patient_row)

    cursor.execute("""
        SELECT pr.*, r.recorded_at, r.source
        FROM patient_predictions pr
        JOIN patient_records r ON pr.record_id = r.record_id
        WHERE pr.patient_id = ?
        ORDER BY r.recorded_at ASC, pr.predicted_at ASC
    """, (patient_id,))
    rows = cursor.fetchall()
    conn.close()

    trend_points = []
    for r in rows:
        d = dict(r)
        trend_points.append({
            "prediction_id": d["prediction_id"],
            "record_id": d["record_id"],
            "recorded_at": d["recorded_at"],
            "predicted_at": d["predicted_at"],
            "model_type": d["model_type"],
            "probability": d["probability"],
            "risk_tier": d["risk_tier"],
            "threshold_used": d["threshold_used"],
        })

    return sanitize_json_obj({
        "patient_id": patient_id,
        "name": patient["name"],
        "age": patient["age"],
        "sex": patient["sex"],
        "total_records": len(trend_points),
        "trend": trend_points
    })


def save_report_session(session_data: Dict[str, Any]) -> str:
    """Persist a report session and serialize structured fields consistently."""
    report_id = session_data.get("report_id")
    if not report_id:
        raise ValueError("report_id is required")
    created_at = session_data.get("created_at") or datetime.utcnow().isoformat()
    encoded = {
        field: json.dumps(sanitize_json_obj(session_data.get(field)))
        if session_data.get(field) is not None else None
        for field in JSON_REPORT_FIELDS
    }
    with connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO report_sessions
            (report_id, filename, patient_profile, extracted_biomarkers,
             confirmed_biomarkers, prediction_results, explanation, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(report_id), session_data.get("filename"), encoded["patient_profile"],
                encoded["extracted_biomarkers"], encoded["confirmed_biomarkers"],
                encoded["prediction_results"], encoded["explanation"],
                session_data.get("status", "uploaded"), created_at,
            ),
        )
    return str(report_id)


def _decode_report_session(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    report = dict(row)
    for field in JSON_REPORT_FIELDS:
        raw = report.get(field)
        if raw:
            try:
                report[field] = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                report[field] = None
    return report


def get_report_session(report_id: str) -> Optional[Dict[str, Any]]:
    """Read one report session and decode its structured fields."""
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM report_sessions WHERE report_id = ?", (report_id,)
        ).fetchone()
    return _decode_report_session(row)


def update_report_session(report_id: str, updates: Dict[str, Any]) -> bool:
    """Update a whitelist of report fields; ignore unknown keys."""
    allowed = JSON_REPORT_FIELDS | {"filename", "status", "created_at"}
    assignments: list[str] = []
    values: list[Any] = []
    for field, value in updates.items():
        if field not in allowed:
            continue
        if field in JSON_REPORT_FIELDS:
            value = json.dumps(sanitize_json_obj(value)) if value is not None else None
        assignments.append(f"{field} = ?")
        values.append(value)
    if not assignments:
        return False
    values.append(report_id)
    with connection() as conn:
        cursor = conn.execute(
            f"UPDATE report_sessions SET {', '.join(assignments)} WHERE report_id = ?",
            values,
        )
        return cursor.rowcount > 0


def get_latest_report_session() -> Optional[Dict[str, Any]]:
    """Read the most recently created report session."""
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM report_sessions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return _decode_report_session(row)
