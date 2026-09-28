from __future__ import annotations

import hashlib
import io
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from storage import get_dataset, register_dataset


UPLOAD_DIRECTORY = Path(os.environ.get("QDIAGNOSE_UPLOAD_DIR", "data/uploads"))
TARGET_CANDIDATES = ("target", "label", "diagnosis", "class", "outcome")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class DatasetValidationError(ValueError):
    pass


def _target_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame.columns:
            raise DatasetValidationError(f"Target column '{requested}' was not found")
        return requested
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lowered:
            return str(lowered[candidate])
    return str(frame.columns[-1])


def _encode_target(series: pd.Series) -> tuple[np.ndarray, dict[str, str]]:
    if series.isna().any():
        raise DatasetValidationError("The target column contains missing values")
    values = list(pd.unique(series))
    if len(values) != 2:
        raise DatasetValidationError(
            f"The target column must be binary; found {len(values)} unique values"
        )
    if all(isinstance(value, (bool, np.bool_)) for value in values):
        mapping = {str(bool(values[0])): 0, str(bool(values[1])): 1}
        return series.astype(int).to_numpy(), mapping
    if all(isinstance(value, (int, np.integer)) and int(value) in (0, 1) for value in values):
        return series.astype(int).to_numpy(), {"0": "0", "1": "1"}
    try:
        numeric_values = [float(value) for value in values]
    except (TypeError, ValueError):
        numeric_values = []
    if sorted(numeric_values) == [0.0, 1.0]:
        return series.astype(int).to_numpy(), {"0": "0", "1": "1"}
    ordered = sorted(values, key=lambda value: str(value))
    mapping = {str(ordered[0]): 0, str(ordered[1]): 1}
    return series.map(mapping).to_numpy(dtype=np.int64), mapping


def validate_dataframe(
    frame: pd.DataFrame,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, str, dict[str, str], dict[str, Any]]:
    if frame.empty:
        raise DatasetValidationError("The CSV contains no rows")
    if frame.columns.duplicated().any():
        raise DatasetValidationError("CSV column names must be unique")
    selected_target = _target_column(frame, target_column)
    X = frame.drop(columns=[selected_target]).copy()
    y, target_mapping = _encode_target(frame[selected_target])
    missing_by_column = {
        str(column): int(frame[column].isna().sum()) for column in frame.columns
    }
    duplicate_rows = int(frame.duplicated().sum())
    class_distribution = {
        str(int(label)): int(np.sum(y == label)) for label in np.unique(y)
    }
    feature_columns = [str(column) for column in X.columns]
    categorical_columns = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(X[column])
    ]
    summary = {
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "feature_count": int(X.shape[1]),
        "missing_values": int(frame.isna().sum().sum()),
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "class_distribution": class_distribution,
        "target_mapping": target_mapping,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
    }
    return X, y, selected_target, target_mapping, summary


def parse_csv(content: bytes, target_column: str | None = None) -> tuple[pd.DataFrame, np.ndarray, str, dict[str, str], dict[str, Any]]:
    if not isinstance(content, (bytes, bytearray)) or len(content) == 0:
        raise DatasetValidationError("The uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DatasetValidationError(
            f"The uploaded file is {len(content)} bytes; the limit is {MAX_UPLOAD_BYTES} bytes"
        )
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as error:
        detail = " ".join(str(error).split())[:200]
        raise DatasetValidationError(f"Could not parse CSV: {detail}") from error
    return validate_dataframe(frame, target_column=target_column)


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def frame_content_hash(frame: pd.DataFrame) -> str:
    return content_hash(frame.to_csv(index=False).encode("utf-8"))


def save_uploaded_csv(
    content: bytes,
    original_filename: str,
    target_column: str | None = None,
    database: str | Path = "q_diagnose.db",
) -> dict[str, Any]:
    X, y, selected_target, target_mapping, summary = parse_csv(content, target_column)
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stored_path = UPLOAD_DIRECTORY / f"{uuid.uuid4().hex}.csv"
    stored_path.write_bytes(content)
    digest = content_hash(content)
    dataset_id = register_dataset(
        name=Path(original_filename).stem or "uploaded_dataset",
        source=str(stored_path),
        target_column=selected_target,
        n_rows=summary["row_count"],
        n_columns=summary["column_count"],
        class_distribution=summary["class_distribution"],
        metadata={
            "target_mapping": target_mapping,
            "original_filename": original_filename,
        },
        path=database,
        file_path=stored_path,
        summary=summary,
        content_hash=digest,
    )
    return {
        "dataset_id": dataset_id,
        "name": Path(original_filename).stem or "uploaded_dataset",
        "file_path": str(stored_path),
        "target_column": selected_target,
        "target_mapping": target_mapping,
        "content_hash": digest,
        "summary": summary,
        "feature_preview": X.head(5).to_dict(orient="records"),
    }


def register_builtin_dataset(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    source: str,
    database: str | Path = "q_diagnose.db",
    metadata: dict[str, Any] | None = None,
) -> int:
    frame = X.copy()
    frame["target"] = np.asarray(y, dtype=np.int64)
    validated_X, validated_y, target, mapping, summary = validate_dataframe(frame, "target")
    return register_dataset(
        name=name,
        source=source,
        target_column=target,
        n_rows=summary["row_count"],
        n_columns=summary["column_count"],
        class_distribution=summary["class_distribution"],
        metadata={
            "target_mapping": mapping,
            **(metadata or {}),
        },
        path=database,
        summary=summary,
        content_hash=frame_content_hash(frame),
    )


def load_registered_dataset(
    dataset_id: int,
    database: str | Path = "q_diagnose.db",
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    metadata = get_dataset(dataset_id, path=database)
    if metadata is None:
        raise KeyError(f"Dataset {dataset_id} was not found")
    file_path = metadata.get("file_path")
    if file_path:
        frame = pd.read_csv(file_path)
    elif metadata.get("source") == "sklearn.datasets.load_breast_cancer":
        features, labels = breast_cancer_dataframe()
        frame = features.copy()
        frame[metadata["target_column"]] = np.asarray(labels, dtype=np.int64)
    elif metadata.get("source") == "sklearn.datasets.load_diabetes_binary":
        features, labels, _ = diabetes_binary_dataframe()
        frame = features.copy()
        frame[metadata["target_column"]] = np.asarray(labels, dtype=np.int64)
    else:
        raise DatasetValidationError(f"Dataset {dataset_id} has no stored CSV file")
    X, y, target, mapping, summary = validate_dataframe(frame, metadata["target_column"])
    if summary != metadata.get("summary"):
        summary["stored_summary_mismatch"] = True
    return X, y, {
        "dataset_id": dataset_id,
        "name": metadata["name"],
        "source": metadata["source"],
        "target_column": target,
        "target_mapping": mapping,
        "summary": summary,
    }


def breast_cancer_dataframe() -> tuple[pd.DataFrame, np.ndarray]:
    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer()
    frame = pd.DataFrame(dataset.data, columns=[str(name) for name in dataset.feature_names])
    return frame, dataset.target.astype(np.int64)


def breast_cancer_target_mapping() -> dict[str, str]:
    return {"0": "malignant", "1": "benign"}


def diabetes_binary_dataframe() -> tuple[pd.DataFrame, np.ndarray, float]:
    from sklearn.datasets import load_diabetes

    dataset = load_diabetes()
    threshold = float(np.median(dataset.target))
    labels = (dataset.target >= threshold).astype(np.int64)
    frame = pd.DataFrame(dataset.data, columns=[str(name) for name in dataset.feature_names])
    return frame, labels, threshold


def ensure_builtin_dataset(
    database: str | Path = "q_diagnose.db",
) -> int:
    from storage import list_datasets

    for dataset in list_datasets(database):
        if dataset["source"] == "sklearn.datasets.load_breast_cancer":
            return int(dataset["id"])
    X, y = breast_cancer_dataframe()
    return register_builtin_dataset(
        "UCI Breast Cancer Wisconsin (Diagnostic)",
        X,
        y,
        "sklearn.datasets.load_breast_cancer",
        database=database,
        metadata={
            "target_mapping": breast_cancer_target_mapping(),
            "class_names": {
                "0": "malignant",
                "1": "benign",
            },
            "target_source": "sklearn.datasets.load_breast_cancer.target",
        },
    )


def ensure_diabetes_dataset(
    database: str | Path = "q_diagnose.db",
) -> int:
    from storage import list_datasets

    source = "sklearn.datasets.load_diabetes_binary"
    for dataset in list_datasets(database):
        if dataset["source"] == source:
            return int(dataset["id"])
    X, y, threshold = diabetes_binary_dataframe()
    return register_builtin_dataset(
        "Diabetes progression median split",
        X,
        y,
        source,
        database=database,
        metadata={
            "label_definition": "target is 1 when the bundled regression target is at or above its median",
            "target_threshold": threshold,
            "exploratory_binary_task": True,
        },
    )
