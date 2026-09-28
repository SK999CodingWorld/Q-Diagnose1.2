from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATABASE_FILENAME = "q_diagnose.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def encode_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), allow_nan=False, sort_keys=True)


def decode_json(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def database_path(path: str | Path | None = None) -> Path:
    if path is None:
        return Path(DATABASE_FILENAME)
    return Path(path)


@contextmanager
def _connection(path: str | Path):
    connection = sqlite3.connect(database_path(path))
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def initialize_database(path: str | Path | None = None) -> Path:
    db_path = database_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                target_column TEXT NOT NULL,
                n_rows INTEGER NOT NULL,
                n_columns INTEGER NOT NULL,
                class_distribution_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                file_path TEXT,
                summary_json TEXT
            );
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                confusion_matrix_json TEXT NOT NULL,
                roc_curve_json TEXT NOT NULL,
                pr_curve_json TEXT NOT NULL,
                holdout_metrics_json TEXT,
                runtime_seconds REAL NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'completed',
                progress REAL NOT NULL DEFAULT 1.0,
                error_message TEXT,
                feature_names_json TEXT,
                test_data_json TEXT,
                calibration_json TEXT,
                sweep_variable TEXT,
                sweep_value REAL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );
            CREATE TABLE IF NOT EXISTS experiment_folds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                fold INTEGER NOT NULL,
                n_train INTEGER NOT NULL,
                n_validation INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                confusion_matrix_json TEXT NOT NULL,
                roc_curve_json TEXT NOT NULL,
                pr_curve_json TEXT NOT NULL,
                runtime_seconds REAL NOT NULL,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            );
            CREATE TABLE IF NOT EXISTS explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                values_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            );
            """
        )
        _ensure_column(connection, "datasets", "file_path", "TEXT")
        _ensure_column(connection, "datasets", "summary_json", "TEXT")
        _ensure_column(connection, "datasets", "content_hash", "TEXT")
        _ensure_column(connection, "experiments", "status", "TEXT NOT NULL DEFAULT 'completed'")
        _ensure_column(connection, "experiments", "progress", "REAL NOT NULL DEFAULT 1.0")
        _ensure_column(connection, "experiments", "error_message", "TEXT")
        _ensure_column(connection, "experiments", "feature_names_json", "TEXT")
        _ensure_column(connection, "experiments", "test_data_json", "TEXT")
        _ensure_column(connection, "experiments", "calibration_json", "TEXT")
        _ensure_column(connection, "experiments", "sweep_variable", "TEXT")
        _ensure_column(connection, "experiments", "sweep_value", "REAL")
        _ensure_column(connection, "explanations", "pipeline_revision", "TEXT")
    return db_path


def register_dataset(
    name: str,
    source: str,
    target_column: str,
    n_rows: int,
    n_columns: int,
    class_distribution: dict[str, int],
    metadata: dict[str, Any] | None = None,
    path: str | Path | None = None,
    file_path: str | Path | None = None,
    summary: dict[str, Any] | None = None,
    content_hash: str | None = None,
) -> int:
    db_path = initialize_database(path)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO datasets (
                name, source, target_column, n_rows, n_columns,
                class_distribution_json, metadata_json, created_at,
                file_path, summary_json, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                source,
                target_column,
                int(n_rows),
                int(n_columns),
                encode_json(class_distribution),
                encode_json(metadata or {}),
                utc_now(),
                str(file_path) if file_path is not None else None,
                encode_json(summary) if summary is not None else None,
                content_hash,
            ),
        )
        return int(cursor.lastrowid)


def create_experiment(
    dataset_id: int,
    model_type: str,
    config: dict[str, Any],
    path: str | Path | None = None,
) -> int:
    db_path = initialize_database(path)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiments (
                dataset_id, model_type, config_json, metrics_json,
                confusion_matrix_json, roc_curve_json, pr_curve_json,
                runtime_seconds, created_at, status, progress
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(dataset_id),
                model_type,
                encode_json(config),
                "{}",
                "{}",
                "{}",
                "{}",
                0.0,
                utc_now(),
                "running",
                0.0,
            ),
        )
        return int(cursor.lastrowid)


def update_experiment_progress(
    experiment_id: int,
    progress: float,
    path: str | Path | None = None,
) -> None:
    db_path = initialize_database(path)
    with _connection(db_path) as connection:
        connection.execute(
            "UPDATE experiments SET progress = ? WHERE id = ?",
            (float(max(0.0, min(1.0, progress))), int(experiment_id)),
        )


def complete_experiment(
    experiment_id: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    confusion_matrix: Any,
    roc_curve: dict[str, Any],
    pr_curve: dict[str, Any],
    runtime_seconds: float,
    fold_results: Iterable[dict[str, Any]] = (),
    holdout_metrics: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    test_data: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    sweep_variable: str | None = None,
    sweep_value: float | None = None,
    path: str | Path | None = None,
) -> None:
    db_path = initialize_database(path)
    with _connection(db_path) as connection:
        connection.execute(
            """
            UPDATE experiments SET
                config_json = ?, metrics_json = ?, confusion_matrix_json = ?,
                roc_curve_json = ?, pr_curve_json = ?, holdout_metrics_json = ?,
                runtime_seconds = ?, status = 'completed', progress = 1.0,
                feature_names_json = ?, test_data_json = ?, calibration_json = ?,
                sweep_variable = ?, sweep_value = ?
            WHERE id = ?
            """,
            (
                encode_json(config),
                encode_json(metrics),
                encode_json(confusion_matrix),
                encode_json(roc_curve),
                encode_json(pr_curve),
                encode_json(holdout_metrics) if holdout_metrics is not None else None,
                float(runtime_seconds),
                encode_json(feature_names) if feature_names is not None else None,
                encode_json(test_data) if test_data is not None else None,
                encode_json(calibration) if calibration is not None else None,
                sweep_variable,
                sweep_value,
                int(experiment_id),
            ),
        )
        for fold_result in fold_results:
            connection.execute(
                """
                INSERT INTO experiment_folds (
                    experiment_id, seed, fold, n_train, n_validation,
                    metrics_json, confusion_matrix_json, roc_curve_json,
                    pr_curve_json, runtime_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(experiment_id),
                    int(fold_result["seed"]),
                    int(fold_result["fold"]),
                    int(fold_result["n_train"]),
                    int(fold_result["n_validation"]),
                    encode_json(fold_result["metrics"]),
                    encode_json(fold_result["confusion_matrix"]),
                    encode_json(fold_result["roc_curve"]),
                    encode_json(fold_result["pr_curve"]),
                    float(fold_result["runtime_seconds"]),
                ),
            )


def fail_experiment(
    experiment_id: int,
    error_message: str,
    path: str | Path | None = None,
) -> None:
    db_path = initialize_database(path)
    with _connection(db_path) as connection:
        connection.execute(
            """
            UPDATE experiments SET status = 'failed', error_message = ?, progress = 1.0
            WHERE id = ?
            """,
            (str(error_message), int(experiment_id)),
        )


def save_experiment(
    dataset_id: int,
    model_type: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    confusion_matrix: Any,
    roc_curve: dict[str, Any],
    pr_curve: dict[str, Any],
    runtime_seconds: float,
    fold_results: Iterable[dict[str, Any]] = (),
    holdout_metrics: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> int:
    db_path = initialize_database(path)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiments (
                dataset_id, model_type, config_json, metrics_json,
                confusion_matrix_json, roc_curve_json, pr_curve_json,
                holdout_metrics_json, runtime_seconds, created_at, status, progress
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', 1.0)
            """,
            (
                int(dataset_id),
                model_type,
                encode_json(config),
                encode_json(metrics),
                encode_json(confusion_matrix),
                encode_json(roc_curve),
                encode_json(pr_curve),
                encode_json(holdout_metrics) if holdout_metrics is not None else None,
                float(runtime_seconds),
                utc_now(),
            ),
        )
        experiment_id = int(cursor.lastrowid)
        for fold_result in fold_results:
            connection.execute(
                """
                INSERT INTO experiment_folds (
                    experiment_id, seed, fold, n_train, n_validation,
                    metrics_json, confusion_matrix_json, roc_curve_json,
                    pr_curve_json, runtime_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    int(fold_result["seed"]),
                    int(fold_result["fold"]),
                    int(fold_result["n_train"]),
                    int(fold_result["n_validation"]),
                    encode_json(fold_result["metrics"]),
                    encode_json(fold_result["confusion_matrix"]),
                    encode_json(fold_result["roc_curve"]),
                    encode_json(fold_result["pr_curve"]),
                    float(fold_result["runtime_seconds"]),
                ),
            )
    return experiment_id


def _decode_row(row: sqlite3.Row | dict[str, Any], json_columns: Iterable[str]) -> dict[str, Any]:
    result = dict(row)
    for key in json_columns:
        if key in result:
            result[key.removesuffix("_json")] = decode_json(result.pop(key))
    return result


def get_dataset(dataset_id: int, path: str | Path | None = None) -> dict[str, Any] | None:
    db_path = database_path(path)
    if not db_path.exists():
        return None
    with _connection(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM datasets WHERE id = ?",
            (int(dataset_id),),
        ).fetchone()
    if row is None:
        return None
    return _decode_row(row, ("class_distribution_json", "metadata_json", "summary_json"))


def list_datasets(path: str | Path | None = None) -> list[dict[str, Any]]:
    db_path = database_path(path)
    initialize_database(db_path)
    with _connection(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM datasets ORDER BY id").fetchall()
    return [_decode_row(row, ("class_distribution_json", "metadata_json", "summary_json")) for row in rows]


def get_experiment(experiment_id: int, path: str | Path | None = None) -> dict[str, Any] | None:
    db_path = database_path(path)
    if not db_path.exists():
        return None
    with _connection(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (int(experiment_id),),
        ).fetchone()
    if row is None:
        return None
    return _decode_row(
        row,
        (
            "config_json",
            "metrics_json",
            "confusion_matrix_json",
            "roc_curve_json",
            "pr_curve_json",
            "holdout_metrics_json",
            "feature_names_json",
            "test_data_json",
            "calibration_json",
        ),
    )


def list_experiments(
    dataset_id: int | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    db_path = initialize_database(path)
    with _connection(db_path) as connection:
        connection.row_factory = sqlite3.Row
        if dataset_id is None:
            rows = connection.execute("SELECT * FROM experiments ORDER BY id").fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM experiments WHERE dataset_id = ? ORDER BY id",
                (int(dataset_id),),
            ).fetchall()
    return [
        _decode_row(
            row,
            (
                "config_json",
                "metrics_json",
                "confusion_matrix_json",
                "roc_curve_json",
                "pr_curve_json",
                "holdout_metrics_json",
                "feature_names_json",
                "test_data_json",
                "calibration_json",
            ),
        )
        for row in rows
    ]


def get_experiment_folds(
    experiment_id: int,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    db_path = database_path(path)
    if not db_path.exists():
        return []
    with _connection(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM experiment_folds WHERE experiment_id = ? ORDER BY seed, fold",
            (int(experiment_id),),
        ).fetchall()
    return [
        _decode_row(
            row,
            ("metrics_json", "confusion_matrix_json", "roc_curve_json", "pr_curve_json"),
        )
        for row in rows
    ]


def save_explanation(
    experiment_id: int,
    method: str,
    values: dict[str, Any],
    path: str | Path | None = None,
    pipeline_revision: str | None = None,
) -> int:
    db_path = initialize_database(path)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO explanations (
                experiment_id, method, values_json, created_at, pipeline_revision
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(experiment_id),
                method,
                encode_json(values),
                utc_now(),
                pipeline_revision,
            ),
        )
    return int(cursor.lastrowid)


def get_latest_explanation(
    experiment_id: int,
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    db_path = database_path(path)
    if not db_path.exists():
        return None
    with _connection(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT * FROM explanations WHERE experiment_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (int(experiment_id),),
        ).fetchone()
    if row is None:
        return None
    return _decode_row(row, ("values_json",))


def update_calibration(
    experiment_id: int,
    calibration: dict[str, Any],
    path: str | Path | None = None,
) -> None:
    db_path = initialize_database(path)
    with _connection(db_path) as connection:
        connection.execute(
            "UPDATE experiments SET calibration_json = ? WHERE id = ?",
            (encode_json(calibration), int(experiment_id)),
        )
