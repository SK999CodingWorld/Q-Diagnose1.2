from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from datasets import (
    breast_cancer_dataframe,
    ensure_diabetes_dataset,
    ensure_builtin_dataset,
    load_registered_dataset,
    parse_csv,
    register_builtin_dataset,
)
from experiment_runner import (
    ExperimentConfigurationError,
    ExperimentSettings,
    run_experiment,
)
from explanations import explain_experiment
from model_registry import (
    MODEL_TYPES,
    QuantumBackendUnavailable,
    build_model,
    real_hardware_status,
)
from phase5 import run_cross_dataset_comparison
from storage import get_experiment, register_dataset


def test_all_model_factories_build() -> None:
    for model_type in MODEL_TYPES:
        try:
            model, metadata = build_model(
                model_type,
                n_features=2,
                seed=7,
                max_iterations=10,
                shots=32,
            )
        except QuantumBackendUnavailable as error:
            assert model_type == "qnn"
            assert "Refusing to report unreliable numbers" in str(error)
            continue
        assert model is not None
        assert metadata["model_type"] == model_type


def test_quantum_feature_cap_is_enforced() -> None:
    settings = ExperimentSettings(dataset_id=1, model_type="qsvc", n_features=9)
    with pytest.raises(ExperimentConfigurationError, match="at most 8 features"):
        settings.validate()
    with pytest.raises(ValueError, match="at most 8 features"):
        build_model("qsvc", n_features=9, seed=1)
    build_model("qsvc", n_features=8, seed=1)


def test_insufficient_class_support_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "small.db"
    frame = pd.DataFrame(
        {
            "a": np.arange(10, dtype=float),
            "b": np.linspace(0, 1, 10),
            "target": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    csv_path = tmp_path / "tiny.csv"
    frame.to_csv(csv_path, index=False)
    _, _, target, mapping, summary = parse_csv(csv_path.read_bytes(), "target")
    dataset_id = register_dataset(
        name="tiny",
        source=str(csv_path),
        target_column=target,
        n_rows=summary["row_count"],
        n_columns=summary["column_count"],
        class_distribution=summary["class_distribution"],
        metadata={"target_mapping": mapping},
        path=database,
        file_path=csv_path,
        summary=summary,
    )
    with pytest.raises(ExperimentConfigurationError, match="at least cv_folds training rows"):
        run_experiment(
            ExperimentSettings(
                dataset_id=dataset_id,
                model_type="logistic_regression",
                n_features=1,
                cv_folds=5,
                seeds=(11, 22, 33),
            ),
            database=database,
        )


def test_noisy_qsvc_kernel_is_not_ideal_statevector() -> None:
    values = np.linspace(0, 1, 6).reshape(-1, 2)
    ideal, _ = build_model("qsvc", n_features=2, seed=4)
    noisy, _ = build_model(
        "qsvc",
        n_features=2,
        seed=4,
        backend="noisy_simulator",
        noise_multiplier=0.05,
    )
    ideal_kernel = ideal.quantum_kernel.evaluate(values)
    noisy_kernel = noisy.quantum_kernel.evaluate(values)
    assert not np.allclose(ideal_kernel, noisy_kernel)
    assert np.allclose(noisy_kernel, noisy_kernel.T)


def test_builtin_datasets_are_loadable(tmp_path: Path) -> None:
    database = tmp_path / "datasets.db"
    breast_cancer_id = ensure_builtin_dataset(database)
    diabetes_id = ensure_diabetes_dataset(database)
    X_cancer, y_cancer, cancer_metadata = load_registered_dataset(breast_cancer_id, database=database)
    X_diabetes, y_diabetes, diabetes_metadata = load_registered_dataset(diabetes_id, database=database)
    assert X_cancer.shape == (569, 30)
    assert set(np.unique(y_cancer)) == {0, 1}
    assert X_diabetes.shape == (442, 10)
    assert set(np.unique(y_diabetes)) == {0, 1}
    assert diabetes_metadata["source"] == "sklearn.datasets.load_diabetes_binary"


def test_runner_persists_metrics_and_calibration(tmp_path: Path) -> None:
    database = tmp_path / "runner.db"
    dataset_id = ensure_builtin_dataset(database)
    result = run_experiment(
        ExperimentSettings(
            dataset_id=dataset_id,
            model_type="logistic_regression",
            n_features=2,
            cv_folds=3,
            seeds=(11, 22, 33),
        ),
        database=database,
    )
    stored = get_experiment(result["id"], path=database)
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["progress"] == 1.0
    assert len(stored["feature_names"]) == 2
    assert stored["calibration"]["brier_score"] >= 0
    assert stored["test_data"]["y_test"]


def test_phase5_comparison_runs_on_both_datasets(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "phase5.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    client = TestClient(main.app)
    response = client.post(
        "/api/phase5/validate",
        json={
            "model_types": ["logistic_regression"],
            "n_features": 2,
            "cv_folds": 3,
            "seeds": [11, 22, 33],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["datasets"]) == 2
    assert len(body["rows"]) == 2
    assert {row["dataset_id"] for row in body["rows"]} == {row["id"] for row in body["datasets"]}
    for row in body["rows"]:
        assert row["model_type"] == "logistic_regression"
        assert get_experiment(row["experiment_id"], path=database)["status"] == "completed"
    assert client.get("/api/comparisons").status_code == 200


def test_explanation_reports_numeric_base_value(tmp_path: Path) -> None:
    database = tmp_path / "explain.db"
    dataset_id = ensure_builtin_dataset(database)
    result = run_experiment(
        ExperimentSettings(
            dataset_id=dataset_id,
            model_type="logistic_regression",
            n_features=2,
            cv_folds=3,
            seeds=(11, 22, 33),
        ),
        database=database,
    )
    explanation = explain_experiment(result["id"], database=database, sample_limit=5)
    assert explanation["method"].startswith("shap")
    assert isinstance(explanation["base_values"], float)
    assert explanation["top_features"]


def test_phase5_deduplicates_identical_datasets(tmp_path: Path) -> None:
    database = tmp_path / "phase5-dupes.db"
    first = ensure_builtin_dataset(database)
    cancer_features, cancer_labels = breast_cancer_dataframe()
    duplicate = register_builtin_dataset(
        "UCI Breast Cancer Wisconsin (Diagnostic)",
        cancer_features,
        cancer_labels,
        "sklearn.datasets.load_breast_cancer",
        database=database,
    )
    second = ensure_diabetes_dataset(database)
    result = run_cross_dataset_comparison(
        ExperimentSettings(
            dataset_id=first,
            model_type="logistic_regression",
            n_features=2,
            cv_folds=3,
            seeds=(11, 22, 33),
        ),
        database=database,
        dataset_ids=[first, duplicate, second],
        model_types=["logistic_regression"],
    )
    assert [row["id"] for row in result["datasets"]] == [first, second]
    assert len(result["rows"]) == 2


def test_phase5_requires_two_distinct_datasets(tmp_path: Path) -> None:
    database = tmp_path / "phase5-single.db"
    only = ensure_builtin_dataset(database)
    with pytest.raises(ValueError, match="at least two distinct datasets"):
        run_cross_dataset_comparison(
            ExperimentSettings(
                dataset_id=only,
                model_type="logistic_regression",
                n_features=2,
                cv_folds=3,
                seeds=(11, 22, 33),
            ),
            database=database,
            dataset_ids=[only],
            model_types=["logistic_regression"],
        )


def test_api_exposes_health_and_hardware_gate(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "api.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    dataset_id = ensure_builtin_dataset(database)
    client = TestClient(main.app)
    assert client.get("/api/health").status_code == 200
    payload = {
        "dataset_id": dataset_id,
        "model_type": "logistic_regression",
        "n_features": 2,
        "cv_folds": 3,
        "seeds": [11, 22, 33],
    }
    response = client.post("/api/experiments", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    hardware_payload = {**payload, "backend": "real_hardware"}
    hardware_response = client.post("/api/experiments", json=hardware_payload)
    assert hardware_response.status_code == 503
    assert "not configured" in hardware_response.json()["detail"]
    cap_response = client.post(
        "/api/experiments",
        json={**payload, "model_type": "qsvc", "n_features": 9},
    )
    assert cap_response.status_code == 400
    assert "at most 8 features" in cap_response.json()["detail"]
    assert real_hardware_status()["configured"] is False
