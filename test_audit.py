from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import datasets as datasets_module
from datasets import ensure_builtin_dataset
from evaluate import (
    METRIC_NAMES,
    aggregate_metric_values,
    build_classical_model,
    compute_metrics,
    mean_confusion_matrix,
)
from experiment_runner import (
    PIPELINE_REVISION,
    ExperimentSettings,
    run_experiment,
    settings_from_experiment,
)
from model_registry import (
    SUPPORTED_BACKENDS,
    normalize_backend,
    normalize_model_type,
    qnn_probability_from_score,
    real_hardware_status,
)
from pipeline import fit_tabular_preprocessor, transform_with_tabular_preprocessor
from storage import get_dataset, list_datasets


def test_pca_is_fit_only_on_training_rows() -> None:
    rng = np.random.default_rng(0)
    full = rng.normal(loc=5.0, scale=3.0, size=(100, 3))
    frame = pd.DataFrame(full, columns=["a", "b", "c"])
    train_only = fit_tabular_preprocessor(frame.iloc[:80], n_components=2)
    on_everything = fit_tabular_preprocessor(frame, n_components=2)
    raw_pca = PCA(n_components=2).fit(full[:80])
    train_scaler_mean = train_only.transformer.named_transformers_["numeric"].named_steps["scaler"].mean_
    all_scaler_mean = on_everything.transformer.named_transformers_["numeric"].named_steps["scaler"].mean_
    print("raw PCA(80 rows).mean_     :", np.round(raw_pca.mean_, 4))
    print("mean of rows 0:80          :", np.round(full[:80].mean(axis=0), 4))
    print("mean of all 100 rows       :", np.round(full.mean(axis=0), 4))
    print("scaler mean_ fit on 0:80   :", np.round(train_scaler_mean, 4))
    print("scaler mean_ fit on 0:100  :", np.round(all_scaler_mean, 4))
    print("fit-on-train == rows 0:80  :", bool(np.allclose(train_scaler_mean, full[:80].mean(axis=0))))
    print("fit-on-train == all 100    :", bool(np.allclose(train_scaler_mean, full.mean(axis=0))))
    assert np.allclose(raw_pca.mean_, full[:80].mean(axis=0))
    assert not np.allclose(raw_pca.mean_, full.mean(axis=0))
    assert np.allclose(train_scaler_mean, full[:80].mean(axis=0))
    assert not np.allclose(train_scaler_mean, full.mean(axis=0))
    assert not np.allclose(train_scaler_mean, all_scaler_mean, atol=1e-3)


def test_every_fold_refits_preprocessing_on_fold_train_rows(
    tmp_path, monkeypatch
) -> None:
    import experiment_runner

    database = tmp_path / "leak.db"
    dataset_id = ensure_builtin_dataset(database)
    seen: list[int] = []
    original = experiment_runner.fit_tabular_preprocessor

    def recorder(frame, *args, **kwargs):
        seen.append(len(frame))
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(experiment_runner, "fit_tabular_preprocessor", recorder)
    run_experiment(
        ExperimentSettings(
            dataset_id=dataset_id,
            model_type="logistic_regression",
            n_features=2,
            cv_folds=3,
            seeds=(11, 22, 33),
            test_size=0.2,
        ),
        database=database,
    )
    X, y, _ = __import__("datasets").load_registered_dataset(dataset_id, database=database)
    training_rows = int(np.floor(len(y) * 0.8))
    fold_rows = int(np.floor(training_rows * (2 / 3)))
    print("rows in dataset            :", len(y))
    print("rows used per fit call     :", seen)
    print("expected holdout-fit rows  :", training_rows)
    print("expected per-fold-train rows:", fold_rows, "..", int(np.ceil(training_rows * 2 / 3)))
    assert len(seen) == 10
    assert max(seen) == training_rows
    assert sorted(seen) == sorted(
        [training_rows] + [fold_rows, fold_rows, fold_rows + 1] * 3
    )
    assert all(count < len(y) for count in seen)


def test_holdout_rows_never_enter_preprocessor_fit() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    train = frame.iloc[:40]
    test = frame.iloc[40:]
    preprocessor = fit_tabular_preprocessor(train, n_components=2)
    transformed_test = transform_with_tabular_preprocessor(preprocessor, test)
    expected = preprocessor.pca.transform(
        preprocessor.transformer.transform(test[list(preprocessor.numeric_columns)])
    )
    print("test transform shape:", transformed_test.shape)
    print("max abs diff vs manual apply:", float(np.max(np.abs(transformed_test - expected))))
    assert transformed_test.shape == (10, 2)
    assert np.allclose(transformed_test, expected)


def test_metrics_match_sklearn_exactly() -> None:
    rng = np.random.default_rng(7)
    truth = rng.integers(0, 2, size=200)
    predictions = np.where(rng.random(200) < 0.7, truth, 1 - truth)
    probabilities = np.clip(rng.random(200) * 0.6 + 0.2 * truth, 0, 1)
    metrics, matrix, roc_data, pr_data = compute_metrics(truth, predictions, probabilities)
    reference = {
        "accuracy": accuracy_score(truth, predictions),
        "precision": precision_score(truth, predictions, pos_label=1, zero_division=0),
        "sensitivity": recall_score(truth, predictions, pos_label=1, zero_division=0),
        "specificity": recall_score(truth, predictions, pos_label=0, zero_division=0),
        "f1": f1_score(truth, predictions, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(truth, probabilities),
        "pr_auc": average_precision_score(truth, probabilities),
    }
    for name in METRIC_NAMES:
        print(f"{name:12s} ours={metrics[name]:.12f} sklearn={reference[name]:.12f}")
        assert metrics[name] == pytest.approx(reference[name], abs=1e-12)
    print("confusion matrix (rows=actual, cols=predicted):", matrix)
    assert matrix == confusion_matrix(truth, predictions, labels=[0, 1]).tolist()
    assert len(roc_data["fpr"]) == len(roc_data["tpr"]) == len(roc_data["thresholds"])
    assert len(pr_data["precision"]) == len(pr_data["recall"])
    assert len(pr_data["thresholds"]) == len(pr_data["recall"]) - 1
    recall = np.asarray(pr_data["recall"])
    precision = np.asarray(pr_data["precision"])
    print("pr recall is non-increasing :", bool(np.all(np.diff(recall) <= 1e-12)))
    print("pr recall spans 1.0 -> 0.0  :", recall[0] == 1.0 and recall[-1] == 0.0)
    print("pr precision stays in [0,1] :", bool(precision.min() >= 0.0 and precision.max() <= 1.0))
    assert np.all(np.diff(recall) <= 1e-12)
    assert recall[0] == 1.0 and recall[-1] == 0.0
    assert precision.min() >= 0.0 and precision.max() <= 1.0


def test_confusion_matrix_orientation_is_documented_and_consistent() -> None:
    truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    predictions = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    metrics, matrix, _, _ = compute_metrics(
        truth, predictions, np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.4, 0.35])
    )
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    print("truth     :", truth.tolist())
    print("predicted :", predictions.tolist())
    print("matrix rows=actual, cols=predicted:", matrix)
    print("tn,fp,fn,tp =", tn, fp, fn, tp)
    print("sensitivity (tp/(tp+fn)) =", metrics["sensitivity"])
    print("specificity (tn/(tn+fp)) =", metrics["specificity"])
    assert (tn, fp, fn, tp) == (3, 1, 2, 2)
    assert metrics["sensitivity"] == pytest.approx(tp / (tp + fn)) == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(tn / (tn + fp)) == pytest.approx(0.75)
    assert tn / (tn + fp) != tp / (tp + fn)


def test_every_fit_including_holdout_uses_quantum_range_scaling(
    tmp_path, monkeypatch
) -> None:
    import experiment_runner

    database = tmp_path / "range.db"
    dataset_id = ensure_builtin_dataset(database)
    calls: list[bool | None] = []
    original = experiment_runner.fit_tabular_preprocessor

    def recorder(frame, *args, **kwargs):
        calls.append(kwargs.get("range_scaling"))
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(experiment_runner, "fit_tabular_preprocessor", recorder)
    run_experiment(
        ExperimentSettings(
            dataset_id=dataset_id,
            model_type="qsvc",
            n_features=2,
            cv_folds=3,
            seeds=(11, 22, 33),
            backend="ideal_simulator",
            shots=128,
        ),
        database=database,
    )
    print("range_scaling flag per fit call (qsvc):", calls)
    print("number of fit calls:", len(calls))
    assert len(calls) == 10
    assert all(flag is True for flag in calls)


def test_classical_models_never_range_scale(tmp_path, monkeypatch) -> None:
    import experiment_runner

    database = tmp_path / "classical.db"
    dataset_id = ensure_builtin_dataset(database)
    calls: list[bool | None] = []
    original = experiment_runner.fit_tabular_preprocessor

    def recorder(frame, *args, **kwargs):
        calls.append(kwargs.get("range_scaling"))
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(experiment_runner, "fit_tabular_preprocessor", recorder)
    run_experiment(
        ExperimentSettings(
            dataset_id=dataset_id,
            model_type="logistic_regression",
            n_features=2,
            cv_folds=3,
            seeds=(11, 22, 33),
        ),
        database=database,
    )
    print("range_scaling flag per fit call (logistic_regression):", calls)
    assert all(flag is False for flag in calls)


def test_repeated_subject_rows_leak_with_row_level_split() -> None:
    rng = np.random.default_rng(3)
    subjects = 40
    rows = []
    labels = []
    for subject in range(subjects):
        subject_signal = 1.0 if subject % 2 == 0 else 0.0
        for _ in range(5):
            rows.append([subject, subject_signal + rng.normal(scale=0.05)])
            labels.append(int(subject_signal))
    frame = pd.DataFrame(rows, columns=["subject_id", "value"])
    frame["target"] = labels
    y = frame["target"].to_numpy()
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    train_subjects = set(frame["subject_id"].iloc[train_idx])
    test_subjects = set(frame["subject_id"].iloc[test_idx])
    shared = train_subjects & test_subjects
    print("rows:", len(frame), "subjects:", subjects)
    print("subjects in both train and test:", len(shared))
    assert len(shared) > 0, "row-level split is expected to share subjects; guard the test"


def test_spsa_optimizer_is_seeded_and_deterministic_per_seed() -> None:
    from qiskit_machine_learning.utils import algorithm_globals

    from model_registry import _optimizer

    def draws(seed: int) -> list[list[int]]:
        _optimizer("spsa", 30, seed)
        return [
            algorithm_globals.random.binomial(1, 0.5, size=4).tolist() for _ in range(2)
        ]

    first, metadata = _optimizer("spsa", 30, 42)
    again = draws(42)
    other = draws(7)
    print("spsa metadata:", metadata)
    print("seed 42 draws  :", again)
    print("seed 7  draws  :", other)
    assert metadata["name"] == "SPSA"
    assert metadata["seed"] == 42
    assert "random_source" in metadata
    assert again == draws(42)
    assert again != other


def test_cobyla_metadata_declares_determinism() -> None:
    from model_registry import _optimizer

    optimizer, metadata = _optimizer("cobyla", 30, 42)
    print("cobyla metadata:", metadata)
    assert metadata["seed"] == 42
    assert "random_source" in metadata
    assert "deterministic" not in metadata
    _, seeded = _optimizer("cobyla", 30, 43)
    assert seeded["seed"] == 43
    assert optimizer is not None


def test_quantum_circuits_use_one_qubit_per_feature() -> None:
    from model_registry import build_model, feature_map_circuit, real_amplitudes

    for n_features in (2, 4, 8):
        feature_map = feature_map_circuit(n_features)
        ansatz = real_amplitudes(n_features, reps=2)
        combined = feature_map.compose(ansatz)
        print(
            f"n_features={n_features} feature_map.qubits={feature_map.num_qubits} "
            f"ansatz.qubits={ansatz.num_qubits} combined.qubits={combined.num_qubits}"
        )
        assert feature_map.num_qubits == n_features
        assert ansatz.num_qubits == n_features
        assert combined.num_qubits == n_features
    _, vqc_metadata = build_model(
        model_type="vqc",
        n_features=4,
        seed=5,
        backend="ideal_simulator",
        circuit_depth=2,
    )
    print("vqc metadata qubits:", vqc_metadata["qubits"])
    assert vqc_metadata["qubits"] == 4
    _, qsvc_metadata = build_model(
        model_type="qsvc", n_features=4, seed=5, backend="noisy_simulator", shots=128
    )
    print("qsvc noisy metadata simulation :", qsvc_metadata["simulation"])
    print("qsvc noisy metadata noise_mult :", qsvc_metadata["noise_multiplier"])
    print("qsvc noisy metadata depth      :", qsvc_metadata["circuit_depth"])
    assert qsvc_metadata["simulation"] == "density_matrix"
    assert qsvc_metadata["noise_multiplier"] == 0.01
    assert qsvc_metadata["circuit_depth"] == feature_map_circuit(4).depth()
    _, ideal_metadata = build_model(
        model_type="qsvc", n_features=4, seed=5, backend="ideal_simulator", shots=128
    )
    assert ideal_metadata["simulation"] == "statevector"
    assert ideal_metadata["noise_multiplier"] == 0.0


def test_sweep_variable_applicability_matrix() -> None:
    from main import _sweep_variable_applies

    cases = [
        ("logistic_regression", "ideal_simulator", "n_features", True),
        ("logistic_regression", "ideal_simulator", "circuit_depth", False),
        ("logistic_regression", "ideal_simulator", "shots", False),
        ("logistic_regression", "ideal_simulator", "noise_multiplier", False),
        ("logistic_regression", "ideal_simulator", "max_iterations", False),
        ("qsvc", "ideal_simulator", "circuit_depth", False),
        ("qsvc", "ideal_simulator", "shots", True),
        ("qsvc", "noisy_simulator", "noise_multiplier", True),
        ("qsvc", "ideal_simulator", "noise_multiplier", False),
        ("vqc", "ideal_simulator", "circuit_depth", True),
        ("vqc", "ideal_simulator", "shots", True),
        ("vqc", "ideal_simulator", "max_iterations", True),
        ("qnn", "ideal_simulator", "shots", False),
        ("qnn", "ideal_simulator", "circuit_depth", True),
        ("qnn", "noisy_simulator", "noise_multiplier", True),
    ]
    for model_type, backend, variable, expected in cases:
        actual = _sweep_variable_applies(model_type, backend, variable)
        print(f"{model_type:20s} {backend:18s} {variable:16s} -> {actual}")
        assert actual is expected


def test_sweep_endpoint_rejects_ineffective_variables(tmp_path, monkeypatch) -> None:
    import main
    from fastapi.testclient import TestClient

    database = tmp_path / "sweeps.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    dataset_id = ensure_builtin_dataset(database)
    client = TestClient(main.app)
    blocked = [
        ("qsvc", "circuit_depth"),
        ("qnn", "shots"),
        ("logistic_regression", "noise_multiplier"),
        ("logistic_regression", "circuit_depth"),
    ]
    for model_type, variable in blocked:
        response = client.post(
            "/api/sweeps",
            json={
                "dataset_id": dataset_id,
                "model_type": model_type,
                "variable": variable,
                "values": [1, 2],
                "cv_folds": 3,
                "seeds": [11, 22, 33],
            },
        )
        print(f"{model_type} + {variable} -> {response.status_code}: {response.json().get('detail')}")
        assert response.status_code == 422
        assert variable in response.json()["detail"]


def test_qsvc_circuit_endpoint_reports_no_ansatz(tmp_path, monkeypatch) -> None:
    import main
    from fastapi.testclient import TestClient

    from storage import create_experiment

    database = tmp_path / "circuit.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    dataset_id = ensure_builtin_dataset(database)
    client = TestClient(main.app)
    qsvc_id = create_experiment(
        dataset_id=dataset_id,
        model_type="qsvc",
        config={
            "preprocessing": {"n_components": 2},
            "circuit_depth": 2,
            "model": {"feature_map": {"reps": 2}},
        },
        path=database,
    )
    response = client.get(f"/api/experiments/{qsvc_id}/circuit")
    body = response.json()
    print("qsvc circuit response keys:", sorted(body))
    print("qsvc ansatz is None       :", body["ansatz"] is None)
    print("qsvc feature map has RY   :", "RY" in body["feature_map"])
    print("qsvc message              :", body["message"])
    assert response.status_code == 200
    assert body["quantum"] is True
    assert body["ansatz"] is None
    assert "P(2*x[0])" in body["feature_map"]
    assert "RY" not in body["feature_map"]
    assert "no trainable ansatz" in body["message"]
    vqc_id = create_experiment(
        dataset_id=dataset_id,
        model_type="vqc",
        config={
            "preprocessing": {"n_components": 2},
            "circuit_depth": 2,
            "model": {"feature_map": {"reps": 2}},
        },
        path=database,
    )
    vqc_body = client.get(f"/api/experiments/{vqc_id}/circuit").json()
    print("vqc ansatz present        :", vqc_body["ansatz"] is not None)
    print("vqc qubits                :", vqc_body["qubits"])
    assert vqc_body["ansatz"] is not None
    assert vqc_body["qubits"] == 2


def test_upload_rejects_empty_binary_and_oversized_files(
    tmp_path, monkeypatch
) -> None:
    import main
    from fastapi.testclient import TestClient

    from datasets import MAX_UPLOAD_BYTES

    database = tmp_path / "upload.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    monkeypatch.setattr(main, "UPLOAD_DIRECTORY", tmp_path / "uploads")
    monkeypatch.setattr(datasets_module, "UPLOAD_DIRECTORY", tmp_path / "uploads")
    client = TestClient(main.app)
    empty = client.post(
        "/api/datasets/upload", files={"file": ("empty.csv", b"", "text/csv")}
    )
    print("empty upload          ->", empty.status_code, empty.json().get("detail"))
    assert empty.status_code == 422
    assert "empty" in empty.json()["detail"].lower()
    binary = client.post(
        "/api/datasets/upload",
        files={"file": ("payload.exe", bytes(range(256)) * 8, "application/octet-stream")},
    )
    print("binary upload         ->", binary.status_code, binary.json().get("detail"))
    assert binary.status_code == 422
    detail = binary.json()["detail"]
    assert len(detail) <= 260
    oversized = b"a,b,target\n" + b"1,2,1\n" * (MAX_UPLOAD_BYTES // 6 + 64)
    print("oversized payload len :", len(oversized), "limit:", MAX_UPLOAD_BYTES)
    big = client.post(
        "/api/datasets/upload", files={"file": ("big.csv", oversized, "text/csv")}
    )
    print("oversized upload      ->", big.status_code, big.json().get("detail"))
    assert big.status_code == 422
    assert str(MAX_UPLOAD_BYTES) in big.json()["detail"]


def test_upload_stored_file_is_uuid_named_and_outside_attack_paths(
    tmp_path, monkeypatch
) -> None:
    import main
    from fastapi.testclient import TestClient

    uploads = tmp_path / "uploads"
    database = tmp_path / "traversal.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    monkeypatch.setattr(main, "UPLOAD_DIRECTORY", uploads)
    monkeypatch.setattr(datasets_module, "UPLOAD_DIRECTORY", uploads)
    client = TestClient(main.app)
    payload = b"a,b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n"
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("../../../../evil name.csv", payload, "text/csv")},
    )
    print("status:", response.status_code)
    assert response.status_code == 200
    body = response.json()
    stored = Path(body["file_path"])
    print("stored path      :", stored)
    print("stored parent    :", stored.parent)
    print("stored file name :", stored.name)
    print("uploads contents :", sorted(p.name for p in uploads.iterdir()))
    print("dataset name     :", body["name"])
    assert stored.parent == uploads
    assert stored.name.endswith(".csv")
    assert len(stored.stem) == 32
    assert stored.is_file()
    assert not (tmp_path / "evil name.csv").exists()
    assert not Path("evil name.csv").exists()
    assert sorted(p.name for p in uploads.iterdir()) == [stored.name]


def test_dataset_names_with_sql_metacharacters_are_stored_literally(
    tmp_path, monkeypatch
) -> None:
    import main
    from fastapi.testclient import TestClient

    database = tmp_path / "sqli.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    monkeypatch.setattr(main, "UPLOAD_DIRECTORY", tmp_path / "uploads")
    monkeypatch.setattr(datasets_module, "UPLOAD_DIRECTORY", tmp_path / "uploads")
    client = TestClient(main.app)
    payload = b"a,b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n"
    marker = "'; DROP TABLE datasets; --"
    response = client.post(
        "/api/datasets/upload",
        files={"file": (f"{marker}.csv", payload, "text/csv")},
    )
    print("status:", response.status_code)
    assert response.status_code == 200
    dataset_id = response.json()["id"]
    stored = get_dataset(dataset_id, path=database)
    print("stored name  :", stored["name"])
    print("datasets kept:", len(list_datasets(path=database)))
    assert stored["name"] == marker
    assert len(list_datasets(path=database)) >= 1
    assert client.get("/api/datasets").status_code == 200


def test_upload_oserror_does_not_leak_filesystem_paths(
    tmp_path, monkeypatch
) -> None:
    import main
    from fastapi.testclient import TestClient

    database = tmp_path / "oserror.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    monkeypatch.setattr(main, "UPLOAD_DIRECTORY", tmp_path / "uploads")
    monkeypatch.setattr(datasets_module, "UPLOAD_DIRECTORY", tmp_path / "uploads")
    client = TestClient(main.app)

    def boom(*args, **kwargs):
        raise OSError(13, "Permission denied", str(tmp_path / "secret" / "file.csv"))

    monkeypatch.setattr(main, "save_uploaded_csv", boom)
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("x.csv", b"a,b,target\n1,2,0\n3,4,1\n", "text/csv")},
    )
    print("status:", response.status_code)
    print("detail:", response.json().get("detail"))
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "secret" not in detail
    assert str(tmp_path) not in detail
    assert "Permission denied" not in detail
    assert "OSError" not in detail


def test_seed_count_is_bounded(tmp_path, monkeypatch) -> None:
    import main
    from fastapi.testclient import TestClient

    database = tmp_path / "seeds.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    client = TestClient(main.app)
    response = client.post(
        "/api/experiments",
        json={
            "dataset_id": 1,
            "model_type": "logistic_regression",
            "n_features": 2,
            "cv_folds": 3,
            "seeds": list(range(50)),
        },
    )
    print("50 seeds ->", response.status_code)
    print("body     ->", response.json())
    assert response.status_code == 422
    assert "seeds" in str(response.json())
    assert "at most 10 items" in str(response.json())


def test_bounds_are_enforced_by_the_api(tmp_path, monkeypatch) -> None:
    import main
    from fastapi.testclient import TestClient

    database = tmp_path / "bounds.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    client = TestClient(main.app)
    cases = [
        {"n_features": 33},
        {"n_features": 0},
        {"cv_folds": 2},
        {"cv_folds": 21},
        {"shots": 0},
        {"max_iterations": 501},
        {"noise_multiplier": 0.3},
        {"test_size": 0},
        {"seeds": [1, 2]},
    ]
    for override in cases:
        payload = {
            "dataset_id": 1,
            "model_type": "logistic_regression",
            "n_features": 2,
            "cv_folds": 3,
            "seeds": [11, 22, 33],
        }
        payload.update(override)
        response = client.post("/api/experiments", json=payload)
        print(f"{override} -> {response.status_code}")
        assert response.status_code == 422


def test_stored_experiment_reproduces_from_its_own_record(tmp_path) -> None:
    from experiment_runner import fit_final_for_experiment
    from storage import get_experiment

    database = tmp_path / "repro.db"
    dataset_id = ensure_builtin_dataset(database)
    recorded = run_experiment(
        ExperimentSettings(
            dataset_id=dataset_id,
            model_type="logistic_regression",
            n_features=2,
            cv_folds=3,
            seeds=(11, 22, 33),
            split_seed=7,
            test_size=0.25,
        ),
        database=database,
    )
    experiment_id = recorded["id"]
    stored = get_experiment(experiment_id, path=database)
    print("stored keys          :", sorted(stored))
    print("pipeline_revision    :", stored["config"].get("pipeline_revision"))
    print("config top-level keys:", sorted(stored["config"]))
    model, preprocessor, X_train, X_test, y_train, y_test = fit_final_for_experiment(
        stored, database=database
    )
    reproduced = model.predict_proba(X_test)[:, 1]
    original = np.asarray(stored["test_data"]["probabilities"], dtype=np.float64)
    print("holdout rows         :", len(original))
    print("max abs difference   :", float(np.max(np.abs(reproduced - original))))
    print("holdout accuracy     : %.6f" % stored["holdout_metrics"]["metrics"]["accuracy"])
    assert stored["config"].get("pipeline_revision")
    assert len(original) == len(y_test) == len(X_test)
    assert np.allclose(reproduced, original, atol=1e-9)
    assert stored["test_data"]["y_test"] == [int(value) for value in y_test]
    assert stored["feature_names"] == list(preprocessor.feature_names)
    preprocessing = stored["config"]["preprocessing"]
    cross_validation = stored["config"]["cross_validation"]
    print("preprocessing        :", json.dumps(preprocessing, sort_keys=True))
    print("cross_validation     :", json.dumps(cross_validation, sort_keys=True))
    print("model random_state   :", stored["config"]["model"]["random_state"])
    assert preprocessing["n_components"] == 2
    assert preprocessing["split_random_state"] == 7
    assert preprocessing["test_size"] == 0.25
    assert preprocessing["fit_scope"] == "training_partition_only"
    assert cross_validation["n_splits"] == 3
    assert cross_validation["seeds"] == [11, 22, 33]
    assert cross_validation["fit_preprocessing_per_fold"] is True
    assert cross_validation["strategy"] == "StratifiedKFold"
    assert stored["config"]["model"]["random_state"] == 7
    print("config[dataset]      :", json.dumps(stored["config"]["dataset"], sort_keys=True))
    assert stored["config"]["dataset"]["positive_class"] == 1
    assert stored["config"]["dataset"]["target_column"] == "target"
    assert stored["config"]["pipeline_revision"] == PIPELINE_REVISION


def test_superseded_and_current_runs_are_distinguishable(tmp_path, monkeypatch) -> None:
    import main
    from fastapi.testclient import TestClient

    from storage import create_experiment

    database = tmp_path / "superseded.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    client = TestClient(main.app)
    stale_id = create_experiment(
        dataset_id=1,
        model_type="qsvc",
        config={
            "pipeline_revision": "2.1.0-quantum-range-scaling",
            "preprocessing": {"n_components": 2},
        },
        path=database,
    )
    legacy_id = create_experiment(
        dataset_id=1,
        model_type="classical_svm",
        config={"preprocessing": {"n_components": 2}},
        path=database,
    )
    dataset_id = ensure_builtin_dataset(database)
    fresh = run_experiment(
        ExperimentSettings(
            dataset_id=dataset_id,
            model_type="logistic_regression",
            n_features=2,
            cv_folds=3,
            seeds=(11, 22, 33),
        ),
        database=database,
    )
    listing = {row["id"]: row for row in client.get("/api/experiments").json()}
    print("stale row superseded :", listing[stale_id]["superseded"])
    print("legacy row superseded:", listing[legacy_id]["superseded"])
    print("fresh row superseded :", listing[fresh["id"]]["superseded"])
    assert listing[stale_id]["superseded"] is True
    assert listing[stale_id]["pipeline_revision"] == "2.1.0-quantum-range-scaling"
    assert listing[legacy_id]["superseded"] is True
    assert listing[legacy_id]["pipeline_revision"] is None
    assert listing[fresh["id"]]["superseded"] is False
    assert listing[fresh["id"]]["pipeline_revision"] == PIPELINE_REVISION
    assert listing[fresh["id"]]["current_pipeline_revision"] == PIPELINE_REVISION
    detail = client.get(f"/api/experiments/{stale_id}").json()
    print("detail superseded    :", detail["superseded"])
    assert detail["superseded"] is True
    fresh_detail = client.get(f"/api/experiments/{fresh['id']}").json()
    assert fresh_detail["superseded"] is False


def test_real_hardware_is_refused_before_any_work_starts(
    tmp_path, monkeypatch
) -> None:
    import main
    from fastapi.testclient import TestClient

    database = tmp_path / "hardware.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database)
    client = TestClient(main.app)
    for payload in (
        {
            "dataset_id": 1,
            "model_type": "qsvc",
            "backend": "real_hardware",
            "cv_folds": 3,
            "seeds": [11, 22, 33],
        },
        {
            "variable": "n_features",
            "values": [2, 3],
            "dataset_id": 1,
            "model_type": "qsvc",
            "backend": "real_hardware",
            "cv_folds": 3,
            "seeds": [11, 22, 33],
        },
    ):
        route = "/api/experiments" if "variable" not in payload else "/api/sweeps"
        response = client.post(route, json=payload)
        print(f"{route} -> {response.status_code}: {response.json()['detail']}")
        assert response.status_code == 503
    assert list_datasets(path=database) == []
    assert client.get("/api/experiments").json() == []
    monkeypatch.setattr(
        main,
        "real_hardware_status",
        lambda: {
            "configured": True,
            "token_present": True,
            "runtime_package_present": True,
            "execution_available": False,
            "message": "Real hardware executor is not available; simulator fallback is disabled",
        },
    )
    response = client.post(
        "/api/experiments",
        json={
            "dataset_id": 1,
            "model_type": "qsvc",
            "backend": "real_hardware",
            "cv_folds": 3,
            "seeds": [11, 22, 33],
        },
    )
    print("token+package but no executor ->", response.status_code)
    print("message:", response.json()["detail"])
    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Real hardware executor is not available; simulator fallback is disabled"
    )
    assert client.get("/api/experiments").json() == []


def test_positive_class_is_recorded_for_metric_interpretation(tmp_path) -> None:
    database = tmp_path / "labels.db"
    dataset_id = ensure_builtin_dataset(database)
    recorded = run_experiment(
        ExperimentSettings(
            dataset_id=dataset_id,
            model_type="logistic_regression",
            n_features=2,
            cv_folds=3,
            seeds=(11, 22, 33),
        ),
        database=database,
    )
    dataset_config = recorded["config"]["dataset"]
    print("positive_class       :", dataset_config["positive_class"])
    print("positive_class_label :", dataset_config["positive_class_label"])
    print("target_mapping       :", json.dumps(dataset_config["target_mapping"], sort_keys=True))
    print("holdout sensitivity  : %.4f" % recorded["holdout_metrics"]["metrics"]["sensitivity"])
    print("holdout specificity  : %.4f" % recorded["holdout_metrics"]["metrics"]["specificity"])
    assert dataset_config["positive_class"] == 1
    assert dataset_config["positive_class_label"] == "benign"
    assert dataset_config["target_mapping"] == {"0": "malignant", "1": "benign"}


def test_dataset_records_capture_provenance_and_are_immutable(tmp_path) -> None:
    import main
    from fastapi.testclient import TestClient

    uploads = tmp_path / "uploads"
    database = tmp_path / "immutable.db"
    monkey = pytest.MonkeyPatch()
    monkey.setattr(main, "DATABASE_PATH", database)
    monkey.setattr(main, "UPLOAD_DIRECTORY", uploads)
    monkey.setattr(datasets_module, "UPLOAD_DIRECTORY", uploads)
    try:
        client = TestClient(main.app)
        payload = b"a,b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n"
        first = client.post(
            "/api/datasets/upload", files={"file": ("v1.csv", payload, "text/csv")}
        ).json()
        mutated = payload + b"13,14,0\n"
        second = client.post(
            "/api/datasets/upload", files={"file": ("v2.csv", mutated, "text/csv")}
        ).json()
        print("dataset 1 rows:", first["n_rows"], "dataset 2 rows:", second["n_rows"])
        print("distinct file paths:", first["file_path"] != second["file_path"])
        print(
            "first file still on disk with original bytes:",
            Path(first["file_path"]).read_bytes() == payload,
        )
        print("first file size:", Path(first["file_path"]).stat().st_size)
        assert first["id"] != second["id"]
        assert first["n_rows"] == 6
        assert second["n_rows"] == 7
        assert first["file_path"] != second["file_path"]
        assert Path(first["file_path"]).read_bytes() == payload
        assert first["metadata"]["original_filename"] == "v1.csv"
        assert first["metadata"]["target_mapping"] == {"0": "0", "1": "1"}
        assert first["registration"]["target_mapping"] == {"0": "0", "1": "1"}
        assert first["summary"]["row_count"] == 6
        assert second["summary"]["row_count"] == 7
        assert len(first["content_hash"]) == 64
        assert first["content_hash"] != second["content_hash"]
        assert client.get(f"/api/datasets/{first['id']}").status_code == 200
        detail = client.get(f"/api/datasets/{first['id']}").json()
        print("hash from dataset endpoint:", detail["content_hash"][:16], "...")
        assert detail["content_hash"] == first["content_hash"]
        same_content = client.post(
            "/api/datasets/upload", files={"file": ("v1-copy.csv", payload, "text/csv")}
        ).json()
        print("identical content -> identical hash:", same_content["content_hash"] == first["content_hash"])
        print("but still a separate record       :", same_content["id"] != first["id"])
        assert same_content["content_hash"] == first["content_hash"]
        assert same_content["id"] != first["id"]
    finally:
        monkey.undo()


def test_calibration_brier_and_bins_match_reference_definitions() -> None:
    from experiment_runner import _calibration

    rng = np.random.default_rng(7)
    truth = rng.integers(0, 2, size=500)
    probabilities = np.clip(truth * 0.6 + rng.normal(0.5, 0.22, size=500), 0.0, 1.0)
    calibration = _calibration(truth, probabilities, bins=10)
    reference_brier = brier_score_loss(truth, probabilities)
    print("brier score        :", calibration["brier_score"])
    print("sklearn reference  :", reference_brier)
    assert abs(calibration["brier_score"] - reference_brier) < 1e-12
    assert 0.0 <= calibration["brier_score"] <= 1.0
    counts = [row["count"] for row in calibration["bins"]]
    print("bin counts         :", counts)
    print("sum of bin counts  :", sum(counts), "vs sample count", len(truth))
    assert sum(counts) == len(truth)
    for row in calibration["bins"]:
        if row["count"] == 0:
            assert row["predicted_probability"] is None
            assert row["observed_frequency"] is None
            continue
        mask = (probabilities >= row["lower"]) & (probabilities <= row["upper"])
        expected_probability = float(np.mean(probabilities[mask]))
        expected_frequency = float(np.mean(truth[mask]))
        assert abs(row["predicted_probability"] - expected_probability) < 1e-12
        assert abs(row["observed_frequency"] - expected_frequency) < 1e-12
    assert calibration["bin_count"] == 10
    filled = [row for row in calibration["bins"] if row["count"]]
    assert all(
        row["lower"] <= row["upper"] for row in filled
    )
    assert abs(sum(row["observed_frequency"] * row["count"] for row in filled) - truth.sum()) < 1e-9
    perfect = _calibration(np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 0.0, 1.0]), bins=10)
    print("perfect brier      :", perfect["brier_score"])
    assert perfect["brier_score"] == pytest.approx(0.0, abs=1e-12)
    worst = _calibration(np.array([0, 1]), np.array([1.0, 0.0]), bins=10)
    print("inverted brier     :", worst["brier_score"])
    assert worst["brier_score"] == pytest.approx(1.0)


def test_aggregation_uses_population_statistics_over_all_seeds_and_folds() -> None:
    rng = np.random.default_rng(3)
    fold_results = []
    for seed in (11, 22, 33):
        for fold in range(3):
            fold_results.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "metrics": {name: float(rng.random()) for name in METRIC_NAMES},
                    "confusion_matrix": [
                        [float(rng.integers(0, 30)), float(rng.integers(0, 30))],
                        [float(rng.integers(0, 30)), float(rng.integers(0, 30))],
                    ],
                }
            )
    aggregate = aggregate_metric_values(fold_results)
    print("fold count:", len(fold_results), "= 3 seeds x 3 folds")
    assert len(fold_results) == 9
    for name in METRIC_NAMES:
        values = [row["metrics"][name] for row in fold_results]
        expected_mean = float(np.mean(values))
        expected_std = float(np.std(values, ddof=0))
        sample_std = float(statistics.stdev(values))
        assert abs(aggregate[name]["mean"] - expected_mean) < 1e-12
        assert abs(aggregate[name]["std"] - expected_std) < 1e-12
        assert aggregate[name]["std"] != pytest.approx(sample_std)
        assert len(aggregate[name]["values"]) == 9
    print("population std (ddof=0) differs from sample std (ddof=1), as documented")
    matrices = np.asarray(
        [row["confusion_matrix"] for row in fold_results], dtype=np.float64
    )
    mean_matrix = np.asarray(mean_confusion_matrix(fold_results), dtype=np.float64)
    print("mean matrix row sums:", np.round(mean_matrix.sum(axis=1), 6))
    assert np.allclose(mean_matrix, matrices.mean(axis=0))
    per_fold_totals = matrices.sum(axis=(1, 2))
    assert np.allclose(mean_matrix.sum(), per_fold_totals.mean())
    assert not np.allclose(mean_matrix.sum(axis=1), matrices[0].sum(axis=1))


def test_noisy_simulator_changes_the_learned_quantum_model() -> None:
    from model_registry import build_model

    X_train = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]] * 6)
    y_train = np.asarray([0, 1] * 12)
    y_test = np.asarray([0, 1, 1, 0] * 3)
    X_test = np.asarray([[0.15, 0.25], [0.65, 0.75], [0.45, 0.55], [0.05, 0.35]] * 3)
    ideal, ideal_metadata = build_model(
        model_type="qsvc",
        n_features=2,
        seed=11,
        backend="ideal_simulator",
        shots=1024,
        noise_multiplier=0.0,
    )
    noisy, noisy_metadata = build_model(
        model_type="qsvc",
        n_features=2,
        seed=11,
        backend="noisy_simulator",
        shots=1024,
        noise_multiplier=1.0,
    )
    print("ideal metadata:", ideal_metadata["backend"], ideal_metadata["simulation"], ideal_metadata["noise_multiplier"])
    print("noisy metadata:", noisy_metadata["backend"], noisy_metadata["simulation"], noisy_metadata["noise_multiplier"])
    assert ideal_metadata["backend"] == "ideal_simulator"
    assert ideal_metadata["simulation"] == "statevector"
    assert ideal_metadata["noise_multiplier"] == 0.0
    assert noisy_metadata["backend"] == "noisy_simulator"
    assert noisy_metadata["simulation"] == "density_matrix"
    assert noisy_metadata["noise_multiplier"] == 1.0
    ideal.fit(X_train, y_train)
    noisy.fit(X_train, y_train)
    ideal_accuracy = float(ideal.score(X_test, y_test))
    noisy_accuracy = float(noisy.score(X_test, y_test))
    print("ideal holdout accuracy:", ideal_accuracy)
    print("noisy holdout accuracy:", noisy_accuracy)
    assert 0.0 <= ideal_accuracy <= 1.0
    assert 0.0 <= noisy_accuracy <= 1.0
    assert abs(ideal_accuracy - noisy_accuracy) > 1e-9


def test_quantum_training_is_reproducible_within_one_process(tmp_path) -> None:
    from experiment_runner import ExperimentConfigurationError

    database = tmp_path / "reproducible.db"
    dataset_id = ensure_builtin_dataset(database)
    settings = dict(
        dataset_id=dataset_id,
        n_features=2,
        cv_folds=3,
        seeds=(11, 22, 33),
        shots=128,
        max_iterations=15,
    )
    np.random.seed(999)
    np.random.random(2000)
    first = run_experiment(ExperimentSettings(model_type="qsvc", **settings), database=database)
    np.random.seed(123)
    np.random.random(2000)
    second = run_experiment(ExperimentSettings(model_type="qsvc", **settings), database=database)
    print("qsvc run1 holdout accuracy:", first["holdout_metrics"]["metrics"]["accuracy"])
    print("qsvc run2 holdout accuracy:", second["holdout_metrics"]["metrics"]["accuracy"])
    assert first["metrics"]["accuracy"]["values"] == second["metrics"]["accuracy"]["values"]
    assert first["holdout_metrics"]["metrics"] == second["holdout_metrics"]["metrics"]
    assert first["test_data"]["probabilities"] == second["test_data"]["probabilities"]
    vqc_first = run_experiment(ExperimentSettings(model_type="vqc", **settings), database=database)
    np.random.seed(77)
    np.random.random(2000)
    vqc_second = run_experiment(ExperimentSettings(model_type="vqc", **settings), database=database)
    print("vqc run1 holdout accuracy:", vqc_first["holdout_metrics"]["metrics"]["accuracy"])
    print("vqc run2 holdout accuracy:", vqc_second["holdout_metrics"]["metrics"]["accuracy"])
    assert vqc_first["metrics"]["accuracy"]["values"] == vqc_second["metrics"]["accuracy"]["values"]
    assert vqc_first["holdout_metrics"]["metrics"] == vqc_second["holdout_metrics"]["metrics"]
    assert vqc_first["test_data"]["probabilities"] == vqc_second["test_data"]["probabilities"]
    with pytest.raises(ExperimentConfigurationError):
        ExperimentSettings(model_type="not_a_model", **settings).validate()


def test_qnn_is_refused_when_its_forward_pass_is_not_reproducible(tmp_path, monkeypatch) -> None:
    import main
    from fastapi.testclient import TestClient

    from model_registry import QuantumBackendUnavailable, build_model

    database = tmp_path / "qnn_guard.db"
    dataset_id = ensure_builtin_dataset(database)
    settings = dict(
        dataset_id=dataset_id,
        n_features=2,
        cv_folds=3,
        seeds=(11, 22, 33),
        shots=128,
        max_iterations=15,
    )
    from experiment_runner import run_experiment

    try:
        result = run_experiment(ExperimentSettings(model_type="qnn", **settings), database=database)
    except QuantumBackendUnavailable as error:
        print("qnn refused:", str(error)[:120])
        message = str(error)
    else:
        print("qnn completed in this environment; status:", result["status"])
        assert result["status"] == "completed"
        message = None
    from storage import list_experiments

    records = list_experiments(path=database)
    print("qnn experiments recorded:", [(row["id"], row["status"]) for row in records])
    if message is not None:
        assert "Refusing to report unreliable numbers" in message
        assert all(row["status"] == "failed" for row in records)
        assert records[0]["error_message"]
        assert records[0]["metrics"] in (None, {})
        monkeypatch.setattr(main, "DATABASE_PATH", database)
        client = TestClient(main.app)
        response = client.post(
            "/api/experiments",
            json={
                "dataset_id": dataset_id,
                "model_type": "qnn",
                "n_features": 2,
                "cv_folds": 3,
                "seeds": [11, 22, 33],
                "shots": 128,
                "max_iterations": 15,
            },
        )
        print("api status:", response.status_code)
        print("api detail:", response.json()["detail"][:120])
        assert response.status_code == 422
        assert "Refusing to report unreliable numbers" in response.json()["detail"]
    else:
        assert records[0]["status"] == "completed"

    from model_registry import _require_deterministic_qnn

    class Unstable:
        num_weights = 2

        def __init__(self) -> None:
            self.calls = 0

        def forward(self, values, weights):
            self.calls += 1
            return np.asarray([[0.5 + 0.1 * self.calls]])

    with pytest.raises(QuantumBackendUnavailable) as unstable:
        _require_deterministic_qnn(Unstable(), 2)
    print("unstable forward ->", str(unstable.value)[:90])
    assert "different values for identical inputs" in str(unstable.value)

    class WrongWidth:
        num_weights = 2

        def forward(self, values, weights):
            return np.zeros((len(values), 3))

    with pytest.raises(QuantumBackendUnavailable) as width:
        _require_deterministic_qnn(WrongWidth(), 2)
    print("wrong width ->", str(width.value)[:90])
    assert "output contract is unreliable" in str(width.value)

    class Stable:
        num_weights = 2

        def forward(self, values, weights):
            return np.zeros((len(values), 1))

    _require_deterministic_qnn(Stable(), 2)
    print("stable single-output QNN accepted")


def test_shots_none_is_a_clear_configuration_error_not_a_crash() -> None:
    from experiment_runner import ExperimentConfigurationError

    settings = ExperimentSettings(
        dataset_id=1, model_type="qsvc", n_features=2, cv_folds=3, seeds=(11, 22, 33)
    )
    object.__setattr__(settings, "shots", None)
    with pytest.raises(ExperimentConfigurationError) as error:
        settings.validate()
    print("shots=None ->", error.value)
    assert "shots must be an integer" in str(error.value)
    object.__setattr__(settings, "shots", True)
    with pytest.raises(ExperimentConfigurationError):
        settings.validate()
    object.__setattr__(settings, "shots", 0)
    with pytest.raises(ExperimentConfigurationError) as error:
        settings.validate()
    print("shots=0    ->", error.value)
    assert "positive" in str(error.value)
    object.__setattr__(settings, "shots", 512)
    settings.validate()
    print("shots=512  -> valid")


@pytest.mark.parametrize("model_type", ["vqc"])
def test_variational_models_train_end_to_end_and_explain(tmp_path, model_type) -> None:
    from explanations import explain_experiment
    from storage import get_experiment

    database = tmp_path / f"{model_type}.db"
    dataset_id = ensure_builtin_dataset(database)
    result = run_experiment(
        ExperimentSettings(
            dataset_id=dataset_id,
            model_type=model_type,
            n_features=2,
            cv_folds=3,
            seeds=(11, 22, 33),
            shots=128,
            max_iterations=15,
        ),
        database=database,
    )
    metrics = result["metrics"]
    holdout = result["holdout_metrics"]["metrics"]
    matrix = np.asarray(result["holdout_metrics"]["confusion_matrix"], dtype=np.float64)
    holdout_rows = result["test_data"]["y_test"]
    probabilities = np.asarray(result["test_data"]["probabilities"], dtype=np.float64)
    predictions = np.asarray(result["test_data"]["predictions"], dtype=np.int64)
    print(f"{model_type} status        :", result["status"])
    print(f"{model_type} holdout acc  :", holdout["accuracy"])
    print(f"{model_type} holdout auc  :", holdout["roc_auc"])
    print(f"{model_type} holdout f1   :", holdout["f1"])
    print(f"{model_type} cv acc mean  :", metrics["accuracy"]["mean"], "+/-", metrics["accuracy"]["std"])
    print(f"{model_type} confusion     :", result["holdout_metrics"]["confusion_matrix"])
    assert result["status"] == "completed"
    for name in ("accuracy", "f1", "precision", "sensitivity", "specificity", "roc_auc", "pr_auc"):
        assert 0.0 <= holdout[name] <= 1.0, name
    assert "recall" not in holdout
    assert holdout["f1"] == pytest.approx(
        2 * holdout["precision"] * holdout["sensitivity"]
        / (holdout["precision"] + holdout["sensitivity"]),
        abs=1e-12,
    )
    print(f"{model_type} confusion sum:", matrix.sum(), "vs holdout rows", len(holdout_rows))
    assert matrix.sum() == len(holdout_rows)
    total = matrix.sum()
    assert holdout["accuracy"] == pytest.approx(matrix.trace() / total, abs=1e-12)
    assert holdout["sensitivity"] == pytest.approx(matrix[1, 1] / matrix[1].sum(), abs=1e-12)
    assert holdout["specificity"] == pytest.approx(matrix[0, 0] / matrix[0].sum(), abs=1e-12)
    assert holdout["sensitivity"] + matrix[1, 0] / matrix[1].sum() == pytest.approx(1.0, abs=1e-12)
    assert len(predictions) == len(holdout_rows) == len(probabilities)
    assert set(np.unique(predictions)) <= {0, 1}
    assert np.array_equal(predictions, (probabilities >= 0.5).astype(np.int64))
    confusion_from_predictions = confusion_matrix(
        np.asarray(holdout_rows), predictions, labels=[0, 1]
    )
    print(f"{model_type} cm recomputed from stored predictions:", confusion_from_predictions.tolist())
    assert np.array_equal(confusion_from_predictions, matrix.astype(int))
    assert matrix[0].sum() == int((np.asarray(holdout_rows) == 0).sum())
    assert matrix[1].sum() == int((np.asarray(holdout_rows) == 1).sum())
    calibration_counts = sum(row["count"] for row in result["calibration"]["bins"])
    print(f"{model_type} calibration bin total:", calibration_counts)
    assert calibration_counts == len(holdout_rows)
    assert 0.0 <= result["calibration"]["brier_score"] <= 1.0
    assert len(result["fold_results"]) == 9
    assert metrics["n_folds"] == 9
    for name in ("accuracy", "roc_auc", "f1"):
        assert len(metrics[name]["values"]) == 9
    assert result["config"]["probability_output"]["source"] == "predict_proba"
    stored = get_experiment(result["id"], path=database)
    assert stored["config"] == result["config"]
    explanation = explain_experiment(result["id"], database=database)
    print(f"{model_type} explanation method:", explanation.get("method"))
    print(f"{model_type} explanation samples:", explanation["sample_count"])
    assert explanation["sample_count"] == 20
    assert explanation["model_type"] == model_type
    assert len(explanation["feature_names"]) == 2
    if model_type == "qnn":
        assert result["config"]["probability_output"]["qnn_mapping"] == "sigmoid_of_network_output"
        assert explanation.get("method") is None
        quantum = explanation["quantum"]
        print(f"{model_type} method  :", quantum["method"])
        print(f"{model_type} epsilon :", quantum["epsilon"])
        print(f"{model_type} base prob:", quantum["base_probability"])
        print(f"{model_type} feature map:", quantum["feature_map"])
        print(f"{model_type} sensitivity:", [(row["feature"], round(row["mean_absolute_probability_change"], 6), row["rank"]) for row in quantum["feature_scores"]])
        assert quantum["method"] == "parameter_perturbation_on_positive_probability"
        assert quantum["epsilon"] > 0.0
        assert len(quantum["feature_scores"]) == 2
        assert quantum["feature_map"]["name"] == "ZZFeatureMap"
        assert quantum["feature_map"]["qubits"] == 2
        assert quantum["feature_map"]["depth"] > 0
        assert 0.0 <= quantum["base_probability"] <= 1.0
        changes = [row["mean_absolute_probability_change"] for row in quantum["feature_scores"]]
        assert all(value >= 0.0 for value in changes)
        assert changes == sorted(changes, reverse=True)
        assert [row["rank"] for row in quantum["feature_scores"]] == [1, 2]
        assert {row["feature"] for row in quantum["feature_scores"]} == set(explanation["feature_names"])
        assert all(0.0 <= row["base_probability"] <= 1.0 for row in quantum["feature_scores"])
    else:
        assert result["config"]["probability_output"]["qnn_mapping"] is None
        method = explanation.get("method")
        assert method == "unavailable" or str(method).startswith("shap")
        assert explanation["top_features"]
        if str(method).startswith("shap"):
            contributions = np.asarray(explanation["sample_contributions"], dtype=np.float64)
            print(f"{model_type} contribution shape:", contributions.shape)
            print(f"{model_type} base value:", explanation["base_values"])
            assert contributions.shape == (20, 2)
            assert np.isfinite(contributions).all()
            assert explanation["base_values"] is None or np.isfinite(explanation["base_values"])


def test_qnn_score_mapping_is_a_bounded_monotone_sigmoid() -> None:
    scores = np.asarray([-800.0, -40.0, -1.0, -0.25, 0.0, 0.25, 1.0, 40.0, 800.0])
    probabilities = qnn_probability_from_score(scores)
    print("scores      :", np.round(scores, 3))
    print("probabilities:", np.round(probabilities, 6))
    assert probabilities.shape == scores.shape
    assert np.all(np.isfinite(probabilities))
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert np.all(np.diff(probabilities) >= 0.0)
    unsaturated = scores[np.abs(scores) <= 40.0]
    unsaturated_probabilities = qnn_probability_from_score(unsaturated)
    print("strictly increasing in the unsaturated range:", np.all(np.diff(unsaturated_probabilities) > 0.0))
    assert np.all(np.diff(unsaturated_probabilities) > 0.0)
    reference = 1.0 / (1.0 + np.exp(-np.clip(scores, -709.0, 709.0)))
    assert np.allclose(probabilities, reference)
    assert probabilities[int(np.where(scores == 0.0)[0][0])] == pytest.approx(0.5)
    overflow = qnn_probability_from_score(np.asarray([1e308, -1e308]))
    print("extreme inputs stay finite:", overflow)
    assert np.isfinite(overflow).all()
    assert overflow[0] == pytest.approx(1.0)
    assert overflow[1] == pytest.approx(0.0)


def test_normalize_backend_accepts_every_stored_shape() -> None:
    assert normalize_backend("ideal_simulator") == "ideal_simulator"
    assert normalize_backend("noisy_simulator") == "noisy_simulator"
    assert normalize_backend("real_hardware") == "real_hardware"
    assert normalize_backend(None) == "ideal_simulator"
    assert normalize_backend("classical") == "ideal_simulator"
    legacy_noiseless = {
        "name": "qiskit_aer",
        "simulator": "statevector",
        "noiseless": True,
        "shots": None,
        "statevector_precision": "double",
    }
    print("legacy noiseless descriptor ->", normalize_backend(legacy_noiseless))
    assert normalize_backend(legacy_noiseless) == "ideal_simulator"
    assert normalize_backend({**legacy_noiseless, "noiseless": False}) == "noisy_simulator"
    for bad in ("statevector", "qiskit_aer", 7, ["ideal_simulator"]):
        with pytest.raises(ValueError):
            normalize_backend(bad)


def test_normalize_model_type_aliases_legacy_classical_svm() -> None:
    from model_registry import _classical_model

    assert normalize_model_type("svm") == "svm"
    assert normalize_model_type("qsvc") == "qsvc"
    assert normalize_model_type("classical_svm") == "svm"
    with pytest.raises(ValueError):
        normalize_model_type("quantum_wizard")
    legacy = build_classical_model(42)
    current, _ = _classical_model("svm", 42)
    print("legacy SVC params    :", legacy.get_params())
    print("current svm params   :", current.get_params())
    shared = set(legacy.get_params()) & set(current.get_params())
    assert all(legacy.get_params()[key] == current.get_params()[key] for key in shared)
    assert legacy.get_params()["kernel"] == "rbf"


def test_legacy_phase1_config_is_replayable() -> None:
    legacy_classical = {
        "dataset_id": 1,
        "model_type": "classical_svm",
        "config": {
            "backend": "classical",
            "circuit_depth": 0,
            "preprocessing": {"n_components": 4, "split_random_state": 42, "test_size": 0.2},
            "cross_validation": {"n_splits": 5, "seeds": [42, 123, 2025]},
        },
    }
    legacy_quantum = {
        "dataset_id": 1,
        "model_type": "qsvc",
        "config": {
            "backend": {
                "name": "qiskit_aer",
                "simulator": "statevector",
                "noiseless": True,
                "shots": None,
                "statevector_precision": "double",
            },
            "circuit_depth": 6,
            "preprocessing": {"n_components": 4, "split_random_state": 42, "test_size": 0.2},
            "cross_validation": {"n_splits": 5, "seeds": [42, 123, 2025]},
        },
    }
    for experiment, expected_model, expected_backend in (
        (legacy_classical, "svm", "ideal_simulator"),
        (legacy_quantum, "qsvc", "ideal_simulator"),
    ):
        settings = settings_from_experiment(experiment)
        print("replayed", experiment["model_type"], "->", settings.model_type, settings.backend)
        assert settings.model_type == expected_model
        assert settings.backend == expected_backend
        assert settings.backend in SUPPORTED_BACKENDS
        settings.validate()


def test_replayed_legacy_backend_would_not_have_been_stringified() -> None:
    legacy_backend = {
        "name": "qiskit_aer",
        "simulator": "statevector",
        "noiseless": True,
        "shots": None,
    }
    before = str(legacy_backend)
    print("pre-fix String(config.backend) was:", before)
    assert before not in SUPPORTED_BACKENDS
    assert normalize_backend(legacy_backend) in SUPPORTED_BACKENDS


def test_created_experiment_response_carries_provenance(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    import main as main_module

    main_module.DATABASE_PATH = str(tmp_path / "create_provenance.db")
    _register_seed_dataset(main_module.DATABASE_PATH)
    client = TestClient(main_module.app)
    response = client.post(
        "/api/experiments",
        json={
            "dataset_id": 1,
            "model_type": "logistic_regression",
            "n_features": 3,
            "backend": "ideal_simulator",
            "cv_folds": 3,
            "seeds": [11, 22, 33],
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    for field in ("pipeline_revision", "current_pipeline_revision", "superseded"):
        assert field in payload, "{} missing from create response".format(field)
    assert payload["pipeline_revision"] == PIPELINE_REVISION
    assert payload["current_pipeline_revision"] == PIPELINE_REVISION
    assert payload["superseded"] is False
    detail = client.get("/api/experiments/{}".format(payload["id"])).json()
    assert detail["superseded"] is False
    assert detail["current_pipeline_revision"] == PIPELINE_REVISION


def test_cached_explanation_is_provenance_stamped(tmp_path: Path) -> None:
    from explanations import _with_provenance, explain_experiment
    from storage import get_latest_explanation, initialize_database, save_explanation

    database = tmp_path / "cache_stamp.db"
    initialize_database(database)
    experiment = {
        "id": 1,
        "dataset_id": 1,
        "model_type": "logistic_regression",
        "config": {
            "backend": "ideal_simulator",
            "preprocessing": {"n_components": 3, "split_random_state": 42, "test_size": 0.2},
            "cross_validation": {"n_splits": 3, "seeds": [11, 22, 33]},
        },
    }
    legacy_payload = {"experiment_id": 1, "model_type": "logistic_regression", "sample_count": 1}
    save_explanation(
        1,
        "feature_attribution",
        legacy_payload,
        path=database,
        pipeline_revision="2.1.0-quantum-range-scaling",
    )
    stored = get_latest_explanation(1, path=database)
    assert stored is not None
    print("stored revision column:", stored.get("pipeline_revision"))
    assert stored.get("pipeline_revision") == "2.1.0-quantum-range-scaling"
    annotated = _with_provenance(stored["values"], stored.get("pipeline_revision"))
    print("annotated:", {k: annotated[k] for k in ("pipeline_revision", "current_pipeline_revision", "superseded")})
    assert annotated["superseded"] is True
    assert annotated["pipeline_revision"] == "2.1.0-quantum-range-scaling"
    assert annotated["current_pipeline_revision"] == PIPELINE_REVISION
    fresh = _with_provenance(legacy_payload, PIPELINE_REVISION)
    assert fresh["superseded"] is False


def test_explanation_table_has_revision_column_and_defaults_to_null(tmp_path: Path) -> None:
    import sqlite3

    from storage import initialize_database, save_explanation

    database = tmp_path / "schema.db"
    initialize_database(database)
    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(explanations)")]
    print("explanations columns:", columns)
    assert "pipeline_revision" in columns
    save_explanation(1, "feature_attribution", {"a": 1}, path=database)
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT pipeline_revision FROM explanations").fetchone()
    print("unstamped row ->", row)
    assert row[0] is None


def _register_seed_dataset(database: str) -> int:
    from datasets import ensure_builtin_dataset

    return ensure_builtin_dataset(database)
