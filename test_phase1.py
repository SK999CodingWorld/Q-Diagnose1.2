from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from evaluate import evaluate_classical, evaluate_quantum
from pipeline import fit_preprocessor, load_dataset, preprocess_data, stratified_split, transform_with_preprocessor
from storage import get_experiment, initialize_database, register_dataset, save_experiment


def assert_metrics_match_reference(y_true: np.ndarray, output: object) -> None:
    predictions = output.predictions
    probabilities = output.probabilities
    expected = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "sensitivity": recall_score(y_true, predictions, zero_division=0),
        "specificity": recall_score(y_true, predictions, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }
    for name, reference in expected.items():
        assert np.isclose(output.metrics[name], reference)
    assert output.confusion_matrix == confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).tolist()


def test_preprocessing_fits_scaler_and_pca_only_on_training_indices() -> None:
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=17,
    )
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=17)
    fitted = fit_preprocessor(X_train, n_components=3, random_state=17)
    transformed_train = transform_with_preprocessor(fitted, X_train)
    transformed_test = transform_with_preprocessor(fitted, X_test)
    expected_train = fitted.pca.transform(fitted.scaler.transform(X_train))
    expected_test = fitted.pca.transform(fitted.scaler.transform(X_test))
    assert np.allclose(transformed_train, expected_train)
    assert np.allclose(transformed_test, expected_test)
    assert np.allclose(fitted.scaler.mean_, X_train.mean(axis=0))
    assert np.allclose(fitted.pca.mean_, fitted.scaler.transform(X_train).mean(axis=0))
    assert not np.allclose(fitted.scaler.mean_, X.mean(axis=0))
    assert transformed_train.shape == (48, 3)
    assert transformed_test.shape == (12, 3)
    assert y_train.shape == (48,)
    assert y_test.shape == (12,)


def test_public_preprocess_data_returns_fitted_objects() -> None:
    X_train, X_test, y_train, y_test, scaler, pca = preprocess_data(
        n_components=4,
        random_state=42,
    )
    assert X_train.shape == (455, 4)
    assert X_test.shape == (114, 4)
    assert y_train.shape == (455,)
    assert y_test.shape == (114,)
    assert scaler.n_samples_seen_ == 455
    assert pca.n_samples_ == 455


def test_classical_evaluation_matches_sklearn_on_toy_data() -> None:
    X, y = make_classification(
        n_samples=48,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=23,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=23, stratify=y
    )
    output = evaluate_classical(X_train, y_train, X_test, y_test, seed=23)
    assert_metrics_match_reference(y_test, output)


def test_quantum_evaluation_matches_sklearn_on_toy_data() -> None:
    X, y = make_classification(
        n_samples=40,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=29,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=29, stratify=y
    )
    output = evaluate_quantum(X_train, y_train, X_test, y_test, seed=29, reps=1)
    assert_metrics_match_reference(y_test, output)


def test_experiment_is_persisted_with_reproducibility_fields(tmp_path) -> None:
    database = tmp_path / "experiments.db"
    dataset_id = register_dataset(
        name="toy",
        source="test",
        target_column="target",
        n_rows=4,
        n_columns=2,
        class_distribution={"0": 2, "1": 2},
        path=database,
    )
    experiment_id = save_experiment(
        dataset_id=dataset_id,
        model_type="classical_svm",
        config={"seed": 7},
        metrics={"accuracy": {"mean": 0.75, "std": 0.1}},
        confusion_matrix=[[1, 1], [0, 1]],
        roc_curve={"fpr": [0.0], "tpr": [1.0], "thresholds": [None]},
        pr_curve={"precision": [1.0], "recall": [1.0], "thresholds": []},
        runtime_seconds=0.2,
        fold_results=[
            {
                "seed": 7,
                "fold": 0,
                "n_train": 3,
                "n_validation": 1,
                "metrics": {"accuracy": 0.75},
                "confusion_matrix": [[1, 1], [0, 1]],
                "roc_curve": {"fpr": [0.0], "tpr": [1.0], "thresholds": [None]},
                "pr_curve": {"precision": [1.0], "recall": [1.0], "thresholds": []},
                "runtime_seconds": 0.2,
            }
        ],
        path=database,
    )
    stored = get_experiment(experiment_id, path=database)
    assert stored is not None
    assert stored["model_type"] == "classical_svm"
    assert stored["config"]["seed"] == 7
    assert stored["metrics"]["accuracy"]["mean"] == 0.75
    assert initialize_database(database) == database
