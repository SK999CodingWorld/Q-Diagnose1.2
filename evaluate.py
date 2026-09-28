from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import BaseKernel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from pipeline import (
    fit_preprocessor,
    load_dataset,
    preprocessing_config,
    stratified_split,
    transform_with_preprocessor,
)
from storage import register_dataset, save_experiment


METRIC_NAMES = (
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
)
DISCLAIMER = "Research and decision-support prototype. Not a clinically validated medical diagnostic device."
DEFAULT_SEEDS = (42, 123, 2025)

warnings.filterwarnings(
    "ignore",
    message="The `probability` parameter was deprecated",
    category=FutureWarning,
)


@dataclass
class EvaluationOutput:
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    roc_curve: dict[str, list[float | None]]
    pr_curve: dict[str, list[float]]
    runtime_seconds: float
    predictions: np.ndarray
    probabilities: np.ndarray

    def as_dict(self, include_predictions: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "metrics": self.metrics,
            "confusion_matrix": self.confusion_matrix,
            "roc_curve": self.roc_curve,
            "pr_curve": self.pr_curve,
            "runtime_seconds": self.runtime_seconds,
        }
        if include_predictions:
            result["predictions"] = self.predictions.tolist()
            result["probabilities"] = self.probabilities.tolist()
        return result


class AerStatevectorKernel(BaseKernel):
    def __init__(
        self,
        feature_map: QuantumCircuit,
        seed: int,
        noise_model: Any | None = None,
    ):
        super().__init__(feature_map=feature_map, enforce_psd=True)
        self._seed = int(seed)
        self._noisy = noise_model is not None
        backend_options: dict[str, Any] = {
            "method": "density_matrix" if self._noisy else "statevector",
            "precision": "double",
        }
        if noise_model is not None:
            backend_options["noise_model"] = noise_model
        self._backend = AerSimulator(**backend_options)
        self._state_cache: dict[bytes, np.ndarray] = {}

    def _statevectors(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        keys = [np.ascontiguousarray(row).tobytes() for row in values]
        missing_keys: list[bytes] = []
        missing_values: list[np.ndarray] = []
        for key, row in zip(keys, values):
            if key not in self._state_cache and key not in missing_keys:
                missing_keys.append(key)
                missing_values.append(row)
        if missing_values:
            circuits = []
            for row in missing_values:
                circuit = self.feature_map.assign_parameters(row)
                if self._noisy:
                    circuit.save_density_matrix()
                else:
                    circuit.save_statevector()
                circuits.append(circuit)
            result = self._backend.run(circuits, seed_simulator=self._seed).result()
            for key, index in zip(missing_keys, range(len(missing_keys))):
                if self._noisy:
                    self._state_cache[key] = np.asarray(
                        result.data(index)["density_matrix"], dtype=np.complex128
                    )
                else:
                    self._state_cache[key] = np.asarray(
                        result.get_statevector(index), dtype=np.complex128
                    )
        if self._noisy:
            return np.stack([self._state_cache[key] for key in keys], axis=0)
        return np.vstack([self._state_cache[key] for key in keys])

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:
        x_values, y_values = self._validate_input(x_vec, y_vec)
        if y_values is None:
            y_values = x_values
        x_states = self._statevectors(x_values)
        y_states = self._statevectors(y_values)
        if self._noisy:
            kernel_matrix = np.einsum(
                "xij,yji->xy",
                x_states,
                y_states,
            ).real
            return np.maximum(kernel_matrix, 0.0)
        kernel_matrix = np.abs(np.conjugate(x_states) @ y_states.T) ** 2
        return kernel_matrix


def build_zz_feature_map(n_features: int, reps: int = 2) -> QuantumCircuit:
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if reps < 1:
        raise ValueError("reps must be positive")
    return zz_feature_map(
        feature_dimension=n_features,
        reps=reps,
        entanglement="full",
        alpha=2.0,
    )


def build_classical_model(seed: int) -> SVC:
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=int(seed),
    )


def build_quantum_model(
    n_features: int,
    seed: int,
    reps: int = 2,
) -> tuple[QSVC, AerStatevectorKernel]:
    feature_map = build_zz_feature_map(n_features, reps=reps)
    quantum_kernel = AerStatevectorKernel(feature_map, seed=seed)
    model = QSVC(
        quantum_kernel=quantum_kernel,
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=int(seed),
    )
    return model, quantum_kernel


def positive_probability(model: Any, X: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("The model did not return two-class probabilities")
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError("The model classes must contain label 1 exactly once")
    return probabilities[:, int(positive_indices[0])]


def finite_curve(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values)]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> tuple[dict[str, float], list[list[int]], dict[str, list[float | None]], dict[str, list[float]]]:
    truth = np.asarray(y_true, dtype=np.int64)
    predictions = np.asarray(y_pred, dtype=np.int64)
    probabilities = np.asarray(y_probability, dtype=np.float64)
    if truth.shape != predictions.shape or truth.shape != probabilities.shape:
        raise ValueError("y_true, y_pred, and y_probability must have matching shapes")
    if not np.isin(truth, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Metrics require binary labels encoded as 0 and 1")
    fpr, tpr, thresholds = roc_curve(truth, probabilities, pos_label=1)
    precision_values, recall_values, pr_thresholds = precision_recall_curve(
        truth, probabilities, pos_label=1
    )
    metrics = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(
            precision_score(truth, predictions, pos_label=1, zero_division=0)
        ),
        "sensitivity": float(
            recall_score(truth, predictions, pos_label=1, zero_division=0)
        ),
        "specificity": float(
            recall_score(truth, predictions, pos_label=0, zero_division=0)
        ),
        "f1": float(f1_score(truth, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "pr_auc": float(average_precision_score(truth, probabilities)),
    }
    matrix = confusion_matrix(truth, predictions, labels=[0, 1]).astype(int).tolist()
    roc_data = {
        "fpr": finite_curve(fpr),
        "tpr": finite_curve(tpr),
        "thresholds": finite_curve(thresholds),
    }
    pr_data = {
        "precision": finite_curve(precision_values),
        "recall": finite_curve(recall_values),
        "thresholds": finite_curve(pr_thresholds),
    }
    return metrics, matrix, roc_data, pr_data


def fit_and_evaluate(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> EvaluationOutput:
    started = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = np.asarray(model.predict(X_validation), dtype=np.int64)
    probabilities = positive_probability(model, X_validation)
    metrics, matrix, roc_data, pr_data = compute_metrics(
        y_validation,
        predictions,
        probabilities,
    )
    runtime_seconds = time.perf_counter() - started
    return EvaluationOutput(
        metrics=metrics,
        confusion_matrix=matrix,
        roc_curve=roc_data,
        pr_curve=pr_data,
        runtime_seconds=runtime_seconds,
        predictions=predictions,
        probabilities=probabilities,
    )


def evaluate_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> EvaluationOutput:
    return fit_and_evaluate(
        build_classical_model(seed),
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


def evaluate_quantum(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    reps: int = 2,
) -> EvaluationOutput:
    model, _ = build_quantum_model(X_train.shape[1], seed=seed, reps=reps)
    return fit_and_evaluate(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


def make_cv_splits(
    y: np.ndarray,
    seeds: Iterable[int],
    n_splits: int = 5,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    labels = np.asarray(y, dtype=np.int64)
    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed),
        )
        for fold, (train_indices, validation_indices) in enumerate(splitter.split(labels, labels)):
            splits.append((int(seed), int(fold), train_indices, validation_indices))
    return splits


def aggregate_metric_values(
    fold_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for metric_name in METRIC_NAMES:
        values = np.asarray(
            [result["metrics"][metric_name] for result in fold_results],
            dtype=np.float64,
        )
        aggregate[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "values": values.tolist(),
        }
    return aggregate


def mean_confusion_matrix(fold_results: list[dict[str, Any]]) -> list[list[float]]:
    matrices = np.asarray(
        [result["confusion_matrix"] for result in fold_results],
        dtype=np.float64,
    )
    return np.mean(matrices, axis=0).tolist()


def fold_record(
    seed: int,
    fold: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    output: EvaluationOutput,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "fold": int(fold),
        "n_train": int(len(train_indices)),
        "n_validation": int(len(validation_indices)),
        "metrics": output.metrics,
        "confusion_matrix": output.confusion_matrix,
        "roc_curve": output.roc_curve,
        "pr_curve": output.pr_curve,
        "runtime_seconds": output.runtime_seconds,
    }


def evaluate_cross_validated(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_components: int,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    n_splits: int = 5,
    reps: int = 2,
) -> dict[str, Any]:
    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if model_type not in {"classical_svm", "qsvc"}:
        raise ValueError(f"Unsupported Phase 1 model type: {model_type}")
    if n_splits < 3:
        raise ValueError("At least three folds are required")
    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) < 3:
        raise ValueError("At least three random seeds are required")
    started = time.perf_counter()
    fold_results: list[dict[str, Any]] = []
    for seed, fold, train_indices, validation_indices in make_cv_splits(
        labels,
        seed_values,
        n_splits=n_splits,
    ):
        preprocessor = fit_preprocessor(
            features[train_indices],
            n_components=n_components,
            random_state=seed,
        )
        X_train = transform_with_preprocessor(preprocessor, features[train_indices])
        X_validation = transform_with_preprocessor(preprocessor, features[validation_indices])
        if model_type == "classical_svm":
            output = evaluate_classical(
                X_train,
                labels[train_indices],
                X_validation,
                labels[validation_indices],
                seed=seed,
            )
        else:
            output = evaluate_quantum(
                X_train,
                labels[train_indices],
                X_validation,
                labels[validation_indices],
                seed=seed,
                reps=reps,
            )
        fold_results.append(
            fold_record(seed, fold, train_indices, validation_indices, output)
        )
    representative = fold_results[0]
    metrics = aggregate_metric_values(fold_results)
    metrics["confusion_matrix_mean"] = mean_confusion_matrix(fold_results)
    metrics["n_folds"] = len(fold_results)
    return {
        "model_type": model_type,
        "metrics": metrics,
        "confusion_matrix": representative["confusion_matrix"],
        "roc_curve": representative["roc_curve"],
        "pr_curve": representative["pr_curve"],
        "representative_fold": {
            "seed": representative["seed"],
            "fold": representative["fold"],
        },
        "fold_results": fold_results,
        "runtime_seconds": time.perf_counter() - started,
    }


def evaluate_holdout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    n_components: int,
    seed: int,
    reps: int = 2,
) -> dict[str, Any]:
    preprocessor = fit_preprocessor(
        X_train,
        n_components=n_components,
        random_state=seed,
    )
    X_train_transformed = transform_with_preprocessor(preprocessor, X_train)
    X_test_transformed = transform_with_preprocessor(preprocessor, X_test)
    if model_type == "classical_svm":
        output = evaluate_classical(
            X_train_transformed,
            y_train,
            X_test_transformed,
            y_test,
            seed=seed,
        )
    else:
        output = evaluate_quantum(
            X_train_transformed,
            y_train,
            X_test_transformed,
            y_test,
            seed=seed,
            reps=reps,
        )
    return output.as_dict(include_predictions=False)


def model_config(
    model_type: str,
    n_components: int,
    reps: int,
    seeds: Iterable[int],
    n_splits: int,
    split_random_state: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_type": model_type,
        "preprocessing": preprocessing_config(
            n_components=n_components,
            test_size=0.2,
            random_state=split_random_state,
        ),
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "n_splits": n_splits,
            "shuffle": True,
            "seeds": [int(seed) for seed in seeds],
            "fit_preprocessing_per_fold": True,
            "cv_scope": "fixed_80_percent_training_partition",
        },
        "runtime_target_seconds": 120,
    }
    if model_type == "classical_svm":
        config["estimator"] = {
            "library": "sklearn.svm.SVC",
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state_per_seed": True,
        }
        config["qubits"] = 0
        config["circuit_depth"] = 0
        config["backend"] = "classical"
    else:
        config["estimator"] = {
            "library": "qiskit_machine_learning.algorithms.QSVC",
            "kernel": "evaluate.AerStatevectorKernel",
            "kernel_class": "AerStatevectorKernel",
            "kernel_basis": "statevector simulation of ZZFeatureMap",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state_per_seed": True,
        }
        config["feature_map"] = {
            "name": "ZZFeatureMap",
            "feature_dimension": n_components,
            "reps": reps,
            "entanglement": "full",
            "alpha": 2.0,
        }
        config["qubits"] = n_components
        config["circuit_depth"] = int(
            build_zz_feature_map(n_components, reps=reps).depth()
        )
        config["backend"] = {
            "name": "qiskit_aer",
            "simulator": "statevector",
            "noiseless": True,
            "shots": None,
            "statevector_precision": "double",
        }
    return config


def run_mvp_evaluation(
    n_components: int = 4,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    n_splits: int = 5,
    reps: int = 2,
    split_random_state: int = 42,
    database: str | Path = "q_diagnose.db",
) -> dict[str, Any]:
    started = time.perf_counter()
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = stratified_split(
        X,
        y,
        test_size=0.2,
        random_state=split_random_state,
    )
    class_distribution = {
        str(int(label)): int(np.sum(y == label)) for label in np.unique(y)
    }
    dataset_id = register_dataset(
        name="UCI Breast Cancer Wisconsin (Diagnostic)",
        source="sklearn.datasets.load_breast_cancer",
        target_column="target",
        n_rows=X.shape[0],
        n_columns=X.shape[1],
        class_distribution=class_distribution,
        metadata={"label_mapping": {"0": "malignant", "1": "benign"}},
        path=database,
    )
    seed_values = tuple(int(seed) for seed in seeds)
    model_results: list[dict[str, Any]] = []
    for model_type in ("classical_svm", "qsvc"):
        cv_result = evaluate_cross_validated(
            X_train,
            y_train,
            model_type=model_type,
            n_components=n_components,
            seeds=seed_values,
            n_splits=n_splits,
            reps=reps,
        )
        holdout = evaluate_holdout(
            X_train,
            y_train,
            X_test,
            y_test,
            model_type=model_type,
            n_components=n_components,
            seed=split_random_state,
            reps=reps,
        )
        config = model_config(
            model_type=model_type,
            n_components=n_components,
            reps=reps,
            seeds=seed_values,
            n_splits=n_splits,
            split_random_state=split_random_state,
        )
        runtime_seconds = float(cv_result["runtime_seconds"] + holdout["runtime_seconds"])
        experiment_id = save_experiment(
            dataset_id=dataset_id,
            model_type=model_type,
            config=config,
            metrics=cv_result["metrics"],
            confusion_matrix=cv_result["confusion_matrix"],
            roc_curve=cv_result["roc_curve"],
            pr_curve=cv_result["pr_curve"],
            runtime_seconds=runtime_seconds,
            fold_results=cv_result["fold_results"],
            holdout_metrics=holdout,
            path=database,
        )
        model_results.append(
            {
                "experiment_id": experiment_id,
                "model_type": model_type,
                "config": config,
                "metrics": cv_result["metrics"],
                "holdout": holdout,
                "confusion_matrix": cv_result["confusion_matrix"],
                "roc_curve": cv_result["roc_curve"],
                "pr_curve": cv_result["pr_curve"],
                "representative_fold": cv_result["representative_fold"],
                "runtime_seconds": runtime_seconds,
            }
        )
    return {
        "application": "Q-Diagnose",
        "phase": 1,
        "disclaimer": DISCLAIMER,
        "dataset": {
            "id": dataset_id,
            "name": "UCI Breast Cancer Wisconsin (Diagnostic)",
            "source": "sklearn.datasets.load_breast_cancer",
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "class_distribution": class_distribution,
            "split_random_state": split_random_state,
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
        },
        "preprocessing": preprocessing_config(
            n_components=n_components,
            test_size=0.2,
            random_state=split_random_state,
        ),
        "models": model_results,
        "database_path": str(Path(database)),
        "runtime_seconds": time.perf_counter() - started,
    }


def format_metric(metric: dict[str, Any]) -> str:
    return f"{metric['mean']:.4f} +/- {metric['std']:.4f}"


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    headers = (
        "Model",
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1",
        "ROC-AUC",
        "PR-AUC",
        "Runtime (s)",
    )
    rows = [headers]
    for result in results:
        metrics = result["metrics"]
        rows.append(
            (
                "Classical SVM" if result["model_type"] == "classical_svm" else "QSVC",
                *(format_metric(metrics[name]) for name in METRIC_NAMES),
                f"{result['runtime_seconds']:.2f}",
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows[1:]:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def write_results(results: dict[str, Any], path: str | Path = "phase1_results.json") -> Path:
    output_path = Path(path)
    output_path.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    results = run_mvp_evaluation()
    output_path = write_results(results)
    print(DISCLAIMER)
    print(f"Dataset: {results['dataset']['n_rows']} rows, {results['dataset']['n_features']} features")
    print("Cross-validation: 5 folds x 3 seeds on the fixed 80% training partition")
    print_comparison_table(results["models"])
    print(f"Saved results: {output_path}")
    print(f"Saved SQLite database: {results['database_path']}")
    print(f"Total runtime: {results['runtime_seconds']:.2f} seconds")
    return 0
