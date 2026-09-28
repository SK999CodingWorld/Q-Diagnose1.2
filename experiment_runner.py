from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from datasets import load_registered_dataset
from evaluate import (
    EvaluationOutput,
    aggregate_metric_values,
    compute_metrics,
    mean_confusion_matrix,
    positive_probability,
)
from model_registry import (
    MAX_QUANTUM_FEATURES,
    QUANTUM_MODEL_TYPES,
    HardwareNotConfigured,
    build_model,
    normalize_backend,
    normalize_model_type,
    qnn_probability_from_score,
    real_hardware_status,
    seed_quantum_globals,
)
from pipeline import (
    TabularFittedPreprocessor,
    fit_tabular_preprocessor,
    transform_with_tabular_preprocessor,
)
from storage import (
    complete_experiment,
    create_experiment,
    fail_experiment,
    get_dataset,
    get_experiment,
    update_experiment_progress,
)


class ExperimentConfigurationError(ValueError):
    pass


PIPELINE_REVISION = "2.3.0-quantum-determinism"


@dataclass(frozen=True)
class ExperimentSettings:
    dataset_id: int
    model_type: str
    n_features: int = 4
    backend: str = "ideal_simulator"
    circuit_depth: int = 2
    cv_folds: int = 5
    seeds: tuple[int, ...] = (42, 123, 2025)
    optimizer: str = "cobyla"
    max_iterations: int = 25
    shots: int = 512
    noise_multiplier: float = 0.01
    split_seed: int = 42
    test_size: float = 0.2

    def validate(self) -> None:
        if self.model_type not in {
            "logistic_regression",
            "svm",
            "random_forest",
            "xgboost",
            "qsvc",
            "vqc",
            "qnn",
        }:
            raise ExperimentConfigurationError(f"Unsupported model_type: {self.model_type}")
        if self.backend not in {"ideal_simulator", "noisy_simulator", "real_hardware"}:
            raise ExperimentConfigurationError(f"Unsupported backend: {self.backend}")
        if self.n_features < 1:
            raise ExperimentConfigurationError("n_features must be positive")
        if self.model_type in QUANTUM_MODEL_TYPES and self.n_features > MAX_QUANTUM_FEATURES:
            raise ExperimentConfigurationError(
                f"Quantum models support at most {MAX_QUANTUM_FEATURES} features; got {self.n_features}"
            )
        if self.cv_folds < 3:
            raise ExperimentConfigurationError("cv_folds must be at least 3")
        if len(self.seeds) < 3:
            raise ExperimentConfigurationError("At least three seeds are required")
        if self.circuit_depth < 1:
            raise ExperimentConfigurationError("circuit_depth must be positive")
        if self.max_iterations < 1:
            raise ExperimentConfigurationError("max_iterations must be positive")
        if not isinstance(self.shots, int) or isinstance(self.shots, bool):
            raise ExperimentConfigurationError(
                f"shots must be an integer number of samples; got {self.shots!r}. "
                "Shot sampling is required, so shots=None is not supported."
            )
        if self.shots < 1:
            raise ExperimentConfigurationError("shots must be positive")
        if self.noise_multiplier < 0:
            raise ExperimentConfigurationError("noise_multiplier cannot be negative")
        if not 0 < self.test_size < 1:
            raise ExperimentConfigurationError("test_size must be between zero and one")


def settings_to_config(settings: ExperimentSettings) -> dict[str, Any]:
    return {
        "dataset_id": settings.dataset_id,
        "model_type": settings.model_type,
        "pipeline_revision": PIPELINE_REVISION,
        "preprocessing": {
            "split": "stratified_train_test",
            "test_size": settings.test_size,
            "split_random_state": settings.split_seed,
            "scaler": "StandardScaler",
            "pca": "PCA",
            "n_components": settings.n_features,
            "pca_solver": "full",
            "fit_scope": "training_partition_only",
            "categorical_encoding": "OneHotEncoder fitted on each training fold",
            "post_pca_range_scaler": (
                "MinMaxScaler(0,1) fitted on each training fold"
                if settings.model_type in QUANTUM_MODEL_TYPES
                else None
            ),
        },
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "n_splits": settings.cv_folds,
            "shuffle": True,
            "seeds": list(settings.seeds),
            "fit_preprocessing_per_fold": True,
            "cv_scope": "fixed_80_percent_training_partition",
        },
        "backend": settings.backend,
        "circuit_depth": settings.circuit_depth,
        "optimizer": settings.optimizer,
        "max_iterations": settings.max_iterations,
        "shots": settings.shots,
        "noise_multiplier": settings.noise_multiplier if settings.backend == "noisy_simulator" else 0.0,
        "runtime_target_seconds": 120,
    }


def _split_frame(
    X: pd.DataFrame,
    y: np.ndarray,
    settings: ExperimentSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=settings.test_size,
        random_state=settings.split_seed,
        stratify=y,
    )
    return (
        X.iloc[train_indices].reset_index(drop=True),
        X.iloc[test_indices].reset_index(drop=True),
        y[train_indices],
        y[test_indices],
        train_indices,
        test_indices,
    )


def _fit_output(
    model: Any,
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> EvaluationOutput:
    started = time.perf_counter()
    seed_quantum_globals(seed)
    model.fit(X_train, y_train)
    if model_type == "qnn":
        raw_probabilities = np.asarray(model.predict_proba(X_validation))
        if raw_probabilities.ndim == 1:
            probabilities = qnn_probability_from_score(raw_probabilities)
        elif raw_probabilities.shape[1] == 1:
            probabilities = qnn_probability_from_score(raw_probabilities[:, 0])
        else:
            probabilities = positive_probability(model, X_validation)
    else:
        probabilities = positive_probability(model, X_validation)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{model_type} returned non-finite probabilities")
    predictions = (np.asarray(probabilities, dtype=np.float64) >= 0.5).astype(np.int64)
    metrics, matrix, roc_data, pr_data = compute_metrics(
        y_validation,
        predictions,
        probabilities,
    )
    return EvaluationOutput(
        metrics=metrics,
        confusion_matrix=matrix,
        roc_curve=roc_data,
        pr_curve=pr_data,
        runtime_seconds=time.perf_counter() - started,
        predictions=predictions,
        probabilities=probabilities,
    )


def _make_splits(y: np.ndarray, settings: ExperimentSettings) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for seed in settings.seeds:
        splitter = StratifiedKFold(
            n_splits=settings.cv_folds,
            shuffle=True,
            random_state=int(seed),
        )
        for fold, (train_indices, validation_indices) in enumerate(splitter.split(y, y)):
            splits.append((int(seed), int(fold), train_indices, validation_indices))
    return splits


def _fold_record(
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


def _calibration(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bins - 1:
            mask = (predicted >= lower) & (predicted <= upper)
        else:
            mask = (predicted >= lower) & (predicted < upper)
        count = int(np.sum(mask))
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "predicted_probability": float(np.mean(predicted[mask])) if count else None,
                "observed_frequency": float(np.mean(truth[mask])) if count else None,
            }
        )
    return {
        "brier_score": float(brier_score_loss(truth, predicted)),
        "bin_count": bins,
        "bins": rows,
    }


def _target_mapping(dataset_info: dict[str, Any]) -> dict[str, str]:
    metadata = dataset_info.get("metadata") or {}
    summary = dataset_info.get("summary") or {}
    for candidate in (metadata.get("target_mapping"), metadata.get("label_mapping"), summary.get("target_mapping")):
        if isinstance(candidate, dict) and candidate:
            return {str(key): str(value) for key, value in candidate.items()}
    return {}


def _build_for_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    settings: ExperimentSettings,
    seed: int,
) -> tuple[EvaluationOutput, TabularFittedPreprocessor, np.ndarray, np.ndarray, dict[str, Any]]:
    preprocessor = fit_tabular_preprocessor(
        X_train,
        n_components=settings.n_features,
        random_state=seed,
        range_scaling=settings.model_type in QUANTUM_MODEL_TYPES,
    )
    transformed_train = transform_with_tabular_preprocessor(preprocessor, X_train)
    transformed_validation = transform_with_tabular_preprocessor(preprocessor, X_validation)
    model, model_metadata = build_model(
        model_type=settings.model_type,
        n_features=settings.n_features,
        seed=seed,
        backend=settings.backend,
        circuit_depth=settings.circuit_depth,
        optimizer=settings.optimizer,
        max_iterations=settings.max_iterations,
        noise_multiplier=settings.noise_multiplier,
        shots=settings.shots,
    )
    output = _fit_output(
        model,
        settings.model_type,
        transformed_train,
        y_train,
        transformed_validation,
        y_validation,
        seed,
    )
    return output, preprocessor, transformed_train, transformed_validation, model_metadata


def _holdout_and_artifacts(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    settings: ExperimentSettings,
) -> tuple[EvaluationOutput, TabularFittedPreprocessor, np.ndarray, dict[str, Any]]:
    preprocessor = fit_tabular_preprocessor(
        X_train,
        n_components=settings.n_features,
        random_state=settings.split_seed,
        range_scaling=settings.model_type in QUANTUM_MODEL_TYPES,
    )
    transformed_train = transform_with_tabular_preprocessor(preprocessor, X_train)
    transformed_test = transform_with_tabular_preprocessor(preprocessor, X_test)
    model, model_metadata = build_model(
        model_type=settings.model_type,
        n_features=settings.n_features,
        seed=settings.split_seed,
        backend=settings.backend,
        circuit_depth=settings.circuit_depth,
        optimizer=settings.optimizer,
        max_iterations=settings.max_iterations,
        noise_multiplier=settings.noise_multiplier,
        shots=settings.shots,
    )
    output = _fit_output(
        model,
        settings.model_type,
        transformed_train,
        y_train,
        transformed_test,
        y_test,
        settings.split_seed,
    )
    return output, preprocessor, transformed_test, model_metadata


def run_experiment(
    settings: ExperimentSettings,
    database: str | Path = "q_diagnose.db",
    sweep_variable: str | None = None,
    sweep_value: float | None = None,
) -> dict[str, Any]:
    settings.validate()
    dataset_info = get_dataset(settings.dataset_id, path=database)
    if dataset_info is None:
        raise KeyError(f"Dataset {settings.dataset_id} was not found")
    if settings.backend == "real_hardware" and not real_hardware_status()["configured"]:
        raise HardwareNotConfigured("Real hardware not configured")
    X, y, dataset_metadata = load_registered_dataset(settings.dataset_id, database=database)
    if len(np.unique(y)) != 2:
        raise ExperimentConfigurationError("The selected target is not binary")
    X_train, X_test, y_train, y_test, _, _ = _split_frame(X, y, settings)
    class_counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=2)
    if int(class_counts.min()) < settings.cv_folds:
        raise ExperimentConfigurationError(
            f"Each class needs at least cv_folds training rows; found {class_counts.tolist()}"
        )
    config = settings_to_config(settings)
    if sweep_variable is not None:
        config["sweep"] = {
            "variable": sweep_variable,
            "value": sweep_value,
        }
    experiment_id = create_experiment(
        dataset_id=settings.dataset_id,
        model_type=settings.model_type,
        config=config,
        path=database,
    )
    started = time.perf_counter()
    try:
        fold_results: list[dict[str, Any]] = []
        model_metadata: dict[str, Any] | None = None
        splits = _make_splits(y_train, settings)
        for index, (seed, fold, train_indices, validation_indices) in enumerate(splits):
            output, _, _, _, fold_metadata = _build_for_fold(
                X_train.iloc[train_indices].reset_index(drop=True),
                y_train[train_indices],
                X_train.iloc[validation_indices].reset_index(drop=True),
                y_train[validation_indices],
                settings,
                seed,
            )
            model_metadata = fold_metadata
            fold_results.append(
                _fold_record(seed, fold, train_indices, validation_indices, output)
            )
            update_experiment_progress(
                experiment_id,
                0.1 + 0.75 * (index + 1) / len(splits),
                path=database,
            )
        if model_metadata is None:
            raise ExperimentConfigurationError("No cross-validation folds were produced")
        holdout_output, holdout_preprocessor, transformed_test, holdout_metadata = _holdout_and_artifacts(
            X_train,
            y_train,
            X_test,
            y_test,
            settings,
        )
        model_metadata = holdout_metadata
        metrics = aggregate_metric_values(fold_results)
        metrics["confusion_matrix_mean"] = mean_confusion_matrix(fold_results)
        metrics["n_folds"] = len(fold_results)
        representative = fold_results[0]
        calibration = _calibration(y_test, holdout_output.probabilities)
        config["model"] = model_metadata
        config["dataset"] = {
            "id": settings.dataset_id,
            "name": dataset_metadata["name"],
            "n_rows": int(dataset_info["n_rows"]),
            "n_features": int(dataset_info["n_columns"]) - 1,
            "target_column": dataset_info.get("target_column"),
            "target_mapping": _target_mapping(dataset_info),
            "positive_class": 1,
            "positive_class_label": _target_mapping(dataset_info).get("1"),
        }
        config["probability_output"] = {
            "source": "predict_proba",
            "qnn_mapping": "sigmoid_of_network_output" if settings.model_type == "qnn" else None,
        }
        test_data = {
            "X_test": transformed_test.tolist(),
            "y_test": np.asarray(y_test, dtype=np.int64).tolist(),
            "probabilities": holdout_output.probabilities.tolist(),
            "predictions": holdout_output.predictions.tolist(),
            "feature_names": list(holdout_preprocessor.feature_names),
            "epsilon": 0.1,
        }
        holdout = holdout_output.as_dict(include_predictions=False)
        runtime_seconds = time.perf_counter() - started
        complete_experiment(
            experiment_id=experiment_id,
            config=config,
            metrics=metrics,
            confusion_matrix=representative["confusion_matrix"],
            roc_curve=representative["roc_curve"],
            pr_curve=representative["pr_curve"],
            runtime_seconds=runtime_seconds,
            fold_results=fold_results,
            holdout_metrics=holdout,
            feature_names=list(holdout_preprocessor.feature_names),
            test_data=test_data,
            calibration=calibration,
            sweep_variable=sweep_variable,
            sweep_value=sweep_value,
            path=database,
        )
        result = get_experiment(experiment_id, path=database)
        if result is None:
            raise RuntimeError("Experiment was not persisted")
        result["fold_results"] = fold_results
        result["dataset_metadata"] = dataset_metadata
        return result
    except Exception as error:
        fail_experiment(experiment_id, str(error), path=database)
        raise


def settings_from_experiment(experiment: dict[str, Any]) -> ExperimentSettings:
    config = experiment.get("config", {})
    preprocessing = config.get("preprocessing", {})
    cross_validation = config.get("cross_validation", {})
    model = config.get("model", {})
    return ExperimentSettings(
        dataset_id=int(experiment["dataset_id"]),
        model_type=normalize_model_type(experiment["model_type"]),
        n_features=int(preprocessing.get("n_components", 4)),
        backend=normalize_backend(config.get("backend")),
        circuit_depth=max(
            1,
            int(config.get("circuit_depth", model.get("circuit_depth", 2) or 2) or 1),
        ),
        cv_folds=int(cross_validation.get("n_splits", 5)),
        seeds=tuple(int(seed) for seed in cross_validation.get("seeds", [42, 123, 2025])),
        optimizer=str(config.get("optimizer", "cobyla")),
        max_iterations=int(config.get("max_iterations", 25)),
        shots=int(config.get("shots", 512)),
        noise_multiplier=float(config.get("noise_multiplier", 0.0)),
        split_seed=int(preprocessing.get("split_random_state", 42)),
        test_size=float(preprocessing.get("test_size", 0.2)),
    )


def fit_final_for_experiment(
    experiment: dict[str, Any],
    database: str | Path = "q_diagnose.db",
) -> tuple[Any, TabularFittedPreprocessor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    settings = settings_from_experiment(experiment)
    X, y, _ = load_registered_dataset(settings.dataset_id, database=database)
    X_train, X_test, y_train, y_test, _, _ = _split_frame(X, y, settings)
    preprocessor = fit_tabular_preprocessor(
        X_train,
        n_components=settings.n_features,
        random_state=settings.split_seed,
        range_scaling=settings.model_type in QUANTUM_MODEL_TYPES,
    )
    transformed_train = transform_with_tabular_preprocessor(preprocessor, X_train)
    transformed_test = transform_with_tabular_preprocessor(preprocessor, X_test)
    model, _ = build_model(
        model_type=settings.model_type,
        n_features=settings.n_features,
        seed=settings.split_seed,
        backend=settings.backend,
        circuit_depth=settings.circuit_depth,
        optimizer=settings.optimizer,
        max_iterations=settings.max_iterations,
        noise_multiplier=settings.noise_multiplier,
        shots=settings.shots,
    )
    seed_quantum_globals(settings.split_seed)
    model.fit(transformed_train, y_train)
    return model, preprocessor, transformed_train, transformed_test, y_train, y_test
