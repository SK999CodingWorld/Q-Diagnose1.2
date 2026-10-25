"""
Q-Diagnose Evaluation Module
==============================
Runs models under stratified k-fold cross-validation with multiple seeds.
Computes: accuracy, precision, sensitivity, specificity, F1, ROC-AUC, PR-AUC.
Outputs JSON results and printed comparison tables.

Every metric comes from actual model execution — no fabricated numbers.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    balanced_accuracy_score,
    brier_score_loss,
)
from sklearn.svm import SVC

from qdiagnose.domain.pipeline import run_pipeline_cv_fold

def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity = TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all required metrics.

    Parameters
    ----------
    y_true : ground truth labels
    y_pred : predicted labels
    y_prob : predicted probabilities for the positive class (for AUC)

    Returns
    -------
    dict with keys: accuracy, precision, sensitivity, specificity, f1,
                    roc_auc, pr_auc
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(compute_specificity(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["pr_auc"] = float("nan")
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        metrics["brier_score"] = float("nan")

    return metrics


def get_confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return confusion matrix as a labeled dict."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def get_roc_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, List[float]]:
    """Compute ROC curve data points."""
    if y_prob is None:
        return {"fpr": [], "tpr": [], "thresholds": []}
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full metric set and confusion matrix at a specific operating threshold.
    Crucial for early disease detection triage where sensitivity vs specificity
    must be tuned dynamically.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    threshold = float(np.clip(threshold, 0.0, 1.0))
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "metrics": {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
        }
    }


def generate_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    steps: int = 100,
) -> List[Dict[str, Any]]:
    """
    Generate dense operating characteristics over swept threshold values (0.01 to 0.99).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.linspace(0.01, 0.99, steps)
    results = []

    for t in thresholds:
        res = compute_metrics_at_threshold(y_true, y_prob, threshold=float(t))
        cm = res["confusion_matrix"]
        m = res["metrics"]
        tpr = m["sensitivity"]
        fpr = float(cm["fp"] / (cm["tn"] + cm["fp"])) if (cm["tn"] + cm["fp"]) > 0 else 0.0
        results.append({
            "threshold": round(float(t), 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "accuracy": round(m["accuracy"], 4),
            "confusion_matrix": cm,
        })
    return results


from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_machine_learning.kernels import BaseKernel


class StatevectorQuantumKernel(BaseKernel):
    """
    Fidelity kernel with an exact statevector path when noise_rate is zero and
    a density-matrix Aer path with configured depolarizing gate errors otherwise.
    """
    def __init__(self, feature_map, noise_rate: float = 0.0):
        self._feature_map = feature_map
        self.noise_rate = float(noise_rate)
        self._density_cache = {}
        self._noisy_backend = None
        if self.noise_rate > 0.0:
            noise = NoiseModel()
            one_qubit = depolarizing_error(self.noise_rate, 1)
            two_qubit = depolarizing_error(min(0.5, self.noise_rate * 1.5), 2)
            noise.add_all_qubit_quantum_error(one_qubit, "u")
            noise.add_all_qubit_quantum_error(two_qubit, "cx")
            self._noisy_backend = AerSimulator(method="density_matrix", noise_model=noise)
        super().__init__(feature_map=feature_map)

    def _density_matrices(self, values):
        matrices = []
        for row in np.asarray(values, dtype=np.float64):
            key = np.ascontiguousarray(row).tobytes()
            if key not in self._density_cache:
                circuit = self._feature_map.assign_parameters(row)
                circuit.save_density_matrix()
                compiled = transpile(circuit, self._noisy_backend, optimization_level=0)
                result = self._noisy_backend.run(compiled).result()
                self._density_cache[key] = np.asarray(
                    result.data(0)["density_matrix"], dtype=np.complex128
                )
            matrices.append(self._density_cache[key])
        return np.stack(matrices, axis=0)

    def evaluate(self, x_vec, y_vec=None):
        if y_vec is None:
            y_vec = x_vec

        if self._noisy_backend is not None:
            rho_x = self._density_matrices(x_vec)
            rho_y = rho_x if y_vec is x_vec else self._density_matrices(y_vec)
            return np.einsum("xij,yji->xy", rho_x, rho_y).real

        sv_x = np.array([
            Statevector.from_instruction(self._feature_map.assign_parameters(x)).data
            for x in x_vec
        ])
        if x_vec is y_vec or (isinstance(x_vec, np.ndarray) and isinstance(y_vec, np.ndarray) and np.array_equal(x_vec, y_vec)):
            sv_y = sv_x
        else:
            sv_y = np.array([
                Statevector.from_instruction(self._feature_map.assign_parameters(y)).data
                for y in y_vec
            ])
        return np.abs(np.dot(np.conj(sv_x), sv_y.T)) ** 2


def build_classical_svm():
    """Build the classical SVM baseline."""
    return SVC(kernel="rbf", probability=True, random_state=42)


def build_qsvc(n_features: int):
    """
    Build the Quantum SVC using Qiskit Machine Learning.

    Uses ZZFeatureMap with feature_dimension=n_features, reps=2,
    linear entanglement, evaluated via statevector quantum kernel.
    Number of qubits = number of PCA components (n_features).
    Uses standard angle encoding (scaling PCA features to [0, pi]
    fitted strictly on the training fold).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.algorithms import QSVC

    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=2,
        entanglement="linear",
    )
    kernel = StatevectorQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel, random_state=42)

    return Pipeline([
        ("angle_scaler", MinMaxScaler(feature_range=(0, np.pi))),
        ("qsvc", qsvc)
    ])


def evaluate_model(
    model_builder,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 4,
    n_folds: int = 5,
    seeds: List[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model under stratified k-fold CV with multiple seeds.

    Parameters
    ----------
    model_builder : callable that returns a fresh model instance
    model_name : str for display/logging
    X : raw feature array (pre-pipeline)
    y : label array
    n_components : PCA components
    n_folds : number of CV folds
    seeds : list of random seeds
    verbose : whether to print progress

    Returns
    -------
    dict with aggregated metrics, per-fold results, and representative
    fold data (confusion matrix, ROC curve)
    """
    if seeds is None:
        seeds = [42, 123, 7]

    class_labels, class_counts = np.unique(y, return_counts=True)
    if len(class_labels) != 2:
        raise ValueError(f"Binary evaluation requires exactly two target classes; found {len(class_labels)}.")
    if int(class_counts.min()) < n_folds:
        raise ValueError(
            f"Each target class needs at least {n_folds} rows for stratified cross-validation; "
            f"the smallest class has {int(class_counts.min())}."
        )

    all_fold_metrics = []
    out_of_fold_predictions = np.empty(len(y), dtype=np.int64)
    out_of_fold_probabilities = np.empty(len(y), dtype=np.float64)
    out_of_fold_seen = np.zeros(len(y), dtype=bool)
    total_start = time.time()

    for seed_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            if verbose:
                print(
                    f"  [{model_name}] Seed {seed}, Fold {fold_idx + 1}/{n_folds}...",
                    end="",
                    flush=True,
                )

            fold_start = time.time()

            # Run pipeline for this fold (no leakage: fit on train only)
            pipe = run_pipeline_cv_fold(
                X, y, train_idx, test_idx, n_components=n_components
            )

            # Build fresh model
            model = model_builder()

            # Train
            model.fit(pipe.X_train, pipe.y_train)

            # Predict
            y_pred = model.predict(pipe.X_test)

            # Get probabilities if available
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(pipe.X_test)
                    if proba.ndim == 2 and proba.shape[1] == 2:
                        y_prob = proba[:, 1]
                    elif proba.ndim == 1:
                        y_prob = proba
                    else:
                        raise ValueError(f"Unsupported predict_proba shape: {proba.shape}")
                except Exception as error:
                    raise RuntimeError(
                        f"{model_name} failed to produce probabilities for seed {seed}, fold {fold_idx + 1}."
                    ) from error
            elif hasattr(model, "decision_function"):
                try:
                    from scipy.special import expit
                    df_val = model.decision_function(pipe.X_test)
                    y_prob = expit(df_val)
                except Exception as error:
                    raise RuntimeError(
                        f"{model_name} failed to produce scores for seed {seed}, fold {fold_idx + 1}."
                    ) from error

            if y_prob is None or len(y_prob) != len(pipe.y_test) or not np.isfinite(y_prob).all():
                raise ValueError(f"{model_name} returned invalid probability values for seed {seed}, fold {fold_idx + 1}.")

            fold_time = time.time() - fold_start

            # Compute metrics
            metrics = compute_all_metrics(pipe.y_test, y_pred, y_prob)
            metrics["runtime_seconds"] = fold_time
            metrics["seed"] = seed
            metrics["fold"] = fold_idx
            all_fold_metrics.append(metrics)

            if verbose:
                print(
                    f" Acc={metrics['accuracy']:.4f}, "
                    f"F1={metrics['f1']:.4f}, "
                    f"AUC={metrics['roc_auc']:.4f} "
                    f"({fold_time:.1f}s)"
                )

            # One stratified CV repetition gives every row exactly one
            # out-of-fold prediction; use it for the detailed charts.
            if seed_idx == 0:
                out_of_fold_predictions[test_idx] = y_pred
                out_of_fold_probabilities[test_idx] = y_prob
                out_of_fold_seen[test_idx] = True

    if not out_of_fold_seen.all():
        raise RuntimeError("The first cross-validation repetition did not produce predictions for every row.")
    representative_fold = {
        "aggregation": "all_folds_first_seed_out_of_fold",
        "seed": seeds[0],
        "confusion_matrix": get_confusion_matrix_dict(y, out_of_fold_predictions),
        "roc_curve": get_roc_curve_data(y, out_of_fold_probabilities),
        "y_true": np.asarray(y).tolist(),
        "y_pred": out_of_fold_predictions.tolist(),
        "y_prob": out_of_fold_probabilities.tolist(),
    }

    total_time = time.time() - total_start

    # Aggregate metrics across all folds and seeds
    metric_names = [
        "accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity",
        "f1", "roc_auc", "pr_auc", "brier_score",
    ]
    aggregated = {}
    for m in metric_names:
        values = [f[m] for f in all_fold_metrics if not np.isnan(f[m])]
        if values:
            aggregated[m] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            aggregated[m] = {"mean": float("nan"), "std": float("nan")}

    runtime_values = [f["runtime_seconds"] for f in all_fold_metrics]
    aggregated["runtime_seconds"] = {
        "mean": float(np.mean(runtime_values)),
        "total": float(sum(runtime_values)),
    }

    result = {
        "model_name": model_name,
        "n_components": n_components,
        "n_folds": n_folds,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "total_evaluations": len(all_fold_metrics),
        "total_runtime_seconds": total_time,
        "aggregated_metrics": aggregated,
        "per_fold_metrics": all_fold_metrics,
        "representative_fold": representative_fold,
    }

    return result


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table of model results."""
    print("\n" + "=" * 100)
    print("Q-DIAGNOSE -- MODEL COMPARISON (Stratified 5-Fold CV x 3 Seeds)")
    print("Dataset: UCI Breast Cancer Wisconsin (Diagnostic) -- 569 samples, 30 -> 4 PCA features")
    print("[DISCLAIMER] Research prototype. Not a clinically validated diagnostic device.")
    print("=" * 100)

    metric_names = [
        "accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity",
        "f1", "roc_auc", "pr_auc", "brier_score",
    ]
    headers = ["Model"] + [m.upper().replace("_", "-") for m in metric_names] + ["Runtime(s)"]

    # Compute column widths
    col_widths = [max(20, len(h) + 2) for h in headers]
    col_widths[0] = 22  # Model name column

    # Header row
    header_str = ""
    for h, w in zip(headers, col_widths):
        header_str += h.center(w)
    print(header_str)
    print("-" * sum(col_widths))

    # Data rows
    for r in results:
        agg = r["aggregated_metrics"]
        row_values = [r["model_name"]]
        for m in metric_names:
            if m in agg and not np.isnan(agg[m]["mean"]):
                row_values.append(
                    f"{agg[m]['mean']:.4f} +/- {agg[m]['std']:.4f}"
                )
            else:
                row_values.append("N/A")
        row_values.append(f"{r['total_runtime_seconds']:.1f}")

        row_str = ""
        for v, w in zip(row_values, col_widths):
            row_str += str(v).center(w)
        print(row_str)

    print("=" * sum(col_widths))
    print()


def save_results(
    results: List[Dict[str, Any]],
    output_path: str = "results.json",
) -> None:
    """Save results to JSON file."""

    def convert(obj):
        """Make numpy types JSON-serializable."""
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"Results saved to {path.resolve()}")
