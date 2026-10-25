"""
Q-Diagnose MVP Runner
======================
Runs the complete Phase 1 MVP: Classical SVM vs Quantum SVC
on the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

Usage: python run_mvp.py

Prints a side-by-side comparison table with real cross-validated metrics.
No fabricated results — every number comes from actual model execution.
"""

import sys
import time
import warnings

# Suppress noisy warnings for cleaner output
warnings.filterwarnings("ignore")

from pipeline import load_breast_cancer_data
from evaluate import (
    build_classical_svm,
    build_qsvc,
    evaluate_model,
    print_comparison_table,
    save_results,
)


def main():
    print("=" * 70)
    print("Q-DIAGNOSE -- Phase 1 MVP")
    print("Hybrid Quantum-Classical ML Platform for Early Disease Risk Detection")
    print("=" * 70)
    print()
    print("[DISCLAIMER] Research and decision-support prototype.")
    print("             Not a clinically validated medical diagnostic device.")
    print()

    # Configuration
    N_COMPONENTS = 4  # PCA components = number of qubits
    N_FOLDS = 5
    SEEDS = [42, 123, 7]

    # Load data
    print("Loading UCI Breast Cancer Wisconsin (Diagnostic) dataset...")
    X, y, feature_names = load_breast_cancer_data()
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Classes: Malignant={sum(y == 0)}, Benign={sum(y == 1)}")
    print(f"  PCA components (qubits): {N_COMPONENTS}")
    print(f"  CV folds: {N_FOLDS}, Seeds: {SEEDS}")
    print()

    results = []

    # --- Classical SVM ---
    print("=" * 50)
    print("TRAINING: Classical SVM (RBF kernel)")
    print("=" * 50)
    start = time.time()
    svm_result = evaluate_model(
        model_builder=build_classical_svm,
        model_name="Classical SVM",
        X=X,
        y=y,
        n_components=N_COMPONENTS,
        n_folds=N_FOLDS,
        seeds=SEEDS,
        verbose=True,
    )
    results.append(svm_result)
    print(f"Classical SVM total time: {time.time() - start:.1f}s")
    print()

    # --- Quantum SVC ---
    print("=" * 50)
    print("TRAINING: Quantum SVC (ZZFeatureMap, 4 qubits, ideal simulator)")
    print("=" * 50)
    start = time.time()

    def qsvc_builder():
        return build_qsvc(n_features=N_COMPONENTS)

    qsvc_result = evaluate_model(
        model_builder=qsvc_builder,
        model_name="Quantum SVC",
        X=X,
        y=y,
        n_components=N_COMPONENTS,
        n_folds=N_FOLDS,
        seeds=SEEDS,
        verbose=True,
    )
    results.append(qsvc_result)
    print(f"Quantum SVC total time: {time.time() - start:.1f}s")
    print()

    # --- Comparison ---
    print_comparison_table(results)

    # --- Save results ---
    save_results(results, "results/mvp_results.json")

    # --- Print representative fold details ---
    for r in results:
        rep = r["representative_fold"]
        if rep:
            cm = rep["confusion_matrix"]
            print(f"\n{r['model_name']} - Representative Fold Confusion Matrix:")
            if "tn" in cm:
                print(f"  TN={cm['tn']}  FP={cm['fp']}")
                print(f"  FN={cm['fn']}  TP={cm['tp']}")

    print("\n[SUCCESS] Phase 1 MVP complete. All metrics are from actual model runs.")
    print("  Results saved to results/mvp_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
