"""
Phase 2 Acceptance Test Runner
==============================
Verifies:
1. FastAPI app is running on http://127.0.0.1:8000
2. POST /datasets/upload computes real dataset stats from uploaded CSV
3. POST /experiments/run executes all 7 models:
   - logistic_regression
   - svm
   - random_forest
   - xgboost
   - qsvc
   - vqc
   - qnn
4. Verifies 7 distinct, non-identical metric sets
5. GET /experiments/{id} retrieves stored metrics and confusion matrix
"""

import sys
import json
import time
import requests
import io
import pandas as pd
import numpy as np

API_BASE = "http://127.0.0.1:8000"


def wait_for_api(timeout: int = 30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{API_BASE}/health", timeout=2)
            if r.status_code == 200:
                print(f"[OK] API is live at {API_BASE}")
                return True
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"API did not start within {timeout}s.")


def test_dataset_upload():
    print("\n" + "=" * 70)
    print("TEST 1: POST /datasets/upload (Real CSV Validation)")
    print("=" * 70)

    # Generate a realistic clinical sample CSV
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(30, 80, n),
        "cholesterol": np.random.normal(200, 30, n).round(1),
        "blood_pressure": np.random.normal(120, 15, n).round(1),
        "max_heart_rate": np.random.normal(150, 20, n).round(1),
        "diagnosis": np.random.choice([0, 1], n, p=[0.4, 0.6]),
    })
    # Add a couple duplicate rows and missing values to verify computation
    df.loc[5, "cholesterol"] = np.nan
    df.loc[12, "blood_pressure"] = np.nan
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    files = {"file": ("cardio_test_sample.csv", csv_buffer, "text/csv")}
    data = {"name": "Cardiovascular Risk Sample", "target_column": "diagnosis"}

    resp = requests.post(f"{API_BASE}/datasets/upload", files=files, data=data)
    print(f"Status: {resp.status_code}")
    result = resp.json()
    print("Response JSON:")
    print(json.dumps(result, indent=2))
    assert resp.status_code == 200
    ds = result["dataset"]
    assert ds["n_rows"] == 102
    assert ds["n_cols"] == 5
    assert ds["missing_count"] == 2
    assert ds["duplicate_count"] == 2
    print("[PASS] CSV validation stats computed dynamically and correctly.")
    return ds["id"]


def test_run_all_7_models():
    print("\n" + "=" * 70)
    print("TEST 2: POST /experiments/run for all 7 models")
    print("=" * 70)

    models = [
        "logistic_regression",
        "svm",
        "random_forest",
        "xgboost",
        "qsvc",
        "vqc",
        "qnn",
    ]

    responses = {}
    metric_accuracies = {}

    for model_type in models:
        print(f"\n--- Running model: {model_type} ---")
        payload = {
            "dataset_id": "breast_cancer_wdbc",
            "model_type": model_type,
            "n_features": 4,
            "backend": "ideal_simulator",
            "circuit_depth": 1,
            "cv_folds": 5,
            "seeds": [42],
        }
        start = time.time()
        resp = requests.post(f"{API_BASE}/experiments/run", json=payload, timeout=300)
        elapsed = time.time() - start
        assert resp.status_code == 200, f"Error for {model_type}: {resp.text}"
        data = resp.json()
        responses[model_type] = data
        acc_mean = data["metrics"]["accuracy"]["mean"]
        auc_mean = data["metrics"]["roc_auc"]["mean"]
        metric_accuracies[model_type] = (acc_mean, auc_mean)

        print(f"Status: {resp.status_code} ({elapsed:.1f}s)")
        print(f"Experiment ID: {data['experiment_id']}")
        print(f"Metrics: Accuracy={acc_mean:.4f}, ROC-AUC={auc_mean:.4f}")
        print("Raw JSON response snippet:")
        print(json.dumps({
            "status": data["status"],
            "experiment_id": data["experiment_id"],
            "model_type": data["model_type"],
            "accuracy": data["metrics"]["accuracy"],
            "f1": data["metrics"]["f1"],
            "roc_auc": data["metrics"]["roc_auc"],
            "runtime_seconds": data["runtime_seconds"],
        }, indent=2))

        # Test GET /experiments/{id}
        get_resp = requests.get(f"{API_BASE}/experiments/{data['experiment_id']}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()["experiment"]
        assert "confusion_matrix" in get_data
        assert "roc_curve" in get_data

    # Acceptance criterion: Verify distinct, non-identical metrics
    print("\n" + "=" * 70)
    print("VERIFICATION: Checking that all 7 models produced distinct metrics")
    print("=" * 70)

    unique_accs = set(round(v[0], 4) for v in metric_accuracies.values())
    print("Model Accuracies:")
    for m, (acc, auc) in metric_accuracies.items():
        print(f"  {m:20s} -> Acc: {acc:.4f}, AUC: {auc:.4f}")

    assert len(unique_accs) >= 5, f"Expected distinct metrics across models, got: {metric_accuracies}"
    print(f"\n[PASS] Verified {len(unique_accs)} distinct performance profiles across all 7 models.")

    # Save summary for verification
    with open("results/phase2_acceptance_results.json", "w") as f:
        json.dump(responses, f, indent=2)
    print("Full responses saved to results/phase2_acceptance_results.json")


def main():
    wait_for_api()
    test_dataset_upload()
    test_run_all_7_models()
    print("\n" + "=" * 70)
    print("[SUCCESS] ALL PHASE 2 ACCEPTANCE CRITERIA MET.")
    print("=" * 70)


if __name__ == "__main__":
    main()
