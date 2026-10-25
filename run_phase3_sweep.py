"""
Phase 3 Sweep & Noise Study Runner
==================================
Calls POST /experiments/sweep to run VQC across circuit depths [1, 2, 3, 4].
Captures actual execution metrics and prints the real observed trend.
"""

import sys
import json
import time
import requests

API_BASE = "http://127.0.0.1:8000"


def main():
    print("=" * 70)
    print("PHASE 3: Running VQC Circuit Depth Sweep [1, 2, 3, 4] via API")
    print("=" * 70)

    payload = {
        "dataset_id": "breast_cancer_wdbc",
        "model_type": "vqc",
        "sweep_param": "circuit_depth",
        "sweep_values": [1, 2, 3, 4],
        "n_features": 4,
        "backend": "ideal_simulator",
        "circuit_depth": 1,
        "cv_folds": 3,
        "seeds": [42],
    }

    start = time.time()
    resp = requests.post(f"{API_BASE}/experiments/sweep", json=payload, timeout=600)
    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()

    print(f"\n[OK] Sweep completed in {elapsed:.1f}s. HTTP Status: {resp.status_code}")
    print("\nACTUAL SWEEP JSON OUTPUT:")
    print("----------------------------------------------------------------------")
    print(json.dumps(data, indent=2))
    print("----------------------------------------------------------------------")

    # Save to disk
    with open("results/phase3_sweep_results.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\nSaved sweep results to results/phase3_sweep_results.json")

    # Trend Summary
    print("\nObserved Trend Across Circuit Depths:")
    print(f"{'Depth':<8}{'Accuracy (Mean +/- Std)':<26}{'ROC-AUC':<14}{'Runtime(s)':<12}")
    for item in data["results"]:
        d = item["param_value"]
        m = item["metrics"]
        acc_str = f"{m['accuracy']['mean']:.4f} +/- {m['accuracy']['std']:.4f}"
        auc_str = f"{m['roc_auc']['mean']:.4f}"
        rt_str = f"{item['runtime_seconds']:.1f}s"
        print(f"{d:<8}{acc_str:<26}{auc_str:<14}{rt_str:<12}")


if __name__ == "__main__":
    main()
