"""
Phase 2 Benchmark Runner
=========================
Executes all 7 supported models on UCI Breast Cancer Wisconsin (Diagnostic):
- Classical: Logistic Regression, SVM, Random Forest, XGBoost
- Quantum: QSVC, VQC, QNN

Under stratified 5-fold cross-validation with 3 random seeds (15 folds each for classical & QSVC,
3 folds for iterative VQC/QNN to manage quantum iteration times).
Saves results to SQLite database and prints the comparison table.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from main import app
import database as db

client = TestClient(app)

print("=" * 110)
print("Q-DIAGNOSE -- PHASE 2 FULL ARCHITECTURE BENCHMARK (All 7 Models)")
print("Dataset: UCI Breast Cancer Wisconsin (Diagnostic) -- 569 samples, 30 -> 4 PCA features")
print("[DISCLAIMER] Research prototype. Not a clinically validated diagnostic device.")
print("=" * 110)

models_to_run = [
    ("logistic_regression", 5, [42, 123, 7]),
    ("svm", 5, [42, 123, 7]),
    ("random_forest", 5, [42, 123, 7]),
    ("xgboost", 5, [42, 123, 7]),
    ("qsvc", 5, [42, 123, 7]),
    ("vqc", 3, [42]),
    ("qnn", 3, [42]),
]

for model_type, cv_folds, seeds in models_to_run:
    print(f"\nRunning {model_type.upper()} ({cv_folds} folds, seeds={seeds})...", end="", flush=True)
    resp = client.post("/experiments/run", json={
        "dataset_id": "breast_cancer_wdbc",
        "model_type": model_type,
        "n_features": 4,
        "backend": "ideal_simulator",
        "circuit_depth": 1,
        "cv_folds": cv_folds,
        "seeds": seeds,
    })
    if resp.status_code == 200:
        data = resp.json()
        m = data["metrics"]
        print(f" Done ({data['runtime_seconds']:.1f}s)")
        print(f"  Acc: {m['accuracy']['mean']:.4f} +/- {m['accuracy']['std']:.4f} | "
              f"F1: {m['f1']['mean']:.4f} +/- {m['f1']['std']:.4f} | "
              f"ROC-AUC: {m['roc_auc']['mean']:.4f} +/- {m['roc_auc']['std']:.4f}")
    else:
        print(f" ERROR: {resp.text}")

print("\n" + "=" * 110)
print("FETCHING BENCHMARK TABLE FROM DATABASE VIA GET /experiments")
print("=" * 110)

all_resp = client.get("/experiments?dataset_id=breast_cancer_wdbc")
exps = all_resp.json()["experiments"]

headers = ["MODEL TYPE", "ACCURACY", "PRECISION", "SENSITIVITY", "SPECIFICITY", "F1 SCORE", "ROC-AUC", "PR-AUC", "RUNTIME(s)"]
col_widths = [22, 18, 18, 18, 18, 18, 18, 18, 12]

header_str = "".join(h.center(w) for h, w in zip(headers, col_widths))
print(header_str)
print("-" * sum(col_widths))

# Group by model_type (latest run per model)
seen = set()
for exp in exps:
    mt = exp["model_type"]
    if mt in seen:
        continue
    seen.add(mt)
    m = exp["metrics"]
    row = [
        mt.upper(),
        f"{m['accuracy']['mean']:.4f} +/- {m['accuracy']['std']:.4f}",
        f"{m['precision']['mean']:.4f} +/- {m['precision']['std']:.4f}",
        f"{m['sensitivity']['mean']:.4f} +/- {m['sensitivity']['std']:.4f}",
        f"{m['specificity']['mean']:.4f} +/- {m['specificity']['std']:.4f}",
        f"{m['f1']['mean']:.4f} +/- {m['f1']['std']:.4f}",
        f"{m['roc_auc']['mean']:.4f} +/- {m['roc_auc']['std']:.4f}",
        f"{m['pr_auc']['mean']:.4f} +/- {m['pr_auc']['std']:.4f}",
        f"{exp['runtime_seconds']:.1f}",
    ]
    print("".join(str(v).center(w) for v, w in zip(row, col_widths)))

print("=" * sum(col_widths))
print("\n[SUCCESS] Phase 2 benchmark successfully logged to database.")
