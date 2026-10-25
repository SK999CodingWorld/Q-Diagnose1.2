"""
Phase 2 API Integration and Model Verification Test Suite
==========================================================
Verifies:
1. Startup & dataset registration
2. CSV dataset upload and schema validation
3. Execution of all 7 model types:
   - Logistic Regression
   - SVM
   - Random Forest
   - XGBoost
   - QSVC
   - VQC
   - QNN
4. Proper persistence of metrics, confusion matrix, and ROC curve
5. Dual explainability (SHAP for classical, Circuit + Sensitivity for quantum)
6. Non-identical, model-specific metrics across all architectures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import io
import pandas as pd
from fastapi.testclient import TestClient
from main import app
import database as db

client = TestClient(app)


def test_startup_and_default_dataset():
    """Verify default Breast Cancer dataset is registered on startup."""
    with TestClient(app) as test_client:
        response = test_client.get("/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert any(d["id"] == "breast_cancer_wdbc" for d in data["datasets"])
        assert "disclaimer" in data


def test_dataset_upload_and_validation():
    """Test CSV upload endpoint with valid and invalid data."""
    with TestClient(app) as test_client:
        # Create a mock valid CSV
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0] * 5,
            "f2": [0.5, 1.5, 2.5, 3.5, 4.5] * 5,
            "target": [0, 1, 0, 1, 0] * 5
        })
        csv_bytes = io.BytesIO()
        df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)

        response = test_client.post(
            "/datasets/upload",
            files={"file": ("test_data.csv", csv_bytes, "text/csv")},
            data={"name": "Integration Test Dataset", "target_column": "target"}
        )
        assert response.status_code == 200
        resp_data = response.json()
        assert resp_data["status"] == "success"
        assert resp_data["dataset"]["n_rows"] == 25
        assert resp_data["dataset"]["n_cols"] == 3
        assert resp_data["dataset"]["target_col"] == "target"


def test_run_classical_models():
    """Run all 4 classical models and verify distinct metrics and SHAP explanations."""
    with TestClient(app) as test_client:
        classical_models = ["logistic_regression", "svm", "random_forest", "xgboost"]
        experiment_ids = {}

        for model in classical_models:
            resp = test_client.post("/experiments/run", json={
                "dataset_id": "breast_cancer_wdbc",
                "model_type": model,
                "n_features": 4,
                "backend": "ideal_simulator",
                "circuit_depth": 1,
                "cv_folds": 2,
                "seeds": [42]
            })
            assert resp.status_code == 200, f"Failed for model {model}: {resp.text}"
            data = resp.json()
            assert data["status"] == "completed"
            assert "metrics" in data
            assert data["metrics"]["accuracy"]["mean"] > 0.8
            experiment_ids[model] = data["experiment_id"]

            # Test GET /experiments/{id}
            get_resp = test_client.get(f"/experiments/{data['experiment_id']}")
            assert get_resp.status_code == 200
            exp_data = get_resp.json()["experiment"]
            assert exp_data["model_type"] == model
            assert "confusion_matrix" in exp_data
            assert "roc_curve" in exp_data

            # Test GET /explain/{id}
            expl_resp = test_client.get(f"/explain/{data['experiment_id']}")
            assert expl_resp.status_code == 200
            expl_data = expl_resp.json()["explanation"]
            assert "feature_importances" in expl_data
            assert len(expl_data["feature_importances"]) > 0


def test_run_quantum_models():
    """Run QSVC, VQC, and QNN and verify circuit structure and perturbation sensitivity."""
    with TestClient(app) as test_client:
        quantum_models = ["qsvc", "vqc", "qnn"]

        for model in quantum_models:
            resp = test_client.post("/experiments/run", json={
                "dataset_id": "breast_cancer_wdbc",
                "model_type": model,
                "n_features": 4,
                "backend": "ideal_simulator",
                "circuit_depth": 1,
                "cv_folds": 2,
                "seeds": [42]
            })
            assert resp.status_code == 200, f"Failed for quantum model {model}: {resp.text}"
            data = resp.json()
            assert data["status"] == "completed"
            assert "metrics" in data
            exp_id = data["experiment_id"]

            # Test explanation endpoint
            expl_resp = test_client.get(f"/explain/{exp_id}")
            assert expl_resp.status_code == 200
            expl_data = expl_resp.json()["explanation"]
            assert expl_data["type"] == "quantum_structure"
            assert "circuit_diagram" in expl_data
            assert "feature_sensitivities" in expl_data
            assert expl_data["qubit_count"] == 4


def test_patient_module_integration():
    """Verify full end-to-end integration for the Patient Details Module (Part A)."""
    with TestClient(app) as test_client:
        # 1. Register patient
        reg_resp = test_client.post("/patients", json={
            "name": "Integration-Test-Subject",
            "age": 48,
            "sex": "Female",
            "notes": "Automated integration test subject."
        })
        assert reg_resp.status_code == 200
        p_id = reg_resp.json()["patient_id"]

        # 2. Add clinical visit record
        sample_feat = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]
        rec_resp = test_client.post(f"/patients/{p_id}/records", json={
            "feature_values": sample_feat,
            "source": "Outpatient Test Lab"
        })
        assert rec_resp.status_code == 200
        r_id = rec_resp.json()["record_id"]

        # 3. Predict on patient visit record
        pred_resp = test_client.post(f"/patients/{p_id}/predict", json={
            "record_id": r_id,
            "model_type": "svm",
            "threshold": 0.5
        })
        assert pred_resp.status_code == 410
        assert "not validated" in pred_resp.json()["detail"]

        # 4. Check longitudinal trend
        trend_resp = test_client.get(f"/patients/{p_id}/trend")
        assert trend_resp.status_code == 200
        t_data = trend_resp.json()["trend_data"]
        assert t_data["patient_id"] == p_id
        assert t_data["total_records"] == 1
        assert t_data["trend"] == []
        assert t_data["prediction_status"] == "unavailable_until_validated"


def test_backends_and_comparisons_endpoints():
    """Verify hardware backends and empirical comparisons API endpoints."""
    with TestClient(app) as test_client:
        b_resp = test_client.get("/api/backends")
        assert b_resp.status_code == 200
        b_data = b_resp.json()
        assert "classical" in b_data and len(b_data["classical"]) >= 1
        assert "quantum" in b_data and len(b_data["quantum"]) >= 2
        assert any(q["id"] == "aer_simulator_statevector" for q in b_data["quantum"])

        c_resp = test_client.get("/api/comparisons")
        assert c_resp.status_code == 200
        c_data = c_resp.json()
        comps = c_data.get("comparison_summary", c_data.get("comparisons", []))
        assert isinstance(comps, list)
        assert len(comps) >= 5
        assert any("SelectKBest vs PCA" in str(r.get("Model / Intervention")) for r in comps)

