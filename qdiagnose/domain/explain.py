"""
Q-Diagnose Explainability Module
=================================
Dual explainability engine:
1. Classical models: SHAP values (Shapley Additive exPlanations) on the test set.
2. Quantum models: Quantum circuit diagram (text/QASM), qubit count, circuit depth,
   and perturbation-based feature sensitivity scores (dP/dx_i).
"""

import numpy as np
from typing import Dict, Any, List
import shap
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes


def explain_classical_model(model, X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
    """
    Compute SHAP explanations for classical models.
    """
    n_features = X_train.shape[1]
    if not feature_names or len(feature_names) != n_features:
        feature_names = [f"PC_{i+1}" for i in range(n_features)]

    # Subsample background for fast explainer creation
    background = X_train[:50]
    eval_set = X_test[:30]

    try:
        # Check if model has a tree-based structure (RandomForest, XGBoost)
        if hasattr(model, "estimators_") or "XGB" in type(model).__name__:
            explainer = shap.TreeExplainer(model, background)
            shap_values = explainer.shap_values(eval_set)
        elif hasattr(model, "coef_"):
            # Linear model (LogisticRegression)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(eval_set)
        else:
            # Kernel explainer for SVM / other
            def predict_fn(x):
                try:
                    if hasattr(model, "predict_proba"):
                        p = model.predict_proba(x)
                        if p.ndim == 2 and p.shape[1] > 1:
                            return np.nan_to_num(p[:, 1], nan=0.5, posinf=1.0, neginf=0.0)
                        return np.nan_to_num(p.flatten(), nan=0.5, posinf=1.0, neginf=0.0)
                    from scipy.special import expit
                    return np.nan_to_num(expit(model.decision_function(x)), nan=0.5, posinf=1.0, neginf=0.0)
                except Exception:
                    return np.full(len(x), 0.5)

            explainer = shap.KernelExplainer(predict_fn, background[:20])
            shap_values = explainer.shap_values(eval_set[:10], nsamples=30)

        # Normalize shap_values array
        if isinstance(shap_values, list):
            vals = np.array(shap_values[-1])  # Class 1
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            vals = shap_values[:, :, 1]
        else:
            vals = np.array(shap_values)

        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        mean_abs_shap = np.nan_to_num(mean_abs_shap, nan=0.0, posinf=0.0, neginf=0.0)

        feature_importances = []
        for name, imp in zip(feature_names, mean_abs_shap):
            feature_importances.append({
                "feature": str(name),
                "importance": float(imp),
            })

        feature_importances.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "type": "classical_shap",
            "method": "SHAP (Shapley Additive exPlanations)",
            "feature_importances": feature_importances,
            "sample_shap_values": vals[:5].tolist(),
            "evaluated_samples": len(eval_set),
            "disclaimer": "Classical SHAP values quantify average marginal contribution of each principal component."
        }

    except Exception as e:
        # Fallback to permutation-like importance if explainer encounters format issues
        print(f"SHAP explanation fallback: {e}")
        return {
            "type": "classical_shap_fallback",
            "method": "Feature Variance/Weight Proxy",
            "feature_importances": [
                {"feature": f"PC_{i+1}", "importance": float(np.var(X_train[:, i]))}
                for i in range(n_features)
            ],
            "error_detail": str(e)
        }


def explain_quantum_model(model, X_test: np.ndarray, n_features: int = 4, circuit_depth: int = 1, model_type: str = "qsvc") -> Dict[str, Any]:
    """
    Generate quantum model explainability:
    1. Circuit diagram representation (text/ASCII).
    2. Circuit metadata (qubit count, circuit depth, total gates).
    3. Perturbation-based feature sensitivity score:
       Delta_P = |P(x + eps) - P(x - eps)| / (2 * eps)
    """
    eps = 0.05
    n_samples = min(20, len(X_test))
    eval_set = np.array(X_test[:n_samples], dtype=float, copy=True)

    # 1. Build circuit representation
    fm = ZZFeatureMap(feature_dimension=n_features, reps=2 if model_type == "qsvc" else 1, entanglement="linear")
    if model_type in ["vqc", "qnn"]:
        ansatz = RealAmplitudes(num_qubits=n_features, reps=circuit_depth, entanglement="linear")
        full_circuit = fm.compose(ansatz)
    else:
        full_circuit = fm

    circuit_diagram_str = str(full_circuit.draw(output="text"))
    gate_counts = {str(k.name if hasattr(k, "name") else k): int(v) for k, v in full_circuit.count_ops().items()}

    # 2. Perturbation-based sensitivity
    sensitivities = []
    for j in range(n_features):
        X_plus = eval_set.copy()
        X_minus = eval_set.copy()
        X_plus[:, j] += eps
        X_minus[:, j] -= eps

        try:
            if hasattr(model, "predict_proba"):
                proba_plus = model.predict_proba(X_plus)
                proba_minus = model.predict_proba(X_minus)
                p_plus = proba_plus[:, 1] if (proba_plus.ndim == 2 and proba_plus.shape[1] > 1) else proba_plus.flatten()
                p_minus = proba_minus[:, 1] if (proba_minus.ndim == 2 and proba_minus.shape[1] > 1) else proba_minus.flatten()
            elif hasattr(model, "decision_function"):
                from scipy.special import expit
                p_plus = expit(model.decision_function(X_plus))
                p_minus = expit(model.decision_function(X_minus))
            else:
                p_plus = model.predict(X_plus)
                p_minus = model.predict(X_minus)

            diff = np.abs(p_plus - p_minus) / (2.0 * eps)
            avg_sensitivity = float(np.mean(diff))
        except Exception:
            avg_sensitivity = 0.0

        sensitivities.append({
            "feature": f"Qubit_{j+1} (PC_{j+1})",
            "sensitivity": avg_sensitivity,
        })

    sensitivities.sort(key=lambda x: x["sensitivity"], reverse=True)

    return {
        "type": "quantum_structure",
        "method": "Quantum Circuit Decomposition & Perturbation Sensitivity",
        "qubit_count": n_features,
        "circuit_depth": int(full_circuit.depth()),
        "total_gates": sum(gate_counts.values()),
        "gate_breakdown": gate_counts,
        "circuit_diagram": circuit_diagram_str,
        "feature_sensitivities": sensitivities,
        "disclaimer": "Note: Quantum circuit visualizations provide algorithm structure but do not constitute clinical explanations."
    }
