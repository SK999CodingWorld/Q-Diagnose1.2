from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from evaluate import positive_probability
from experiment_runner import PIPELINE_REVISION, fit_final_for_experiment
from model_registry import feature_map_circuit, qnn_probability_from_score
from storage import get_latest_explanation, save_explanation


def _with_provenance(explanation: dict[str, Any], pipeline_revision: str | None) -> dict[str, Any]:
    return {
        **explanation,
        "pipeline_revision": pipeline_revision,
        "current_pipeline_revision": PIPELINE_REVISION,
        "superseded": pipeline_revision != PIPELINE_REVISION,
    }


def _positive_scores(model: Any, model_type: str, values: np.ndarray) -> np.ndarray:
    if model_type != "qnn":
        return positive_probability(model, values)
    raw = np.asarray(model.predict_proba(values))
    if raw.ndim == 1:
        return qnn_probability_from_score(raw)
    if raw.shape[1] == 1:
        return qnn_probability_from_score(raw[:, 0])
    return positive_probability(model, values)


def _feature_ranking(values: np.ndarray, feature_names: list[str], limit: int) -> list[dict[str, Any]]:
    mean_absolute = np.mean(np.abs(values), axis=0)
    order = np.argsort(mean_absolute)[::-1][:limit]
    return [
        {
            "rank": int(rank + 1),
            "feature": feature_names[int(index)],
            "mean_absolute_contribution": float(mean_absolute[index]),
        }
        for rank, index in enumerate(order)
    ]


def _shap_base_value(base_values: Any, row_count: int) -> float | None:
    if base_values is None:
        return None
    array = np.asarray(base_values, dtype=float)
    if array.size == 0:
        return None
    if array.ndim >= 2:
        return float(np.mean(array[:, -1]))
    if array.shape[0] == row_count:
        return float(np.mean(array))
    return float(array.reshape(-1)[-1])


def _shap_values(
    model: Any,
    background: np.ndarray,
    values: np.ndarray,
    background_targets: np.ndarray,
) -> tuple[np.ndarray, str, float | None]:
    try:
        import shap

        explainer = shap.Explainer(model.predict_proba, background)
        explanation = explainer(values)
        raw_values = np.asarray(explanation.values)
        if raw_values.ndim == 3:
            raw_values = raw_values[:, :, -1]
        elif raw_values.ndim != 2:
            raise ValueError("SHAP returned an unsupported value shape")
        base_value = _shap_base_value(explanation.base_values, len(values))
        return raw_values, "shap.Explainer.predict_proba", base_value
    except Exception:
        from sklearn.inspection import permutation_importance

        scoring = "roc_auc" if len(np.unique(background_targets)) == 2 else "accuracy"
        scorer = permutation_importance(
            model,
            background,
            background_targets,
            n_repeats=3,
            random_state=0,
            scoring=scoring,
        )
        base_value = float(np.mean(positive_probability(model, background)))
        return np.tile(scorer.importances_mean, (values.shape[0], 1)), (
            "permutation_importance_fallback"
        ), base_value


def _quantum_sensitivity(
    model: Any,
    values: np.ndarray,
    feature_names: list[str],
    epsilon: float,
) -> dict[str, Any]:
    base = _positive_scores(model, "qnn", values)
    feature_scores: list[dict[str, Any]] = []
    for index, feature_name in enumerate(feature_names):
        plus = values.copy()
        minus = values.copy()
        plus[:, index] += epsilon
        minus[:, index] -= epsilon
        plus_score = _positive_scores(model, "qnn", plus)
        minus_score = _positive_scores(model, "qnn", minus)
        feature_scores.append(
            {
                "feature": feature_name,
                "epsilon": float(epsilon),
                "mean_absolute_probability_change": float(
                    np.mean(np.abs(plus_score - minus_score))
                ),
                "base_probability": float(np.mean(base)),
            }
        )
    feature_scores.sort(
        key=lambda item: item["mean_absolute_probability_change"],
        reverse=True,
    )
    for rank, item in enumerate(feature_scores, start=1):
        item["rank"] = rank
    circuit = feature_map_circuit(len(feature_names))
    return {
        "method": "parameter_perturbation_on_positive_probability",
        "epsilon": float(epsilon),
        "base_probability": float(np.mean(base)),
        "feature_scores": feature_scores,
        "feature_map": {
            "name": "ZZFeatureMap",
            "qubits": len(feature_names),
            "depth": int(circuit.depth()),
        },
    }


def explain_experiment(
    experiment_id: int,
    database: str | Path = "q_diagnose.db",
    force: bool = False,
    sample_limit: int = 20,
) -> dict[str, Any]:
    from storage import get_experiment

    experiment = get_experiment(experiment_id, path=database)
    if experiment is None:
        raise KeyError(f"Experiment {experiment_id} was not found")
    if not force:
        stored = get_latest_explanation(experiment_id, path=database)
        if stored is not None:
            cached = stored["values"]
            if not isinstance(cached, dict):
                return cached
            return _with_provenance(cached, stored.get("pipeline_revision"))
    model, preprocessor, transformed_train, transformed_test, y_train, _ = fit_final_for_experiment(
        experiment,
        database=database,
    )
    model_type = str(experiment["model_type"])
    feature_names = list(preprocessor.feature_names)
    sample_limit = max(1, min(int(sample_limit), len(transformed_test)))
    values = transformed_test[:sample_limit]
    if model_type == "qnn":
        epsilon = float((experiment.get("test_data") or {}).get("epsilon", 0.1))
        explanation = {
            "experiment_id": int(experiment_id),
            "model_type": model_type,
            "sample_count": int(sample_limit),
            "feature_names": feature_names,
            "quantum": _quantum_sensitivity(model, values, feature_names, epsilon),
        }
    else:
        background = transformed_train[: min(50, len(transformed_train))]
        contributions, method, base_value = _shap_values(
            model,
            background,
            values,
            y_train[: len(background)],
        )
        explanation = {
            "experiment_id": int(experiment_id),
            "model_type": model_type,
            "sample_count": int(sample_limit),
            "feature_names": feature_names,
            "method": method,
            "base_values": base_value,
            "top_features": _feature_ranking(contributions, feature_names, len(feature_names)),
            "sample_contributions": contributions.tolist(),
            "quantum": None,
        }
    save_explanation(
        experiment_id,
        "feature_attribution",
        explanation,
        path=database,
        pipeline_revision=PIPELINE_REVISION,
    )
    return _with_provenance(explanation, PIPELINE_REVISION)
