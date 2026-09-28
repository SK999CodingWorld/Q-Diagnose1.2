from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from datasets import ensure_builtin_dataset, ensure_diabetes_dataset
from evaluate import DISCLAIMER
from experiment_runner import ExperimentSettings, run_experiment
from storage import get_dataset


def ensure_phase5_datasets(database: str | Path = "q_diagnose.db") -> tuple[int, int]:
    return ensure_builtin_dataset(database), ensure_diabetes_dataset(database)


def run_cross_dataset_comparison(
    settings: ExperimentSettings,
    database: str | Path = "q_diagnose.db",
    dataset_ids: Iterable[int] | None = None,
    model_types: Iterable[str] | None = None,
) -> dict:
    requested = tuple(int(value) for value in (dataset_ids or ensure_phase5_datasets(database)))
    selected_datasets: list[int] = []
    seen_sources: set[tuple[str, str]] = set()
    for dataset_id in requested:
        dataset = get_dataset(dataset_id, path=database)
        if dataset is None:
            raise KeyError(f"Dataset {dataset_id} was not found")
        identity = (str(dataset["name"]), str(dataset["source"]))
        if identity in seen_sources:
            continue
        seen_sources.add(identity)
        selected_datasets.append(dataset_id)
    if len(selected_datasets) < 2:
        raise ValueError(
            "Cross-dataset validation requires at least two distinct datasets; "
            f"received {len(selected_datasets)}"
        )
    selected_models = tuple(str(value) for value in (model_types or ("logistic_regression", "svm")))
    rows = []
    for dataset_id in selected_datasets:
        dataset = get_dataset(dataset_id, path=database)
        for model_type in selected_models:
            experiment = run_experiment(
                replace(settings, dataset_id=dataset_id, model_type=model_type),
                database=database,
            )
            rows.append(
                {
                    "experiment_id": experiment["id"],
                    "dataset_id": dataset_id,
                    "dataset_name": dataset["name"],
                    "model_type": model_type,
                    "backend": experiment["config"].get("backend"),
                    "metrics": experiment["metrics"],
                    "runtime_seconds": experiment["runtime_seconds"],
                }
            )
    return {
        "datasets": [
            {
                "id": dataset_id,
                "name": get_dataset(dataset_id, path=database)["name"],
                "source": get_dataset(dataset_id, path=database)["source"],
            }
            for dataset_id in selected_datasets
        ],
        "rows": rows,
        "disclaimer": DISCLAIMER,
    }
