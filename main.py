from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qiskit.circuit.library import real_amplitudes

from datasets import (
    MAX_UPLOAD_BYTES,
    UPLOAD_DIRECTORY,
    DatasetValidationError,
    ensure_builtin_dataset,
    ensure_diabetes_dataset,
    save_uploaded_csv,
)
from evaluate import DISCLAIMER, METRIC_NAMES
from explanations import explain_experiment
from experiment_runner import (
    PIPELINE_REVISION,
    ExperimentSettings,
    _calibration,
    run_experiment,
)
from phase5 import run_cross_dataset_comparison
from model_registry import (
    MODEL_TYPES,
    QUANTUM_MODEL_TYPES,
    SUPPORTED_BACKENDS,
    HardwareNotConfigured,
    QuantumBackendUnavailable,
    feature_map_circuit,
    circuit_text,
    real_hardware_status,
)
from storage import (
    get_dataset,
    get_experiment,
    get_experiment_folds,
    list_datasets,
    list_experiments,
    update_calibration,
)


DATABASE_PATH = Path("q_diagnose.db")
logger = logging.getLogger("q_diagnose.api")
app = FastAPI(
    title="Q-Diagnose API",
    version="2.0.0",
    description=(
        "Research and decision-support prototype for classical and quantum binary-classification experiments."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExperimentRequest(BaseModel):
    dataset_id: int
    model_type: str
    n_features: int = Field(default=4, ge=1, le=32)
    backend: str = "ideal_simulator"
    circuit_depth: int = Field(default=2, ge=1, le=12)
    cv_folds: int = Field(default=5, ge=3, le=20)
    seeds: list[int] = Field(default_factory=lambda: [42, 123, 2025], min_length=3, max_length=10)
    optimizer: str = "cobyla"
    max_iterations: int = Field(default=25, ge=10, le=500)
    shots: int = Field(default=512, ge=1, le=100000)
    noise_multiplier: float = Field(default=0.01, ge=0, le=0.25)
    split_seed: int = 42
    test_size: float = Field(default=0.2, gt=0, lt=1)

    def settings(self) -> ExperimentSettings:
        return ExperimentSettings(
            dataset_id=self.dataset_id,
            model_type=self.model_type,
            n_features=self.n_features,
            backend=self.backend,
            circuit_depth=self.circuit_depth,
            cv_folds=self.cv_folds,
            seeds=tuple(self.seeds),
            optimizer=self.optimizer,
            max_iterations=self.max_iterations,
            shots=self.shots,
            noise_multiplier=self.noise_multiplier,
            split_seed=self.split_seed,
            test_size=self.test_size,
        )


class SweepRequest(ExperimentRequest):
    variable: str
    values: list[float] = Field(min_length=1, max_length=20)


class Phase5Request(BaseModel):
    dataset_ids: list[int] | None = None
    model_types: list[str] | None = None
    n_features: int = Field(default=4, ge=1, le=32)
    backend: str = "ideal_simulator"
    circuit_depth: int = Field(default=2, ge=1, le=12)
    cv_folds: int = Field(default=5, ge=3, le=20)
    seeds: list[int] = Field(default_factory=lambda: [42, 123, 2025], min_length=3, max_length=10)
    optimizer: str = "cobyla"
    max_iterations: int = Field(default=25, ge=10, le=500)
    shots: int = Field(default=512, ge=1, le=100000)
    noise_multiplier: float = Field(default=0.01, ge=0, le=0.25)
    split_seed: int = 42
    test_size: float = Field(default=0.2, gt=0, lt=1)

    def settings(self) -> ExperimentSettings:
        model_types = self.model_types or ["logistic_regression"]
        dataset_ids = self.dataset_ids or [1]
        return ExperimentSettings(
            dataset_id=int(dataset_ids[0]),
            model_type=model_types[0],
            n_features=self.n_features,
            backend=self.backend,
            circuit_depth=self.circuit_depth,
            cv_folds=self.cv_folds,
            seeds=tuple(self.seeds),
            optimizer=self.optimizer,
            max_iterations=self.max_iterations,
            shots=self.shots,
            noise_multiplier=self.noise_multiplier,
            split_seed=self.split_seed,
            test_size=self.test_size,
        )


def _provenance(experiment: dict[str, Any]) -> dict[str, Any]:
    config = experiment.get("config") or {}
    revision = config.get("pipeline_revision")
    return {
        "pipeline_revision": revision,
        "current_pipeline_revision": PIPELINE_REVISION,
        "superseded": revision != PIPELINE_REVISION,
    }


def _experiment_summary(experiment: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: value
        for key, value in experiment.items()
        if key not in {"test_data", "fold_results"}
    }
    summary.update(_provenance(experiment))
    return summary


def _raise_backend_error(error: HardwareNotConfigured) -> HTTPException:
    return HTTPException(status_code=503, detail=str(error))


def _run_settings(settings: ExperimentSettings) -> dict[str, Any]:
    if settings.backend == "real_hardware":
        status = real_hardware_status()
        if not status["execution_available"]:
            raise HTTPException(status_code=503, detail=status["message"])
    try:
        return run_experiment(settings, database=DATABASE_PATH)
    except HardwareNotConfigured as error:
        raise _raise_backend_error(error) from error
    except QuantumBackendUnavailable as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except DatasetValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (KeyError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/")
def root() -> Any:
    if Path("frontend/dist/index.html").exists():
        return FileResponse("frontend/dist/index.html")
    return {
        "application": "Q-Diagnose",
        "version": "2.0.0",
        "disclaimer": DISCLAIMER,
        "api": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    datasets = list_datasets(DATABASE_PATH)
    experiments = list_experiments(path=DATABASE_PATH)
    return {
        "status": "ok",
        "disclaimer": DISCLAIMER,
        "datasets": len(datasets),
        "experiments": len(experiments),
        "hardware": real_hardware_status(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {
        "model_types": list(MODEL_TYPES),
        "quantum_model_types": sorted(QUANTUM_MODEL_TYPES),
        "backends": list(SUPPORTED_BACKENDS),
        "metric_names": list(METRIC_NAMES),
    }


@app.get("/api/backends")
def backends() -> dict[str, Any]:
    return {
        "backends": [
            {
                "id": "ideal_simulator",
                "name": "Ideal simulator",
                "description": "Deterministic Qiskit Aer statevector or sampler execution.",
            },
            {
                "id": "noisy_simulator",
                "name": "Noisy simulator",
                "description": "Aer execution with a configured depolarizing noise model.",
            },
            {
                "id": "real_hardware",
                "name": "IBM Quantum hardware",
                "description": real_hardware_status()["message"],
            },
        ],
        "hardware": real_hardware_status(),
    }


@app.get("/api/datasets")
def datasets() -> list[dict[str, Any]]:
    return list_datasets(DATABASE_PATH)


@app.post("/api/datasets/builtin")
def create_builtin_dataset() -> dict[str, Any]:
    try:
        dataset_id = ensure_builtin_dataset(DATABASE_PATH)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    dataset = get_dataset(dataset_id, path=DATABASE_PATH)
    if dataset is None:
        raise HTTPException(status_code=500, detail="Built-in dataset was not persisted")
    return dataset


@app.post("/api/datasets/builtin/diabetes")
def create_diabetes_dataset() -> dict[str, Any]:
    try:
        dataset_id = ensure_diabetes_dataset(DATABASE_PATH)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    dataset = get_dataset(dataset_id, path=DATABASE_PATH)
    if dataset is None:
        raise HTTPException(status_code=500, detail="Diabetes dataset was not persisted")
    return dataset


@app.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
) -> dict[str, Any]:
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        registration = save_uploaded_csv(
            content,
            file.filename or "uploaded_dataset.csv",
            target_column=target_column,
            database=DATABASE_PATH,
        )
        dataset = get_dataset(registration["dataset_id"], path=DATABASE_PATH)
        if dataset is None:
            raise DatasetValidationError("Uploaded dataset was not persisted")
        return {**dataset, "registration": registration}
    except DatasetValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except OSError as error:
        logger.exception("Dataset upload failed while writing to %s", UPLOAD_DIRECTORY)
        raise HTTPException(
            status_code=500,
            detail="The dataset could not be stored on the server",
        ) from error


@app.get("/api/datasets/{dataset_id}")
def dataset_detail(dataset_id: int) -> dict[str, Any]:
    dataset = get_dataset(dataset_id, path=DATABASE_PATH)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset["experiments"] = [
        _experiment_summary(item)
        for item in list_experiments(dataset_id=dataset_id, path=DATABASE_PATH)
    ]
    return dataset


@app.post("/api/experiments")
def create_experiment_endpoint(request: ExperimentRequest) -> dict[str, Any]:
    experiment = _run_settings(request.settings())
    return {**experiment, **_provenance(experiment)}


@app.get("/api/experiments")
def experiments() -> list[dict[str, Any]]:
    return [_experiment_summary(item) for item in list_experiments(path=DATABASE_PATH)]


@app.get("/api/experiments/{experiment_id}")
def experiment_detail(experiment_id: int) -> dict[str, Any]:
    experiment = get_experiment(experiment_id, path=DATABASE_PATH)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {**experiment, **_provenance(experiment)}


@app.get("/api/experiments/{experiment_id}/progress")
def experiment_progress(experiment_id: int) -> dict[str, Any]:
    experiment = get_experiment(experiment_id, path=DATABASE_PATH)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {
        "experiment_id": experiment_id,
        "status": experiment["status"],
        "progress": experiment["progress"],
        "error_message": experiment.get("error_message"),
    }


@app.get("/api/experiments/{experiment_id}/folds")
def experiment_folds(experiment_id: int) -> list[dict[str, Any]]:
    if get_experiment(experiment_id, path=DATABASE_PATH) is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return get_experiment_folds(experiment_id, path=DATABASE_PATH)


@app.get("/api/experiments/{experiment_id}/explanation")
def explanation(
    experiment_id: int,
    force: bool = False,
    sample_limit: int = 20,
) -> dict[str, Any]:
    if get_experiment(experiment_id, path=DATABASE_PATH) is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    try:
        return explain_experiment(
            experiment_id,
            database=DATABASE_PATH,
            force=force,
            sample_limit=sample_limit,
        )
    except DatasetValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (KeyError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/experiments/{experiment_id}/calibration")
def calibration(experiment_id: int) -> dict[str, Any]:
    experiment = get_experiment(experiment_id, path=DATABASE_PATH)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment.get("calibration") or {}


@app.post("/api/experiments/{experiment_id}/calibrate")
def calibrate_experiment(experiment_id: int) -> dict[str, Any]:
    experiment = get_experiment(experiment_id, path=DATABASE_PATH)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    test_data = experiment.get("test_data") or {}
    if "y_test" not in test_data or "probabilities" not in test_data:
        raise HTTPException(status_code=422, detail="Experiment has no holdout probabilities")
    values = _calibration(
        np.asarray(test_data["y_test"]),
        np.asarray(test_data["probabilities"]),
    )
    update_calibration(experiment_id, values, path=DATABASE_PATH)
    return values


@app.get("/api/experiments/{experiment_id}/circuit")
def experiment_circuit(experiment_id: int) -> dict[str, Any]:
    experiment = get_experiment(experiment_id, path=DATABASE_PATH)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    config = experiment.get("config", {})
    model_type = str(experiment["model_type"])
    if model_type not in QUANTUM_MODEL_TYPES:
        return {
            "model_type": model_type,
            "quantum": False,
            "message": "No quantum circuit is associated with this classical model.",
        }
    n_features = int((config.get("preprocessing") or {}).get("n_components", 4))
    circuit_depth = int(config.get("circuit_depth", 2))
    model_config = config.get("model") or {}
    feature_map_reps = int((model_config.get("feature_map") or {}).get("reps", 2))
    feature_map = feature_map_circuit(n_features, reps=feature_map_reps)
    response: dict[str, Any] = {
        "model_type": model_type,
        "quantum": True,
        "qubits": n_features,
        "feature_map": circuit_text(feature_map),
    }
    if model_type == "qsvc":
        response["ansatz"] = None
        response["combined"] = circuit_text(feature_map)
        response["message"] = (
            "QSVC has no trainable ansatz; only the fixed feature map is used as a kernel."
        )
        return response
    ansatz = real_amplitudes(n_features, reps=circuit_depth)
    circuit = feature_map.compose(ansatz)
    response["ansatz"] = circuit_text(ansatz)
    response["combined"] = circuit_text(circuit)
    return response


def _sweep_variable_applies(model_type: str, backend: str, variable: str) -> bool:
    quantum = model_type in QUANTUM_MODEL_TYPES
    if variable == "n_features":
        return True
    if variable == "circuit_depth":
        return model_type in {"vqc", "qnn"}
    if variable == "shots":
        return model_type in {"qsvc", "vqc"}
    if variable == "noise_multiplier":
        return quantum and backend == "noisy_simulator"
    if variable == "max_iterations":
        return model_type in {"vqc", "qnn"}
    return False


def _sweep_variable_reason(model_type: str, backend: str, variable: str) -> str:
    reasons = {
        "circuit_depth": (
            "only vqc and qnn have a variational ansatz whose depth is configurable; "
            "the qsvc feature map uses a fixed 2 repetitions"
        ),
        "shots": (
            "only qsvc and vqc sample circuits; qnn uses exact expectation values and "
            "classical models are not sampled"
        ),
        "noise_multiplier": (
            "noise applies only to quantum models on the noisy_simulator backend"
        ),
        "max_iterations": (
            "only vqc and qnn train with a classical optimizer over circuit parameters"
        ),
        "n_features": "no reason",
    }
    return reasons.get(variable, "the variable has no effect on this model")


@app.post("/api/sweeps")
def create_sweep(request: SweepRequest) -> dict[str, Any]:
    numeric_variables = {
        "n_features",
        "circuit_depth",
        "max_iterations",
        "shots",
        "noise_multiplier",
    }
    if request.variable not in numeric_variables:
        raise HTTPException(
            status_code=422,
            detail=f"Sweep variable must be one of {sorted(numeric_variables)}",
        )
    base = request.settings()
    if base.backend == "real_hardware":
        status = real_hardware_status()
        if not status["execution_available"]:
            raise HTTPException(status_code=503, detail=status["message"])
    if not _sweep_variable_applies(base.model_type, base.backend, request.variable):
        effective = sorted(
            variable
            for variable in numeric_variables
            if _sweep_variable_applies(base.model_type, base.backend, variable)
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"Cannot sweep {request.variable} for model_type={base.model_type} and "
                f"backend={base.backend}: "
                f"{_sweep_variable_reason(base.model_type, base.backend, request.variable)}. "
                f"Variables that do affect this configuration: {effective}"
            ),
        )
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for raw_value in request.values:
        value = float(raw_value)
        if request.variable == "n_features":
            value = int(value)
        elif request.variable in {"circuit_depth", "max_iterations", "shots"}:
            value = int(value)
        settings = replace(base, **{request.variable: value})
        try:
            experiment = run_experiment(
                settings,
                database=DATABASE_PATH,
                sweep_variable=request.variable,
                sweep_value=float(value),
            )
            results.append(
                {
                    "sweep_value": float(value),
                    "experiment_id": experiment["id"],
                    "status": experiment["status"],
                    "metrics": experiment["metrics"],
                }
            )
        except QuantumBackendUnavailable as error:
            errors.append({"sweep_value": float(value), "error": str(error)})
        except (ValueError, KeyError, DatasetValidationError, FileNotFoundError) as error:
            errors.append({"sweep_value": float(value), "error": str(error)})
    return {
        "variable": request.variable,
        "values": request.values,
        "results": results,
        "errors": errors,
    }


@app.post("/api/phase5/validate")
def phase5_validate(request: Phase5Request) -> dict[str, Any]:
    try:
        return run_cross_dataset_comparison(
            request.settings(),
            database=DATABASE_PATH,
            dataset_ids=request.dataset_ids,
            model_types=request.model_types,
        )
    except HardwareNotConfigured as error:
        raise _raise_backend_error(error) from error
    except DatasetValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (KeyError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/comparisons")
def comparisons(
    dataset_ids: str | None = None,
    model_types: str | None = None,
) -> dict[str, Any]:
    selected_datasets = {
        int(value.strip())
        for value in (dataset_ids or "").split(",")
        if value.strip()
    } or None
    selected_models = {
        value.strip()
        for value in (model_types or "").split(",")
        if value.strip()
    } or None
    rows: list[dict[str, Any]] = []
    for experiment in list_experiments(path=DATABASE_PATH):
        if selected_datasets and experiment["dataset_id"] not in selected_datasets:
            continue
        if selected_models and experiment["model_type"] not in selected_models:
            continue
        if experiment["status"] != "completed":
            continue
        rows.append(
            {
                "experiment_id": experiment["id"],
                "dataset_id": experiment["dataset_id"],
                "model_type": experiment["model_type"],
                "backend": (experiment.get("config") or {}).get("backend"),
                "metrics": experiment.get("metrics"),
                "runtime_seconds": experiment["runtime_seconds"],
            }
        )
    return {"rows": rows, "disclaimer": DISCLAIMER}


@app.exception_handler(HardwareNotConfigured)
def hardware_exception_handler(request: Any, error: HardwareNotConfigured) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


frontend_dist = Path("frontend/dist")
_assets_dir = frontend_dist / "assets"

if _assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="frontend-assets")


def _missing_frontend_build(name: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                f"'{name}' needs the built frontend, which is not present at {frontend_dist}. "
                "Build it with: cd frontend && npm install && npm run build"
            ),
            "expected_path": str(frontend_dist),
        },
    )


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> Any:
    index = frontend_dist / "index.html"
    if not index.is_file():
        return _missing_frontend_build("dashboard")
    return FileResponse(index)


@app.get("/landing", include_in_schema=False)
def landing() -> Any:
    landing_page = frontend_dist / "landing.html"
    if not landing_page.is_file():
        return _missing_frontend_build("landing")
    return FileResponse(landing_page)
