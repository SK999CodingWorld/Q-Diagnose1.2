from __future__ import annotations

import importlib.util
import os
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, QSVC, VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from evaluate import AerStatevectorKernel, build_zz_feature_map


MODEL_TYPES = (
    "logistic_regression",
    "svm",
    "random_forest",
    "xgboost",
    "qsvc",
    "vqc",
    "qnn",
)
QUANTUM_MODEL_TYPES = {"qsvc", "vqc", "qnn"}
MAX_QUANTUM_FEATURES = 8
SUPPORTED_BACKENDS = {"ideal_simulator", "noisy_simulator", "real_hardware"}


class HardwareNotConfigured(RuntimeError):
    pass


class QuantumBackendUnavailable(RuntimeError):
    pass


LEGACY_MODEL_TYPE_ALIASES = {"classical_svm": "svm"}


def normalize_model_type(value: Any) -> str:
    """Map a stored model type onto the current vocabulary when replaying a run.

    Phase 1 wrote ``classical_svm`` with an RBF SVC (C=1.0, gamma="scale",
    probability=True) that is hyperparameter-for-hyperparameter identical to the
    current ``svm`` entry, so the alias replays faithfully. Unknown types are
    rejected rather than guessed at.
    """
    text = str(value)
    if text in MODEL_TYPES:
        return text
    if text in LEGACY_MODEL_TYPE_ALIASES:
        return LEGACY_MODEL_TYPE_ALIASES[text]
    raise ValueError(f"Unsupported model type: {text}")


def normalize_backend(value: Any) -> str:
    """Map a stored ``config.backend`` onto the current backend vocabulary.

    Phase 1 stored ``"classical"`` for classical models and a nested descriptor
    dict for QSVC (``{"name": "qiskit_aer", "simulator": "statevector",
    "noiseless": true, ...}``). Stringifying either produced a value that no
    backend check could accept, which made those rows impossible to replay. The
    modern pipeline already records ``ideal_simulator`` for classical models, so
    the legacy ``"classical"`` marker maps to the same value rather than
    inventing a new one.
    """
    if value is None:
        return "ideal_simulator"
    if isinstance(value, str):
        if value in SUPPORTED_BACKENDS:
            return value
        if value == "classical":
            return "ideal_simulator"
        raise ValueError(f"Unsupported backend: {value}")
    if isinstance(value, dict):
        noiseless = value.get("noiseless")
        if noiseless is None:
            return "ideal_simulator"
        return "ideal_simulator" if bool(noiseless) else "noisy_simulator"
    raise ValueError(f"Unsupported backend: {value!r}")


def real_hardware_status() -> dict[str, Any]:
    token = os.getenv("QISKIT_IBM_TOKEN") or os.getenv("IBM_QUANTUM_TOKEN")
    runtime_available = importlib.util.find_spec("qiskit_ibm_runtime") is not None
    configured = bool(token) and runtime_available
    return {
        "configured": configured,
        "token_present": bool(token),
        "runtime_package_present": runtime_available,
        "execution_available": False,
        "message": (
            "Real hardware executor is not available; simulator fallback is disabled"
            if configured
            else "Real hardware not configured"
        ),
    }


def require_hardware() -> None:
    status = real_hardware_status()
    if not status["configured"]:
        raise HardwareNotConfigured("Real hardware not configured")
    raise HardwareNotConfigured(
        "Real hardware execution is not implemented; simulator fallback is disabled"
    )


def build_noise_model(noise_multiplier: float = 0.01) -> NoiseModel:
    level = max(0.0, min(0.25, float(noise_multiplier)))
    if level == 0:
        return NoiseModel()
    one_qubit_error = depolarizing_error(level, 1)
    two_qubit_error = depolarizing_error(min(0.5, level * 1.5), 2)
    noise_model = NoiseModel()
    for gate in ("h", "x", "y", "z", "rx", "ry", "rz", "p", "u", "u1", "u2", "u3"):
        noise_model.add_all_qubit_quantum_error(one_qubit_error, gate)
    for gate in ("cx", "cz", "swap"):
        noise_model.add_all_qubit_quantum_error(two_qubit_error, gate)
    return noise_model


def backend_noise(backend: str, noise_multiplier: float) -> NoiseModel | None:
    if backend == "noisy_simulator":
        return build_noise_model(noise_multiplier)
    return None


def seed_quantum_globals(seed: int) -> None:
    algorithm_globals.random_seed = int(seed)


def _require_deterministic_qnn(qnn: Any, n_features: int) -> None:
    probe = np.linspace(0.05, 0.95, 4 * n_features).reshape(4, n_features)
    weights = np.zeros(int(qnn.num_weights), dtype=np.float64)
    first = np.asarray(qnn.forward(probe, weights), dtype=np.float64).ravel()
    second = np.asarray(qnn.forward(probe, weights), dtype=np.float64).ravel()
    if first.shape != second.shape or not np.array_equal(first, second):
        raise QuantumBackendUnavailable(
            "The QNN forward pass returned different values for identical inputs, so its "
            "metrics cannot be reproduced. Refusing to report unreliable numbers; use "
            "qsvc or vqc, or run qnn in an environment with a newer "
            "qiskit / qiskit-machine-learning release."
        )
    if first.size != probe.shape[0]:
        raise QuantumBackendUnavailable(
            f"The QNN returned {first.size} values for {probe.shape[0]} samples, so its "
            "output contract is unreliable. Refusing to report unreliable numbers; use "
            "qsvc or vqc, or run qnn in an environment with a newer "
            "qiskit / qiskit-machine-learning release."
        )


def _optimizer(name: str, max_iterations: int, seed: int) -> tuple[Any, dict[str, Any]]:
    normalized = name.lower()
    iterations = max(10, int(max_iterations))
    if normalized == "spsa":
        seed_quantum_globals(seed)
        optimizer = SPSA(
            maxiter=iterations,
            blocking=False,
            learning_rate=0.1,
            perturbation=0.1,
        )
        return optimizer, {
            "name": "SPSA",
            "maxiter": iterations,
            "seed": int(seed),
            "random_source": "qiskit_machine_learning.utils.algorithm_globals.random_seed",
        }
    if normalized != "cobyla":
        raise ValueError(f"Unsupported optimizer: {name}")
    optimizer = COBYLA(maxiter=iterations)
    return optimizer, {
        "name": "COBYLA",
        "maxiter": iterations,
        "seed": int(seed),
        "random_source": "qiskit_machine_learning.utils.algorithm_globals.random_seed",
    }


def _classical_model(
    model_type: str,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    if model_type == "logistic_regression":
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        )
        hyperparameters = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
        }
    elif model_type == "svm":
        model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        )
        hyperparameters = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
        }
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            random_state=seed,
            n_jobs=1,
        )
        hyperparameters = {
            "n_estimators": 150,
            "max_depth": None,
        }
    elif model_type == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        hyperparameters = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "binary:logistic",
        }
    else:
        raise ValueError(f"Unsupported classical model: {model_type}")
    return model, {
        "library": type(model).__module__ + "." + type(model).__name__,
        "hyperparameters": hyperparameters,
        "random_state": int(seed),
        "qubits": 0,
        "circuit_depth": 0,
    }


def _sampler(
    seed: int,
    noise_model: NoiseModel | None,
    shots: int,
) -> SamplerV2:
    backend_options: dict[str, Any] = {
        "method": "density_matrix" if noise_model is not None else "statevector"
    }
    if noise_model is not None:
        backend_options["noise_model"] = noise_model
    return SamplerV2(
        default_shots=int(shots),
        seed=int(seed),
        options={"backend_options": backend_options},
    )


def _estimator(
    noise_model: NoiseModel | None,
) -> EstimatorV2:
    backend_options: dict[str, Any] = {
        "method": "density_matrix" if noise_model is not None else "statevector"
    }
    if noise_model is not None:
        backend_options["noise_model"] = noise_model
    return EstimatorV2(options={"backend_options": backend_options})


def build_model(
    model_type: str,
    n_features: int,
    seed: int,
    backend: str = "ideal_simulator",
    circuit_depth: int = 2,
    optimizer: str = "cobyla",
    max_iterations: int = 25,
    noise_multiplier: float = 0.01,
    shots: int = 512,
) -> tuple[Any, dict[str, Any]]:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    if model_type in QUANTUM_MODEL_TYPES and n_features > MAX_QUANTUM_FEATURES:
        raise ValueError(
            f"Quantum models support at most {MAX_QUANTUM_FEATURES} features; got {n_features}"
        )
    if backend == "real_hardware":
        require_hardware()
    if model_type in {"logistic_regression", "svm", "random_forest", "xgboost"}:
        model, metadata = _classical_model(model_type, int(seed))
        metadata.update(
            {
                "model_type": model_type,
                "backend": backend,
                "noise_multiplier": noise_multiplier if backend == "noisy_simulator" else 0.0,
            }
        )
        return model, metadata
    seed_quantum_globals(seed)
    if model_type == "qsvc":
        if backend == "real_hardware":
            raise HardwareNotConfigured("Real hardware not configured for the local QSVC kernel")
        noise_model = backend_noise(backend, noise_multiplier)
        feature_map = build_zz_feature_map(n_features, reps=2)
        kernel = AerStatevectorKernel(
            feature_map,
            seed=int(seed),
            noise_model=noise_model,
        )
        model = QSVC(
            quantum_kernel=kernel,
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=int(seed),
        )
        return model, {
            "model_type": model_type,
            "library": "qiskit_machine_learning.algorithms.QSVC",
            "hyperparameters": {
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
            },
            "feature_map": {
                "name": "ZZFeatureMap",
                "feature_dimension": int(n_features),
                "reps": 2,
            },
            "backend": backend,
            "simulation": "density_matrix" if backend == "noisy_simulator" else "statevector",
            "noise_multiplier": noise_multiplier if backend == "noisy_simulator" else 0.0,
            "shots": None,
            "qubits": int(n_features),
            "circuit_depth": int(feature_map.depth()),
        }
    noise_model = backend_noise(backend, noise_multiplier)
    feature_map = build_zz_feature_map(n_features, reps=2)
    optimizer_name = optimizer.lower()
    if model_type == "vqc":
        ansatz = real_amplitudes(n_features, reps=int(circuit_depth))
        optimizer_instance, optimizer_metadata = _optimizer(optimizer_name, max_iterations, int(seed))
        sampler = _sampler(int(seed), noise_model, shots)
        model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer_instance,
            sampler=sampler,
        )
        return model, {
            "model_type": model_type,
            "library": "qiskit_machine_learning.algorithms.VQC",
            "hyperparameters": optimizer_metadata,
            "feature_map": {
                "name": "ZZFeatureMap",
                "feature_dimension": int(n_features),
                "reps": 2,
            },
            "ansatz": {
                "name": "RealAmplitudes",
                "reps": int(circuit_depth),
            },
            "backend": backend,
            "simulation": "density_matrix" if backend == "noisy_simulator" else "statevector",
            "noise_multiplier": noise_multiplier if backend == "noisy_simulator" else 0.0,
            "shots": int(shots),
            "qubits": int(n_features),
            "circuit_depth": int(feature_map.depth() + ansatz.depth()),
            "optimizer": optimizer_metadata,
        }
    if model_type == "qnn":
        ansatz = real_amplitudes(n_features, reps=int(circuit_depth))
        circuit: QuantumCircuit = feature_map.compose(ansatz)
        estimator = _estimator(noise_model)
        qnn = EstimatorQNN(
            circuit=circuit,
            estimator=estimator,
            input_params=list(feature_map.parameters),
            weight_params=list(ansatz.parameters),
            input_gradients=False,
        )
        _require_deterministic_qnn(qnn, int(n_features))
        optimizer_instance, optimizer_metadata = _optimizer(optimizer_name, max_iterations, int(seed))
        model = NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=optimizer_instance,
        )
        return model, {
            "model_type": model_type,
            "library": "qiskit_machine_learning.algorithms.NeuralNetworkClassifier",
            "hyperparameters": optimizer_metadata,
            "feature_map": {
                "name": "ZZFeatureMap",
                "feature_dimension": int(n_features),
                "reps": 2,
            },
            "ansatz": {
                "name": "RealAmplitudes",
                "reps": int(circuit_depth),
            },
            "backend": backend,
            "simulation": "density_matrix" if backend == "noisy_simulator" else "statevector",
            "noise_multiplier": noise_multiplier if backend == "noisy_simulator" else 0.0,
            "shots": None,
            "qubits": int(n_features),
            "circuit_depth": int(feature_map.depth() + ansatz.depth()),
            "optimizer": optimizer_metadata,
        }
    raise ValueError(f"Unsupported quantum model: {model_type}")


def feature_map_circuit(n_features: int, reps: int = 2) -> QuantumCircuit:
    return build_zz_feature_map(n_features, reps=reps)


def circuit_text(circuit: QuantumCircuit) -> str:
    return str(circuit.draw(output="text"))


def circuit_qasm(circuit: QuantumCircuit) -> str:
    try:
        return circuit.qasm()
    except Exception:
        return circuit_text(circuit)


def model_supports_probability(model_type: str) -> bool:
    return model_type != "qnn"


def qnn_probability_from_score(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    clipped = np.clip(values, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-clipped))
