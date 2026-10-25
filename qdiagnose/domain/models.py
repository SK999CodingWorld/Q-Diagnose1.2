"""
Q-Diagnose Models Module
=========================
Model builders for all 7 supported architectures:
- Classical: Logistic Regression, SVM (RBF), Random Forest, XGBoost
- Quantum: QSVC (Quantum Kernel), VQC (Variational), QNN (Estimator-based)

All models adhere to the standard scikit-learn estimator interface (fit, predict, predict_proba).
"""

import numpy as np
from typing import Dict, Any, Callable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import QSVC, VQC, NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from qdiagnose.domain.evaluate import StatevectorQuantumKernel


def build_logistic_regression(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)


def build_svm(random_state: int = 42) -> SVC:
    return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=random_state, n_jobs=-1,
    )


def build_extra_trees(random_state: int = 42) -> ExtraTreesClassifier:
    """Strong randomized tree ensemble for tabular data and nonlinear feature effects."""
    return ExtraTreesClassifier(
        n_estimators=400, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=random_state, n_jobs=-1,
    )


def build_hist_gradient_boosting(random_state: int = 42) -> HistGradientBoostingClassifier:
    """Regularized histogram gradient boosting baseline for medium-sized tabular datasets."""
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, random_state=random_state,
    )


def build_xgboost(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


# --- Risk Tier Configuration & Stratification ---
DEFAULT_RISK_TIERS = {
    "low": {"max": 0.30, "label": "Low Risk", "color": "#10B981", "badge_class": "text-emerald-400 bg-emerald-950/60 border-emerald-800/50"},
    "moderate": {"min": 0.30, "max": 0.60, "label": "Moderate Risk", "color": "#FBBF24", "badge_class": "text-yellow-400 bg-yellow-950/60 border-yellow-800/50"},
    "high": {"min": 0.60, "max": 0.85, "label": "High Risk", "color": "#F97316", "badge_class": "text-orange-400 bg-orange-950/60 border-orange-800/50"},
    "very_high": {"min": 0.85, "label": "Very High Risk", "color": "#EF4444", "badge_class": "text-rose-400 bg-rose-950/60 border-rose-800/50"},
}


def assign_risk_tier(probability: float, tiers: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Stratify a disease risk probability into auditable tiers.
    
    Default boundaries:
    - Low Risk: prob < 0.30
    - Moderate Risk: 0.30 <= prob < 0.60
    - High Risk: 0.60 <= prob < 0.85
    - Very High Risk: prob >= 0.85
    """
    if tiers is None:
        tiers = DEFAULT_RISK_TIERS
    
    if probability is None or np.isnan(probability):
        raise ValueError("Invalid risk probability: value cannot be None or NaN")

    prob = float(np.clip(probability, 0.0, 1.0))
    if prob < tiers["low"]["max"]:
        tier_key = "low"
    elif prob < tiers["moderate"]["max"]:
        tier_key = "moderate"
    elif prob < tiers["high"]["max"]:
        tier_key = "high"
    else:
        tier_key = "very_high"
        
    return {
        "tier": tier_key,
        "label": tiers[tier_key]["label"],
        "color": tiers[tier_key]["color"],
        "probability": prob,
        "threshold_bounds": {
            "low_max": tiers["low"]["max"],
            "moderate_max": tiers["moderate"]["max"],
            "high_max": tiers["high"]["max"],
        }
    }


class QSVCWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for Qiskit QSVC providing standard predict_proba() via
    a sigmoid transform of the quantum kernel decision score. This output is
    a bounded model score and is not calibrated as a clinical probability.
    """
    def __init__(self, qsvc=None):
        self.qsvc = qsvc

    def fit(self, X, y):
        self.qsvc.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.qsvc.predict(X)

    def decision_function(self, X):
        return self.qsvc.decision_function(X)

    def predict_proba(self, X):
        """
        Convert the decision score to a bounded [0, 1] score for sklearn API
        compatibility. The sigmoid transform is not probability calibration.
        """
        from scipy.special import expit
        df = self.qsvc.decision_function(X)
        p1 = expit(df)
        return np.column_stack([1.0 - p1, p1])


def build_qsvc(n_features: int = 4, backend: str = "ideal_simulator", random_state: int = 42) -> Pipeline:
    """QSVC using ZZFeatureMap on ideal or noisy simulator with angle scaling and predict_proba support."""
    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=2,
        entanglement="linear",
    )
    noise_rate = 0.02 if backend == "noisy_simulator" else 0.0
    kernel = StatevectorQuantumKernel(feature_map=feature_map, noise_rate=noise_rate)
    raw_qsvc = QSVC(quantum_kernel=kernel, random_state=random_state)
    wrapped = QSVCWrapper(raw_qsvc)
    return Pipeline([
        ("angle_scaler", MinMaxScaler(feature_range=(0, np.pi))),
        ("qsvc", wrapped)
    ])


def train_classical_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """Train classical SVM (RBF kernel) on X_train, y_train."""
    model = build_svm(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_qsvc(X_train: np.ndarray, y_train: np.ndarray, n_qubits: int = 4, backend: str = "ideal_simulator") -> Pipeline:
    """Train QSVC using ZZFeatureMap on ideal or noisy simulator."""
    model = build_qsvc(n_features=n_qubits, backend=backend, random_state=42)
    model.fit(X_train, y_train)
    return model


from sklearn.base import BaseEstimator, ClassifierMixin


class SubsampledEstimatorWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for variational quantum models (VQC/QNN) that subsamples
    large training sets during iterative optimization to ensure fast,
    interactive training times while preserving validation integrity.
    """
    def __init__(self, base_estimator=None, max_train_samples: int = 80, random_state: int = 42):
        self.base_estimator = base_estimator
        self.max_train_samples = max_train_samples
        self.random_state = random_state

    def __getattr__(self, name):
        if name in ["base_estimator", "max_train_samples", "random_state"]:
            return super().__getattribute__(name)
        if hasattr(self, "base_estimator") and self.base_estimator is not None:
            return getattr(self.base_estimator, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def fit(self, X, y):
        n_samples = len(y)
        if n_samples > self.max_train_samples:
            use_stratify = False
            try:
                counts = np.bincount(y)
                if len(counts) >= 2 and np.min(counts) >= 2 and (n_samples - self.max_train_samples) >= 2:
                    use_stratify = True
            except Exception:
                use_stratify = False

            if use_stratify:
                try:
                    from sklearn.model_selection import train_test_split
                    _, X_sub, _, y_sub = train_test_split(
                        X, y,
                        test_size=self.max_train_samples,
                        stratify=y,
                        random_state=self.random_state
                    )
                except Exception:
                    rng = np.random.RandomState(self.random_state)
                    indices = rng.choice(n_samples, size=self.max_train_samples, replace=False)
                    X_sub = X[indices]
                    y_sub = y[indices]
            else:
                rng = np.random.RandomState(self.random_state)
                indices = rng.choice(n_samples, size=self.max_train_samples, replace=False)
                X_sub = X[indices]
                y_sub = y[indices]
        else:
            X_sub, y_sub = X, y
        self.base_estimator.fit(X_sub, y_sub)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        if hasattr(self.base_estimator, "predict_proba"):
            return self.base_estimator.predict_proba(X)
        elif hasattr(self.base_estimator, "decision_function"):
            from scipy.special import expit
            df = self.base_estimator.decision_function(X)
            p1 = expit(df)
            return np.column_stack([1 - p1, p1])
        else:
            preds = self.base_estimator.predict(X)
            preds_flat = np.asarray(preds).flatten()
            p1 = np.where(preds_flat > 0.5, 0.9, 0.1)
            return np.column_stack([1 - p1, p1])


def build_vqc(n_features: int = 4, circuit_depth: int = 1, backend: str = "ideal_simulator", random_state: int = 42) -> Pipeline:
    """Variational Quantum Classifier with RealAmplitudes ansatz and COBYLA."""
    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=1,
        entanglement="linear",
    )
    ansatz = RealAmplitudes(
        num_qubits=n_features,
        reps=circuit_depth,
        entanglement="linear",
    )
    optimizer = COBYLA(maxiter=20)
    if backend == "noisy_simulator":
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.primitives import Sampler as AerSampler
        fake_b = GenericBackendV2(num_qubits=n_features, seed=random_state)
        nm = NoiseModel.from_backend(fake_b)
        sampler = AerSampler(backend_options={"noise_model": nm})
    else:
        sampler = StatevectorSampler()

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
    )
    wrapped = SubsampledEstimatorWrapper(vqc, max_train_samples=60, random_state=random_state)
    return Pipeline([
        ("angle_scaler", MinMaxScaler(feature_range=(0, np.pi))),
        ("vqc", wrapped)
    ])


class QNNClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for EstimatorQNN ensuring scikit-learn binary classification compatibility."""
    def __init__(self, n_features: int = 4, circuit_depth: int = 1, backend: str = "ideal_simulator", random_state: int = 42):
        self.n_features = n_features
        self.circuit_depth = circuit_depth
        self.backend = backend
        self.random_state = random_state

        fm = ZZFeatureMap(feature_dimension=n_features, reps=1, entanglement="linear")
        ansatz = RealAmplitudes(num_qubits=n_features, reps=circuit_depth, entanglement="linear")
        qc = fm.compose(ansatz)
        if backend == "noisy_simulator":
            from qiskit.providers.fake_provider import GenericBackendV2
            from qiskit_aer.noise import NoiseModel
            from qiskit_aer.primitives import Estimator as AerEstimator
            fake_b = GenericBackendV2(num_qubits=n_features, seed=random_state)
            nm = NoiseModel.from_backend(fake_b)
            estimator = AerEstimator(backend_options={"noise_model": nm})
        else:
            estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=qc,
            input_params=fm.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )
        self.clf = NeuralNetworkClassifier(self.qnn, optimizer=COBYLA(maxiter=20))
        self.subsampler = SubsampledEstimatorWrapper(self.clf, max_train_samples=60, random_state=random_state)

    def fit(self, X, y):
        y_scaled = np.where(y == 1, 1, -1)
        self.subsampler.fit(X, y_scaled)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        raw = self.subsampler.predict(X)
        raw_flat = np.asarray(raw).flatten()
        return np.where(raw_flat > 0, 1, 0)

    def predict_proba(self, X):
        from scipy.special import expit
        raw = self.subsampler.predict(X)
        raw_flat = np.asarray(raw).flatten()
        p1 = expit(raw_flat)
        return np.column_stack([1 - p1, p1])


def build_qnn(n_features: int = 4, circuit_depth: int = 1, backend: str = "ideal_simulator", random_state: int = 42) -> Pipeline:
    """EstimatorQNN classifier with ZZFeatureMap and RealAmplitudes."""
    qnn_model = QNNClassifierWrapper(n_features=n_features, circuit_depth=circuit_depth, backend=backend, random_state=random_state)
    return Pipeline([
        ("angle_scaler", MinMaxScaler(feature_range=(0, np.pi))),
        ("qnn", qnn_model)
    ])


MODEL_REGISTRY: Dict[str, Callable] = {
    "logistic_regression": build_logistic_regression,
    "svm": build_svm,
    "random_forest": build_random_forest,
    "extra_trees": build_extra_trees,
    "hist_gradient_boosting": build_hist_gradient_boosting,
    "xgboost": build_xgboost,
    "qsvc": build_qsvc,
    "vqc": build_vqc,
    "qnn": build_qnn,
}


def get_model(
    model_type: str,
    n_features: int = 4,
    circuit_depth: int = 1,
    backend: str = "ideal_simulator",
    random_state: int = 42
):
    """Factory function for retrieving model instances."""
    m_type = model_type.lower()
    if m_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose from {list(MODEL_REGISTRY.keys())}")

    if m_type == "qsvc":
        return MODEL_REGISTRY[m_type](n_features=n_features, backend=backend, random_state=random_state)
    elif m_type in ["vqc", "qnn"]:
        return MODEL_REGISTRY[m_type](n_features=n_features, circuit_depth=circuit_depth, backend=backend, random_state=random_state)
    else:
        return MODEL_REGISTRY[m_type](random_state=random_state)
