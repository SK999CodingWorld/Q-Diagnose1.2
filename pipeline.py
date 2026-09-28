from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FittedPreprocessor:
    scaler: StandardScaler
    pca: PCA
    n_components: int
    random_state: int


@dataclass(frozen=True)
class TabularFittedPreprocessor:
    transformer: ColumnTransformer
    pca: PCA
    feature_names: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    n_components: int
    random_state: int
    range_scaler: MinMaxScaler | None = None


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    dataset = load_breast_cancer()
    return dataset.data.astype(np.float64, copy=False), dataset.target.astype(np.int64, copy=False)


def validate_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if features.shape[0] < 2:
        raise ValueError("At least two samples are required")
    if not np.isfinite(features).all():
        raise ValueError("X contains non-finite values")
    if len(np.unique(labels)) != 2:
        raise ValueError("The target must contain exactly two classes")
    return features, labels


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels = validate_inputs(X, y)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_train, X_test, y_train, y_test


def fit_preprocessor(
    X_train: np.ndarray,
    n_components: int = 4,
    random_state: int = 42,
) -> FittedPreprocessor:
    features = np.asarray(X_train, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("X_train must be a two-dimensional feature matrix")
    if not np.isfinite(features).all():
        raise ValueError("X_train contains non-finite values")
    maximum_components = min(features.shape)
    if n_components < 1 or n_components > maximum_components:
        raise ValueError(
            f"n_components must be between 1 and {maximum_components}, got {n_components}"
        )
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(features)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state)
    pca.fit(scaled_train)
    return FittedPreprocessor(
        scaler=scaler,
        pca=pca,
        n_components=n_components,
        random_state=random_state,
    )


def transform_with_preprocessor(
    preprocessor: FittedPreprocessor,
    X: np.ndarray,
) -> np.ndarray:
    features = np.asarray(X, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if features.shape[1] != preprocessor.pca.n_features_in_:
        raise ValueError(
            f"Expected {preprocessor.pca.n_features_in_} features, got {features.shape[1]}"
        )
    if not np.isfinite(features).all():
        raise ValueError("X contains non-finite values")
    return preprocessor.pca.transform(preprocessor.scaler.transform(features))


def preprocess_data(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    n_components: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA]:
    if X is None or y is None:
        if X is not None or y is not None:
            raise ValueError("X and y must be supplied together")
        X, y = load_dataset()
    X_train, X_test, y_train, y_test = stratified_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    fitted = fit_preprocessor(
        X_train,
        n_components=n_components,
        random_state=random_state,
    )
    X_train_transformed = transform_with_preprocessor(fitted, X_train)
    X_test_transformed = transform_with_preprocessor(fitted, X_test)
    return (
        X_train_transformed,
        X_test_transformed,
        y_train,
        y_test,
        fitted.scaler,
        fitted.pca,
    )


def _as_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        frame = pd.DataFrame(np.asarray(X, dtype=np.float64))
    else:
        frame = X.copy()
    if frame.columns.duplicated().any():
        raise ValueError("Feature column names must be unique")
    if frame.shape[1] < 1:
        raise ValueError("At least one feature column is required")
    return frame


def fit_tabular_preprocessor(
    X_train: pd.DataFrame,
    n_components: int = 4,
    random_state: int = 42,
    range_scaling: bool = False,
) -> TabularFittedPreprocessor:
    frame = _as_feature_frame(X_train)
    numeric_columns = tuple(
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    )
    categorical_columns = tuple(
        str(column) for column in frame.columns if str(column) not in numeric_columns
    )
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_columns),
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                list(categorical_columns),
            )
        )
    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    encoded_train = transformer.fit_transform(frame)
    if hasattr(encoded_train, "toarray"):
        encoded_train = encoded_train.toarray()
    encoded_train = np.asarray(encoded_train, dtype=np.float64)
    maximum_components = min(encoded_train.shape)
    if n_components < 1 or n_components > maximum_components:
        raise ValueError(
            f"n_components must be between 1 and {maximum_components}, got {n_components}"
        )
    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state)
    pca.fit(encoded_train)
    range_scaler = None
    if range_scaling:
        range_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        range_scaler.fit(pca.transform(encoded_train))
    transformed_names = [str(name) for name in transformer.get_feature_names_out()]
    feature_names = tuple(f"PC{index + 1}" for index in range(n_components))
    return TabularFittedPreprocessor(
        transformer=transformer,
        pca=pca,
        feature_names=feature_names,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        n_components=n_components,
        random_state=random_state,
        range_scaler=range_scaler,
    )


def transform_with_tabular_preprocessor(
    preprocessor: TabularFittedPreprocessor,
    X: pd.DataFrame,
) -> np.ndarray:
    frame = _as_feature_frame(X)
    expected_columns = list(preprocessor.numeric_columns) + list(preprocessor.categorical_columns)
    if list(frame.columns) != expected_columns:
        if set(frame.columns) != set(expected_columns):
            raise ValueError("Feature columns do not match the fitted training columns")
        frame = frame[expected_columns]
    transformed = preprocessor.transformer.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=np.float64)
    components = preprocessor.pca.transform(transformed)
    if preprocessor.range_scaler is not None:
        components = preprocessor.range_scaler.transform(components)
    return np.asarray(components, dtype=np.float64)


def preprocessing_config(
    n_components: int,
    test_size: float,
    random_state: int,
    range_scaling: bool = False,
) -> dict[str, Any]:
    return {
        "split": "stratified_train_test",
        "test_size": test_size,
        "split_random_state": random_state,
        "scaler": "StandardScaler",
        "pca": "PCA",
        "n_components": n_components,
        "pca_solver": "full",
        "fit_scope": "training_partition_only",
        "post_pca_range_scaler": "MinMaxScaler(0,1)" if range_scaling else None,
    }
