"""Validated request models grouped by API resource."""

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ExperimentRunRequest(BaseModel):
    dataset_id: str = "breast_cancer_wdbc"
    model_type: str = Field(..., description="logistic_regression | svm | random_forest | extra_trees | hist_gradient_boosting | xgboost | qsvc | vqc | qnn")
    n_features: int = Field(default=4, ge=2, le=10, description="PCA components = Qubit count")
    backend: Literal["ideal_simulator", "noisy_simulator"] = Field(default="ideal_simulator", description="ideal_simulator | noisy_simulator")
    circuit_depth: int = Field(default=1, ge=1, le=4, description="Ansatz depth for VQC/QNN")
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of cross-validation folds")
    seeds: List[int] = Field(default_factory=lambda: [42, 123, 7], min_length=1, max_length=5)


class SweepRunRequest(BaseModel):
    dataset_id: str = "breast_cancer_wdbc"
    model_type: str = Field(default="vqc", description="logistic_regression | svm | random_forest | extra_trees | hist_gradient_boosting | xgboost | qsvc | vqc | qnn")
    sweep_param: str = Field(default="circuit_depth", description="circuit_depth | n_features | backend")
    sweep_values: List[Any] = Field(default_factory=lambda: [1, 2, 3, 4])
    n_features: int = Field(default=4, ge=2, le=10)
    backend: Literal["ideal_simulator", "noisy_simulator"] = "ideal_simulator"
    circuit_depth: int = Field(default=1, ge=1, le=4)
    cv_folds: int = Field(default=3, ge=2, le=10)
    seeds: List[int] = Field(default_factory=lambda: [42], min_length=1, max_length=5)


class PredictRequest(BaseModel):
    dataset_id: str = "breast_cancer_wdbc"
    model_type: str = Field(default="svm", description="logistic_regression | svm | random_forest | xgboost | qsvc | vqc | qnn")
    features: List[Optional[float]] = Field(
        ..., description="Values ordered exactly as returned in the selected dataset's feature_names; null values are imputed from training data."
    )
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    n_features: Optional[int] = Field(default=4, ge=2, le=10)
    circuit_depth: Optional[int] = Field(default=1, ge=1, le=4)
    backend: Optional[Literal["ideal_simulator", "noisy_simulator"]] = "ideal_simulator"


class PatientCreateRequest(BaseModel):
    patient_id: Optional[str] = Field(default=None, max_length=100)
    name: Optional[str] = Field(default="De-identified Subject", max_length=120)
    age: Optional[int] = Field(default=None, ge=0, le=130)
    sex: Optional[str] = Field(default="Unknown", max_length=40)
    notes: Optional[str] = Field(default="", max_length=4000)


class PatientRecordCreateRequest(BaseModel):
    feature_values: List[float] = Field(min_length=1, max_length=1024)
    recorded_at: Optional[str] = None
    source: Optional[str] = Field(default="manual_entry", max_length=160)

    @field_validator("feature_values")
    @classmethod
    def feature_values_must_be_finite(cls, values: List[float]) -> List[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Feature values must be finite numbers.")
        return values


class PatientPredictRequest(BaseModel):
    record_id: Optional[str] = None
    model_type: str = "svm"
    dataset_id: str = "breast_cancer_wdbc"
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    n_features: Optional[int] = Field(default=4, ge=2, le=10)
    circuit_depth: Optional[int] = Field(default=1, ge=1, le=4)
    backend: Optional[Literal["ideal_simulator", "noisy_simulator"]] = "ideal_simulator"


class LabReportAnalyzeRequest(BaseModel):
    report_text: Optional[str] = Field(default=None, max_length=100_000)
    biomarkers: Optional[Dict[str, float]] = Field(default=None, max_length=200)
    report_title: Optional[str] = Field(default=None, max_length=160)

    @field_validator("biomarkers")
    @classmethod
    def biomarkers_must_be_finite(cls, values: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if values is not None and not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Biomarker values must be finite numbers.")
        return values


class SaveLabReportToPatientRequest(BaseModel):
    patient_id: str
    report_title: str
    biomarkers: Dict[str, float] = Field(max_length=200)
    primary_disease: Optional[str] = None
    chance_percentage: Optional[float] = Field(default=None, ge=0, le=100)
    risk_tier: Optional[str] = None

    @field_validator("biomarkers")
    @classmethod
    def biomarkers_must_be_finite(cls, values: Dict[str, float]) -> Dict[str, float]:
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Biomarker values must be finite numbers.")
        return values


class ReportConfirmRequest(BaseModel):
    biomarkers: Dict[str, float] = Field(max_length=200)
    patient_profile: Optional[Dict[str, Any]] = Field(default=None, max_length=32)

    @field_validator("biomarkers")
    @classmethod
    def biomarkers_must_be_finite(cls, values: Dict[str, float]) -> Dict[str, float]:
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Biomarker values must be finite numbers.")
        return values


class RiskPredictRequest(BaseModel):
    report_id: Optional[str] = None
    biomarkers: Optional[Dict[str, float]] = None
    patient_profile: Optional[Dict[str, Any]] = None
    model_type: Optional[str] = "hybrid_quantum"
