"""Leakage-aware cohort preparation primitives for approved clinical research data.

This module deliberately does not choose a disease, outcome definition, feature set,
or prediction horizon. A study owner must supply those choices explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


class CohortValidationError(ValueError):
    """Raised when a cohort cannot safely satisfy its declared study contract."""


@dataclass(frozen=True)
class CohortSpec:
    """Column contract for one row per patient prediction/index date."""

    patient_id: str
    index_date: str
    outcome_date: str
    followup_end_date: str
    feature_columns: Sequence[str]
    horizon_days: int

    def __post_init__(self) -> None:
        if not isinstance(self.horizon_days, int) or isinstance(self.horizon_days, bool) or self.horizon_days <= 0:
            raise CohortValidationError("horizon_days must be a positive integer")
        columns = [self.patient_id, self.index_date, self.outcome_date,
                   self.followup_end_date, *self.feature_columns]
        if any(not isinstance(column, str) or not column.strip() for column in columns):
            raise CohortValidationError("All configured column names must be non-empty strings")
        if len(set(columns)) != len(columns):
            raise CohortValidationError("Patient, date, and feature columns must be distinct")
        if not self.feature_columns:
            raise CohortValidationError("At least one explicitly approved feature column is required")


@dataclass(frozen=True)
class CohortBuildResult:
    cohort: pd.DataFrame
    excluded_rows: int
    exclusion_reasons: dict[str, int]


@dataclass(frozen=True)
class CohortSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp
    purged_train_rows: int


def build_fixed_horizon_cohort(frame: pd.DataFrame, spec: CohortSpec) -> CohortBuildResult:
    """Create binary future-outcome labels and exclude rows without full follow-up.

    A positive row has a first confirmed outcome strictly after the index date and
    on or before the horizon end. A negative row has no outcome through the
    horizon and documented follow-up reaching the horizon end. Outcomes before
    or on the index date are prevalent/ambiguous and excluded. This function
    assumes outcome_date is the first confirmed outcome date for the target.
    """
    required = {spec.patient_id, spec.index_date, spec.outcome_date,
                spec.followup_end_date, *spec.feature_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise CohortValidationError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise CohortValidationError("The input dataset has no rows")

    data = frame.loc[:, list(dict.fromkeys([
        spec.patient_id, spec.index_date, spec.outcome_date,
        spec.followup_end_date, *spec.feature_columns,
    ]))].copy()
    data[spec.index_date] = pd.to_datetime(data[spec.index_date], errors="coerce", utc=True)
    data[spec.outcome_date] = pd.to_datetime(data[spec.outcome_date], errors="coerce", utc=True)
    data[spec.followup_end_date] = pd.to_datetime(data[spec.followup_end_date], errors="coerce", utc=True)

    blank_patient = data[spec.patient_id].isna() | data[spec.patient_id].astype(str).str.strip().eq("")
    invalid_index = data[spec.index_date].isna()
    invalid_followup = data[spec.followup_end_date].isna()
    duplicate_index = data.duplicated([spec.patient_id, spec.index_date], keep=False)
    horizon_end = data[spec.index_date] + pd.to_timedelta(spec.horizon_days, unit="D")
    prevalent = data[spec.outcome_date].notna() & (data[spec.outcome_date] <= data[spec.index_date])
    event_after_followup = data[spec.outcome_date].notna() & (
        data[spec.outcome_date] > data[spec.followup_end_date]
    )
    complete_followup = data[spec.followup_end_date] >= horizon_end
    positive = data[spec.outcome_date].notna() & (
        data[spec.outcome_date] > data[spec.index_date]
    ) & (data[spec.outcome_date] <= horizon_end) & ~event_after_followup
    no_event_within_horizon = data[spec.outcome_date].isna() | (data[spec.outcome_date] > horizon_end)
    negative = no_event_within_horizon & complete_followup

    invalid = blank_patient | invalid_index | invalid_followup | duplicate_index | prevalent | event_after_followup
    eligible = ~invalid & (positive | negative)

    reasons = {
        "missing_patient_id": int(blank_patient.sum()),
        "invalid_index_date": int(invalid_index.sum()),
        "invalid_followup_date": int(invalid_followup.sum()),
        "duplicate_patient_index_date": int(duplicate_index.sum()),
        "prevalent_or_index_date_outcome": int(prevalent.sum()),
        "outcome_after_recorded_followup": int(event_after_followup.sum()),
        "insufficient_followup_or_unresolved": int((~invalid & ~eligible).sum()),
    }
    cohort = data.loc[eligible].copy()
    cohort["outcome"] = positive.loc[eligible].astype("int8")
    cohort["horizon_end"] = horizon_end.loc[eligible]
    if cohort.empty:
        raise CohortValidationError("No rows meet the declared fixed-horizon outcome and follow-up rules")
    if cohort["outcome"].nunique() < 2:
        raise CohortValidationError("Eligible cohort contains only one outcome class")
    return CohortBuildResult(
        cohort=cohort.reset_index(drop=True),
        excluded_rows=int(len(frame) - len(cohort)),
        exclusion_reasons=reasons,
    )


def patient_temporal_split(
    cohort: pd.DataFrame,
    *,
    patient_id: str,
    index_date: str,
    cutoff: str | pd.Timestamp,
    label: str = "outcome",
) -> CohortSplit:
    """Split chronologically and purge patients appearing on both sides.

    Rows on the cutoff date belong to test. Any earlier rows for those test
    patients are purged from training so the patient groups remain disjoint.
    Both partitions must contain both outcome classes.
    """
    required = {patient_id, index_date, label}
    missing = sorted(required.difference(cohort.columns))
    if missing:
        raise CohortValidationError(f"Missing split columns: {', '.join(missing)}")
    if cohort.empty:
        raise CohortValidationError("Cannot split an empty cohort")
    if cohort[patient_id].isna().any():
        raise CohortValidationError("Patient IDs cannot be missing for temporal splitting")
    dates = pd.to_datetime(cohort[index_date], errors="coerce", utc=True)
    boundary = pd.to_datetime(cutoff, errors="coerce", utc=True)
    if pd.isna(boundary):
        raise CohortValidationError("cutoff must be a valid date or timestamp")
    if dates.isna().any():
        raise CohortValidationError("Index dates must all be valid before splitting")

    test_mask = dates >= boundary
    candidate_train = cohort.loc[~test_mask]
    test = cohort.loc[test_mask]
    test_patients = set(test[patient_id].dropna())
    train = candidate_train.loc[~candidate_train[patient_id].isin(test_patients)]
    purged = int(len(candidate_train) - len(train))

    if train.empty or test.empty:
        raise CohortValidationError("The temporal cutoff produced an empty train or test partition")
    if set(train[patient_id]) & set(test[patient_id]):
        raise CohortValidationError("Patient overlap remains after temporal purging")
    for name, partition in (("train", train), ("test", test)):
        if partition[label].nunique(dropna=True) < 2:
            raise CohortValidationError(f"The {name} partition must contain both outcome classes")
    return CohortSplit(
        train=train.copy(), test=test.copy(), cutoff=boundary, purged_train_rows=purged
    )


def patient_group_holdout(
    cohort: pd.DataFrame,
    *,
    patient_id: str,
    label: str = "outcome",
    test_size: float = 0.2,
    random_state: int = 42,
    attempts: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find a reproducible patient-disjoint holdout with both classes per side.

    This is a fallback when no defensible temporal cutoff exists; it does not
    replace external or temporal validation.
    """
    required = {patient_id, label}
    missing = sorted(required.difference(cohort.columns))
    if missing:
        raise CohortValidationError(f"Missing split columns: {', '.join(missing)}")
    if not 0 < test_size < 1:
        raise CohortValidationError("test_size must be between 0 and 1")
    if cohort[patient_id].isna().any():
        raise CohortValidationError("Patient IDs cannot be missing for grouped splitting")
    if cohort[label].nunique(dropna=True) < 2:
        raise CohortValidationError("A grouped holdout requires both outcome classes")

    splitter = GroupShuffleSplit(n_splits=attempts, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(cohort, cohort[label], groups=cohort[patient_id]):
        train, test = cohort.iloc[train_idx], cohort.iloc[test_idx]
        if train[label].nunique(dropna=True) == 2 and test[label].nunique(dropna=True) == 2:
            if set(train[patient_id]) & set(test[patient_id]):
                raise CohortValidationError("Patient overlap detected in grouped holdout")
            return train.copy(), test.copy()
    raise CohortValidationError(
        "Could not create a patient-disjoint holdout with both outcome classes; "
        "review event counts, cohort size, and test_size"
    )
