# Clinical cohort pipeline (research preparation)

`qdiagnose/domain/clinical_cohort.py` contains reusable cohort preparation tools. It is not a diagnostic model and does not establish clinical validity. Use it only with data whose provenance and permitted use have been approved, after the intended disease, outcome, index date, horizon, population, and features have been specified by the study team.

## Expected cohort layout

Supply one candidate prediction row per patient and index date, with columns for:

- a stable pseudonymous patient grouping key;
- the index/prediction date;
- first confirmed target outcome date (blank if none observed);
- end of documented follow-up;
- an explicitly reviewed set of candidate features available at the index date.

Repeated measurements need to be summarized or aligned to the index date by an approved study rule before modeling. This module does not parse arbitrary reports, normalize laboratory units, resolve test synonyms, select clinical features, or decide how outcomes should be confirmed.

## Build a fixed-horizon cohort

```python
from qdiagnose.domain.clinical_cohort import CohortSpec, build_fixed_horizon_cohort

spec = CohortSpec(
    patient_id="subject_key",
    index_date="index_date",
    outcome_date="first_confirmed_outcome_date",
    followup_end_date="followup_end_date",
    feature_columns=["reviewed_feature_a", "reviewed_feature_b"],
    horizon_days=365,  # Set from the approved study protocol.
)
result = build_fixed_horizon_cohort(frame, spec)
cohort = result.cohort
print(result.excluded_rows, result.exclusion_reasons)
```

An outcome after index and within the declared horizon is labeled positive. A no-outcome row is labeled negative only if follow-up reaches the horizon end. Prevalent outcomes, invalid dates, duplicate patient/index rows, outcomes after recorded follow-up, and unresolved/incompletely followed rows are excluded. Counts by reason can overlap when one row has multiple data-quality problems.

## Split without patient leakage

Prefer a protocol-approved calendar cutoff and chronological test period:

```python
from qdiagnose.domain.clinical_cohort import patient_temporal_split

split = patient_temporal_split(
    cohort,
    patient_id="subject_key",
    index_date="index_date",
    cutoff="YYYY-MM-DD",  # Replace with a protocol-approved cutoff.
)
```

Rows on the cutoff date enter the test set; earlier rows for patients in test are purged from training. If no defensible temporal cutoff exists, `patient_group_holdout` creates a reproducible patient-disjoint random holdout, but that is not external or temporal validation. Both split functions fail closed if either partition lacks both classes.

## Before fitting any model

1. Complete `dataset-intake-template.md` with the data steward and clinical reviewer.
2. Verify the target date and feature availability rules against source records.
3. Inspect exclusion counts, class counts, missingness, site/lab effects, and subgroup coverage.
4. Lock splits before feature selection or hyperparameter tuning.
5. Fit imputation, scaling, feature selection, calibration, and models on training data only.
6. Evaluate baselines and quantum candidates on identical patient-level partitions; report confidence intervals and calibration as well as discrimination metrics.

Do not feed these outputs to an individual patient workflow or present model scores as clinical risk probabilities without the separate clinical validation and governance gates in the roadmap.

