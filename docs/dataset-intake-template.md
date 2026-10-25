# PS 26139 clinical dataset intake template

Complete this with the dataset owner and a clinical reviewer before using a dataset for patient-level prediction. Do not put names, addresses, medical record numbers, or other direct identifiers in this document or an uploaded dataset.

## Study target

- First disease or clinical outcome:
- Outcome definition and source of confirmation:
- Prediction/index date definition:
- Prediction horizon (how far into the future the outcome is predicted):
- Feature lookback window (how far before index date measurements may come from):
- Intended population and care setting:
- Inclusion and exclusion criteria:
- Positive, negative, and indeterminate outcome rules:
- Follow-up end date and handling of loss to follow-up:
- Competing events and how they are handled:
- Clinical owner and reviewer:

## Dataset provenance and permissions

- Dataset name, version, and source:
- Data steward / owner:
- Collection period and care setting(s):
- Geography and population description:
- De-identification method and re-identification risk review:
- Written authorization / permitted use reference:
- Restrictions on storage, access, retention, and publication:
- External validation cohort available (yes/no; describe):

## File and record layout

- File format and encoding:
- One row represents (patient / encounter / measurement / other):
- Number of patients:
- Number of rows:
- Number of positive outcomes:
- Number of negative outcomes:
- Patient grouping key (pseudonymous, stable across visits):
- Measurement/encounter timestamp field and timezone:
- Outcome timestamp/confirmation field:
- Follow-up/censoring timestamp field:
- Outcome label field and allowed values:

### Measurement fields

For each field, document its source, unit, normalizing rule (if any), expected range, missing-value codes, and whether it is available at the prediction/index date.

| Field name | Meaning | Type | Unit / allowed values | Missing codes | Source / notes |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

### Labeling rules

Write an executable, unambiguous rule for assigning each patient/index date to a label. Confirm that no data recorded after the index date is used as a predictor. Document how repeat measurements, multiple outcomes, and patients with insufficient follow-up are treated.

## Evaluation protocol

- Patient-level grouping split required (yes/no):
- Temporal cutoff for the locked test cohort:
- External test cohort:
- Minimum event counts agreed before modeling:
- Primary metric and rationale:
- Secondary metrics (include sensitivity, specificity, ROC-AUC, PR-AUC, calibration, and confidence intervals where estimable):
- Subgroups requiring review:
- Clinical operating threshold and who selected it:
- Missingness and site/lab shift analyses:
- Baseline models to compare:
- Quantum simulator/backend and compute budget:
- Model review and sign-off owner:

## Intake decision

- [ ] Provenance and permitted use reviewed
- [ ] De-identification and access controls reviewed
- [ ] Outcome definition and time horizon approved
- [ ] Data dictionary complete
- [ ] Patient grouping key and timestamps verified
- [ ] Post-index leakage review complete
- [ ] Patient/time split protocol approved before tuning
- [ ] Clinical and data owners approved use for this study

Decision: `pending` / `approved for research preparation` / `rejected`

Decision date and approvers:

