# Q-Diagnose implementation roadmap

This roadmap reflects the current repository and distinguishes the research prototype from a future patient-report prediction product. A phase is not considered complete because code exists: its exit criteria must also be met.

## Current prototype

**Status: available for research exploration.** The app supports tabular cohort experiments with classical and quantum models. The patient report workflow extracts recognizable lab values and compares them with general reference intervals. It does not currently connect a disease-specific model to an uploaded report, diagnose disease, or estimate future illness.

Saved experiment files in `results/phase2_acceptance_results.json` and `results/phase3_sweep_results.json` are historical research artifacts, not clinical validation. The recorded Phase 2 acceptance results report mean ROC-AUC values of about 0.994 for logistic regression, 0.882 for QSVC, 0.565 for VQC, and 0.509 for QNN on the benchmark dataset. The Phase 3 VQC depth sweep reports ROC-AUC values from about 0.505 to 0.548. These results do not establish performance on blood reports or patient populations.

## PS 26139 deliverable alignment

The official problem statement sets a broad research-platform objective. The current repository is a partial research prototype; the following distinction prevents research results from being mistaken for a validated patient-facing predictor.

| PS 26139 objective | Current evidence | Status / next work |
| --- | --- | --- |
| Hybrid quantum-classical architecture | Classical preprocessing and research-cohort classical/quantum experiment code are present. | Partial; document supported backends and reproducibility limits. |
| Biomedical data ingestion and feature engineering | Cohort CSV experimentation and patient lab value extraction exist. | Partial; no validated, longitudinal patient-report feature pipeline. |
| Quantum-enhanced classification | QSVC, VQC, and QNN research experiments are recorded. | Research prototype; historical metrics do not show improvement over the classical baseline. |
| Compare with classical baselines | Historical benchmark results include logistic regression and quantum models. | Partial; require matched patient-level splits, uncertainty intervals, calibration, and external validation for the intended target. |
| Explainability | Research workflows include feature attribution/interpretation artifacts. | Partial; explanations are not validated for clinical decisions or attached to report-based predictions. |
| Early disease prediction from biomedical data | No defined disease, prediction horizon, population, or approved labeled longitudinal dataset is configured. | Blocked on Phase 1 inputs; uploaded reports currently receive extraction and reference-interval review only. |
| Scalable / near-term hardware compatibility | Simulator-oriented experiments exist. | Not established for hardware execution, throughput, or production scale. |
| Comprehensive documentation | README, architecture, API, limitations, and this roadmap describe the prototype and its limits. | In progress; update alongside implementation and validation evidence. |

The PS's linked Drive file is supplementary material, not a dataset specification by itself. Its contents have not been verified in this workspace. Treat any linked data as unavailable until access, provenance, permitted use, schema, and outcome labels are reviewed.

## Remaining phases

### Phase 1 — Define the clinical prediction target and obtain an approved dataset

**Status: blocked on project inputs.** An intake worksheet is available at [`dataset-intake-template.md`](dataset-intake-template.md). Before model work, a clinical owner must specify:

- The first disease to study and the intended prediction horizon (for example, a defined outcome within a defined follow-up period).
- The intended population, exclusions, and how the outcome is confirmed.
- An authorized, de-identified dataset with patient identifiers for grouping, timestamped lab values and units, and timestamped outcomes/follow-up.
- Data-use approval and a data dictionary describing each feature, unit, reference interval source, missing-value convention, and outcome label.

Do not use the current research-cohort results or unlabelled/synthetic demo reports as substitutes for this dataset.

### Phase 2 — Build and validate the patient-report data pipeline

**Status: generic safeguards implemented; disease-specific mapping and validation await Phase 1.** The intake worksheet defines the required patient grouping, timestamps, outcome/follow-up fields, measurement dictionary, and evaluation protocol. `qdiagnose/domain/clinical_cohort.py` now provides configurable fixed-horizon outcome construction, follow-up exclusions, patient-disjoint holdouts, and chronological holdouts that purge patients appearing on both sides. These utilities do not define a disease or validate a dataset. Once an approved dataset and target are supplied, add reviewed report-to-feature mappings; normalize units only where conversions are unambiguous; preserve source values, units, and laboratory reference intervals; report missing or unrecognized fields; and lock an untouched holdout set before tuning. No dataset currently in the repository is confirmed to meet these requirements.

**Exit criteria:** reviewed mappings, reproducible preprocessing, patient-level leakage checks, and documented handling of missing values and lab-to-lab variation.

### Phase 3 — Train and evaluate disease-specific baselines and quantum candidates

**Status: not started for patient blood reports.** Start with interpretable classical baselines, then compare quantum candidates using identical splits and features. Report calibration, sensitivity, specificity, ROC-AUC, PR-AUC, confidence intervals, subgroup performance, and a clinically selected operating threshold. Compare against a simple clinical baseline. Do not show an individual risk estimate until validation supports its intended use.

**Exit criteria:** independent external validation and clinical review of the model card, dataset shift risks, limitations, and patient-facing explanation.

### Phase 4 — Clinical governance and prospective evaluation

**Status: not started.** Establish intended use, human review workflow, incident handling, monitoring for drift, update controls, and the regulatory/privacy review appropriate to the deployment jurisdiction. If required, perform a prospective evaluation before use in care.

### Phase 5 — Secure deployment

**Status: partial local safeguard; not production-ready.** Optional Supabase Auth validates a session and allows one configured user. Local mode remains unauthenticated and loopback-only. The data store remains shared SQLite without per-user row security. Before storing real patient records or exposing the service beyond a trusted local research environment, migrate data to an appropriately secured store, implement and review multi-user authorization, secure transport and storage, consent, audit logs, retention/deletion policy, backups, monitoring, and operational recovery.

## Current release gate

Until Phases 1–4 are completed for a defined intended use, patient report pages must remain in extraction-and-reference-review mode. Research experiment scores must remain clearly separated from patient reports and must not be described as patient-specific disease probabilities.
