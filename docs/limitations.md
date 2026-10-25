# Q-Diagnose: Limitations and Release Boundaries

> **Research prototype only.** This application has not been validated or authorized as a medical diagnostic device. It must not be used as the sole basis for diagnosis, triage, or treatment.

## Patient report workflow

- Report handling extracts recognized values and compares them with general reference intervals. Extraction may fail or interpret units incorrectly; users must check the source report.
- No disease-specific model is connected to uploaded reports. The workflow does not estimate an individual patient's future disease risk.
- General reference intervals vary by laboratory, method, age, sex, pregnancy status, and clinical context. An out-of-range flag is not a diagnosis.
- Do not upload real patient-identifiable information to this prototype. Local mode is intended for loopback development. Optional Supabase Auth mode restricts API access to one configured user, but data is still held in shared local SQLite and this does not provide multi-user isolation or production data-security controls.

## Research model evidence

- Saved benchmark metrics describe the dataset, split, and run stored in the corresponding research artifact. They do not validate performance on patient reports or future outcomes.
- Current benchmark evidence is historical and uses a single seed for the Phase 2 acceptance run. There is no reported external validation, prospective evaluation, or clinically selected operating threshold for a patient-report model.
- Classical baselines outperformed QSVC, VQC, and QNN in the recorded benchmark run. No quantum advantage has been demonstrated.
- The QSVC wrapper converts decision scores with a sigmoid for API compatibility; these values are not calibrated probabilities. Other model outputs also lack calibration for clinical use unless separately established.
- Cohort explanations are research tools. PCA-space feature attributions may not map directly to source measurements, SHAP can use approximations or fallback behavior, and quantum perturbation sensitivities are not clinical explanations.

## Dataset and evaluation limits

- Existing public datasets in the app are small or task-specific benchmark cohorts. Several are cross-sectional and do not contain the timestamps, patient grouping, follow-up, and future outcomes required for longitudinal prediction.
- The app has not established a disease target, prediction horizon, target population, outcome definition, or approved longitudinal clinical dataset for patient-report forecasting.
- A fixed-horizon cohort builder and patient-disjoint split utilities are now available for research preparation. They do not verify source provenance, clinical label quality, feature availability at the index date, unit conversion, or suitability of a dataset.
- No patient-level model should be trained or shown to users until the study protocol, data permissions, and feature/outcome definitions have been reviewed and an independent evaluation plan is locked.

## Quantum execution and operations

- Quantum models currently run on simulators. The repository does not demonstrate execution on a physical quantum processor, production-scale throughput, or robust deployment under hardware noise.
- Local mode can run without authentication for loopback development. Supabase mode validates tokens and allowlists one user, but does not move SQLite data into Supabase or add per-user RLS. The app is not ready for real patient records or internet-facing deployment.
- Before any clinical or production deployment, the responsible organization must determine applicable privacy, security, regulatory, and clinical-governance requirements and implement appropriate controls.
