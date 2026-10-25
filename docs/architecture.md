# Q-Diagnose: System Architecture & Data Flow

> **Smart India Hackathon 2025 | Problem Statement ID: 26139**  
> **Organization:** Egreen Quanta | **Theme:** MedTech / BioTech / HealthTech  

---

## 1. System Architecture Overview

Q-Diagnose is structured into a modular, decoupled hybrid quantum-classical architecture that separates deterministic classical feature preprocessing from quantum state preparation and Hilbert space projection.

```mermaid
flowchart TD
    subgraph INGESTION["1. Data Ingestion & Governance Layer"]
        CSV[Clinical CSV Ingestion] --> VAL[Schema & Binary Target Validation]
        VAL --> DB[(SQLite Database Metadata Store)]
    end

    subgraph PREPROCESSING["2. Fold-local cohort preprocessing"]
        VAL --> SPLIT[Configured stratified cross-validation folds and seeds]
        SPLIT --> TRAIN_SPLIT[Training Fold (80%)]
        SPLIT --> TEST_SPLIT[Held-Out Test Fold (20%)]
        TRAIN_SPLIT --> SCALER[Fit StandardScaler]
        SCALER --> PCA_FIT[Fit PCA: N Components = Qubits]
        TEST_SPLIT -.->|Transform Only| SCALER
        TEST_SPLIT -.->|Transform Only| PCA_FIT
        PCA_FIT --> ANGLE[Scale Angles to [0, π]]
    end

    subgraph QUANTUM_LAYER["3. Quantum Processing Unit (QPU) / Simulator"]
        ANGLE --> FM[ZZFeatureMap Encoding (2nd Order Entanglement)]
        FM --> KERNEL[Statevector Quantum Kernel Evaluator]
        FM --> ANSATZ[RealAmplitudes Parameterized Circuit (Depth 1-4)]
        ANSATZ --> COBYLA[COBYLA Classical Optimizer Loop]
        KERNEL --> QSVC[Quantum Support Vector Classifier]
        COBYLA --> VQC[Variational Quantum Classifier]
        COBYLA --> QNN[Estimator Quantum Neural Network]
    end

    subgraph DECISION_SUPPORT["4. Research Evaluation"]
        QSVC --> PROB[Research model score]
        VQC --> PROB
        QNN --> PROB
        PROB --> THRESH[Research threshold evaluation]
        THRESH --> STRAT[Dataset-level classification output]
        PROB --> ROC_SWEEP[Dense ROC / PR Curve Operating Sweep]
    end

    subgraph SERVING["5. Presentation & API Layer"]
        DB --> REST[FastAPI REST Backend (Uvicorn 8000)]
        DECISION_SUPPORT --> REST
        REST --> UI[React 18 + Tailwind CSS + Recharts Dashboard]
    end
```

---

## 2. Classical Front-End vs QPU/Simulator Boundary

1. **Classical Responsibilities:**
   - Missing-value imputation, scaling, and PCA fitted within training folds for cohort experiments.
   - ROC-AUC and PR-AUC metric computation and threshold evaluation for research cohorts.
   - QSVC's sigmoid transformation is for estimator API compatibility; it is not probability calibration.
   - Classical explainability via SHAP (TreeExplainer and KernelExplainer).

2. **Quantum Processing Unit (QPU) / Simulator Responsibilities:**
   - Non-linear feature mapping into $2^N$-dimensional Hilbert space using $N$-qubit parameterized quantum circuits.
   - Quantum kernel matrix generation: $K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$.
   - Quantum variational optimization using `StatevectorSampler` and `StatevectorEstimator` with `COBYLA`.
   - Depolarizing noise simulation via Qiskit Aer `GenericBackendV2` to evaluate NISQ viability.

---

## 3. Data Encoding & Quantum Circuits

- **Feature Map:** `ZZFeatureMap(feature_dimension=N, reps=2, entanglement='linear')`
  - $U_{\Phi(x)} = \exp\left(i \sum_{j} x_j Z_j + \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k\right)$
  - Standard angle encoding scales PCA inputs into $[0, \pi]$ to prevent periodicity aliasing.
- **Ansatz:** `RealAmplitudes(num_qubits=N, reps=d, entanglement='linear')`
  - Alternating layers of parameterized $R_y(\theta)$ rotation gates and linear CNOT entanglement.

---

## 4. Patient Lab Report Review

The current patient report workflow is an extraction and review tool, not a multi-disease prediction engine:

1. **Supported inputs:** `.pdf`, `.docx`, `.csv`, `.txt`, and `.json` reports are accepted by the report workflow.
2. **Biomarker extraction:** `lab_analyzer.py` recognizes supported biomarker names and numeric values from report text. Recognition can fail or misread units, so users must verify extracted values against the original report.
3. **Reference review:** Values may be compared with general reference intervals. An interval flag is not a diagnosis; ranges vary by laboratory and patient context.
4. **Prediction boundary:** No validated disease-specific ML or quantum model is connected to uploaded patient reports. The report workflow does not diagnose disease or predict future illness. Research-cohort model scores must not be applied to individual reports.

---

## 5. Research Patient Records and Prediction History

The database layer (`database.py`) provides SQLite persistence for research demonstrations:

- **Patient Profiles:** De-identified research subjects with demographics (ID, age, sex, clinical notes).
- **Visit Records (`patient_records`):** Serial clinical visits storing raw feature arrays and data sources (e.g. "Lab Report: Inpatient Lipid Panel").
- **Prediction Logs (`patient_predictions`):** Timestamped results from configured cohort-model workflows. They are not validated longitudinal disease forecasts from uploaded lab reports.
- **Scope:** This local research record store has no configured authentication and is not ready for real patient-identifiable data or clinical monitoring.

---

## 6. Delivery Table (Expected Deliverables)

| Deliverable ID | Module | Implementation Scope | Tech Stack | Status |
| :--- | :--- | :--- | :--- | :---: |
| **DEL-01** | **Data Ingestion** | Tabular cohort preprocessing; separate lab report extraction/review | Python 3.11, scikit-learn | `PARTIAL` |
| **DEL-02** | **Classical Baselines** | Classical models for labeled research cohorts | scikit-learn, XGBoost | `IMPLEMENTED (RESEARCH)` |
| **DEL-03** | **Quantum Models** | QSVC, VQC, QNN candidates for cohort experiments | Qiskit, Qiskit ML | `IMPLEMENTED (RESEARCH)` |
| **DEL-04** | **Patient Report Prediction** | Biomarker extraction and general interval review; no disease-specific report model | Python, report parsers | `PARTIAL` |
| **DEL-05** | **Explainability** | Research model and circuit explanations | SHAP, Qiskit | `IMPLEMENTED (RESEARCH)` |
| **DEL-06** | **Benchmarking** | Cohort CV, threshold, noise, and depth studies | scikit-learn, Aer | `RESEARCH ARTIFACTS PRESENT` |
| **DEL-07** | **Interactive Web UI** | Report-review and research dashboard prototype | React, Vite, FastAPI | `PROTOTYPE` |
| **DEL-08** | **Documentation** | Current implementation scope and future validation roadmap | Markdown, Docx | `IN PROGRESS` |

