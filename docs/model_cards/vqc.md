# Model Card: Variational Quantum Classifier (VQC)

- **Architecture:** Qiskit VQC with `ZZFeatureMap`, `RealAmplitudes`, and COBYLA optimization.
- **Experiment input:** Four PCA components; circuit depth 1; training wrapper caps optimization to a subsample.
- **Recorded artifact:** `results/phase2_acceptance_results.json`, five folds, seed 42.
- **Recorded fold means:** Accuracy 0.568; sensitivity 0.583; specificity 0.543; ROC-AUC 0.565.
- **Output interpretation:** Research classifier scores only; no clinical probability calibration or patient-report use.
- **Limitations:** Historical single-seed benchmark; performance is near chance and below the classical baselines. No patient-report, external, or clinical validation is available.
