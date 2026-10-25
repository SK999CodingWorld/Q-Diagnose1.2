# Model Card: Estimator Quantum Neural Network (QNN)

- **Architecture:** Qiskit `EstimatorQNN` with `NeuralNetworkClassifier`, a `ZZFeatureMap`, `RealAmplitudes`, and COBYLA.
- **Experiment input:** Four PCA components; circuit depth 1.
- **Recorded artifact:** `results/phase2_acceptance_results.json`, five folds, seed 42.
- **Recorded fold means:** Accuracy 0.508; sensitivity 0.493; specificity 0.533; ROC-AUC 0.509.
- **Output interpretation:** Research classifier scores only; no clinical probability calibration or patient-report use.
- **Limitations:** Historical single-seed benchmark; performance is near chance and below classical baselines. No patient-report, external, or clinical validation is available.
