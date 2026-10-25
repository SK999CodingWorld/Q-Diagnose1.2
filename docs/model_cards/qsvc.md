# Model Card: Quantum Support Vector Classifier (QSVC)

- **Architecture:** Quantum kernel SVM using `ZZFeatureMap` and an exact statevector fidelity kernel.
- **Experiment input:** Four PCA components angle-scaled to `[0, pi]` from the built-in breast cancer benchmark.
- **Recorded artifact:** `results/phase2_acceptance_results.json`, five folds, seed 42.
- **Recorded fold means:** Accuracy 0.814; sensitivity 0.905; specificity 0.660; ROC-AUC 0.882.
- **Output interpretation:** `QSVCWrapper.predict_proba` applies a sigmoid to the decision score for scikit-learn API compatibility. This is not probability calibration and must not be called a disease probability.
- **Limitations:** Historical single-seed benchmark only; quantum runtime is much higher than classical baselines in the recorded run. No patient-report, external, or clinical validation is available.
