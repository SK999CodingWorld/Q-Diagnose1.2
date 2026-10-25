# Model Card: XGBoost

- **Architecture:** XGBoost classifier with shallow trees, subsampling, regularization, and log-loss evaluation.
- **Experiment input:** Four PCA components from the built-in breast cancer benchmark; preprocessing is fit within each training fold.
- **Recorded artifact:** `results/phase2_acceptance_results.json`, five folds, seed 42.
- **Recorded fold means:** Accuracy 0.935; sensitivity 0.947; specificity 0.915; ROC-AUC 0.988.
- **Output interpretation:** Research benchmark output. Calibration for clinical use has not been established.
- **Limitations:** Historical single-seed benchmark only. No patient-report, external, or clinical validation is available.
