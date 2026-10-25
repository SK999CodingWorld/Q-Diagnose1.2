# Model Card: Random Forest

- **Architecture:** scikit-learn RandomForestClassifier (`n_estimators=400`, `min_samples_leaf=2`, balanced subsample weights).
- **Experiment input:** Four PCA components from the built-in breast cancer benchmark; preprocessing is fit within each training fold.
- **Recorded artifact:** `results/phase2_acceptance_results.json`, five folds, seed 42.
- **Recorded fold means:** Accuracy 0.946; sensitivity 0.955; specificity 0.929; ROC-AUC 0.986.
- **Output interpretation:** Research benchmark output. Calibration for clinical use has not been established.
- **Limitations:** Historical single-seed benchmark only. No patient-report, external, or clinical validation is available.
