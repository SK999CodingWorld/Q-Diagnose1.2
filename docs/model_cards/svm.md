# Model Card: Support Vector Machine (RBF)

- **Architecture:** scikit-learn SVC with RBF kernel, class-weight balanced, probability output enabled.
- **Experiment input:** Four PCA components from the built-in breast cancer benchmark; preprocessing is fit within each training fold.
- **Recorded artifact:** `results/phase2_acceptance_results.json`, five folds, seed 42.
- **Recorded fold means:** Accuracy 0.958; sensitivity 0.969; specificity 0.939; ROC-AUC 0.992.
- **Output interpretation:** Research benchmark output. The classifier's probability output has not been calibrated or validated for clinical use.
- **Limitations:** Historical single-seed benchmark only. No patient-report, external, or clinical validation is available.
