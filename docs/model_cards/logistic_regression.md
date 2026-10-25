# Model Card: Logistic Regression

- **Architecture:** L2-regularized scikit-learn logistic regression (`max_iter=2000`, class-weight balanced).
- **Experiment input:** Four PCA components from the built-in breast cancer benchmark; imputation, scaling, and PCA are fit within each training fold.
- **Recorded artifact:** `results/phase2_acceptance_results.json`, five folds, seed 42.
- **Recorded fold means:** Accuracy 0.965; sensitivity 0.980; specificity 0.939; ROC-AUC 0.994.
- **Output interpretation:** Research benchmark output. Calibration for clinical use has not been established.
- **Limitations:** Historical single-seed benchmark only. PCA-component explanations are not direct explanations in original lab measurements. No patient-report, external, or clinical validation is available.
