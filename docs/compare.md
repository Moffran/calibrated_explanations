# Calibrated Explanations comparisons (CE-first)

All code examples in this repo use `WrapCalibratedExplainer`.

## vs SHAP

- ✅ Uncertainty bounds on feature importance (aleatoric + epistemic)
- ✅ Calibrated predictions included
- ⚠️ Requires held-out calibration set for guarantees (SHAP does not)

## vs LIME

- ✅ Built-in uncertainty quantification and more stable rules
- ⚠️ Needs calibration set

## vs Conformal Prediction packages

- ✅ Provides both calibrated predictions and feature-level uncertainty / explanations
- ✅ Builds on conformal methods (Venn-Abers, CPS)
- ⚠️ Conformal focuses on prediction sets or intervals
