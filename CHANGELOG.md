# Changelog

## [Unreleased]
### Features
- Added a `WrapCalibratedExplainer` class which can be used for both classificaiton and regression.  
### Fixes
- Removed the dependency on `shap` and `scikit-learn` and closed issue #8.
- Added [LIME_comparison](https://github.com/Moffran/calibrated_explanations/notebooks/LIME_comparison.ipynb) to the notebooks folder. 
- Updated the weights to match LIME's weights (to ensure that a positive weight has the same meaning in both). 
- Changed name of parameter `y` (representing the threshold in probabilistic regression) to `threshold`. 

## [v0.1.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.1.1) - 2023-09-14
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.1.0...v0.1.1)
### Features
- Exchanged the slow `VennABERS_by_def` function for the `VennAbers` class in the `venn-abers` package.
### Fixes
- Low and high weights are correctly assigned, so that low < high is always the case.
- Adjusted the number of decimals in counterfactual rules to 2.
## [v0.1.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.1.0) - 2023-09-04

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.0.2...v0.1.0)

### Features

- **Performance**: Fast, reliable, stable and robust feature importance explanations.
- **Calibrated Explanations**: Calibration of the underlying model to ensure that predictions reflect reality.
- **Uncertainty Quantification**: Uncertainty quantification of the prediction from the underlying model and the feature importance weights.
- **Interpretation**: Rules with straightforward interpretation in relation to the feature weights.
- **Factual and Counterfactual Explanations**: Possibility to generate counterfactual rules with uncertainty quantification of the expected predictions achieved.
- **Conjunctive Rules**: Conjunctive rules conveying joint contribution between features.
- **Multiclass Support**: Multiclass support has been added since the original version developed for the paper [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://arxiv.org/pdf/2305.02305.pdf).
- **Regression Support**: Support for explanations from standard regression was developed and is described in the paper [Calibrated Explanations for Regression](https://arxiv.org/pdf/2308.16245.pdf).
- **Probabilistic Regression Support**: Support for probabilistic explanations from standard regression was added together with regression and is described in the paper mentioned above.
- **Conjunctive Rules**: Since the original version, conjunctive rules has also been added.
- **Code Structure**: The code structure has been improved a lot. The `CalibratedExplainer`, when applied to a model and a collection of test instances, creates a collection class, `CalibratedExplanations`, holding `CalibratedExplanation` objects, which are either `FactualExplanation` or `CounterfactualExplanation` objects. Operations can be applied to all explanations in the collection directly through `CalibratedExplanations` or through each individual `CalibratedExplanation` (see the [documentation](https://calibrated-explanations.readthedocs.io)).

### Fixes
Numerous. The code has been refactored and improved a lot since the original version. The code is now also tested and documented.