# Changelog

## [Unreleased]
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.2.3...main)
### Features
- Updated to version 1.4.1 of venn_abers. Added `precision=4` to the fitting of the venn_abers model to increase speed. 
- Preparation for weighted categorical rules implemented but not yet activated.
### Fixes
- Filtered out extreme target values in the quickstart notebook to make the regression examples more realistic. 

## [v0.2.3](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.2.3) - 2023-11-04
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.2.2...v0.2.3)
### Features
- Added an evaluation folder with scripts and notebooks for evaluating the performance of the method.
  - One evaluation focuses on stability and robustness of the method: see [Classification_Experiment_stab_rob.py](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Experiment_stab_rob.py) and [Classification_Analysis_stab_rob.ipynb](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Analysis_stab_rob.ipynb) for running and evaluating the experiment.
  - One evaluation focuses on how different parameters affect the method regarding time and robustness: see [Classification_Experiment_Ablation.py](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Experiment_Ablation.py) and [Classification_Analysis_Ablation.ipynb](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Analysis_Ablation.ipynb) for running and evaluating the experiment.

### Fixes
- Fix in `CalibratedExplainer` to ensure that greater-than works identical as less-than.
- Bugfix in `FactualExplanation._get_rules()` which caused an error when categorical labels where missing.

## [v0.2.2](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.2.2) - 2023-10-03
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.2.1...v0.2.2)
### Fixes
Smaller adjustments and fixes.

## [v0.2.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.2.1) - 2023-09-20
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.2.0...v0.2.1)
### Fixes
The wrapper file with helper classes `CalibratedAsShapExplainer` and `CalibratedAsLimeTabularExplanainer` has been removed. The `as_shap` and `as_lime` functions are still working.

## [v0.2.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.2.0) - 2023-09-19
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.1.1...v0.2.0)
### Features
- Added a `WrapCalibratedExplainer` class which can be used for both classificaiton and regression.
- Added [quickstart_wrap](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/quickstart_wrap.ipynb) to the notebooks folder.
- Added [LIME_comparison](https://github.com/Moffran/calibrated_explanations/notebooks/LIME_comparison.ipynb) to the notebooks folder. 
### Fixes
- Removed the dependency on `shap` and `scikit-learn` and closed issue #8.
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
