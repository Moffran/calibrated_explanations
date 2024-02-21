# Changelog

## [Unreleased]
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.0...main)
### Features
- Added support for Mondrian explanations, using the `bins` attribute. The `bins` attribute takes a categorical feature of the size of the calibration or test set (depending on context) indicating the category of each instance. For continuous attributes, the `crepes.extras.binning`can be used to define categories through binning.  
- Added `BinaryRegressorDiscretizer` and `RegressorDiscretizer` which are similar to `BinaryEntropyDiscretizer` and `EntropyDiscretizer` in that it uses a decision tree to identify suitable discretizations for numerical features. `explain_factual` and `explain_counterfactual` have been updated to use these discretizers for regression by default. In a future version, the possibility to assign your own discretizer may be removed.
- Updated the [Further reading and citing](https://github.com/Moffran/calibrated_explanations#further-reading-and-citing) section in the README: 
  - Updated the reference and bibtex to the published version of the introductory paper:
    - Löfström, H., Löfström, T., Johansson, U., and Sönströd, C. (2024). [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://doi.org/10.1016/j.eswa.2024.123154). Expert Systems with Applications, 1-27.

    - ```bibtex
      @article{lofstrom2024calibrated,
        title = 	{Calibrated explanations: With uncertainty information and counterfactuals},
        journal = 	{Expert Systems with Applications},
        pages = 	{123154},
        year = 	{2024},
        issn = 	{0957-4174},
        doi = 	{https://doi.org/10.1016/j.eswa.2024.123154},
        url = 	{https://www.sciencedirect.com/science/article/pii/S0957417424000198},
        author = 	{Helena Löfström and Tuwe Löfström and Ulf Johansson and Cecilia Sönströd},
        keywords = 	{Explainable AI, Feature importance, Calibrated explanations, Venn-Abers, Uncertainty quantification, Counterfactual explanations},
        abstract = 	{While local explanations for AI models can offer insights into individual predictions, such as feature importance, they are plagued by issues like instability. The unreliability of feature weights, often skewed due to poorly calibrated ML models, deepens these challenges. Moreover, the critical aspect of feature importance uncertainty remains mostly unaddressed in Explainable AI (XAI). The novel feature importance explanation method presented in this paper, called Calibrated Explanations (CE), is designed to tackle these issues head-on. Built on the foundation of Venn-Abers, CE not only calibrates the underlying model but also delivers reliable feature importance explanations with an exact definition of the feature weights. CE goes beyond conventional solutions by addressing output uncertainty. It accomplishes this by providing uncertainty quantification for both feature weights and the model’s probability estimates. Additionally, CE is model-agnostic, featuring easily comprehensible conditional rules and the ability to generate counterfactual explanations with embedded uncertainty quantification. Results from an evaluation with 25 benchmark datasets underscore the efficacy of CE, making it stand as a fast, reliable, stable, and robust solution.}
      }
      ```
  - Added [Code and results](https://github.com/tuvelofstrom/calibrating-explanations) for the [Investigating the impact of calibration on the quality of explanations](https://link.springer.com/article/10.1007/s10472-023-09837-2) paper, inspiring the idea behind Calibrated Explanations.
  - Added a bibtex to the software repository:
    - ```bibtex
      @software{Lofstrom_Calibrated_Explanations_2024,
        author = 	{Löfström, Helena and Löfström, Tuwe and Johansson, Ulf and Sönströd, Cecilia and Matela, Rudy},
        license = 	{BSD-3-Clause},
        title = 	{Calibrated Explanations},
        url = 	{https://github.com/Moffran/calibrated_explanations},
        version = 	{v0.3.0},
        month = 	jan,
        year = 	{2024}
      }
      ``` 
  - Updated the [docs/citing.md](https://github.com/Moffran/calibrated_explanations/blob/main/docs/citing.md) with the above changes.
- Added a [CITATION.cff](https://github.com/Moffran/calibrated_explanations/blob/main/CITATION.cff) with citation data for the software repository.
### Fixes
- Extended `__repr__` to include additional fields when `verbose=True`.
- Fixed a minor bug in the example provided in the [README.md](https://github.com/Moffran/calibrated_explanations/blob/main/README.md#classification) and the [getting_started.md](https://github.com/Moffran/calibrated_explanations/blob/main/docs/getting_started.md#classification), as described in issue #26. 
- Added `utils.transform_to_numeric` and a clarification about known limitations in [README.md](https://github.com/Moffran/calibrated_explanations/blob/main/README.md#classification) as a response to issue #28.
- Fixed a minor bug in `FactualExplanation.__plot_probabilistic` that was triggered when no features where to be shown.

## [v0.3.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.0) - 2024-01-02
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.2.3...v0.3.0)
### Features
- Updated to version 1.4.1 of venn_abers. Added `precision=4` to the fitting of the venn_abers model to increase speed. 
- Preparation for weighted categorical rules implemented but not yet activated. 
- Added a state-of-the-art comparison with scripts and notebooks for evaluating the performance of the method in comparison with `LIME` and `SHAP`: see [Classification_Experiment_sota.py](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Experiment_sota.py) and [Classification_Analysis_sota.ipynb](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Analysis_sota.ipynb) for running and evaluating the experiment. Unzip [results_sota.zip](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/results_sota.zip) and run [Classification_Analysis_sota.ipynb](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Analysis_sota.ipynb) to get the results used in the paper [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://arxiv.org/abs/2305.02305).
- Updated the parameters used by `plot_all` and `plot_explanation`.
### Fixes
- Filtered out extreme target values in the quickstart notebook to make the regression examples more realistic. 
- Fixed bugs related to how plots can be saved to file.
- Fixed an issue where add_conjunctions with `max_rule_size=3` did not work.

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
