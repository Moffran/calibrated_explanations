<!-- markdownlint-disable-file -->
# Changelog

## [Unreleased]

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.6.1...main)

### Added

- Interval calibrators now resolve through the plugin registry for both legacy
  and FAST paths, enabling trusted override chains and capturing telemetry about
  the `interval_source`/`proba_source` used for each explanation batch.
- CalibratedExplainer accepts new keyword overrides (`interval_plugin`,
  `fast_interval_plugin`, `plot_style`) and reads environment/pyproject
  fallbacks so operators can configure intervals and plots without code changes.
- Packaged a `ce.plugins` console script with smoke tests covering list/show and
  trust management workflows.

### Changed

- Interval plugin contexts surface FAST reuse hints and metadata to keep
  calibrator reuse efficient while routing through the registry.
- README, developer docs, and contributing guides document the new CLI,
  configuration knobs, and plugin telemetry expectations.

### Docs

- ADR-013 and ADR-015 marked Accepted with implementation notes summarising the
  registry-backed runtime.
- ADR-017/ADR-018 ratified with quick-reference style excerpts in
  `CONTRIBUTING.md` and contributor docs.
- Harmonised `core.validation` docstrings with numpy-style lint guardrails (ADR-018).

### CI

- Shared `.coveragerc` published and the test workflow now enforces
  `--cov-fail-under=80` to meet ADR-019 phase 1 requirements (with gradual increase for each new version).
- Lint workflow surfaces Ruff naming warnings and docstring lint/coverage
  reports, providing guardrails for ADR-017/ADR-018 adoption.

#### Public API updates in v0.7.0
This document summarises the signature adjustments introduced while aligning the
codebase with the new Ruff style baseline. Reference it from the v0.7.0 changelog
when communicating breaking or user-visible updates.
##### Function parameter renames
The following parameters have been renamed across multiple functions and methods:
- X_test → x
- y_test → y

##### Wrapper keyword normalisation
The following `WrapCalibratedExplainer` entry points now strip deprecated alias
arguments after emitting a `DeprecationWarning`:
- `calibrate`
- `explain_factual`
- `explore_alternatives`
- `explain_fast`
- `predict`
- `predict_proba`
Alias keys such as `alpha`, `alphas`, and `n_jobs` are therefore ignored going
forward. Callers must provide the canonical keyword names (`low_high_percentiles`,
`parallel_workers`, etc.) for custom behaviour to take effect.【F:src/calibrated_explanations/core/wrap_explainer.py†L201-L409】【F:src/calibrated_explanations/api/params.py†L16-L70】
##### Explanation plugin toggle
`CalibratedExplainer` now exposes a keyword-only `_use_plugin` flag across all
explanation factories (`explain_factual`, `explore_alternatives`, `explain_fast`,
`explain`, and the `__call__` shorthand). The flag defaults to `True`, enabling
the plugin orchestrator. Pass `_use_plugin=False` to route through the legacy
implementation when needed.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1489-L1665】
##### Conjunction helper parameters
All `add_conjunctions` helpers across explanation containers use the renamed
keyword arguments `n_top_features` and `max_rule_size` (previously exposed as
`num_to_include` and `num_rule_size`). Update downstream code, documentation,
and notebooks accordingly.【F:src/calibrated_explanations/explanations/explanations.py†L460-L501】

### Fixed

- Fixed test helper stubs and plugin descriptors to satisfy Ruff naming guardrails (ADR-017), keeping `ruff check --select N` green.

## [v0.6.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.6.1) - 2025-10-05

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.6.0...v0.6.1)

### Tests

- Added runtime regression coverage to compare plugin-orchestrated factual, alternative, and fast explanations against the legacy `_use_plugin=False` code paths (`tests/integration/core/test_explanation_parity.py`).
- Exercised schema v1 guardrails by asserting that payloads missing required keys are rejected when `jsonschema` is installed (`tests/unit/core/test_serialization_and_quick.py::test_validate_payload_rejects_missing_required_fields`).
- Locked in `WrapCalibratedExplainer` keyword defaults and alias handling when using configuration objects (`tests/unit/core/test_wrap_keyword_defaults.py`).

### Docs

- Documented the v0.6.x hardening checklist covering plugin parity, schema validation, and wrapper default tests in `docs/plugins.md`.

## [v0.6.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.6.0) - 2025-09-04

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.5.1...v0.6.0)

### Highlights (Contract-first)

- Internal domain model for explanations (ADR-008): added `Explanation` and `FeatureRule` types with adapters to preserve legacy dict outputs. No public API changes required; golden outputs unchanged.
- Explanation Schema v1 (ADR-005): shipped a versioned JSON Schema (`schemas/explanation_schema_v1.json`) and utilities for `to_json`/`from_json` with validation. Round-trip tests included.
- Preprocessing policy hooks (ADR-009): wrapper now supports configurable preprocessing with mapping persistence and unseen-category policy; default behavior unchanged for numeric inputs.
- Optional extras split (Phase 2S): declared `viz`, `lime`, `notebooks`, `dev`, `eval` extras and lazy plotting import. Tests that need matplotlib are marked `@pytest.mark.viz` and skipped when extras are absent.
- Docs: new Schema v1 and Migration (0.5.x → 0.6.0) pages; evaluation README and API reference updates.

### Deprecations

- Parameter alias deprecations wired; warnings emitted once per session (removal not before v0.8.0). See migration guide.

### CI

- Added docs build + linkcheck job and a core-only test job without viz extras to ensure core independence.

### Notes

- This release focuses on contract stability and does not change public serialized outputs. Performance features remain behind flags and will arrive in v0.7.x.

### Acknowledgements

We thank community contributors for overlapping PR work and early prototypes that informed the v0.6.0 contract-first implementation (domain model, schema/serialization, preprocessing hooks). Your feedback and ideas helped refine the final design.

Also added explicit credit files:

- AUTHORS.md (Main authors, authors listed in papers)
- CONTRIBUTORS.md (community contributions)

### Maintenance / Phase 1B completion

- Phase 1B concluded: parameter canonicalization and lightweight validation wired at predict/predict_proba boundaries; strict typing with py.typed and targeted mypy overrides; documentation for error handling/validation/params added and linked; removed OnlineCalibratedExplainer and pruned legacy mentions; CI hygiene (branch conditions, perf guard) and repo lint/type gates green.

### Features

- Updated references to the paper [Calibrated Explanations for Regression](https://doi.org/10.1007/s10994-024-06642-8) in README and citing. The paper is now published in Machine Learning Journal.
  - [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U., Sönströd, C., and [Matela, R](https://github.com/rudymatela). (2025). [Calibrated Explanations for Regression](https://doi.org/10.1007/s10994-024-06642-8). Machine Learning 114, 100.
- Plot Style Control: With [a series of commits](https://github.com/Moffran/calibrated_explanations/compare/4cbc4ff410df19a32899071daa2568e8904c2c47...4093496b1d938f3470f2d33715f0af286f239728), a `plot_config.ini` file and a `test_plot_config.py` file have been added. The style parameters are used by the plots. The style parameters for all plot functions can now be overridden using the `style_override` parameter. [Controlling figure width is also added](https://github.com/Moffran/calibrated_explanations/compare/86babfa1afed75fc8959cf072c21e932c3d08f07...e0d13f32907185a144781ff76b553ad5c8cc0f8d).
- Optional extras and lazy plotting: Added optional dependency extras in `pyproject.toml` (`viz` for matplotlib, `lime` for LIME). Made matplotlib a lazy optional import used only when plotting is invoked, with a friendly runtime hint to install `calibrated_explanations[viz]`. Updated README with installation examples.

### Breaking Changes

- Removed the experimental `OnlineCalibratedExplainer` and its tests/docs. All references were purged from code, docs, configs, and packaging metadata.

### Fixes

- [fix: ensure figures are closed when not shown in plotting functions](https://github.com/Moffran/calibrated_explanations/commit/f20a047b2c4acb0eae6b5f6aed876f2db7d4d389)

## [v0.5.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.5.1) - 2024-11-27

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.5.0...v0.5.1)

### Features

- String Targets Support: Added support for string targets, enhancing flexibility in handling diverse datasets. Special thanks to our new contributor [ww-jermaine](https://github.com/ww-jermaine) for the efforts on this feature ([issue #27](https://github.com/Moffran/calibrated_explanations/issues/27)).
- Out-of-Bag Calibration: Introduced support for out-of-bag calibration when using random forests from `sklearn`, enabling improved calibration techniques directly within ensemble models. See the new [notebook](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/quickstart_wrap_oob.ipynb) for examples.
- Documentation Enhancements: Updated and refined [documentation](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest), including fixes to existing sections and the addition of doctests for helper functions to ensure accuracy and reliability.
- Minor updates: Added a `calibrated` parameter to the `predict` and `predict_proba` methods to allow uncalibrated results.

### Fixes

- Bug Fixes: Resolved multiple bugs to enhance stability and performance across the library.


## [v0.5.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.5.0) - 2024-10-15

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.4.0...v0.5.0)

### Features

- Improved the introduction in README.
- Added `calibrated_confusion_matrix` in `CalibratedExplainer` and `WrapCalibratedExplainer`, providing a leave-one-out calibrated confusion matrix using the calibration set. The insights from the confusion matrix are useful when analyzing explanations, to determine general prediction and error distributions of the model. An example of using the confusion matrix in the analysis is given in paper [Calibrated Explanations for Multi-class](https://raw.githubusercontent.com/mlresearch/v230/main/assets/lofstrom24a/lofstrom24a.pdf).
- Embraced the update of `crepes` version 0.7.1, making it possible to add a seed when fitting. Addresses issue #43.
- Updating terminology and functionality:
  - Introducing the concept of _ensured_ explanations.
    - Changed the name of `CounterfactualExplanation` to `AlternativeExplanation`, as it better reflects the purpose and functionality of the class.
    - Added a collection subclass `AlternativeExplanations` inheriting from `CalibratedExplanations`, which is used for collections of `AlternativeExplanation`'s. Collection methods referring to methods only available in the `AlternativeExplanation` are included in the new collection class.
    - Added an `explore_alternatives` method in `CalibratedExplainer` and `WrapCalibratedExplainer` to be used instead of `explain_counterfactual`, as the name of the later is ambiguous. The `explain_counterfactual` is still kept for compatibility reasons but only forwards the call to `explore_alternatives`. All files and notebooks have been updated to only call `explore_alternatives`. All references to counterfactuals have been changed to alternatives, with obvious exceptions.
    - Added both filtering methods and a ranking metric that can help filter out ensured explanations.
      - The parameters `rnk_metric` and `rnk_weight` has been added to the plotting functions and is applicable to all kinds of plots.
      - Both the `AlternativeExplanation` class (for an individual instance) and the collection subclass `AlternativeExplanations` contains filter functions only applicable to alternative explanations, such as `counter_explanations`, `semi_explanations`, `super_explanations`, and `ensured_explanations`.
        - `counter_explanations` removes all alternatives except those changing prediction.
        - `semi_explanations` removes all alternatives except those reducing the probability while not changing prediction.
        - `super_explanations` removes all alternatives except those increasing the probability for the prediction.
        - The concept of potential (uncertain) explanations is introduced. When the uncertainty interval spans across probability 0.5, an explanation is considered a potential. It will normally only be counter-potential and semi-potential, but can in some cases also be super-potential. Potential alternatives can be included or excluded from the above methods using the boolean parameter `include_potentials`.
        - `ensured_explanations` removes all alternatives except those with lower uncertainty (i.e. smaller uncertainty interval) than the original prediction.
    - Added a new form of plot for probabilistic predictions is added, clearly visualizing both the aleatoric and the epistemic uncertainty.
      - A global plot is added, plotting all test instances with probability and uncertainty as the x- and y-axes. The area corresponding to potential (uncertain) predictions is marked. The plot can be invoked using the `plot(X_test)` or `plot(X_test, y_test)` call.
      - A local plot for alternative explanations, with probability and uncertainty as the x- and y-axes, is added, which can be invoked from an `AlternativeExplanation` or a `AlternativeExplanations` using `plot(style='triangular')`. The optimal use is when combined with the `filter_top` parameter (see below), to include all alternatives, as follows: `plot(style='triangular', filter_top=None)`.
    - Added prerpint and bibtex to the paper introducing _ensured_ explanations:
      - [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., and [Hallberg Szabadvary, J](https://github.com/egonmedhatten). (2024). [Ensured: Explanations for Decreasing the Epistemic Uncertainty in Predictions](https://arxiv.org/abs/2410.05479). arXiv preprint arXiv:2410.05479.
      - Bibtex:

        ```bibtex
        @misc{lofstrom2024ce_ensured,
          title =        {Ensured: Explanations for Decreasing the Epistemic Uncertainty in Predictions},
          author =          {L\"ofstr\"om, Helena and L\"ofstr\"om, Tuwe and Hallberg Szabadvary, Johan},
          year =            {2024},
          eprint =          {2410.05479},
          archivePrefix =   {arXiv},
          primaryClass =    {cs.LG}
        }
        ```

  - Introduced _fast_ explanations
    - Introduced a new type of explanation called `FastExplanation` which can be extracted using the `explain_fast` method. It differs from a `FactualExplanation` in that it does not define a rule condition but only provides a feature weight.
    - The new type of explanation is using ideas from [ConformaSight](https://github.com/rabia174/ConformaSight), a recently proposed global explanation algorithm based on conformal classification. Acknowledgements have been added.
  - Introduced a new form av probabilistic regression explanation:
    - Introduced the possibility to get explanations for the probability of being inside an interval. This is achieved by assigning a tuple with lower and upper bounds as threshold, e.g., `threshold=(low,high)` to get the probability of the prediction falling inside the interval (low, high].
    - To the best of our knowledge, this is the only package that provide this functionality with epistemic uncertainty.
  - Introduced the possibility to add new user defined rule conditions, using the `add_new_rule_condition` method. This is only applicable to numerical features.
    - Factual explanations will create new conditions covering the instance value. Categorical features already get a condition for the instance value during the invocation of `explain_factual`.
    - Alternative explanations will create new conditions that exclude the instance value. Categorical features already get conditions for all alternative categories during the invocation of `explore_alternatives`.
  - Parameter naming:
    - The parameter indicating the number of rules to plot is renamed to `filter_top` (previously `n_features_to_show`), making the call including all rules (`filter_top=None`) makes a lot more sense.

### Fixes

- Added checks to ensure that the learner is not called unless the `WrapCalibratedExplainer` is fitted.
- Added checks to ensure that the explainer is not called unless the `WrapCalibratedExplainer` is calibrated.
- Fixed incorrect use of `np.random.seed`.

## [v0.4.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.4.0) - 2024-08-23
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.5...v0.4.0)
### Features
- Paper updates:
  - [Calibrated Explanations for Regression](https://arxiv.org/abs/2308.16245) has been accepted to Machine Learning. It is currently in press.
- Code improvements:
  - __Substantial speedup__ achieved through the newly implemented `explain` method! This method implements the core algorithm while minimizing the number of calls to core._predict, substantially speeding up the code without altering the algorithmic logic of `calibrated_explanations`. The `explain` method is used exclusively from this version on when calling `explain_factual` or `explain_counterfactual`.
    - Re-ran the ablation study for classification, looking at the impact of calibration set size, number of percentile samplings for numeric features and the number of features.
      - Uploaded a pdf version of the [ablation study](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Calibrated_Explanations_Ablation.pdf), making the results easier to overview.
    - Re-ran the evaluation for regression, measuring stability, robustness and running times with and without normalization.
  - Improved the `safe_import` to allow `import ... from ...` constructs.
  - Restructured package
    - Added a utils folder:
      - Moved discretizers.py to utils
      - Moved utils.py to utils and renamed to helper.py
    - Made explanations public
    - Made VennAbers and interval_regressor restricted
  - Experimental functionality introduced:
    - Several new experimental features have been introduced. These will be presented as Features once they are thoroughly tested and evaluated.
- Code interface improvements:
  - Added support for the `MondrianCategorizer` from crepes in the `WrapCalibratedExplainer`.
  - Added wrapper functions in `WrapCalibratedExplainer` redirecting to `CalibratedExplainer`:
    - Including `predict`, `predict_proba`, and `set_difficulty_estimator`.
    - Moved any remaining implementations of functions in `WrapCalibratedExplainer` to `CalibratedExplainer`.
  - Renamed the `plot_all` and `plot_explanation` functions to `plot`. Updated all usages of the `plot` function.
  - Added `__len__` and `__getitem__` to `CalibratedExplanations`.
    - `__getitem__` allow indexing with `int`, `slice`, and lists (both boolean and integer lists). When more than one explanation is retrieved, a new `CalibratedExplanations` is returned, otherwise, the indexed `CalibratedExplanation` is returned.
- Documentation improvements:
  - Restructured and extended the [documentation](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest).
    - Updated the information at the entry page
    - Added an API reference
- Improvements of the CI setup:
  - Updated CI to run pytest before pylint.
  - Updated CI to avoid running tests when commit message starts with 'info:' or 'docs:'.
- Testing improvements
  - Improved tests to test `predict` and `predict_proba` functions in `CalibratedExplainer` better.
  - Added several other tests to increase [coverage](https://app.codecov.io/github/Moffran/calibrated_explanations).
### Fixes
- Fixed minor errors in the `predict` and `predict_proba` functions in `CalibratedExplainer`.
- Several other minor bug fixes have also been made.

## [v0.3.5](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.5) - 2024-07-24
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.4...v0.3.5)
### Features
- Made several improvements of the `WrapCalibratedExplainer`:
    - `WrapCalibratedExplainer` is introduced as the default way to interact with `calibrated-explanations` in README.md. The benefit of having a wrapper class as interface is that it makes it easier to add different kinds of explanations.
    - Documentation of the functions has been updated.
    - Initialization:
      - The `WrapCalibratedExplainer` can now be initialized with an unfitted model as well as with a fitted model.
      - The `WrapCalibratedExplainer` can now be initialized with an already initialized `CalibratedExplainer` instance, providing access to the `predict` and `predict_proba` functions.
    - The `fit` method will reinitialize the explainer if the `WrapCalibratedExplainer` has already been calibrated, to ensure that the `explainer` is adapted to the re-fitted model.
    - Added improved error handling.
    - Made several other minor quality improving adjustments.
- Code coverage tests are added and monitored at [Codecov](https://app.codecov.io/github/Moffran/calibrated_explanations).
  - Tests are added in order to increase code coverage.
  - Unused code is cleaned up.
- Updated the [Further reading and citing](https://github.com/Moffran/calibrated_explanations#further-reading-and-citing) section in the README:
  - Added a reference and bibtex to:
    - [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U. (2024). [Calibrated Explanations for Multi-class](https://easychair.org/publications/preprint/rqdD). <i>Proceedings of the Thirteenth Workshop on Conformal and Probabilistic Prediction and Applications</i>, in <i>Proceedings of Machine Learning Research</i>. In press.
    - ```bibtex
      @Booklet{lofstrom2024ce_multiclass,
        author = {Tuwe Löfström and Helena Löfström and Ulf Johansson},
        title = {Calibrated Explanations for Multi-Class},
        howpublished = {EasyChair Preprint no. 14106},
        year = {EasyChair, 2024}
      }
      ```
  - Updated the [docs/citing.md](https://github.com/Moffran/calibrated_explanations/blob/main/docs/citing.md) with the above changes.
### Fixes
- Discretizers are limited to the default alternatives for classification and regression. BinaryDiscretizer removed. `__repr__` functions added.
- Changed the `check_is_fitted` function to remove ties to sklearn.
- Made the `safe_import` throw an `ImportError` when an import fail.

## [v0.3.4](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.4) - 2024-07-10
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.3...v0.3.4)
### Features
- Updated the [Further reading and citing](https://github.com/Moffran/calibrated_explanations#further-reading-and-citing) section in the README:
  - Added a reference and bibtex to:
    - [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom). (2024). [Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty](https://doi.org/10.1007/978-3-031-63787-2_17). In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2153. Springer, Cham.
    - ```bibtex
      @InProceedings{lofstrom2024ce_conditional,
      author="L{\"o}fstr{\"o}m, Helena
      and L{\"o}fstr{\"o}m, Tuwe",
      editor="Longo, Luca
      and Lapuschkin, Sebastian
      and Seifert, Christin",
      title="Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty",
      booktitle="Explainable Artificial Intelligence",
      year="2024",
      publisher="Springer Nature Switzerland",
      address="Cham",
      pages="332--355",
      abstract="While Artificial Intelligence and Machine Learning models are becoming increasingly prevalent, it is essential to remember that they are not infallible or inherently objective. These models depend on the data they are trained on and the inherent bias of the chosen machine learning algorithm. Therefore, selecting and sampling data for training is crucial for a fair outcome of the model. A model predicting, e.g., whether an applicant should be taken further in the job application process, could create heavily biased predictions against women if the data used to train the model mostly contained information about men. The well-known concept of conditional categories used in Conformal Prediction can be utilised to address this type of bias in the data. The Conformal Prediction framework includes uncertainty quantification methods for classification and regression. To help meet the challenges of data sets with potential bias, conditional categories were incorporated into an existing explanation method called Calibrated Explanations, relying on conformal methods. This approach allows users to try out different settings while simultaneously having the possibility to study how the uncertainty in the predictions is affected on an individual level. Furthermore, this paper evaluated how the uncertainty changed when using conditional categories based on attributes containing potential bias. It showed that the uncertainty significantly increased, revealing that fairness came with a cost of increased uncertainty.",
      isbn="978-3-031-63787-2"
      }
      ```
  - Updated the [docs/citing.md](https://github.com/Moffran/calibrated_explanations/blob/main/docs/citing.md) with the above changes.
### Fixes
- Changed np.Inf to np.inf for compatibility reasons (numpy v2.0.0).
- Updated requirements for numpy and crepes to include versions v2.0.0 and v0.7.0, respecitvely.

## [v0.3.3](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.3) - 2024-05-25
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.2...v0.3.3)
### Features
- Changed how probabilistic regression is done to handle both validity and speed by dividing the calibration set into two sets to allow pre-computation of the CPS. Credits to anonymous reviewer for this suggestion.
- Added updated regression experiments and plotting for revised paper.
- Added a new `under the hood` demo notebook to show how to access the information used in the plots,  like conditions and uncertainties etc.
### Fixes
- Several minor updates to descrptions and notebooks in the repository.

## [v0.3.2](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.2) - 2024-04-14
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.1...v0.3.2)
### Features
- Added Fairness experiments and plotting for the XAI 2024 paper. Added a `Fairness` tag for the weblinks.
- Added multi-class experiments and plotting for upcoming submissions. Added a `Multi-class` tag for weblinks.
- Some improvements were made to the multi-class functionality. The updates included updating the VennAbers class to a more robust handling of multi-class (with or without Mondrian bins).
### Fixes
- Updated the requirement for crepes to v0.6.2, to address known issues with some versions of python.
- The pythonpath for pytest was added to pyprojects.toml to avoid module not found error when running pytest locally.

## [v0.3.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.1) - 2024-02-23
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.0...v0.3.1)
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
        version = 	{v0.3.1},
        month = 	feb,
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
- Fixed a bug with the discretizers in `core`.
- Fixed a bug with saving plots to file using the `filename` parameter.

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
