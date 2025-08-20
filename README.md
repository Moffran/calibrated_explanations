# Calibrated Explanations ([Documentation](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest))
<!-- ======================= -->

[![Calibrated Explanations PyPI version][pypi-version]][calibrated-explanations-on-pypi]
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/calibrated-explanations.svg)](https://anaconda.org/conda-forge/calibrated-explanations)
[![GitHub (Pre-)Release Date](https://img.shields.io/github/release-date-pre/Moffran/calibrated_explanations)](https://github.com/Moffran/calibrated_explanations/blob/main/CHANGELOG.md)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/moffran/calibrated_explanations/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/calibrated-explanations)](https://pepy.tech/project/calibrated-explanations)
<!-- [![Documentation Status](https://readthedocs.org/projects/calibrated-explanations/badge/?version=latest)](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest)
[![Build Status for Calibrated Explanations][build-status]][build-log] -->
<!-- [![Lint Status for Calibrated Explanations][lint-status]][lint-log] -->
<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Moffran/calibrated_explanations/main?urlpath=https%3A%2F%2Fgithub.com%2FMoffran%2Fcalibrated_explanations%2Fblob%2Fmain%2Fnotebooks%2Fquickstart.ipynb) -->

## Table of Contents
- [Calibrated Explanations (Documentation)](#calibrated-explanations-documentation)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Core Features:](#core-features)
    - [Distinctive Characteristics](#distinctive-characteristics)
    - [Example Explanation](#example-explanation)
  - [Quick Start](#quick-start)
  - [Usage](#usage)
    - [Classification](#classification)
      - [Factual Explanations](#factual-explanations)
      - [Explore Alternative Explanations](#explore-alternative-explanations)
    - [Regression](#regression)
      - [Factual Explanations](#factual-explanations-1)
      - [Explore Alternative Explanations](#explore-alternative-explanations-1)
    - [Probabilistic Regression](#probabilistic-regression)
    - [Initializing `WrapCalibratedExplainer`s](#initializing-wrapcalibratedexplainers)
    - [Known Limitations](#known-limitations)
  - [Installation](#installation)
    - [From PyPI:](#from-pypi)
    - [From conda-forge:](#from-conda-forge)
    - [Dependencies:](#dependencies)
  - [Contributing](#contributing)
  - [Documentation](#documentation)
  - [License](#license)
  - [Further Reading and Citing](#further-reading-and-citing)
    - [Published Papers](#published-papers)
    - [Preprints:](#preprints)
    - [Citing and Bibtex](#citing-and-bibtex)
  - [Acknowledgements](#acknowledgements)
  - [Support](#support)

## Introduction

**Calibrated Explanations** is an explanation method for machine learning designed to enhance both the interpretability of model predictions and the quantification of uncertainty. In many real-world applications, understanding how confident a model is about its predictions is just as important as the predictions themselves. This framework provides calibrated explanations for both predictions and feature importance by minimizing **aleatoric** and quantifying **epistemic uncertainty** — two types of uncertainty that offer critical insights into both data and model reliability.

- **Aleatoric uncertainty** represents the noise inherent in the data. It affects the spread of probability distributions (for probabilistic outcomes) and predictions (for regression). This uncertainty is **irreducible** because it reflects limitations in the data generation process itself. Incorporating calibration ensures accurate decision support. The difference between the calibrated and the uncalibrated predictions is a measure of the aleatoric uncertainty.

- **Epistemic uncertainty** arises from the model's lack of knowledge due to limited training or calibration data or insufficient complexity. It affects the model’s confidence in its output when it encounters unfamiliar or out-of-distribution data. Unlike aleatoric uncertainty, epistemic uncertainty is **reducible** — it can be minimized by gathering more data, improving the model architecture, or refining features.

Through calibration of the model prediction, **Calibrated Explanations** minimize the aleatoric uncertainty. By providing an uncertainty interval for the prediction and the feature importance estimates, it highlights the epistemic uncertainty for individual instances. This offers a more comprehensive and accurate support for decision making, both in terms of accuracy and confidence. This is particularly valuable in high-stakes environments where model reliability and interpretability are essential, such as in healthcare, finance, and autonomous systems.

To get a very condensed example on how Calibrated Explanations can be used, see the [Quick Start](#quick-start). For an in-depth guide on how to start using Calibrated Explanations, refer to the [Usage](#usage) section below.

### Core Features:
- **Calibrated Prediction Confidence**: Obtain well-calibrated uncertainty estimates for predictions, helping users make informed decisions based on the model’s confidence.
- **Uncertainty-Aware Feature Importance**: Understand not only which features are important but also how uncertain the model is about the contribution of those features.
- **Support for Various Tasks**: The framework supports classification, regression, and probabilistic regression, making it adaptable to a wide range of machine learning problems.

The ability to quantify both aleatoric and epistemic uncertainty provides practitioners with actionable insights into the reliability of predictions and explanations, fostering **appropriate trust** ([read paper](https://ieeexplore.ieee.org/abstract/document/10487244)) and transparency in machine learning models.

### Distinctive Characteristics

Calibrated Explanations offers a range of features designed to enhance both the interpretability and reliability of machine learning models. These characteristics can be summarized as follows:

* **Fast, reliable, stable, and robust feature importance explanations** for:
  - **Binary classification models** ([Read paper](https://doi.org/10.1016/j.eswa.2024.123154)).
  - **Multi-class classification models** ([Read paper](https://raw.githubusercontent.com/mlresearch/v230/main/assets/lofstrom24a/lofstrom24a.pdf), [Slides](https://copa-conference.com/presentations/Lofstrom.pdf)).
  - **Regression models** ([Read paper](https://arxiv.org/abs/2308.16245)), including:
    - **Probabilistic explanations**: Provides the probability that the target exceeds a user-defined threshold.
    - **Difficulty-adaptable explanations**: Adjust explanations based on conformal normalization for varying levels of data difficulty.

* **Aleatoric and epistemic uncertainty estimates**: These estimates are provided by **Venn-Abers** for probabilistic explanations and by **Conformal Predictive Systems** for regression tasks. Both these techniques are grounded in solid theoretical foundations, leveraging **conformal prediction** and **Venn prediction** to ensure reliability and robustness in uncertainty quantification.

* **Calibration of the underlying model**: Ensures that predictions accurately reflect reality, improving trust in model outputs.

* **Comprehensive uncertainty quantification**:
  - **Prediction uncertainty**: Quantifies both aleatoric and epistemic uncertainties for the model’s predictions.
  - **Feature importance uncertainty**: Measures uncertainty in feature importance scores, helping to assess the reliability of each feature's contribution.

* **Proximity-based rules for straightforward interpretation**: Generates rules that are easily interpretable by relating instance values to feature importance weights.

* **Alternative explanations with uncertainty quantification**: Provides explanations for how predicted outcomes would change if specific input features were modified, including uncertainty estimates for these alternative outcomes.
  - **Ensured Explanations**: Ensured explanations aims to help users find alternative explanations that reduce epistemic uncertainty (read more in the [changelog](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.5.0) or [read the paper](https://arxiv.org/abs/2410.05479)). This includes:
    - Categories for uncertain explanations, such as counter-potential, semi-potential, and super-potential.
    - A new ranking metric, called _ensured_ ranking, to help balance uncertainty and probability among alternative explanations.
    - A new plot to help visualize uncertainties among alternative explanations.

* **Conjunctional rules**: Provides feature importance explanations for interactions between multiple features, highlighting joint contributions (discussed in detail in the [regression paper](https://arxiv.org/abs/2308.16245)).

* **Conditional rules for contextual explanations**: Allows users to create explanations conditioned on specific criteria, enabling better handling of e.g. fairness and bias constraints ([Read paper](https://doi.org/10.1007/978-3-031-63787-2_17)). Using conformal terminology, this means that Mondrian categories are supported.

### Example Explanation
Below is an example of an alternative explanation for an instance from the Wine classification dataset. The triangular plots (subfigures (a) and (b)) show the prediction and uncertainty for the instance (red) and alternatives (blue). The uncertainty is represented by the width of the probability interval, shown in subfigure (c). The greyed area cannot be populated, as the probability is a regularised mean of the interval endpoints (the regularisation explains the curved shape). All explanations falling inside the rhombus-like shape in the middle are uncertain since their uncertainty interval covers probability 0.5, indicating that they can potentially be either positive or negative.

* **Subfigure (a)**: Displays all generated alternatives, including conjunctive alternative explanations, indicating the outcome when changing feature value(s).
* **Subfigures (b) and (c)**: Show a subset of ten alternative explanations, filtered by probability estimate (`rnk_metric='ensured'` and `rnk_weight=1`). The light red area in the background of (c) represents the calibrated probability interval on the original instance for the positive class, corresponding to the red dot in (a) and (b).

The darker red bars for each rule (left) show the probability intervals provided by Venn-Abers, indicating how the likelihood of the outcome changes when original instance values (right) are modified according to the rule conditions. As an example, if _Chloride_ is reduced to below 0.04 (instead of the instance value 0.06), the probability for the positive class is increased and the uncertainty is decreased.
<p align="center">
  <a href="https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest">
    <!-- <img src="https://github.com/Moffran/calibrated_explanations/blob/main/docs/images/counterfactual_probabilistic_house_regression.jpg" alt="Probabilistic alternative explanation for California Housing"> -->
    <img src="https://github.com/Moffran/calibrated_explanations/blob/main/docs/images/alternatives_wine_ensured.png" alt="Alternative explanations for Wine (classification)">
  </a>
</p>

[Table of Content](#table-of-contents)

## Quick Start
Here is a very condensed example to get you started:

```python
   from calibrated_explanations import WrapCalibratedExplainer
   # Load and pre-process your data
   # Divide it into proper training, calibration, and test sets

   # Initialize the WrapCalibratedExplainer with your model
   classifier = WrapCalibratedExplainer(ClassifierOfYourChoice())
   regressor = WrapCalibratedExplainer(RegressorOfYourChoice())

   # Train your model using the proper training set
   classifier.fit(X_proper_training_cls, y_proper_training_cls)
   regressor.fit(X_proper_training_reg, y_proper_training_reg)

   # Initialize the CalibratedExplainer
   classifier.calibrate(X_calibration_cls, y_calibration_cls)
   regressor.calibrate(X_calibration_reg, y_calibration_reg)

   # Factual Explanations
   # Create factual explanations for classification
   factual_explanations = classifier.explain_factual(X_test_cls)
   # Create factual standard explanations for regression with default 90 % uncertainty interval
   factual_explanations = regressor.explain_factual(X_test_reg) # low_high_percentiles=(5,95)
   # Create factual standard explanations for regression with user assigned uncertainty interval
   factual_explanations = regressor.explain_factual(X_test_reg, low_high_percentiles=(10,90))
   # Create factual probabilistic explanations for regression with user assigned threshold
   your_threshold = 1000
   factual_explanations = regressor.explain_factual(X_test_reg, threshold=your_threshold)

   # Alternative Explanations
   # Create alternative explanations for classification
   alternative_explanations = classifier.explore_alternatives(X_test_cls)
   # Create alternative standard explanations for regression with default 90 % uncertainty interval
   alternative_explanations = regressor.explore_alternatives(X_test_reg) # low_high_percentiles=(5,95)
   # Create alternative standard explanations for regression with user assigned uncertainty interval
   alternative_explanations = regressor.explore_alternatives(X_test_reg, low_high_percentiles=(10,90))
   # Create alternative probabilistic explanations for regression with user assigned threshold
   alternative_explanations = regressor.explore_alternatives(X_test_reg, threshold=your_threshold)

   # Plot the explanations, works the same for classification and regression
   factual_explanations.plot()
   factual_explanations.plot(uncertainty=True)
   alternative_explanations.plot()

   # Add conjunctions to the explanations, works the same for classification and regression
   factual_conjunctions.add_conjunctions()
   alternative_conjunctions.add_conjunctions()

   # One-sided and asymmetric explanations for regression are easily created
   factual_upper_bounded = regressor.explain_factual(X_test_reg, low_high_percentiles=(-np.inf,90))
   alternative_lower_bounded = regressor.explore_alternatives(X_test_reg, low_high_percentiles=(10,np.inf))
   alternative_asymmetric = regressor.explore_alternatives(X_test_reg, low_high_percentiles=(10,70))
```
It is easy to access the predictions and probabilities from the calibrator and model.

```python
   # Train your model using the proper training set
   classifier.fit(X_proper_training_cls, y_proper_training_cls)
   regressor.fit(X_proper_training_reg, y_proper_training_reg)

   # Output the model predictions and probabilities (without calibration)
   uncal_proba_cls = classifier.predict_proba(X_test_cls)
   uncal_y_hat_cls = classifier.predict(X_test_cls)
   uncal_y_hat_reg = regressor.predict(X_test_reg)

   # Initialize the CalibratedExplainer
   classifier.calibrate(X_calibration_cls, y_calibration_cls)
   regressor.calibrate(X_calibration_reg, y_calibration_reg)

   # Output the model predictions and probabilities (without calibration).
   uncal_proba_cls = classifier.predict_proba(X_test_cls, calibrated=False)
   uncal_y_hat_cls = classifier.predict(X_test_cls, calibrated=False)
   uncal_y_hat_reg = regressor.predict(X_test_reg, calibrated=False)

   # Output the calibrated predictions and probabilities
   calib_proba_cls = classifier.predict_proba(X_test_cls)
   calib_y_hat_cls = classifier.predict(X_test_cls)
   calib_y_hat_reg = regressor.predict(X_test_reg)
   # Get thresholded regression predictions and probabilities for labels 'y_hat > threshold' and 'y_hat <= threshold'
   your_threshold = 1000
   thrld_y_hat_reg = regressor.predict(X_test_reg, threshold=your_threshold)
   thrld_proba_reg = regressor.predict_proba(X_test_reg, threshold=your_threshold)

   # Include uncertainty interval, outputted as a tuple (low, high)
   calib_proba_cls, low_high = classifier.predict_proba(X_test_cls, uq_interval=True)
   calib_y_hat_cls, (low, high) = classifier.predict(X_test_cls, uq_interval=True)
   calib_y_hat_reg, low_high = regressor.predict(X_test_reg, uq_interval=True) # default low_high_percentiles=(5, 95)
   calib_y_hat_reg, low_high = regressor.predict(X_test_reg, low_high_percentiles=(10,90), uq_interval=True)
   thrld_y_hat_reg, low_high = regressor.predict(X_test_reg, threshold=your_threshold, uq_interval=True)
   thrld_proba_reg, (low, high) = regressor.predict_proba(X_test_reg, threshold=your_threshold, uq_interval=True)
```

[Table of Content](#table-of-contents)

## Usage
The [notebooks folder](https://github.com/Moffran/calibrated_explanations/tree/main/notebooks) contains a number of notebooks illustrating different use cases for `calibrated-explanations`. The [quickstart_wrap](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/quickstart_wrap.ipynb) notebook is similar to this section in structure and includes plots and output.

The notebooks listed below are using the `CalibratedExplainer` class. They showcase a number of different use cases, as indicated by their names:
* [quickstart](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/quickstart.ipynb) - similar to this Getting Started, but without a wrapper class.
* [demo_binary_classification](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_binary_classification.ipynb) - with examples for binary classification
* [demo_multiclass](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_multiclass_glass.ipynb) - with examples for multi-class classification
* [demo_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_regression.ipynb) - with examples for regression
* [demo_probabilistic_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_probabilistic_regression.ipynb) - with examples for regression with thresholds
* [demo_under_the_hood](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_under_the_hood.ipynb) - illustrating how to access the information composing the explanations

### Classification
Let us illustrate how we may use `calibrated_explanations` to generate explanations from a classifier trained on a dataset from
[www.openml.org](https://www.openml.org), which we first split into a
training and a test set using `train_test_split` from
[sklearn](https://scikit-learn.org), and then further split the
training set into a proper training set and a calibration set:


```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

dataset = fetch_openml(name="wine", version=7, as_frame=True, parser='auto')

X = dataset.data.values.astype(float)
y = (dataset.target.values == 'True').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, stratify=y)

X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train,
                                                            test_size=0.25)

```

We now create our wrapper object, using a `RandomForestClassifier` as learner.


```python
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer, __version__

print(f"calibrated_explanations {__version__}")

classifier = WrapCalibratedExplainer(RandomForestClassifier())
display(classifier)
```

We now fit our model using the proper training set.


```python
classifier.fit(X_prop_train, y_prop_train)
display(classifier)
```

The `WrapCalibratedExplainer` class has a `predict` and a `predict_proba` method that returns the predictions and probability estimates of the underlying classifier. If the model is not yet calibrated, then the underlying models `predict` and `predict_proba` methods are used. If the model is calibrated, then the `predict` and `predict_proba` method of the calibration model is used.


```python
predict = classifier.predict(X_test)
proba = classifier.predict_proba(X_test)
print('Uncalibrated prediction (probability estimates):')
print(f'{predict} ({proba})')
```

Before we can generate explanations, we need to calibrate our model using the calibration set.


```python
classifier.calibrate(X_cal, y_cal)
display(classifier)
```

Once the model is calibrated, the `predict` and `predict_proba` methods produce calibrated predictions and probability estimates.


```python
predict = classifier.predict(X_test)
proba, (low, high) = classifier.predict_proba(X_test, uq_interval=True)
print('Calibrated prediction (probability estimates):')
print(f'{predict} ({proba})')
print('Calibrated uncertainty interval for the positive class:')
print([(low[i], high[i]) for i in range(len(low))])
```

#### Factual Explanations
Let us explain a test instance using our `WrapCalibratedExplainer` object. The method used to get factual explanations is `explain_factual`.


```python
factual_explanations = classifier.explain_factual(X_test)
display(classifier)
```

Once we have the explanations, we can plot all of them using the `plot` function. Default, a regular plot, without uncertainty intervals included, is created. To include uncertainty intervals, change the parameter `uncertainty=True`.


```python
factual_explanations.plot()
factual_explanations.plot(uncertainty=True)
```

You can also add and remove conjunctive rules.


```python
factual_explanations.add_conjunctions().plot(uncertainty=True)
factual_explanations.remove_conjunctions().plot(uncertainty=True)
```

All explanations support indexing using integer indexing, slices or boolean indexing.


```python
factual_explanations[0].plot(uncertainty=True)
factual_explanations[:1].plot(uncertainty=True)
```

#### Explore Alternative Explanations
An alternative to factual rules is to extract alternative rules, which is done using the `explore_alternatives` function. Alternative explanations provides insights on how predicted outcomes would change if specific input features were modified, including uncertainty estimates for these alternative outcomes.


```python
alternative_explanations = classifier.explore_alternatives(X_test)
display(classifier)
```

Alternatives are also visualized using the `plot` function. Plotting an individual alternative explanation is easiest done using indexing. Adding or removing conjunctions is done as before.


```python
alternative_explanations.plot()
alternative_explanations.add_conjunctions().plot()
```

`calibrated_explanations` supports multiclass which is demonstrated in [demo_multiclass](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_multiclass_glass.ipynb). That notebook also demonstrates how both feature names and target and categorical labels can be added to improve the interpretability.

### Regression
Extracting explanations for regression is very similar to how it is done for classification. First we load and divide the dataset. The target is divided by 1000, meaning that the target is in thousands of dollars.


```python
dataset = fetch_openml(name="house_sales", version=3)

X = dataset.data.values.astype(float)
y = dataset.target.values/1000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, random_state=42)

X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train,
                                                            test_size=200)
```

We now create our wrapper object, using a `RandomForestRegressor` as learner.


```python
from sklearn.ensemble import RandomForestRegressor

regressor = WrapCalibratedExplainer(RandomForestRegressor())
display(regressor)
```

We now fit our model using the proper training set.


```python
regressor.fit(X_prop_train, y_prop_train)
display(regressor)
```

The `WrapCalibratedExplainer` class has a `predict` method that returns the predictions and probability estimates of the underlying classifier. If the model is not yet calibrated, then the underlying models `predict` method is used. If the model is calibrated, then the `predict` method of the calibration model is used.


```python
predict = regressor.predict(X_test)
print(f'Uncalibrated model prediction: \n{predict}')
```

Before we can generate explanations, we need to calibrate our model using the calibration set.


```python
regressor.calibrate(X_cal, y_cal)
display(regressor)
```

We can easily add a difficulty estimator by assigning a `DifficultyEstimator` to the `difficulty_estimator` attribute when calibrating the model.


```python
from crepes.extras import DifficultyEstimator

de = DifficultyEstimator().fit(X=X_prop_train, learner=regressor.learner, scaler=True)
regressor.calibrate(X_cal, y_cal, difficulty_estimator=de)
display(regressor)
```

A `DifficultyEstimator` can also be assigned to an already calibrated model using the `set_difficult_estimator` method. Using `set_difficult_estimator(None)` removes any previously assigned `DifficultyEstimator`.

Once the model is calibrated, the `predict` method produce calibrated predictions with uncertainties. The default confidence is 90 per cent, which can be altered using the `low_high_percentiles` parameter.


```python
prediction, (low, high) = regressor.predict(X_test, uq_interval=True) # default low_high_percentiles=(5, 95)
print('Calibrated prediction:')
print(prediction)
print('Calibrated uncertainty interval:')
print([(low[i], high[i]) for i in range(len(low))])
```

You can also get the probability of the prediction being below a certain threshold using `predict_proba` by assigning the `threshold` parameter.


```python
import numpy as np
prediction = regressor.predict(X_test, threshold=200)
print('Calibrated probabilistic prediction:')
print(prediction)

proba, (low, high) = regressor.predict_proba(X_test, uq_interval=True, threshold=200)
print('Calibrated probabilistic probability estimate [y_hat > threshold, y_hat <= threshold]:')
print(proba)
print('Calibrated probabilistic uncertainty interval for y_hat <= threshold ([lower_bound, upper_bound]):')
print(np.array([(low[i], high[i]) for i in range(len(low))]))
```

#### Factual Explanations
Let us explain a test instance using our `WrapCalibratedExplainer` object. The method used to get factual explanations is `explain_factual`.


```python
factual_explanations = regressor.explain_factual(X_test) # default low_high_percentiles=(5, 95)
display(regressor)
```

Regression also offer both regular and uncertainty plots for factual explanations with or without conjunctive rules, in almost exactly the same way as for classification.


```python
factual_explanations.plot()
factual_explanations.plot(uncertainty=True)

factual_explanations.add_conjunctions().plot(uncertainty=True)
```

Default, the confidence interval is set to a symmetric interval of 90% (defined as `low_high_percentiles=(5,95)`). The intervals can cover any user specified interval, including one-sided intervals. To define a one-sided upper-bounded 90% interval, set `low_high_percentiles=(-np.inf,90)`, and to define a one-sided lower-bounded 95% interval, set `low_high_percentiles=(5,np.inf)`. Percentiles can also be set to any other values in the range (0,100) (exclusive), and intervals do not have to be symmetric.


```python
lower_bounded_explanations = regressor.explain_factual(X_test, low_high_percentiles=(5,np.inf))
asymmetric_explanations = regressor.explain_factual(X_test, low_high_percentiles=(5,75))
```

#### Explore Alternative Explanations
The `explore_alternatives` will work exactly the same as for classification.


```python
alternative_explanations = regressor.explore_alternatives(X_test) # default low_high_percentiles=(5, 95)
display(regressor)
```

Alternative plots work as for classification.


```python
alternative_explanations.plot()
alternative_explanations.add_conjunctions().plot()
```

### Probabilistic Regression
The difference between probabilistic regression and regular regression is that the former returns a probability of the prediction being below a certain threshold. This could for example be useful when the prediction is a time to an event, such as time to death or time to failure.


```python
probabilistic_factual_explanations = regressor.explain_factual(X_test, threshold=200)
probabilistic_factual_explanations.plot()
probabilistic_factual_explanations.plot(uncertainty=True)
```


```python
probabilistic_alternative_explanations = regressor.explore_alternatives(X_test, threshold=200)
probabilistic_alternative_explanations.plot()
```

Regression offers many more options but to learn more about them, see the [demo_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_regression.ipynb) or the [demo_probabilistic_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_probabilistic_regression.ipynb) notebooks.

### Initializing `WrapCalibratedExplainer`s

A `WrapCalibratedExplainer` can also be initialized with a trained model or with a `CalibratedExplainer` object, as is examplified below.

```python
fitted_classifier = WrapCalibratedExplainer(classifier.learner)
display(fitted_classifier)
calibrated_classifier = WrapCalibratedExplainer(classifier.explainer)
display(calibrated_classifier)

fitted_regressor = WrapCalibratedExplainer(regressor.learner)
display(fitted_regressor)
calibrated_regressor = WrapCalibratedExplainer(regressor.explainer)
display(calibrated_regressor)
```

When a calibrated explainer is re-fitted, the explainer is reinitialized.

### Known Limitations
The implementation currently only support numerical input. Use the `utils.helper.transform_to_numeric` (released in version v0.3.1) to transform a `DataFrame` with text data into numerical form and at the same time extracting `categorical_features`, `categorical_labels`, `target_labels` (if text labels) and `mappings` (used to apply the same mappings to new data) to be used as input to the `CalibratedExplainer`. The algorithm does not currently support image data.

See e.g. the [Conditional Fairness Experiment](evaluation/Conditional_Fairness_Experiment.ipynb) for examples on how `utils.helper.transform_to_numeric` can be used.

[Table of Content](#table-of-contents)

## Installation

### From PyPI:
Install `calibrated-explanations` from PyPI:

```bash
pip install calibrated-explanations
```

### From conda-forge:
Alternatively, you can install it from conda-forge:

```bash
conda install -c conda-forge calibrated-explanations
```

<!-- ### From GitHub:
To install the latest version directly from the GitHub repository, use the following command:

```bash
conda install git+https://github.com/Moffran/calibrated_explanations.git
``` -->

### Dependencies:
The following dependencies are required and will be installed automatically if not already present:

* [crepes](https://github.com/henrikbostrom/crepes)
* [venn-abers](https://github.com/ip200/venn-abers)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)

[Table of Content](#table-of-contents)


## Contributing
Contributions are welcome.
Please send bug reports, feature requests or pull requests through
the project page on [GitHub](https://github.com/Moffran/calibrated_explanations).
You can find a detailed guide for contributions in
[CONTRIBUTING.md](https://github.com/Moffran/calibrated_explanations/blob/main/CONTRIBUTING.md).

[Table of Content](#table-of-contents)


## Documentation
For documentation, see [calibrated-explanations.readthedocs.io](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest).

[Table of Content](#table-of-contents)


## License
The license used by `calibrated-explanations` is the [BSD 3-Clause License](https://github.com/Moffran/calibrated_explanations/blob/main/LICENSE).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

[Table of Content](#table-of-contents)


## Further Reading and Citing

If you use `calibrated-explanations` for a scientific publication, you are kindly requested to cite one of the following papers:
### Published Papers
- [Löfström, H](https://github.com/Moffran). (2023). [Trustworthy explanations: Improved decision support through well-calibrated uncertainty quantification](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1810440&dswid=6197) (Doctoral dissertation, Jönköping University, Jönköping International Business School).
- [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom)., Johansson, U., and Sönströd, C. (2024). [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://doi.org/10.1016/j.eswa.2024.123154). Expert Systems with Applications, 1-27.
- [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U., Sönströd, C., and [Matela, R](https://github.com/rudymatela). (2025). [Calibrated Explanations for Regression](https://doi.org/10.1007/s10994-024-06642-8). Machine Learning 114, 100.
- [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom). (2024). [Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty](https://doi.org/10.1007/978-3-031-63787-2_17). In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2153. Springer, Cham.
- [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U. (2024). [Calibrated Explanations for Multi-class](https://raw.githubusercontent.com/mlresearch/v230/main/assets/lofstrom24a/lofstrom24a.pdf). <i>Proceedings of the Thirteenth Workshop on Conformal and Probabilistic Prediction and Applications</i>, in <i>Proceedings of Machine Learning Research</i>, PMLR 230:175-194. [Presentation](https://copa-conference.com/presentations/Lofstrom.pdf)

The paper that originated the idea of `calibrated-explanations` is:

- [Löfström, H.](https://github.com/Moffran), [Löfström, T.](https://github.com/tuvelofstrom), Johansson, U., & Sönströd, C. (2023). [Investigating the impact of calibration on the quality of explanations](https://link.springer.com/article/10.1007/s10472-023-09837-2). Annals of Mathematics and Artificial Intelligence, 1-18. [Code and results](https://github.com/tuvelofstrom/calibrating-explanations).

### Preprints:
- [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom)., and [Hallberg Szabadvary, J](https://github.com/egonmedhatten). (2024). [Ensured: Explanations for Decreasing the Epistemic Uncertainty in Predictions](https://arxiv.org/abs/2410.05479). arXiv preprint arXiv:2410.05479.
- [Löfström, T](https://github.com/tuvelofstrom)., [Rabia Yapicioglu, F](https://github.com/rabia174)., Stramiglio A., [Löfström, H](https://github.com/Moffran)., and Vitali F. (2024). [Fast Calibrated Explanations: Efficient and Uncertainty-Aware Explanations for Machine Learning Models](https://arxiv.org/abs/2410.21129). arXiv preprint arXiv:2410.21129.

### Citing and Bibtex
If you use `calibrated-explanations` for a scientific publication, you are kindly requested to cite one of the papers above. Bibtex entries can be found in [citing](https://github.com/Moffran/calibrated_explanations/blob/main/docs/citing.md#bibtex-entries).

[Table of Content](#table-of-contents)


## Acknowledgements
This research is funded by the [Swedish Knowledge Foundation](https://www.kks.se/) together with industrial partners supporting the research and education environment on Knowledge Intensive Product Realization SPARK at Jönköping University, Sweden, through projects: AFAIR grant no. 20200223, ETIAI grant no. 20230040, and PREMACOP grant no. 20220187. Helena Löfström was initially a PhD student in the Industrial Graduate School in Digital Retailing (INSiDR) at the University of Borås, funded by the Swedish Knowledge Foundation, grant no. 20160035.

[Rudy Matela](https://github.com/rudymatela) has been our git guru and has helped us with the release process.

We have used both the `ConformalPredictiveSystem` and `DifficultyEstimator` classes from [Henrik Boström](https://github.com/henrikbostrom)s [crepes](https://github.com/henrikbostrom/crepes) package to provide support for regression. The `MondrianCategorizer` class is also supported in the `WrapCalibratedExplainer` as an alternative to using the `bins` parameter to create conditional explanations.

We have used the `VennAbers` class from [Ivan Petej](https://github.com/ip200)s [venn-abers](https://github.com/ip200/venn-abers) package to provide support for probabilistic explanations (both classification and probabilistic regression).

The `FastExplanation`, created using the `explain_fast` method, is incorporating ideas and code from [ConformaSight](https://github.com/rabia174/ConformaSight) developed by [Fatima Rabia Yapicioglu](https://github.com/rabia174), Allesandra Stramiglio, and Fabio Vitali.

We are using Decision Trees from `scikit-learn` in the discretizers.

We have copied code from [Marco Tulio Correia Ribeiro](https://github.com/marcotcr)s [lime](https://github.com/marcotcr/lime) package for the `Discretizer` class.

The `check_is_fitted` and `safe_instance` functions in `calibrated_explanations.utils` are copied from `sklearn` and `shap`.

[Table of Content](#table-of-contents)


## Support
For any questions or issues, please open an [issue](https://github.com/Moffran/calibrated_explanations/issues), open a [discussion](https://github.com/Moffran/calibrated_explanations/discussions) or contact the maintainers.

[Table of Content](#table-of-contents)

[build-log]:    https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml
[build-status]: https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml/badge.svg
[pypi-version]: https://img.shields.io/pypi/v/calibrated-explanations
[calibrated-explanations-on-pypi]: https://pypi.org/project/calibrated-explanations
