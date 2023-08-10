Calibrated Explanations
=======================

[![Calibrated Explanations PyPI version][pypi-version]][calibrated-explanations-on-pypi]
[![Build Status for Calibrated Explanations][build-status]][build-log]
[![Lint Status for Calibrated Explanations][lint-status]][lint-log]
[![Documentation Status](https://readthedocs.org/projects/calibrated-explanations/badge/?version=latest)](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest)


`calibrated-explanations` is a Python package for the Calibrated Explanations method, supporting both classification and regression.
The proposed method is based on Venn-Abers (classification) and Conformal Predictive Systems (regression) and has the following characteristics:
* Fast, reliable, stable and robust feature importance explanations.
* Calibration of the underlying model to ensure that predictions reflect reality.
* Uncertainty quantification of the prediction from the underlying model and the feature importance weights. 
* Rules with straightforward interpretation in relation to the feature weights.
* Possibility to generate counterfactual rules with uncertainty quantification of the expected predictions achieved.
* Conjunctional rules conveying joint contribution between features.


Install
-------

First, you need a Python environment installed with pip.

`calibrated-explanations` can be installed from PyPI:

	pip install calibrated-explanations

The dependencies are:

* [crepes](https://github.com/henrikbostrom/crepes)
* [lime](https://github.com/marcotcr/lime)
* [matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)
* [SHAP](https://pypi.org/project/shap/)


Getting started
---------------
The notebooks folder contains a number of notebooks illustrating different use cases for `calibrated-explanations`. The following are commented and should be a good start:
* [demo_binary_classification](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_binary_classification.ipynb) 
* [demo_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_regression.ipynb) 

### Classification
Let us illustrate how we may use `calibrated-explanations` to generate explanations from a classifier trained on a dataset from
[www.openml.org](https://www.openml.org), which we first split into a
training and a test set using `train_test_split` from
[sklearn](https://scikit-learn.org), and then further split the
training set into a proper training set and a calibration set:

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

dataset = fetch_openml(name="qsar-biodeg")

X = dataset.data.values.astype(float)
y = dataset.target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, stratify=y)

X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train,
                                                            test_size=0.25)

```

We now fit a model on our data. 

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1)

rf.fit(X_prop_train, y_prop_train)
```

#### Factual Explanations
Lets extract explanations for our test set using the `calibrated-explanations` package by importing `CalibratedExplainer` from `calibrated_explanations`.

```python
from calibrated_explanations import CalibratedExplainer, __version__
print(__version__)

explainer = CalibratedExplainer(rf, X_cal, y_cal)

if __version__ >= "0.0.8":
    factual_explanations = explainer.get_factuals(X_test)
else:
    factual_explanations = explainer(X_test)
```

Once we have the explanations, we can plot them using `plot_regular` or `plot_uncertainty`. You can also add and remove conjunctive rules.

```python
factual_explanations.plot_regular()
factual_explanations.plot_uncertainty()

factual_explanations.add_conjunctive_factual_rules().plot_regular()
factual_explanations.remove_conjunctive_rules().plot_regular()
```

#### Counterfactual Explanations
An alternative to factual rules is to extract counterfactual rules. 
From version 0.0.8, `get_counterfactuals` can be called to get counterfactual rules with an appropriate discretizer automatically assigned. An alternative is to first change the discretizer to `entropy` (for classification) and then call the `CalibratedExplainer` object as above. 

```python
if __version__ >= "0.0.8":
    counterfactual_explanations = explainer.get_counterfactuals(X_test)
else:
    explainer.set_discretizer('entropy')
    counterfactual_explanations = explainer(X_test)
```

Counterfactuals are visualized using the `plot_counterfactuals`. Adding or removing conjunctions is done as before. 

```python
counterfactual_explanations.plot_counterfactuals()
counterfactual_explanations.add_conjunctive_counterfactual_rules().plot_counterfactuals()
counterfactual_explanations.remove_counterfactual_rules().plot_counterfactuals()
```
#### Support for multiclass
`calibrated-explanations` supports multiclass which is demonstrated in [demo_multiclass](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_multiclass.ipynb). That notebook also demonstrates how both feature names and target and categorical labels can be added to improve the interpretability. 

### Regression
Extracting explanations for regression is very similar to how it is done for classification. 

```python
dataset = fetch_openml(name="house_sales", version=3)

X = dataset.data.values.astype(float)
y = dataset.target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train,
                                                            test_size=0.25)
```

Let us now fit a `RandomForestRegressor` from
[sklearn](https://scikit-learn.org) to the proper training
set:

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_prop_train, y_prop_train)
```

#### Factual Explanations
Define a `CalibratedExplainer` object using the new model and data. The `mode` parameter must be explicitly set to regression. Regular and uncertainty plots work in the same way as for classification.

```python
explainer = CalibratedExplainer(rf, X_cal, y_cal, mode='regression')

if __version__ >= '0.0.8':
    factual_explanations = explainer.get_factuals(X_test)
else:
    factual_explanations = explainer(X_test)

factual_explanations.plot_regular()
factual_explanations.plot_uncertainty()

factual_explanations.add_conjunctive_factual_rules().plot_regular()
factual_explanations.remove_conjunctive_rules().plot_regular()
```

#### Counterfactual Explanations
From version 0.0.8, the `get_counterfactuals` will work exactly the same as for classification. Otherwise, the discretizer must be set explicitly and the 'decile' discretizer is recommended. Counterfactual plots work in the same way as for classification.

```python
if __version__ >= '0.0.8':
    counterfactual_explanations = explainer.get_counterfactuals(X_test)
else:
    explainer.set_discretizer('decile')
    counterfactual_explanations = explainer(X_test)

counterfactual_explanations.plot_counterfactuals()
counterfactual_explanations.add_conjunctive_counterfactual_rules().plot_counterfactuals()
counterfactual_explanations.remove_counterfactual_rules().plot_counterfactuals()
```

#### Additional Regression Use Cases
Regression offers many more options but to learn more about them, see the [demo_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_regression.ipynb) or the [demo_probabilistic_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_probabilistic_regression.ipynb) notebooks.


Development
-----------

This project has tests that can be executed using `pytest`.
Just run the following command from the project root.

```bash
pytest
```


Further reading
---------------

The calibrated explanations library is based on the paper
["Calibrated Explanations: with Uncertainty Information and Counterfactuals"](https://arxiv.org/abs/2305.02305)
by
[Helena Löfström](https://github.com/Moffran),
[Tuwe Löfström](https://github.com/tuvelofstrom),
Ulf Johansson and
Cecilia Sönströd.

If you would like to cite this work, please cite the above paper.

[build-log]:    https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml
[build-status]: https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml/badge.svg
[lint-log]:    https://github.com/Moffran/calibrated_explanations/actions/workflows/pylint.yml
[lint-status]: https://github.com/Moffran/calibrated_explanations/actions/workflows/pylint.yml/badge.svg
[pypi-version]: https://img.shields.io/pypi/v/calibrated-explanations
[calibrated-explanations-on-pypi]: https://pypi.org/project/calibrated-explanations
