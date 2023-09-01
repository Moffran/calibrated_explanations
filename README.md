Calibrated Explanations
=======================

[![Calibrated Explanations PyPI version][pypi-version]][calibrated-explanations-on-pypi]
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/calibrated-explanations.svg)](https://anaconda.org/conda-forge/calibrated-explanations)
[![Build Status for Calibrated Explanations][build-status]][build-log]
[![Lint Status for Calibrated Explanations][lint-status]][lint-log]
[![Documentation Status](https://readthedocs.org/projects/calibrated-explanations/badge/?version=latest)](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest)
[![License](https://badgen.net/github/license/moffran/calibrated_explanations)](https://github.com/moffran/calibrated_explanations/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/calibrated-explanations)](https://pepy.tech/project/calibrated-explanations)


`calibrated-explanations` is a Python package for the Calibrated Explanations method, supporting both classification and regression.
The proposed method is based on Venn-Abers (classification) and Conformal Predictive Systems (regression) and has the following characteristics:
* Fast, reliable, stable and robust feature importance explanations.
* Calibration of the underlying model to ensure that predictions reflect reality.
* Uncertainty quantification of the prediction from the underlying model and the feature importance weights. 
* Rules with straightforward interpretation in relation to the feature weights.
* Possibility to generate counterfactual rules with uncertainty quantification of the expected predictions achieved.
* Conjunctional rules conveying joint contribution between features.

![Counterfactual explanation](https://github.com/Moffran/calibrated_explanations/blob/main/docs/images/counterfactual_diabetes.png "Counterfactual explanation")

Documentation
-------------
For documentation, see [calibrated-explanations.readthedocs.io](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest).

Further reading
---------------
The `calibrated-explanations` method for classification is introduced in the paper:

[Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom)., Johansson, U., and Sönströd, C. [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://arxiv.org/abs/2305.02305). arXiv preprint arXiv:2305.02305.

The extensions for regression are introduced in the paper:

[Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U., Sönströd, C., and [Matela, R](https://github.com/rudymatela). [Calibrated Explanations for Regression](https://arxiv.org/abs/2308.16245). arXiv preprint arXiv:2308.16245.

The paper that originated the idea of `calibrated-explanations` is:

[Löfström, H.](https://github.com/Moffran), [Löfström, T.](https://github.com/tuvelofstrom), Johansson, U., & Sönströd, C. (2023). [Investigating the impact of calibration on the quality of explanations](https://link.springer.com/article/10.1007/s10472-023-09837-2). Annals of Mathematics and Artificial Intelligence, 1-18.
  
Install
-------

First, you need a Python environment.

Then `calibrated-explanations` can be installed from PyPI:

	pip install calibrated-explanations

or from conda-forge:
	
 	conda install -c conda-forge calibrated-explanations

or by following further instructions at [conda-forge](https://github.com/conda-forge/calibrated-explanations-feedstock#installing-calibrated-explanations).

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
The [notebooks folder](https://github.com/Moffran/calibrated_explanations/tree/main/notebooks) contains a number of notebooks illustrating different use cases for `calibrated-explanations`. The following are commented and should be a good start:
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

dataset = fetch_openml(name="wine", version=7, as_frame=True)

X = dataset.data.values.astype(float)
y = dataset.target.values

feature_names = dataset.feature_names

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

explainer = CalibratedExplainer(rf, X_cal, y_cal, feature_names=feature_names)

factual_explanations = explainer.explain_factual(X_test)
```

Once we have the explanations, we can plot all of them using `plot_all`. Default, a regular plot, without uncertainty intervals included, is created. To include uncertainty intervals, change the parameter `uncertainty=True`. To plot only a single instance, the `plot_factual` function can be called, submitting the index of the test instance to plot. You can also add and remove conjunctive rules.

```python
factual_explanations.plot_all()
factual_explanations.plot_all(uncertainty=True)

factual_explanations.plot_factual(0, uncertainty=True)

factual_explanations.add_conjunctions().plot_all()
factual_explanations.remove_conjunctions().plot_all()
```

#### Counterfactual Explanations
An alternative to factual rules is to extract counterfactual rules. 
`explain_counterfactual` can be called to get counterfactual rules with an appropriate discretizer automatically assigned.  

```python
counterfactual_explanations = explainer.explain_counterfactual(X_test)
```

Counterfactuals are also visualized using the `plot_all`. Plotting an individual counterfactual explanation is done using `plot_counterfactual`, submitting the index to plot. Adding or removing conjunctions is done as before. 

```python
counterfactual_explanations.plot_all()
counterfactual_explanations.add_conjunctions().plot_all()
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

factual_explanations = explainer.explain_factual(X_test)

factual_explanations.plot_all()
factual_explanations.plot_all(uncertainty=True)

factual_explanations.add_conjunctions().plot_all()
```
Default, the confidence interval is set to a symmetric interval of 90% (defined as `low_high_percentiles=(5,95)`). The intervals can cover any user specified interval, including one-sided intervals. To define a one-sided upper-bounded 90% interval, set `low_high_percentiles=(-np.inf,90)`, and to define a one-sided lower-bounded 95% interval, set `low_high_percentiles=(5,np.inf)`. Percentiles can also be set to any other values in the range [0,100] (exclusive), and intervals do not have to be symmetric. 

```python
lower_bounded_explanations = explainer.explain_factual(X_test, low_high_percentiles=(5,np.inf))
asymmetric_explanations = explainer.explain_factual(X_test, low_high_percentiles=(5,75))
```

#### Counterfactual Explanations
The `explain_counterfactual` will work exactly the same as for classification. Otherwise, the discretizer must be set explicitly and the 'decile' discretizer is recommended. Counterfactual plots work in the same way as for classification.

```python
counterfactual_explanations = explainer.explain_counterfactual(X_test)

counterfactual_explanations.plot_all()
counterfactual_explanations.add_conjunctions().plot_all()

counterfactual_explanations.plot_counterfactual(0)
```
The parameter `low_high_percentiles` works in the same way as for factual explanations. 

#### Probabilistic Regression Explanations
It is possible to create probabilistic explanations for regression, providing the probability that the target value is below the provided threshold (which is 180 000 in the examples below). All methods are the same as for normal regression and classification.

```python
factual_explanations = explainer.explain_factual(X_test, 180000)

factual_explanations.plot_all()
factual_explanations.plot_all(uncertainty=True)

factual_explanations.add_conjunctions().plot_all()

counterfactual_explanations = explainer.explain_counterfactual(X_test, 180000)

counterfactual_explanations.plot_all()
counterfactual_explanations.add_conjunctions().plot_all()
```

#### Additional Regression Use Cases
Regression offers many more options and to learn more about them, see the [demo_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_regression.ipynb) or the [demo_probabilistic_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_probabilistic_regression.ipynb) notebooks.


Development
-----------

This project has tests that can be executed using `pytest`.
Just run the following command from the project root.

```bash
pytest
```

To build the Sphinx documentation,
run the following command in the project root:

```bash
sphinx-build docs docs/_build
```

Then open the `docs/_build/index.html` file on your web browser.

[calibrated-explanations documentation on readthedocs]: https://calibrated-explanations.readthedocs.io/

The [calibrated-explanations documentation on readthedocs] is automatically
updated form GitHub's main branch.
If there is an issue with the documentation build there,
the [build tab](https://readthedocs.org/projects/calibrated-explanations/builds/)
of the [project page](https://readthedocs.org/projects/calibrated-explanations/)
should have the logs.

To make a new release on PyPI,
just follow the [release guide](docs/release.md).


Citing
------
If you use `calibrated-explanations` for a scientific publication, you are kindly requested to cite one of the following papers:

* [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom)., Johansson, U., and Sönströd, C. [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://arxiv.org/abs/2305.02305). arXiv preprint arXiv:2305.02305.

* [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U., Sönströd, C., and [Matela, R](https://github.com/rudymatela). [Calibrated Explanations for Regression](https://arxiv.org/abs/2308.16245). arXiv preprint arXiv:2308.16245.

Bibtex entry for the original paper:

```bibtex
@misc{calibrated-explanations,
      title = 	      {Calibrated Explanations: with Uncertainty Information and Counterfactuals},
      author =          {L\"ofstr\"om, Helena and L\"ofstr\"om, Tuwe and Johansson, Ulf and S\"onstr\"od, Cecilia},
      year =            {2023},
      eprint =          {2305.02305},
      archivePrefix =   {arXiv},
      primaryClass =    {cs.AI}
}
```
Bibtex entry for the regression paper:

```bibtex
@misc{cal-expl-regression,
      title = 	      {Calibrated Explanations for Regression},
      author =          {L\"ofstr\"om, Tuwe and L\"ofstr\"om, Helena and Johansson, Ulf and S\"onstr\"od, Cecilia and Matela, Rudy},
      year =            {2023},
      eprint =          {2308.16245},
      archivePrefix =   {arXiv},
      primaryClass =    {cs.LG}
}
```

[build-log]:    https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml
[build-status]: https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml/badge.svg
[lint-log]:    https://github.com/Moffran/calibrated_explanations/actions/workflows/pylint.yml
[lint-status]: https://github.com/Moffran/calibrated_explanations/actions/workflows/pylint.yml/badge.svg
[pypi-version]: https://img.shields.io/pypi/v/calibrated-explanations
[calibrated-explanations-on-pypi]: https://pypi.org/project/calibrated-explanations
