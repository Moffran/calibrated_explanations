Calibrated Explanations
=======================

[![Build Status for Calibrated Explanations][build-status]][build-log]

Calibrated Explanations is a Python library
that is able to explain predictions of a black-box model
using Venn Abers predictors (classification)
or conformal predictive systems (regression) and perturbations.

Install
-------

First, you need a Python environment installed with pip.

Calibrated Explanations can be installed from PyPI:

	pip install calibrated-explanations

The dependencies are:

* [crepes](https://github.com/henrikbostrom/crepes)
* [lime](https://github.com/marcotcr/lime)
* [matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)
* [SHAP](https://pypi.org/project/shap/)
* [tqdm](https://tqdm.github.io/)


Getting started
---------------

```python
>>> import calibrated_explanations
... TODO: write me...
```


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
"Calibrated Explanations for Black-Box Predictions"
by
Helena Löfström,
[Tuwe Löfström](https://github.com/tuvelofstrom),
Ulf Johansson and
Cecilia Sönströd.

If you would like to cite this work, please cite the above paper.

[build-log]:    https://github.com/Moffran/calibrated_explanations/actions/workflows/ce.yml
[build-status]: https://github.com/Moffran/calibrated_explanations/actions/workflows/ce.yml/badge.svg
