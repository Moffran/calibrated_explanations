Calibrated Explanations
=======================

[![Calibrated Explanations PyPI version][pypi-version]][calibrated-explanations-on-pypi]
[![Build Status for Calibrated Explanations][build-status]][build-log]

Calibrated Explanations is a Python library for the Calibrated Explanations method, initially developed for classification but is now extended for regression.
The proposed method is based on Venn-Abers (classification) and Conformal Predictive Systems (regression) and has the following characteristics:
* Fast, reliable, stable and robust feature importance explanations.
* Calibration of the underlying model to ensure that probability estimates are closer to reality (classification).
* Uncertainty quantification of the prediction from the underlying model and the feature importance weights. 
* Rules with straightforward interpretation in relation to the feature weights.
* Possibility to generate counterfactual rules with uncertainty quantification of the expected predictions achieved.
* Conjunctional rules conveying joint contribution between features.


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
["Calibrated Explanations: with Uncertainty Information and Counterfactuals"](https://arxiv.org/abs/2305.02305)
by
[Helena Löfström](https://github.com/Moffran),
[Tuwe Löfström](https://github.com/tuvelofstrom),
Ulf Johansson and
Cecilia Sönströd.

If you would like to cite this work, please cite the above paper.

[build-log]:    https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml
[build-status]: https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml/badge.svg
[pypi-version]: https://img.shields.io/pypi/v/calibrated-explanations
[calibrated-explanations-on-pypi]: https://pypi.org/project/calibrated-explanations
