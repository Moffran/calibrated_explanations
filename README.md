Calibrated Explanations ([Documentation](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest))
=======================

[![Calibrated Explanations PyPI version][pypi-version]][calibrated-explanations-on-pypi]
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/calibrated-explanations.svg)](https://anaconda.org/conda-forge/calibrated-explanations)
[![GitHub (Pre-)Release Date](https://img.shields.io/github/release-date-pre/Moffran/calibrated_explanations)](https://github.com/Moffran/calibrated_explanations/blob/main/CHANGELOG.md)
[![Documentation Status](https://readthedocs.org/projects/calibrated-explanations/badge/?version=latest)](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest)
[![Build Status for Calibrated Explanations][build-status]][build-log]
[![Lint Status for Calibrated Explanations][lint-status]][lint-log]
[![License](https://badgen.net/github/license/moffran/calibrated_explanations)](https://github.com/moffran/calibrated_explanations/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/calibrated-explanations)](https://pepy.tech/project/calibrated-explanations)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Moffran/calibrated_explanations/v0.3.3)

`calibrated-explanations` is a Python package for the local feature importance explanation method called Calibrated Explanations, supporting both [classification](https://doi.org/10.1016/j.eswa.2024.123154) and [regression](https://arxiv.org/abs/2308.16245).
The proposed method is based on Venn-Abers (classification & regression) and Conformal Predictive Systems (regression) and has the following characteristics:
* Fast, reliable, stable and robust feature importance explanations for:
	- Binary classification models
	- Multi-class classification models
	- Regression models
		* Including probabilistic explanations of the probability that the target exceeds a user-defined threshold 
		* With difficulty adaptable explanations (conformal normalization) 
* Calibration of the underlying model to ensure that predictions reflect reality.
* Uncertainty quantification of the prediction from the underlying model and the feature importance weights. 
* Rules with straightforward interpretation in relation to instance values and feature weights.
* Possibility to generate counterfactual rules with uncertainty quantification of the expected predictions.
* Conjunctional rules conveying feature importance for the interaction of included features.
* Conditional rules, allowing users the ability to create contextual explanations to handle e.g. bias and fairness constraints. 

Below is an example of a probabilistic counterfactual explanation for an instance of the regression dataset California Housing (with the threshold 180 000). The light red area in the background is representing the calibrated probability interval (for the prediction being below the threshold) of the underlying model, as indicated by a Conformal Predictive System and calibrated through Venn-Abers. The darker red bars for each rule show the probability intervals that Venn-Abers indicate for an instance changing a feature value in accordance with the rule condition.
<p align="center">
  <a href="https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest">
    <img src="https://github.com/Moffran/calibrated_explanations/blob/main/docs/images/counterfactual_probabilistic_house_regression.jpg" alt="Probabilistic counterfactual explanation for California Housing">
  </a>
</p>

The table summarizes the characteristics of Calibrated Explanations.
<table align="center" style="border-collapse: collapse;">
  <tr>
    <th style="border-bottom: 0px; text-align: left;"></th>
    <th style="text-align: center; border-left: 1px solid; border-bottom: 0px;" colspan="3"></th>
	<th style="text-align: center; border-left: 1px solid; border-bottom: 0px;" colspan="3">Standard</th>
	<th style="text-align: center; border-left: 1px solid; border-bottom: 0px;" colspan="3">Probabilistic</th>	
  </tr>
  <tr>
    <th style="border-bottom: 0px; text-align: left;"></th>
    <th style="text-align: center; border-left: 1px solid; border-bottom: 0px;" colspan="3">Classification</th>
	<th style="text-align: center; border-left: 1px solid; border-bottom: 0px;" colspan="3">Regression</th>
	<th style="text-align: center; border-left: 1px solid; border-bottom: 0px;" colspan="3">Regression</th>	
  </tr>
  <tr>
    <th style="border-bottom: 1px solid; text-align: left;">Characteristics</th>
    <th style="border-bottom: 1px solid; border-left: 1px solid; ">FR</th>
    <th style="border-bottom: 1px solid;">FU</th>
    <th style="border-bottom: 1px solid;">CF</th>
    <th style="border-bottom: 1px solid; border-left: 1px solid; ">FR</th>
    <th style="border-bottom: 1px solid;">FU</th>
    <th style="border-bottom: 1px solid;">CF</th>
    <th style="border-bottom: 1px solid; border-left: 1px solid; ">FR</th>
    <th style="border-bottom: 1px solid;">FU</th>
    <th style="border-bottom: 1px solid;">CF</th>
  </tr>
  <tr>
    <td style="text-align: left;">Feature Weight w/o CI</td>
    <td style="border-left: 1px solid; ">X</td>
    <td></td>
    <td></td>
    <td style="border-left: 1px solid; ">X</td>
    <td></td>
    <td></td>
    <td style="border-left: 1px solid; ">X</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td style="text-align: left;">Feature Weight with CI</td>
    <td style="border-left: 1px solid; "></td>
    <td>X</td>
    <td></td>
    <td style="border-left: 1px solid; "></td>
    <td>X</td>
    <td></td>
    <td style="border-left: 1px solid; "></td>
    <td>X</td>
    <td></td>
  </tr>
  <tr>
    <td style="border-bottom: 1px solid; text-align: left;">Rule Prediction with CI</td>
    <td style="border-bottom: 1px solid; border-left: 1px solid; "></td>
    <td style="border-bottom: 1px solid;"></td>
    <td style="border-bottom: 1px solid;">X</td>
    <td style="border-bottom: 1px solid; border-left: 1px solid;"></td>
    <td style="border-bottom: 1px solid;"></td>
    <td style="border-bottom: 1px solid;">X</td>
    <td style="border-bottom: 1px solid; border-left: 1px solid;"></td>
    <td style="border-bottom: 1px solid;"></td>
    <td style="border-bottom: 1px solid;">X</td>
  </tr>
  <tr>
    <td style="text-align: left;">Two-sided CI</td>
    <td style="border-left: 1px solid; ">I</td>
    <td>I</td>
    <td>I</td>
    <td style="border-left: 1px solid; ">I</td>
    <td>I</td>
    <td>I</td>
    <td style="border-left: 1px solid; ">I</td>
    <td>I</td>
    <td>I</td>
  </tr>
  <tr>
    <td style="text-align: left;">Lower-bounded CI</td>
    <td style="border-left: 1px solid; "></td>
    <td></td>
    <td></td>
    <td style="border-left: 1px solid; ">I</td>
    <td></td>
    <td>I</td>
    <td style="border-left: 1px solid; "></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td style="border-bottom: 1px solid; text-align: left;">Upper-bounded CI</td>
    <td style="border-bottom: 1px solid; border-left: 1px solid; "></td>
    <td style="border-bottom: 1px solid;"></td>
    <td style="border-bottom: 1px solid;"></td>
    <td style="border-bottom: 1px solid; border-left: 1px solid;">I</td>
    <td style="border-bottom: 1px solid;"></td>
    <td style="border-bottom: 1px solid;">I</td>
    <td style="border-bottom: 1px solid; border-left: 1px solid;"></td>
    <td style="border-bottom: 1px solid;"></td>
    <td style="border-bottom: 1px solid;"></td>
  </tr>
  <tr>
    <td style="text-align: left;">Conjunctive Rules</td>
    <td style="border-left: 1px solid; ">O</td>
    <td>O</td>
    <td>O</td>
    <td style="border-left: 1px solid; ">O</td>
    <td>O</td>
    <td>O</td>
    <td style="border-left: 1px solid; ">O</td>
    <td>O</td>
    <td>O</td>
  </tr>
  <tr>
    <td style="text-align: left;">Conditional Rules</td>
    <td style="border-left: 1px solid; ">O</td>
    <td>O</td>
    <td>O</td>
    <td style="border-left: 1px solid; ">O</td>
    <td>O</td>
    <td>O</td>
    <td style="border-left: 1px solid; ">O</td>
    <td>O</td>
    <td>O</td>
  </tr>
  <tr>
    <td style="text-align: left;">Difficulty Estimation</td>
    <td style="border-left: 1px solid; "></td>
    <td></td>
    <td></td>
    <td style="border-left: 1px solid; ">O</td>
    <td>O</td>
    <td>O</td>
    <td style="border-left: 1px solid; ">O</td>
    <td>O</td>
    <td>O</td>
  </tr>
  <tr>
    <td style="text-align: left;"># Alternative Setups</td>
    <td style="border-left: 1px solid; ">1</td>
    <td>1</td>
    <td>1</td>
    <td style="border-left: 1px solid; ">5</td>
    <td>5</td>
    <td>5</td>
    <td style="border-left: 1px solid; ">5</td>
    <td>5</td>
    <td>5</td>
  </tr>
</table>

All explanations include the *calibrated prediction*, with *confidence intervals* (**CI**), of the explained instance. 
- **FR** refers to *factual* explanations visualized using *regular* plots
- **FU** refers to *factual* explanations visualized using *uncertainty* plots
- **CF** refers to *counterfactual* explanations and plots
- **X** marks a *core alternative*
- **I** marks possible *interval type(s)*
- **O** marks *optional additions*
  
The example plot above, showing a counterfactual probabilistic regression explanation, corresponds to the last column without any optional additions.



<!-- <p align="center">
  <a href="https://arxiv.org/abs/2308.16245">
    <img src="https://github.com/Moffran/calibrated_explanations/blob/main/docs/images/Table1.png" alt="Characteristics of Calibrated Explanantions">
  </a>
</p> -->

Getting started
---------------
The [notebooks folder](https://github.com/Moffran/calibrated_explanations/tree/main/notebooks) contains a number of notebooks illustrating different use cases for `calibrated-explanations`. The following are commented and should be a good start:
* [quickstart](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/quickstart.ipynb) - similar to this Getting Started, including plots.
* [quickstart_wrap](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/quickstart_wrap.ipynb) - similar to this Getting Started, but with a wrapper class for easier use.
* [demo_binary_classification](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_binary_classification.ipynb) - with examples for binary classification 
* [demo_multiclass](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_multiclass_glass.ipynb) - with examples for multi-class classification
* [demo_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_regression.ipynb) - with examples for regression
* [demo_probabilistic_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_probabilistic_regression.ipynb) - with examples for regression with thresholds
* [demo_under_the_hood](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_under_the_hood.ipynb) - illustrating how to access the information composing the explanations

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
y = (dataset.target.values == 'True').astype(int)

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

Once we have the explanations, we can plot all of them using `plot_all`. Default, a regular plot, without uncertainty intervals included, is created. To include uncertainty intervals, change the parameter `uncertainty=True`. To plot only a single instance, the `plot_explanation` function can be called, submitting the index of the test instance to plot. You can also add and remove conjunctive rules.

```python
factual_explanations.plot_all()
factual_explanations.plot_all(uncertainty=True)

factual_explanations.plot_explanation(0, uncertainty=True)

factual_explanations.add_conjunctions().plot_all()
factual_explanations.remove_conjunctions().plot_all()
```

#### Counterfactual Explanations
An alternative to factual rules is to extract counterfactual rules. 
`explain_counterfactual` can be called to get counterfactual rules with an appropriate discretizer automatically assigned.  

```python
counterfactual_explanations = explainer.explain_counterfactual(X_test)
```

Counterfactuals are also visualized using the `plot_all`. Plotting an individual counterfactual explanation is done using `plot_explanation`, submitting the index of the test instance to plot. Adding or removing conjunctions is done as before. 

```python
counterfactual_explanations.plot_all()
counterfactual_explanations.plot_explanation(0)
counterfactual_explanations.add_conjunctions().plot_all()
```

Individual explanations can also be plotted using `plot_explanation`.
      
```python
factual_explanations.get_explanation(0).plot_explanation()
counterfactual_explanations.get_explanation(0).plot_explanation()
```
#### Support for multiclass
`calibrated-explanations` supports multiclass which is demonstrated in [demo_multiclass](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_multiclass_glass.ipynb). That notebook also demonstrates how both feature names and target and categorical labels can be added to improve the interpretability. 

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
Default, the confidence interval is set to a symmetric interval of 90% (defined as `low_high_percentiles=(5,95)`). The intervals can cover any user specified interval, including one-sided intervals. To define a one-sided upper-bounded 90% interval, set `low_high_percentiles=(-np.inf,90)`, and to define a one-sided lower-bounded 95% interval, set `low_high_percentiles=(5,np.inf)`. Percentiles can also be set to any other values in the range (0,100) (exclusive), and intervals do not have to be symmetric. 

```python
lower_bounded_explanations = explainer.explain_factual(X_test, low_high_percentiles=(5,np.inf))
asymmetric_explanations = explainer.explain_factual(X_test, low_high_percentiles=(5,75))
```

#### Counterfactual Explanations
The `explain_counterfactual` will work exactly the same as for classification. Counterfactual plots work in the same way as for classification.

```python
counterfactual_explanations = explainer.explain_counterfactual(X_test)

counterfactual_explanations.plot_all()
counterfactual_explanations.add_conjunctions().plot_all()

counterfactual_explanations.plot_explanation(0)
```
The parameter `low_high_percentiles` works in the same way as for factual explanations. 

#### Probabilistic Regression Explanations
It is possible to create probabilistic explanations for regression, providing the probability that the target value is below the provided threshold (which is 180 000 in the examples below). All methods are the same as for normal regression and classification, except that the `explain_factual` and `explain_counterfactual` methods need the additional threshold value (here 180 000).

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

[Top](#calibrated-explanations-documentation)

Known Limitations
-----------------
The implementation currently only support numerical input. Use the `utils.transform_to_numeric` (released in version v0.3.1) to transform a `DataFrame` with text data into numerical form and at the same time extracting `categorical_features`, `categorical_labels`, `target_labels` (if text labels) and `mappings` (used to apply the same mappings to new data) to be used as input to the `CalibratedExplainer`. The algorithm does not currently support image data.

[Top](#calibrated-explanations-documentation)

Install
-------

`calibrated-explanations` is implemented in Python, so you need a Python environment.

Install `calibrated-explanations` from PyPI:

	pip install calibrated-explanations

or from conda-forge:
	
 	conda install -c conda-forge calibrated-explanations

or by following further instructions at [conda-forge](https://github.com/conda-forge/calibrated-explanations-feedstock#installing-calibrated-explanations).

The dependencies are:

* [crepes](https://github.com/henrikbostrom/crepes)
* [venn-abers](https://github.com/ip200/venn-abers)
* [lime](https://github.com/marcotcr/lime)
* [matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)
  
[Top](#calibrated-explanations-documentation)


Contributing
------------

Contributions are welcome.
Please send bug reports, feature requests or pull requests through
the project page on [GitHub](https://github.com/Moffran/calibrated_explanations).
You can find a detailed guide for contributions in
[CONTRIBUTING.md](https://github.com/Moffran/calibrated_explanations/blob/main/CONTRIBUTING.md).

[Top](#calibrated-explanations-documentation)


Documentation
-------------
For documentation, see [calibrated-explanations.readthedocs.io](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest).
  
[Top](#calibrated-explanations-documentation)

Further reading and citing
--------------------------
The `calibrated-explanations` method for classification is introduced in the paper:

- [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom)., Johansson, U., and Sönströd, C. (2024). [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://doi.org/10.1016/j.eswa.2024.123154). Expert Systems with Applications, 1-27.

The extensions for regression are introduced in the paper:

- [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U., Sönströd, C., and [Matela, R](https://github.com/rudymatela). [Calibrated Explanations for Regression](https://arxiv.org/abs/2308.16245). arXiv preprint arXiv:2308.16245.

The paper that originated the idea of `calibrated-explanations` is:

- [Löfström, H.](https://github.com/Moffran), [Löfström, T.](https://github.com/tuvelofstrom), Johansson, U., & Sönströd, C. (2023). [Investigating the impact of calibration on the quality of explanations](https://link.springer.com/article/10.1007/s10472-023-09837-2). Annals of Mathematics and Artificial Intelligence, 1-18. [Code and results](https://github.com/tuvelofstrom/calibrating-explanations).

If you use `calibrated-explanations` for a scientific publication, you are kindly requested to cite one of the papers above.

Bibtex entry for the original paper:

```bibtex
@article{lofstrom2024ce_classification,
	title = 	{Calibrated explanations: With uncertainty information and counterfactuals},
	journal = 	{Expert Systems with Applications},
	pages = 	{123154},
	year = 		{2024},
	issn = 		{0957-4174},
	doi = 		{https://doi.org/10.1016/j.eswa.2024.123154},
	url = 		{https://www.sciencedirect.com/science/article/pii/S0957417424000198},
	author = 	{Helena Löfström and Tuwe Löfström and Ulf Johansson and Cecilia Sönströd},
	keywords = 	{Explainable AI, Feature importance, Calibrated explanations, Venn-Abers, Uncertainty quantification, Counterfactual explanations},
	abstract = 	{While local explanations for AI models can offer insights into individual predictions, such as feature importance, they are plagued by issues like instability. The unreliability of feature weights, often skewed due to poorly calibrated ML models, deepens these challenges. Moreover, the critical aspect of feature importance uncertainty remains mostly unaddressed in Explainable AI (XAI). The novel feature importance explanation method presented in this paper, called Calibrated Explanations (CE), is designed to tackle these issues head-on. Built on the foundation of Venn-Abers, CE not only calibrates the underlying model but also delivers reliable feature importance explanations with an exact definition of the feature weights. CE goes beyond conventional solutions by addressing output uncertainty. It accomplishes this by providing uncertainty quantification for both feature weights and the model’s probability estimates. Additionally, CE is model-agnostic, featuring easily comprehensible conditional rules and the ability to generate counterfactual explanations with embedded uncertainty quantification. Results from an evaluation with 25 benchmark datasets underscore the efficacy of CE, making it stand as a fast, reliable, stable, and robust solution.}
}
```
Bibtex entry for the regression paper:

```bibtex
@misc{lofstrom2023ce_regression,
      title = 	      	{Calibrated Explanations for Regression},
      author =          {L\"ofstr\"om, Tuwe and L\"ofstr\"om, Helena and Johansson, Ulf and S\"onstr\"od, Cecilia and Matela, Rudy},
      year =            {2023},
      eprint =          {2308.16245},
      archivePrefix =   {arXiv},
      primaryClass =    {cs.LG}
}
```

To cite this software, use the following bibtex entry:

```bibtex
@software{lofstrom2024ce_repository,
	author = 	{Löfström, Helena and Löfström, Tuwe and Johansson, Ulf and Sönströd, Cecilia and Matela, Rudy},
	license = 	{BSD-3-Clause},
	title = 	{Calibrated Explanations},
	url = 		{https://github.com/Moffran/calibrated_explanations},
	version = 	{v0.3.3},
	month = 	May,
	year = 		{2024}
}
```
  
[Top](#calibrated-explanations-documentation)

Acknowledgements
----------------
This research is funded by the Swedish Knowledge Foundation together with industrial partners supporting the research and education environment on Knowledge Intensive Product Realization SPARK at Jönköping University, Sweden, through projects: AFAIR grant no. 20200223, ETIAI grant no. 20230040, and PREMACOP grant no. 20220187. Helena Löfström was a PhD student in the Industrial Graduate School in Digital Retailing (INSiDR) at the University of Borås, funded by the Swedish Knowledge Foundation, grant no. 20160035. 

[Rudy Matela](https://github.com/rudymatela) has been our git guru and has helped us with the release process.

We have used both the `ConformalPredictiveSystem` and `DifficultyEstimator` classes from [Henrik Boström](https://github.com/henrikbostrom)s [crepes](https://github.com/henrikbostrom/crepes) package to provide support for regression.

We have used the `VennAbers` class from [Ivan Petej](https://github.com/ip200)s [venn-abers](https://github.com/ip200/venn-abers) package to provide support for probabilistic explanations (both classification and probabilistic regression). 

We have used code from [Marco Tulio Correia Ribeiro](https://github.com/marcotcr)s [lime](https://github.com/marcotcr/lime) package for the `Discretizer` class.

The `check_is_fitted` and `safe_instance` functions in `calibrated_explanations.utils` are copied from `sklearn` and `shap`.  
  
[Top](#calibrated-explanations-documentation)

[build-log]:    https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml
[build-status]: https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml/badge.svg
[lint-log]:    https://github.com/Moffran/calibrated_explanations/actions/workflows/pylint.yml
[lint-status]: https://github.com/Moffran/calibrated_explanations/actions/workflows/pylint.yml/badge.svg
[pypi-version]: https://img.shields.io/pypi/v/calibrated-explanations
[calibrated-explanations-on-pypi]: https://pypi.org/project/calibrated-explanations
