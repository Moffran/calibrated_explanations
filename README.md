Calibrated Explanations ([Documentation](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest))
=======================

[![Calibrated Explanations PyPI version][pypi-version]][calibrated-explanations-on-pypi]
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/calibrated-explanations.svg)](https://anaconda.org/conda-forge/calibrated-explanations)
[![GitHub (Pre-)Release Date](https://img.shields.io/github/release-date-pre/Moffran/calibrated_explanations)](https://github.com/Moffran/calibrated_explanations/blob/main/CHANGELOG.md)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/moffran/calibrated_explanations/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/calibrated-explanations)](https://pepy.tech/project/calibrated-explanations)
<!-- [![Documentation Status](https://readthedocs.org/projects/calibrated-explanations/badge/?version=latest)](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest)
[![Build Status for Calibrated Explanations][build-status]][build-log] -->
<!-- [![Lint Status for Calibrated Explanations][lint-status]][lint-log] -->
<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Moffran/calibrated_explanations/main?urlpath=https%3A%2F%2Fgithub.com%2FMoffran%2Fcalibrated_explanations%2Fblob%2Fmain%2Fnotebooks%2Fquickstart.ipynb) -->

`calibrated-explanations` is a Python package for Calibrated Explanations, a local feature importance explanation method providing uncertainty quantification, supporting both [classification](https://doi.org/10.1016/j.eswa.2024.123154) and [regression](https://arxiv.org/abs/2308.16245).
The proposed method is based on Venn-Abers (classification & regression) and Conformal Predictive Systems (regression) and has the following characteristics:
* __Fast__, __reliable__, __stable__ and __robust__ __feature importance explanations__ for:
	- __Binary classification__ models ([read paper](https://doi.org/10.1016/j.eswa.2024.123154)).
	- __Multi-class classification__ models ([read paper](https://easychair.org/publications/preprint/rqdD)).
	- __Regression__ models ([read paper](https://arxiv.org/abs/2308.16245)).
		* Including __probabilistic explanations__ of the probability that the target exceeds a user-defined threshold.
		* With __difficulty adaptable explanations__ (conformal normalization).
* __Calibration of the underlying model__ to ensure that predictions reflect reality.
* __Uncertainty quantification__ of both the prediction from the underlying model and the feature importance weights. 
* __Proximity-based rules with straightforward interpretation__ in relation to instance values and feature weights.
* Possibility to generate __counterfactual rules with uncertainty quantification__ of the expected predictions.
* __Conjunctional rules__ conveying feature importance for the interaction of included features  (described in the [regression paper](https://arxiv.org/abs/2308.16245)).
* __Conditional rules__, allowing users the ability to create contextual explanations to handle e.g. bias and fairness constraints ([read paper](https://doi.org/10.1007/978-3-031-63787-2_17)). 

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
The [notebooks folder](https://github.com/Moffran/calibrated_explanations/tree/main/notebooks) contains a number of notebooks illustrating different use cases for `calibrated-explanations`. The [quickstart_wrap](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/quickstart_wrap.ipynb), using the `WrapCalibratedExplainer` class, is similar to this Getting Started, including plots and output.

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

feature_names = dataset.feature_names

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
print(f'Uncalibrated probability estimates: \n{classifier.predict_proba(X_test)}')
```

Before we can generate explanations, we need to calibrate our model using the calibration set. 


```python
classifier.calibrate(X_cal, y_cal, feature_names=feature_names)
display(classifier)
```

Once the model is calibrated, the `predict` and `predict_proba` methods produce calibrated predictions and probability estimates.


```python
proba, (low, high) = classifier.predict_proba(X_test, uq_interval=True)
print(f'Calibrated probability estimates: \n{proba}')
print(f'Calibrated uncertainty interval for the positive class: [{[(low[i], high[i]) for i in range(len(low))]}]')
```

#### Factual Explanations
Let us explain a test instance using our `WrapCalibratedExplainer` object. The method used to get factual explanations is `explain_factual`. 


```python
factual_explanations = classifier.explain_factual(X_test)
display(classifier)
```

Once we have the explanations, we can plot all of them using the `plot` function. Default, a regular plot, without uncertainty intervals included, is created. To include uncertainty intervals, change the parameter `uncertainty=True`. To plot only a single instance, the `plot` function can be called, submitting the index of the test instance to plot.


```python
factual_explanations.plot()
factual_explanations.plot(uncertainty=True)

factual_explanations.plot(0, uncertainty=True)
```

You can also add and remove conjunctive rules.


```python
factual_explanations.add_conjunctions().plot(0)
factual_explanations.plot(0, uncertainty=True)
factual_explanations.remove_conjunctions().plot(0, uncertainty=True)
```

#### Counterfactual Explanations
An alternative to factual rules is to extract counterfactual rules, which is done using the `explain_counterfactual` function. 


```python
counterfactual_explanations = classifier.explain_counterfactual(X_test)
display(classifier)
```

Counterfactuals are also visualized using the `plot` function. Plotting an individual counterfactual explanation is done using `plot`, submitting the index to plot. Adding or removing conjunctions is done as before. 


```python
counterfactual_explanations.plot()
counterfactual_explanations.add_conjunctions().plot()

counterfactual_explanations.plot(0)
```

`calibrated_explanations` supports multiclass which is demonstrated in [demo_multiclass](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_multiclass.ipynb). That notebook also demonstrates how both feature names and target and categorical labels can be added to improve the interpretability. 
### Regression
Extracting explanations for regression is very similar to how it is done for classification. First we load and divide the dataset. The target is divided by 1000, meaning that the target is in thousands of dollars. 


```python
dataset = fetch_openml(name="house_sales", version=3)

X = dataset.data.values.astype(float)
y = dataset.target.values/1000

feature_names = dataset.feature_names

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
print(f'Uncalibrated model prediction: \n{regressor.predict(X_test)}')
```

Before we can generate explanations, we need to calibrate our model using the calibration set. 


```python
regressor.calibrate(X_cal, y_cal, feature_names=feature_names)
display(regressor)
```

We can easily add a difficulty estimator by assigning a `DifficultyEstimator` to the `difficulty_estimator` attribute when calibrating the model.


```python
from crepes.extras import DifficultyEstimator

regressor.calibrate(X_cal, y_cal, feature_names=feature_names, 
                    difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, learner=regressor.learner, scaler=True))
display(regressor)
```

Once the model is calibrated, the `predict`  method produce calibrated predictions with uncertainties. The default confidence is 90 per cent, which can be altered using the `low_high_percentiles` parameter. 


```python
prediction, (low, high) = regressor.predict(X_test, uq_interval=True, low_high_percentiles=(5, 95))
print(f'Calibrated prediction: \n{prediction}')
print(f'Calibrated uncertainty interval: [{[(low[i], high[i]) for i in range(len(low))]}]')
```

You can also get the probability of the prediction being below a certain threshold using `predict_proba` by assigning the `threshold` parameter.


```python
prediction = regressor.predict(X_test, threshold=200)
print(f'Calibrated probabilistic prediction: {prediction}')

proba, (low, high) = regressor.predict_proba(X_test, uq_interval=True, threshold=200)
print(f'Calibrated probabilistic probability estimate [y_hat > threshold, y_hat <= threshold]: \n{proba}')
print(f'Calibrated probabilistic uncertainty interval for y_hat <= threshold: [{[(low[i], high[i]) for i in range(len(low))]}]')
```

#### Factual Explanations
Let us explain a test instance using our `WrapCalibratedExplainer` object. The method used to get factual explanations is `explain_factual`. 


```python
factual_explanations = regressor.explain_factual(X_test)
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

#### Counterfactual Explanations
The `explain_counterfactual` will work exactly the same as for classification. 


```python
counterfactual_explanations = regressor.explain_counterfactual(X_test)
display(regressor)
```

Counterfactual plots work as for classification.


```python
counterfactual_explanations.plot()
counterfactual_explanations.add_conjunctions().plot()
```

### Probabilistic Regression
The difference between probabilistic regression and regular regression is that the former returns a probability of the prediction being below a certain threshold. This could for example be useful when the prediction is a time to an event, such as time to death or time to failure. 


```python
probabilistic_factual_explanations = regressor.explain_factual(X_test, threshold=200)
probabilistic_factual_explanations.plot()
probabilistic_factual_explanations.plot(uncertainty=True)
```


```python
probabilistic_counterfactual_explanations = regressor.explain_counterfactual(X_test, threshold=200)
probabilistic_counterfactual_explanations.plot()
```

Regression offers many more options but to learn more about them, see the [demo_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_regression.ipynb) or the [demo_probabilistic_regression](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_probabilistic_regression.ipynb) notebooks.

### Alternatives

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

[Top](#calibrated-explanations-documentation)

Known Limitations
-----------------
The implementation currently only support numerical input. Use the `utils.helper.transform_to_numeric` (released in version v0.3.1) to transform a `DataFrame` with text data into numerical form and at the same time extracting `categorical_features`, `categorical_labels`, `target_labels` (if text labels) and `mappings` (used to apply the same mappings to new data) to be used as input to the `CalibratedExplainer`. The algorithm does not currently support image data.

See e.g. the [Conditional Fairness Experiment](evaluation/Conditional_Fairness_Experiment.ipynb) for examples on how it can be used.

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
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
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
If you use `calibrated-explanations` for a scientific publication, you are kindly requested to cite one of the following papers:

- [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom)., Johansson, U., and Sönströd, C. (2024). [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://doi.org/10.1016/j.eswa.2024.123154). Expert Systems with Applications, 1-27.
- [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U., Sönströd, C., and [Matela, R](https://github.com/rudymatela). [Calibrated Explanations for Regression](https://arxiv.org/abs/2308.16245). arXiv preprint arXiv:2308.16245. Accepted to Machine Learning. In press.
-  [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom). (2024). [Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty](https://doi.org/10.1007/978-3-031-63787-2_17). In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2153. Springer, Cham.
- [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U. (2024). [Calibrated Explanations for Multi-class](https://easychair.org/publications/preprint/rqdD). <i>Proceedings of the Thirteenth Workshop on Conformal and Probabilistic Prediction and Applications</i>, in <i>Proceedings of Machine Learning Research</i>. In press. 

The paper that originated the idea of `calibrated-explanations` is:

- [Löfström, H.](https://github.com/Moffran), [Löfström, T.](https://github.com/tuvelofstrom), Johansson, U., & Sönströd, C. (2023). [Investigating the impact of calibration on the quality of explanations](https://link.springer.com/article/10.1007/s10472-023-09837-2). Annals of Mathematics and Artificial Intelligence, 1-18. [Code and results](https://github.com/tuvelofstrom/calibrating-explanations).

If you use `calibrated-explanations` for a scientific publication, you are kindly requested to cite one of the papers above. Bibtex entries can be found in [citing](https://github.com/Moffran/calibrated_explanations/blob/main/docs/citing.md#bibtex-entries).
  
[Top](#calibrated-explanations-documentation)

Acknowledgements
----------------
This research is funded by the Swedish Knowledge Foundation together with industrial partners supporting the research and education environment on Knowledge Intensive Product Realization SPARK at Jönköping University, Sweden, through projects: AFAIR grant no. 20200223, ETIAI grant no. 20230040, and PREMACOP grant no. 20220187. Helena Löfström was a PhD student in the Industrial Graduate School in Digital Retailing (INSiDR) at the University of Borås, funded by the Swedish Knowledge Foundation, grant no. 20160035. 

[Rudy Matela](https://github.com/rudymatela) has been our git guru and has helped us with the release process.

We have used both the `ConformalPredictiveSystem` and `DifficultyEstimator` classes from [Henrik Boström](https://github.com/henrikbostrom)s [crepes](https://github.com/henrikbostrom/crepes) package to provide support for regression. The `MondrianCategorizer` class is also supported in the `WrapCalibratedExplainer` as an alternative to using the `bins` parameter to create conditional explanations.

We have used the `VennAbers` class from [Ivan Petej](https://github.com/ip200)s [venn-abers](https://github.com/ip200/venn-abers) package to provide support for probabilistic explanations (both classification and probabilistic regression). 

We are using Decision Trees from `scikit-learn` in the discretizers.

We have copied code from [Marco Tulio Correia Ribeiro](https://github.com/marcotcr)s [lime](https://github.com/marcotcr/lime) package for the `Discretizer` class.

The `check_is_fitted` and `safe_instance` functions in `calibrated_explanations.utils` are copied from `sklearn` and `shap`.  
  
[Top](#calibrated-explanations-documentation)

[build-log]:    https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml
[build-status]: https://github.com/Moffran/calibrated_explanations/actions/workflows/test.yml/badge.svg
[pypi-version]: https://img.shields.io/pypi/v/calibrated-explanations
[calibrated-explanations-on-pypi]: https://pypi.org/project/calibrated-explanations
