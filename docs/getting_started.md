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

Once we have the explanations, we can plot all of them using `plot`. Default, a regular plot, without uncertainty intervals included, is created. To include uncertainty intervals, change the parameter `uncertainty=True`. To plot only a single instance, the `plot` function can be called, submitting the index of the test instance to plot.


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

#### Alternative Explanations
An alternative to factual rules is to extract alternative rules, which is done using the `explore_alternatives` function. 


```python
alternative_explanations = classifier.explore_alternatives(X_test)
display(classifier)
```

Alternatives are also visualized using the `plot`. Plotting an individual alternative explanation is done using `plot`, submitting the index to plot. Adding or removing conjunctions is done as before. 


```python
alternative_explanations.plot()
alternative_explanations.add_conjunctions().plot()

alternative_explanations.plot(0)
```

`calibrated_explanations` supports multiclass which is demonstrated in [demo_multiclass](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_multiclass.ipynb). That notebook also demonstrates how both feature names and target and categorical labels can be added to improve the interpretability. 
### Regression
Extracting explanations for regression is very similar to how it is done for classification. First we load and divide the dataset. The target is divided by 1000, meaning that the target is in thousands of dollars. 


```python
dataset = fetch_openml(name="house_sales", version=3)

X = dataset.data.values.astype(float)
y = dataset.target.values/1000
y_filter = y < 400
X = X[y_filter,:]
y = y[y_filter]

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

#### Alternative Explanations
The `explore_alternatives` will work exactly the same as for classification. 


```python
alternative_explanations = regressor.explore_alternatives(X_test)
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