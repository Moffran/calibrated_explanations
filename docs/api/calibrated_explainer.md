# Calibrated Explainer API

This reference documents the public API for the `calibrated_explanations` package. It includes the main explainer classes, the explanation containers, and the specific explanation types.

## Core Explainers

The core of the library is the `CalibratedExplainer`, which handles the calibration of the underlying model. For a scikit-learn compatible interface, use `WrapCalibratedExplainer`.

### CalibratedExplainer

The `CalibratedExplainer` is the core class of the library. It takes a machine learning model (classifier or regressor) and a calibration dataset. It fits Venn-Abers calibrators (for classification) or Conformal Predictive Systems (for regression) to the model's predictions. This process ensures that the explanations generated are calibrated, meaning the predicted probabilities or intervals reflect the true underlying uncertainty.

**Example Usage:**

```python
from calibrated_explanations import CalibratedExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize explainer
explainer = CalibratedExplainer(model, X_cal, y_cal, mode='classification')

# Explain a test instance
X_test = X_cal[:1]
explanations = explainer.explain(X_test)
```

`{eval-rst}
.. autoclass:: calibrated_explanations.core.calibrated_explainer.CalibratedExplainer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
`

### WrapCalibratedExplainer

The `WrapCalibratedExplainer` acts as a wrapper around `CalibratedExplainer` to provide a standard scikit-learn interface (`fit`, `predict`, `predict_proba`). It simplifies the workflow by handling the splitting of data into calibration and training sets (if needed) and managing the underlying `CalibratedExplainer` instance.

**Example Usage:**

```python
from calibrated_explanations import WrapCalibratedExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Initialize wrapper with a base model
model = RandomForestClassifier()
wrapper = WrapCalibratedExplainer(model)

# Fit and calibrate (automatically handles data splitting if not provided)
wrapper.fit(X, y)

# Explain
X_test = X[:1]
explanations = wrapper.explain(X_test)
```

`{eval-rst}
.. autoclass:: calibrated_explanations.core.wrap_explainer.WrapCalibratedExplainer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
`

## Explanation Containers

When you request explanations for a set of instances, the result is returned in a `CalibratedExplanations` object. This collection holds the individual explanations and provides methods for visualization and export.

### CalibratedExplanations

A `CalibratedExplanations` object is returned when you call `explain()` on a `CalibratedExplainer`. It is a collection of explanations for the provided test instances. It serves as a container that allows you to iterate over individual explanations, visualize them, or export them.

**Example Usage:**

```python
# Assuming 'explanations' is a CalibratedExplanations object
# Iterate over explanations
for explanation in explanations:
    print(explanation.prediction)

# Plot all explanations
explanations.plot()

# Get a specific explanation
first_explanation = explanations[0]
```

`{eval-rst}
.. autoclass:: calibrated_explanations.explanations.explanations.CalibratedExplanations
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
`

## Explanation Types

Individual explanations are represented by specific classes depending on the type of explanation requested.

### CalibratedExplanation

This is the abstract base class for a single explanation. It contains the instance data, the prediction, and the feature weights (rules) that explain the prediction. It provides methods for plotting the explanation.

`{eval-rst}
.. autoclass:: calibrated_explanations.explanations.explanation.CalibratedExplanation
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
`

### FactualExplanation

A `FactualExplanation` explains *why* a model made a specific prediction for an instance. It identifies the feature values (conditions) that, if changed, would most likely alter the prediction. It focuses on the features present in the instance.

**Example Usage:**

```python
# Plot a factual explanation
factual_explanation.plot()

# Add a conjunction to the explanation
factual_explanation.add_conjunctions()
```

`{eval-rst}
.. autoclass:: calibrated_explanations.explanations.explanation.FactualExplanation
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
`

### AlternativeExplanation

An `AlternativeExplanation` (often called a counterfactual explanation) explores *what if* scenarios. It suggests changes to the feature values that would result in a different prediction (e.g., flipping a classification label or moving a regression prediction into a different range).

**Example Usage:**

```python
# Plot an alternative explanation
alternative_explanation.plot()
```

`{eval-rst}
.. autoclass:: calibrated_explanations.explanations.explanation.AlternativeExplanation
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
`
