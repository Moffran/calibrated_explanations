---
name: ce-docstring-author
description: >
  Write or repair NumPy-style docstrings for public CE APIs following STD-002 and
  contributor documentation rules.
---

# CE Docstring Author

You are writing or fixing Numpy-style docstrings for CE source code.
Numpy style is mandatory for all public symbols in `src/calibrated_explanations/`.

---

## Canonical section order

```python
def function_name(param1: TypeA, param2: TypeB, *, kwparam: TypeC = default) -> ReturnType:
    """Short one-line summary (imperative mood, no trailing period).

    Optional extended description: one or more paragraphs that explain *why*
    and *when* to use this function, not just *what* it does. Wrap at 88 chars.

    Parameters
    ----------
    param1 : TypeA
        Description of param1. Use "of shape (n, m)" for arrays.
    param2 : TypeB
        Description of param2.
    kwparam : TypeC, optional
        Description. State the default clearly. Default ``default``.

    Returns
    -------
    ReturnType
        Description of the return value. Name the variable if the caller
        needs to index it (e.g. ``ndarray of shape (n_samples, n_features)``).

    Raises
    ------
    ValidationError
        If ``param1`` does not satisfy the expected contract.
    ConfigurationError
        If the explainer is not calibrated before calling this method.

    Notes
    -----
    Technical background, algorithmic details, or references go here.
    Math can use inline LaTeX: :math:`p(y|x)`.

    References
    ----------
    .. [1] Author, Title, Journal, Year. https://doi.org/...

    Examples
    --------
    >>> from calibrated_explanations import WrapCalibratedExplainer
    >>> explainer = WrapCalibratedExplainer(RandomForestClassifier())
    >>> explainer.fit(X_proper, y_proper)
    >>> explainer.calibrate(X_cal, y_cal)
    >>> explanations = explainer.explain_factual(X_query)
    """
```

---

## Section presence rules

| Section | When required |
|---|---|
| One-line summary | Always |
| `Parameters` | When there are ≥ 1 parameters |
| `Returns` | When the function returns a non-None value |
| `Raises` | When the function raises documented exceptions |
| `Notes` | Optional — use for non-obvious behaviour |
| `References` | Optional — when citing papers or ADRs |
| `Examples` | Strongly encouraged for public API; required for CE-First entry points |

---

## Type annotation format (in docstrings)

```
param : np.ndarray of shape (n_samples, n_features)
param : int or None, optional
param : {'regular', 'triangular', 'ensured'}, optional  ← enum choices
param : list of str
param : Mapping[str, Any]
param : CalibratedExplanations
param : float, optional. Default 0.5.
```

---

## Common CE-specific patterns

### CalibratedExplanations return

```python
Returns
-------
CalibratedExplanations
    Collection of per-instance factual explanations. Access individual
    explanations with ``explanations[i]``.
```

### Prediction interval return

```python
Returns
-------
dict
    Keys: ``'predict'`` (point estimate), ``'low'`` (lower bound),
    ``'high'`` (upper bound). Invariant: ``low <= predict <= high``.
```

### threshold parameter

```python
threshold : float or tuple of float, optional
    Decision threshold(s) for probabilistic or thresholded regression.
    Pass a single float for one-sided; pass ``(low, high)`` for two-sided.
    See `ce-regression-intervals` for full semantics.
```

### expertise_level parameter

```python
expertise_level : str or tuple of str, optional
    Narrative verbosity level(s). One of ``'beginner'``, ``'intermediate'``, ``'advanced'``,
    or a tuple for both. Default ``('beginner', 'advanced')``.
```

---

## Class docstring

```python
class WrapCalibratedExplainer:
    """Scikit-learn compatible wrapper for calibrated explanations.

    Implements the CE-First pipeline: fit → calibrate → explain.
    For most use cases prefer this class over ``CalibratedExplainer`` directly.

    Parameters
    ----------
    model : sklearn estimator
        An unfitted or fitted scikit-learn compatible model. Must implement
        ``fit`` and ``predict_proba`` (classification) or ``predict`` (regression).
    difficulty_estimator : DifficultyEstimator, optional
        Custom difficulty estimator for Mondrian calibration. Defaults to
        the standard KNN-based estimator.

    Attributes
    ----------
    fitted : bool
        True after ``fit()`` has been called successfully.
    calibrated : bool
        True after ``calibrate()`` has been called successfully.

    Examples
    --------
    >>> from calibrated_explanations import WrapCalibratedExplainer
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> explainer = WrapCalibratedExplainer(RandomForestClassifier())
    >>> explainer.fit(X_proper, y_proper)
    >>> explainer.calibrate(X_cal, y_cal)
    """
```

---

## Deprecation in docstrings

When a parameter is deprecated, add a `.. deprecated::` directive in its entry:
```
param : int, optional
    .. deprecated:: 0.11.0
        Use ``new_param`` instead. Will be removed in v0.13.0.
```

---

## Running docstring coverage

```bash
python scripts/quality/check_docstring_coverage.py src/
# or check docstring_coverage.txt in the repo root
```

The coverage gate is tracked in `docstring_coverage.txt`. Public members with
missing docstrings appear as failures in CI.

---

## Anti-patterns

```python
# ❌ Google style
def fit(self, X, y):
    """
    Args:
        X: feature matrix
        y: label vector
    """

# ❌ reStructuredText (Sphinx :param:) style
def fit(self, X, y):
    """
    :param X: feature matrix
    :type X: np.ndarray
    """

# ❌ No type in docstring when type hints are present but vague
param : Any  # missing concrete description

# ❌ Describing implementation instead of contract
"""Loops through X and calls _internal() for each row."""
```

---

## Evaluation Checklist

- [ ] Summary is one line, imperative mood, no trailing period.
- [ ] `Parameters` section present for all non-trivial inputs.
- [ ] Types match the function signature (or are more specific when `Any` is used).
- [ ] `Returns` section describes the return value and its shape/keys.
- [ ] `Raises` documents all exceptions the caller must handle.
- [ ] `Examples` section present for CE-First entry points.
- [ ] Deprecated parameters annotated with `.. deprecated:: <version>`.
- [ ] Section order matches the canonical order above.
