---
name: cee-parity-test
description: >
  Author and validate CEE numerical parity tests that verify enterprise wrappers preserve exact OSS CE mathematical behaviour for regression, binary, and multiclass tasks.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee Parity Test — Core Instructions

# CEE Parity Testing

## Use this skill when
- Writing new parity tests for a CEE wrapper class
- Debugging a parity test failure (enterprise output diverges from OSS CE)
- Checking whether a change to adaptive or governance could break numerical parity
- Running the parity suite before a PR merge
- Interpreting `np.allclose` tolerance failures in parity tests

## Inputs
- `tests/parity/` — existing parity test suite
- `packages/adaptive/src/.../calibrated_adaptive_explainer.py` — enterprise wrapper
- `packages/governance/src/.../calibrated_governance_explainer.py` — governance wrapper
- `AGENTS.md` §"Parity assertions" under Test Conventions

## Parity Tolerance Rules

| Context | Tolerance | Assertion |
|---|---|---|
| Offline CE calls (no online update) | `rtol=0, atol=1e-10` | `np.allclose(enterprise, oss, rtol=0, atol=1e-10)` |
| Online/recalibrated outputs | `rtol=0, atol=1e-6` | `np.allclose(enterprise, oss, rtol=0, atol=1e-6)` |

**Never use** `rtol=1e-5` or other looser tolerances unless explicitly justified.

## Parity Test Structure

```python
"""
Parity tests verify that enterprise wrappers produce numerically identical
outputs to the OSS calibrated-explanations library for the same inputs.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations_enterprise.adaptive import CalibratedAdaptiveExplainer


@pytest.fixture
def binary_classification_data():
    """Deterministic binary classification dataset."""
    X, y = load_iris(return_X_y=True)
    # Binary: class 0 vs rest
    y_binary = (y == 0).astype(int)
    return X[:100], y_binary[:100], X[100:130], y_binary[100:130]


def test_adaptive_explainer_predict_proba_matches_oss_binary(binary_classification_data):
    """Enterprise predict_proba output must match OSS exactly for binary classification."""
    X_proper, y_proper, X_cal, y_cal = binary_classification_data
    X_test = X_cal[:5]

    # OSS CE baseline
    oss = WrapCalibratedExplainer(RandomForestClassifier(random_state=42))
    oss.fit(X_proper, y_proper)
    oss.calibrate(X_cal, y_cal)
    oss_proba = oss.predict_proba(X_test)

    # Enterprise wrapper
    enterprise = CalibratedAdaptiveExplainer(
        RandomForestClassifier(random_state=42)
    )
    enterprise.fit(X_proper, y_proper)
    enterprise.calibrate(X_cal, y_cal)
    enterprise_proba = enterprise.predict_proba(X_test)

    assert np.allclose(enterprise_proba, oss_proba, rtol=0, atol=1e-10), (
        f"predict_proba diverged: max diff = {np.abs(enterprise_proba - oss_proba).max()}"
    )
```

## Existing Parity Test Files

| File | Coverage |
|---|---|
| `tests/parity/test_regression_parity.py` | Regression: predict, uncertainty intervals |
| `tests/parity/test_binary_classification_parity.py` | Binary: predict_proba, explain_factual |
| `tests/parity/test_multiclass_parity.py` | Multiclass: predict_proba, class handling |

## Workflow

### Diagnosing a parity failure

1. Run the failing test with verbose output: `pytest tests/parity/ -v -s`
2. Inspect the diff:
   ```python
   diff = enterprise_output - oss_output
   print(f"Max abs diff: {np.abs(diff).max()}")
   print(f"Mean abs diff: {np.abs(diff).mean()}")
   ```
3. If diff > 1e-10: enterprise wrapper is altering OSS behaviour — this is a bug
4. If diff is exactly 0 everywhere: check that the enterprise wrapper actually called OSS (not a mock)
5. Check for different `random_state` seeds, different calibration splits, or different data slicing

### Adding parity for a new enterprise method

1. Identify the OSS CE method being wrapped
2. Write a parity test following the template above
3. Add to the appropriate parity test file (or create a new one for new task types)
4. Tolerance: use `atol=1e-10` unless the method involves online updates

## Verification
```bash
pytest tests/parity/ -q          # must all pass
pytest tests/parity/ -v          # verbose output for debugging
pytest tests/parity/ --tb=short  # short tracebacks on failure
```

## Output contract
For parity test authoring, return:
1. Complete test file with docstring explaining what is being checked
2. `atol` value with justification (1e-10 for offline, 1e-6 for online)
3. Assertion message showing the actual max diff on failure
4. Confirmation all tests pass with `pytest tests/parity/ -q`

## Constraints
- Every parity test must have a deterministic OSS CE baseline (fixed `random_state`, fixed data split)
- Never use `MagicMock` for the OSS CE component in parity tests
- Never compare to a previously-saved golden file unless the golden file was generated by OSS CE itself
- Parity tests must be in `tests/parity/` not `tests/unit/`
- Add `@pytest.mark.parity` marker if creating tests that should run separately from unit tests