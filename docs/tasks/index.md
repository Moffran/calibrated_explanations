# Tasks & capabilities

Calibrated Explanations supports three primary task types, each with canonical API signatures and calibration mechanisms.

| Task | Core Class | Capability |
| :--- | :--- | :--- |
| **{doc}`Classification <classification>`** | `VennAbers` | Calibrated probabilities + uncertainty intervals |
| **{doc}`Conformal Regression <regression>`** | `ConformalPredictiveSystem` | Point estimates + CPS-conformal intervals |
| **{doc}`Probabilistic Regression <probabilistic_regression>`** | `ConformalPredictiveSystem` + `VennAbers` | Thresholded probability queries on numeric targets |

All tasks support the core explanation methods:
- `explain_factual(x)`
- `explore_alternatives(x)`

```{toctree}
:maxdepth: 1

classification
regression
probabilistic_regression
```
