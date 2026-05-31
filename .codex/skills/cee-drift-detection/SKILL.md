---
name: cee-drift-detection
description: >
  Implement, configure, and test CEE drift detectors (KS, MMD, Martingale) and CompositeDriftDetector strategies for production model monitoring.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee Drift Detection — Core Instructions

# CEE Drift Detection

## Use this skill when
- Implementing a new drift detection method or extending existing detectors
- Configuring `DriftMonitoringStrategy` for a customer use case
- Debugging false positive or false negative drift alerts
- Writing tests for drift detection logic
- Asked "which drift strategy should I use for X scenario?"

## Inputs
- `packages/adaptive/src/.../semi_online/composite_drift.py`
- `packages/adaptive/src/.../semi_online/drift.py` (KS-test)
- `packages/adaptive/src/.../semi_online/feature_drift.py` (MMD)
- `packages/adaptive/src/.../semi_online/martingale_drift.py` (Martingale)
- `packages/adaptive/src/.../semi_online/config.py` (DriftPolicy, DriftMonitoringStrategy)
- `development/strategic-pillars/02_online_calibration.md`

## Drift Strategy Reference

| Strategy | Detectors Used | Cost | Best For |
|---|---|---|---|
| `FEATURE` | MMD only | High (feature-space) | Feature distribution shift |
| `PREDICTION` | KS-test on scores | Low | Prediction score distribution shift |
| `CALIBRATION` | Martingale on calibration quality | Low | Calibration degradation |
| `FEATURE_PREDICTION` | MMD + KS | Medium | Combined feature+prediction monitoring |
| `FEATURE_CALIBRATION` | MMD + Martingale | Medium | Feature shift + calibration quality |
| `PREDICTION_CALIBRATION` | KS + Martingale | Low | Prediction+calibration monitoring |
| `FULL` | All three | High | Maximum sensitivity |
| `AUTO` | KS + Martingale (= PREDICTION_CALIBRATION) | Low | Default; recommended starting point |

## Detector Implementations

### KS-test (drift.py)
- Tests: Kolmogorov-Smirnov statistic on prediction score distributions
- Trigger: p-value < `ks_pvalue_floor` (default 0.1)
- Input: baseline scores, current scores (1D arrays)

### MMD (feature_drift.py)
- Tests: Maximum Mean Discrepancy on feature space
- Input: baseline feature matrix, current feature matrix (2D arrays)
- Computationally expensive for high-dimensional features

### Martingale (martingale_drift.py)
- Tests: Conformal martingale on calibration quality stream
- Trigger: martingale value > `martingale_threshold` (default 100.0)
- Input: sequential calibration quality scores

## Workflow

### Implementing a new detector

1. Create detector class in a new file under `semi_online/`
2. Implement the interface expected by `CompositeDriftDetector`:
   ```python
   class MyDetector:
       def evaluate(
           self,
           baseline: np.ndarray,
           current: np.ndarray,
       ) -> DriftSignal:
           """Returns DriftSignal with triggered: bool and statistic: float"""
   ```
3. Add a new `DriftMonitoringStrategy` enum value in `config.py` if needed
4. Wire up in `CompositeDriftDetector._build_detectors()` or equivalent
5. Add `needs_features: bool` property if detector requires feature-space data

### Debugging drift alerts

1. Inspect `DriftResult.signals` — which detector triggered?
2. Check `DriftResult.statistic` values against configured thresholds
3. For false positives: increase `ks_pvalue_floor` or `martingale_threshold`
4. For false negatives: switch to a more sensitive strategy (e.g., `FULL`)

## Test Patterns

```python
# Test drift detector with synthetic shift
def test_ks_detects_shift_when_distributions_diverge():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 200)
    shifted = rng.normal(2, 1, 200)   # mean shift of 2 sigma

    detector = KSDriftDetector(pvalue_floor=0.1)
    result = detector.evaluate(baseline, shifted)

    assert result.triggered is True
    assert result.statistic > 0.0

# Test no drift with same distribution
def test_ks_no_false_positive_for_same_distribution():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 200)
    current = rng.normal(0, 1, 200)   # same distribution

    detector = KSDriftDetector(pvalue_floor=0.05)  # stricter threshold
    result = detector.evaluate(baseline, current)

    assert result.triggered is False
```

## Verification
```bash
pytest packages/adaptive/tests/semi_online/test_drift* -q
pytest tests/integration/ -m integration -q
ruff check packages/adaptive/
```

## Output contract
For implementation tasks, return:
1. New detector file(s) with complete implementation
2. Integration in `CompositeDriftDetector`
3. Unit tests covering: triggers on shift, no trigger on stable data, edge cases
4. Updated `DriftMonitoringStrategy` if new strategy added

## Constraints
- Drift detection must be independent of calibration engines (usable with both `crepes` and future `online-cp`)
- Never use wall-clock time in tests — seed all RNGs
- Detector implementations must be in `adaptive` package only (not `governance`)
- `CompositeDriftDetector` is internal — never expose it as public API
