# Decision Policies Guide

This guide helps practitioners convert Calibrated Explanations uncertainty outputs into actionable decision policies.

## From Uncertainty to Decisions

Calibrated Explanations provide uncertainty-aware outputs. This guide shows how to transform these into systematic decision policies for production systems.

## Common Decision Patterns

### 1. Confidence Threshold Policy

Act only when prediction confidence exceeds a threshold:

```python
probs, (low, high) = explainer.predict_proba(x_test, uq_interval=True)

# Only act when lower bound exceeds threshold
confidence_threshold = 0.7
confident_mask = low[:, 1] > confidence_threshold

# Route confident predictions to automation
automated = x_test[confident_mask]
# Route uncertain predictions to review
needs_review = x_test[~confident_mask]
```

**Use when**: False positives are costly and you need high confidence before acting.

### 2. Interval Width Policy

Reject predictions where uncertainty is too high:

```python
prediction, (low, high) = explainer.predict(x_test, uq_interval=True)
interval_width = high - low

# Define acceptable uncertainty
max_acceptable_width = 20  # domain-specific threshold

# Flag wide intervals for review
uncertain_mask = interval_width > max_acceptable_width
```

**Use when**: You need bounded uncertainty regardless of the prediction value.

### 3. Interval Straddle Policy (Classification)

Reject when the probability interval straddles the decision boundary:

```python
probs, (low, high) = explainer.predict_proba(x_test, uq_interval=True)
decision_boundary = 0.5

# Reject if interval contains the boundary
straddles = (low[:, 1] < decision_boundary) & (high[:, 1] > decision_boundary)

# These predictions could go either way
ambiguous = x_test[straddles]
```

**Use when**: The decision boundary matters and you can't tolerate ambiguity.

### 4. RejectPolicy Enum (Built-in)

Use the built-in `RejectPolicy` for systematic handling:

```python
from calibrated_explanations.core.reject.policy import RejectPolicy

# FLAG: Include all results with rejection status
result = explainer.explain_factual(
    x_test,
    reject_policy=RejectPolicy.FLAG
)

# Check rejection status
if hasattr(result, 'rejected'):
    flagged = result.rejected
    print(f"Flagged {flagged.sum()} instances for review")
```

Available policies:

| Policy | Behavior |
| :--- | :--- |
| `NONE` | No rejection logic (default) |
| `FLAG` | Include all, mark rejected instances |
| `ONLY_ACCEPTED` | Return only accepted predictions |
| `ONLY_REJECTED` | Return only rejected predictions |
| `RAISE` | Raise exception on rejection |

## Policy Selection Matrix

| Use Case | Recommended Policy | Implementation |
| :--- | :--- | :--- |
| Audit logging | `RejectPolicy.FLAG` | Log all with rejection status |
| Conservative automation | Confidence threshold | Only act on high confidence |
| Human-in-the-loop | `ONLY_REJECTED` | Route uncertain to experts |
| Full automation | Confidence + width | Multiple checks before acting |
| Regulatory compliance | Interval straddle | Never automate ambiguous cases |

## Combining Policies

For production systems, combine multiple policies:

```python
def make_decision(explainer, x, config):
    """Multi-stage decision policy."""
    probs, (low, high) = explainer.predict_proba(x, uq_interval=True)

    decisions = []
    for i in range(len(x)):
        prob = probs[i, 1]
        interval_width = high[i, 1] - low[i, 1]

        # Stage 1: Check interval width
        if interval_width > config['max_width']:
            decisions.append(('DEFER', 'high_uncertainty'))
            continue

        # Stage 2: Check if straddles boundary
        if low[i, 1] < 0.5 < high[i, 1]:
            decisions.append(('DEFER', 'ambiguous'))
            continue

        # Stage 3: Check confidence threshold
        if prob > config['accept_threshold']:
            decisions.append(('ACCEPT', 'confident_positive'))
        elif prob < config['reject_threshold']:
            decisions.append(('REJECT', 'confident_negative'))
        else:
            decisions.append(('DEFER', 'moderate_confidence'))

    return decisions

# Example configuration
config = {
    'max_width': 0.2,
    'accept_threshold': 0.7,
    'reject_threshold': 0.3
}

decisions = make_decision(explainer, x_test, config)
```

## Regression Decision Policies

For regression tasks, policies focus on interval bounds:

```python
prediction, (low, high) = explainer.predict(
    x_test,
    uq_interval=True,
    low_high_percentiles=(5, 95)  # 90% intervals
)

# Policy: Accept only if entire interval is within spec
spec_min, spec_max = 100, 200

within_spec = (low >= spec_min) & (high <= spec_max)
out_of_spec = (high < spec_min) | (low > spec_max)
uncertain = ~within_spec & ~out_of_spec  # interval crosses boundary
```

## Probabilistic Regression Policies

For threshold probability queries:

```python
# Query probability of meeting threshold
probs, (low, high) = explainer.predict_proba(
    x_test,
    threshold=150,  # P(y <= 150)
    uq_interval=True
)

# Policy: Accept if high probability of being within spec
# Reject if high probability of exceeding spec
# Defer if uncertain

accept_threshold = 0.8
reject_threshold = 0.2

accept = low[:, 1] > accept_threshold  # Confident y <= 150
reject = high[:, 1] < reject_threshold  # Confident y > 150
defer = ~accept & ~reject  # Uncertain
```

## Documenting Decision Policies

For audit trails and reproducibility:

```python
POLICY_DOC = """
Decision Policy: Credit Approval v2.1
=====================================
Date: 2024-01-15
Owner: Risk Team

Acceptance Criteria:
- Probability interval lower bound > 0.7
- Interval width < 0.15
- No interval straddle at 0.5 boundary

Deferral Criteria:
- Interval width > 0.15 OR
- Interval straddles 0.5 boundary

Rejection Criteria:
- Probability interval upper bound < 0.3

Calibration Requirements:
- Minimum 1000 calibration samples
- Refreshed monthly
"""
```

## Cross-References

* {doc}`../advanced/reject-policy` - Full RejectPolicy API documentation
* {doc}`ensured-explanations` - Uncertainty-based alternative filtering
* {doc}`../../tasks/capabilities` - Full capability manifest
* {doc}`../../improvement/adrs/ADR-029-reject-integration-strategy` - Reject architecture
