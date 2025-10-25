# Concepts & architecture

{{ hero_calibrated_explanations }}

Understand the theory powering calibrated explanations across binary and
multiclass classification plus probabilistic and interval regression. Start with
interpretation and alternatives, then dive into telemetry and architecture.

For proofs, benchmarks, and citations that underpin these concepts, visit the {doc}`../research/index` hub before diving into individual guides.

| Concept | Why it matters |
| --- | --- |
| [Interpret explanations](../how-to/interpret_explanations.md) | Reuses notebook screenshots and walks through dual uncertainty plus the triangular plot. |
| [Probabilistic & interval regression](probabilistic_regression.md) | Shows how calibrated probabilities and interval regression stay in lockstep across quickstarts and notebooks. |
| [Alternatives & triangular plots](alternatives.md) | Explains how the triangular view pairs with rule tables for calibrated alternatives. |
| [Telemetry & observability](telemetry.md) | Documents provenance fields and opt-in instrumentation. |
| [Architecture overview](../architecture.md) | Connects runtime components, caching, and plugin guardrails. |

```{toctree}
:maxdepth: 1

probabilistic_regression
alternatives
../architecture
telemetry
../error_handling
```

{{ optional_extras_template }}
