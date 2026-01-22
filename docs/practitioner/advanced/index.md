# Practitioner advanced topics

Reserve these references for teams that need optional telemetry, performance
profiling, or PlotSpec visual narratives. Each topic extends the core
explanation workflow without introducing mandatory dependencies.

- {doc}`use_plugins` - Opt in to curated external plugins (e.g., FAST) and wire them safely.
- {doc}`../../foundations/how-to/configure_telemetry` - Opt-in telemetry hooks
  with privacy defaults and governance cross-links.
- {doc}`../../foundations/how-to/tune_runtime_performance` - Parallelism,
  caching, and profiling guidance for high-throughput deployments.
- {doc}`reject-policy` - Configure `RejectPolicy` defaults, per-call overrides,
  and envelopes for reject-aware predictions and explanations.
- {doc}`parallel_execution_playbook` - Heuristics for switching between
  sequential, feature-, and instance-parallel explain strategies.
- {doc}`../../foundations/how-to/plot_with_plotspec` - Optional PlotSpec and
  triangular plot tooling to visualise factual vs. alternative trade-offs.
- {doc}`../../foundations/concepts/telemetry` - Conceptual framing for the
  telemetry model and its boundaries.

```{toctree}
:maxdepth: 1
:hidden:

use_plugins
../../foundations/how-to/configure_telemetry
../../foundations/how-to/tune_runtime_performance
reject-policy
parallel_execution_playbook
../../foundations/how-to/plot_with_plotspec
../../foundations/concepts/telemetry
../performance-tuning
```
