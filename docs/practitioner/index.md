# Practitioner hub

Practitioners adopt calibrated explanations to deliver trustworthy models while
keeping day-to-day workflows lightweight. Start with the parity quickstarts,
anchor your interpretation in the shared foundations, and only pull in optional
telemetry or performance tooling once your core checks pass.

## Core workflow

1. Run the {doc}`../get-started/quickstart_classification` and
   {doc}`../get-started/quickstart_regression` tutorials to validate calibrated
   explanations across classification and regression.
2. Apply the {doc}`../foundations/how-to/interpret_explanations` checklist to
   review factual and alternative narratives before deployment.
3. Consult the {doc}`task_api_comparison` to choose between the wrapper API
   and direct CalibratedExplainer API for your use case.
4. When integrating with existing systems, follow the
   {doc}`playbooks/index` playbooks to keep exports and governance aligned.

## Advanced lane

Keep optional extras in reserve until the calibrated workflows above are in
place. The {doc}`advanced/index` catalogue groups telemetry, performance, and
PlotSpec guidance so you can adopt them deliberately without overwhelming
stakeholders who only need the core explanations.

```{toctree}
:maxdepth: 1
:hidden:

task_api_comparison
playbooks/index
advanced/index
```
