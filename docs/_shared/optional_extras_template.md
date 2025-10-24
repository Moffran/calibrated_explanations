### Optional extras

These integrations are **opt-in**—all primary quickstarts, notebooks, and
runtime APIs work without them. Enable only the tooling you need:

- **Telemetry & compliance logging**: Follow the
  {doc}`how-to/configure_telemetry` guide to emit
  optional provenance payloads for regulated environments. Skip this step when
  you only need calibrated explanations.
- **PlotSpec visualisation plugins**: Install PlotSpec styles with
  ``pip install "calibrated-explanations[plotspec]"`` and read the
  {doc}`viz_plotspec` guide to render calibrated factual
  and alternative plots. These styles stay optional for all workflows.
- **CLI & registry governance**: The plugin CLI is an opt-in helper for
  discovery and denylist management. Run ``python -m
  calibrated_explanations.plugins.cli --help`` only when you want to inspect
  plugin routing; it honours the ``CE_DENY_PLUGIN`` toggle.
- **External plugin bundle**: Fast explanations and other vetted extensions live
  in the external plugin lane. Install them with
  ``pip install "calibrated-explanations[external-plugins]"`` and review the
  {doc}`external_plugins/index` catalogue before enabling anything in
  production.

> 💡 Keep these extras at the end of the page so calibration-first workflows stay
> front and centre.
