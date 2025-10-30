# Contributor hub

Help shape calibrated explanations by improving the core library, documentation,
and plugin ecosystem. Start with the plugin contract so every change preserves
classification and regression parity, then follow the extending guides to wire
CLI, configuration, and documentation updates.

## Core contributor path

1. Review the {doc}`plugin-contract` to understand the surfaces that must remain
   calibrated-first across explanations, intervals, and PlotSpec outputs.
2. Follow the {doc}`../foundations/governance/test_policy` so every Python
   example added to the docs or README gains a matching test in `tests/docs/`.
3. Set up your environment using the workflow in {doc}`../foundations/governance/pr_guide` and run
   the formatter, tests, and doc builds locally.
4. Use the {doc}`extending/guides/index` collection when adding new plugins or
   command-line tooling.

## Advanced contributor tooling

Need telemetry hooks or performance instrumentation for plugins? The
{doc}`extending/advanced` page summarises the optional extras and how they tie
back to governance requirements.

```{toctree}
:maxdepth: 1
:hidden:

plugin-contract
extending/guides/index
extending/advanced
```
