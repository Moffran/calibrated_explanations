# Calibrated Explanations documentation

Use this site by audience and by task.

The public source, issue tracker, tags, and releases are maintained in
[`Moffran/calibrated_explanations`](https://github.com/Moffran/calibrated_explanations).
Install published packages from
[PyPI](https://pypi.org/project/calibrated-explanations/), report problems in the
[public issue tracker](https://github.com/Moffran/calibrated_explanations/issues),
and read the
[release notes](https://github.com/Moffran/calibrated_explanations/blob/main/CHANGELOG.md).

```{admonition} 1.0.0 stable release
:class: important

The public API, plugin trust model, and visualization extension contract are
stable as of `1.0.0`; only patch-level defect fixes will be accepted against
these contracts going forward. Install with
`pip install calibrated-explanations`. Before adoption, review the
{doc}`v1.0.0 upgrade checklist <upgrade/v1.0.0-upgrade-checklist>`, the
{doc}`Explanation Schema v1 freeze <schema_v1>`, and the
[release notes](https://github.com/Moffran/calibrated_explanations/blob/main/CHANGELOG.md).
```

```{admonition} Open source, demo, and commercial solutions
:class: note

Calibrated Explanations is an independent, BSD-licensed open-source project.
For optional commercial integration, operational deployment or pilot discussions,
[visit Grejna AB](https://grejna.ai/?utm_source=ce_docs&utm_medium=referral&utm_campaign=ce_ecosystem).
These services are separate from the open-source library and its license.

For simulated learning scenarios, [explore Decision Labs](https://games.grejna.ai/).
```

```{admonition} Try Calibrated Explanations in your browser
:class: tip

[Open the interactive web demo](https://calibrated-explanations-demo.ju.se/)
to explore a guided workflow or use the expert interface without installing the
package locally.

The demo is illustrative. Use this versioned documentation as the authority for
supported APIs, behavior, assumptions, and guarantees.
```

- New users start in {doc}`get-started/index`.
- Practitioners use {doc}`practitioner/index`.
- Researchers use {doc}`researcher/index`.
- Contributors use {doc}`contributor/index`.

Semantics are mode-specific. Classification, interval regression, and
probabilistic regression do not share one guarantee statement. For semantics,
assumptions, and non-guarantees, use
{doc}`foundations/concepts/calibrated_interval_semantics`.

## Companion projects

Use {doc}`plugins` to find optional companion extensions and
{doc}`researcher/replication/index` to reproduce published studies with their
version-specific environments.

```{toctree}
:maxdepth: 1
:caption: Start here

get-started/index
practitioner/index
researcher/index
contributor/index
```

```{toctree}
:maxdepth: 1
:caption: Core references

foundations/index
tasks/index
api/index
```

```{toctree}
:maxdepth: 1
:caption: Companion projects

plugins
researcher/replication/index
```

```{toctree}
:maxdepth: 1
:caption: Migration and project docs

migration/index
upgrade/index
appendices/changelog_links
appendices/rtd_tier_map
get-started/ce_first_agent_guide
get-started/copilot-setup
maintenance/legacy-plotting-reference
compare
ROADMAP
citing
```

Entry-point tier: Tier 1.
