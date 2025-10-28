# External plugins

The external plugin index tracks optional extensions that remain outside the
core package. Each entry must:

- Reuse the calibrated prediction bridge and respect ADR-024/025/026 guardrails.
- Highlight binary & multiclass classification plus probabilistic and interval
  regression coverage.
- Document optional telemetry or compliance hooks inside the `Optional extras`
  section of the hosting page.

## Vetted plugins

| Identifier | Summary | Install | Notes |
| --- | --- | --- | --- |
| `core.explanation.fast` / `core.interval.fast` | FAST explanations and interval calibrators packaged as an opt-in bundle. | `pip install "calibrated-explanations[external-plugins]"` then `external_plugins.fast_explanations.register()` | Treat as an optional speed-up; probabilistic regression parity remains the headline capability. |

## Community submissions

This table is reserved for community-maintained plugins. Open an issue with the
plugin metadata, ADR alignment notes, and calibration guarantees to request
listing.

| Identifier | Contact | Status |
| --- | --- | --- |
| _TBD_ | _community_ | _Pending_ |

### Optional: aggregated install extra

`pip install "calibrated-explanations[external-plugins]"` installs every curated
external plugin plus the pinned versions of ``numpy``, ``pandas``, and ``scikit-learn`` required by FAST mode. Import the relevant module and call its
`register()` helper to populate the registry.

### Optional: telemetry disclosure

External plugins should clearly mark telemetry emission as opt-in and link back
to :doc:`../governance/optional_telemetry` whenever instrumentation is enabled.

{{ optional_extras_template }}
