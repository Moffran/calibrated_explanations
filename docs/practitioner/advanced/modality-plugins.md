# Modality plugin selection

Use modality-aware plugin selection when you want to choose explanation plugins
for non-tabular workflows while keeping CE-first behavior unchanged.

## Canonical modalities and aliases

The modality resolver accepts canonical names and normalises aliases:

| Canonical | Aliases |
| --- | --- |
| `tabular` | none |
| `vision` | `image`, `images`, `img` |
| `audio` | none |
| `text` | none |
| `timeseries` | `time_series`, `time-series` |
| `multimodal` | `multi-modal`, `multi_modal` |

The same alias map applies to API and CLI modality inputs.

## CE-first invariant

Modality extension is additive. Existing `CalibratedExplainer` tabular workflows
are unchanged.

`find_explanation_plugin_for` is an opt-in helper in
`calibrated_explanations.plugins.registry`; it is not called automatically by
`CalibratedExplainer`.

## Resolve a plugin by modality

Use `find_explanation_plugin_for` when you want deterministic modality-aware
selection for a given `mode` and `task`.

```python
from calibrated_explanations.plugins.registry import find_explanation_plugin_for

identifier, plugin = find_explanation_plugin_for(
    modality="vision",
    mode="factual",
    task="classification",
    model=clf,
)
print(identifier)
```

Alias inputs resolve to the same canonical modality. For example,
`modality="image"` resolves exactly as `modality="vision"`.

## CLI filtering by modality

Use the plugin CLI to inspect only plugins matching a modality:

```bash
python -m calibrated_explanations.plugins.cli list all --modality vision
```

Alias filtering behaves the same:

```bash
python -m calibrated_explanations.plugins.cli list all --modality image
```

## Extension shim imports and availability checks

Optional extension shims are import-triggered only:

- `calibrated_explanations.vision`
- `calibrated_explanations.audio`

When the companion package is not installed, import raises
`MissingExtensionError`, which is an `ImportError` subclass. The recommended
availability check pattern is therefore:

```python
try:
    import calibrated_explanations.vision as ce_vision
except ImportError:
    ce_vision = None
```

Install companion packages when needed:

- `pip install calibrated-explanations[vision]`
- `pip install calibrated-explanations[audio]`

## Migration timeline

| Version | Behaviour |
| --- | --- |
| `v0.11.0` | `data_modalities` defaults to `("tabular",)` with no warning |
| `v0.11.1` | Missing `data_modalities` in entry-point metadata emits `DeprecationWarning` |
| `v0.12.0/v1.0.0-rc` | Explicit `data_modalities` is required; default fallback removed |

## Related references

- {doc}`../../contributor/plugin-contract`
- {doc}`../../contributor/extending/plugin-advanced-contract`
- {doc}`use_plugins`
- [ADR-033 - modality extension plugin contract and packaging](https://github.com/Moffran/calibrated_explanations/blob/main/docs/improvement/adrs/ADR-033-modality-extension-plugin-contract-and-packaging.md)
