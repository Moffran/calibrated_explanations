# Migration Guide: v0.5.x → v0.6.0

This guide highlights changes relevant for upgrading to v0.6.0.

## Summary

- Public API surface unchanged (no new root exports). Adapters provide legacy-shape compatibility.
- New internal domain model for explanations; round-trip converters keep legacy outputs stable.
- JSON Schema v1 for Explanation added with optional validation utilities.
- Optional extras introduced (`viz`, `lime`, `notebooks`, `dev`, `eval`).

## Parameter aliases

Calls to public methods now emit a deprecation warning if using alias keys; behavior is unchanged for now. Prefer canonical names as documented. Warnings appear once per session.

## Serialization

Use `calibrated_explanations.serialization` utilities when you need portable JSON:

- `to_json(Explanation)` / `from_json(payload)`
- `validate_payload(payload)` (requires `jsonschema`)

See `docs/schema_v1.md` for field reference.

## Preprocessing

Wrapper supports a user-supplied preprocessor via `ExplainerConfig` and `_from_config`. Automatic encoding remains conservative; existing numeric pipelines are unaffected.

## Notebooks and viz

If you depend on plots or notebook authoring, install extras:

```powershell
pip install "calibrated_explanations[viz,notebooks]"
```

## FAQ

- Do I need to change how I access explanations? No; legacy dict output remains for public surfaces; internal model is used under the hood.
- Is the new `quick_explain` public? It lives under `calibrated_explanations.api.quick` and is not re-exported at root to avoid API snapshot churn.
