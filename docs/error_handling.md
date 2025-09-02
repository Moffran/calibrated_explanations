# Error handling and validation

This package raises a small set of library-specific exceptions to make failures predictable and easy to handle. They are conservative in Phase 1B (no behavior changes to successful paths) and surfaced early during input checking.

## Key exceptions

- `calibrated_explanations.core.exceptions.ValidationError`: Inputs or configuration failed validation.
- `calibrated_explanations.core.exceptions.DataShapeError`: Shapes or dtypes are incompatible (e.g., X/y length mismatch, non-2D X).
- `calibrated_explanations.core.exceptions.ConfigurationError`: Conflicting or invalid parameter combinations.
- `calibrated_explanations.core.exceptions.ModelNotSupportedError`: Model lacks required methods.
- `calibrated_explanations.core.exceptions.NotFittedError`: Operation requires a fitted explainer or estimator.
- `calibrated_explanations.core.exceptions.ConvergenceError`: Optimization or calibration failed to converge.
- `calibrated_explanations.core.exceptions.SerializationError`: Failure to (de)serialize artifacts.

## Validation helpers

- `calibrated_explanations.core.validation.validate_inputs_matrix(X, y, ...)`: Ensures 2D X, matching y, and finiteness.
- `calibrated_explanations.core.validation.validate_model(model)`: Checks for a `predict` method and defers finer checks to call sites.
- `calibrated_explanations.core.validation.validate_fit_state(obj)`: Ensures the object is in a fitted state where required.
- `calibrated_explanations.core.validation.infer_task(...)`: Heuristic task inference based on model/y.

## Parameter canonicalization

- `calibrated_explanations.api.params.canonicalize_kwargs(kwargs)`: Maps known aliases to canonical keys without removing the originals (Phase 1B is no-op on unknowns).
- `calibrated_explanations.api.params.validate_param_combination(kwargs)`: Reserved for future stricter checks; a no-op in Phase 1B.

## Usage tips

- Callers should prefer canonical keys when reading kwargs.
- Catch `ValidationError` (or specific subclasses) at integration boundaries to provide user-friendly messages.
- For libraries that consume this package, prefer catching these exceptions over generic `Exception` to avoid masking other issues.
