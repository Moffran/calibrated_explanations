"""Minimal modality plugin fixture for ADR-033 packaging smoke tests."""


class MinimalModalityPlugin:
    """Minimal ExplainerPlugin satisfying the ADR-033 contract for vision modality."""

    plugin_meta = {
        "schema_version": 1,
        "name": "ce-mock-modality-plugin",
        "version": "0.1.0",
        "provider": "tests.fixtures",
        "capabilities": ("explain",),
        "modes": ("factual",),
        "tasks": ("classification",),
        "dependencies": (),
        "data_modalities": ("vision",),
        "plugin_api_version": "1.0",
        "trusted": False,
    }

    def supports(self, model):
        return True

    def explain(self, model, x, **kw):
        raise NotImplementedError
