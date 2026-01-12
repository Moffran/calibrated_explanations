from calibrated_explanations.plugins.registry import (
    register_explanation_plugin,
    register_interval_plugin,
)
from calibrated_explanations.logging import get_logging_context


def test_should_set_logging_context_during_explanation_plugin_registration():
    class ContextCheckingPlugin:
        def __init__(self):
            self.captured_context = None

        @property
        def plugin_meta(self):
            # Capture context when registry accesses plugin_meta
            ctx = get_logging_context()
            if ctx.get("plugin_identifier"):  # Only capture if set
                self.captured_context = ctx
            return {"name": "test_plugin", "provider": "test"}

    checking_plugin = ContextCheckingPlugin()

    # In the current registry, register_explanation_plugin takes (identifier, plugin)
    try:
        register_explanation_plugin("test.test_plugin", checking_plugin)
    except Exception:
        pass

    assert checking_plugin.captured_context is not None
    assert checking_plugin.captured_context.get("plugin_identifier") == "test.test_plugin"


def test_should_set_logging_context_during_interval_plugin_registration():
    class ContextCheckingPlugin:
        def __init__(self):
            self.captured_context = None

        @property
        def plugin_meta(self):
            ctx = get_logging_context()
            if ctx.get("plugin_identifier"):
                self.captured_context = ctx
            return {"name": "test_interval", "provider": "test"}

    checking_plugin = ContextCheckingPlugin()
    try:
        register_interval_plugin("test.test_interval", checking_plugin)
    except Exception:
        pass

    assert checking_plugin.captured_context is not None
    assert checking_plugin.captured_context.get("plugin_identifier") == "test.test_interval"


def test_should_clear_logging_context_after_registration():
    class DummyPlugin:
        @property
        def plugin_meta(self):
            return {"name": "dummy", "provider": "test"}

    try:
        register_explanation_plugin("test.dummy", DummyPlugin())
    except Exception:
        pass

    ctx = get_logging_context()
    assert ctx.get("plugin_identifier") is None
