from calibrated_explanations.plugins import registry as reg


def test_normalise_trust_various_forms():
    assert reg.normalise_trust({"trusted": True}) is True
    assert reg.normalise_trust({"trust": True}) is True
    assert reg.normalise_trust({"trust": {"trusted": True}}) is True
    assert reg.normalise_trust({"trust": {"default": True}}) is True
    assert reg.normalise_trust({}) is False


def test_env_trusted_names_and_clear_cache(monkeypatch):
    monkeypatch.setenv("CE_TRUST_PLUGIN", "a,b; c")
    names = reg.env_trusted_names()
    assert "a" in names and "b" in names and "c" in names
    # clearing should reset cached value
    reg.clear_env_trust_cache()
    monkeypatch.setenv("CE_TRUST_PLUGIN", "x")
    assert "x" in reg.env_trusted_names()


def test_should_trust_with_pyproject_cache(monkeypatch):
    # non-builtin requires explicit trust
    reg.set_pyproject_trust_cache_for_testing(["trusted-id"])
    meta = {"name": "trusted-id"}
    assert reg.should_trust(meta, identifier="trusted-id", source="entrypoint") is True
    # builtin is always trusted
    assert reg.should_trust({}, identifier="anything", source="builtin") is True
    # cleanup
    reg.set_pyproject_trust_cache_for_testing(None)


def test_update_and_propagate_trust_metadata():
    meta = {"name": "p", "provider": "prov"}

    class DummyPlugin:
        def __init__(self):
            self.plugin_meta = {}

    p = DummyPlugin()
    reg.update_trust_keys(meta, True)
    # propagate should write the keys back into plugin.plugin_meta
    reg.propagate_trust_metadata(p, meta)
    assert p.plugin_meta.get("trusted") is True
    assert p.plugin_meta.get("trust") is not None
