import pytest

try:
    from calibrated_explanations.viz import builders, serializers, matplotlib_adapter
    from calibrated_explanations.viz.plotspec import SaveBehavior
except ModuleNotFoundError:
    pytest.skip("viz not available", allow_module_level=True)

pytestmark = pytest.mark.viz




def test_triangular_roundtrip_and_save_behavior():
    spec = builders.build_triangular_plotspec(
        title="tri",
        proba=[0.1, 0.9],
        uncertainty=[0.2, 0.1],
        rule_proba=[0.2, 0.8],
        rule_uncertainty=[0.1, 0.2],
        num_to_show=2,
        is_probabilistic=True,
    )

    # add save behavior and provenance
    spec.save_behavior = SaveBehavior(path=None, title="tri", default_exts=("png", "svg"))
    spec.data_slice_id = "slice-1"

    d = serializers.triangular_plotspec_to_dict(spec)
    spec2 = serializers.triangular_plotspec_from_dict(d)

    assert spec2.kind == "triangular"
    assert spec2.triangular is not None
    assert spec2.triangular.num_to_show == 2
    assert spec2.save_behavior is not None
    assert spec2.data_slice_id == "slice-1"


def test_global_roundtrip_preserves_entries():
    spec = builders.build_global_plotspec(
        title="glob",
        proba=[0.1, 0.2, 0.8],
        predict=None,
        low=[0.05, 0.1, 0.7],
        high=[0.15, 0.3, 0.9],
        uncertainty=[0.1, 0.2, 0.1],
        y_test=[0, 1, 1],
        is_regularized=True,
    )

    # add save behavior
    spec.save_behavior = SaveBehavior(path=None, title="glob", default_exts=("png",))

    d = serializers.global_plotspec_to_dict(spec)
    spec2 = serializers.global_plotspec_from_dict(d)

    assert spec2.kind.startswith("global")
    assert spec2.global_entries is not None
    assert len(spec2.global_entries.proba) == 3


