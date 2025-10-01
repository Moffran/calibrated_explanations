from calibrated_explanations.viz import matplotlib_adapter as mpl_adapter
from tests.unit.viz.test_plot_parity_fixtures import (
    factual_probabilistic_zero_crossing,
    factual_probabilistic_no_uncertainty,
)


def test_adapter_coordinate_spaces_for_factual_probabilistic_zero_crossing():
    spec = factual_probabilistic_zero_crossing()
    # Should not raise assertion in adapter coordinate-space checks
    wrapper = mpl_adapter.render(spec, export_drawn_primitives=True)
    assert isinstance(wrapper, dict)


def test_adapter_coordinate_spaces_for_factual_probabilistic_no_uncertainty():
    spec = factual_probabilistic_no_uncertainty()
    wrapper = mpl_adapter.render(spec, export_drawn_primitives=True)
    assert isinstance(wrapper, dict)
