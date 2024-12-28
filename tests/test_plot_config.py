"""Tests for the plot configuration module"""
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations._plots import load_plot_config, update_plot_config
# pylint: disable=invalid-name, redefined-outer-name
@pytest.fixture
def styled_explainer():
    """Create a fitted and calibrated explainer for testing plot styles"""
    X_data = np.random.rand(100, 5)
    y_data = (X_data[:,0] + X_data[:,1] > 1).astype(int)

    X_train, X_cal = X_data[:70], X_data[70:90]
    y_train, y_cal = y_data[:70], y_data[70:90]

    explainer = WrapCalibratedExplainer(RandomForestClassifier())
    explainer.fit(X_train, y_train)
    explainer.calibrate(X_cal, y_cal)

    return explainer, X_data[90:], y_data[90:]

def test_default_plot_config():
    """Test the default plot configuration loads correctly"""
    config = load_plot_config()
    assert config['style']['base'] == 'seaborn-v0_8-whitegrid'
    assert config['fonts']['family'] == 'sans-serif'

def test_update_plot_config():
    """Test updating plot configuration"""
    new_config = {
        'style': {'base': 'default'},
        'fonts': {'family': 'serif'}
    }
    update_plot_config(new_config)

    config = load_plot_config()
    assert config['style']['base'] == 'default'
    assert config['fonts']['family'] == 'serif'
    new_config = {
        'style': {'base': 'seaborn-v0_8-whitegrid'},
        'fonts': {'family': 'sans-serif'}
    }
    update_plot_config(new_config)

    config = load_plot_config()
    assert config['style']['base'] == 'seaborn-v0_8-whitegrid'
    assert config['fonts']['family'] == 'sans-serif'

@pytest.mark.parametrize("style_section,style_params", [
    ('style', {'base': 'default'}),
    ('fonts', {
        'family': 'serif',
        'sans_serif': 'Times',
        'axes_label_size': '14',
        'tick_label_size': '12',
        'legend_size': '11',
        'title_size': '16'
    }),
    ('lines', {'width': '3'}),
    ('grid', {'style': ':', 'alpha': '0.7'}),
    ('figure', {
        'dpi': '150',
        'save_dpi': '150',
        'facecolor': 'gray',
        'axes_facecolor': 'lightgray'
    }),
    ('colors', {
        'background': '#ffffff',
        'text': '#000000',
        'grid': '#dddddd',
        'regression': 'red',
        'positive': 'green',
        'negative': 'orange',
        'uncertainty': 'gray',
        'alpha': '0.4'
    })
])
def test_style_override(styled_explainer, style_section, style_params):
    """Test that style overrides work for all configurable parameters"""
    explainer, X_test, y_test = styled_explainer

    # Test global plot
    explainer.plot(X_test, y_test, show=False,
                  style_override={style_section: style_params})

    # Test factual explanation plot
    explanation = explainer.explain_factual(X_test)
    explanation.plot(show=False,
                    style_override={style_section: style_params})

    # Test alternative explanation plot
    explanation = explainer.explore_alternatives(X_test)
    explanation.plot(show=False,
                    style_override={style_section: style_params})

    # # No errors should occur with any style override
    # assert True

def test_invalid_style_override(styled_explainer):
    """Test that invalid style overrides are handled gracefully"""
    explainer, X_test, _ = styled_explainer

    # with pytest.raises(Warning):
    explainer.plot(X_test, show=False,
                    style_override={'invalid_section': {'param': 'value'}})

def test_style_override_persistence(styled_explainer):
    """Test that style overrides don't persist between plots"""
    explainer, X_test, _ = styled_explainer

    # Plot with override
    explainer.plot(X_test, show=False,
                  style_override={'fonts': {'family': 'serif'}})

    # Get default config
    config1 = load_plot_config()

    # Plot without override
    explainer.plot(X_test, show=False)

    # Get config again
    config2 = load_plot_config()

    # Configs should be identical
    assert config1['fonts']['family'] == config2['fonts']['family']
