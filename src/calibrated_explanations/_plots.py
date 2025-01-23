"""
This module contains all plotting functionality for calibrated explanations.

Functions
---------
- _plot_probabilistic(explanation, instance, predict, feature_weights, features_to_plot,
                      num_to_show, column_names, title, path, show, interval=False,
                      idx=None, save_ext=None)
    Plot regular and uncertainty explanations.

- _plot_regression(explanation, instance, predict, feature_weights, features_to_plot, num_to_show,
                   column_names, title, path, show, interval=False, idx=None,
                   save_ext=None)
    Plot regular and uncertainty explanations.

- _plot_triangular(explanation, proba, uncertainty, rule_proba, rule_uncertainty,
                   num_to_show, title, path, show, save_ext=None)
    Plot triangular explanations.

- _plot_alternative(explanation, instance, predict, feature_predict, features_to_plot,
                    num_to_show, column_names, title, path, show, save_ext=None)
    Plot alternative explanations.

- _plot_global(explainer, X_test, y_test=None, threshold=None, **kwargs)
    Generate a global explanation plot for the given test data.

- _plot_proba_triangle()
    Plot a probability triangle. Used internally by several other plotting functions.

- __color_brew(n)
    Generate a list of colors.

- __get_fill_color(venn_abers, reduction=1)
    Get the fill color for the plot.
"""
import os
import contextlib
import math
import warnings
import configparser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# pylint: disable=unknown-option-value
# pylint: disable=too-many-arguments, too-many-statements, too-many-branches, too-many-locals, too-many-positional-arguments, fixme

def load_plot_config():
    """Load plot configuration from INI file."""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__),
                               '../calibrated-explanations/utils/configurations/plot_config.ini')

    # Set default values
    config['style'] = {'base': 'seaborn-v0_8-whitegrid'}
    config['fonts'] = {
        'family': 'sans-serif',
        'sans_serif': 'Arial',
        'axes_label_size': '12',
        'tick_label_size': '10',
        'legend_size': '10',
        'title_size': '14'
    }
    config['lines'] = {'width': '2'}
    config['grid'] = {'style': '--', 'alpha': '0.5'}
    config['figure'] = {
        'dpi': '300',
        'save_dpi': '300',
        'facecolor': 'white',
        'axes_facecolor': 'white',
        'width': '10'  # Add default width value
    }
    config['colors'] = {
        'background': '#f8f9fa',
        'text': '#666666',
        'grid': '#cccccc',
        'regression': 'red',
        'positive': 'red',
        'negative': 'blue',
        'uncertainty': 'lightgrey',
        'alpha': '0.2'
    }

    # Read config file if it exists
    config.read(config_path)
    return config

def update_plot_config(new_config):
    """Update plot configuration file with new values."""
    config = load_plot_config()

    # Update configuration with new values
    for section, values in new_config.items():
        if section not in config:
            config[section] = {}
        for key, value in values.items():
            config[section][key] = str(value)

    # Write updated config to file
    config_path = os.path.join(os.path.dirname(__file__),
                               '../calibrated-explanations/utils/configurations/plot_config.ini')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)  # Ensure directory exists
    with open(config_path, 'w', encoding='utf-8') as f:
        config.write(f)

def __setup_plot_style(style_override=None):
    """Set up plot style using configuration with optional runtime overrides."""
    config = load_plot_config()

    # Apply style overrides if provided
    if style_override:
        for section, values in style_override.items():
            if section not in config:
                config[section] = {}
                warnings.warn(f'Unknown style section "{section}" in style override.', Warning)
            for key, value in values.items():
                config[section][key] = str(value)

    plt.style.use(config['style']['base'])

    # Font settings
    plt.rcParams['font.family'] = config['fonts']['family']
    plt.rcParams['font.sans-serif'] = [config['fonts']['sans_serif']]
    plt.rcParams['axes.labelsize'] = float(config['fonts']['axes_label_size'])
    plt.rcParams['xtick.labelsize'] = float(config['fonts']['tick_label_size'])
    plt.rcParams['ytick.labelsize'] = float(config['fonts']['tick_label_size'])
    plt.rcParams['legend.fontsize'] = float(config['fonts']['legend_size'])
    plt.rcParams['axes.titlesize'] = float(config['fonts']['title_size'])

    # Line settings
    plt.rcParams['lines.linewidth'] = float(config['lines']['width'])

    # Grid settings
    plt.rcParams['grid.linestyle'] = config['grid']['style']
    plt.rcParams['grid.alpha'] = float(config['grid']['alpha'])

    # Figure settings
    plt.rcParams['figure.dpi'] = float(config['figure']['dpi'])
    plt.rcParams['savefig.dpi'] = float(config['figure']['save_dpi'])
    plt.rcParams['figure.facecolor'] = config['figure']['facecolor']
    plt.rcParams['axes.facecolor'] = config['figure']['axes_facecolor']

    return config

def _plot_probabilistic(explanation, instance, predict, feature_weights, features_to_plot,
                        num_to_show, column_names, title, path, show, interval=False,
                        idx=None, save_ext=None, style_override=None):
    """
    Plot regular and uncertainty explanations.

    Parameters
    ----------
    explanation : object
        The explanation object containing the details of the explanation.
    instance : array-like
        The instance for which the explanation is generated.
    predict : dict
        The prediction details including 'predict', 'low', and 'high'.
    feature_weights : dict or array-like
        The weights of the features.
    features_to_plot : list
        The list of features to plot.
    num_to_show : int
        The number of features to show in the plot.
    column_names : list
        The names of the columns.
    title : str
        The title of the plot.
    path : str
        The path to save the plot.
    show : bool
        Whether to show the plot.
    interval : bool, optional
        Whether to plot intervals.
    idx : int, optional
        The index for interval plotting.
    save_ext : list, optional
        The list of file extensions to save the plot.
    """
    config = __setup_plot_style(style_override)

    if save_ext is None:
        save_ext=['svg','pdf','png']
    if interval is True:
        assert idx is not None
    # Get figure width from config, with fallback to default value
    fig_width = float(config['figure'].get('width', 10))
    fig = plt.figure(figsize=(fig_width, num_to_show*.5+2))
    subfigures = fig.subfigures(3, 1, height_ratios=[1, 1, num_to_show+2])

    if interval and (explanation.is_one_sided()):
        raise Warning('Interval plot is not supported for one-sided explanations.')

    ax_positive = subfigures[0].add_subplot(111)
    ax_negative = subfigures[1].add_subplot(111)

    ax_main = subfigures[2].add_subplot(111)

    # plot the probabilities at the top
    x = np.linspace(0, 1, 2)
    xj = np.linspace(x[0]-0.2, x[0]+0.2,2)
    pred = predict['predict']
    pred_low = predict['low'] if predict['low'] != -np.inf else explanation.y_minmax[0]
    pred_high = predict['high'] if predict['high'] != np.inf else explanation.y_minmax[1]

    alpha_val = float(config['colors']['alpha'])
    pos_color = config['colors']['positive']
    neg_color = config['colors']['negative']

    ax_negative.fill_betweenx(xj, 1-pred, 1-pred, color=pos_color)
    ax_negative.fill_betweenx(xj, 0, 1-pred_high, color=pos_color)
    ax_negative.fill_betweenx(xj, 1-pred_low, 1-pred_high, color=pos_color, alpha=alpha_val)
    ax_negative.set_xlim([0,1])
    ax_negative.set_yticks(range(1))
    ax_negative.set_xticks(np.linspace(0,1,6))
    ax_positive.fill_betweenx(xj, pred, pred, color=neg_color)
    ax_positive.fill_betweenx(xj, 0, pred_low, color=neg_color)
    ax_positive.fill_betweenx(xj, pred_low, pred_high, color=neg_color, alpha=alpha_val)
    ax_positive.set_xlim([0,1])
    ax_positive.set_yticks(range(1))
    ax_positive.set_xticks([])

    if explanation.is_thresholded():
        if np.isscalar(explanation.y_threshold):
            ax_negative.set_yticklabels(labels=[f'P(y>{float(explanation.y_threshold) :.2f})'])
            ax_positive.set_yticklabels(labels=[f'P(y<={float(explanation.y_threshold) :.2f})'])
        else: # interval threshold
            ax_negative.set_yticklabels(labels=[f'y_hat <= {explanation.y_threshold[0]:.3f} || y_hat > {explanation.y_threshold[1]:.3f}']) # pylint: disable=line-too-long
            ax_positive.set_yticklabels(labels=[f'{explanation.y_threshold[0]:.3f} < y_hat <= {explanation.y_threshold[1]:.3f}'])# pylint: disable=line-too-long
    elif explanation.get_class_labels() is None:
        if explanation._get_explainer().is_multiclass(): # pylint: disable=protected-access
            ax_negative.set_yticklabels(labels=[f'P(y!={explanation.prediction["classes"]})'])
            ax_positive.set_yticklabels(labels=[f'P(y={explanation.prediction["classes"]})'])
        else:
            ax_negative.set_yticklabels(labels=['P(y=0)'])
            ax_positive.set_yticklabels(labels=['P(y=1)'])
    elif explanation.is_multiclass: # pylint: disable=protected-access
        ax_negative.set_yticklabels(labels=[f'P(y!={explanation.get_class_labels()[int(explanation.prediction["classes"])]})']) # pylint: disable=line-too-long
        ax_positive.set_yticklabels(labels=[f'P(y={explanation.get_class_labels()[int(explanation.prediction["classes"])]})']) # pylint: disable=line-too-long
    else:
        ax_negative.set_yticklabels(labels=[f'P(y={explanation.get_class_labels()[0]})']) # pylint: disable=line-too-long
        ax_positive.set_yticklabels(labels=[f'P(y={explanation.get_class_labels()[1]})']) # pylint: disable=line-too-long
    ax_negative.set_xlabel('Probability')

    # Plot the base prediction in black/grey
    if num_to_show > 0:
        x = np.linspace(0, num_to_show-1, num_to_show)
        xl = np.linspace(-0.5, x[0] if len(x) > 0 else 0, 2)
        xh = np.linspace(x[-1], x[-1]+0.5 if len(x) > 0 else 0.5, 2)
        ax_main.fill_betweenx(x, [0], [0], color='k')
        ax_main.fill_betweenx(xl, [0], [0], color='k')
        ax_main.fill_betweenx(xh, [0], [0], color='k')
        if interval:
            gwl = predict['low'] - pred
            gwh = predict['high'] - pred

            gwh, gwl = np.max([gwh, gwl]), np.min([gwl, gwh])
            ax_main.fill_betweenx([-0.5,num_to_show-0.5], gwl, gwh, color='k', alpha=alpha_val)

        # For each feature, plot the weight
        for jx, j in enumerate(features_to_plot):
            xj = np.linspace(x[jx]-0.2, x[jx]+0.2,2)
            min_val,max_val = 0,0
            if interval:
                width = feature_weights['predict'][j]
                wl = feature_weights['low'][j]
                wh = feature_weights['high'][j]
                wh, wl = np.max([wh, wl]), np.min([wh, wl])
                max_val = wh if width < 0 else 0
                min_val = wl if width > 0 else 0
                # If uncertainty cover zero, then set to 0 to avoid solid plotting
                if wl < 0 < wh:
                    min_val = 0
                    max_val = 0
            else:
                width = feature_weights[j]
                min_val = min(width, 0)
                max_val = max(width, 0)
            color = neg_color if width > 0 else pos_color
            ax_main.fill_betweenx(xj, min_val, max_val, color=color)
            if interval:
                if wl < 0 < wh and explanation.get_mode() == 'classification':
                    ax_main.fill_betweenx(xj, 0, wl, color=pos_color, alpha=alpha_val)
                    ax_main.fill_betweenx(xj, wh, 0, color=neg_color, alpha=alpha_val)
                else:
                    ax_main.fill_betweenx(xj, wl, wh, color=color, alpha=alpha_val)

        ax_main.set_yticks(range(num_to_show))
        ax_main.set_yticklabels(labels=[column_names[i] for i in features_to_plot]) \
                if column_names is not None else ax_main.set_yticks(range(num_to_show)) # pylint: disable=expression-not-assigned
        ax_main.set_ylim(-0.5,x[-1]+0.5 if len(x) > 0 else 0.5)
        ax_main.set_ylabel('Features')
        ax_main.set_xlabel('Feature weights')
        ax_main_twin = ax_main.twinx()
        ax_main_twin.set_yticks(range(num_to_show))
        ax_main_twin.set_yticklabels([instance[i] for i in features_to_plot])
        ax_main_twin.set_ylim(-0.5,x[-1]+0.5 if len(x) > 0 else 0.5)
        ax_main_twin.set_ylabel('Instance values')

    # Enhance subplot appearance
    for ax in [ax_positive, ax_negative, ax_main]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add background colors for better readability
    ax_main.patch.set_facecolor(config['colors']['background'])

    # Enhance legend appearance
    ax_main_twin.tick_params(axis='y', colors=config['colors']['text'])

    for ext in save_ext:
        fig.savefig(path + title + ext, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


# pylint: disable=too-many-branches, too-many-statements, too-many-locals
def _plot_regression(explanation, instance, predict, feature_weights, features_to_plot, num_to_show,
                column_names, title, path, show, interval=False, idx=None,
                save_ext=None, style_override=None):
    """
    Plot regular and uncertainty explanations.

    Parameters
    ----------
    explanation : object
        The explanation object containing the details of the explanation.
    instance : array-like
        The instance for which the explanation is generated.
    predict : dict
        The prediction details including 'predict', 'low', and 'high'.
    feature_weights : dict or array-like
        The weights of the features.
    features_to_plot : list
        The list of features to plot.
    num_to_show : int
        The number of features to show in the plot.
    column_names : list
        The names of the columns.
    title : str
        The title of the plot.
    path : str
        The path to save the plot.
    show : bool
        Whether to show the plot.
    interval : bool, optional
        Whether to plot intervals.
    idx : int, optional
        The index for interval plotting.
    save_ext : list, optional
        The list of file extensions to save the plot.
    """
    config = __setup_plot_style(style_override)

    if save_ext is None:
        save_ext=['svg','pdf','png']
    if interval is True:
        assert idx is not None
    # Get figure width from config, with fallback to default value
    fig_width = float(config['figure'].get('width', 10))
    fig = plt.figure(figsize=(fig_width, num_to_show*.5+2))
    subfigures = fig.subfigures(2, 1, height_ratios=[1, num_to_show+2])

    if interval and (explanation.is_one_sided()):
        raise Warning('Interval plot is not supported for one-sided explanations.')

    ax_regression = subfigures[0].add_subplot(111)
    ax_main = subfigures[1].add_subplot(111)

    # plot the probabilities at the top
    x = np.linspace(0, 1, 2)
    xj = np.linspace(x[0]-0.2, x[0]+0.2,2)
    pred = predict['predict']
    pred_low = predict['low'] if predict['low'] != -np.inf \
                                    else explanation.y_minmax[0]
    pred_high = predict['high'] if predict['high'] != np.inf \
                                    else explanation.y_minmax[1]

    alpha_val = float(config['colors']['alpha'])
    reg_color = config['colors']['regression']

    ax_regression.fill_betweenx(xj, pred_low, pred_high, color=reg_color, alpha=alpha_val)
    ax_regression.fill_betweenx(xj, pred, pred, color=reg_color)
    ax_regression.set_xlim([np.min([pred_low, explanation.y_minmax[0]]),np.max([pred_high, explanation.y_minmax[1]])]) # pylint: disable=line-too-long
    ax_regression.set_yticks(range(1))

    ax_regression.set_xlabel(f'Prediction interval with {explanation.calibrated_explanations.get_confidence()}% confidence') # pylint: disable=line-too-long
    ax_regression.set_yticklabels(labels=['Median prediction'])

    # Plot the base prediction in black/grey
    x = np.linspace(0, num_to_show-1, num_to_show)
    xl = np.linspace(-0.5, x[0], 2)
    xh = np.linspace(x[-1], x[-1]+0.5, 2)
    ax_main.fill_betweenx(x, [0], [0], color='k')
    ax_main.fill_betweenx(xl, [0], [0], color='k')
    ax_main.fill_betweenx(xh, [0], [0], color='k')
    x_min, x_max = 0,0
    if interval:
        gwl = pred - predict['low']
        gwh = pred - predict['high']

        gwh, gwl = np.max([gwh, gwl]), np.min([gwl, gwh])
        # ax_main.fill_betweenx([-0.5,num_to_show-0.5], gwl, gwh, color='k', alpha=0.2)

        x_min, x_max = gwl,gwh
    # For each feature, plot the weight
    for jx, j in enumerate(features_to_plot):
        xj = np.linspace(x[jx]-0.2, x[jx]+0.2,2)
        min_val,max_val = 0,0
        if interval:
            width = feature_weights['predict'][j]
            wl = feature_weights['low'][j]
            wh = feature_weights['high'][j]
            wh, wl = np.max([wh, wl]), np.min([wh, wl])
            max_val = wh if width < 0 else 0
            min_val = wl if width > 0 else 0
            # If uncertainty cover zero, then set to 0 to avoid solid plotting
            if wl < 0 < wh:
                min_val = 0
                max_val = 0
        else:
            width = feature_weights[j]
            min_val = min(width, 0)
            max_val = max(width, 0)
        color = config['colors']['positive'] if width > 0 else config['colors']['negative']
        ax_main.fill_betweenx(xj, min_val, max_val, color=color)
        if interval:
            ax_main.fill_betweenx(xj, wl, wh, color=color, alpha=alpha_val)

            x_min = np.min([x_min, min_val, max_val, wl, wh])
            x_max = np.max([x_max, min_val, max_val, wl, wh])
        else:
            x_min = np.min([x_min, min_val, max_val])
            x_max = np.max([x_max, min_val, max_val])

    ax_main.set_yticks(range(num_to_show))
    ax_main.set_yticklabels(labels=[column_names[i] for i in features_to_plot]) \
                if column_names is not None else ax_main.set_yticks(range(num_to_show)) # pylint: disable=expression-not-assigned
    ax_main.set_ylim(-0.5,x[-1]+0.5 if len(x) > 0 else 0.5)
    ax_main.set_ylabel('Rules', fontweight='bold')
    ax_main.set_xlabel('Feature weights', fontweight='bold')
    ax_main.set_xlim(x_min, x_max)
    ax_main_twin = ax_main.twinx()
    ax_main_twin.set_yticks(range(num_to_show))
    ax_main_twin.set_yticklabels([instance[i] for i in features_to_plot])
    ax_main_twin.set_ylim(-0.5,x[-1]+0.5 if len(x) > 0 else 0.5)
    ax_main_twin.set_ylabel('Instance values')

    # Enhance subplot appearance
    for ax in [ax_regression, ax_main]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add subtle background color
    ax_main.patch.set_facecolor(config['colors']['background'])

    for ext in save_ext:
        fig.savefig(path + title + ext, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


# pylint: disable=duplicate-code
def _plot_triangular(explanation, proba, uncertainty, rule_proba, rule_uncertainty,
                     num_to_show, title, path, show, save_ext=None, style_override=None):
    """
    Plot triangular explanations.

    Parameters
    ----------
    explanation : object
        The explanation object containing the details of the explanation.
    proba : array-like
        The probabilities of the predictions.
    uncertainty : array-like
        The uncertainties of the predictions.
    rule_proba : array-like
        The probabilities of the rules.
    rule_uncertainty : array-like
        The uncertainties of the rules.
    num_to_show : int
        The number of rules to show in the plot.
    title : str
        The title of the plot.
    path : str
        The path to save the plot.
    show : bool
        Whether to show the plot.
    save_ext : list, optional
        The list of file extensions to save the plot.
    """
    config = __setup_plot_style(style_override)

    if save_ext is None:
        save_ext=['svg','pdf','png']
    # assert self._get_explainer().mode == 'classification' or \
    #     (self._get_explainer().mode == 'regression' and self._is_thresholded()), \
    #     'Triangular plot is only available for classification or thresholded regression'
    marker_size = 50
    min_x, min_y = 0,0
    max_x, max_y = 1,1
    is_probabilistic = True
    plt.figure()
    if explanation.get_mode() == 'classification' or \
            (explanation.get_mode() == 'regression' and explanation.is_thresholded()):
        __plot_proba_triangle()
    else:
        min_x = min(np.min(rule_proba), np.min(proba))#np.min(self._get_explainer().y_cal) # pylint: disable=protected-access
        max_x = max(np.max(rule_proba), np.max(proba))#np.max(self._get_explainer().y_cal) # pylint: disable=protected-access
        min_y = min(np.min(rule_uncertainty), np.min(uncertainty))
        max_y = max(np.max(rule_uncertainty), np.max(uncertainty))
        if math.isclose(min_x, max_x, rel_tol=1e-9):
            warnings.warn("All uncertainties are (almost) identical.", Warning)
        min_y = min_y - 0.1 * (max_y - min_y)
        max_y = max_y + 0.1 * (max_y - min_y)
        min_x = min_x - 0.1 * (max_x - min_x)
        max_x = max_x + 0.1 * (max_x - min_x)
        is_probabilistic = False

    plt.quiver([proba]*num_to_show, [uncertainty]*num_to_show,
                rule_proba[:num_to_show] - proba,
                rule_uncertainty[:num_to_show] - uncertainty,
                angles='xy', scale_units='xy', scale=1, color='lightgrey',
                width=0.005, headwidth=3, headlength=3)
    plt.scatter(rule_proba, rule_uncertainty, label='Alternative Explanations',
                marker='.', s=marker_size, alpha=0.7)
    plt.scatter(proba, uncertainty, color='red', label='Original Prediction',
                marker='*', s=marker_size*1.5)
    if is_probabilistic:
        plt.xlabel('Probability')
    else:
        plt.xlabel('Prediction')
    plt.ylabel('Uncertainty')
    plt.title('Alternative Explanations')
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # Add legend
    plt.legend(frameon=True, fancybox=True, framealpha=0.95,
              edgecolor=config['colors']['grid'], loc='best')

    for ext in save_ext:
        plt.savefig(path + title + ext, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

# `__plot_triangular`
def __plot_proba_triangle():
    x = np.arange(0, 1, 0.01)
    plt.plot((x / (1 + x)), x, color='black')
    plt.plot(x, ((1 - x) / x), color='black')
    x = np.arange(0.5, 1, 0.005)
    plt.plot((0.5 + x - 0.5)/(1 + x - 0.5), x - 0.5, color='black')
    x = np.arange(0, 0.5, 0.005)
    plt.plot((x + 0.5 - x)/(1 + x), x, color='black')


# pylint: disable=too-many-arguments, too-many-locals, invalid-name, too-many-branches, too-many-statements
def _plot_alternative(explanation, instance, predict, feature_predict, features_to_plot,
                            num_to_show, column_names, title, path, show, save_ext=None,
                            style_override=None):
    """
    Plot alternative explanations.

    Parameters
    ----------
    explanation : object
        The explanation object containing the details of the explanation.
    instance : array-like
        The instance for which the explanation is generated.
    predict : dict
        The prediction details including 'predict', 'low', and 'high'.
    feature_predict : dict
        The predictions for the features.
    features_to_plot : list
        The list of features to plot.
    num_to_show : int
        The number of features to show in the plot.
    column_names : list
        The names of the columns.
    title : str
        The title of the plot.
    path : str
        The path to save the plot.
    show : bool
        Whether to show the plot.
    save_ext : list, optional
        The list of file extensions to save the plot.
    """
    config = __setup_plot_style(style_override)

    if save_ext is None:
        save_ext=['svg','pdf','png']
    # Get figure width from config, with fallback to default value
    fig_width = float(config['figure'].get('width', 10))
    fig = plt.figure(figsize=(fig_width, num_to_show*.5))
    ax_main = fig.add_subplot(111)
    x = np.linspace(0, num_to_show-1, num_to_show)
    pred = predict['predict']
    pred_low = predict['low'] if predict['low'] != -np.inf \
                            else explanation.y_minmax[0]
    pred_high = predict['high'] if predict['high'] != np.inf \
                            else explanation.y_minmax[1]
    venn_abers={'low_high': [pred_low,pred_high],'predict':pred}
    alpha_val = float(config['colors']['alpha'])
    pos_color = config['colors']['positive']
    # Fill original Venn Abers interval
    xl = np.linspace(-0.5, x[0], 2) if len(x) > 0 else np.linspace(-0.5, 0, 2)
    xh = np.linspace(x[-1], x[-1]+0.5, 2) if len(x) > 0 else np.linspace(0, 0.5, 2)
    if (pred_low < 0.5 and pred_high < 0.5) or (pred_low > 0.5 and pred_high > 0.5) or \
                            'regression' in explanation.get_mode():
        color = __get_fill_color({'predict':1},0.15) \
                            if 'regression' in explanation.get_mode() \
                            else __get_fill_color(venn_abers,0.15)
        ax_main.fill_betweenx(x, [pred_low]*(num_to_show), [pred_high]*(num_to_show),color=color)
        # Fill up to the edges
        ax_main.fill_betweenx(xl, [pred_low]*(2), [pred_high]*(2),color=color)
        ax_main.fill_betweenx(xh, [pred_low]*(2), [pred_high]*(2),color=color)
        if 'regression' in explanation.get_mode():
            ax_main.fill_betweenx(x, pred, pred, color=pos_color, alpha=alpha_val)
            # Fill up to the edges
            ax_main.fill_betweenx(xl, pred, pred, color=pos_color, alpha=alpha_val)
            ax_main.fill_betweenx(xh, pred, pred, color=pos_color, alpha=alpha_val)
    else:
        venn_abers['predict'] = pred_low
        color = __get_fill_color(venn_abers, 0.15)
        ax_main.fill_betweenx(x, [pred_low]*(num_to_show), [0.5]*(num_to_show),color=color)
        # Fill up to the edges
        ax_main.fill_betweenx(xl, [pred_low]*(2), [0.5]*(2),color=color)
        ax_main.fill_betweenx(xh, [pred_low]*(2), [0.5]*(2),color=color)
        venn_abers['predict'] = pred_high
        color = __get_fill_color(venn_abers, 0.15)
        ax_main.fill_betweenx(x, [0.5]*(num_to_show), [pred_high]*(num_to_show),color=color)
        # Fill up to the edges
        ax_main.fill_betweenx(xl, [0.5]*(2), [pred_high]*(2),color=color)
        ax_main.fill_betweenx(xh, [0.5]*(2), [pred_high]*(2),color=color)

    for jx, j in enumerate(features_to_plot):
        pred_low = feature_predict['low'][j] if feature_predict['low'][j] != -np.inf \
                                            else explanation.y_minmax[0]
        pred_high = feature_predict['high'][j] if feature_predict['high'][j] != np.inf \
                                            else explanation.y_minmax[1]
        pred = feature_predict['predict'][j]
        xj = np.linspace(x[jx]-0.2, x[jx]+0.2,2)
        venn_abers={'low_high': [pred_low,pred_high],'predict':pred}
        # Fill each feature impact
        if 'regression' in explanation.get_mode():
            ax_main.fill_betweenx(xj, pred_low,pred_high, color=pos_color, alpha= alpha_val)
            ax_main.fill_betweenx(xj, pred, pred, color=pos_color)
        elif (pred_low < 0.5 and pred_high < 0.5) or (pred_low > 0.5 and pred_high > 0.5) :
            ax_main.fill_betweenx(xj, pred_low,pred_high,color=__get_fill_color(venn_abers, 0.99))
        else:
            venn_abers['predict'] = pred_low
            ax_main.fill_betweenx(xj, pred_low,0.5,color=__get_fill_color(venn_abers, 0.99))
            venn_abers['predict'] = pred_high
            ax_main.fill_betweenx(xj, 0.5,pred_high,color=__get_fill_color(venn_abers, 0.99))

    ax_main.set_yticks(range(num_to_show))
    ax_main.set_yticklabels(labels=[column_names[i] for i in features_to_plot]) \
            if column_names is not None else ax_main.set_yticks(range(num_to_show)) # pylint: disable=expression-not-assigned
    ax_main.set_ylim(-0.5,x[-1]+0.5 if len(x) > 0 else 0.5)
    ax_main.set_ylabel('Alternative rules')
    ax_main_twin = ax_main.twinx()
    ax_main_twin.set_yticks(range(num_to_show))
    ax_main_twin.set_yticklabels([instance[i] for i in features_to_plot])
    ax_main_twin.set_ylim(-0.5,x[-1]+0.5 if len(x) > 0 else 0.5)
    ax_main_twin.set_ylabel('Instance values')
    if explanation.is_thresholded():
        # pylint: disable=unsubscriptable-object
        if np.isscalar(explanation.y_threshold):
            ax_main.set_xlabel(
                f'Probability of target being below {float(explanation.y_threshold):.2f}'
            )
        elif isinstance(explanation.y_threshold, tuple):
            ax_main.set_xlabel(
                f'Probability of target being between {float(explanation.y_threshold[0]):.3f} and '
                + f'{float(explanation.y_threshold[1]):.3f}'
            )
        else:
            ax_main.set_xlabel(
                f'Probability of target being below {float(explanation.y_threshold):.2f}'
            )
        ax_main.set_xlim(0,1)
        ax_main.set_xticks(np.linspace(0, 1, 11))
    elif 'regression' in explanation.get_mode():
        # pylint: disable=line-too-long
        ax_main.set_xlabel(f'Prediction interval with {explanation.calibrated_explanations.get_confidence()}% confidence')
        ax_main.set_xlim([explanation.y_minmax[0], explanation.y_minmax[1]])
    else:
        if explanation.get_class_labels() is None:
            if explanation._get_explainer().is_multiclass(): # pylint: disable=protected-access
                ax_main.set_xlabel(f'Probability for class \'{explanation.prediction["classes"]}\'')
            else:
                ax_main.set_xlabel('Probability for the positive class')
        elif explanation._get_explainer().is_multiclass(): # pylint: disable=protected-access
            # pylint: disable=line-too-long
            ax_main.set_xlabel(
                f'Probability for class \'{explanation.get_class_labels()[explanation.prediction["classes"]]}\''
            )
        else:
            ax_main.set_xlabel(
                f"Probability for class \'{explanation.get_class_labels()[1]}\'"
            )
        ax_main.set_xlim(0,1)
        ax_main.set_xticks(np.linspace(0, 1, 11))

    with contextlib.suppress(Exception):
        fig.tight_layout()
    for ext in save_ext:
        fig.savefig(path + title + ext, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


# pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
def _plot_global(explainer, X_test, y_test=None, threshold=None, style_override=None,
                 show=True, bins=None):
    """
    Generate a global explanation plot for the given test data.

    This plot is based on the 
    probability distribution and the uncertainty quantification intervals.
    The plot is only available for calibrated probabilistic learners (both classification and 
    thresholded regression).

    Parameters
    ----------
    explainer : object
        The explainer object used to generate the explanations.
    X_test : array-like
        The test data for which predictions are to be made.
    y_test : array-like, optional
        The true labels of the test data.
    threshold : float, int, optional
        The threshold value used with regression to get probability of being below the threshold.
    **kwargs : dict
        Additional keyword arguments.
    """
    config = __setup_plot_style(style_override)

    is_regularized = True
    if 'predict_proba' not in dir(explainer.learner) and threshold is None:
        predict, (low, high) = explainer.predict(X_test, uq_interval=True, bins=bins)
        is_regularized = False
    else:
        proba, (low, high) = explainer.predict_proba(X_test,
                                                     uq_interval=True,
                                                     threshold=threshold,
                                                     bins=bins)
    uncertainty = np.array(high - low)

    marker_size = 50
    min_x, min_y = 0,0
    max_x, max_y = 1,1
    ax = None
    if is_regularized:
        fig = _plot_proba_triangle()
    else:
        fig, ax = plt.subplots()
        # draw a line from (0,0) to (0.5,1) and from (1,0) to (0.5,1)
        min_x = np.min(explainer.y_cal)
        max_x = np.max(explainer.y_cal)
        min_y = np.min(uncertainty)
        max_y = np.max(uncertainty)
        if math.isclose(min_x, max_x, rel_tol=1e-9):
            warnings.warn("All uncertainties are (almost) identical.", Warning)

        min_x = min_x - min(0.1 * (max_x - min_x), 0) if min_x > 0 else min_x - 0.1 * (max_x - min_x) # pylint: disable=line-too-long
        max_x = max_x + 0.1 * (max_x - min_x)
        # min_y = min_y - max(0.1 * (max_y - min_y), 0) # uncertainty is always positive
        max_y = max_y + 0.1 * (max_y - min_y)
        # mid_x = (min_x + max_x) / 2
        # mid_y = (min_y + max_y) / 2
        # ax.plot([min_x, mid_x], [min_y, max_y], color='black')
        # ax.plot([max_x, mid_x], [min_y, max_y], color='black')
        # # draw a line from (0.5,0) to halfway between (0.5,0) and (0,1)
        # ax.plot([mid_x, mid_x / 2], [min_y, mid_y], color='black')
        # # draw a line from (0.5,0) to halfway between (0.5,0) and (1,1)
        # ax.plot([mid_x, mid_x + mid_x / 2], [min_y, mid_y], color='black')

    if y_test is None:
        if 'predict_proba' not in dir(explainer.learner) and threshold is None: # not probabilistic
            plt.scatter(predict, uncertainty, label='Predictions', marker='.', s=marker_size)
        else:
            if explainer.is_multiclass(): # pylint: disable=protected-access
                predicted = np.argmax(proba, axis=1)
                proba = proba[np.arange(len(proba)), predicted]
                uncertainty = uncertainty[np.arange(len(uncertainty)), predicted]
            else:
                proba = proba[:,1]
            plt.scatter(proba, uncertainty, label='Predictions', marker='.', s=marker_size)

    elif 'predict_proba' not in dir(explainer.learner) and threshold is None: # not probabilistic
        norm = mcolors.Normalize(vmin=y_test.min(), vmax=y_test.max())
        # Choose a colormap
        colormap = plt.cm.viridis  # pylint: disable=no-member
        # Map the normalized values to colors
        colors = colormap(norm(y_test))
        ax.scatter(predict,
                   uncertainty, label='Predictions', color=colors, marker='.', s=marker_size)
        # # Create a new axes for the colorbar
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # # Add the colorbar to the new axes
        # plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap),
        #                                       cax=cax, label='Target Values')
    else:
        if 'predict_proba' not in dir(explainer.learner):
            assert np.isscalar(threshold), "The threshold parameter must be a single constant value for all instances when used in plot_global." # pylint: disable=line-too-long
            y_test = np.array([0 if y_test[i] >= threshold else 1 for i in range(len(y_test))])
            labels = [f'Y >= {threshold}', f'Y < {threshold}']
        else:
            labels = (
                [f'Y = {i}' for i in explainer.class_labels.values()]
                if explainer.class_labels is not None
                else [f'Y = {i}' for i in np.unique(y_test)]
            )
        marker_size = 25
        if len(labels) == 2:
            colors = ['blue', 'red']
            markers = ['o', 'x']
            proba = proba[:,1]
        else:
            colormap = plt.get_cmap('tab10', len(labels))
            colors = [colormap(i) for i in range(len(labels))]
            markers = ['o', 'x', 's', '^', 'v', 'D', 'P', '*', 'h', 'H','o', 'x', 's', '^', 'v', 'D', 'P', '*', 'h', 'H'][:len(labels)]# pylint: disable=line-too-long
            proba = proba[np.arange(len(proba)), y_test]
            uncertainty = uncertainty[np.arange(len(uncertainty)), y_test]
        for i, c in enumerate(np.unique(y_test)):
            plt.scatter(proba[y_test == c],
                        uncertainty[y_test == c],
                        color=colors[i],
                        label=labels[i],
                        marker=markers[i],
                        s=marker_size,
                        alpha=0.7)
        plt.legend(frameon=True, fancybox=True, framealpha=0.95,
              edgecolor=config['colors']['grid'], bbox_to_anchor=(1.05, 1), loc='upper left')
    if 'predict_proba' not in dir(explainer.learner) and threshold is None: # not probabilistic
        plt.xlabel('Predictions',loc='center')
        plt.ylabel('Uncertainty',loc='center')
    else:
        plt.ylabel('Uncertainty')
        if 'predict_proba' not in dir(explainer.learner):
            if np.isscalar(threshold):
                plt.xlabel(f'Probability of Y < {threshold}')
            else:
                plt.xlabel(f'Probability of {threshold[0]} <= Y < {threshold[1]}')
        elif explainer.is_multiclass(): # pylint: disable=protected-access
            if y_test is None:
                plt.xlabel('Probability of Y = predicted class')
            else:
                plt.xlabel('Probability of Y = actual class')
        else:
            plt.xlabel('Probability of Y = 1')
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y) # TODO: UserWarning: Attempting to set identical low and high ylims makes transformation singular; automatically expanding.
    plt.grid(True, linestyle=config['grid']['style'], alpha=float(config['grid']['alpha']))
    if show:
        plt.show()
    else:
        plt.close(fig)
    # TODO: Add save functionality

def _plot_proba_triangle():
    fig = plt.figure()
    x = np.arange(0, 1, 0.01)
    plt.plot((x / (1 + x)), x, color='black')
    plt.plot(x, ((1 - x) / x), color='black')
    x = np.arange(0.5, 1, 0.005)
    plt.plot((0.5 + x - 0.5)/(1 + x - 0.5), x - 0.5, color='black')
    x = np.arange(0, 0.5, 0.005)
    plt.plot((x + 0.5 - x)/(1 + x), x, color='black')
    return fig


# pylint: disable=invalid-name
def __color_brew(n):
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    # for h in np.arange(25, 385, 360. / n).astype(int):
    for h in np.arange(5, 385, 490. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
            (int(255 * (g + m))),
            (int(255 * (b + m)))]
        color_list.append(rgb)
    color_list.reverse()
    return color_list



def __get_fill_color(venn_abers, reduction=1): # pylint: disable=unused-private-member
    colors = __color_brew(2)
    winner_class = int(venn_abers['predict']>= 0.5)
    color = colors[winner_class]

    alpha = venn_abers['predict'] if winner_class==1 else 1-venn_abers['predict']
    alpha = ((alpha - 0.5) / (1 - 0.5)) * (1-.25) + .25 # normalize values to the range [.25,1]
    if reduction != 1:
        alpha = reduction

    # unpack numpy scalars
    alpha = float(alpha)
    # compute the color as alpha against white
    color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
    # Return html color code in #RRGGBB format
    return '#%2x%2x%2x' % tuple(color) # pylint: disable=consider-using-f-string
