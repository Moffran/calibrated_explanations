# pylint: disable=line-too-long
'''
This module provides utility functions for various tasks such as directory creation, safe importing,
type checking, data transformation, and metric calculation. It includes functions to handle 
categorical data, check if an estimator is fitted, and determine if the code is running in a 
Jupyter notebook.
Functions:
- make_directory: Create a directory if it does not exist.
- safe_isinstance: Safely check if an object is an instance of a specified class.
- safe_import: Safely import a module or class, with error handling for missing modules or classes.
- check_is_fitted: Check if an estimator is fitted by verifying the presence of fitted attributes.
- is_notebook: Check if the code is running in a Jupyter notebook.
- transform_to_numeric: Transform categorical features in a DataFrame to numeric values.
- assert_threshold: Validate and return thresholds.
- calculate_metrics: Calculate various metrics based on uncertainty and prediction values.
- convert_targets_to_numeric: Convert string/categorical targets to numeric values while preserving labels.
Created on 2023-07-01
Author: Tuwe Löfström
'''
import os
import sys
import importlib
from inspect import isclass
import numbers
import numpy as np
from pandas import CategoricalDtype

def make_directory(path: str, save_ext=None) -> None:    # pylint: disable=unused-private-member
    """ create directory if it does not exist
    """
    if save_ext is not None and len(save_ext) == 0:
        return
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    if not os.path.isdir(f'plots/{path}'):
        os.mkdir(f'plots/{path}')


# copied from shap.utils._general.safe_isinstance
def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, (list, tuple)):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']

    # try each module path in order
    for _class_path_str in class_path_strs:
        if "." not in _class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurence of "."
        module_name, class_name = _class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        #Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False

def safe_import(module_name, class_name=None):
    '''safely import a module, if it is not installed, print a message and return None
    '''
    try:
        imported_module = importlib.import_module(module_name)
        if class_name is None:
            return imported_module
        if isinstance(class_name, (list, np.ndarray)):
            return [getattr(imported_module, name) for name in class_name]
        return getattr(imported_module, class_name)
    except ImportError as exc:
        raise ImportError(f"The required module '{module_name}' is not installed. "
                f"Please install it using 'pip install {module_name}' or another method.") from exc
    except AttributeError as exc:
        raise ImportError(f"The class or function '{class_name}' does "+
                f"not exist in the module '{module_name}'.") from exc

# copied from sklearn.utils.validation.check_is_fitted
def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to specify if the
    estimator is fitted or not.

    Parameters
    ----------
    estimator : estimator instance
        estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError(f"{estimator} is a class, not an instance.")
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError(f"{estimator} is not an estimator instance.")

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        fitted = all_or_any([hasattr(estimator, attr) for attr in attributes])
    elif hasattr(estimator, "__sklearn_is_fitted__"):
        fitted = estimator.__sklearn_is_fitted__()
    else:
        fitted = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    if not fitted or fitted == []:
        raise RuntimeError(msg % {"name": type(estimator).__name__})

def is_notebook():
    '''
    Check if the code is running in a Jupyter notebook.

    This returns False when not running from a notebook:

    >>> is_notebook()
    False
    '''
    try:
        # pylint: disable=import-outside-toplevel
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except (ImportError, AttributeError):
        return False
    return True

# pylint: disable=too-many-locals, too-many-branches
def transform_to_numeric(df, target, mappings=None):
    '''
    Transform the categorical features to numeric
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to transform
    target : str
        The target column name
    categorical_features : list, optional
        The list of categorical features, by default None
    mappings : dict, optional
        The mapping created by previous calls to this function, by default None
    
    Returns
    -------
    pd.DataFrame
        The transformed dataframe
    Categorical features
        A list of the indexes to categorical features
    Categorical labels
        A dictionary with a list of categorical labels (value) for each categorical feature (key)
    Target labels
        A dictionary with target label-index pairs
    Mappings
        A dictionary with the mapping of each categorical feature and the target
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'target': ['a','b']}) 
    >>> transform_to_numeric(df,'target')        
    (  target
    0      0
    1      1, None, None, {0: 'a', 1: 'b'}, {'target': {'a': 0, 'b': 1}})
    
    >>> df = pd.DataFrame({'numerical': [2,3], 'nominal': ['c','d'], 'target': ['a','b']})
    >>> ndf, categorical_features, categorical_labels, target_labels, mappings = transform_to_numeric(df,'target')
    >>> ndf
       numerical nominal target
    0          2       0      0
    1          3       1      1
    >>> categorical_features
    [1]
    >>> categorical_labels
    {1: {0: 'c', 1: 'd'}}
    >>> target_labels
    {0: 'a', 1: 'b'}
    >>> mappings
    {'nominal': {'c': 0, 'd': 1}, 'target': {'a': 0, 'b': 1}}
    
    >>> ddf = pd.DataFrame({'numerical': [2,3], 'nominal': ['d','c'], 'target': ['b','a']})
    >>> nddf, _, _, _, _ = transform_to_numeric(ddf,'target', mappings)
    >>> nddf
       numerical  nominal  target
    0          2        1       1
    1          3        0       0
    '''
    if mappings is None:
        categorical_features = []
        categorical_labels = {}
        target_labels = None
        mappings = {}
    else:
        categorical_features = [c for c in range(len(df.columns)) if df.columns[c] in mappings and df.columns[c] != target]
        categorical_labels = {c:mappings[df.columns[c]] for c in categorical_features}
        target_labels = mappings.get(target)
    for c, col in enumerate(df.columns):
        if col in mappings:
            df[col] = df[col].fillna('nan')
            df[col] = df[col].astype(str)
            df[col] = df[col].map(mappings[col])
        elif isinstance(df[col], CategoricalDtype) or df[col].dtype in (object, str):
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace("'", "")
            df[col] = df[col].str.replace('"', '')
            if isinstance(df[col], CategoricalDtype) and 'nan' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['nan'])
            df[col] = df[col].fillna('nan')
            df[col] = df[col].astype('category')
            uniques = []
            for v in df[col]:
                # if v is None or v is np.nan:
                #     v = 'nan'
                    # df[col][i] = v
                if v not in uniques:
                    uniques.append(v)

            if col != target:
                categorical_features.append(c)
                categorical_labels[c] = dict(zip(range(len(uniques)), uniques))
            else:
                target_labels = dict(zip(range(len(uniques)), uniques))
            mapping = dict(zip(uniques, range(len(uniques))))
            if len(mapping) > 5:
                counts = df[col].value_counts().sort_values(ascending=False)
                idx = 0
                for key, count in counts.items():
                    if count > 5:
                        idx += 1
                        continue
                    mapping[key] = idx
            mappings[col] = mapping
            df[col] = df[col].map(mapping)
    if categorical_features:
        return df, categorical_features, categorical_labels, target_labels, mappings
    return df, None, None, target_labels, mappings

def assert_threshold(threshold, x):
    '''
    Test if the thresholds are valid.
    
    Parameters
    ----------
    threshold : int, float, tuple, list, or np.ndarray
        The threshold(s) to be validated. It can be a scalar (int or float), 
        a tuple with two values, or a list/np.ndarray of scalars or tuples.
    x : list or np.ndarray
        The data against which the thresholds are validated. Used to check 
        the length of list/np.ndarray thresholds.
    
    Returns
    -------
    int, float, tuple, or list
        The validated threshold(s).
    
    Raises
    ------
    AssertionError
        If the length of the list/np.ndarray threshold is not equal to the number of samples.
        if the tuple threshold does not have two values.
    ValueError
        If the threshold is not a scalar, binary tuple, or list of scalars or binary tuples.
    
    Examples
    --------
    >>> assert_threshold(0.5, [1, 2, 3])
    0.5
    >>> assert_threshold((0.2, 0.8), [1, 2, 3])
    (0.2, 0.8)
    >>> assert_threshold([0.1, 0.2, 0.3], [1, 2, 3])
    [0.1, 0.2, 0.3]
    >>> assert_threshold([(0.1, 0.9), (0.2, 0.8)], [1, 2])
    [(0.1, 0.9), (0.2, 0.8)]
    >>> assert_threshold(None, [1, 2, 3])
    >>> assert_threshold([0.1, 0.2], [1])
    Traceback (most recent call last):
        ...
    AssertionError: list thresholds must have the same length as the number of samples
    '''
    if threshold is None:
        return threshold
    if np.isscalar(threshold) and isinstance(threshold, (numbers.Integral, numbers.Real)):
        return threshold
    if isinstance(threshold, tuple):
        assert len(threshold) == 2, 'tuple thresholds must be a tuple with two values'
        return threshold
    if isinstance(threshold, (list, np.ndarray)):
        assert len(threshold) == np.asarray(x).shape[0], \
            'list thresholds must have the same length as the number of samples'
        return [assert_threshold(t, [x[i]]) for i,t in enumerate(threshold)]
    raise ValueError(
        'thresholds must be a scalar, binary tuple or list of scalars or binary tuples')

# pylint: disable=too-many-arguments, too-many-statements
def calculate_metrics(uncertainty=None,
                      prediction=None,
                      w=0.5,
                      metric=None,
                      normalize=False,):
    '''
    The function `calculate_metrics` calculates different metrics based on the uncertainty and 
    probability values.

    Parameters
    ----------
    uncertainty : float
        The `uncertainty` parameter is a float value that represents the uncertainty of the
        explanation. Uncertainty is a measure of the confidence of the explanation. For 
        classification, this is a value between 0 and 1, where 0 means the explanation is certain 
        and 1 means the explanation is uncertain. For regression, this is the width of the 
        uncertainty interval determined by the user defined percentiles.
    prediction : float
        The `prediction` parameter is a float value that represents the prediction of the 
        explanation. For classification, this is the probability of the predicted class. For 
        regression, this is the predicted value.
    w : float, default=0.5
        The `w` parameter is a float value that represents the weight of the uncertainty in the 
        metric calculation. The weight must be between -1 and 1. The default value is 0.5.
    metric : str, list of str, or None, default=None
        The `metric` parameter is a string that represents the metric to calculate.
        If `metric` is set to None, the function will calculate all available metrics.
        If `metric` is set to a list of metrics, the function will calculate only those
        metrics. The available metrics are:
        - 'ensured' : Weighted Sum Method 
    normalize : bool, default=False
        The `normalize` parameter is a boolean value that represents whether to normalize the 
        uncertainty and prediction values. The default value is False.

    Notes
    -----
    If the method is called with no arguments, it will return the list of available metrics.
    '''

    # Count the number of arguments passed
    if uncertainty is None and prediction is None:
        return ['ensured'
                ]

    assert uncertainty is not None and prediction is not None, \
            'Both uncertainty and prediction must be provided if any other argument is provided'
    uncertainty = np.array(uncertainty) if isinstance(uncertainty, list) else uncertainty
    prediction = np.array(prediction) if isinstance(prediction, list) else prediction
    metrics = {}
    assert -1 <= w <= 1, 'The weight must be between -1 and 1.'
    inverse_prediction = False
    if w < 0:
        w = -w
        inverse_prediction = True
    if metric is None:
        metric = calculate_metrics()
    elif isinstance(metric, str):
        metric = [metric]
    if normalize:
        min_uncertainty, max_uncertainty = np.min(uncertainty), np.max(uncertainty)
        min_prediction, max_prediction = np.min(prediction), np.max(prediction)
        uncertainty = (uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty)
        prediction = (prediction - min_prediction) / (max_prediction - min_prediction)
    prediction = -1*prediction if inverse_prediction and prediction is not None else prediction

    if 'ensured' in metric:
        metrics['ensured'] = (1-w) * (1 - uncertainty) + w * (prediction)

    return metrics if len(metrics) > 1 else metrics[list(metrics.keys())[0]]

def convert_targets_to_numeric(y):
    """Convert string/categorical targets to numeric values while preserving labels.
    
    Parameters
    ----------
    y (array-like): Array of target values that may be strings or categorical.
        
    Returns
    -------
    tuple:
        - array-like: Numeric version of the target values
        - dict or None: Mapping of original labels to numeric values if conversion was needed
    """
    if (any(isinstance(val, str) for val in y) or
            any(isinstance(val, (np.str_, np.object_)) for val in y)):
        unique_labels = np.unique(y)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        numeric_y = np.array([label_map[label] for label in y])
        return numeric_y, label_map
    return y, None

def concatenate_thresholds(perturbed_threshold, threshold, indices):
    """
    Concatenates the given threshold values to the perturbed_threshold based on the provided indices.

    Parameters
    ----------
    perturbed_threshold : np.ndarray
        The existing perturbed thresholds.
    threshold : list or np.ndarray
        The original thresholds.
    indices : np.ndarray
        The indices to select from the threshold.

    Returns
    -------
    np.ndarray
        The concatenated thresholds.
    """
    if threshold is not None and isinstance(threshold, (list, np.ndarray)):
        if isinstance(threshold[0], tuple) and len(perturbed_threshold) == 0:
            perturbed_threshold = [threshold[i] for i in indices]
        else:
            perturbed_threshold = np.concatenate((perturbed_threshold, [threshold[i] for i in indices]))
    return perturbed_threshold


if __name__ == "__main__":
    import doctest
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
