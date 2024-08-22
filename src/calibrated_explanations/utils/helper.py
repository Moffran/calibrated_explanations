'''
Created on 2021-07-01
Author: Tuwe Löfström
'''
import os
import sys
import importlib
from inspect import isclass
import numpy as np
from pandas.api.types import is_categorical_dtype

def make_directory(path: str, save_ext=None) -> None: # pylint: disable=unused-private-member
    """ create directory if it does not exist
    """
    if save_ext is not None and len(save_ext) == 0:
        return
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    if not os.path.isdir('plots/'+path):
        os.mkdir('plots/'+path)


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
    Check if the code is running in a Jupyter notebook
    '''
    try:
        # pylint: disable=import-outside-toplevel
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

# pylint: disable=too-many-locals, too-many-branches
def transform_to_numeric(df, target, categorical_features=None, mappings=None):
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
    '''
    if categorical_features is None:
        categorical_features = []
        mappings = {}
    else:
        assert mappings is not None, 'mapping must be provided if categorical_features is provided'
    categorical_labels = {}
    target_labels = None
    for c, col in enumerate(df.columns):
        if is_categorical_dtype(df[col]) or df[col].dtype in (object, str):
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace("'", "")
            df[col] = df[col].str.replace('"', '')
            if is_categorical_dtype(df[col]) and 'nan' not in df[col].cat.categories:
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
        elif c in categorical_features:
            df[col] = df[col].fillna('nan')
            df[col] = df[col].astype(str)
            df[col] = df[col].map(mappings[col])
    if len(categorical_features) > 0:
        return df, categorical_features , categorical_labels, target_labels, mappings
    return df, None, None, target_labels, mappings

def assert_threshold(threshold, x):
    '''
    Test if the thresholds are valid
    
    Parameters
    ----------
    thresholds : list
        The list of thresholds
    
    Returns
    -------
    list
        The list of thresholds
    '''
    if threshold is None:
        return threshold
    if np.isscalar(threshold) and isinstance(threshold, (int, float)):
        return threshold
    if isinstance(threshold, tuple):
        assert len(threshold) == 2, 'tuple thresholds must be a tuple with two values'
        return threshold
    if isinstance(threshold, (list, np.ndarray)):
        assert len(threshold) == x.shape[0], \
            'list thresholds must have the same length as the number of samples'
        return [assert_threshold(t, [x[i]]) for i,t in enumerate(threshold)]
    raise ValueError(
        'thresholds must be a scalar, binary tuple or list of scalars or binary tuples')
