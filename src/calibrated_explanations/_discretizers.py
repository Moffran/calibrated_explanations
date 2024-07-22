# pylint: disable=too-many-arguments
# flake8: noqa: E501
"""This module defines the discretizers used by CalibratedExplainer.
The discretizers are defined using the same super class as the discretizers from the LIME package.
"""
from lime.discretize import BaseDiscretizer, QuartileDiscretizer, DecileDiscretizer, EntropyDiscretizer  # pylint: disable=unused-import, line-too-long
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np

# pylint: disable=unused-argument
def __repr__(self):
    return 'EntropyDiscretizer()'
EntropyDiscretizer.__repr__ = __repr__

class BinaryEntropyDiscretizer(BaseDiscretizer):
    """a binary entropy discretizer for the CalibratedExplainer

    Args:        
        data: numpy 2d array
        categorical_features: list of indices (ints) corresponding to the
            categorical columns. These features will not be discretized.
            Everything else will be considered continuous, and will be
            discretized.
        categorical_names: map from int to list of names, where
            categorical_names[x][y] represents the name of the yth value of
            column x.
        feature_names: list of names (strings) corresponding to the columns
            in the training data.
        labels: must have target labels of the data  
    """
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        if labels is None:
            raise ValueError('Labels must not be None when using '+\
                             'BinaryEntropyDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)
    def __repr__(self):
        return 'BinaryEntropyDiscretizer()'
    # pylint: disable=invalid-name
    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            # Entropy splitting / at most 2 bins so max_depth=1
            dt = DecisionTreeClassifier(criterion='entropy',
                                                     max_depth=1,
                                                     random_state=self.random_state)
            x = np.reshape(data[:, feature], (-1, 1))
            dt.fit(x, labels)
            qts = dt.tree_.threshold[np.where(dt.tree_.children_left > -1)]

            if qts.shape[0] == 0:
                qts = np.array([np.median(data[:, feature])])
            else:
                qts = np.sort(qts)

            bins.append(qts)

        return bins


class RegressorDiscretizer(BaseDiscretizer):
    """a dynamic Regressor discretizer for the CalibratedExplainer

    Args:        
        data: numpy 2d array
        categorical_features: list of indices (ints) corresponding to the
            categorical columns. These features will not be discretized.
            Everything else will be considered continuous, and will be
            discretized.
        categorical_names: map from int to list of names, where
            categorical_names[x][y] represents the name of the yth value of
            column x.
        feature_names: list of names (strings) corresponding to the columns
            in the training data.
        labels: must have target values of the data
    """
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        if labels is None:
            raise ValueError('Labels must not be None when using '+\
                             'RegressorDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)
    def __repr__(self):
        return 'RegressorDiscretizer()'
    # pylint: disable=invalid-name
    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            # Entropy splitting / at most 2 bins so max_depth=1
            dt = DecisionTreeRegressor(criterion='absolute_error',
                                                     max_depth=3,
                                                     random_state=self.random_state)
            x = np.reshape(data[:, feature], (-1, 1))
            dt.fit(x, labels)
            qts = dt.tree_.threshold[np.where(dt.tree_.children_left > -1)]

            if qts.shape[0] == 0:
                qts = np.array([np.median(data[:, feature])])
            else:
                qts = np.sort(qts)

            bins.append(qts)

        return bins


class BinaryRegressorDiscretizer(BaseDiscretizer):
    """a dynamic binary Regressor discretizer for the CalibratedExplainer

    Args:        
        data: numpy 2d array
        categorical_features: list of indices (ints) corresponding to the
            categorical columns. These features will not be discretized.
            Everything else will be considered continuous, and will be
            discretized.
        categorical_names: map from int to list of names, where
            categorical_names[x][y] represents the name of the yth value of
            column x.
        feature_names: list of names (strings) corresponding to the columns
            in the training data.
        labels: must have target values of the data
    """
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        if labels is None:
            raise ValueError('Labels must not be None when using '+\
                             'BinaryRegressorDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)
    def __repr__(self):
        return 'BinaryRegressorDiscretizer()'
    # pylint: disable=invalid-name
    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            # Entropy splitting / at most 2 bins so max_depth=1
            dt = DecisionTreeRegressor(criterion='absolute_error',
                                                     max_depth=1,
                                                     random_state=self.random_state)
            x = np.reshape(data[:, feature], (-1, 1))
            dt.fit(x, labels)
            qts = dt.tree_.threshold[np.where(dt.tree_.children_left > -1)]

            if qts.shape[0] == 0:
                qts = np.array([np.median(data[:, feature])])
            else:
                qts = np.sort(qts)

            bins.append(qts)

        return bins
