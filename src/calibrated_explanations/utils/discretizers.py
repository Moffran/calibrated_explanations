# pylint: disable=too-many-arguments, too-many-positional-arguments
"""This module defines the discretizers used by CalibratedExplainer.
The discretizers are defined using the same super class as the discretizers from the LIME package.
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state

# pylint: disable=too-many-instance-attributes
class BaseDiscretizer():
    """
    Abstract class - Build a class that inherits from this class to implement
    a custom discretizer.
    Method bins() is to be redefined in the child class, as it is the actual
    custom part of the discretizer.
    """

    __metaclass__ = ABCMeta  # abstract class

    # pylint: disable=too-many-locals
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        """Initializer
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
            data_stats: must have 'means', 'stds', 'mins' and 'maxs', use this
                if you don't want these values to be computed from data
        """
        self.to_discretize = ([x for x in range(data.shape[1])
                                if x not in categorical_features])
        self.names = {}
        self.lambdas = {}
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        self.random_state = check_random_state(random_state)

        # To override when implementing a custom binning
        bins = self.bins(data, labels)
        bins = [np.unique(x) for x in bins]

        for feature, qts in zip(self.to_discretize, bins):
            n_bins = qts.shape[0]  # Actually number of borders (= #bins-1)
            boundaries = np.min(data[:, feature]), np.max(data[:, feature])
            name = feature_names[feature]

            self.names[feature] = [f'{name} <= {qts[0]:.2f}']
            for i in range(n_bins - 1):
                self.names[feature].append(f'{qts[i]:.2f} < {name} <= {qts[i + 1]:.2f}')
            self.names[feature].append(f'{name} > {qts[n_bins - 1]:.2f}')

            self.lambdas[feature] = lambda x, qts=qts: np.searchsorted(qts, x)
            discretized = self.lambdas[feature](data[:, feature])

            self.means[feature] = []
            self.stds[feature] = []
            for x in range(n_bins + 1):
                selection = data[discretized == x, feature]
                mean = 0 if len(selection) == 0 else np.mean(selection)
                self.means[feature].append(mean)
                std = 0 if len(selection) == 0 else np.std(selection)
                std += 0.00000000001
                self.stds[feature].append(std)
            self.mins[feature] = [boundaries[0]] + qts.tolist()
            self.maxs[feature] = qts.tolist() + [boundaries[1]]

    @abstractmethod
    def bins(self, data, labels):
        """
        To be overridden
        Returns for each feature to discretize the boundaries
        that form each bin of the discretizer
        """
        raise NotImplementedError("Must override bins() method")

    def discretize(self, data):
        """Discretizes the data.
        Args:
            data: numpy 2d or 1d array
        Returns:
            numpy array of same dimension, discretized.
        """
        ret = data.copy()
        for feature, value in self.lambdas.items():
            if len(data.shape) == 1:
                ret[feature] = int(value(ret[feature]))
            else:
                ret[:, feature] = value(ret[:, feature]).astype(int)
        return ret


class EntropyDiscretizer(BaseDiscretizer):
    """a dynamic entropy discretizer for the CalibratedExplainer

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
            raise ValueError('Labels must be not None when using \
                            EntropyDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                feature_names, labels=labels,
                                random_state=random_state)
    def __repr__(self):
        return 'EntropyDiscretizer()'

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            # Entropy splitting / at most 8 bins so max_depth=3
            dt = DecisionTreeClassifier(criterion='entropy',
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
