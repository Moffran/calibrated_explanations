# pylint: disable=unknown-option-value
# pylint: disable=too-many-arguments, too-many-positional-arguments, line-too-long
"""
This module contains classes and functions for discretizing data.

This module defines the discretizers used by CalibratedExplainer.

The discretizers are defined using the same super class as the discretizers from the LIME package.

Classes
-------
BaseDiscretizer
    Abstract class for creating custom discretizers.
EntropyDiscretizer
    A dynamic entropy discretizer for the CalibratedExplainer.
BinaryEntropyDiscretizer
    A binary entropy discretizer for the CalibratedExplainer.
RegressorDiscretizer
    A dynamic Regressor discretizer for the CalibratedExplainer.
BinaryRegressorDiscretizer
    A dynamic binary Regressor discretizer for the CalibratedExplainer.

Methods
-------
BaseDiscretizer.__init__(self, data, categorical_features, feature_names, labels=None, random_state=None)
    Initialize the BaseDiscretizer.
BaseDiscretizer.bins(self, data, labels)
    To be overridden. Returns the boundaries that form each bin of the discretizer.
BaseDiscretizer.discretize(self, data)
    Discretizes the data.
EntropyDiscretizer.__init__(self, data, categorical_features, feature_names, labels=None, random_state=None)
    Initialize the EntropyDiscretizer.
EntropyDiscretizer.__repr__(self)
    Return a string representation of the EntropyDiscretizer.
EntropyDiscretizer.bins(self, data, labels)
    Calculate the bins for the EntropyDiscretizer.
BinaryEntropyDiscretizer.__init__(self, data, categorical_features, feature_names, labels=None, random_state=None)
    Initialize the BinaryEntropyDiscretizer.
BinaryEntropyDiscretizer.__repr__(self)
    Return a string representation of the BinaryEntropyDiscretizer.
BinaryEntropyDiscretizer.bins(self, data, labels)
    Calculate the bins for the BinaryEntropyDiscretizer.
RegressorDiscretizer.__init__(self, data, categorical_features, feature_names, labels=None, random_state=None)
    Initialize the RegressorDiscretizer.
RegressorDiscretizer.__repr__(self)
    Return a string representation of the RegressorDiscretizer.
RegressorDiscretizer.bins(self, data, labels)
    Calculate the bins for the RegressorDiscretizer.
BinaryRegressorDiscretizer.__init__(self, data, categorical_features, feature_names, labels=None, random_state=None)
    Initialize the BinaryRegressorDiscretizer.
BinaryRegressorDiscretizer.__repr__(self)
    Return a string representation of the BinaryRegressorDiscretizer.
BinaryRegressorDiscretizer.bins(self, data, labels)
    Calculate the bins for the BinaryRegressorDiscretizer.
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state

# pylint: disable=too-many-instance-attributes
class BaseDiscretizer():
    """
    Abstract class. Build a class that inherits from this class to implement a custom discretizer.

    Method bins() is to be redefined in the child class, as it is the actual
    custom part of the discretizer.

    Methods
    -------
    __init__(self, data, categorical_features, feature_names, labels=None, random_state=None)
        Initialize the BaseDiscretizer.
    bins(self, data, labels)
        To be overridden. Returns the boundaries that form each bin of the discretizer.
    discretize(self, data)
        Discretizes the data.
    """

    __metaclass__ = ABCMeta  # abstract class

    # pylint: disable=too-many-locals
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        """
        Initialize the BaseDiscretizer.

        Parameters
        ----------
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
        To be overridden.

        Returns for each feature to discretize the boundaries
        that form each bin of the discretizer.
        """
        raise NotImplementedError("Must override bins() method")

    def discretize(self, data):
        """
        Discretizes the data.

        Parameters
        ----------
            data: numpy 2d or 1d array

        Returns
        -------
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
    """A dynamic entropy discretizer for the CalibratedExplainer.

    Arguments
    ---------        
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

    Methods
    -------
    __init__(self, data, categorical_features, feature_names, labels=None, random_state=None)
        Initialize the EntropyDiscretizer.
    __repr__(self)
        Return a string representation of the EntropyDiscretizer.
    bins(self, data, labels)
        Calculate the bins for the EntropyDiscretizer.
    """

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        """Initialize the EntropyDiscretizer."""
        if labels is None:
            raise ValueError('Labels must be not None when using \
                            EntropyDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                feature_names, labels=labels,
                                random_state=random_state)

    def __repr__(self):
        """Return a string representation of the EntropyDiscretizer."""
        return 'EntropyDiscretizer()'

    def bins(self, data, labels):
        """Calculate the bins for the EntropyDiscretizer."""
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
    """A binary entropy discretizer for the CalibratedExplainer.

    Arguments
    ---------
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

    Methods
    -------
    __init__(self, data, categorical_features, feature_names, labels=None, random_state=None)
        Initialize the BinaryEntropyDiscretizer.
    __repr__(self)
        Return a string representation of the BinaryEntropyDiscretizer.
    bins(self, data, labels)
        Calculate the bins for the BinaryEntropyDiscretizer.
    """

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        """Initialize the BinaryEntropyDiscretizer."""
        if labels is None:
            raise ValueError('Labels must not be None when using '+\
                            'BinaryEntropyDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                feature_names, labels=labels,
                                random_state=random_state)

    def __repr__(self):
        """Return a string representation of the BinaryEntropyDiscretizer."""
        return 'BinaryEntropyDiscretizer()'

    # pylint: disable=invalid-name
    def bins(self, data, labels):
        """Calculate the bins for the BinaryEntropyDiscretizer."""
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
    """A dynamic Regressor discretizer for the CalibratedExplainer.

    Arguments
    ---------
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

    Methods
    -------
    __init__(self, data, categorical_features, feature_names, labels=None, random_state=None)
        Initialize the RegressorDiscretizer.
    __repr__(self)
        Return a string representation of the RegressorDiscretizer.
    bins(self, data, labels)
        Calculate the bins for the RegressorDiscretizer.
    """

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        """Initialize the RegressorDiscretizer."""
        if labels is None:
            raise ValueError('Labels must not be None when using '+\
                            'RegressorDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                feature_names, labels=labels,
                                random_state=random_state)

    def __repr__(self):
        """Return a string representation of the RegressorDiscretizer."""
        return 'RegressorDiscretizer()'

    # pylint: disable=invalid-name
    def bins(self, data, labels):
        """Calculate the bins for the RegressorDiscretizer."""
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
    """A dynamic binary Regressor discretizer for the CalibratedExplainer.

    Arguments
    ---------
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

    Methods
    -------
    __init__(self, data, categorical_features, feature_names, labels=None, random_state=None)
        Initialize the BinaryRegressorDiscretizer.
    __repr__(self)
        Return a string representation of the BinaryRegressorDiscretizer.
    bins(self, data, labels)
        Calculate the bins for the BinaryRegressorDiscretizer.
    """

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        """Initialize the BinaryRegressorDiscretizer."""
        if labels is None:
            raise ValueError('Labels must not be None when using '+\
                            'BinaryRegressorDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                feature_names, labels=labels,
                                random_state=random_state)

    def __repr__(self):
        """Return a string representation of the BinaryRegressorDiscretizer."""
        return 'BinaryRegressorDiscretizer()'

    # pylint: disable=invalid-name
    def bins(self, data, labels):
        """Calculate the bins for the BinaryRegressorDiscretizer."""
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
