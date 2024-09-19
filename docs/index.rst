.. Calibrated-explanations documentation master file, created by
   sphinx-quickstart on Mon Aug  7 14:45:06 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Calibrated-explanations's documentation!
===================================================

.. raw:: html

   <hr>

.. title:: Calibrated-explanations

``calibrated-explanations`` is a Python package that implements the calibrated-explanations method for classification and regression models. 

.. raw:: html

   <hr>

Introduction
------------

`calibrated-explanations` is a Python package for the local feature importance explanation method called Calibrated Explanations, supporting both `classification <https://doi.org/10.1016/j.eswa.2024.123154/>`_ and `regression <https://arxiv.org/abs/2308.16245/>`_.
The proposed method is based on Venn-Abers (classification & regression) and Conformal Predictive Systems (regression) and has the following characteristics:

   * Fast, reliable, stable and robust feature importance explanations for:
      * Binary classification models (`read paper <https://doi.org/10.1016/j.eswa.2024.123154/>`_)
      * Multi-class classification models (`read paper <https://easychair.org/publications/preprint/rqdD/>`_)
      * Regression models (`read paper <https://arxiv.org/abs/2308.16245/>`_)
         * Including probabilistic explanations of the probability that the target exceeds a user-defined threshold 
         * With difficulty adaptable explanations (conformal normalization) 
   * Calibration of the underlying model to ensure that predictions reflect reality.
   * Uncertainty quantification of the prediction from the underlying model and the feature importance weights. 
   * Rules with straightforward interpretation in relation to instance values and feature weights.
   * Possibility to generate alternative rules with uncertainty quantification of the expected predictions.
   * Conjunctional rules conveying feature importance for the interaction of included features.
   * Conditional rules, allowing users the ability to create contextual explanations to handle e.g. bias and fairness constraints (`read paper <https://doi.org/10.1007/978-3-031-63787-2_17/>`_). 

Below is an example of a probabilistic alternative explanation for an instance of the regression dataset California Housing (with the threshold 180 000). The light red area in the background is representing the calibrated probability interval (for the prediction being below the threshold) of the underlying model, as indicated by a Conformal Predictive System and calibrated through Venn-Abers. The darker red bars for each rule show the probability intervals that Venn-Abers indicate for an instance changing a feature value in accordance with the rule condition.

.. image:: images/counterfactual_probabilistic_house_regression.png
   :width: 800
   :align: center

Installation
------------

To install the package using pip:

.. code-block:: bash

    pip install calibrated-explanations

to install the package from conda:

.. code-block:: bash

    conda install -c conda-forge calibrated-explanations

Getting Started
---------------

Here is a basic example to get you started:

.. code-block:: python

   from calibrated_explanations import WrapCalibratedExplainer
   # Load and pre-process your data
   # Divide it into proper training, calibration, and test sets

   # Initialize the WrapCalibratedExplainer with your model
   classifier = WrapCalibratedExplainer(ClassifierOfYourChoice())
   regressor = WrapCalibratedExplainer(RegressorOfYourChoice())

   # Train your model using the proper training set
   classifier.fit(X_proper_training, y_proper_training)
   regressor.fit(X_proper_training, y_proper_training)

   # Initialize the CalibratedExplainer
   classifier.calibrate(X_calibration, y_calibration)
   regressor.calibrate(X_calibration, y_calibration)

   # Factual Explanations
   # Create factual explanations for classification
   factual_explanations = classifier.explain_factual(X_test)
   # Create factual standard explanations for regression with default 90 % uncertainty interval
   factual_explanations = regressor.explain_factual(X_test) # low_high_percentiles=(5,95)
   # Create factual standard explanations for regression with user assigned uncertainty interval
   factual_explanations = regressor.explain_factual(X_test, low_high_percentiles=(10,90))
   # Create factual probabilistic explanations for regression with user assigned threshold
   your_threshold = 1000
   factual_explanations = regressor.explain_factual(X_test, threshold=your_threshold)

   # Alternative Explanations
   # Create alternative explanations for classification
   alternative_explanations = classifier.explore_alternatives(X_test)
   # Create alternative standard explanations for regression with default 90 % uncertainty interval
   alternative_explanations = regressor.explore_alternatives(X_test) # low_high_percentiles=(5,95)
   # Create alternative standard explanations for regression with user assigned uncertainty interval
   alternative_explanations = regressor.explore_alternatives(X_test, low_high_percentiles=(10,90))
   # Create alternative probabilistic explanations for regression with user assigned threshold
   alternative_explanations = regressor.explore_alternatives(X_test, threshold=your_threshold)
   
   # Plot the explanations
   factual_explanations.plot()
   factual_explanations.plot(uncertainty=True)
   alternative_explanations.plot()

   # Add conjunctions to the explanations
   factual_conjunctions.add_conjunctions()
   alternative_conjunctions.add_conjunctions()

   # One-sided explanations for regression are easily created
   factual_upper_bounded = regressor.explain_factual(X_test, 
                              low_high_percentiles=(-np.inf,90))
   alternative_lower_bounded = regressor.explore_alternatives(X_test, 
                              low_high_percentiles=(10,np.inf))

Contents
--------
.. toctree::
    :maxdepth: 1

    Getting started <getting_started.md>
    Citing calibrated-explanations <citing.md>
    The calibrated-explanations package <calibrated_explanations>
    API Reference <api_reference>

.. autosummary::
   :toctree: 

   calibrated_explanations.core
   calibrated_explanations.explanations
   calibrated_explanations.utils.helper
