> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Archived as of v0.8.x delivery · Implementation window: Historical (≤v0.8.x).

# API Contract

This define the minimum api contract for WrapCalibratedExplainer that may only be extended, not removed.

```python
   from calibrated_explanations import WrapCalibratedExplainer
   # Load and pre-process your data
   # Divide it into proper training, calibration, and test sets

   # Initialize the WrapCalibratedExplainer with your model
   classifier = WrapCalibratedExplainer(ClassifierOfYourChoice())
   regressor = WrapCalibratedExplainer(RegressorOfYourChoice())

   # Train your model using the proper training set
   classifier.fit(X_proper_training_cls, y_proper_training_cls)
   regressor.fit(X_proper_training_reg, y_proper_training_reg)

   # Initialize the CalibratedExplainer
   classifier.calibrate(X_calibration_cls, y_calibration_cls)
   regressor.calibrate(X_calibration_reg, y_calibration_reg)

   # Factual Explanations
   # Create factual explanations for classification
   factual_explanations = classifier.explain_factual(X_test_cls)
   # Create factual standard explanations for regression with default 90 % uncertainty interval
   factual_explanations = regressor.explain_factual(X_test_reg) # low_high_percentiles=(5,95)
   # Create factual standard explanations for regression with user assigned uncertainty interval
   factual_explanations = regressor.explain_factual(X_test_reg, low_high_percentiles=(10,90))
   # Create factual probabilistic explanations for regression with user assigned threshold
   your_threshold = 1000
   factual_explanations = regressor.explain_factual(X_test_reg, threshold=your_threshold)

   # Alternative Explanations
   # Create alternative explanations for classification
   alternative_explanations = classifier.explore_alternatives(X_test_cls)
   # Create alternative standard explanations for regression with default 90 % uncertainty interval
   alternative_explanations = regressor.explore_alternatives(X_test_reg) # low_high_percentiles=(5,95)
   # Create alternative standard explanations for regression with user assigned uncertainty interval
   alternative_explanations = regressor.explore_alternatives(X_test_reg, low_high_percentiles=(10,90))
   # Create alternative probabilistic explanations for regression with user assigned threshold
   alternative_explanations = regressor.explore_alternatives(X_test_reg, threshold=your_threshold)

   # Plot the explanations, works the same for classification and regression
   factual_explanations.plot()
   factual_explanations.plot(uncertainty=True)
   alternative_explanations.plot()

   # Add conjunctions to the explanations, works the same for classification and regression
   factual_conjunctions.add_conjunctions()
   alternative_conjunctions.add_conjunctions()

   # One-sided and asymmetric explanations for regression are easily created
   factual_upper_bounded = regressor.explain_factual(X_test_reg, low_high_percentiles=(-np.inf,90))
   alternative_lower_bounded = regressor.explore_alternatives(X_test_reg, low_high_percentiles=(10,np.inf))
   alternative_asymmetric = regressor.explore_alternatives(X_test_reg, low_high_percentiles=(10,70))

   # Output the model predictions and probabilities (without calibration)
   uncal_proba_cls = classifier.predict_proba(X_test_cls)
   uncal_y_hat_cls = classifier.predict(X_test_cls)
   uncal_y_hat_reg = regressor.predict(X_test_reg)

   # Initialize the CalibratedExplainer
   classifier.calibrate(X_calibration_cls, y_calibration_cls)
   regressor.calibrate(X_calibration_reg, y_calibration_reg)

   # Output the model predictions and probabilities (without calibration).
   uncal_proba_cls = classifier.predict_proba(X_test_cls, calibrated=False)
   uncal_y_hat_cls = classifier.predict(X_test_cls, calibrated=False)
   uncal_y_hat_reg = regressor.predict(X_test_reg, calibrated=False)

   # Output the calibrated predictions and probabilities
   calib_proba_cls = classifier.predict_proba(X_test_cls)
   calib_y_hat_cls = classifier.predict(X_test_cls)
   calib_y_hat_reg = regressor.predict(X_test_reg)
   # Get thresholded regression predictions and probabilities for labels 'y_hat > threshold' and 'y_hat <= threshold'
   your_threshold = 1000
   thrld_y_hat_reg = regressor.predict(X_test_reg, threshold=your_threshold)
   thrld_proba_reg = regressor.predict_proba(X_test_reg, threshold=your_threshold)

   # Include uncertainty interval, outputted as a tuple (low, high)
   calib_proba_cls, low_high = classifier.predict_proba(X_test_cls, uq_interval=True)
```
