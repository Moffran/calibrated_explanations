The calibrated-explanations package
===================================

.. toctree::
   :maxdepth: 1
   :caption: Explainers:

   _autosummary/calibrated_explanations.core.CalibratedExplainer
   _autosummary/calibrated_explanations.core.WrapCalibratedExplainer
.. toctree::
   :maxdepth: 1
   :caption: Collections:

   _autosummary/calibrated_explanations.explanations.CalibratedExplanations
   _autosummary/calibrated_explanations.explanations.AlternativeExplanations
.. toctree::
   :maxdepth: 1
   :caption: Explanations:

   _autosummary/calibrated_explanations.explanations.CalibratedExplanation
   _autosummary/calibrated_explanations.explanations.FactualExplanation
   _autosummary/calibrated_explanations.explanations.AlternativeExplanation
   _autosummary/calibrated_explanations.explanations.FastExplanation
.. toctree::
   :maxdepth: 1
   :caption: Other:

   _autosummary/calibrated_explanations.utils.helper
   _autosummary/calibrated_explanations.core.exceptions
   _autosummary/calibrated_explanations.core.validation
   _autosummary/calibrated_explanations.api.params
   _autosummary/calibrated_explanations.api.config

API notes
---------

Parameter aliases and deprecations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function ``calibrated_explanations.api.params.canonicalize_kwargs`` maps known
aliases to canonical parameter names in a non-destructive way (e.g., ``alpha``/``alphas``
→ ``low_high_percentiles``; ``n_jobs`` → ``parallel_workers``). The helper
``warn_on_aliases`` emits a ``DeprecationWarning`` at public call sites when alias keys
are used, guiding users to the canonical names. During the current phase, behavior is
unchanged: original keys are preserved and only warnings are raised.

Configuration scaffolding
~~~~~~~~~~~~~~~~~~~~~~~~~

``calibrated_explanations.api.config`` provides a light-weight ``ExplainerConfig`` dataclass
and an ``ExplainerBuilder`` fluent API to assemble configuration in a typed way. This is
future-facing scaffolding; it currently does not alter runtime behavior unless used via
internal/private paths. Public API exports remain unchanged to keep the snapshot stable.

Config-driven defaults and preprocessing (Phase 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When constructing the wrapper through the private helper ``WrapCalibratedExplainer._from_config``,
some configuration fields are honored without changing the public API:

- ``threshold`` and ``low_high_percentiles``: used as defaults in ``explain_factual``,
   ``explore_alternatives``, and ``explain_fast`` when not explicitly provided.
- ``preprocessor``: if supplied (e.g., a scikit-learn ``Pipeline``/``ColumnTransformer``), the
   wrapper will fit it on the first call to ``fit``/``calibrate`` and apply it for
   ``fit``, ``calibrate``, and explain calls.

Notes:

- This wiring is intentionally private in this phase to avoid public API changes. If you don’t
   use ``_from_config``, behavior remains identical to earlier versions.
- ``auto_encode`` and ``unseen_category_policy`` are stored for future use but not active yet.
