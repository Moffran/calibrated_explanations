```{admonition} Explore alternatives with the triangular plot
:class: tip

Pair `explore_alternatives` with the triangular plot walkthrough to interpret
probability shifts and uncertainty trade-offs:

1. Render `alternatives.plot(style="triangular")` to compare the calibrated
   prediction (red) against alternative scenarios (blue) across probability and
   interval axes.
2. Use the rhombus overlay to spot alternatives whose calibrated interval still
   crosses 0.5; these remain uncertain candidates that require additional
   evidence before action.
3. Read the accompanying rule table to understand which feature adjustments drive
   each alternative inside or outside the rhombus, keeping an eye on interval
   width changes.

Fast explanations now live in the external plugin lane. Install the vetted
bundle with ``pip install "calibrated-explanations[external-plugins]"`` and
follow the external plugins index before enabling it in production.
```
