# Alternatives and triangular plots

Alternative explanations quantify how predictions and uncertainty move when you
adjust feature values. Use them alongside the triangular plot to stay aligned
with calibrated decision boundaries.

1. Start from the same calibrated explainer used in the quickstarts.
2. Call `explore_alternatives` to generate candidate rules and uncertainty
   intervals for each scenario.
3. Plot the batch with `style="triangular"` to compare the calibrated base point
   (red) against the alternative set (blue).

> 🧭 **Interpretation link:** Revisit the
> {doc}`../how-to/interpret_explanations` guide for a narrated walkthrough of the
> triangular overlays, rule ranking heuristics, and how probabilistic and
> interval regression appear side by side.
