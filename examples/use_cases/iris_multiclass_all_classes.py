"""Iris multiclass example with all-classes explanations enabled."""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer


def main() -> None:
    """Run a multiclass Iris workflow and print per-class intervals."""
    x, y = load_iris(return_X_y=True)
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.4, random_state=42, stratify=y
    )
    x_cal, x_test, y_cal, _ = train_test_split(
        x_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    wrapper = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(
        x_cal,
        y_cal,
        mode="classification",
        class_labels={0: "setosa", 1: "versicolor", 2: "virginica"},
    )

    probs, (low, high) = wrapper.predict_proba(x_test[:1], uq_interval=True)
    print("Per-class calibrated probabilities with intervals (sample 0):")
    for class_idx, prob in enumerate(probs[0]):
        print(
            f"  class={class_idx}: p={prob:.4f}, "
            f"interval=({low[0, class_idx]:.4f}, {high[0, class_idx]:.4f})"
        )

    all_classes_factual = wrapper.explain_factual(x_test[:1], multi_labels_enabled=True)
    all_classes_alternative = wrapper.explore_alternatives(x_test[:1], multi_labels_enabled=True)

    print("\nPer-class factual narratives (sample 0):")
    for class_idx in sorted(all_classes_factual.explanations[0].keys()):
        explanation = all_classes_factual[(0, class_idx)]
        print(f"\nClass {class_idx}")
        print(explanation.to_narrative(output_format="text"))

    # Saves a per-instance multiclass plot without opening a GUI window.
    all_classes_alternative.plot(index=0, show=False, filename="iris_multiclass_alternatives.png")


if __name__ == "__main__":
    main()
