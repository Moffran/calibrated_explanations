from calibrated_explanations.viz.coloring import get_fill_color


def test_get_color_exception_predict():
    # Trigger first try-except in get_fill_color
    color = get_fill_color({"predict": None})  # float(None) raises TypeError
    assert color.startswith("#")


def test_get_color_exception_alpha():
    # Trigger second try-except in get_fill_color
    # We need pred to be a float to pass the first try-except,
    # but then alpha calculation to fail.
    # This is hard because alpha is derived from pred.
    # Wait, if pred is a custom object that supports >= but not subtraction?
    class BadFloat(float):
        def __sub__(self, other):
            raise ValueError("Bad subtraction")

    color = get_fill_color({"predict": BadFloat(0.8)})
    assert color.startswith("#")
