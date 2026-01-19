from calibrated_explanations import plotting


def test_derive_threshold_labels_plotting():
    assert plotting.derive_threshold_labels((1, 2))[0].startswith("1.00 <= Y < 2.00")
    assert plotting.derive_threshold_labels(3) == ("Y < 3.00", "Y >= 3.00")


def test_split_csv_and_format_save_path(tmp_path):
    assert plotting.split_csv(None) == ()
    assert plotting.split_csv("") == ()
    assert plotting.split_csv("a,b, c") == ("a", "b", "c")
    assert plotting.split_csv(["x", "y"]) == ("x", "y")

    p = tmp_path / "out"
    p.mkdir()
    res = plotting.format_save_path(p, "file.png")
    assert res.endswith("file.png")

    res2 = plotting.format_save_path("", "file.png")
    assert res2 == "file.png"
