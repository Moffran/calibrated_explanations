import calibrated_explanations.plotting as legacy_plotting


def test_format_save_path_variants(tmp_path):
    p = legacy_plotting.format_save_path(tmp_path, "x.png")
    assert str(tmp_path) in p

    p2 = legacy_plotting.format_save_path("", "file.txt")
    assert p2 == "file.txt"

    p3 = legacy_plotting.format_save_path("dir/", "f.txt")
    assert p3.endswith("dir/f.txt") or p3.endswith("dir\f.txt")
