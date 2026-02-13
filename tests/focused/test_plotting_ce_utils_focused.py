import json

from calibrated_explanations import ce_agent_utils
from calibrated_explanations import plotting




def test_format_save_path_variants(tmp_path):
    p = plotting.format_save_path(tmp_path, "x.png")
    assert str(tmp_path) in p

    p2 = plotting.format_save_path("", "file.txt")
    assert p2 == "file.txt"

    p3 = plotting.format_save_path("dir/", "f.txt")
    assert p3.endswith("dir/f.txt") or p3.endswith("dir\\f.txt")




def test_policy_and_serialization():
    d = ce_agent_utils.policy_as_dict()
    assert isinstance(d, dict)
    assert "requires_library" in d

    s = ce_agent_utils.serialize_policy()
    # ensure it's valid json and contains expected key
    parsed = json.loads(s)
    assert parsed["requires_library"] == d["requires_library"]


def test_optional_cache_behavior():
    calls = {"n": 0}

    @ce_agent_utils.optional_cache(enabled=True, maxsize=16)
    def f():
        calls["n"] += 1
        return calls["n"]

    v1 = f()
    v2 = f()
    assert v1 == v2
    assert calls["n"] == 1

    @ce_agent_utils.optional_cache(enabled=False)
    def g():
        calls["n"] += 1
        return calls["n"]

    a1 = g()
    a2 = g()
    assert a2 == a1 + 1


