import importlib.util
import os


def test_execute_core_reject_module_from_file():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    module_path = os.path.join(
        repo_root,
        "src",
        "calibrated_explanations",
        "core",
        "reject.py",
    )

    spec = importlib.util.spec_from_file_location("ce_core_reject_file", module_path)
    module = importlib.util.module_from_spec(spec)
    # execute module to ensure its top-level lines are covered
    spec.loader.exec_module(module)  # type: ignore

    # basic smoke assertions
    assert hasattr(module, "RejectPolicy")
    assert hasattr(module, "is_policy_enabled")
    assert module.is_policy_enabled("none") is False
    assert module.is_policy_enabled(object()) is False
