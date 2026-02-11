def test_load_template_file_valid_yaml(tmp_path):
    from calibrated_explanations.core.narrative_generator import load_template_file

    valid_yaml = tmp_path / "test.yaml"
    valid_yaml.write_text("key: value\nlist:\n  - item1\n  - item2")
    result = load_template_file(str(valid_yaml))
    assert result == {"key": "value", "list": ["item1", "item2"]}
