# Manual Triage — Prioritized Next Actions

This file lists prioritized private-member symbols with recommended actions.

## Top Symbols to Remediate

| Name | Count | In Src | Samples |
| :--- | :--- | :--- | :--- |
| _plugin_manager | 9 | True | tests\unit\test_explanations_collection.py:60;tests\unit\test_explanations_collection.py:64;tests\unit\test_explanations_collection.py:68 |
| _EMITTED_PER_TEST | 8 | False | tests\unit\test_utils_deprecations.py:53;tests\unit\test_utils_deprecations.py:72;tests\unit\utils\test_deprecations_helper.py:24 |
| _should_raise | 8 | True | tests\unit\utils\test_deprecations_helper.py:36;tests\unit\utils\test_deprecations_helper.py:41;tests\unit\utils\test_deprecations_helper.py:46 |
| _EMITTED | 7 | False | tests\unit\test_utils_deprecations.py:55;tests\unit\test_utils_deprecations.py:64;tests\unit\test_utils_deprecations.py:74 |
| _cmd_list | 6 | True | tests\plugins\test_cli.py:146;tests\plugins\test_cli.py:169;tests\plugins\test_cli.py:181 |
| _explanation_attr | 5 | False | tests\plugins\test_runtime_validation.py:37;tests\unit\plugins\test_execution_strategy_wrappers.py:144;tests\unit\plugins\test_execution_strategy_wrappers.py:158 |
| _mode | 5 | False | tests\plugins\test_runtime_validation.py:44;tests\unit\plugins\test_execution_strategy_wrappers.py:143;tests\unit\plugins\test_execution_strategy_wrappers.py:157 |
| _append_bins | 5 | True | tests\unit\core\test_interval_regressor.py:190;tests\unit\core\test_interval_regressor.py:531;tests\unit\core\test_interval_regressor.py:547 |
| _bins_storage | 5 | True | tests\unit\core\test_interval_regressor.py:529;tests\unit\core\test_interval_regressor.py:662;tests\unit\core\test_interval_regressor.py:217 |
| _validate_prediction_result | 5 | True | tests\unit\core\prediction\test_orchestrator_coverage.py:818;tests\unit\core\prediction\test_orchestrator_coverage.py:823;tests\unit\core\prediction\test_orchestrator_coverage.py:97 |
| _explanation_plugin_instances | 4 | True | tests\integration\core\test_plugin_resolution.py:139;tests\unit\plugins\test_manager.py:37;tests\unit\plugins\test_manager.py:227 |
| _plot_proba_triangle | 4 | True | tests\integration\viz\test_plots_integration.py:96;tests\integration\viz\test_plots_more.py:12;tests\unit\test_plotting.py:629 |
| _register_builtins | 4 | True | tests\plugins\test_builtins_behaviour.py:659;tests\plugins\test_builtins_behaviour.py:673;tests\plugins\test_builtins_module.py:622 |
| _cmd_show | 4 | True | tests\plugins\test_cli.py:218;tests\plugins\test_cli.py:227;tests\plugins\test_cli.py:247 |
| _cmd_trust | 4 | True | tests\plugins\test_cli.py:264;tests\plugins\test_cli.py:288;tests\plugins\test_cli.py:302 |
| _read_plot_pyproject | 4 | True | tests\unit\test_plots_configuration.py:35;tests\unit\test_plots_configuration.py:26;tests\unit\test_plots_configuration.py:45 |
| _fast_interval_plugin_override | 4 | True | tests\unit\core\test_calibrated_explainer_runtime_helpers.py:30;tests\unit\plugins\test_manager.py:28;tests\unit\plugins\test_manager.py:79 |
| _interval_plugin_override | 4 | True | tests\unit\core\test_calibrated_explainer_runtime_helpers.py:530;tests\unit\plugins\test_manager.py:27;tests\unit\plugins\test_manager.py:78 |
| _bins_size | 4 | True | tests\unit\core\test_interval_regressor.py:530;tests\unit\core\test_interval_regressor.py:533;tests\unit\core\test_interval_regressor.py:550 |
| _pre_fit_preprocess | 4 | True | tests\unit\core\test_wrap_explainer_helpers.py:148;tests\unit\core\test_wrap_explainer_helpers.py:170;tests\unit\core\test_wrap_explainer_helpers.py:405 |
| _format_proba_output | 4 | True | tests\unit\core\test_wrap_explainer_helpers.py:197;tests\unit\core\test_wrap_explainer_helpers.py:202;tests\unit\core\test_wrap_explainer_helpers.py:206 |
| _normalize_auto_encode_flag | 4 | True | tests\unit\core\test_wrap_explainer_helpers.py:88;tests\unit\core\test_wrap_explainer_helpers.py:90;tests\unit\core\test_wrap_explainer_helpers.py:92 |
| _bridge_monitors | 4 | True | tests\unit\core\explain\test_explain_orchestrator.py:23;tests\unit\plugins\test_manager.py:36;tests\unit\plugins\test_manager.py:192 |
| _predict | 3 | True | tests\unit\test_explanations_collection.py:56;tests\unit\test_explanations_collection.py:81;tests\unit\core\test_calibrated_explainer_lime.py:105 |
| _initialized | 3 | False | tests\unit\core\test_calibration_helpers.py:83;tests\unit\core\test_calibration_helpers.py:87;tests\unit\core\test_calibration_helpers.py:91 |
| _y_cal_hat_size | 3 | True | tests\unit\core\test_interval_regressor.py:521;tests\unit\core\test_interval_regressor.py:524;tests\unit\core\test_interval_regressor.py:543 |
| _append_calibration_buffer | 3 | True | tests\unit\core\test_interval_regressor.py:182;tests\unit\core\test_interval_regressor.py:522;tests\unit\core\test_interval_regressor.py:541 |
| _parallel_executor | 3 | False | tests\unit\core\test_parallel_benchmark.py:32;tests\unit\core\test_parallel_benchmark.py:37;tests\unit\core\test_parallel_benchmark.py:41 |
| _serialise_preprocessor_value | 3 | True | tests\unit\core\test_wrap_explainer_helpers.py:109;tests\unit\core\test_wrap_explainer_helpers.py:372;tests\unit\core\test_wrap_explainer_helpers.py:378 |
| _resolve_strategy | 3 | True | tests\unit\core\explain\test_parallel_lifecycle.py:75;tests\unit\core\explain\test_parallel_lifecycle.py:81;tests\unit\core\explain\test_parallel_lifecycle.py:91 |
| _build_uncertainty_payload | 3 | True | tests\unit\explanations\test_explanation_helpers.py:117;tests\unit\explanations\test_explanation_helpers.py:515;tests\unit\explanations\test_explanation_helpers.py:526 |
| _build_condition_payload | 3 | True | tests\unit\explanations\test_explanation_helpers.py:149;tests\unit\explanations\test_explanation_helpers.py:605;tests\unit\explanations\test_explanation_helpers.py:615 |
| _build_instance_uncertainty | 3 | True | tests\unit\explanations\test_explanation_helpers.py:559;tests\unit\explanations\test_explanation_helpers.py:568;tests\unit\explanations\test_explanation_helpers.py:574 |
| _safe_feature_name | 3 | True | tests\unit\explanations\test_explanation_helpers.py:583;tests\unit\explanations\test_explanation_helpers.py:584;tests\unit\explanations\test_explanation_helpers.py:585 |
| _supports_calibrated_explainer | 2 | True | tests\plugins\test_builtins_behaviour.py:136;tests\plugins\test_builtins_behaviour.py:137 |
| _emit_explanation_descriptor | 2 | True | tests\plugins\test_cli.py:102;tests\plugins\test_cli_additional.py:57 |
| _emit_interval_descriptor | 2 | True | tests\plugins\test_cli.py:103;tests\plugins\test_cli_additional.py:72 |
| _emit_plot_descriptor | 2 | True | tests\plugins\test_cli.py:104;tests\plugins\test_cli_additional.py:91 |
| _emit_plot_builder_descriptor | 2 | True | tests\plugins\test_cli.py:105;tests\plugins\test_cli_additional.py:113 |
| _hash_part | 2 | True | tests\unit\test_perf_cache_shim.py:19;tests\unit\test_perf_cache_shim.py:19 |
| _legacy_get_fill_color | 2 | True | tests\unit\test_plot_spec_default_builder.py:10;tests\unit\test_plot_spec_default_builder.py:11 |
| _sequential_plugin | 2 | True | tests\unit\core\test_calibrated_explainer_additional.py:572;tests\unit\core\explain\test_parallel_executors.py:148 |
| _pyproject_intervals | 2 | True | tests\unit\core\test_calibrated_explainer_runtime_helpers.py:40;tests\unit\core\test_calibrated_explainer_runtime_helpers.py:131 |
| _pyproject_plots | 2 | True | tests\unit\core\test_calibrated_explainer_runtime_helpers.py:41;tests\unit\core\test_calibrated_explainer_runtime_helpers.py:132 |
| _CalibratedExplainer__noise_type | 2 | False | tests\unit\core\test_calibrated_explainer_runtime_helpers.py:557;tests\unit\core\prediction\test_orchestrator_coverage.py:687 |
| _residual_cal_storage | 2 | True | tests\unit\core\test_interval_regressor.py:173;tests\unit\core\test_interval_regressor.py:652 |
| _sigma_cal_storage | 2 | True | tests\unit\core\test_interval_regressor.py:174;tests\unit\core\test_interval_regressor.py:653 |
| _schema_json | 2 | True | tests\unit\core\test_schema_payload_validation.py:20;tests\unit\core\test_schema_payload_validation.py:30 |
| _build_preprocessor_metadata | 2 | True | tests\unit\core\test_wrap_explainer_helpers.py:136;tests\unit\core\test_wrap_explainer_helpers.py:132 |
| _pre_transform | 2 | True | tests\unit\core\test_wrap_explainer_helpers.py:152;tests\unit\core\test_wrap_explainer_helpers.py:174 |
