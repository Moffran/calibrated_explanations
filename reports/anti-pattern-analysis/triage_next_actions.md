# Manual Triage — Prioritized Next Actions

This file lists prioritized private-member symbols with recommended actions.

## Top Symbols to Remediate

| Name | Count | In Src | Samples |
| :--- | :--- | :--- | :--- |
| _dll | 88 | False | .conda\Lib\ctypes\test\test_cfuncs.py:14;.conda\Lib\ctypes\test\test_cfuncs.py:16;.conda\Lib\ctypes\test\test_cfuncs.py:19 |
| _str | 51 | False | .conda\Lib\tkinter\test\widget_tests.py:29;.conda\Lib\tkinter\test\test_tkinter\test_geometry_managers.py:154;.conda\Lib\tkinter\test\test_tkinter\test_geometry_managers.py:157 |
| _Call | 50 | False | .conda\Lib\unittest\test\testmock\testhelpers.py:93;.conda\Lib\unittest\test\testmock\testhelpers.py:106;.conda\Lib\unittest\test\testmock\testhelpers.py:112 |
| _objects | 42 | False | .conda\Lib\ctypes\test\test_cast.py:36;.conda\Lib\ctypes\test\test_cast.py:38;.conda\Lib\ctypes\test\test_cast.py:42 |
| _testfunc_p_p | 25 | False | .conda\Lib\ctypes\test\test_as_parameter.py:30;.conda\Lib\ctypes\test\test_checkretval.py:20;.conda\Lib\ctypes\test\test_checkretval.py:23 |
| _assert_highlighting | 25 | False | .conda\Lib\idlelib\idle_test\test_colorizer.py:470;.conda\Lib\idlelib\idle_test\test_colorizer.py:473;.conda\Lib\idlelib\idle_test\test_colorizer.py:477 |
| _run_test | 24 | False | .conda\Lib\unittest\test\test_result.py:679;.conda\Lib\unittest\test\test_result.py:681;.conda\Lib\unittest\test\test_result.py:683 |
| _runnable_test | 23 | False | .conda\Lib\unittest\test\testmock\testasync.py:809;.conda\Lib\unittest\test\testmock\testasync.py:935;.conda\Lib\unittest\test\testmock\testasync.py:942 |
| _logs | 21 | False | .conda\Lib\distutils\tests\test_config_cmd.py:24;.conda\Lib\distutils\tests\test_config_cmd.py:44;.conda\Lib\distutils\tests\test_config_cmd.py:20 |
| _fields_ | 17 | False | .conda\Lib\ctypes\test\test_pep3118.py:91;.conda\Lib\ctypes\test\test_pep3118.py:108;.conda\Lib\ctypes\test\test_structures.py:45 |
| _config_vars | 17 | False | .conda\Lib\distutils\tests\test_build_ext.py:39;.conda\Lib\distutils\tests\test_build_ext.py:51;.conda\Lib\distutils\tests\test_build_ext.py:52 |
| _dt_test | 15 | False | .conda\Lib\doctest.py:2200;.conda\Lib\doctest.py:2205;.conda\Lib\doctest.py:2212 |
| _do_discovery | 15 | False | .conda\Lib\unittest\test\test_discovery.py:585;.conda\Lib\unittest\test\test_discovery.py:597;.conda\Lib\unittest\test\test_discovery.py:609 |
| _module_cleanups | 15 | False | .conda\Lib\unittest\test\test_runner.py:606;.conda\Lib\unittest\test\test_runner.py:614;.conda\Lib\unittest\test\test_runner.py:629 |
| _create_files | 14 | False | .conda\Lib\distutils\tests\test_archive_util.py:69;.conda\Lib\distutils\tests\test_archive_util.py:76;.conda\Lib\distutils\tests\test_archive_util.py:81 |
| _get_cmd | 13 | False | .conda\Lib\distutils\tests\test_register.py:107;.conda\Lib\distutils\tests\test_register.py:162;.conda\Lib\distutils\tests\test_register.py:173 |
| _type_ | 12 | False | .conda\Lib\ctypes\test\test_arrays.py:166;.conda\Lib\ctypes\test\test_arrays.py:168;.conda\Lib\ctypes\test\test_arrays.py:171 |
| _indent | 10 | False | .conda\Lib\doctest.py:1291;.conda\Lib\doctest.py:1276;.conda\Lib\doctest.py:1737 |
| _runtime_warn | 10 | False | .conda\Lib\unittest\test\test_case.py:1481;.conda\Lib\unittest\test\test_case.py:1484;.conda\Lib\unittest\test\test_case.py:1492 |
| _test | 9 | False | .conda\Lib\doctest.py:2832;.conda\Lib\unittest\test\testmock\testpatch.py:934;.conda\Lib\unittest\test\testmock\testpatch.py:1327 |
| _run | 9 | False | .conda\Lib\distutils\tests\test_check.py:44;.conda\Lib\distutils\tests\test_check.py:53;.conda\Lib\distutils\tests\test_check.py:58 |
| _exes | 9 | False | .conda\Lib\distutils\tests\test_cygwinccompiler.py:19;.conda\Lib\distutils\tests\test_cygwinccompiler.py:39;.conda\Lib\distutils\tests\test_cygwinccompiler.py:56 |
| _click_increment_arrow | 9 | False | .conda\Lib\tkinter\test\test_ttk\test_widgets.py:1187;.conda\Lib\tkinter\test\test_ttk\test_widgets.py:1197;.conda\Lib\tkinter\test\test_ttk\test_widgets.py:1207 |
| _get_module_from_name | 9 | False | .conda\Lib\unittest\test\test_discovery.py:73;.conda\Lib\unittest\test\test_discovery.py:119;.conda\Lib\unittest\test\test_discovery.py:174 |
| _subtest | 9 | False | .conda\Lib\unittest\test\test_result.py:406;.conda\Lib\unittest\test\test_result.py:488;.conda\Lib\unittest\test\test_result.py:495 |
| _testfunc_byval | 8 | False | .conda\Lib\ctypes\test\test_as_parameter.py:148;.conda\Lib\ctypes\test\test_as_parameter.py:156;.conda\Lib\ctypes\test\test_as_parameter.py:157 |
| _remove_original_values | 8 | False | .conda\Lib\distutils\tests\test_util.py:91;.conda\Lib\distutils\tests\test_util.py:105;.conda\Lib\distutils\tests\test_util.py:114 |
| _click_decrement_arrow | 8 | False | .conda\Lib\tkinter\test\test_ttk\test_widgets.py:1191;.conda\Lib\tkinter\test\test_ttk\test_widgets.py:1198;.conda\Lib\tkinter\test\test_ttk\test_widgets.py:1219 |
| _top_level_dir | 8 | False | .conda\Lib\unittest\test\test_discovery.py:32;.conda\Lib\unittest\test\test_discovery.py:83;.conda\Lib\unittest\test\test_discovery.py:129 |
| _find_tests | 8 | False | .conda\Lib\unittest\test\test_discovery.py:401;.conda\Lib\unittest\test\test_discovery.py:798;.conda\Lib\unittest\test\test_discovery.py:84 |
| _mk_TestSuite | 8 | False | .conda\Lib\unittest\test\test_suite.py:33;.conda\Lib\unittest\test\test_suite.py:33;.conda\Lib\unittest\test\test_suite.py:36 |
| _name2ft | 7 | False | .conda\Lib\doctest.py:1234;.conda\Lib\doctest.py:1592;.conda\Lib\doctest.py:1446 |
| _testfunc_callback_i_if | 7 | False | .conda\Lib\ctypes\test\test_as_parameter.py:54;.conda\Lib\ctypes\test\test_as_parameter.py:73;.conda\Lib\ctypes\test\test_as_parameter.py:109 |
| _created_files | 7 | False | .conda\Lib\distutils\tests\test_archive_util.py:120;.conda\Lib\distutils\tests\test_archive_util.py:179;.conda\Lib\distutils\tests\test_archive_util.py:180 |
| _tarinfo | 7 | False | .conda\Lib\distutils\tests\test_archive_util.py:120;.conda\Lib\distutils\tests\test_archive_util.py:179;.conda\Lib\distutils\tests\test_archive_util.py:180 |
| _warnings | 7 | False | .conda\Lib\distutils\tests\test_check.py:45;.conda\Lib\distutils\tests\test_check.py:54;.conda\Lib\distutils\tests\test_check.py:62 |
| _delayed_completion_id | 7 | False | .conda\Lib\idlelib\idle_test\test_autocomplete.py:117;.conda\Lib\idlelib\idle_test\test_autocomplete.py:122;.conda\Lib\idlelib\idle_test\test_autocomplete.py:146 |
| _name | 7 | False | .conda\Lib\idlelib\idle_test\test_config_key.py:131;.conda\Lib\idlelib\idle_test\test_config_key.py:133;.conda\Lib\tkinter\test\test_ttk\test_extensions.py:19 |
| _close | 7 | False | .conda\Lib\idlelib\idle_test\test_editor.py:31;.conda\Lib\idlelib\idle_test\test_editor.py:114;.conda\Lib\idlelib\idle_test\test_editor.py:225 |
| _tk_type | 7 | False | .conda\Lib\idlelib\idle_test\test_macosx.py:17;.conda\Lib\idlelib\idle_test\test_macosx.py:21;.conda\Lib\idlelib\idle_test\test_macosx.py:62 |
| _default_root | 7 | False | .conda\Lib\tkinter\test\test_tkinter\test_misc.py:808;.conda\Lib\tkinter\test\test_tkinter\test_misc.py:812;.conda\Lib\tkinter\test\test_tkinter\test_misc.py:814 |
| _truncateMessage | 7 | False | .conda\Lib\unittest\test\test_case.py:889;.conda\Lib\unittest\test\test_case.py:901;.conda\Lib\unittest\test\test_case.py:934 |
| _class_cleanups | 7 | False | .conda\Lib\unittest\test\test_runner.py:255;.conda\Lib\unittest\test\test_runner.py:268;.conda\Lib\unittest\test\test_runner.py:380 |
| _verbose | 6 | False | .conda\Lib\doctest.py:844;.conda\Lib\doctest.py:995;.conda\Lib\doctest.py:1227 |
| _dt_optionflags | 6 | False | .conda\Lib\doctest.py:2198;.conda\Lib\doctest.py:2225;.conda\Lib\doctest.py:2324 |
| _dt_checker | 6 | False | .conda\Lib\doctest.py:2199;.conda\Lib\doctest.py:2233;.conda\Lib\doctest.py:2325 |
| _dt_setUp | 6 | False | .conda\Lib\doctest.py:2201;.conda\Lib\doctest.py:2208;.conda\Lib\doctest.py:2209 |
| _dt_tearDown | 6 | False | .conda\Lib\doctest.py:2202;.conda\Lib\doctest.py:2214;.conda\Lib\doctest.py:2215 |
| _length_ | 6 | False | .conda\Lib\ctypes\test\test_arrays.py:167;.conda\Lib\ctypes\test\test_arrays.py:169;.conda\Lib\ctypes\test\test_arrays.py:172 |
| _log | 6 | False | .conda\Lib\distutils\tests\support.py:24;.conda\Lib\distutils\tests\support.py:25;.conda\Lib\distutils\tests\support.py:25 |
