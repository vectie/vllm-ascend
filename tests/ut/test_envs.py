#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import inspect
import os

import vllm.envs as envs_vllm
import vllm_ascend.envs as envs_ascend
from tests.ut.base import TestBase
from vllm_ascend.platform import _register_graph_semantic_env_factors


class TestEnvVariables(TestBase):
    def setUp(self):
        self.env_vars = list(envs_ascend.env_variables.keys())

    def test_env_vars_behavior(self):
        for var_name in self.env_vars:
            with self.subTest(var=var_name):
                original_val = os.environ.get(var_name)
                var_handler = envs_ascend.env_variables[var_name]

                try:
                    if var_name in os.environ:
                        del os.environ[var_name]
                    self.assertEqual(getattr(envs_ascend, var_name), var_handler())

                    handler_source = inspect.getsource(var_handler)
                    if "int(" in handler_source:
                        test_vals = ["123", "456"]
                    elif "bool(int(" in handler_source:
                        test_vals = ["0", "1"]
                    else:
                        test_vals = [f"test_{var_name}", f"custom_{var_name}"]

                    for test_val in test_vals:
                        os.environ[var_name] = test_val
                        self.assertEqual(getattr(envs_ascend, var_name), var_handler())

                finally:
                    if original_val is None:
                        os.environ.pop(var_name, None)
                    else:
                        os.environ[var_name] = original_val

    def test_dir_and_getattr(self):
        self.assertEqual(sorted(envs_ascend.__dir__()), sorted(self.env_vars))
        for var_name in self.env_vars:
            with self.subTest(var=var_name):
                getattr(envs_ascend, var_name)

    def test_graph_semantic_env_is_registered_for_vllm_aot_hash(self):
        name = "VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS"
        original_getter = envs_vllm.environment_variables.pop(name, None)
        original_value = os.environ.get(name)
        try:
            _register_graph_semantic_env_factors()
            os.environ[name] = "0"
            self.assertFalse(envs_vllm.environment_variables[name]())
            os.environ[name] = "1"
            self.assertTrue(envs_vllm.environment_variables[name]())
        finally:
            if original_getter is None:
                envs_vllm.environment_variables.pop(name, None)
            else:
                envs_vllm.environment_variables[name] = original_getter
            if original_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original_value
