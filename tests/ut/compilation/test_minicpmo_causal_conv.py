# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def test_torchair_converter_module_stays_lazy() -> None:
    module = Path("vllm_ascend/compilation/minicpmo_causal_conv.py")
    tree = ast.parse(module.read_text())
    top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported_modules = {
        alias.name for node in top_level_imports for alias in node.names
    }
    assert "torchair" not in imported_modules
    assert "torch_npu.dynamo" not in imported_modules
