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


def test_block_converter_exposes_compute_to_ge() -> None:
    module = Path("vllm_ascend/compilation/minicpmo_causal_conv.py")
    tree = ast.parse(module.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "convert_minicpmo_causal_conv_block"
    )
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }

    assert "ge.MatMulV2" in calls
    assert "ge.LayerNormV4" in calls
    assert "ge.Mish" in calls
    assert "ge.Mul" in calls
    assert "ge.Add" in calls
    assert "ge.ConcatV2" in calls
    assert "custom_op" in calls

    custom_op_names = {
        node.args[0].value
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "custom_op"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert custom_op_names == {"MinicpmoCausalConvPack"}
