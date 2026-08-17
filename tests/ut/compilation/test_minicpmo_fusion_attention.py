# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def test_fusion_attention_v3_converter_stays_lazy_and_ge_visible() -> None:
    module = Path("vllm_ascend/compilation/minicpmo_fusion_attention.py")
    tree = ast.parse(module.read_text())
    top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported_modules = {alias.name for node in top_level_imports for alias in node.names}
    assert "torchair" not in imported_modules
    assert "torch_npu.dynamo" not in imported_modules

    converter = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "convert_minicpmo_fusion_attention_v3"
    )
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(converter)
        if isinstance(node, ast.Call)
    }
    assert "ge.FlashAttentionScore" in calls
    assert "Const" in calls
