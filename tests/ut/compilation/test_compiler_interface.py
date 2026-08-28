import sys
from types import SimpleNamespace

from vllm_ascend.compilation.compiler_interface import (
    _ensure_npugraph_weight_quant_view_option,
)


class _FakeOptionValue:
    def __init__(self, default, optional=None):
        self.default = default
        self.optional = optional


def test_repairs_missing_npugraph_weight_quant_view_option(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "npugraph_ex.configs._option_base",
        SimpleNamespace(OptionValue=_FakeOptionValue),
    )
    experimental_config = SimpleNamespace(_fixed_attrs=[])
    config = SimpleNamespace(experimental_config=experimental_config)

    _ensure_npugraph_weight_quant_view_option(config)
    _ensure_npugraph_weight_quant_view_option(config)

    option = experimental_config.enable_view_optimize
    assert option.default is True
    assert option.optional == [True, False]
    assert experimental_config._fixed_attrs == ["enable_view_optimize"]
