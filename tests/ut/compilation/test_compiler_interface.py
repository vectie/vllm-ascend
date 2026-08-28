import sys
from types import SimpleNamespace

from vllm_ascend.compilation.compiler_interface import (
    AscendCompiler,
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


def test_add_rms_norm_bias_capability_partitions_compile_cache(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS", raising=False)
    assert AscendCompiler._add_rms_norm_bias_cache_key()

    for disabled_value in ("0", "false", "off", "OFF"):
        monkeypatch.setenv("VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS", disabled_value)
        assert not AscendCompiler._add_rms_norm_bias_cache_key()

    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS", "1")
    assert AscendCompiler._add_rms_norm_bias_cache_key()
