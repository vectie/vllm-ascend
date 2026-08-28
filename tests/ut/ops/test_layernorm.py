from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.utils import enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw

enable_custom_op()


@pytest.mark.parametrize("disabled_value", ["0", "false", "off", "OFF"])
@patch("vllm_ascend.ops.layernorm.enable_custom_op", return_value=True)
def test_add_rms_norm_bias_can_be_disabled(mock_enable_custom_op, monkeypatch, disabled_value):
    from vllm_ascend.ops.layernorm import _enable_add_rms_norm_bias

    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS", disabled_value)

    assert not _enable_add_rms_norm_bias()
    mock_enable_custom_op.assert_not_called()


@patch("vllm_ascend.ops.layernorm.enable_custom_op", return_value=True)
def test_add_rms_norm_bias_remains_enabled_by_default(mock_enable_custom_op, monkeypatch):
    from vllm_ascend.ops.layernorm import _enable_add_rms_norm_bias

    monkeypatch.delenv("VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS", raising=False)

    assert _enable_add_rms_norm_bias()
    mock_enable_custom_op.assert_called_once_with()


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


def mock_rms_norm(x, weight, eps):
    return x + 1, None


def mock_add_rms_norm(x, residual, weight, eps):
    return 2 * x, None, 2 * residual


def mock_add_rms_norm_bias(x, residual, weight, bias, eps):
    if bias is None:
        return 2 * x, None, 2 * residual
    else:
        return 2 * x + bias, None, 2 * residual


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]

    with set_current_vllm_config(mock_config):
        yield mock_config


@pytest.mark.skip("Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
@pytest.mark.parametrize("residual", [None, torch.randn(4, 8, dtype=torch.float32)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
@patch("torch.ops._C_ascend.npu_add_rms_norm_bias", side_effect=mock_add_rms_norm_bias)
def test_RMSNorm_forward(
    mock_add_rms_norm_bias, mock_add_rmsnorm, mock_rmsnorm, residual, dummy_tensor, default_vllm_config
):
    layer = RMSNorm(hidden_size=8, eps=1e-05)
    if residual is not None:
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = 2 * dummy_tensor
        expected_out_residual = 2 * residual
        mock_add_rms_norm_bias.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
        assert torch.allclose(out_residual, expected_out_residual)
    else:
        out_x = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = dummy_tensor + 1

        mock_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)


@pytest.mark.skipif(not is_310p_hw(), reason="310P device unittest case.")
@pytest.mark.parametrize("residual", [None, torch.randn(4, 8, dtype=torch.float16)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
def test_RMSNorm_forward_310p(mock_add_rmsnorm, mock_rmsnorm, residual, dummy_tensor, default_vllm_config):
    layer = RMSNorm(hidden_size=8, eps=1e-05)
    if residual is not None:
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = 2 * dummy_tensor
        expected_out_residual = 2 * residual
        mock_add_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
        assert torch.allclose(out_residual, expected_out_residual)
    else:
        out_x = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = dummy_tensor + 1
        mock_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
