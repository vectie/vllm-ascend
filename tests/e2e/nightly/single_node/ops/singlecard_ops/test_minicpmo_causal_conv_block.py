# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


def _reference(
    hidden: torch.Tensor,
    conv_input: torch.Tensor,
    cnn_cache: torch.Tensor,
    gate_conv: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache1, cache2 = cnn_cache.split((512, 512), dim=1)
    first = torch.cat((cache1, conv_input.transpose(1, 2)), dim=2)
    new_cache1 = first[:, :, -2:]
    convolution = F.conv1d(first, conv1_weight.reshape(512, 3, 512).permute(0, 2, 1), conv1_bias)
    convolution = F.layer_norm(convolution.transpose(1, 2), (512,), norm_weight, norm_bias, 1e-5)
    convolution = F.mish(convolution)
    second = torch.cat((cache2, convolution.transpose(1, 2)), dim=2)
    new_cache2 = second[:, :, -2:]
    convolution = F.conv1d(second, conv2_weight.reshape(512, 3, 512).permute(0, 2, 1), conv2_bias)
    return hidden + gate_conv * convolution.transpose(1, 2), torch.cat((new_cache1, new_cache2), dim=1)


def test_minicpmo_causal_conv_block_fp32() -> None:
    torch.manual_seed(19)
    device = torch.device("npu")
    hidden = torch.randn(2, 50, 512, dtype=torch.float32, device=device)
    conv_input = torch.randn_like(hidden)
    cnn_cache = torch.randn(2, 1024, 2, dtype=torch.float32, device=device)
    gate_conv = torch.randn(2, 1, 512, dtype=torch.float32, device=device)
    conv1_weight = torch.randn(512, 1536, dtype=torch.float32, device=device) * 0.01
    conv1_bias = torch.randn(512, dtype=torch.float32, device=device) * 0.01
    norm_weight = torch.randn(512, dtype=torch.float32, device=device)
    norm_bias = torch.randn(512, dtype=torch.float32, device=device)
    conv2_weight = torch.randn(512, 1536, dtype=torch.float32, device=device) * 0.01
    conv2_bias = torch.randn(512, dtype=torch.float32, device=device) * 0.01

    expected_hidden, expected_cache = _reference(
        hidden,
        conv_input,
        cnn_cache,
        gate_conv,
        conv1_weight,
        conv1_bias,
        norm_weight,
        norm_bias,
        conv2_weight,
        conv2_bias,
    )
    actual_hidden, actual_cache = torch.ops._C_ascend.npu_minicpmo_causal_conv_block(
        hidden,
        conv_input,
        cnn_cache,
        gate_conv,
        conv1_weight,
        conv1_bias,
        norm_weight,
        norm_bias,
        conv2_weight,
        conv2_bias,
    )

    # The MIX kernel uses the 910C Vector-core transcendental instructions for
    # Mish. Their bounded FP32 approximation is much faster than dispatching a
    # separate high-level activation kernel, but is not bit-identical to
    # torch_npu's Mish implementation. Keep both the worst-case and aggregate
    # error bounded so a broad accuracy regression cannot hide behind atol.
    hidden_error = (actual_hidden - expected_hidden).abs()
    cache_error = (actual_cache - expected_cache).abs()
    assert hidden_error.max().item() <= 5e-3
    assert hidden_error.mean().item() <= 3e-4
    assert cache_error.max().item() <= 1.2e-2
    assert cache_error.mean().item() <= 4e-4
