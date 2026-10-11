"""CUDA extension parity checks on synthetic inputs; no project phenotypes."""

import copy

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device is not visible in this session"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_causal_convolution_forward_and_backward(dtype):
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref

    torch.manual_seed(17)
    inputs = [
        torch.randn(2, 8, 64, device="cuda", dtype=dtype, requires_grad=True),
        torch.randn(8, 4, device="cuda", dtype=dtype, requires_grad=True),
        torch.randn(8, device="cuda", dtype=dtype, requires_grad=True),
    ]
    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    actual = causal_conv1d_fn(*inputs, activation="silu")
    expected = causal_conv1d_ref(*reference_inputs, activation="silu")
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    gradient = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, inputs, gradient)
    reference_gradients = torch.autograd.grad(expected, reference_inputs, gradient)
    for actual_gradient, expected_gradient in zip(gradients, reference_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=tolerance * 4, rtol=tolerance * 4
        )
    torch.cuda.synchronize()


def test_selective_scan_forward_and_backward_against_reference():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

    torch.manual_seed(19)
    inputs = [
        torch.randn(2, 8, 32, device="cuda"),
        torch.randn(2, 8, 32, device="cuda"),
        -torch.rand(8, 4, device="cuda"),
        torch.randn(2, 4, 32, device="cuda"),
        torch.randn(2, 4, 32, device="cuda"),
        torch.randn(8, device="cuda"),
    ]
    inputs = [x.requires_grad_() for x in inputs]
    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    actual = selective_scan_fn(*inputs, delta_softplus=True)
    expected = selective_scan_ref(*reference_inputs, delta_softplus=True)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    gradient = torch.randn_like(actual)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual, inputs, gradient),
        torch.autograd.grad(expected, reference_inputs, gradient),
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-3, rtol=1e-3)
    torch.cuda.synchronize()


def test_teacher_fused_residual_rmsnorm_against_reference():
    from mamba_ssm.ops.triton.layer_norm import rms_norm_fn

    torch.manual_seed(21)
    inputs = [
        torch.randn(2, 600, 768, device="cuda", requires_grad=True),
        torch.randn(2, 600, 768, device="cuda", requires_grad=True),
        torch.randn(768, device="cuda", requires_grad=True),
    ]
    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    actual, residual = rms_norm_fn(
        inputs[0], inputs[2], None, residual=inputs[1], eps=1e-5,
        prenorm=True, residual_in_fp32=True,
    )
    reference_residual = reference_inputs[0] + reference_inputs[1]
    expected = reference_residual * torch.rsqrt(
        reference_residual.square().mean(dim=-1, keepdim=True) + 1e-5
    ) * reference_inputs[2]
    torch.testing.assert_close(residual, reference_residual, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    gradient = torch.randn_like(actual)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual, inputs, gradient),
        torch.autograd.grad(expected, reference_inputs, gradient),
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-3, rtol=1e-3)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mamba2_fused_and_unfused_paths(dtype):
    from mamba_ssm import Mamba2

    torch.manual_seed(23)
    # Match the teacher's width: its projected channel stride is divisible by
    # eight, as required by causal-conv1d's channel-last kernel.
    fused = Mamba2(
        d_model=768, d_state=64, headdim=64, chunk_size=256,
        use_mem_eff_path=True, device="cuda", dtype=dtype,
    ).eval()
    unfused = copy.deepcopy(fused)
    unfused.use_mem_eff_path = False
    sequence = torch.randn(2, 600, 768, device="cuda", dtype=dtype, requires_grad=True)
    reference_sequence = sequence.detach().clone().requires_grad_()
    actual = fused(sequence)
    expected = unfused(reference_sequence)
    tolerance = 3e-3 if dtype == torch.float16 else 1e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    gradient = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad(actual, sequence, gradient)[0]
    expected_gradient = torch.autograd.grad(expected, reference_sequence, gradient)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=tolerance * 4, rtol=tolerance * 4
    )
    torch.cuda.synchronize()
