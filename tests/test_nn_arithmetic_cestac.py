from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from noisefloat import configure, reset_chopper_cache
from noisefloat.nn import (
    ArithmeticNFloatLinear,
    ArithmeticNFloatModule,
    ArithmeticNFloatMultiheadAttention,
    NFloatShadowModel,
    NFloatTensor,
    clear_kernel_reports,
    get_kernel_reports,
)
from noisefloat.nn.tensor import (
    _as_samples,
    _gemm_perturbation_threshold,
    _nfloat_gemm,
    _nfloat_rms_norm,
)


class CompositeCESTACLayer(torch.nn.Module):
    """Small proxy for softmax(sigmoid(A @ (B - C)) / D)."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(
                [[1.0, -1.0], [0.5, 0.25]],
                dtype=torch.float64,
            )
        )

    def forward(self, x):
        b = x[:, :2]
        c = x[:, 2:]
        mixed = torch.nn.functional.linear(b - c, self.weight)
        activated = torch.sigmoid(mixed)
        return torch.softmax(activated / 3.0, dim=-1)


@pytest.fixture(autouse=True)
def _configure():
    configure(
        backend="torch",
        exp_bits=8,
        sig_bits=23,
        n_samples=3,
        random_state=123,
        trace=False,
        digits_threshold=0.5,
        zero_digits_threshold=0.5,
        data_perturbation_threshold=1e-7,
    )
    reset_chopper_cache()
    clear_kernel_reports()
    yield
    clear_kernel_reports()


def test_arithmetic_module_dispatches_through_torch_primitives():
    model = ArithmeticNFloatModule(
        torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(3),
        ),
        name="TinyArithmetic",
    )
    value = NFloatTensor(torch.randn(2, 4, dtype=torch.float64))

    output = model(value)

    assert isinstance(output, NFloatTensor)
    assert output.samples.shape == (3, 2, 3)
    reports = get_kernel_reports()
    assert [report.kernel_name for report in reports] == ["nfloat/TinyArithmetic"]


def test_arithmetic_prelu_scalar_weight_broadcasts_across_batch_and_features():
    value = NFloatTensor(torch.randn(128, 256, dtype=torch.float64))
    weight = torch.tensor([0.25], dtype=torch.float64)

    output = torch.nn.functional.prelu(value, weight)

    assert isinstance(output, NFloatTensor)
    assert output.samples.shape == (3, 128, 256)
    assert torch.isfinite(output.samples).all()


def test_arithmetic_prelu_channel_weight_broadcasts_over_spatial_dims():
    value = NFloatTensor(torch.randn(4, 3, 5, 5, dtype=torch.float64))
    weight = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    output = torch.nn.functional.prelu(value, weight)

    assert isinstance(output, NFloatTensor)
    assert output.samples.shape == (3, 4, 3, 5, 5)
    assert torch.isfinite(output.samples).all()


def test_arithmetic_rms_norm_dispatches_through_torch_function():
    if not hasattr(torch.nn.functional, "rms_norm"):
        pytest.skip("torch.nn.functional.rms_norm is unavailable in this torch")

    value = NFloatTensor(torch.randn(2, 4, dtype=torch.float64))
    weight = torch.ones(4, dtype=torch.float64)

    output = torch.nn.functional.rms_norm(value, (4,), weight, 1e-6)

    assert isinstance(output, NFloatTensor)
    assert output.samples.shape == (3, 2, 4)
    assert torch.isfinite(output.samples).all()


def test_arithmetic_rms_norm_uses_nfloat_primitives():
    base = torch.randn(2, 4, dtype=torch.float64)
    weight = torch.linspace(0.5, 1.25, steps=4, dtype=torch.float64)
    value = NFloatTensor(base)

    output = _nfloat_rms_norm(value, (4,), weight, 1e-6)

    expected = base / torch.sqrt(base.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = expected * weight
    assert isinstance(output, NFloatTensor)
    assert output.samples.shape == (3, 2, 4)
    assert torch.isfinite(output.samples).all()
    assert torch.allclose(output.value, expected, rtol=1e-4, atol=1e-4)


def test_arithmetic_rms_norm_module_supports_nfloat_tensor():
    if not hasattr(torch.nn, "RMSNorm"):
        pytest.skip("torch.nn.RMSNorm is unavailable in this torch")

    model = ArithmeticNFloatModule(
        torch.nn.RMSNorm(4, eps=1e-6).double(),
        name="RMSNorm",
    )
    value = NFloatTensor(torch.randn(2, 4, dtype=torch.float64))

    output = model(value)

    assert isinstance(output, NFloatTensor)
    assert output.samples.shape == (3, 2, 4)
    assert torch.isfinite(output.samples).all()
    reports = get_kernel_reports()
    assert [report.kernel_name for report in reports] == ["nfloat/RMSNorm"]


def test_plain_tensor_samples_can_use_broadcast_view_without_rounding():
    weight = torch.randn(5, 3, dtype=torch.float64)
    samples = _as_samples(weight, 3, round_plain=False)

    assert samples.shape == (3, 5, 3)
    assert samples.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    assert torch.allclose(samples[0], weight)
    assert torch.allclose(samples[1], weight)


def test_arithmetic_linear_rounds_during_inner_product():
    layer = ArithmeticNFloatLinear(2, 1, bias=False)
    with torch.no_grad():
        layer.inner.weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float64))
    value = NFloatTensor(torch.tensor([[1.0e8 + 1.0, 1.0e8]], dtype=torch.float64))

    output = layer(value)

    assert isinstance(output, NFloatTensor)
    # Full CESTAC neural GEMM uses data perturbation when the stochastic
    # operands are indistinguishable, so this cancellation probe should not
    # collapse to three identical samples.
    assert torch.unique(output.samples.reshape(-1)).numel() > 1


def test_gemm_data_perturbation_threshold_is_configurable():
    configure(data_perturbation_threshold=1e-3)

    assert _gemm_perturbation_threshold() == pytest.approx(1e-3)

    layer = ArithmeticNFloatLinear(2, 1, bias=False)
    with torch.no_grad():
        layer.inner.weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float64))
    identical = torch.tensor(
        [
            [[1.0e8, 1.0e8]],
            [[1.0e8, 1.0e8]],
            [[1.0e8, 1.0e8]],
        ],
        dtype=torch.float64,
    )

    output = layer(NFloatTensor.from_samples(identical))

    assert torch.unique(output.samples.reshape(-1)).numel() > 1


def test_gemm_data_perturbation_can_be_disabled():
    configure(exp_bits=11, sig_bits=52, data_perturbation_threshold=0.0)

    assert _gemm_perturbation_threshold() == 0.0

    layer = ArithmeticNFloatLinear(2, 1, bias=False)
    with torch.no_grad():
        layer.inner.weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float64))
    identical = torch.tensor(
        [
            [[1.0e8, 1.0e8]],
            [[1.0e8, 1.0e8]],
            [[1.0e8, 1.0e8]],
        ],
        dtype=torch.float64,
    )

    output = layer(NFloatTensor.from_samples(identical))

    assert output.samples.detach().abs().max().item() < 1e-6


def test_gemm_data_perturbation_only_applies_to_stochastic_operands():
    configure(exp_bits=11, sig_bits=52, data_perturbation_threshold=1e-3)
    lhs = torch.tensor(
        [
            [[1.0, 2.0]],
            [[1.0, 2.0]],
            [[1.0, 2.0]],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor(
        [
            [[3.0], [4.0]],
            [[3.0], [4.0]],
            [[3.0], [4.0]],
        ],
        dtype=torch.float64,
    )

    deterministic_kernel = _nfloat_gemm(
        lhs,
        rhs,
        perturb_lhs=False,
        perturb_rhs=False,
    )
    stochastic_input_kernel = _nfloat_gemm(
        lhs,
        rhs,
        perturb_lhs=True,
        perturb_rhs=False,
    )
    deterministic_spread = (
        deterministic_kernel.samples.max() - deterministic_kernel.samples.min()
    ).item()
    stochastic_spread = (
        stochastic_input_kernel.samples.max() - stochastic_input_kernel.samples.min()
    ).item()

    assert deterministic_spread < 1e-12
    assert stochastic_spread > 1e-4


def test_gemm_data_perturbation_default_is_precision_aware():
    configure(sig_bits=52, data_perturbation_threshold=None)

    assert _gemm_perturbation_threshold() == pytest.approx(2.0**-52)

    configure(sig_bits=23, data_perturbation_threshold=None)

    assert _gemm_perturbation_threshold() == pytest.approx(2.0**-23)


def test_composite_layer_propagates_cestac_except_trusted_gemm():
    configure(data_perturbation_threshold=1e-3)
    model = ArithmeticNFloatModule(CompositeCESTACLayer(), name="CompositeCESTAC")
    samples = torch.tensor(
        [
            [[1.0e8 + 1.0, 1.0e8, 1.0e8, 1.0e8 - 1.0]],
            [[1.0e8 + 1.0, 1.0e8, 1.0e8, 1.0e8 - 1.0]],
            [[1.0e8 + 1.0, 1.0e8, 1.0e8, 1.0e8 - 1.0]],
        ],
        dtype=torch.float64,
    )

    output = model(NFloatTensor.from_samples(samples))

    assert isinstance(output, NFloatTensor)
    assert output.samples.shape == (3, 1, 2)
    assert torch.isfinite(output.samples).all()
    assert torch.unique(output.samples.detach().reshape(-1)).numel() > 1
    reports = get_kernel_reports()
    assert [report.kernel_name for report in reports] == ["nfloat/CompositeCESTAC"]


def test_shadow_model_syncs_arithmetic_wrapper_weights():
    reference = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Sigmoid()).double()
    nfloat_model = torch.nn.Sequential(
        ArithmeticNFloatLinear(3, 2),
        ArithmeticNFloatModule(torch.nn.Sigmoid(), name="Sigmoid"),
    )
    shadow = NFloatShadowModel(reference, nfloat_model, backend="torch")
    value = NFloatTensor(torch.randn(4, 3, dtype=torch.float64))

    output = shadow(value)

    assert isinstance(output, NFloatTensor)
    assert output.samples.shape == (3, 4, 2)
    assert torch.allclose(
        nfloat_model[0].inner.weight,
        reference[0].weight,
    )


def test_arithmetic_multihead_attention_supports_masks():
    layer = ArithmeticNFloatMultiheadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
        batch_first=True,
    )
    value = NFloatTensor(torch.randn(2, 4, 8, dtype=torch.float64))
    attn_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    key_padding_mask = torch.tensor(
        [[False, False, False, True], [False, False, True, True]]
    )

    output, weights = layer(
        value,
        value,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )

    assert isinstance(output, NFloatTensor)
    assert isinstance(weights, NFloatTensor)
    assert output.samples.shape == (3, 2, 4, 8)
    assert weights.samples.shape == (3, 2, 4, 4)
    assert torch.isfinite(output.samples).all()
