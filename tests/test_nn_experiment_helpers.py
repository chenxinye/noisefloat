from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from noisefloat.nn import (
    NFloatLinear,
    NFloatReLU,
    NFloatShadowModel,
    NFloatTensor,
    KernelDigitTracker,
    clear_kernel_reports,
    configure_noisefloat_torch,
    get_kernel_reports,
    sync_model_weights,
)


class ReferenceMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 5, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2, dtype=torch.float64),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class NFloatMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            NFloatLinear(4, 5),
            NFloatReLU(),
            NFloatLinear(5, 2),
        )

    def forward(self, inputs):
        return self.layers(inputs)


@pytest.fixture(autouse=True)
def reset_noisefloat():
    configure_noisefloat_torch(n_samples=3, random_state=42)
    clear_kernel_reports()
    yield
    clear_kernel_reports()


def test_package_exports_experiment_helpers():
    assert NFloatShadowModel is not None
    assert KernelDigitTracker is not None
    assert sync_model_weights is not None


def test_shadow_model_auto_syncs_wrapped_pytorch_weights():
    reference = ReferenceMLP()
    nfloat = NFloatMLP()
    shadow = NFloatShadowModel(reference, nfloat)

    with torch.no_grad():
        reference.layers[0].weight.fill_(0.25)
        reference.layers[0].bias.fill_(-0.5)
        reference.layers[2].weight.fill_(0.125)
        reference.layers[2].bias.fill_(0.75)

    output = shadow(NFloatTensor(torch.ones(3, 4, dtype=torch.float64)))

    assert isinstance(output, NFloatTensor)
    assert torch.equal(nfloat.layers[0].inner.weight, reference.layers[0].weight)
    assert torch.equal(nfloat.layers[0].inner.bias, reference.layers[0].bias)
    assert torch.equal(nfloat.layers[2].inner.weight, reference.layers[2].weight)
    assert torch.equal(nfloat.layers[2].inner.bias, reference.layers[2].bias)


def test_shadow_model_capture_runs_and_records_reports():
    reference = ReferenceMLP()
    nfloat = NFloatMLP()
    tracker = KernelDigitTracker("shadow_demo")
    shadow = NFloatShadowModel(reference, nfloat)

    output = shadow.capture(
        tracker,
        NFloatTensor(torch.ones(2, 4, dtype=torch.float64)),
        epoch=0,
        iteration=1,
        split="train",
        global_iteration=1,
        metadata={"backend": "torch"},
    )

    assert isinstance(output, NFloatTensor)
    assert tracker.rows
    assert tracker.rows[0]["details"]["backend"] == "torch"
    assert get_kernel_reports() == []
