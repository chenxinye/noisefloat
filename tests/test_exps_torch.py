from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from noisefloat.nn import NFloatTensor, clear_kernel_reports
from noisefloat.nn.report import record_kernel

torch = pytest.importorskip("torch", reason="torch not installed")

from exps.common_torch import (
    KernelDigitTracker,
    PlotStyle,
    configure_noisefloat_torch,
    tensor_add,
    tensor_concat,
    tensor_flatten,
    tensor_permute,
    tensor_scale,
)


@pytest.fixture(autouse=True)
def reset_noisefloat():
    configure_noisefloat_torch(n_samples=3, random_state=42)
    clear_kernel_reports()
    yield
    clear_kernel_reports()


def test_tensor_helpers_preserve_nfloat_type():
    base = NFloatTensor(torch.randn(2, 3, dtype=torch.float64))
    other = torch.randn(2, 3, dtype=torch.float64)

    added = tensor_add(base, other)
    scaled = tensor_scale(base, 2.0)
    flattened = tensor_flatten(base, start_dim=1)
    concatenated = tensor_concat((base, other), dim=1)
    permuted = tensor_permute(NFloatTensor(torch.randn(2, 3, 4, dtype=torch.float64)), (1, 0, 2))

    assert isinstance(added, NFloatTensor)
    assert isinstance(scaled, NFloatTensor)
    assert isinstance(flattened, NFloatTensor)
    assert isinstance(concatenated, NFloatTensor)
    assert isinstance(permuted, NFloatTensor)
    assert concatenated.shape == (2, 6)
    assert flattened.shape == (2, 3)
    assert permuted.shape == (3, 2, 4)


def test_tracker_exports_and_captures_metadata(tmp_path):
    record_kernel("demo_kernel", "forward", NFloatTensor(torch.ones(2, 2, dtype=torch.float64)))
    tracker = KernelDigitTracker("demo_task")
    tracker.capture(epoch=1, iteration=2, split="train", global_iteration=5, metadata={"task_seed": 42})
    tracker.export(tmp_path)

    rows = (tmp_path / "kernel_digits.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["epoch"] == 1
    assert row["iteration"] == 2
    assert row["details"]["task_seed"] == 42
    assert row["details"]["detail_level"] == "summary"
    assert "samples_shape" in row["details"]
    assert "samples" not in row["details"]
    assert "representative_value" not in row["details"]
    assert "digits" not in row["details"]
    assert (tmp_path / "kernel_digits.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_tracker_plot_writes_jpg(tmp_path):
    matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
    assert matplotlib is not None

    record_kernel("plot_kernel", "forward", NFloatTensor(torch.ones(2, 2, dtype=torch.float64)))
    tracker = KernelDigitTracker("plot_task")
    tracker.capture(epoch=0, iteration=0, split="train", global_iteration=0)
    record_kernel("plot_kernel", "forward", NFloatTensor(torch.full((2, 2), 2.0, dtype=torch.float64)))
    tracker.capture(epoch=1, iteration=0, split="train", global_iteration=1)

    tracker.plot(tmp_path, style=PlotStyle(font_size=10), metrics=("avg_digits",))
    plots = list(tmp_path.glob("*.jpg"))
    assert plots


def test_diffusion_reconstruction_progression_writes_comparison_figures(tmp_path):
    matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
    assert matplotlib is not None

    from exps.full_cestac_diffusion_fashionmnist_digits_th import (
        QualitativeBatch,
        ReconstructionSnapshot,
        plot_reconstruction_progression,
    )

    images = torch.zeros(2, 1, 4, 4, dtype=torch.float64)
    noisy = torch.full_like(images, 0.2)
    batch = QualitativeBatch(
        images=images,
        noisy=noisy,
        labels=torch.tensor([0, 1]),
        timestep=5,
    )
    snapshots = [
        ReconstructionSnapshot(
            epoch=1,
            reconstructed=torch.full_like(images, 0.15),
            mse_per_sample=torch.full((2,), 0.0225, dtype=torch.float64),
            mean_mse=0.0225,
            mean_psnr=22.5,
        ),
        ReconstructionSnapshot(
            epoch=2,
            reconstructed=torch.full_like(images, 0.05),
            mse_per_sample=torch.full((2,), 0.0025, dtype=torch.float64),
            mean_mse=0.0025,
            mean_psnr=32.0,
        ),
    ]
    args = SimpleNamespace(formats=["png"], qualitative_dpi=72)

    plot_reconstruction_progression(
        batch=batch,
        snapshots=snapshots,
        output_dir=tmp_path,
        args=args,
        style=PlotStyle(font_size=8),
    )

    output_dir = tmp_path / "figures" / "diffusion" / "reconstructions"
    expected = {
        "epoch_reconstruction_progression.png",
        "epoch_absolute_error_progression.png",
        "epoch_reconstruction_change_progression.png",
    }
    assert {path.name for path in output_dir.glob("*.png")} == expected
    assert all((output_dir / name).stat().st_size > 0 for name in expected)
