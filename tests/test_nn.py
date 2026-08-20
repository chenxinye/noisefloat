"""
Tests for noisefloat.nn – NFloat wrappers for deep learning kernels.

Requires PyTorch.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

import noisefloat as nf
from noisefloat import configure
from noisefloat.nn import (
    NFloatTensor,
    NFloatModule,
    NFloatIterationTracker,
    NFloatOperator,
    wrap,
    wrap_model,
    nfloat_analysis,
    nfloat_operator,
    # Layers
    NFloatLinear,
    NFloatConv1d,
    NFloatConv2d,
    NFloatConv3d,
    NFloatConvTranspose1d,
    NFloatConvTranspose2d,
    NFloatConvTranspose3d,
    NFloatBatchNorm1d,
    NFloatBatchNorm2d,
    NFloatLayerNorm,
    NFloatGroupNorm,
    NFloatInstanceNorm1d,
    NFloatInstanceNorm2d,
    NFloatInstanceNorm3d,
    # Pooling
    NFloatMaxPool1d,
    NFloatMaxPool2d,
    NFloatMaxPool3d,
    NFloatAvgPool1d,
    NFloatAvgPool2d,
    NFloatAvgPool3d,
    NFloatAdaptiveAvgPool1d,
    NFloatAdaptiveAvgPool2d,
    NFloatAdaptiveAvgPool3d,
    NFloatAdaptiveMaxPool1d,
    NFloatAdaptiveMaxPool2d,
    NFloatAdaptiveMaxPool3d,
    # Activations
    NFloatReLU,
    NFloatGELU,
    NFloatSigmoid,
    NFloatTanh,
    NFloatSoftmax,
    NFloatLogSoftmax,
    NFloatLeakyReLU,
    NFloatELU,
    NFloatSiLU,
    NFloatMish,
    NFloatPReLU,
    NFloatHardswish,
    NFloatHardtanh,
    # Dropout
    NFloatDropout,
    NFloatDropout2d,
    NFloatDropout3d,
    # Embedding
    NFloatEmbedding,
    NFloatEmbeddingBag,
    # Padding
    NFloatZeroPad2d,
    NFloatReflectionPad2d,
    NFloatReplicationPad2d,
    # Recurrent
    NFloatRNN,
    NFloatLSTM,
    NFloatGRU,
    # Attention
    NFloatMultiheadAttention,
    # Transformer
    NFloatTransformerEncoderLayer,
    NFloatTransformerDecoderLayer,
    # Loss
    NFloatCrossEntropyLoss,
    NFloatMSELoss,
    NFloatL1Loss,
    NFloatNLLLoss,
    NFloatBCELoss,
    NFloatBCEWithLogitsLoss,
    NFloatSmoothL1Loss,
    NFloatHuberLoss,
    NFloatKLDivLoss,
    NFloatCTCLoss,
    # Optimizer
    NFloatOptimizer,
    # Reporting
    get_kernel_reports,
    clear_kernel_reports,
    print_kernel_report,
    summary,
)


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def reset():
    """Reset noisefloat config and kernel reports before each test."""
    configure(
        backend="numpy",
        exp_bits=8,
        sig_bits=23,
        n_samples=3,
        random_state=42,
        confidence=0.95,
        trace=False,
        digits_threshold=0.5,
        zero_digits_threshold=0.5,
        kernel_report_detail="summary",
    )
    from noisefloat.core import _chopper
    _chopper._config_hash = ()
    clear_kernel_reports()
    yield


# --------------------------------------------------------------------------- #
#  1. NFloatTensor                                                             #
# --------------------------------------------------------------------------- #

class TestNFloatTensor:
    def test_creation_from_scalar(self):
        ct = NFloatTensor(3.14)
        assert ct.n_samples == 3
        assert ct.shape == torch.Size([])
        assert abs(ct.mean - 3.14) < 1e-3

    def test_creation_from_tensor(self):
        t = torch.randn(4, 5)
        ct = NFloatTensor(t)
        assert ct.n_samples == 3
        assert ct.shape == (4, 5)
        assert ct.samples.shape == (3, 4, 5)

    def test_custom_n_samples(self):
        ct = NFloatTensor(torch.ones(3), n_samples=7)
        assert ct.n_samples == 7

    def test_from_samples(self):
        s = torch.randn(5, 2, 3)
        ct = NFloatTensor.from_samples(s)
        assert ct.n_samples == 5
        assert ct.shape == (2, 3)

    def test_representative_value_is_sample_mean_tensor(self):
        samples = torch.tensor(
            [[[1.0, 3.0]], [[2.0, 5.0]], [[3.0, 7.0]]],
            dtype=torch.float64,
        )
        ct = NFloatTensor.from_samples(samples)
        expected = torch.tensor([[2.0, 5.0]], dtype=torch.float64)
        torch.testing.assert_close(ct.value, expected)
        torch.testing.assert_close(ct.to_tensor(), expected)

    def test_significant_digits(self):
        ct = NFloatTensor(torch.tensor(3.14))
        d = ct.digits
        assert np.isscalar(d) or d.shape == ()
        assert d >= 0

    def test_avg_digits(self):
        ct = NFloatTensor(torch.randn(10))
        avg = ct.avg_digits()
        assert isinstance(avg, float)
        assert avg >= 0

    def test_min_digits(self):
        ct = NFloatTensor(torch.randn(10))
        m = ct.min_digits()
        assert isinstance(m, float)
        assert m >= 0

    def test_is_stable(self):
        ct = NFloatTensor(torch.tensor(1.0))
        assert isinstance(ct.is_stable(), bool)

    def test_repr(self):
        ct = NFloatTensor(torch.randn(3, 4))
        r = repr(ct)
        assert "NFloatTensor" in r
        assert "avg_digits" in r

    def test_std(self):
        ct = NFloatTensor(torch.tensor(2.0))
        assert ct.std >= 0

    def test_ndim(self):
        ct = NFloatTensor(torch.randn(3, 4))
        assert ct.ndim == 2

    def test_len(self):
        ct = NFloatTensor(torch.randn(5, 3))
        assert len(ct) == 5


# --------------------------------------------------------------------------- #
#  2. NFloatModule (base wrapper)                                               #
# --------------------------------------------------------------------------- #

class TestNFloatModule:
    def test_wrap_relu(self):
        m = NFloatModule(torch.nn.ReLU(), name="relu_test")
        ct = NFloatTensor(torch.randn(4, 8))
        out = m(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 8)

    def test_passthrough_plain_tensor(self):
        m = NFloatModule(torch.nn.ReLU())
        t = torch.randn(3, 5)
        out = m(t)
        assert isinstance(out, torch.Tensor)
        assert not isinstance(out, NFloatTensor)

    def test_wrap_function(self):
        linear = torch.nn.Linear(10, 5)
        wrapped = wrap(linear, name="my_linear")
        assert isinstance(wrapped, NFloatModule)
        assert wrapped.kernel_name == "my_linear"

    def test_report_recorded(self):
        m = NFloatModule(torch.nn.ReLU(), name="relu_report")
        ct = NFloatTensor(torch.randn(4, 8))
        _ = m(ct)
        reports = get_kernel_reports()
        assert len(reports) >= 1
        assert reports[-1].kernel_name == "nfloat/relu_report"
        assert reports[-1].phase == "forward"


# --------------------------------------------------------------------------- #
#  3. Layers                                                                   #
# --------------------------------------------------------------------------- #

class TestLayers:
    def test_linear(self):
        layer = NFloatLinear(10, 5)
        ct = NFloatTensor(torch.randn(4, 10))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 5)

    def test_conv1d(self):
        layer = NFloatConv1d(3, 8, kernel_size=3, padding=1)
        ct = NFloatTensor(torch.randn(2, 3, 16))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape[0] == 2  # batch
        assert out.shape[1] == 8  # channels

    def test_conv2d(self):
        layer = NFloatConv2d(3, 16, kernel_size=3, padding=1)
        ct = NFloatTensor(torch.randn(2, 3, 8, 8))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape[:2] == (2, 16)

    def test_batchnorm1d(self):
        layer = NFloatBatchNorm1d(10)
        ct = NFloatTensor(torch.randn(4, 10))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 10)

    def test_batchnorm2d(self):
        layer = NFloatBatchNorm2d(3)
        ct = NFloatTensor(torch.randn(2, 3, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 4, 4)

    def test_layernorm(self):
        layer = NFloatLayerNorm(10)
        ct = NFloatTensor(torch.randn(4, 10))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 10)


# --------------------------------------------------------------------------- #
#  4. Activations                                                              #
# --------------------------------------------------------------------------- #

class TestActivations:
    def test_relu(self):
        act = NFloatReLU()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        # ReLU output should be >= 0
        assert np.all(out.mean >= -1e-7)

    def test_gelu(self):
        act = NFloatGELU()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 8)

    def test_sigmoid(self):
        act = NFloatSigmoid()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        assert np.all(out.mean >= -1e-7) and np.all(out.mean <= 1.0 + 1e-7)

    def test_tanh(self):
        act = NFloatTanh()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        assert np.all(out.mean >= -1.0 - 1e-7) and np.all(out.mean <= 1.0 + 1e-7)

    def test_softmax(self):
        act = NFloatSoftmax(dim=-1)
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        # Softmax outputs should sum to ~1 along dim
        row_sums = out.mean.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_logsoftmax(self):
        act = NFloatLogSoftmax(dim=-1)
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        # LogSoftmax outputs should be <= 0
        assert np.all(out.mean <= 1e-6)


# --------------------------------------------------------------------------- #
#  5. Recurrent layers                                                         #
# --------------------------------------------------------------------------- #

class TestRNN:
    def test_rnn_basic(self):
        rnn = NFloatRNN(input_size=10, hidden_size=20, batch_first=True)
        ct = NFloatTensor(torch.randn(2, 5, 10))  # batch=2, seq=5, features=10
        out, h_n = rnn(ct)
        assert isinstance(out, NFloatTensor)
        assert isinstance(h_n, NFloatTensor)
        assert out.shape == (2, 5, 20)

    def test_lstm_basic(self):
        lstm = NFloatLSTM(input_size=10, hidden_size=20, batch_first=True)
        ct = NFloatTensor(torch.randn(2, 5, 10))
        out, (h_n, c_n) = lstm(ct)
        assert isinstance(out, NFloatTensor)
        assert isinstance(h_n, NFloatTensor)
        assert isinstance(c_n, NFloatTensor)
        assert out.shape == (2, 5, 20)

    def test_gru_basic(self):
        gru = NFloatGRU(input_size=10, hidden_size=20, batch_first=True)
        ct = NFloatTensor(torch.randn(2, 5, 10))
        out, h_n = gru(ct)
        assert isinstance(out, NFloatTensor)
        assert isinstance(h_n, NFloatTensor)
        assert out.shape == (2, 5, 20)

    def test_rnn_passthrough(self):
        rnn = NFloatRNN(input_size=10, hidden_size=20, batch_first=True)
        t = torch.randn(2, 5, 10, dtype=torch.float64)
        out, h_n = rnn(t)
        assert isinstance(out, torch.Tensor)
        assert not isinstance(out, NFloatTensor)

    def test_lstm_with_hidden(self):
        lstm = NFloatLSTM(input_size=5, hidden_size=8, batch_first=True)
        ct = NFloatTensor(torch.randn(2, 3, 5))
        h0 = NFloatTensor(torch.zeros(1, 2, 8))
        c0 = NFloatTensor(torch.zeros(1, 2, 8))
        out, (h_n, c_n) = lstm(ct, (h0, c0))
        assert isinstance(out, NFloatTensor)


# --------------------------------------------------------------------------- #
#  6. Attention                                                                #
# --------------------------------------------------------------------------- #

class TestAttention:
    def test_multihead_attention(self):
        attn = NFloatMultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        q = NFloatTensor(torch.randn(2, 5, 16))
        k = NFloatTensor(torch.randn(2, 5, 16))
        v = NFloatTensor(torch.randn(2, 5, 16))
        out, weights = attn(q, k, v)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 5, 16)

    def test_attention_passthrough(self):
        attn = NFloatMultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        q = torch.randn(2, 3, 8, dtype=torch.float64)
        out, w = attn(q, q, q)
        assert isinstance(out, torch.Tensor)


# --------------------------------------------------------------------------- #
#  7. Loss functions                                                           #
# --------------------------------------------------------------------------- #

class TestLoss:
    def test_cross_entropy(self):
        loss_fn = NFloatCrossEntropyLoss()
        ct = NFloatTensor(torch.randn(4, 10))
        target = torch.randint(0, 10, (4,))
        loss = loss_fn(ct, target)
        assert isinstance(loss, NFloatTensor)
        assert loss.mean > 0

    def test_mse_loss(self):
        loss_fn = NFloatMSELoss()
        ct = NFloatTensor(torch.randn(4, 5))
        target = torch.randn(4, 5, dtype=torch.float64)
        loss = loss_fn(ct, target)
        assert isinstance(loss, NFloatTensor)
        assert loss.mean >= 0

    def test_l1_loss(self):
        loss_fn = NFloatL1Loss()
        ct = NFloatTensor(torch.randn(4, 5))
        target = torch.randn(4, 5, dtype=torch.float64)
        loss = loss_fn(ct, target)
        assert isinstance(loss, NFloatTensor)
        assert loss.mean >= 0

    def test_nll_loss(self):
        loss_fn = NFloatNLLLoss()
        log_probs = NFloatTensor(torch.nn.functional.log_softmax(torch.randn(4, 10), dim=-1))
        target = torch.randint(0, 10, (4,))
        loss = loss_fn(log_probs, target)
        assert isinstance(loss, NFloatTensor)

    def test_loss_passthrough(self):
        loss_fn = NFloatMSELoss()
        x = torch.randn(4, 5, dtype=torch.float64)
        y = torch.randn(4, 5, dtype=torch.float64)
        loss = loss_fn(x, y)
        assert isinstance(loss, torch.Tensor)
        assert not isinstance(loss, NFloatTensor)


# --------------------------------------------------------------------------- #
#  8. Optimizer                                                                #
# --------------------------------------------------------------------------- #

class TestOptimizer:
    def test_optimizer_step(self):
        model = torch.nn.Linear(10, 2).double()
        opt = NFloatOptimizer(torch.optim.SGD(model.parameters(), lr=0.01))
        x = torch.randn(4, 10, dtype=torch.float64)
        y = model(x).sum()
        y.backward()
        opt.step()
        reports = get_kernel_reports()
        # Should have recorded gradient precision for weight and bias
        grad_reports = [r for r in reports if r.phase == "gradient"]
        assert len(grad_reports) >= 1

    def test_optimizer_zero_grad(self):
        model = torch.nn.Linear(10, 2).double()
        opt = NFloatOptimizer(torch.optim.Adam(model.parameters()))
        opt.zero_grad()
        for p in model.parameters():
            assert p.grad is None or (p.grad == 0).all()

    def test_optimizer_param_groups(self):
        model = torch.nn.Linear(10, 2).double()
        opt = NFloatOptimizer(torch.optim.SGD(model.parameters(), lr=0.01))
        assert len(opt.param_groups) >= 1


# --------------------------------------------------------------------------- #
#  9. Reporting                                                                #
# --------------------------------------------------------------------------- #

class TestReporting:
    def test_kernel_reports_recorded(self):
        linear = NFloatLinear(8, 4)
        relu = NFloatReLU()
        ct = NFloatTensor(torch.randn(2, 8))
        y = relu(linear(ct))
        reports = get_kernel_reports()
        assert len(reports) == 2  # Linear + ReLU
        assert reports[0].kernel_name == "nfloat/Linear"
        assert reports[1].kernel_name == "nfloat/ReLU"

    def test_sequential_reports_each_nfloat_layer_with_summary_details(self):
        configure(backend="torch", n_samples=3, random_state=42)
        model = torch.nn.Sequential(
            NFloatLinear(10, 32),
            NFloatReLU(),
            NFloatLinear(32, 2),
        )
        x = NFloatTensor(torch.randn(8, 10, dtype=torch.float64))

        out = model(x)

        assert isinstance(out, NFloatTensor)
        assert out.samples.shape == (3, 8, 2)
        torch.testing.assert_close(out.value, out.samples.mean(dim=0))
        reports = get_kernel_reports()
        assert [report.kernel_name for report in reports] == [
            "nfloat/Linear",
            "nfloat/ReLU",
            "nfloat/Linear",
        ]
        last = reports[-1]
        assert last.details["backend"] == "torch"
        assert last.details["n_samples"] == 3
        assert last.details["detail_level"] == "summary"
        assert last.details["sample_shape"] == (8, 2)
        assert last.details["samples_shape"] == (3, 8, 2)
        assert last.details["representative_shape"] == (8, 2)
        assert last.details["digits_shape"] == (8, 2)
        assert "samples" not in last.details
        assert "representative_value" not in last.details
        assert "digits" not in last.details

    def test_full_kernel_report_details_are_opt_in(self):
        configure(
            backend="torch",
            n_samples=3,
            random_state=42,
            kernel_report_detail="full",
        )
        model = torch.nn.Sequential(
            NFloatLinear(10, 32),
            NFloatReLU(),
            NFloatLinear(32, 2),
        )
        x = NFloatTensor(torch.randn(8, 10, dtype=torch.float64))

        out = model(x)

        assert isinstance(out, NFloatTensor)
        reports = get_kernel_reports()
        last = reports[-1]
        assert last.details["backend"] == "torch"
        assert last.details["n_samples"] == 3
        assert last.details["detail_level"] == "full"
        assert last.details["samples"].shape == (3, 8, 2)
        assert last.details["representative_value"].shape == (8, 2)
        assert last.details["digits"].shape == (8, 2)

    def test_clear_reports(self):
        linear = NFloatLinear(8, 4)
        _ = linear(NFloatTensor(torch.randn(2, 8)))
        assert len(get_kernel_reports()) >= 1
        clear_kernel_reports()
        assert len(get_kernel_reports()) == 0

    def test_print_report(self, capsys):
        clear_kernel_reports()
        print_kernel_report()
        captured = capsys.readouterr()
        assert "No kernel reports" in captured.out

        linear = NFloatLinear(8, 4)
        _ = linear(NFloatTensor(torch.randn(2, 8)))
        print_kernel_report()
        captured = capsys.readouterr()
        assert "Linear" in captured.out

    def test_summary(self):
        linear = NFloatLinear(8, 4)
        relu = NFloatReLU()
        ct = NFloatTensor(torch.randn(2, 8))
        y = relu(linear(ct))
        s = summary()
        assert "forward" in s
        assert s["total_kernels"] == 2
        assert s["forward"]["num_kernels"] == 2

    def test_no_track(self):
        m = NFloatModule(torch.nn.ReLU(), track=False)
        ct = NFloatTensor(torch.randn(4, 8))
        _ = m(ct)
        reports = get_kernel_reports()
        assert len(reports) == 0

    def test_analysis_context_auto_converts_plain_tensor_and_records_metadata(self):
        linear = NFloatLinear(8, 4)
        x = torch.randn(2, 8, dtype=torch.float64)

        with nfloat_analysis(mode="test", use_ste=True, metadata={"case": "auto"}):
            out = linear(x)

        assert isinstance(out, NFloatTensor)
        report = get_kernel_reports()[-1]
        assert report.kernel_name == "nfloat/Linear"
        assert report.details["mode"] == "test"
        assert report.details["use_ste"] is True
        assert report.details["case"] == "auto"

    def test_custom_nfloat_operator_records_fine_grained_kernel(self):
        matmul = nfloat_operator(torch.matmul, name="matmul")
        x = NFloatTensor(torch.randn(2, 3, dtype=torch.float64))
        y = torch.randn(3, 4, dtype=torch.float64)

        out = matmul(x, y)

        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 4)
        assert get_kernel_reports()[-1].kernel_name == "nfloat/matmul"

    def test_iteration_tracker_captures_and_clears_reports(self):
        tracker = NFloatIterationTracker("demo")
        relu = NFloatReLU()

        with tracker.iteration(
            epoch=1,
            iteration=2,
            split="train",
            global_iteration=7,
            metadata={"granularity": "layer"},
        ):
            _ = relu(torch.randn(2, 4, dtype=torch.float64))

        assert len(tracker.rows) == 1
        row = tracker.rows[0]
        assert row["kernel_name"] == "nfloat/ReLU"
        assert row["epoch"] == 1
        assert row["iteration"] == 2
        assert row["global_iteration"] == 7
        assert row["details"]["mode"] == "train"
        assert row["details"]["granularity"] == "layer"
        assert len(get_kernel_reports()) == 0


# --------------------------------------------------------------------------- #
#  10. wrap_model                                                              #
# --------------------------------------------------------------------------- #

class TestWrapModel:
    def test_wrap_sequential(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
        )
        wrapped = wrap_model(model)
        assert isinstance(wrapped[0], NFloatModule)
        assert isinstance(wrapped[1], NFloatModule)
        assert isinstance(wrapped[2], NFloatModule)

    def test_wrapped_model_forward(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
        )
        wrapped = wrap_model(model)
        ct = NFloatTensor(torch.randn(2, 10))
        out = wrapped(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 4)
        reports = get_kernel_reports()
        assert len(reports) == 3  # 3 layers

    def test_wrap_nested_model(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 4)
                self.act = torch.nn.ReLU()

            def forward(self, x):
                return self.act(self.fc(x))

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(10, 8)
                self.sub = SubModule()

            def forward(self, x):
                return self.sub(self.encoder(x))

        model = MyModel()
        wrapped = wrap_model(model)
        ct = NFloatTensor(torch.randn(2, 10))
        out = wrapped(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 4)


# --------------------------------------------------------------------------- #
#  11. End-to-end pipeline                                                     #
# --------------------------------------------------------------------------- #

class TestEndToEnd:
    def test_full_pipeline(self):
        """Linear → ReLU → Linear → Softmax → CrossEntropy."""
        clear_kernel_reports()

        linear1 = NFloatLinear(10, 8)
        relu = NFloatReLU()
        linear2 = NFloatLinear(8, 5)
        softmax = NFloatSoftmax(dim=-1)

        x = NFloatTensor(torch.randn(4, 10))
        h = relu(linear1(x))
        logits = linear2(h)
        probs = softmax(logits)

        assert isinstance(probs, NFloatTensor)
        assert probs.shape == (4, 5)

        reports = get_kernel_reports()
        assert len(reports) == 4
        names = [r.kernel_name for r in reports]
        assert names == ["nfloat/Linear", "nfloat/ReLU", "nfloat/Linear", "nfloat/Softmax"]

        # All forward-pass reports
        assert all(r.phase == "forward" for r in reports)
        # Each report has avg_digits
        assert all(r.avg_digits >= 0 for r in reports)

    def test_pipeline_with_loss(self):
        """Full pipeline including loss computation."""
        clear_kernel_reports()

        linear = NFloatLinear(10, 5)
        loss_fn = NFloatCrossEntropyLoss()

        x = NFloatTensor(torch.randn(4, 10))
        logits = linear(x)
        target = torch.randint(0, 5, (4,))
        loss = loss_fn(logits, target)

        assert isinstance(loss, NFloatTensor)
        assert loss.avg_digits() >= 0

        reports = get_kernel_reports()
        assert len(reports) == 2  # Linear + CrossEntropy
        assert reports[1].kernel_name == "nfloat/CrossEntropyLoss"

    def test_digit_report_summary(self, capsys):
        """Verify human-readable output."""
        clear_kernel_reports()

        linear = NFloatLinear(10, 5)
        sigmoid = NFloatSigmoid()
        x = NFloatTensor(torch.randn(4, 10))
        y = sigmoid(linear(x))

        print_kernel_report()
        captured = capsys.readouterr()
        assert "Linear" in captured.out
        assert "Sigmoid" in captured.out
        assert "STABLE" in captured.out or "UNSTABLE" in captured.out


# --------------------------------------------------------------------------- #
#  12. New Layers – Convolutions                                               #
# --------------------------------------------------------------------------- #

class TestNewConvLayers:
    def test_conv3d(self):
        layer = NFloatConv3d(3, 8, kernel_size=3, padding=1)
        ct = NFloatTensor(torch.randn(1, 3, 4, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape[1] == 8

    def test_conv_transpose1d(self):
        layer = NFloatConvTranspose1d(8, 3, kernel_size=3, padding=1)
        ct = NFloatTensor(torch.randn(2, 8, 16))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape[1] == 3

    def test_conv_transpose2d(self):
        layer = NFloatConvTranspose2d(16, 3, kernel_size=3, padding=1)
        ct = NFloatTensor(torch.randn(2, 16, 8, 8))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape[1] == 3

    def test_conv_transpose3d(self):
        layer = NFloatConvTranspose3d(8, 3, kernel_size=3, padding=1)
        ct = NFloatTensor(torch.randn(1, 8, 4, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape[1] == 3


# --------------------------------------------------------------------------- #
#  13. New Layers – Normalization                                              #
# --------------------------------------------------------------------------- #

class TestNewNormLayers:
    def test_group_norm(self):
        layer = NFloatGroupNorm(num_groups=4, num_channels=8)
        ct = NFloatTensor(torch.randn(2, 8, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 8, 4)

    def test_instance_norm1d(self):
        layer = NFloatInstanceNorm1d(8)
        ct = NFloatTensor(torch.randn(2, 8, 16))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 8, 16)

    def test_instance_norm2d(self):
        layer = NFloatInstanceNorm2d(3)
        ct = NFloatTensor(torch.randn(2, 3, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 4, 4)

    def test_instance_norm3d(self):
        layer = NFloatInstanceNorm3d(3)
        ct = NFloatTensor(torch.randn(1, 3, 4, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (1, 3, 4, 4, 4)


# --------------------------------------------------------------------------- #
#  14. Pooling layers                                                          #
# --------------------------------------------------------------------------- #

class TestPooling:
    def test_maxpool1d(self):
        layer = NFloatMaxPool1d(kernel_size=2)
        ct = NFloatTensor(torch.randn(2, 3, 16))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 8)

    def test_maxpool2d(self):
        layer = NFloatMaxPool2d(kernel_size=2)
        ct = NFloatTensor(torch.randn(2, 3, 8, 8))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 4, 4)

    def test_maxpool3d(self):
        layer = NFloatMaxPool3d(kernel_size=2)
        ct = NFloatTensor(torch.randn(1, 3, 4, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (1, 3, 2, 2, 2)

    def test_avgpool1d(self):
        layer = NFloatAvgPool1d(kernel_size=2)
        ct = NFloatTensor(torch.randn(2, 3, 16))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 8)

    def test_avgpool2d(self):
        layer = NFloatAvgPool2d(kernel_size=2)
        ct = NFloatTensor(torch.randn(2, 3, 8, 8))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 4, 4)

    def test_avgpool3d(self):
        layer = NFloatAvgPool3d(kernel_size=2)
        ct = NFloatTensor(torch.randn(1, 3, 4, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (1, 3, 2, 2, 2)

    def test_adaptive_avgpool1d(self):
        layer = NFloatAdaptiveAvgPool1d(output_size=4)
        ct = NFloatTensor(torch.randn(2, 3, 16))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 4)

    def test_adaptive_avgpool2d(self):
        layer = NFloatAdaptiveAvgPool2d(output_size=(1, 1))
        ct = NFloatTensor(torch.randn(2, 3, 8, 8))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 1, 1)

    def test_adaptive_avgpool3d(self):
        layer = NFloatAdaptiveAvgPool3d(output_size=(1, 1, 1))
        ct = NFloatTensor(torch.randn(1, 3, 4, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (1, 3, 1, 1, 1)

    def test_adaptive_maxpool1d(self):
        layer = NFloatAdaptiveMaxPool1d(output_size=4)
        ct = NFloatTensor(torch.randn(2, 3, 16))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 4)

    def test_adaptive_maxpool2d(self):
        layer = NFloatAdaptiveMaxPool2d(output_size=(2, 2))
        ct = NFloatTensor(torch.randn(2, 3, 8, 8))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 2, 2)

    def test_adaptive_maxpool3d(self):
        layer = NFloatAdaptiveMaxPool3d(output_size=(1, 1, 1))
        ct = NFloatTensor(torch.randn(1, 3, 4, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (1, 3, 1, 1, 1)


# --------------------------------------------------------------------------- #
#  15. New Activations                                                         #
# --------------------------------------------------------------------------- #

class TestNewActivations:
    def test_leaky_relu(self):
        act = NFloatLeakyReLU()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 8)

    def test_elu(self):
        act = NFloatELU()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 8)

    def test_silu(self):
        act = NFloatSiLU()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 8)

    def test_mish(self):
        act = NFloatMish()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 8)

    def test_prelu(self):
        act = NFloatPReLU()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 8)

    def test_hardswish(self):
        act = NFloatHardswish()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 8)

    def test_hardtanh(self):
        act = NFloatHardtanh()
        ct = NFloatTensor(torch.randn(4, 8))
        out = act(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 8)
        # Hardtanh output should be in [-1, 1] (with stochastic rounding tolerance)
        assert np.all(out.mean >= -1.0 - 1e-4) and np.all(out.mean <= 1.0 + 1e-4)


# --------------------------------------------------------------------------- #
#  16. Dropout                                                                 #
# --------------------------------------------------------------------------- #

class TestDropout:
    def test_dropout_eval(self):
        layer = NFloatDropout(p=0.5)
        layer.eval()
        ct = NFloatTensor(torch.randn(4, 8))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 8)

    def test_dropout2d_eval(self):
        layer = NFloatDropout2d(p=0.5)
        layer.eval()
        ct = NFloatTensor(torch.randn(2, 3, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 4, 4)

    def test_dropout3d_eval(self):
        layer = NFloatDropout3d(p=0.5)
        layer.eval()
        ct = NFloatTensor(torch.randn(1, 3, 4, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (1, 3, 4, 4, 4)


# --------------------------------------------------------------------------- #
#  17. Embedding                                                               #
# --------------------------------------------------------------------------- #

class TestEmbedding:
    def test_embedding(self):
        layer = NFloatEmbedding(num_embeddings=100, embedding_dim=16)
        indices = torch.randint(0, 100, (4, 5))
        out = layer(indices)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (4, 5, 16)
        assert out.n_samples == 3
        assert get_kernel_reports()[-1].kernel_name == "nfloat/Embedding"

    def test_embedding_bag(self):
        layer = NFloatEmbeddingBag(num_embeddings=100, embedding_dim=16, mode="mean")
        indices = torch.randint(0, 100, (8,))
        offsets = torch.tensor([0, 3, 5])
        out = layer(indices, offsets)
        assert isinstance(out, NFloatTensor)
        assert out.shape[1] == 16
        assert out.n_samples == 3
        assert get_kernel_reports()[-1].kernel_name == "nfloat/EmbeddingBag"


# --------------------------------------------------------------------------- #
#  18. Padding                                                                 #
# --------------------------------------------------------------------------- #

class TestPadding:
    def test_zero_pad2d(self):
        layer = NFloatZeroPad2d(padding=1)
        ct = NFloatTensor(torch.randn(2, 3, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 6, 6)

    def test_reflection_pad2d(self):
        layer = NFloatReflectionPad2d(padding=1)
        ct = NFloatTensor(torch.randn(2, 3, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 6, 6)

    def test_replication_pad2d(self):
        layer = NFloatReplicationPad2d(padding=1)
        ct = NFloatTensor(torch.randn(2, 3, 4, 4))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 3, 6, 6)


# --------------------------------------------------------------------------- #
#  19. Transformer layers                                                      #
# --------------------------------------------------------------------------- #

class TestTransformer:
    def test_encoder_layer(self):
        layer = NFloatTransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)
        layer.eval()
        ct = NFloatTensor(torch.randn(2, 5, 16))
        out = layer(ct)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 5, 16)

    def test_decoder_layer(self):
        layer = NFloatTransformerDecoderLayer(d_model=16, nhead=4, batch_first=True)
        layer.eval()
        tgt = NFloatTensor(torch.randn(2, 5, 16))
        memory = torch.randn(2, 5, 16, dtype=torch.float64)
        # Decoder requires tgt and memory; pass memory as extra arg
        # The base NFloatModule.forward passes *args through
        out = layer(tgt, memory)
        assert isinstance(out, NFloatTensor)
        assert out.shape == (2, 5, 16)


# --------------------------------------------------------------------------- #
#  20. New Loss functions                                                      #
# --------------------------------------------------------------------------- #

class TestNewLoss:
    def test_bce_loss(self):
        loss_fn = NFloatBCELoss()
        ct = NFloatTensor(torch.sigmoid(torch.randn(4, 5)))
        target = torch.rand(4, 5).double()
        loss = loss_fn(ct, target)
        assert isinstance(loss, NFloatTensor)
        assert loss.mean >= 0

    def test_bce_with_logits_loss(self):
        loss_fn = NFloatBCEWithLogitsLoss()
        ct = NFloatTensor(torch.randn(4, 5))
        target = torch.rand(4, 5).double()
        loss = loss_fn(ct, target)
        assert isinstance(loss, NFloatTensor)
        assert loss.mean >= 0

    def test_smooth_l1_loss(self):
        loss_fn = NFloatSmoothL1Loss()
        ct = NFloatTensor(torch.randn(4, 5))
        target = torch.randn(4, 5).double()
        loss = loss_fn(ct, target)
        assert isinstance(loss, NFloatTensor)
        assert loss.mean >= 0

    def test_huber_loss(self):
        loss_fn = NFloatHuberLoss()
        ct = NFloatTensor(torch.randn(4, 5))
        target = torch.randn(4, 5).double()
        loss = loss_fn(ct, target)
        assert isinstance(loss, NFloatTensor)
        assert loss.mean >= 0

    def test_kl_div_loss(self):
        loss_fn = NFloatKLDivLoss(reduction="batchmean")
        log_probs = NFloatTensor(torch.nn.functional.log_softmax(torch.randn(4, 10), dim=-1))
        target = torch.nn.functional.softmax(torch.randn(4, 10), dim=-1).double()
        loss = loss_fn(log_probs, target)
        assert isinstance(loss, NFloatTensor)

    def test_ctc_loss(self):
        loss_fn = NFloatCTCLoss(blank=0, zero_infinity=True)
        # T=10, N=2, C=5
        log_probs = NFloatTensor(
            torch.nn.functional.log_softmax(torch.randn(10, 2, 5), dim=-1)
        )
        targets = torch.tensor([1, 2, 3, 1, 2], dtype=torch.long)
        input_lengths = torch.tensor([10, 10], dtype=torch.long)
        target_lengths = torch.tensor([3, 2], dtype=torch.long)
        loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
        assert isinstance(loss, NFloatTensor)
