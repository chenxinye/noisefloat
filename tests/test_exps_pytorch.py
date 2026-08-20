from __future__ import annotations

import json
import math
import warnings

import pytest
import torch

from noisefloat.nn import NFloatTensor, clear_kernel_reports
from noisefloat.nn.report import record_kernel

from exps.common_torch import (
    KernelDigitTracker,
    PlotStyle,
    configure_noisefloat_torch,
    sync_model_weights,
    tensor_add,
    tensor_concat,
    tensor_flatten,
    tensor_permute,
    tensor_scale,
)
from exps.models import ResNet18Classifier, Seq2SeqTransformer, UNetSegmentationModel
from exps.multi30k_transformer_translation_digits_th import (
    SimpleVocab,
    aggregate_operator_features,
    correlation_summary,
    digit_decile_summary,
    epoch_correlation_summary,
    epoch_detrended_correlation_summary,
    join_digits_decision,
    corpus_bleu,
    mask_decode_log_probs,
)
from exps.wmt14_ende_transformer_digits_th import (
    decision_metrics_from_loss_logits,
    select_informative_token_metric,
)
from exps.gpt_decoder_language_model_digits_th import (
    correlation_summary as gpt_correlation_summary,
    epoch_correlation_summary as gpt_epoch_correlation_summary,
    epoch_detrended_correlation_summary as gpt_epoch_detrended_correlation_summary,
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
    permuted = tensor_permute(
        NFloatTensor(torch.randn(2, 3, 4, dtype=torch.float64)), (1, 0, 2)
    )

    assert isinstance(added, NFloatTensor)
    assert isinstance(scaled, NFloatTensor)
    assert isinstance(flattened, NFloatTensor)
    assert isinstance(concatenated, NFloatTensor)
    assert isinstance(permuted, NFloatTensor)
    assert concatenated.shape == (2, 6)
    assert flattened.shape == (2, 3)
    assert permuted.shape == (3, 2, 4)


def test_sync_model_weights_handles_nfloat_wrappers():
    reference = ResNet18Classifier(num_classes=10, nfloat=False)
    nfloat = ResNet18Classifier(num_classes=10, nfloat=True)
    with torch.no_grad():
        reference.stem.layers[0].weight.fill_(0.125)
        reference.stem.layers[1].running_mean.fill_(0.5)
        reference.fc.weight.fill_(0.25)
        reference.fc.bias.fill_(-0.75)

    sync_model_weights(reference, nfloat)

    assert torch.equal(
        nfloat.stem.layers[0].inner.weight, reference.stem.layers[0].weight
    )
    assert torch.equal(
        nfloat.stem.layers[1].inner.running_mean, reference.stem.layers[1].running_mean
    )
    assert torch.equal(nfloat.fc.inner.weight, reference.fc.weight)
    assert torch.equal(nfloat.fc.inner.bias, reference.fc.bias)


def test_sync_model_weights_handles_transformer_and_unet_wrappers():
    transformer = Seq2SeqTransformer(
        17,
        19,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        pad_idx=0,
        max_len=8,
        nfloat=False,
    )
    nfloat_transformer = Seq2SeqTransformer(
        17,
        19,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        pad_idx=0,
        max_len=8,
        nfloat=True,
    )
    with torch.no_grad():
        transformer.src_embedding.weight.fill_(0.25)
        transformer.encoder_layers[0].self_attn.in_proj_weight.fill_(0.5)
        transformer.generator.weight.fill_(-0.25)

    sync_model_weights(transformer, nfloat_transformer)

    assert torch.equal(
        nfloat_transformer.src_embedding.inner.weight,
        transformer.src_embedding.weight,
    )
    assert torch.equal(
        nfloat_transformer.encoder_layers[0].self_attn.inner.in_proj_weight,
        transformer.encoder_layers[0].self_attn.in_proj_weight,
    )
    assert torch.equal(
        nfloat_transformer.generator.inner.weight, transformer.generator.weight
    )

    unet = UNetSegmentationModel(in_channels=3, num_classes=3, nfloat=False)
    nfloat_unet = UNetSegmentationModel(in_channels=3, num_classes=3, nfloat=True)
    with torch.no_grad():
        unet.inc.layers.layers[0].weight.fill_(0.375)
        unet.outc.weight.fill_(0.625)

    sync_model_weights(unet, nfloat_unet)

    assert torch.equal(
        nfloat_unet.inc.layers.layers[0].inner.weight, unet.inc.layers.layers[0].weight
    )
    assert torch.equal(nfloat_unet.outc.inner.weight, unet.outc.weight)


def test_tracker_exports_and_captures_metadata(tmp_path):
    record_kernel(
        "demo_kernel", "forward", NFloatTensor(torch.ones(2, 2, dtype=torch.float64))
    )
    tracker = KernelDigitTracker("demo_task")
    tracker.capture(
        epoch=1,
        iteration=2,
        split="train",
        global_iteration=5,
        metadata={"task_seed": 42},
    )
    tracker.export(tmp_path)

    rows = (
        (tmp_path / "kernel_digits.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["epoch"] == 1
    assert row["iteration"] == 2
    assert row["details"]["task_seed"] == 42
    assert (tmp_path / "kernel_digits.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_tracker_plot_writes_jpg(tmp_path):
    matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
    assert matplotlib is not None

    record_kernel(
        "plot_kernel", "forward", NFloatTensor(torch.ones(2, 2, dtype=torch.float64))
    )
    tracker = KernelDigitTracker("plot_task")
    tracker.capture(epoch=0, iteration=0, split="train", global_iteration=0)
    record_kernel(
        "plot_kernel",
        "forward",
        NFloatTensor(torch.full((2, 2), 2.0, dtype=torch.float64)),
    )
    tracker.capture(epoch=1, iteration=0, split="train", global_iteration=1)

    tracker.plot(tmp_path, style=PlotStyle(font_size=10), metrics=("avg_digits",))
    plots = list(tmp_path.glob("*.jpg"))
    assert plots


def test_multi30k_translation_helpers_encode_and_score_bleu():
    vocab = SimpleVocab(
        {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "ein": 4,
            "hund": 5,
            "rennt": 6,
        }
    )

    encoded = vocab.encode("ein hund rennt", max_length=8)

    assert vocab.decode(encoded.tolist()) == ["ein", "hund", "rennt"]
    assert corpus_bleu(
        [["ein", "hund", "rennt"]], [["ein", "hund", "rennt"]]
    ) == pytest.approx(100.0)
    assert corpus_bleu([["ein", "hund", "rennt"]], [["hund", "rennt"]]) < 100.0


def test_multi30k_decode_masks_special_tokens_and_early_eos():
    log_probs = torch.zeros(8)
    content_ids = {4, 5, 6}
    non_content_ids = {7}

    masked = mask_decode_log_probs(
        log_probs,
        pad_idx=0,
        bos_idx=1,
        eos_idx=2,
        unk_idx=3,
        tokens=[1],
        min_decode_tokens=3,
        min_decode_content_tokens=2,
        content_ids=content_ids,
        non_content_ids=non_content_ids,
        allow_unk_decode=False,
    )

    assert torch.isneginf(masked[0])
    assert torch.isneginf(masked[1])
    assert torch.isneginf(masked[2])
    assert torch.isneginf(masked[3])
    assert torch.isfinite(masked[4])
    assert torch.isneginf(masked[7])

    after_min_length = mask_decode_log_probs(
        log_probs,
        pad_idx=0,
        bos_idx=1,
        eos_idx=2,
        unk_idx=3,
        tokens=[1, 4, 5, 6],
        min_decode_tokens=3,
        min_decode_content_tokens=2,
        content_ids=content_ids,
        non_content_ids=non_content_ids,
        allow_unk_decode=True,
    )

    assert torch.isneginf(after_min_length[0])
    assert torch.isneginf(after_min_length[1])
    assert torch.isfinite(after_min_length[2])
    assert torch.isfinite(after_min_length[3])
    assert torch.isfinite(after_min_length[7])


def test_multi30k_improved_batch_analysis_tables_join_decision_metrics():
    kernel_rows = [
        {
            "split": "train",
            "epoch": 0,
            "iteration": 0,
            "global_iteration": 1,
            "kernel_name": "nfloat/MultiheadAttention",
            "phase": "forward",
            "avg_digits": 6.0,
            "min_digits": 4.0,
        },
        {
            "split": "train",
            "epoch": 0,
            "iteration": 0,
            "global_iteration": 1,
            "kernel_name": "nfloat/Linear",
            "phase": "forward",
            "avg_digits": 8.0,
            "min_digits": 5.0,
        },
        {
            "split": "train",
            "epoch": 0,
            "iteration": 1,
            "global_iteration": 2,
            "kernel_name": "nfloat/Linear",
            "phase": "forward",
            "avg_digits": 9.0,
            "min_digits": 6.0,
        },
    ]
    decision_rows = [
        {
            "split": "train",
            "epoch": 0,
            "iteration": 0,
            "global_iteration": 1,
            "gold_token_top1_rate": 0.2,
            "p05_logit_margin": 0.1,
            "p05_decision_snr": 2.0,
        },
        {
            "split": "train",
            "epoch": 0,
            "iteration": 1,
            "global_iteration": 2,
            "gold_token_top1_rate": 0.4,
            "p05_logit_margin": 0.3,
            "p05_decision_snr": 4.0,
        },
    ]

    operator_rows = aggregate_operator_features(kernel_rows)
    detail_rows = aggregate_operator_features(kernel_rows, group_by_kernel=True)
    joined = join_digits_decision(operator_rows, decision_rows)
    correlations = correlation_summary(joined)
    deciles = digit_decile_summary(joined, n_bins=2)

    assert {row["operator_group"] for row in operator_rows} == {
        "attention",
        "linear_matmul",
    }
    assert {row["kernel_name"] for row in detail_rows} == {
        "nfloat/MultiheadAttention",
        "nfloat/Linear",
    }
    assert joined
    assert correlations
    assert deciles


def test_gpt_correlation_summary_handles_constant_metrics_without_warning():
    joined = [
        {
            "operator_group": "linear_matmul",
            "p10_digits": 6.0,
            "gold_token_top1_rate": 0.5,
        },
        {
            "operator_group": "linear_matmul",
            "p10_digits": 7.0,
            "gold_token_top1_rate": 0.5,
        },
        {
            "operator_group": "linear_matmul",
            "p10_digits": 8.0,
            "gold_token_top1_rate": 0.5,
        },
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        correlations = gpt_correlation_summary(joined)

    row = next(
        item
        for item in correlations
        if item["target_metric"] == "gold_token_top1_rate"
    )
    assert math.isnan(row["pearson_r"])
    assert math.isnan(row["spearman_r"])
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_epoch_correlation_diagnostics_are_epoch_aware_and_detrended():
    joined = [
        {
            "split": "train",
            "epoch": 0,
            "iteration": 0,
            "global_iteration": 1,
            "operator_group": "linear_matmul",
            "p10_digits": 6.0,
            "median_digits": 7.0,
            "min_digits": 5.0,
            "gold_token_top1_rate": 0.2,
            "p05_logit_margin": 0.1,
            "p05_decision_snr": 2.0,
        },
        {
            "split": "train",
            "epoch": 0,
            "iteration": 1,
            "global_iteration": 2,
            "operator_group": "linear_matmul",
            "p10_digits": 7.0,
            "median_digits": 8.0,
            "min_digits": 6.0,
            "gold_token_top1_rate": 0.3,
            "p05_logit_margin": 0.2,
            "p05_decision_snr": 3.0,
        },
        {
            "split": "train",
            "epoch": 1,
            "iteration": 0,
            "global_iteration": 3,
            "operator_group": "linear_matmul",
            "p10_digits": 8.0,
            "median_digits": 9.0,
            "min_digits": 7.0,
            "gold_token_top1_rate": 0.6,
            "p05_logit_margin": 0.5,
            "p05_decision_snr": 6.0,
        },
        {
            "split": "train",
            "epoch": 1,
            "iteration": 1,
            "global_iteration": 4,
            "operator_group": "linear_matmul",
            "p10_digits": 9.0,
            "median_digits": 10.0,
            "min_digits": 8.0,
            "gold_token_top1_rate": 0.7,
            "p05_logit_margin": 0.6,
            "p05_decision_snr": 7.0,
        },
    ]

    epoch_rows = epoch_correlation_summary(joined)
    detrended_rows = epoch_detrended_correlation_summary(joined)
    gpt_epoch_rows = gpt_epoch_correlation_summary(joined)
    gpt_detrended_rows = gpt_epoch_detrended_correlation_summary(joined)

    assert {row["scope"] for row in epoch_rows} == {"per_epoch"}
    assert {row["scope"] for row in detrended_rows} == {"epoch_detrended"}
    assert {row["epoch"] for row in epoch_rows} == {0, 1}
    assert any(row["num_epochs"] == 2 for row in detrended_rows)
    assert gpt_epoch_rows
    assert gpt_detrended_rows


def test_wmt14_decision_metrics_accept_loss_order_logits():
    samples = torch.randn(3, 2, 5, 4, dtype=torch.float64)
    samples[:, 0, 1, 0] = 6.0
    samples[:, 0, 2, 1] = 5.0
    samples[:, 1, 3, 0] = 4.0
    logits = NFloatTensor.from_samples(samples)
    targets = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]], dtype=torch.long)

    metrics = decision_metrics_from_loss_logits(logits, targets, ignore_index=0)

    assert metrics["n_valid_tokens"] == 3
    assert 0.0 <= metrics["gold_token_top1_rate"] <= 1.0
    assert 0.0 <= metrics["gold_token_top5_rate"] <= 1.0
    assert 0.0 <= metrics["gold_token_top10_rate"] <= 1.0
    assert metrics["mean_gold_token_rank"] >= 1.0
    assert 0.0 < metrics["gold_token_mrr"] <= 1.0
    assert 0.0 <= metrics["mean_gold_token_probability"] <= 1.0
    assert "p05_logit_margin" in metrics
    assert "p05_decision_snr" in metrics


def test_wmt14_selects_informative_token_metric_when_top1_is_zero():
    pd = pytest.importorskip("pandas")
    joined = pd.DataFrame(
        [
            {
                "gold_token_top1_rate": 0.0,
                "gold_token_top5_rate": 0.0,
                "gold_token_top10_rate": 0.1,
                "gold_token_mrr": 0.02,
            },
            {
                "gold_token_top1_rate": 0.0,
                "gold_token_top5_rate": 0.0,
                "gold_token_top10_rate": 0.3,
                "gold_token_mrr": 0.05,
            },
        ]
    )

    metric, title = select_informative_token_metric(joined)

    assert metric == "gold_token_top10_rate"
    assert "top-10" in title
