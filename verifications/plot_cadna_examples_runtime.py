"""Plot runtime comparisons for CADNA tutorial verification runs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class PlotConfig:
    font_size: int = 12
    title_font_size: Optional[int] = None
    label_font_size: Optional[int] = None
    tick_font_size: Optional[int] = None
    legend_font_size: Optional[int] = None
    dpi: int = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/cadna_examples_1_7_torch/cadna_examples_1_7_summary.csv"),
        help="summary CSV produced by cadna_examples_1_7.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cadna_examples_1_7_torch/figures"),
        help="directory for runtime comparison figures",
    )
    parser.add_argument(
        "--stem",
        default="cadna_examples_runtime_ratio",
        help="output filename stem without suffix",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("jpg", "png", "pdf"),
        help="figure formats to save",
    )
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--title-font-size", type=int, default=None)
    parser.add_argument("--label-font-size", type=int, default=None)
    parser.add_argument("--tick-font-size", type=int, default=None)
    parser.add_argument("--legend-font-size", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="plot runtime-ratio axis on a logarithmic scale",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "noisyfloat_mpl_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def configure_matplotlib(config: PlotConfig):
    ensure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": config.dpi,
            "savefig.dpi": config.dpi,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "font.size": config.font_size,
            "axes.titlesize": config.title_font_size or config.font_size + 1,
            "axes.labelsize": config.label_font_size or config.font_size,
            "xtick.labelsize": config.tick_font_size or config.font_size,
            "ytick.labelsize": config.tick_font_size or config.font_size,
            "legend.fontsize": config.legend_font_size or config.font_size - 1,
        }
    )
    return plt


def save_figure(fig, output_stem: Path, formats: Sequence[str]) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_stem.with_suffix(f".{fmt}"), bbox_inches="tight")


def runtime_ratio(row: dict[str, str]) -> float:
    if row.get("runtime_overhead_ratio"):
        return float(row["runtime_overhead_ratio"])
    deterministic_value = row.get("deterministic_runtime_mean_seconds")
    if not deterministic_value:
        deterministic_value = row.get("deterministic_double_runtime_mean_seconds", "")
    deterministic = float(deterministic_value)
    noisefloat = float(row["noisefloat_runtime_mean_seconds"])
    return noisefloat / deterministic if deterministic > 0.0 else float("nan")


def plot_runtime_comparison(
    rows: Sequence[dict[str, str]],
    output_stem: Path,
    *,
    config: PlotConfig,
    formats: Sequence[str],
    log_scale: bool,
) -> None:
    plt = configure_matplotlib(config)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("precision", "double").lower(), []).append(row)

    precisions = [name for name in ("single", "double") if name in grouped]
    if not precisions:
        precisions = sorted(grouped)

    ncols = len(precisions)
    fig_width = max(6.4, 4.9 * ncols)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 4.6), squeeze=False, sharey=True)
    color = "#4C78A8"

    for index, precision in enumerate(precisions):
        ax = axes[0][index]
        precision_rows = sorted(grouped[precision], key=lambda row: int(row["example"]))
        labels = [f"E{row['example']}" for row in precision_rows]
        ratios = [runtime_ratio(row) for row in precision_rows]
        x = list(range(len(labels)))
        ax.bar(
            x,
            ratios,
            width=0.62,
            label="Noisefloat / deterministic baseline",
            color=color,
            edgecolor="#1f1f1f",
            linewidth=0.7,
            hatch="///",
            alpha=0.95,
        )
        ax.axhline(1.0, color="#4a4a4a", linewidth=0.8, linestyle="--", alpha=0.65)
        ax.set_xticks(x, labels)
        ax.set_xlabel("CADNA tutorial example")
        if index == 0:
            ax.set_ylabel("Runtime ratio (x)")
        ax.set_title(f"{precision.capitalize()} stochastic precision")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if log_scale:
            ax.set_yscale("log")

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.suptitle("Runtime Overhead Ratio for CADNA Tutorial Verification")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_figure(fig, output_stem, formats)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_csv_rows(args.input_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.input_csv}")
    plot_runtime_comparison(
        rows,
        args.output_dir / args.stem,
        config=PlotConfig(
            font_size=args.font_size,
            title_font_size=args.title_font_size,
            label_font_size=args.label_font_size,
            tick_font_size=args.tick_font_size,
            legend_font_size=args.legend_font_size,
            dpi=args.dpi,
        ),
        formats=tuple(args.formats),
        log_scale=args.log_scale,
    )
    print(f"Wrote runtime comparison figure(s) to: {args.output_dir}")


if __name__ == "__main__":
    main()
