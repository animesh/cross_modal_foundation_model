"""
Phase 0: Baseline verification of the Tesorai cross-modal pretraining paper.

Runs 5 baselines (control-mean, perturbation-mean, HVG, PCA, additive-linear)
on the two protein perturbation benchmark datasets and compares against
Figure 3 of the paper.

Usage:
    pipenv run python run_phase0.py [--skip-download] [--dataset pxd|ptm|both]

Expected output:
    results/phase0_results.csv     -- per-pair results
    results/phase0_summary.csv     -- mean/std per baseline
    results/phase0_figure3.png     -- recreation of Figure 3

Paper values to match (read from Figure 3):
    PXD014791      Top-100 DE    Full-proteome
    control-mean   NaN           0.000
    perturb-mean   ~0.54         ~0.27
    HVG            ~0.54         ~0.26
    PCA            ~0.54         ~0.26
    additive-lin   ~0.54         ~0.26

    ProTargetMiner Top-100 DE    Full-proteome
    control-mean   NaN           0.000
    perturb-mean   ~0.75         ~0.73
    HVG            -0.652        -0.658    *** anomalously negative ***
    PCA            -0.650        -0.623    *** anomalously negative ***
    additive-lin   -0.652        -0.658
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("phase0.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 0 baseline benchmark")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip PRIDE download (assumes files already present)")
    parser.add_argument("--dataset", choices=["pxd", "ptm", "both"], default="both",
                        help="Which dataset to benchmark (default: both)")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    all_results = []

    # ------------------------------------------------------------------
    # Optional download
    # ------------------------------------------------------------------
    if not args.skip_download:
        logger.info("=== Step 1: Download data from PRIDE ===")
        from download import download_all
        download_all()
    else:
        logger.info("=== Step 1: Skipping download (--skip-download) ===")

    # ------------------------------------------------------------------
    # PXD014791 (cardiomyocyte LFQ)
    # ------------------------------------------------------------------
    if args.dataset in ("pxd", "both"):
        logger.info("\n=== Step 2a: Load PXD014791 (LFQ) ===")
        try:
            from parse_lfq import load_pxd014791
            from benchmark import build_pairs, run_benchmark
            from config import PXD014791_HELD_OUT_DRUGS

            matrix, meta = load_pxd014791()
            _validate_matrix(matrix, "PXD014791")

            pairs, proteins = build_pairs(
                matrix, meta,
                held_out_drugs=PXD014791_HELD_OUT_DRUGS,
                context_col="donor",
            )
            _warn_pair_count(pairs, expected=58, name="PXD014791")

            results_pxd = run_benchmark(pairs, dataset_name="PXD014791")
            results_pxd.to_csv("results/pxd014791_per_pair.csv", index=False)
            all_results.append(results_pxd)
            logger.info(f"  Saved per-pair results: results/pxd014791_per_pair.csv")

        except FileNotFoundError as e:
            logger.error(f"  {e}")
            logger.error("  --> Run without --skip-download, or manually place proteinGroups.txt")

    # ------------------------------------------------------------------
    # ProTargetMiner (TMT)
    # ------------------------------------------------------------------
    if args.dataset in ("ptm", "both"):
        logger.info("\n=== Step 2b: Load ProTargetMiner (TMT) ===")
        try:
            from parse_tmt import load_protargetminer
            from benchmark import build_pairs, run_benchmark
            from config import PTM_HELD_OUT_DRUGS

            matrix, meta = load_protargetminer()
            _validate_matrix(matrix, "ProTargetMiner")

            pairs, proteins = build_pairs(
                matrix, meta,
                held_out_drugs=PTM_HELD_OUT_DRUGS,
                context_col="cell_line",
            )
            _warn_pair_count(pairs, expected=61, name="ProTargetMiner")

            results_ptm = run_benchmark(pairs, dataset_name="ProTargetMiner")
            results_ptm.to_csv("results/protargetminer_per_pair.csv", index=False)
            all_results.append(results_ptm)
            logger.info(f"  Saved per-pair results: results/protargetminer_per_pair.csv")

        except RuntimeError as e:
            logger.error(f"  {e}")
            logger.error(
                "  --> Fill in HARDCODED_CHANNEL_MAP in parse_tmt.py from Saei 2019 Supp Table 2"
            )

    # ------------------------------------------------------------------
    # Summarise and plot
    # ------------------------------------------------------------------
    if not all_results:
        logger.error("No results to summarise. Check errors above.")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("results/phase0_results.csv", index=False)

    from benchmark import summarise
    summary = summarise(combined)
    summary.to_csv("results/phase0_summary.csv")

    logger.info("\n=== Summary (mean Pearson across held-out pairs) ===")
    logger.info("\n" + summary.to_string())

    _plot_figure3(combined, "results/phase0_figure3.png")
    logger.info("\nFigure saved: results/phase0_figure3.png")
    logger.info("\n=== Phase 0 complete ===")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _validate_matrix(matrix: pd.DataFrame, name: str) -> None:
    n_finite = matrix.apply(lambda col: col.notna().sum()).sum()
    frac = n_finite / matrix.size
    logger.info(
        f"  {name}: {matrix.shape[0]} samples x {matrix.shape[1]} proteins "
        f"({frac*100:.1f}% non-NaN)"
    )
    if frac < 0.05:
        logger.warning(
            f"  [!] Very sparse matrix ({frac*100:.1f}% non-NaN). "
            f"Check parsing -- possible 0->NaN over-imputation."
        )


def _warn_pair_count(pairs, expected: int, name: str) -> None:
    from benchmark import Pair
    held_out = sum(p.is_held_out for p in pairs)
    logger.info(f"  {name}: {held_out} held-out pairs (paper reports {expected})")
    if abs(held_out - expected) > 5:
        logger.warning(
            f"  [!] Pair count differs from paper ({held_out} vs {expected}).\n"
            f"      This may indicate a drug name mismatch in the metadata parsing.\n"
            f"      Check parse_lfq.py _parse_sample_name() or parse_tmt.py channel map."
        )


def _plot_figure3(results: pd.DataFrame, out_path: str) -> None:
    """Recreate Figure 3 style bar chart for comparison with paper."""
    datasets = results["dataset"].unique()
    baselines = ["control-mean", "perturbation-mean", "HVG", "PCA", "additive-linear"]
    metrics = ["pearson_top100", "pearson_full"]
    metric_labels = ["Top-100 DE Pearson (primary)", "Full-proteome Pearson"]

    # Paper values from Figure 3 (approximate reads for overlay)
    paper_values = {
        "PXD014791": {
            "pearson_top100": {
                "control-mean": None,
                "perturbation-mean": 0.54,
                "HVG": 0.54,
                "PCA": 0.54,
                "additive-linear": 0.54,
            },
            "pearson_full": {
                "control-mean": 0.000,
                "perturbation-mean": 0.27,
                "HVG": 0.26,
                "PCA": 0.26,
                "additive-linear": 0.26,
            },
        },
        "ProTargetMiner": {
            "pearson_top100": {
                "control-mean": None,
                "perturbation-mean": 0.75,
                "HVG": -0.652,
                "PCA": -0.650,
                "additive-linear": -0.652,
            },
            "pearson_full": {
                "control-mean": 0.000,
                "perturbation-mean": 0.73,
                "HVG": -0.658,
                "PCA": -0.623,
                "additive-linear": -0.658,
            },
        },
    }

    n_datasets = len(datasets)
    fig, axes = plt.subplots(
        len(metrics), n_datasets,
        figsize=(6 * n_datasets, 5 * len(metrics)),
        squeeze=False,
    )

    colors = {
        "control-mean": "#aec7e8",
        "perturbation-mean": "#ffbb78",
        "HVG": "#98df8a",
        "PCA": "#ff9896",
        "additive-linear": "#c5b0d5",
    }

    for col_idx, dataset in enumerate(datasets):
        for row_idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_idx][col_idx]
            ds_res = results[results["dataset"] == dataset]
            x = range(len(baselines))

            for xi, bl in enumerate(baselines):
                bl_res = ds_res[ds_res["baseline"] == bl][metric].dropna()
                if len(bl_res) == 0:
                    continue
                mean_val = bl_res.mean()
                std_val = bl_res.std()
                bar = ax.bar(xi, mean_val, color=colors.get(bl, "grey"),
                             label=bl, alpha=0.85)
                ax.errorbar(xi, mean_val, yerr=std_val, fmt="none",
                            color="black", capsize=3, linewidth=1)

                # Overlay paper value as horizontal tick mark
                pv = paper_values.get(dataset, {}).get(metric, {}).get(bl)
                if pv is not None:
                    ax.hlines(pv, xi - 0.35, xi + 0.35,
                              colors="red", linestyles="dashed",
                              linewidth=1.5, label="_paper" if xi == 0 else "")

            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"{dataset}\n{mlabel}", fontsize=9)
            ax.set_xticks(list(x))
            ax.set_xticklabels(baselines, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("Pearson r" if col_idx == 0 else "")

            if row_idx == 0 and col_idx == 0:
                ax.annotate(
                    "Red dashed = paper value",
                    xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=7, color="red",
                )

    plt.suptitle(
        "Phase 0: Baseline verification vs Tesorai Figure 3\n"
        "(bars = our values, red dashes = paper values)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
