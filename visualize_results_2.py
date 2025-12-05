#!/usr/bin/env python3
"""
visualize_results.py

Visualisation script for the simulation experiments.

This script is designed to be simple, stable, and LaTeX-ready. It expects
CSV files produced by the simulate.py experiments and generates a small
set of PDF figures with minimal styling.

It supports two experiment modes:

Experiment 1 (single-user + alpha sweep):
-----------------------------------------
Input:  experiment1_runs.csv  (via --input)

Figures:
    - exp1_regret.pdf
        Mean regret (oracle_utility - utility) vs. episode,
        for each policy (including BA with different alpha_mix values).

    - exp1_alignment.pdf
        Mean preference alignment vs. episode, for each policy.

Experiment 2 (population + fairness):
-------------------------------------
Inputs:
    - experiment2_user_type_stats.csv  (via --input-stats)
    - experiment2_fairness.csv         (via --input-fairness)

Figures:
    - exp2_mean_utility_per_usertype.pdf
        Grouped bar chart of mean utility for each user type and policy.

    - exp2_fairness.pdf
        Bar chart of overall mean utility per policy, annotated with
        utility variance and min-utility (worst-off users).

    - exp2_fairness_heatmap.pdf  (extra, useful for the paper)
        Heatmap of mean utility per (user_type, policy).

Usage examples
--------------

Experiment 1:
    python visualize_results.py \
        --experiment 1 \
        --input experiment1_runs.csv \
        --output-dir figs

Experiment 2:
    python visualize_results.py \
        --experiment 2 \
        --input-stats experiment2_user_type_stats.csv \
        --input-fairness experiment2_fairness.csv \
        --output-dir figs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_output_dir(path: Path) -> None:
    """
    Ensure that the output directory exists.

    If the directory does not exist, it will be created recursively.
    """
    path.mkdir(parents=True, exist_ok=True)


def _save_and_close(fig_path: Path) -> None:
    """
    Save the current matplotlib figure to the given path and close it.

    This helper keeps the main plotting code compact and avoids memory
    accumulation when generating multiple figures.
    """
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved {fig_path}")


# ---------------------------------------------------------------------------
# Experiment 1 plots
# ---------------------------------------------------------------------------

def plot_experiment1(
    csv_path: Path,
    output_dir: Path,
    user_type_name: Optional[str] = None,
) -> None:
    """
    Generate Experiment 1 plots:

        - exp1_regret.pdf
        - exp1_alignment.pdf

    Parameters
    ----------
    csv_path : Path
        Path to experiment1_runs.csv.
    output_dir : Path
        Directory where PDF files will be stored.
    user_type_name : Optional[str]
        Optional filter for a specific user archetype (e.g., "Minimalist").
        If None, all user types in the CSV are used.
    """
    _ensure_output_dir(output_dir)

    # Load the full run-level CSV.
    df = pd.read_csv(csv_path)

    # Optionally filter by a specific user archetype (e.g., Minimalist).
    if user_type_name is not None:
        df = df[df["user_type"] == user_type_name].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for user_type='{user_type_name}' in {csv_path}"
            )

    # Some policies (or baselines) may not use an oracle; in that case,
    # regret may be NaN. For the regret plot we only keep rows where it is
    # actually defined.
    df_reg = df.dropna(subset=["regret"]).copy()

    # ----------------------------------------------------------------------
    # 1) Regret vs episode
    # ----------------------------------------------------------------------
    # Group by (policy, alpha_mix, episode) and compute mean regret. This
    # naturally handles:
    #   - NE, AE, NO, PO (which have NaN alpha_mix),
    #   - BA policies with different alpha_mix values (0.0, 0.25, ..., 1.0).
    regret_group = df_reg.groupby(
        ["policy", "alpha_mix", "episode"]
    )["regret"].mean().reset_index()

    plt.figure(figsize=(5, 3))

    # Iterate over each (policy, alpha_mix) combination to get a separate line.
    for (policy, alpha), group in regret_group.groupby(["policy", "alpha_mix"]):
        # For non-BA policies, alpha_mix is NaN; in that case we only show
        # the policy name (e.g., "NE", "AE", "NO", "PO").
        if pd.isna(alpha) or alpha == "":
            label = str(policy)
        else:
            # For BA, show α in the legend.
            label = f"{policy} (α={alpha})"

        plt.plot(group["episode"], group["regret"], label=label)

    plt.xlabel("Episode")
    plt.ylabel("Mean Regret (oracle − policy)")
    plt.title("Experiment 1: Regret over Episodes")
    plt.legend(loc="best", fontsize=7)
    reg_path = output_dir / "exp1_regret.pdf"
    _save_and_close(reg_path)

    # ----------------------------------------------------------------------
    # 2) Preference alignment vs episode
    # ----------------------------------------------------------------------
    # Alignment is approximated as the mean of the three binary feedback
    # dimensions: modality, scope, and detail. Each dimension is coded as:
    #   1 = feedback positive / aligned
    #   0 = feedback negative / misaligned
    #
    # This gives a value in [0, 1] per episode, where 1 means the user liked
    # all three aspects of the explanation and 0 means they liked none.
    df = df.copy()
    df["alignment"] = df[
        ["feedback_modality", "feedback_scope", "feedback_detail"]
    ].mean(axis=1)

    align_group = df.groupby(
        ["policy", "alpha_mix", "episode"]
    )["alignment"].mean().reset_index()

    plt.figure(figsize=(5, 3))

    for (policy, alpha), group in align_group.groupby(["policy", "alpha_mix"]):
        if pd.isna(alpha) or alpha == "":
            label = str(policy)
        else:
            label = f"{policy} (α={alpha})"

        plt.plot(group["episode"], group["alignment"], label=label)

    plt.xlabel("Episode")
    plt.ylabel("Mean Alignment")
    plt.title("Experiment 1: Preference Alignment over Episodes")
    plt.legend(loc="best", fontsize=7)
    align_path = output_dir / "exp1_alignment.pdf"
    _save_and_close(align_path)


# ---------------------------------------------------------------------------
# Experiment 2 plots
# ---------------------------------------------------------------------------

def plot_experiment2(
    stats_path: Path,
    fairness_path: Path,
    output_dir: Path,
) -> None:
    """
    Generate Experiment 2 plots:

        - exp2_mean_utility_per_usertype.pdf
        - exp2_fairness.pdf
        - exp2_fairness_heatmap.pdf  (extra)

    Parameters
    ----------
    stats_path : Path
        Path to experiment2_user_type_stats.csv.
    fairness_path : Path
        Path to experiment2_fairness.csv.
    output_dir : Path
        Directory where PDF files will be stored.
    """
    _ensure_output_dir(output_dir)

    # stats_df: per (user_type, policy) summary
    stats_df = pd.read_csv(stats_path)
    # fair_df: global fairness statistics per policy
    fair_df = pd.read_csv(fairness_path)

    # ----------------------------------------------------------------------
    # 1) Mean utility per (user_type, policy): grouped bar chart
    # ----------------------------------------------------------------------
    # Pivot so that:
    #   - rows = user_type
    #   - columns = policy
    #   - cell   = mean_utility
    pivot = stats_df.pivot(
        index="user_type",
        columns="policy",
        values="mean_utility",
    )

    plt.figure(figsize=(6, 3))
    pivot.plot(kind="bar")
    plt.xlabel("User Type")
    plt.ylabel("Mean Utility")
    plt.title("Experiment 2: Mean Utility per User Type and Policy")
    plt.legend(loc="best", fontsize=7)
    util_path = output_dir / "exp2_mean_utility_per_usertype.pdf"
    _save_and_close(util_path)

    # ----------------------------------------------------------------------
    # 2) Fairness summary across policies (bar chart)
    # ----------------------------------------------------------------------
    # Each bar is a policy with its overall_mean_utility. We annotate with:
    #   - utility_variance (how uneven users are treated),
    #   - min_utility (worst-off users).
    plt.figure(figsize=(5, 3))
    x_positions = range(len(fair_df))
    plt.bar(x_positions, fair_df["overall_mean_utility"])

    plt.xticks(x_positions, fair_df["policy"])
    plt.xlabel("Policy")
    plt.ylabel("Overall Mean Utility")
    plt.title("Experiment 2: Fairness Summary")

    for i, row in fair_df.iterrows():
        text = "var={:.2f}\nmin={:.2f}".format(
            row["utility_variance"], row["min_utility"]
        )
        plt.text(
            i,
            row["overall_mean_utility"],
            text,
            ha="center",
            va="bottom",
            fontsize=6,
        )

    fair_path = output_dir / "exp2_fairness.pdf"
    _save_and_close(fair_path)

    # ----------------------------------------------------------------------
    # 3) (Extra) Fairness heatmap: user_type x policy -> mean_utility
    # ----------------------------------------------------------------------
    # This is a compact visualization to show how each policy treats each
    # user type. It can be used directly in the paper or as a sanity check.
    plt.figure(figsize=(5, 3))

    # We already have 'pivot' from above (user_type x policy -> mean_utility).
    im = plt.imshow(pivot.values, aspect="auto")

    # Add colorbar to show the scale of mean utilities.
    cbar = plt.colorbar(im)
    cbar.set_label("Mean Utility")

    # Set tick labels
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.title("Experiment 2: Fairness Heatmap (Mean Utility)")

    # Annotate each cell with its numeric value
    for i, user_type in enumerate(pivot.index):
        for j, policy in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            text = f"{val:.2f}"
            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    heatmap_path = output_dir / "exp2_fairness_heatmap.pdf"
    _save_and_close(heatmap_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse command-line arguments and dispatch to the Experiment 1 or 2
    plotting functions.

    This preserves the original interface you used in your scripts.
    """
    parser = argparse.ArgumentParser(
        description="Visualize simulation results for preference-aware explanations."
    )
    parser.add_argument(
        "--experiment", type=int, choices=[1, 2], required=True,
        help="Which experiment to visualize: 1 (single-user) or 2 (population).",
    )
    parser.add_argument(
        "--input", type=str,
        help="Input CSV for Experiment 1: experiment1_runs.csv.",
    )
    parser.add_argument(
        "--input-stats", type=str,
        help="User-type stats CSV for Experiment 2: experiment2_user_type_stats.csv.",
    )
    parser.add_argument(
        "--input-fairness", type=str,
        help="Fairness CSV for Experiment 2: experiment2_fairness.csv.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to store output figures.",
    )
    parser.add_argument(
        "--user-type", type=str, default=None,
        help="Optional filter by user type in Experiment 1 (e.g., Minimalist).",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.experiment == 1:
        if args.input is None:
            raise ValueError("--input is required for experiment 1.")
        plot_experiment1(
            csv_path=Path(args.input),
            output_dir=output_dir,
            user_type_name=args.user_type,
        )
    else:
        if args.input_stats is None or args.input_fairness is None:
            raise ValueError(
                "--input-stats and --input-fairness are required for experiment 2."
            )
        plot_experiment2(
            stats_path=Path(args.input_stats),
            fairness_path=Path(args.input_fairness),
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
