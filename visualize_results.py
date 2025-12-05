#!/usr/bin/env python3
"""
visualize_results.py

Visualisation script for the simulation experiments.

Generates PDF figures ready for inclusion in LaTeX:

Experiment 1 (single-user + alpha sweep):
    - exp1_regret.pdf:
        Mean regret (oracle_utility - utility) vs. episode,
        per policy (including BA with different alpha_mix).
    - exp1_alignment.pdf:
        Mean preference alignment vs. episode, per policy.

Experiment 2 (population + fairness):
    - exp2_mean_utility_per_usertype.pdf:
        Grouped bar chart of mean utility for each user type and policy.
    - exp2_fairness.pdf:
        Bar chart of overall mean utility per policy, annotated with
        utility variance and min-utility (worst-off users).

Usage examples:

    # Experiment 1
    python visualize_results.py \
        --experiment 1 \
        --input results/exp1_rerun/experiment1_runs.csv \
        --output-dir results/figures

    # Experiment 2
    python visualize_results.py \
        --experiment 2 \
        --input-stats results/exp2_rerun/experiment2_user_type_stats.csv \
        --input-fairness results/exp2_rerun/experiment2_fairness.csv \
        --output-dir results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir(path: Path) -> None:
    """Create the output directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


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

    Args:
        csv_path: Path to experiment1_runs.csv.
        output_dir: Directory where PDF files will be stored.
        user_type_name: Optional filter for a specific user archetype.
    """
    _ensure_output_dir(output_dir)

    df = pd.read_csv(csv_path)

    # Optionally filter by a specific user archetype (usually "Minimalist").
    if user_type_name is not None:
        df = df[df["user_type"] == user_type_name]

    # Some policies (if oracle was not used) may have NaN in regret;
    # we only use rows where regret is defined for plotting.
    df_reg = df.dropna(subset=["regret"]).copy()

    # ----------------------------------------------------------------------
    # 1) Regret vs episode
    # ----------------------------------------------------------------------
    # Group by policy, alpha, and episode; compute mean regret.
    regret_group = df_reg.groupby(
        ["policy", "alpha_mix", "episode"]
    )["regret"].mean().reset_index()

    plt.figure(figsize=(5, 3))
    for (policy, alpha), group in regret_group.groupby(["policy", "alpha_mix"]):
        # alpha_mix is NaN for non-BA policies; pretty-print label accordingly.
        if pd.isna(alpha) or alpha == "":
            label = str(policy)
        else:
            # Use a compact “alpha” format (no trailing .0 if possible).
            label = "{} (α={})".format(policy, alpha)
        plt.plot(group["episode"], group["regret"], label=label)

    plt.xlabel("Episode")
    plt.ylabel("Mean Regret (oracle − policy)")
    plt.title("Experiment 1: Regret over Episodes")
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()

    reg_path = output_dir / "exp1_regret.pdf"
    plt.savefig(reg_path, bbox_inches="tight")
    plt.close()
    print("[Visualization] Saved", reg_path)

    # ----------------------------------------------------------------------
    # 2) Preference alignment vs episode
    # ----------------------------------------------------------------------
    # Alignment = mean of binary feedback across attribute families:
    # here approximated as the mean of feedback_modality/scope/detail.
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
            label = "{} (α={})".format(policy, alpha)
        plt.plot(group["episode"], group["alignment"], label=label)

    plt.xlabel("Episode")
    plt.ylabel("Mean Alignment")
    plt.title("Experiment 1: Preference Alignment over Episodes")
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()

    align_path = output_dir / "exp1_alignment.pdf"
    plt.savefig(align_path, bbox_inches="tight")
    plt.close()
    print("[Visualization] Saved", align_path)


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

    Args:
        stats_path: Path to experiment2_user_type_stats.csv.
        fairness_path: Path to experiment2_fairness.csv.
        output_dir: Directory where PDF files will be stored.
    """
    _ensure_output_dir(output_dir)

    stats_df = pd.read_csv(stats_path)
    fair_df = pd.read_csv(fairness_path)

    # ----------------------------------------------------------------------
    # 1) Mean utility per user_type x policy (grouped bar chart)
    # ----------------------------------------------------------------------
    # Pivot so that user_type are rows and policies are columns.
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
    plt.tight_layout()

    util_path = output_dir / "exp2_mean_utility_per_usertype.pdf"
    plt.savefig(util_path, bbox_inches="tight")
    plt.close()
    print("[Visualization] Saved", util_path)

    # ----------------------------------------------------------------------
    # 2) Fairness summary across policies
    # ----------------------------------------------------------------------
    # Each bar is a policy with overall_mean_utility.
    # We annotate bars with variance and min-utility.
    plt.figure(figsize=(5, 3))
    x_positions = range(len(fair_df))
    plt.bar(x_positions, fair_df["overall_mean_utility"])

    plt.xticks(x_positions, fair_df["policy"])
    plt.xlabel("Policy")
    plt.ylabel("Overall Mean Utility")
    plt.title("Experiment 2: Fairness Summary")

    # Annotate each bar with variance and min-utility (worst-off users).
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

    plt.tight_layout()
    fair_path = output_dir / "exp2_fairness.pdf"
    plt.savefig(fair_path, bbox_inches="tight")
    plt.close()
    print("[Visualization] Saved", fair_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
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
