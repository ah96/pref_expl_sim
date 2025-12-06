#!/usr/bin/env python3
"""
plot_experiments.py

Generate publication-ready figures for Experiment 1 and Experiment 2.

Input CSVs (expected in the working directory by default):
  - experiment1_runs.csv
  - experiment2_user_type_stats.csv
  - experiment2_fairness.csv

Output (in ./figures by default):
  - exp1_regret_panels.pdf
  - exp1_utility_barplots.pdf
  - exp2_user_type_outcomes.pdf
  - exp2_fairness.pdf
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# General plotting style
# ---------------------------------------------------------------------------

def configure_plot_style():
    """Set a clean, consistent style for all plots."""
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Experiment 1: Regret and Utility
# ---------------------------------------------------------------------------

def plot_experiment1_regret_panels(df_exp1: pd.DataFrame, out_path: str):
    """
    Create a 1×2 panel figure for Experiment 1 showing regret over episodes.

    Panel (a): Baseline policies (NE, AE, NO, PO, BA with alpha_mix = 0.5)
    Panel (b): BA policy with alpha sweep (alpha_mix ∈ {0, 0.25, 0.5, 0.75, 1.0})
    """
    # Ensure we only plot experiment 1
    df = df_exp1[df_exp1["experiment"] == 1].copy()

    # Define policy order and labels for consistency
    baseline_policies = ["NE", "AE", "NO", "PO", "BA"]
    policy_labels = {
        "NE": "No Explanations (NE)",
        "AE": "Always Explain (AE)",
        "NO": "Noisy (NO)",
        "PO": "Policy-Only (PO)",
        "BA": "Bayesian (BA, α=0.5)",
    }

    # Split baseline vs BA alpha-sweep
    # Baselines: all policies, with BA restricted to alpha_mix=0.5
    baseline_mask = df["policy"].isin(["NE", "AE", "NO", "PO"]) | (
        (df["policy"] == "BA") & (df["alpha_mix"] == 0.5)
    )
    df_baseline = df[baseline_mask].copy()

    # BA alpha sweep: only BA, all alpha_mix values
    df_ba = df[df["policy"] == "BA"].copy()

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax_base, ax_ba = axes

    # ---------- Panel (a): Baseline policies ----------
    for pol in baseline_policies:
        if pol == "BA":
            sub = df_baseline[(df_baseline["policy"] == "BA") &
                              (df_baseline["alpha_mix"] == 0.5)]
        else:
            sub = df_baseline[df_baseline["policy"] == pol]

        if sub.empty:
            continue

        ax_base.plot(
            sub["episode"],
            sub["regret"],
            marker="o",
            markersize=2,
            linewidth=1.0,
            label=policy_labels.get(pol, pol),
        )

    ax_base.set_title("(a) Baseline policies")
    ax_base.set_xlabel("Episode")
    ax_base.set_ylabel("Regret (oracle utility – actual utility)")
    ax_base.legend(frameon=False)

    # ---------- Panel (b): BA alpha sweep ----------
    # Use a consistent order of alpha values
    alpha_values = sorted([a for a in df_ba["alpha_mix"].dropna().unique()])
    for alpha in alpha_values:
        sub = df_ba[df_ba["alpha_mix"] == alpha]
        if sub.empty:
            continue

        ax_ba.plot(
            sub["episode"],
            sub["regret"],
            marker="o",
            markersize=2,
            linewidth=1.0,
            label=f"α = {alpha:g}",
        )

    ax_ba.set_title("(b) BA policy: α-sweep")
    ax_ba.set_xlabel("Episode")
    ax_ba.legend(frameon=False, title="Mixing parameter")

    # Layout & save
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment1_utility_bars(df_exp1: pd.DataFrame, out_path: str):
    """
    Create a 1×2 panel figure summarizing mean utility for Experiment 1.

    Panel (a): mean utility per baseline policy (NE, AE, NO, PO, BA α=0.5)
    Panel (b): mean utility for BA over alpha_mix ∈ {0, 0.25, 0.5, 0.75, 1.0}
    """
    df = df_exp1[df_exp1["experiment"] == 1].copy()

    # Baseline policies (with BA fixed at alpha_mix = 0.5)
    baseline_mask = df["policy"].isin(["NE", "AE", "NO", "PO"]) | (
        (df["policy"] == "BA") & (df["alpha_mix"] == 0.5)
    )
    df_baseline = df[baseline_mask].copy()

    # Aggregate mean utility per policy for baselines
    baseline_summary = (
        df_baseline
        .groupby("policy")["utility"]
        .agg(["mean", "std"])
        .reindex(["NE", "AE", "NO", "PO", "BA"])
        .reset_index()
    )

    # BA alpha sweep: aggregate mean utility per alpha_mix
    df_ba = df[df["policy"] == "BA"].copy()
    ba_summary = (
        df_ba
        .groupby("alpha_mix")["utility"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("alpha_mix")
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax_left, ax_right = axes

    # ---------- Panel (a): baseline policies ----------
    sns.barplot(
        data=baseline_summary,
        x="policy",
        y="mean",
        ax=ax_left,
        errorbar=None,
    )
    # Add error bars manually (std)
    ax_left.errorbar(
        x=np.arange(len(baseline_summary)),
        y=baseline_summary["mean"],
        yerr=baseline_summary["std"],
        fmt="none",
        capsize=4,
        linewidth=1.0,
    )
    ax_left.set_title("(a) Mean utility per baseline policy")
    ax_left.set_xlabel("Policy")
    ax_left.set_ylabel("Mean utility (±1 SD)")

    # ---------- Panel (b): BA alpha sweep ----------
    sns.barplot(
        data=ba_summary,
        x="alpha_mix",
        y="mean",
        ax=ax_right,
        errorbar=None,
    )
    ax_right.errorbar(
        x=np.arange(len(ba_summary)),
        y=ba_summary["mean"],
        yerr=ba_summary["std"],
        fmt="none",
        capsize=4,
        linewidth=1.0,
    )
    ax_right.set_title("(b) BA policy: mean utility vs α")
    ax_right.set_xlabel("Mixing parameter α")
    ax_right.set_ylabel("Mean utility (±1 SD)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: User-type outcomes and fairness
# ---------------------------------------------------------------------------

def plot_experiment2_user_type_outcomes(df_stats: pd.DataFrame, out_path: str):
    """
    Create a 2×2 grid of subplots for Experiment 2.

    For each user type:
      - X-axis: policy (NE, AE, NO, PO, BA)
      - Bars: mean utility (with standard deviation error bars)
      - Line (secondary axis): mean alignment

    This gives a compact overview of how each policy performs across user types.
    """
    df = df_stats[df_stats["experiment"] == 2].copy()

    # Fix ordering
    user_types_order = ["Minimalist", "ContextHungry", "VisualLearner", "NormFollower"]
    policy_order = ["NE", "AE", "NO", "PO", "BA"]

    # Create figure with 2×2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
    axes = axes.flatten()

    for i, user_type in enumerate(user_types_order):
        ax = axes[i]
        sub = df[df["user_type"] == user_type].copy()
        # Reindex so that all policies appear in the same order
        sub = sub.set_index("policy").reindex(policy_order).reset_index()

        # Primary axis: barplot of mean utility
        sns.barplot(
            data=sub,
            x="policy",
            y="mean_utility",
            ax=ax,
            errorbar=None,
        )
        # Add explicit error bars (std_utility)
        ax.errorbar(
            x=np.arange(len(sub)),
            y=sub["mean_utility"],
            yerr=sub["std_utility"],
            fmt="none",
            capsize=4,
            linewidth=1.0,
        )
        ax.set_title(user_type)
        ax.set_xlabel("Policy")
        ax.set_ylabel("Mean utility")

        # Secondary axis: line plot of mean alignment
        ax2 = ax.twinx()
        ax2.plot(
            np.arange(len(sub)),
            sub["mean_alignment"],
            marker="o",
            linewidth=1.0,
        )
        ax2.set_ylabel("Mean alignment", rotation=270, labelpad=15)

        # Improve tick labels spacing
        ax.set_xticklabels(policy_order)

    fig.suptitle("Experiment 2: Outcomes per user type", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment2_fairness(df_fair: pd.DataFrame, out_path: str):
    """
    Create a fairness summary figure for Experiment 2.

    For each policy:
      - Bar: overall mean utility
      - Vertical line: min to max utility (across user types)
      - Optional marker: mean (again) for clarity

    This visualizes both performance and disparity across user types.
    """
    df = df_fair[df_fair["experiment"] == 2].copy()
    policy_order = ["NE", "AE", "NO", "PO", "BA"]
    df = df.set_index("policy").reindex(policy_order).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Barplot for overall mean utility
    sns.barplot(
        data=df,
        x="policy",
        y="overall_mean_utility",
        ax=ax,
        errorbar=None,
    )

    # Add min–max ranges as vertical lines
    for i, row in df.iterrows():
        ax.vlines(
            x=i,
            ymin=row["min_utility"],
            ymax=row["max_utility"],
            linewidth=1.5,
        )

    ax.set_xlabel("Policy")
    ax.set_ylabel("Overall mean utility")
    ax.set_title("Experiment 2: Fairness across user types\n(min–max range per policy)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for Experiment 1 and Experiment 2."
    )
    parser.add_argument(
        "--exp1_csv",
        type=str,
        default="results/exp1/experiment1_runs.csv",
        help="Path to Experiment 1 runs CSV (default: experiment1_runs.csv)",
    )
    parser.add_argument(
        "--exp2_stats_csv",
        type=str,
        default="results/exp2/experiment2_user_type_stats.csv",
        help="Path to Experiment 2 user-type stats CSV (default: experiment2_user_type_stats.csv)",
    )
    parser.add_argument(
        "--exp2_fair_csv",
        type=str,
        default="results/exp2/experiment2_fairness.csv",
        help="Path to Experiment 2 fairness CSV (default: experiment2_fairness.csv)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/figures",
        help="Directory to save figures into (default: ./figures)",
    )
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.out_dir, exist_ok=True)

    # Load CSVs
    exp1 = pd.read_csv(args.exp1_csv)
    exp2_stats = pd.read_csv(args.exp2_stats_csv)
    exp2_fair = pd.read_csv(args.exp2_fair_csv)

    # Global style
    configure_plot_style()

    # ---------------- Experiment 1 ----------------
    plot_experiment1_regret_panels(
        exp1,
        out_path=os.path.join(args.out_dir, "exp1_regret_panels.pdf"),
    )
    plot_experiment1_utility_bars(
        exp1,
        out_path=os.path.join(args.out_dir, "exp1_utility_barplots.pdf"),
    )

    # ---------------- Experiment 2 ----------------
    plot_experiment2_user_type_outcomes(
        exp2_stats,
        out_path=os.path.join(args.out_dir, "exp2_user_type_outcomes.pdf"),
    )
    plot_experiment2_fairness(
        exp2_fair,
        out_path=os.path.join(args.out_dir, "exp2_fairness.pdf"),
    )


if __name__ == "__main__":
    main()
