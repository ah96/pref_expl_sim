#!/usr/bin/env python3
"""
plot_experiments.py

Generate publication-ready figures for Experiment 1 and Experiment 2.

Expected input CSVs (in the working directory by default):
  - experiment1_runs.csv
  - experiment2_user_type_stats.csv
  - experiment2_fairness.csv

Output (in ./figures by default):

  Experiment 1 (Minimalist user)
  --------------------------------
  - exp1_regret_band.pdf
        Mean ± std regret over episodes for a subset of policies.
        Still shows temporal dynamics, but no spaghetti.

  - exp1_regret_final_bar.pdf   <-- RECOMMENDED FOR THE LBR
        Final-episode regret as a simple bar+dot plot for each
        policy / BA–α variant. Best choice given deterministic
        convergence (zero variance within each variant).

  - exp1_regret_slope.pdf
        Start vs. end mean regret per variant (slope chart).

  - exp1_regret_heatmap.pdf
        Binned heatmap of mean regret over time for each variant.

  Experiment 2 (population-level)
  --------------------------------
  - exp2_user_type_outcomes.pdf
        2×2 grid: mean utility (bars) + mean alignment (line)
        for each user archetype and policy.

  - exp2_fairness.pdf
        Overall mean utility per policy + min–max utility range
        across user types (fairness summary).

Typical usage:
    python3 plot_experiments.py

You can then include, for example:
    figures/exp1_regret_final_bar.pdf
    figures/exp2_user_type_outcomes.pdf
    figures/exp2_fairness.pdf
in your LBR.
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
    """
    Configure a consistent, paper-friendly style for all plots.

    We use seaborn's "paper" context and a whitegrid background, plus a few
    rcParams tweaks for font sizes and spine visibility.
    """
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


# ======================================================================
# EXPERIMENT 1: ALTERNATIVE PLOTTING STYLES (NO SPAGHETTI)
# ======================================================================

def _prepare_exp1_df(df_exp1: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and return only Experiment 1 data with a clean copy.

    Assumes the CSV has:
        - a column 'experiment' (1 or 2)
        - a column 'policy' (NE, AE, NO, PO, BA)
        - a column 'episode' (int)
        - a column 'regret' (float)
        - optionally 'alpha_mix' (float) for BA

    For non-BA policies, alpha_mix is filled with 0.0 for convenience.
    """
    df = df_exp1[df_exp1["experiment"] == 1].copy()

    # Ensure alpha_mix exists for all rows
    if "alpha_mix" not in df.columns:
        df["alpha_mix"] = 0.0

    return df


def _policy_variant_label(row) -> str:
    """
    Helper: construct a compact label combining policy and alpha (for BA).

    - NE, AE, NO, PO stay as-is (e.g., "NE").
    - BA rows become e.g. 'BA-α=0.0', 'BA-α=0.5', etc.

    This keeps plot legends concise and ordering manageable.
    """
    pol = row["policy"]
    if pol != "BA":
        return pol

    alpha = row["alpha_mix"]
    # Use :g to avoid trailing zeros like 0.500000
    return f"BA-α={alpha:g}"


# ----------------------------------------------------------------------
# Option 1: AREA BAND PLOT (mean ± std over time) FOR SELECTED POLICIES
# ----------------------------------------------------------------------

def plot_exp1_regret_band(df_exp1: pd.DataFrame, out_path: str):
    """
    Plot regret over episodes as mean ± std bands for a *subset* of policies
    to avoid spaghetti.

    We show:
      - NE, AE, PO
      - BA with α ∈ {0.0, 0.5, 1.0}

    This still conveys the temporal learning dynamics, but with a small
    number of bands instead of 10 overlapping lines.
    """
    df = _prepare_exp1_df(df_exp1)

    policies_to_show = ["NE", "AE", "PO"]
    ba_alphas_to_show = [0.0, 0.5, 1.0]

    subset = df[
        (df["policy"].isin(policies_to_show)) |
        ((df["policy"] == "BA") & (df["alpha_mix"].isin(ba_alphas_to_show)))
    ].copy()

    # Aggregate across runs: mean and std per (policy, alpha, episode)
    agg = (
        subset
        .groupby(["policy", "alpha_mix", "episode"])["regret"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Create legend labels (e.g., "NE", "BA-α=0.5")
    agg["variant"] = agg.apply(_policy_variant_label, axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))

    for variant, sub in agg.groupby("variant"):
        sub = sub.sort_values("episode")
        episodes = sub["episode"]
        mean = sub["mean"]
        std = sub["std"]

        # Mean line
        ax.plot(episodes, mean, label=variant, linewidth=1.2)
        # Shaded ±1 std band
        ax.fill_between(
            episodes,
            mean - std,
            mean + std,
            alpha=0.2
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Regret (oracle utility – actual utility)")
    ax.set_title("Experiment 1: regret over time (mean ± std)")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Option 2: FINAL-EPISODE BAR + DOT PLOT (RECOMMENDED FOR LBR)
# ----------------------------------------------------------------------

def plot_exp1_final_regret_bar(df_exp1: pd.DataFrame, out_path: str):
    """
    Plot final-episode regret as a simple bar+dot plot for all
    policies and BA–α variants.

    This is the recommended plot for the LBR, especially when each
    policy converges deterministically across runs (zero variance
    within each group). It shows:

        - one value per policy / BA variant
        - clear ordering w.r.t. final regret
        - no spaghetti, no misleading "distribution" when variance=0
    """
    df = _prepare_exp1_df(df_exp1)

    # Identify the last episode index present in the data
    final_ep = df["episode"].max()
    final_df = df[df["episode"] == final_ep].copy()

    # One row per run at final episode; build variant labels
    final_df["variant"] = final_df.apply(_policy_variant_label, axis=1)

    # If all runs converge deterministically, multiple rows per variant
    # will have the same 'regret' value. We aggregate by mean (which
    # therefore equals that deterministic value).
    grouped = final_df.groupby("variant")["regret"].mean().reset_index()

    # Define a consistent ordering:
    #   1) Baselines (NE, AE, NO, PO)
    #   2) BA variants sorted by α
    baseline_order = ["NE", "AE", "NO", "PO"]
    ba_variants = sorted(
        v for v in grouped["variant"].unique() if v.startswith("BA-α=")
    )
    variant_order = baseline_order + ba_variants
    # Keep only variants that actually appear
    variant_order = [v for v in variant_order if v in grouped["variant"].unique()]

    # Reindex in the desired order
    grouped = grouped.set_index("variant").loc[variant_order].reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    # Simple bar plot of final regrets
    sns.barplot(
        data=grouped,
        x="variant",
        y="regret",
        ax=ax,
        color="skyblue",
        edgecolor="black",
        width=0.7,
    )

    # Overlay exact mean value as a black dot
    ax.scatter(
        x=np.arange(len(grouped)),
        y=grouped["regret"],
        color="black",
        zorder=5,
    )

    ax.set_xlabel("Policy / BA variant")
    ax.set_ylabel("Final-episode regret")
    ax.set_title("Experiment 1: final-episode regret per variant")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Option 3: SLOPE CHART (START vs END MEAN REGRET PER VARIANT)
# ----------------------------------------------------------------------

def plot_exp1_regret_slope(df_exp1: pd.DataFrame, out_path: str):
    """
    Plot a slope chart mapping start-episode mean regret to final-episode
    mean regret for each policy / BA–α variant.

    This shows *direction and magnitude* of change with only two time
    points, avoiding full time-series spaghetti.
    """
    df = _prepare_exp1_df(df_exp1)

    start_ep = df["episode"].min()
    final_ep = df["episode"].max()

    df_start = df[df["episode"] == start_ep].copy()
    df_final = df[df["episode"] == final_ep].copy()

    # Average over runs at start
    start_agg = (
        df_start.groupby(["policy", "alpha_mix"])["regret"]
        .mean()
        .reset_index()
        .rename(columns={"regret": "regret_start"})
    )

    # Average over runs at end
    final_agg = (
        df_final.groupby(["policy", "alpha_mix"])["regret"]
        .mean()
        .reset_index()
        .rename(columns={"regret": "regret_final"})
    )

    merged = start_agg.merge(final_agg, on=["policy", "alpha_mix"])
    merged["variant"] = merged.apply(_policy_variant_label, axis=1)

    baseline_order = ["NE", "AE", "NO", "PO"]
    ba_variants = sorted(
        v for v in merged["variant"].unique() if v.startswith("BA-α=")
    )
    variant_order = baseline_order + ba_variants
    variant_order = [v for v in variant_order if v in merged["variant"].unique()]
    merged = merged.set_index("variant").loc[variant_order].reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Positions: x=0 (start), x=1 (end)
    x_positions = [0, 1]
    for _, row in merged.iterrows():
        ax.plot(
            x_positions,
            [row["regret_start"], row["regret_final"]],
            marker="o",
            linewidth=1.5,
            label=row["variant"],
        )

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Start", "End"])
    ax.set_ylabel("Mean regret")
    ax.set_title("Experiment 1: start vs end regret per variant")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Option 4: HEATMAP OF REGRET OVER EPISODE BINS
# ----------------------------------------------------------------------

def plot_exp1_regret_heatmap(df_exp1: pd.DataFrame, out_path: str, n_bins: int = 5):
    """
    Plot a heatmap of mean regret over coarse episode bins for each
    policy / BA–α variant.

    This aggregates over episodes (e.g., 1–10, 11–20, ...) and shows
    temporal trends without per-episode detail, avoiding line clutter.
    """
    df = _prepare_exp1_df(df_exp1)

    df["variant"] = df.apply(_policy_variant_label, axis=1)

    ep_min, ep_max = df["episode"].min(), df["episode"].max()
    # Build bin edges (inclusive on the left, exclusive on the right)
    bins = np.linspace(ep_min, ep_max + 1, n_bins + 1)
    df["episode_bin"] = pd.cut(
        df["episode"],
        bins=bins,
        include_lowest=True,
        labels=[f"{int(bins[i])}–{int(bins[i + 1] - 1)}" for i in range(n_bins)],
    )

    # Mean regret per (variant, episode_bin)
    heat = (
        df.groupby(["variant", "episode_bin"])["regret"]
        .mean()
        .reset_index()
    )

    # Pivot: rows = variants, columns = episode bins
    heat_pivot = heat.pivot(index="variant", columns="episode_bin", values="regret")

    # Order variants for the heatmap
    baseline_order = ["NE", "AE", "NO", "PO"]
    ba_variants = sorted(
        v for v in heat_pivot.index if v.startswith("BA-α=")
    )
    idx_order = baseline_order + ba_variants
    idx_order = [v for v in idx_order if v in heat_pivot.index]
    heat_pivot = heat_pivot.loc[idx_order]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        heat_pivot,
        annot=False,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Mean regret"},
    )

    ax.set_xlabel("Episode bin")
    ax.set_ylabel("Policy / BA variant")
    ax.set_title("Experiment 1: regret over time (binned heatmap)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# EXPERIMENT 2: USER-TYPE OUTCOMES AND FAIRNESS
# ======================================================================

def plot_experiment2_user_type_outcomes(df_stats: pd.DataFrame, out_path: str):
    """
    Create a 2×2 grid of subplots for Experiment 2.

    For each user type:
      - X-axis: policy (NE, AE, NO, PO, BA)
      - Bars: mean utility (with standard deviation error bars)
      - Line (secondary axis): mean alignment

    This gives a compact overview of how each policy performs across
    user archetypes in terms of utility and preference alignment.
    """
    df = df_stats[df_stats["experiment"] == 2].copy()

    user_types_order = ["Minimalist", "ContextHungry", "VisualLearner", "NormFollower"]
    policy_order = ["NE", "AE", "NO", "PO", "BA"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
    axes = axes.flatten()

    for i, user_type in enumerate(user_types_order):
        ax = axes[i]
        sub = df[df["user_type"] == user_type].copy()
        # Ensure consistent policy ordering (some policies may be absent)
        sub = sub.set_index("policy").reindex(policy_order).reset_index()

        # Primary axis: mean utility with std error bars
        sns.barplot(
            data=sub,
            x="policy",
            y="mean_utility",
            ax=ax,
            errorbar=None,
        )
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

        # Secondary axis: mean alignment
        ax2 = ax.twinx()
        ax2.plot(
            np.arange(len(sub)),
            sub["mean_alignment"],
            marker="o",
            linewidth=1.0,
        )
        ax2.set_ylabel("Mean alignment", rotation=270, labelpad=15)

        ax.set_xticklabels(policy_order)

    fig.suptitle("Experiment 2: outcomes per user type", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment2_fairness(df_fair: pd.DataFrame, out_path: str):
    """
    Create a fairness summary figure for Experiment 2.

    For each policy:
      - Bar: overall mean utility across all users / archetypes
      - Vertical line: min to max utility (across user types)

    This visually encodes both *performance* (mean utility) and
    *fairness* (spread and minimum utility).
    """
    df = df_fair[df_fair["experiment"] == 2].copy()
    policy_order = ["NE", "AE", "NO", "PO", "BA"]

    df = df.set_index("policy").reindex(policy_order).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Bars for overall mean utility
    sns.barplot(
        data=df,
        x="policy",
        y="overall_mean_utility",
        ax=ax,
        errorbar=None,
    )

    # Vertical lines for min–max utility range
    for i, row in df.iterrows():
        ax.vlines(
            x=i,
            ymin=row["min_utility"],
            ymax=row["max_utility"],
            linewidth=1.5,
        )

    ax.set_xlabel("Policy")
    ax.set_ylabel("Overall mean utility")
    ax.set_title("Experiment 2: fairness across user types\n(min–max range per policy)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    """
    Command-line entry point.

    By default, expects the three CSV files in the current directory and
    writes all figures into ./figures. Paths can be overridden via flags.
    """
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

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Load CSVs
    exp1 = pd.read_csv(args.exp1_csv)
    exp2_stats = pd.read_csv(args.exp2_stats_csv)
    exp2_fair = pd.read_csv(args.exp2_fair_csv)

    # Configure global style for all plots
    configure_plot_style()

    # ---------------- Experiment 1 (multiple visualizations) ----------------
    plot_exp1_regret_band(
        exp1,
        out_path=os.path.join(args.out_dir, "exp1_regret_band.pdf"),
    )
    plot_exp1_final_regret_bar(
        exp1,
        out_path=os.path.join(args.out_dir, "exp1_regret_final_bar.pdf"),
    )
    plot_exp1_regret_slope(
        exp1,
        out_path=os.path.join(args.out_dir, "exp1_regret_slope.pdf"),
    )
    plot_exp1_regret_heatmap(
        exp1,
        out_path=os.path.join(args.out_dir, "exp1_regret_heatmap.pdf"),
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
