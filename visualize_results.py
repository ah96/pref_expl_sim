#!/usr/bin/env python3
"""
visualize_result.py

Generate publication-ready figures for Experiments.
              
How to run:
python3 visualize_results.py \
  --exp2_stats_csv results/exp2/experiment2_user_type_stats.csv \
  --exp2_fair_csv results/exp2/experiment2_fairness.csv \
  --exp3_csv results/exp3/experiment3_learning_curve.csv \
  --out_dir results/figures

python3 visualize_results.py \
  --exp1_csvs \
    results/exp1/experiment1_runs_Minimalist.csv \
    results/exp1/experiment1_runs_ContextHungry.csv \
    results/exp1/experiment1_runs_VisualLearner.csv \
    results/exp1/experiment1_runs_NormFollower.csv \
  --out_dir results/figures

python3 visualize_results.py \
  --exp1_csvs \
    results/exp1/experiment1_runs_Minimalist.csv \
    results/exp1/experiment1_runs_ContextHungry.csv \
    results/exp1/experiment1_runs_VisualLearner.csv \
    results/exp1/experiment1_runs_NormFollower.csv \
  --exp2_stats_csv results/exp2/experiment2_user_type_stats.csv \
  --exp2_fair_csv  results/exp2/experiment2_fairness.csv \
  --exp3_csv       results/exp3/experiment3_learning_curve.csv \
  --out_dir results/figures  
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
    """Paper-friendly seaborn style."""
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
# EXPERIMENT 1 HELPERS
# ======================================================================

def _prepare_exp1_df(df_exp1: pd.DataFrame) -> pd.DataFrame:
    """
    Filter Exp1 data and normalize alpha_mix so that:
      - missing alpha_mix column => created
      - empty strings / NaN => numeric NaN then filled with 0.0 (for non-BA)
      - BA rows keep their numeric alpha
    """
    df = df_exp1[df_exp1["experiment"] == 1].copy()

    if "alpha_mix" not in df.columns:
        df["alpha_mix"] = 0.0

    # Coerce alpha_mix to numeric (handles "", "0.5", NaN, etc.)
    df["alpha_mix"] = pd.to_numeric(df["alpha_mix"], errors="coerce")

    # For non-BA policies alpha_mix is irrelevant; fill NaN with 0.0 to simplify grouping.
    df.loc[df["policy"] != "BA", "alpha_mix"] = 0.0

    return df


def _policy_variant_label(row) -> str:
    """Compact label: NE/AE/NO/PO stay; BA becomes BA-α=..."""
    pol = row["policy"]
    if pol != "BA":
        return pol
    alpha = float(row["alpha_mix"])
    return f"BA-α={alpha:g}"

def load_and_combine_exp1_csvs(paths):
    """
    Load multiple Experiment 1 CSVs (one per archetype) and concatenate them.
    If 'user_type' exists in the CSV, we keep it.
    Otherwise, we infer user_type from the filename (e.g., experiment1_runs_Minimalist.csv).
    """
    dfs = []
    for p in paths:
        df = pd.read_csv(p)

        # Infer user_type from filename if missing
        if "user_type" not in df.columns:
            base = os.path.basename(p)
            # expects ..._Minimalist.csv etc.
            guess = base.replace(".csv", "").split("_")[-1]
            df["user_type"] = guess

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out


# ----------------------------------------------------------------------
# Exp1: BAND PLOT (mean ± std) for selected variants
# ----------------------------------------------------------------------

def plot_exp1_regret_band(df_exp1: pd.DataFrame, out_path: str):
    df = _prepare_exp1_df(df_exp1)

    policies_to_show = ["NE", "AE", "PO"]
    ba_alphas_to_show = [0.0, 0.5, 1.0]

    subset = df[
        (df["policy"].isin(policies_to_show)) |
        ((df["policy"] == "BA") & (df["alpha_mix"].isin(ba_alphas_to_show)))
    ].copy()

    agg = (
        subset.groupby(["policy", "alpha_mix", "episode"])["regret"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["variant"] = agg.apply(_policy_variant_label, axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Seaborn palette for consistent line colors
    variants = list(agg["variant"].unique())
    palette = sns.color_palette(n_colors=len(variants))
    color_map = {v: palette[i] for i, v in enumerate(variants)}

    for variant, sub in agg.groupby("variant"):
        sub = sub.sort_values("episode")
        episodes = sub["episode"].to_numpy()
        mean = sub["mean"].to_numpy()
        std = sub["std"].to_numpy()

        ax.plot(episodes, mean, label=variant, linewidth=1.2, color=color_map[variant])
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color_map[variant])

    ax.set_xlabel("Episode")
    ax.set_ylabel("Regret (oracle utility – actual utility)")
    ax.set_title("Experiment 1: regret over time (mean ± std)")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Exp1: FINAL-EPISODE BAR + DOT (recommended)
# ----------------------------------------------------------------------

def plot_exp1_final_regret_bar(df_exp1: pd.DataFrame, out_path: str):
    df = _prepare_exp1_df(df_exp1)

    final_ep = int(df["episode"].max())
    final_df = df[df["episode"] == final_ep].copy()
    final_df["variant"] = final_df.apply(_policy_variant_label, axis=1)

    grouped = final_df.groupby("variant")["regret"].mean().reset_index()

    baseline_order = ["NE", "AE", "NO", "PO"]
    ba_variants = sorted([v for v in grouped["variant"].unique() if v.startswith("BA-α=")])
    variant_order = [v for v in (baseline_order + ba_variants) if v in grouped["variant"].unique()]

    grouped = grouped.set_index("variant").loc[variant_order].reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        data=grouped,
        x="variant",
        y="regret",
        hue="variant",          # <-- NEW
        dodge=False,            # <-- NEW (avoid side-by-side bars)
        ax=ax,
        palette="deep",
        errorbar=None,
        legend=False,           # <-- NEW
    )

    ax.scatter(
        x=np.arange(len(grouped)),
        y=grouped["regret"].to_numpy(),
        color="black",
        zorder=5,
        s=20,
    )

    ax.set_xlabel("Policy / BA variant")
    ax.set_ylabel("Final-episode regret")
    ax.set_title("Experiment 1: final-episode regret per variant")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Exp1: SLOPE CHART (start vs end)
# ----------------------------------------------------------------------

def plot_exp1_regret_slope(df_exp1: pd.DataFrame, out_path: str):
    df = _prepare_exp1_df(df_exp1)

    start_ep = int(df["episode"].min())
    final_ep = int(df["episode"].max())

    df_start = df[df["episode"] == start_ep].copy()
    df_final = df[df["episode"] == final_ep].copy()

    start_agg = (
        df_start.groupby(["policy", "alpha_mix"])["regret"]
        .mean().reset_index()
        .rename(columns={"regret": "regret_start"})
    )
    final_agg = (
        df_final.groupby(["policy", "alpha_mix"])["regret"]
        .mean().reset_index()
        .rename(columns={"regret": "regret_final"})
    )

    merged = start_agg.merge(final_agg, on=["policy", "alpha_mix"])
    merged["variant"] = merged.apply(_policy_variant_label, axis=1)

    baseline_order = ["NE", "AE", "NO", "PO"]
    ba_variants = sorted([v for v in merged["variant"].unique() if v.startswith("BA-α=")])
    variant_order = [v for v in (baseline_order + ba_variants) if v in merged["variant"].unique()]
    merged = merged.set_index("variant").loc[variant_order].reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    palette = sns.color_palette(n_colors=len(merged))
    for i, row in merged.iterrows():
        ax.plot(
            [0, 1],
            [row["regret_start"], row["regret_final"]],
            marker="o",
            linewidth=1.5,
            color=palette[i],
            label=row["variant"],
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Start", "End"])
    ax.set_ylabel("Mean regret")
    ax.set_title("Experiment 1: start vs end regret per variant")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Exp1: HEATMAP (binned regret over time)
# ----------------------------------------------------------------------

def plot_exp1_regret_heatmap(df_exp1: pd.DataFrame, out_path: str, n_bins: int = 5):
    df = _prepare_exp1_df(df_exp1)
    df["variant"] = df.apply(_policy_variant_label, axis=1)

    ep_min, ep_max = int(df["episode"].min()), int(df["episode"].max())
    bins = np.linspace(ep_min, ep_max + 1, n_bins + 1)

    df["episode_bin"] = pd.cut(
        df["episode"],
        bins=bins,
        include_lowest=True,
        labels=[f"{int(bins[i])}–{int(bins[i + 1] - 1)}" for i in range(n_bins)],
    )

    heat = (
        df.groupby(["variant", "episode_bin"], observed=False)["regret"]
        .mean()
        .reset_index()
    )
    heat_pivot = heat.pivot(index="variant", columns="episode_bin", values="regret")

    baseline_order = ["NE", "AE", "NO", "PO"]
    ba_variants = sorted([v for v in heat_pivot.index if v.startswith("BA-α=")])
    idx_order = [v for v in (baseline_order + ba_variants) if v in heat_pivot.index]
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


# ----------------------------------------------------------------------
# Exp1: 2x2 final-regret bars
# ----------------------------------------------------------------------

def plot_exp1_all_archetypes_final_regret_bar(df_exp1_all: pd.DataFrame, out_path: str):
    """
    One compact 2×2 plot: final-episode regret per policy variant, per archetype.

    - Facets: user_type (Minimalist, ContextHungry, VisualLearner, NormFollower)
    - X: variant (NE/AE/NO/PO/BA-α=...)
    - Y: final regret (mean)
    """
    df = df_exp1_all.copy()

    # Ensure consistent Exp1 formatting
    df = df[df["experiment"] == 1].copy()
    df = _prepare_exp1_df(df)

    # Final episode only
    final_ep = int(df["episode"].max())
    df = df[df["episode"] == final_ep].copy()

    # Create variant labels (BA-α=...)
    df["variant"] = df.apply(_policy_variant_label, axis=1)

    user_types_order = ["Minimalist", "ContextHungry", "VisualLearner", "NormFollower"]
    baseline_order = ["NE", "AE", "NO", "PO"]
    ba_variants = sorted([v for v in df["variant"].unique() if v.startswith("BA-α=")])
    variant_order = [v for v in (baseline_order + ba_variants) if v in df["variant"].unique()]

    df["user_type"] = pd.Categorical(df["user_type"], categories=user_types_order, ordered=True)
    df["variant"] = pd.Categorical(df["variant"], categories=variant_order, ordered=True)

    # Aggregate across runs if multiple exist
    agg = (
        df.groupby(["user_type", "variant"], observed=False)["regret"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # Standard error for nicer error bars
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))

    g = sns.catplot(
        data=agg,
        x="variant",
        y="mean",
        hue="variant",  
        col="user_type",
        col_wrap=2,
        kind="bar",
        height=3.2,
        aspect=1.4,
        palette="deep",
        errorbar=None,
        sharey=True,
        legend=False, 
    )

    for ax, (ut, sub) in zip(g.axes.flatten(), agg.groupby("user_type", observed=True)):
        sub = sub.sort_values("variant")
        ax.errorbar(
            x=np.arange(len(sub)),
            y=sub["mean"].to_numpy(),
            yerr=sub["sem"].to_numpy(),
            fmt="none",
            capsize=3,
            linewidth=1.0,
            color="black",
        )
        ax.tick_params(axis="x", rotation=45)

    g.set_axis_labels("Policy / BA variant", "Final-episode regret")
    g.set_titles("{col_name}")
    g.fig.suptitle("Experiment 1 (all archetypes): final-episode regret", y=1.02)

    g.fig.tight_layout()
    g.fig.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)



# ======================================================================
# EXPERIMENT 2
# ======================================================================

def plot_experiment2_user_type_outcomes(df_stats: pd.DataFrame, out_path: str):
    df = df_stats[df_stats["experiment"] == 2].copy()

    user_types_order = ["Minimalist", "ContextHungry", "VisualLearner", "NormFollower"]
    policy_order = ["NE", "AE", "NO", "PO", "BA"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
    axes = axes.flatten()

    for i, user_type in enumerate(user_types_order):
        ax = axes[i]
        sub = df[df["user_type"] == user_type].copy()

        # Reindex but drop policies that don't exist in the CSV (prevents NaNs from breaking error bars)
        sub = sub.set_index("policy").reindex(policy_order).reset_index()
        sub = sub.dropna(subset=["mean_utility", "std_utility", "mean_alignment"])

        sns.barplot(
            data=sub,
            x="policy",
            y="mean_utility",
            hue="policy",           # <-- NEW
            dodge=False,            # <-- NEW
            ax=ax,
            errorbar=None,
            palette="deep",
            legend=False,           # <-- NEW
        )

        ax.errorbar(
            x=np.arange(len(sub)),
            y=sub["mean_utility"].to_numpy(),
            yerr=sub["std_utility"].to_numpy(),
            fmt="none",
            capsize=4,
            linewidth=1.0,
            color="black",
        )

        ax.set_title(user_type)
        ax.set_xlabel("Policy")
        ax.set_ylabel("Mean utility")

        ax2 = ax.twinx()
        ax2.plot(
            np.arange(len(sub)),
            sub["mean_alignment"].to_numpy(),
            marker="o",
            linewidth=1.0,
            color="black",
        )
        ax2.set_ylabel("Mean alignment", rotation=270, labelpad=15)

    fig.suptitle("Experiment 2: outcomes per user type", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment2_fairness(df_fair: pd.DataFrame, out_path: str):
    df = df_fair[df_fair["experiment"] == 2].copy()
    policy_order = ["NE", "AE", "NO", "PO", "BA"]

    df = df.set_index("policy").reindex(policy_order).reset_index()
    df = df.dropna(subset=["overall_mean_utility", "min_utility", "max_utility"])

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(
        data=df,
        x="policy",
        y="overall_mean_utility",
        hue="policy",           # <-- NEW
        dodge=False,            # <-- NEW
        ax=ax,
        errorbar=None,
        palette="deep",
        legend=False,           # <-- NEW
    )

    for i, row in df.iterrows():
        ax.vlines(
            x=i,
            ymin=row["min_utility"],
            ymax=row["max_utility"],
            linewidth=1.5,
            color="black",
        )

    ax.set_xlabel("Policy")
    ax.set_ylabel("Overall mean utility")
    ax.set_title("Experiment 2: fairness across user types\n(min–max range per policy)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# EXPERIMENT 3
# ======================================================================

def plot_experiment3_learning_curve(df_exp3: pd.DataFrame, out_path: str):
    """
    Plot final-episode regret vs. episode budget N (learning curve).

    Expects long-form CSV from simulate.py Exp3:
      experiment, user_type, policy, alpha_mix, episodes, rep, final_regret
    """
    df = df_exp3[df_exp3["experiment"] == 3].copy()

    # Robust numeric conversion
    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    df["final_regret"] = pd.to_numeric(df["final_regret"], errors="coerce")
    df = df.dropna(subset=["episodes", "final_regret", "user_type", "policy"])

    # Sort for nicer lines and consistent x-ticks
    df = df.sort_values(["user_type", "policy", "episodes"])

    user_types_order = ["Minimalist", "ContextHungry", "VisualLearner", "NormFollower"]
    policy_order = ["NE", "AE", "NO", "PO", "BA"]

    df["user_type"] = pd.Categorical(df["user_type"], categories=user_types_order, ordered=True)
    df["policy"] = pd.Categorical(df["policy"], categories=policy_order, ordered=True)

    # Use seaborn relplot for faceting; CI bands come "for free" from repeated reps.
    g = sns.relplot(
        data=df,
        x="episodes",
        y="final_regret",
        hue="policy",
        style="policy",
        col="user_type",
        col_wrap=2,
        kind="line",
        markers=True,
        dashes=False,
        errorbar=("ci", 95),
        facet_kws={"sharex": True, "sharey": True},
        height=3.2,
        aspect=1.35,
        palette="deep",
    )

    g.set_axis_labels("Episode budget N", "Final-episode regret")
    g.set_titles("{col_name}")

    # Make x ticks exactly at the N grid values present
    xs = sorted(df["episodes"].unique())
    for ax in g.axes.flatten():
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x)) for x in xs])

    g.fig.suptitle("Experiment 3: learning curves (final regret vs episode budget)", y=1.02)
    g.fig.tight_layout()
    g.fig.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)

    print("\n Experiment 3 visualization DONE!")



# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate figures for Experiment 1 and 2.")
    parser.add_argument("--exp1_csv", type=str, default="results/exp1/experiment1_runs.csv")
    parser.add_argument("--exp1_csvs", type=str, nargs="+", default=None,
                    help="Optional: multiple Exp1 CSVs (one per archetype). If provided, we combine and plot all archetypes.")
    parser.add_argument("--exp2_stats_csv", type=str, default="results/exp2/experiment2_user_type_stats.csv")
    parser.add_argument("--exp2_fair_csv", type=str, default="results/exp2/experiment2_fairness.csv")
    parser.add_argument("--exp3_csv", type=str, default="results/exp3/experiment3_learning_curve.csv")
    parser.add_argument("--out_dir", type=str, default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    configure_plot_style()

    # Exp1
    try:
        if args.exp1_csvs is not None:
            exp1_all = load_and_combine_exp1_csvs(args.exp1_csvs)
            plot_exp1_all_archetypes_final_regret_bar(
                exp1_all,
                os.path.join(args.out_dir, "exp1_all_archetypes_final_regret.pdf"),
            )
        else:
            exp1 = pd.read_csv(args.exp1_csv)
            plot_exp1_regret_band(exp1, os.path.join(args.out_dir, "exp1_regret_band.pdf"))
            plot_exp1_final_regret_bar(exp1, os.path.join(args.out_dir, "exp1_regret_final_bar.pdf"))
            plot_exp1_regret_slope(exp1, os.path.join(args.out_dir, "exp1_regret_slope.pdf"))
            plot_exp1_regret_heatmap(exp1, os.path.join(args.out_dir, "exp1_regret_heatmap.pdf"))
    except Exception as e:
        print(f"[Experiment 1] Skipped/failed: {e}")


    # Exp2 (optional if CSVs exist)
    try:
        exp2_stats = pd.read_csv(args.exp2_stats_csv)
        exp2_fair = pd.read_csv(args.exp2_fair_csv)
        plot_experiment2_user_type_outcomes(exp2_stats, os.path.join(args.out_dir, "exp2_user_type_outcomes.pdf"))
        plot_experiment2_fairness(exp2_fair, os.path.join(args.out_dir, "exp2_fairness.pdf"))
    except:
        print(f"[Experiment 2] Skipped: missing file(s): {args.exp2_stats_csv} or {args.exp2_fair_csv}")

    # Exp3 (optional if CSV exists)
    try:
        exp3 = pd.read_csv(args.exp3_csv)
        plot_experiment3_learning_curve(exp3, os.path.join(args.out_dir, "exp3_learning_curve_final_regret.pdf"))
    except:
        print(f"[Experiment 3] Skipped: file not found: {args.exp3_csv}")



if __name__ == "__main__":
    main()
