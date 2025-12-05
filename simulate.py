#!/usr/bin/env python3
"""
simulate.py

Command-line interface and core simulation logic for the preference-aware
explanation framework.

Provides two experiments:
    Experiment 1:
        - Single user archetype.
        - Policies: NE, AE, NO, PO, BA(alpha grid).
        - OraclePolicy for regret computation.
        - Outputs per-episode CSV (experiment1_runs.csv).

    Experiment 2:
        - Mixed population of user archetypes.
        - Policies: NE, AE, NO, PO, BA(alpha=0.5).
        - Outputs per-user-type stats and fairness summary CSVs.

In this patched version:
    - OraclePolicy now chooses between explain vs no-explain.
    - BayesianAdaptivePolicy also chooses between explain vs no-explain
      (using curiosity_penalty via its user_type attribute).
    - simulate_user_run() sets BA.user_type = user_type so BA has access
      to the correct archetype for curiosity_penalty.

All type hints are Python 3.8 compatible.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy
import random

import numpy as np

from config import (
    ATTRIBUTE_FAMILIES,
    ALL_ARCHETYPES,
    ARCHETYPE_BY_NAME,
    UserType,
    sample_feedback,
)
from policies import (
    ExplanationPolicy,
    NoExplainPolicy,
    AlwaysExplainPolicy,
    NormOnlyPolicy,
    PreferenceOnlyPolicy,
    BayesianAdaptivePolicy,
    OraclePolicy,
    EpisodeContext,
    explanation_cost,
    curiosity_penalty,
)
from metrics import EpisodeLog, UserRunLog, PopulationMetrics


# ---------------------------------------------------------------------------
# Core simulation utilities
# ---------------------------------------------------------------------------

def simulate_user_run(
    user_type: UserType,
    policy: ExplanationPolicy,
    n_episodes: int,
    oracle_policy: Optional[ExplanationPolicy] = None,
    seed: Optional[int] = None,
) -> Tuple[UserRunLog, Optional[UserRunLog]]:
    """
    Simulate n_episodes for a single (user_type, policy) pair.

    If oracle_policy is not None, it is used to compute oracle utility
    and regret for each episode.

    Returns:
        (run_log, oracle_log_or_None)
    """
    rng = random.Random(seed)

    # Make copies so that each run uses a fresh policy instance.
    policy = copy.deepcopy(policy)
    policy.reset_user()

    # If this is a BA policy, attach the correct user_type so it can
    # compute curiosity_penalty(explain=False, ...).
    if isinstance(policy, BayesianAdaptivePolicy):
        policy.user_type = user_type

    oracle_log = None
    if oracle_policy is not None:
        oracle_policy = copy.deepcopy(oracle_policy)
        oracle_policy.reset_user()
        oracle_log = UserRunLog(user_type.name, oracle_policy.name)

    run_log = UserRunLog(user_type.name, policy.name)

    for t in range(n_episodes):
        # Random response type; treat all episodes as salient for simplicity.
        response_type = rng.choice(["status", "why", "what_next"])
        ctx = EpisodeContext(response_type=response_type, event_salient=True)

        # ------------------ Actual policy decision ------------------
        explain, attrs = policy.choose(ctx, rng)
        feedback = sample_feedback(user_type, attrs, rng=rng)

        # Approximate alignment by feedback==1 for each family.
        aligned = {fam: feedback[fam] for fam in ATTRIBUTE_FAMILIES}
        expl_cost = explanation_cost(attrs) if explain else 0.0
        cur_pen = curiosity_penalty(explain, ctx, user_type)
        util = sum(feedback.values()) - expl_cost - cur_pen

        ep_log = EpisodeLog(
            explained=explain,
            attrs=attrs,
            feedback=feedback,
            utility=util,
            curiosity=cur_pen,
            explanation_cost=expl_cost,
            aligned=aligned,
            oracle_utility=None,
            regret=None,
        )

        # ------------------ Oracle decision (if present) ------------------
        if oracle_policy is not None:
            exp_o, attrs_o = oracle_policy.choose(ctx, rng)
            feedback_o = sample_feedback(user_type, attrs_o, rng=rng)
            aligned_o = {fam: feedback_o[fam] for fam in ATTRIBUTE_FAMILIES}
            expl_cost_o = explanation_cost(attrs_o) if exp_o else 0.0
            cur_pen_o = curiosity_penalty(exp_o, ctx, user_type)
            util_o = sum(feedback_o.values()) - expl_cost_o - cur_pen_o

            # Store oracle utility and regret in the episode log.
            ep_log.oracle_utility = util_o
            ep_log.regret = util_o - util

            # Also build a separate log for the oracle.
            oracle_log.add(
                EpisodeLog(
                    explained=exp_o,
                    attrs=attrs_o,
                    feedback=feedback_o,
                    utility=util_o,
                    curiosity=cur_pen_o,
                    explanation_cost=expl_cost_o,
                    aligned=aligned_o,
                    oracle_utility=None,
                    regret=None,
                )
            )

            oracle_policy.update_from_feedback(feedback_o)

        run_log.add(ep_log)
        policy.update_from_feedback(feedback)

    return run_log, oracle_log


def simulate_population(
    policies: List[ExplanationPolicy],
    archetypes: List[UserType],
    n_users_per_type: int,
    n_episodes: int,
) -> Dict[str, PopulationMetrics]:
    """
    Simulate a population of users for each policy.

    For each policy and each archetype, simulate n_users_per_type runs,
    each with n_episodes episodes. Returns a dictionary:
        policy_name -> PopulationMetrics
    """
    policy_runs: Dict[str, List[UserRunLog]] = {p.name: [] for p in policies}

    for p in policies:
        for user_type in archetypes:
            for u in range(n_users_per_type):
                seed = hash((p.name, user_type.name, u)) % (2 ** 32)
                run_log, _ = simulate_user_run(
                    user_type=user_type,
                    policy=p,
                    n_episodes=n_episodes,
                    oracle_policy=None,
                    seed=seed,
                )
                policy_runs[p.name].append(run_log)

    return {name: PopulationMetrics(runs) for name, runs in policy_runs.items()}


# ---------------------------------------------------------------------------
# Experiment 1 — Single-user adaptation + alpha grid
# ---------------------------------------------------------------------------

def run_experiment1(
    user_type_name: str,
    episodes: int,
    alpha_grid: List[float],
    output_dir: Path,
) -> None:
    """
    Run Experiment 1:
        - Single user archetype (given by name).
        - Policies: NE, AE, NO, PO, BA(alpha_grid).
        - OraclePolicy baseline.

    Writes:
        output_dir/experiment1_runs.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "experiment1_runs.csv"

    if user_type_name not in ARCHETYPE_BY_NAME:
        raise ValueError(
            "Unknown user type '{}'. Available: {}".format(
                user_type_name, list(ARCHETYPE_BY_NAME.keys())
            )
        )
    user_type = ARCHETYPE_BY_NAME[user_type_name]

    # Population-average parameters for PreferenceOnly baseline.
    pop_theta = {
        fam: float(np.mean([u.theta_true[fam] for u in ALL_ARCHETYPES]))
        for fam in ATTRIBUTE_FAMILIES
    }

    base_policies = [
        NoExplainPolicy(),
        AlwaysExplainPolicy(),
        NormOnlyPolicy(),
        PreferenceOnlyPolicy(pop_theta),
    ]
    ba_policies = [BayesianAdaptivePolicy(alpha_mix=a) for a in alpha_grid]
    oracle = OraclePolicy(user_type=user_type)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        # Header row for per-episode logs.
        writer.writerow([
            "experiment", "user_type", "policy", "alpha_mix",
            "episode", "explained", "modality", "scope", "detail",
            "utility", "oracle_utility", "regret",
            "curiosity", "explanation_cost",
            "feedback_modality", "feedback_scope", "feedback_detail",
        ])

        def log_policy(policy: ExplanationPolicy, alpha: Optional[float]) -> None:
            """Simulate and log one policy run for this user."""
            seed = hash((user_type.name, policy.name, alpha)) % (2 ** 32)
            run_log, _ = simulate_user_run(
                user_type=user_type,
                policy=policy,
                n_episodes=episodes,
                oracle_policy=oracle,
                seed=seed,
            )
            for ep_i, ep in enumerate(run_log.episodes):
                writer.writerow([
                    1,
                    user_type.name,
                    policy.name,
                    alpha if alpha is not None else "",
                    ep_i,
                    int(ep.explained),
                    ep.attrs["modality"],
                    ep.attrs["scope"],
                    ep.attrs["detail"],
                    ep.utility,
                    ep.oracle_utility if ep.oracle_utility is not None else "",
                    ep.regret if ep.regret is not None else "",
                    ep.curiosity,
                    ep.explanation_cost,
                    ep.feedback["modality"],
                    ep.feedback["scope"],
                    ep.feedback["detail"],
                ])

        # Log non-BA baselines.
        for p in base_policies:
            log_policy(p, None)

        # Log BA for each alpha.
        for a in alpha_grid:
            log_policy(BayesianAdaptivePolicy(alpha_mix=a), a)

    print("[Experiment 1] Wrote:", csv_path)


# ---------------------------------------------------------------------------
# Experiment 2 — Population fairness and robustness
# ---------------------------------------------------------------------------

def run_experiment2(
    episodes: int,
    users_per_type: int,
    output_dir: Path,
) -> None:
    """
    Run Experiment 2:
        - Mixed population of archetypes.
        - Policies: NE, AE, NO, PO, BA(alpha=0.5).
        - users_per_type simulated per archetype.

    Writes:
        output_dir/experiment2_user_type_stats.csv
        output_dir/experiment2_fairness.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "experiment2_user_type_stats.csv"
    fair_path = output_dir / "experiment2_fairness.csv"

    pop_theta = {
        fam: float(np.mean([u.theta_true[fam] for u in ALL_ARCHETYPES]))
        for fam in ATTRIBUTE_FAMILIES
    }

    policies = [
        NoExplainPolicy(),
        AlwaysExplainPolicy(),
        NormOnlyPolicy(),
        PreferenceOnlyPolicy(pop_theta),
        BayesianAdaptivePolicy(alpha_mix=0.5),
    ]

    metrics_by_policy = simulate_population(
        policies=policies,
        archetypes=ALL_ARCHETYPES,
        n_users_per_type=users_per_type,
        n_episodes=episodes,
    )

    # Write user-type stats.
    with stats_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "user_type", "policy",
            "mean_utility", "std_utility",
            "mean_alignment", "mean_curiosity", "mean_expl_load",
        ])
        for pname, pm in metrics_by_policy.items():
            stats = pm.per_user_type_stats()
            for (utype, pol), vals in stats.items():
                writer.writerow([
                    2,
                    utype,
                    pol,
                    vals["mean_utility"],
                    vals["std_utility"],
                    vals["mean_alignment"],
                    vals["mean_curiosity"],
                    vals["mean_expl_load"],
                ])

    # Write fairness summary.
    with fair_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "policy",
            "overall_mean_utility", "utility_variance",
            "min_utility", "max_utility",
        ])
        for pname, pm in metrics_by_policy.items():
            fs = pm.fairness_summary()
            writer.writerow([
                2,
                pname,
                fs["overall_mean_utility"],
                fs["utility_variance"],
                fs["min_utility"],
                fs["max_utility"],
            ])

    print("[Experiment 2] Wrote:", stats_path)
    print("[Experiment 2] Wrote:", fair_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simulation experiments for preference-aware explanations."
    )
    parser.add_argument(
        "--experiment", type=int, choices=[1, 2], required=True,
        help="Experiment ID: 1 (single-user + alpha sweep) or 2 (population).",
    )
    parser.add_argument(
        "--user-type", type=str,
        help="User type name (Experiment 1 only).",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of episodes per run.",
    )
    parser.add_argument(
        "--users-per-type", type=int, default=20,
        help="Number of synthetic users per archetype (Experiment 2).",
    )
    parser.add_argument(
        "--alpha-grid", type=float, nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="List of alpha_mix values to test (Experiment 1).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory where CSV outputs are stored.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.experiment == 1:
        if args.user_type is None:
            raise ValueError("--user-type must be provided for experiment 1.")
        run_experiment1(
            user_type_name=args.user_type,
            episodes=args.episodes,
            alpha_grid=args.alpha_grid,
            output_dir=output_dir,
        )
    else:
        run_experiment2(
            episodes=args.episodes,
            users_per_type=args.users_per_type,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
