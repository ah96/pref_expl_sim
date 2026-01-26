#!/usr/bin/env python3
"""
simulate.py

Command-line interface and core simulation logic for the preference-aware
explanation framework.

How to run (experiment 1):
python3 simulate.py --experiment 1 --user-type Minimalist --episodes 100 --alpha-grid 0 0.25 0.5 0.75 1 --output-dir results/exp1

How to run (experiment 2):
python3 simulate.py --experiment 2 --episodes 50 --users-per-type 20 --output-dir results/exp2

How to run (experiment 3):
python3 simulate.py --experiment 3   --episode-grid 10 25 50 100   --reps 30   --ba-alpha 0.5   --output-dir results/exp3
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
    stable_seed,
    expected_like_prob,
    P_LIKE_MATCH,
    P_LIKE_MISMATCH,
    RESPONSE_TYPES,
    EVENT_SALIENCE_PROB,
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

    If oracle_policy is not None, it is used to compute oracle expected utility
    and regret for each episode. (We keep oracle expected utility noise-free so
    regret is not distorted by sampling variance.)
    """
    rng_main = random.Random(seed)
    # Separate RNG so oracle choices don't perturb the policy RNG stream.
    rng_oracle = random.Random(stable_seed("oracle_rng", seed))

    # Fresh copy of policy per run
    policy = copy.deepcopy(policy)
    policy.reset_user()

    # BA needs user_type for curiosity penalty when choosing silence
    if isinstance(policy, BayesianAdaptivePolicy):
        policy.user_type = user_type

    oracle_log = None
    if oracle_policy is not None:
        oracle_policy = copy.deepcopy(oracle_policy)
        oracle_policy.reset_user()
        oracle_log = UserRunLog(user_type.name, oracle_policy.name)

    run_log = UserRunLog(user_type.name, policy.name)

    for _t in range(n_episodes):
        response_type = rng_main.choice(RESPONSE_TYPES)
        ctx = EpisodeContext(
            response_type=response_type,
            event_salient=(rng_main.random() < EVENT_SALIENCE_PROB),
        )

        # ------------------ Actual policy decision ------------------
        explain, attrs = policy.choose(ctx, rng_main)

        if explain:
            feedback = sample_feedback(user_type, attrs, rng=rng_main)
        else:
            # No explanation => no attribute feedback signal.
            feedback = {fam: 0 for fam in ATTRIBUTE_FAMILIES}

        # Approximate alignment by feedback==1 for each family.
        aligned = {fam: feedback[fam] for fam in ATTRIBUTE_FAMILIES}

        expl_cost = explanation_cost(attrs) if explain else 0.0
        cur_pen = curiosity_penalty(explain, ctx, user_type)
        util_sampled = sum(feedback.values()) - expl_cost - cur_pen

        # Expected (noise-free) utility of the chosen action under true preferences
        if explain:
            exp_reward = 0.0
            for fam in ATTRIBUTE_FAMILIES:
                exp_reward += expected_like_prob(
                    user_type.theta_true[fam],
                    int(attrs[fam]),
                    P_LIKE_MATCH,
                    P_LIKE_MISMATCH,
                )
            exp_util = exp_reward - explanation_cost(attrs) - curiosity_penalty(True, ctx, user_type)
        else:
            # No explanation => no attribute-based reward
            exp_util = -curiosity_penalty(False, ctx, user_type)

        ep_log = EpisodeLog(
            explained=explain,
            attrs=attrs,
            feedback=feedback,
            utility=util_sampled,
            curiosity=cur_pen,
            explanation_cost=expl_cost,
            aligned=aligned,
            oracle_utility=None,
            regret=None,
        )

        # ------------------ Oracle decision (if present) ------------------
        if oracle_policy is not None:
            # CRITICAL FIX:
            # Call choose() exactly once. The old code called choose() twice and mixed
            # feedback/attrs, causing inconsistent oracle computations and negative regret.
            exp_o, attrs_o = oracle_policy.choose(ctx, rng_oracle)

            # (Optional) sampled oracle utility for the oracle log
            if exp_o:
                feedback_o = sample_feedback(user_type, attrs_o, rng=rng_oracle)
            else:
                feedback_o = {fam: 0 for fam in ATTRIBUTE_FAMILIES}
            aligned_o = {fam: feedback_o[fam] for fam in ATTRIBUTE_FAMILIES}
            expl_cost_o = explanation_cost(attrs_o) if exp_o else 0.0
            cur_pen_o = curiosity_penalty(exp_o, ctx, user_type)
            util_o_sampled = sum(feedback_o.values()) - expl_cost_o - cur_pen_o

            # Expected oracle utility (noise-free)
            if exp_o:
                exp_reward_o = 0.0
                for fam in ATTRIBUTE_FAMILIES:
                    exp_reward_o += expected_like_prob(
                        user_type.theta_true[fam],
                        int(attrs_o[fam]),
                        P_LIKE_MATCH,
                        P_LIKE_MISMATCH,
                    )
                exp_util_o = exp_reward_o - explanation_cost(attrs_o) - curiosity_penalty(True, ctx, user_type)
            else:
                exp_util_o = -curiosity_penalty(False, ctx, user_type)

            ep_log.oracle_utility = exp_util_o
            ep_log.regret = exp_util_o - exp_util

            oracle_log.add(
                EpisodeLog(
                    explained=exp_o,
                    attrs=attrs_o,
                    feedback=feedback_o,
                    utility=util_o_sampled,
                    curiosity=cur_pen_o,
                    explanation_cost=expl_cost_o,
                    aligned=aligned_o,
                    oracle_utility=None,
                    regret=None,
                )
            )

        run_log.add(ep_log)

        # Learn only when the policy explained (matches your original intent),
        # but now pass the chosen attrs so BA updates the right direction.
        if explain:
            policy.update_from_feedback(feedback, attrs)

    return run_log, oracle_log


def simulate_population(
    policies: List[ExplanationPolicy],
    archetypes: List[UserType],
    n_users_per_type: int,
    n_episodes: int,
) -> Dict[str, PopulationMetrics]:
    """Simulate a population of users for each policy."""
    policy_runs: Dict[str, List[UserRunLog]] = {p.name: [] for p in policies}

    for p in policies:
        for user_type in archetypes:
            for u in range(n_users_per_type):
                seed = stable_seed("exp2", p.name, user_type.name, u)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / ''.join(["experiment1_runs_", user_type_name, ".csv"]) #"experiment1_runs" + user_type_name + ".csv"

    if user_type_name not in ARCHETYPE_BY_NAME:
        raise ValueError(
            f"Unknown user type '{user_type_name}'. Available: {list(ARCHETYPE_BY_NAME.keys())}"
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
    oracle = OraclePolicy(user_type=user_type)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "user_type", "policy", "alpha_mix",
            "episode", "explained", "modality", "scope", "detail",
            "utility", "oracle_utility", "regret",
            "curiosity", "explanation_cost",
            "feedback_modality", "feedback_scope", "feedback_detail",
        ])

        def log_policy(policy: ExplanationPolicy, alpha: Optional[float]) -> None:
            seed = stable_seed("exp1", user_type.name, policy.name, alpha)
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

        for p in base_policies:
            log_policy(p, None)

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
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "experiment2_user_type_stats.csv"

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

    with stats_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "user_type", "policy",
            "mean_utility", "std_utility",
            "mean_alignment", "mean_curiosity", "mean_expl_load",
        ])
        for _pname, pm in metrics_by_policy.items():
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

    print("[Experiment 2] Wrote:", stats_path)


# ---------------------------------------------------------------------------
# Experiment 3 — Learning curve over episode budget N (ALL archetypes)
# ---------------------------------------------------------------------------

def run_experiment3(
    episode_grid: List[int],
    reps: int,
    ba_alpha: float,
    output_dir: Path,
) -> None:
    """
    For each archetype, policy, and episode budget N, run `reps` independent runs and
    record FINAL-EPISODE regret.

    Output (long-form) CSV: experiment3_learning_curve.csv
      columns:
        experiment, user_type, policy, alpha_mix, episodes, rep, final_regret
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "experiment3_learning_curve.csv"

    # Population-average parameters for PreferenceOnly baseline.
    pop_theta = {
        fam: float(np.mean([u.theta_true[fam] for u in ALL_ARCHETYPES]))
        for fam in ATTRIBUTE_FAMILIES
    }

    policies: List[Tuple[ExplanationPolicy, Optional[float]]] = [
        (NoExplainPolicy(), None),
        (AlwaysExplainPolicy(), None),
        (NormOnlyPolicy(), None),
        (PreferenceOnlyPolicy(pop_theta), None),
        (BayesianAdaptivePolicy(alpha_mix=ba_alpha), ba_alpha),
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "user_type", "policy", "alpha_mix", "episodes", "rep", "final_regret"
        ])

        for user_type in ALL_ARCHETYPES:
            oracle = OraclePolicy(user_type=user_type)

            for (policy_obj, alpha) in policies:
                for N in episode_grid:
                    if N <= 0:
                        raise ValueError(f"Episode budgets must be positive; got N={N}")

                    for rep in range(reps):
                        seed = stable_seed("exp3", user_type.name, policy_obj.name, alpha, N, rep)

                        run_log, _ = simulate_user_run(
                            user_type=user_type,
                            policy=policy_obj,
                            n_episodes=N,
                            oracle_policy=oracle,
                            seed=seed,
                        )

                        # Final episode regret (oracle - policy). Should be >= 0 up to float noise.
                        final_regret = float(run_log.episodes[-1].regret)

                        writer.writerow([
                            3,
                            user_type.name,
                            policy_obj.name,
                            alpha if alpha is not None else "",
                            N,
                            rep,
                            final_regret,
                        ])

    print("[Experiment 3] Wrote:", csv_path)



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simulation experiments for preference-aware explanations."
    )
    parser.add_argument(
        "--experiment", type=int, choices=[1, 2, 3, 4], required=True,
        help="Experiment ID: 1 (single-user + alpha sweep), 2 (population), 3 (learning curve over N), 4 (alpha sweep all archetypes).",
    )
    parser.add_argument(
        "--user-type", type=str,
        help="User type name (Experiment 1 only).",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of episodes per run (Experiments 1 & 2).",
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
    # Experiment 3 args
    parser.add_argument(
        "--episode-grid", type=int, nargs="+",
        default=[10, 25, 50, 100],
        help="Episode budgets N for Experiment 3 (e.g., 10 25 50 100).",
    )
    parser.add_argument(
        "--reps", type=int, default=30,
        help="Number of independent repetitions per (user_type, policy, N) in Experiment 3.",
    )
    parser.add_argument(
        "--ba-alpha", type=float, default=0.5,
        help="Fixed alpha_mix for BA in Experiment 3.",
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
    elif args.experiment == 2:
        run_experiment2(
            episodes=args.episodes,
            users_per_type=args.users_per_type,
            output_dir=output_dir,
        )
    elif args.experiment == 3:
        run_experiment3(
            episode_grid=args.episode_grid,
            reps=args.reps,
            ba_alpha=args.ba_alpha,
            output_dir=output_dir,
        )



if __name__ == "__main__":
    main()