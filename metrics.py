#!/usr/bin/env python3
"""
metrics.py

Defines logging structures and metric computation for the simulation:

- EpisodeLog: per-episode data (actions, feedback, utilities, regret).
- UserRunLog: all episodes for one (user_type, policy) run.
- PopulationMetrics: aggregated metrics across multiple runs for a policy.

These metrics support:
- Preference alignment,
- Explanation load and curiosity cost,
- Mean utility,
- Regret (when oracle is available),
- Fairness summaries (variance across users, etc.).

Python 3.8 compatible typing is used throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from config import ATTRIBUTE_FAMILIES


@dataclass
class EpisodeLog:
    """
    Container for the outcome of a single episode.

    Attributes:
        explained: True if an explanation was given.
        attrs: Mapping family -> {0,1} for the explanation attributes.
        feedback: Mapping family -> {0,1}, where 1 means user liked it.
        utility: Realized utility u_t for this episode.
        curiosity: Curiosity penalty incurred in this episode.
        explanation_cost: Cost of explanation in this episode.
        aligned: Mapping family -> {0,1} approximating preference alignment.
        oracle_utility: Oracle's utility u_t^OR for this episode, if available.
        regret: u_t^OR - u_t, if oracle_utility is available.
    """
    explained: bool
    attrs: Dict[str, int]
    feedback: Dict[str, int]
    utility: float
    curiosity: float
    explanation_cost: float
    aligned: Dict[str, int]

    oracle_utility: Optional[float] = None
    regret: Optional[float] = None


@dataclass
class UserRunLog:
    """
    Log for a single (user_type, policy) run consisting of many episodes.

    Attributes:
        user_type_name: Name of the user archetype.
        policy_name: Name of the policy used.
        episodes: List of EpisodeLog objects.
    """
    user_type_name: str
    policy_name: str
    episodes: List[EpisodeLog] = field(default_factory=list)

    def add(self, ep: EpisodeLog) -> None:
        """Append an EpisodeLog to this run."""
        self.episodes.append(ep)

    # ------------------------------------------------------------------
    # Metrics over the run
    # ------------------------------------------------------------------

    def preference_alignment(self) -> float:
        """
        Fraction of attributes that were liked/aligned across all episodes.

        We approximate alignment by the binary feedback (y=1 means liked)
        for each attribute family.
        """
        if not self.episodes:
            return 0.0

        matches = 0
        total = 0
        for ep in self.episodes:
            for fam in ATTRIBUTE_FAMILIES:
                matches += ep.aligned[fam]
                total += 1
        return float(matches) / float(total)

    def explanation_load(self) -> float:
        """Mean explanation cost across episodes."""
        if not self.episodes:
            return 0.0
        return float(np.mean([ep.explanation_cost for ep in self.episodes]))

    def curiosity_cost(self) -> float:
        """Mean curiosity penalty across episodes."""
        if not self.episodes:
            return 0.0
        return float(np.mean([ep.curiosity for ep in self.episodes]))

    def mean_utility(self) -> float:
        """Mean utility across episodes."""
        if not self.episodes:
            return 0.0
        return float(np.mean([ep.utility for ep in self.episodes]))

    def mean_regret(self) -> Optional[float]:
        """
        Mean regret across episodes, if oracle-based regret is available.

        Returns:
            Mean regret (float) or None if no regret data is present.
        """
        regrets = [ep.regret for ep in self.episodes if ep.regret is not None]
        if not regrets:
            return None
        return float(np.mean(regrets))


@dataclass
class PopulationMetrics:
    """
    Aggregated metrics for a given policy across many runs (users).

    Attributes:
        runs: List of UserRunLog objects for this policy.
    """
    runs: List[UserRunLog]

    def per_user_type_stats(self) -> Dict[tuple, Dict[str, float]]:
        """
        Compute statistics per (user_type, policy) pair.

        Returns:
            Dictionary mapping (user_type_name, policy_name) to a dict with:
                - mean_utility
                - std_utility
                - mean_alignment
                - mean_curiosity
                - mean_expl_load
        """
        stats: Dict[tuple, Dict[str, float]] = {}
        grouped: Dict[tuple, List[UserRunLog]] = {}

        # Group runs by (user_type, policy).
        for r in self.runs:
            key = (r.user_type_name, r.policy_name)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)

        # Compute metrics for each group.
        for key, group_runs in grouped.items():
            utilities = [r.mean_utility() for r in group_runs]
            aligns = [r.preference_alignment() for r in group_runs]
            curios = [r.curiosity_cost() for r in group_runs]
            loads = [r.explanation_load() for r in group_runs]

            stats[key] = {
                "mean_utility": float(np.mean(utilities)),
                "std_utility": float(np.std(utilities)),
                "mean_alignment": float(np.mean(aligns)),
                "mean_curiosity": float(np.mean(curios)),
                "mean_expl_load": float(np.mean(loads)),
            }

        return stats

    def fairness_summary(self) -> Dict[str, float]:
        """
        Compute fairness-related summary statistics for the policy:

        - overall_mean_utility: mean of mean utilities across all runs.
        - utility_variance: variance of mean utilities across runs.
        - min_utility: minimum mean utility (worst-off user).
        - max_utility: maximum mean utility (best-off user).

        Returns:
            Dictionary with the fields above. If no runs are present,
            returns an empty dictionary.
        """
        if not self.runs:
            return {}

        mean_utils = [r.mean_utility() for r in self.runs]
        return {
            "overall_mean_utility": float(np.mean(mean_utils)),
            "utility_variance": float(np.var(mean_utils)),
            "min_utility": float(np.min(mean_utils)),
            "max_utility": float(np.max(mean_utils)),
        }
