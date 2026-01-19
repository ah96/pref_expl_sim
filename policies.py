#!/usr/bin/env python3
"""
policies.py

Defines explanation policies and utility helpers.

This file covers:
- EpisodeContext: context for each decision (response_type, event_salient).
- Utility helpers: explanation_cost, curiosity_penalty.
- Policies:
    * ExplanationPolicy (abstract base)
    * NoExplainPolicy (NE)
    * AlwaysExplainPolicy (AE)
    * NormOnlyPolicy (NO)
    * PreferenceOnlyPolicy (PO)
    * BayesianAdaptivePolicy (BA) with Beta-Bernoulli learning and alpha-mixing,
      now considering both "explain" and "no-explain" choices.
    * OraclePolicy (OR) that knows the user's true preferences, also considering
      "explain" vs "no-explain".

All typing is Python 3.8 compatible (Optional, Dict, List).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import random

from config import ATTRIBUTE_FAMILIES, NORM_PHI, UserType


# ---------------------------------------------------------------------------
# Episode context and utility helpers
# ---------------------------------------------------------------------------


@dataclass
class EpisodeContext:
    """
    Context for a single explanation decision.

    Attributes:
        response_type: Type of explanation requested/appropriate
                       (e.g., "status", "why", "what_next").
        event_salient: Whether something important/surprising happened.
                       If True and the robot does NOT explain, there may be
                       a curiosity penalty.
    """
    response_type: str
    event_salient: bool


def explanation_cost(attrs: Dict[str, int]) -> float:
    """
    Simple cost model for explanations.

    Base assumptions:
        - Any explanation costs 1 unit (cognitive/temporal load).
        - Detailed explanations cost an extra 1 unit.

    Args:
        attrs: Mapping family -> {0,1} for the chosen explanation attributes.

    Returns:
        A scalar cost value (float).
    """
    cost = 1.0  # base cost for explaining at all
    if attrs.get("detail", 1) == 1: # this needs to be checked further
        cost += 1.0  # extra cost for detailed explanations
    return cost


def curiosity_penalty(
    explained: bool,
    ctx: EpisodeContext,
    user_type: UserType,
) -> float:
    """
    Penalty for unmet curiosity when no explanation is given.

    Intuition:
        - If no explanation is given and the event is salient, we incur
          a penalty whose magnitude depends on how much the user cares
          about explanations (here: preference for detail).
        - If event is not salient or an explanation is given, penalty is 0.

    Args:
        explained: True if an explanation was given in this episode.
        ctx: EpisodeContext with event_salient flag.
        user_type: Synthetic user archetype, used to scale penalty.

    Returns:
        Curiosity penalty as a non-negative float.
    """
    if explained:
        return 0.0
    if not ctx.event_salient:
        return 0.0

    # Use preference for "detail" as a proxy for explanation importance.
    return 2.0 * user_type.theta_true["detail"]


# ---------------------------------------------------------------------------
# Base policy interface
# ---------------------------------------------------------------------------


class ExplanationPolicy:
    """
    Abstract interface for explanation policies.

    All policies must implement:
        - reset_user()
        - choose(ctx, rng)
        - update_from_feedback(feedback)
    """

    name: str  # short identifier for the policy (e.g., "BA", "NO")

    def reset_user(self) -> None:
        """
        Reset any per-user state (e.g., Beta posteriors when starting
        a new simulation run for a new user).
        """
        pass

    def choose(
        self,
        ctx: EpisodeContext,
        rng: random.Random,
    ) -> (bool, Dict[str, int]):
        """
        Choose whether to explain and with which attributes.

        Args:
            ctx: EpisodeContext describing the situation.
            rng: Random number generator for any stochastic decisions.

        Returns:
            (explain, attrs):
                explain: bool indicating whether an explanation is provided.
                attrs: mapping family -> {0,1} describing explanation attributes.
        """
        raise NotImplementedError

    def update_from_feedback(self, feedback: Dict[str, int]) -> None:
        """
        Update the policy's internal state from user feedback.

        Stateless policies can leave this empty.
        """
        pass


# ---------------------------------------------------------------------------
# Simple baseline policies
# ---------------------------------------------------------------------------


class NoExplainPolicy(ExplanationPolicy):
    """
    NE: Never explains anything (pure lower bound).

    Intended as a baseline that captures "silent" robots.
    """

    def __init__(self) -> None:
        self.name = "NE"

    def choose(self, ctx: EpisodeContext, rng: random.Random):
        # No explanation, attributes irrelevant (use zeros).
        return False, {fam: 0 for fam in ATTRIBUTE_FAMILIES}


class AlwaysExplainPolicy(ExplanationPolicy):
    """
    AE: Always explains using a fixed maximal explanation.

    This corresponds to always visual, global, detailed explanations,
    independent of context or user.
    """

    def __init__(self) -> None:
        self.name = "AE"

    def choose(self, ctx: EpisodeContext, rng: random.Random):
        attrs = {
            "modality": 1,  # text + visual
            "scope": 1,     # global
            "detail": 1,    # detailed
        }
        return True, attrs


class NormOnlyPolicy(ExplanationPolicy):
    """
    NO: Norm-only explanation policy.

    Chooses attribute values based solely on normative priors NORM_PHI
    for the current response type, ignoring user-specific feedback.
    For simplicity, we assume it explains whenever the event is salient.
    """

    def __init__(self) -> None:
        self.name = "NO"

    def choose(self, ctx: EpisodeContext, rng: random.Random):
        explain = ctx.event_salient
        attrs: Dict[str, int] = {}
        for fam in ATTRIBUTE_FAMILIES:
            p_val1 = NORM_PHI[fam][ctx.response_type]
            attrs[fam] = 1 if rng.random() < p_val1 else 0
        return explain, attrs


class PreferenceOnlyPolicy(ExplanationPolicy):
    """
    PO: Population-level preference policy (no per-user adaptation).

    Uses a fixed population-level distribution over attribute values,
    e.g., averaging theta_true across archetypes.
    For simplicity, it also explains whenever the event is salient.
    """

    def __init__(self, population_theta: Dict[str, float]) -> None:
        """
        Args:
            population_theta: mapping family -> P(value=1) across population.
        """
        self.name = "PO"
        self.population_theta = population_theta

    def choose(self, ctx: EpisodeContext, rng: random.Random):
        explain = ctx.event_salient
        attrs: Dict[str, int] = {}
        for fam in ATTRIBUTE_FAMILIES:
            p_val1 = self.population_theta[fam]
            attrs[fam] = 1 if rng.random() < p_val1 else 0
        return explain, attrs


# ---------------------------------------------------------------------------
# Bayesian Adaptive policy (Beta–Bernoulli + α-mixing)
# ---------------------------------------------------------------------------


@dataclass
class BayesianAdaptivePolicy(ExplanationPolicy):
    """
    BA: Bayesian Adaptive explanation policy.

    Maintains Beta(a_f, b_f) posteriors over each attribute family f,
    which represent the user's preference for value=1. These posteriors
    are updated from binary feedback via Beta-Bernoulli conjugacy.

    It also uses a mixing parameter alpha_mix to blend normative priors
    with learned preferences:
        p_mix = alpha_mix * phi_f(response_type) + (1 - alpha_mix) * E[theta_f].

    NEW: The policy now explicitly considers BOTH:
        - explain = False  (incurs curiosity_penalty if event is salient)
        - explain = True   (with all 2^3 attribute configurations)
      and chooses the option with higher expected utility.
    """

    name: str = "BA"
    alpha_mix: float = 0.5
    alpha_beta: Dict[str, List[float]] = field(default_factory=dict)
    # user_type is needed to evaluate curiosity_penalty(explain=False, ...)
    user_type: Optional[UserType] = None

    def __post_init__(self) -> None:
        # Initialize Beta(1,1) for each attribute family if not provided.
        if not self.alpha_beta:
            self.alpha_beta = {fam: [1.0, 1.0] for fam in ATTRIBUTE_FAMILIES}

    def reset_user(self) -> None:
        """
        Reset Beta parameters when starting a new user.

        Note: user_type is set from the outside in simulate_user_run()
        and is not reset here.
        """
        self.alpha_beta = {fam: [1.0, 1.0] for fam in ATTRIBUTE_FAMILIES}

    def posterior_mean(self, fam: str) -> float:
        """Return the posterior mean E[theta_f] = a_f / (a_f + b_f)."""
        a, b = self.alpha_beta[fam]
        return a / (a + b)

    def choose(self, ctx: EpisodeContext, rng: random.Random):
        """
        Choose whether to explain and which attributes to use.

        We evaluate:
            - Option 1: explain=False
            - Option 2: explain=True with all 2^3 attribute combinations

        and return the argmax of expected utility.
        """
        # -------------------------------
        # OPTION 1: Do NOT explain
        # -------------------------------
        if self.user_type is not None:
            score_no_explain = -curiosity_penalty(False, ctx, self.user_type)
        else:
            # Fallback if user_type was not set; assume no curiosity penalty.
            score_no_explain = 0.0

        best_score = score_no_explain
        best_attrs = {fam: 0 for fam in ATTRIBUTE_FAMILIES}
        best_explain_flag = False

        # -------------------------------
        # OPTION 2: Explain with attributes
        # -------------------------------
        for mod_val in (0, 1):
            for scope_val in (0, 1):
                for detail_val in (0, 1):
                    attrs = {
                        "modality": mod_val,
                        "scope": scope_val,
                        "detail": detail_val,
                    }
                    score = self.expected_attr_utility(attrs, ctx)
                    if score > best_score:
                        best_score = score
                        best_attrs = attrs
                        best_explain_flag = True

        return best_explain_flag, best_attrs

    def expected_attr_utility(
        self,
        attrs: Dict[str, int],
        ctx: EpisodeContext,
    ) -> float:
        """
        Expected utility approximation for EXPLAINING with a given
        attribute configuration.

        We approximate:
            reward = sum_f P(user likes chosen value for family f)
            cost   = explanation_cost(attrs)

        and return reward - cost.

        There is no curiosity penalty here because this function is only
        called in the explain=True branch.
        """
        reward = 0.0
        for fam in ATTRIBUTE_FAMILIES:
            # Posterior mean preference for value=1.
            a, b = self.alpha_beta[fam]
            theta_hat = a / (a + b)

            # Normative prior for this family and response type.
            phi = NORM_PHI[fam][ctx.response_type]

            # α-mixing between norm and learned preference.
            p_mix = self.alpha_mix * phi + (1.0 - self.alpha_mix) * theta_hat

            # If we choose value 1, "like" probability is p_mix; else 1 - p_mix.
            if attrs[fam] == 1:
                p_like = p_mix
            else:
                p_like = 1.0 - p_mix

            reward += p_like

        cost = explanation_cost(attrs)
        return reward - cost

    def update_from_feedback(self, feedback: Dict[str, int]) -> None:
        """
        Beta-Bernoulli posterior update.

        For each family f:
            a_f <- a_f + y_f
            b_f <- b_f + (1 - y_f)
        """
        for fam, y in feedback.items():
            a, b = self.alpha_beta[fam]
            self.alpha_beta[fam] = [a + y, b + (1 - y)]


# ---------------------------------------------------------------------------
# Oracle policy (knows true preferences, "explain" vs "no-explain")
# ---------------------------------------------------------------------------


@dataclass
class OraclePolicy(ExplanationPolicy):
    """
    OR: Oracle explanation policy.

    Knows the user's true theta_true[f] parameters and chooses between:
        - explain=False
        - explain=True with any attribute configuration

    to maximise expected utility under the true preferences.
    """

    user_type: UserType
    name: str = "OR"

    def reset_user(self) -> None:
        # Oracle has no learning state.
        pass

    def choose(self, ctx: EpisodeContext, rng: random.Random):
        """
        Evaluate both:
            - explain=False (with curiosity penalty),
            - explain=True (over all attribute configs),
        and return the argmax.
        """
        # -------------------------------
        # OPTION 1: No explanation
        # -------------------------------
        score_no_explain = -curiosity_penalty(False, ctx, self.user_type)
        best_score = score_no_explain
        best_attrs = {fam: 0 for fam in ATTRIBUTE_FAMILIES}
        best_explain_flag = False

        # -------------------------------
        # OPTION 2: Explain with attributes
        # -------------------------------
        for mod_val in (0, 1):
            for scope_val in (0, 1):
                for detail_val in (0, 1):
                    attrs = {
                        "modality": mod_val,
                        "scope": scope_val,
                        "detail": detail_val,
                    }
                    score = self.expected_attr_utility(attrs)
                    if score > best_score:
                        best_score = score
                        best_attrs = attrs
                        best_explain_flag = True

        return best_explain_flag, best_attrs

    def expected_attr_utility(self, attrs: Dict[str, int]) -> float:
        """
        Expected utility when EXPLAIN=True under true parameters.

        We approximate:
            reward = sum_f P(user likes chosen value for family f)
                   = sum_f [theta_true_f if value=1 else (1 - theta_true_f)]
            cost   = explanation_cost(attrs)
        """
        reward = 0.0
        for fam in ATTRIBUTE_FAMILIES:
            theta_true = self.user_type.theta_true[fam]
            if attrs[fam] == 1:
                p_like = theta_true
            else:
                p_like = 1.0 - theta_true
            reward += p_like

        cost = explanation_cost(attrs)
        return reward - cost
