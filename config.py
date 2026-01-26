#!/usr/bin/env python3
"""
config.py

Configuration module for the preference-aware explanation simulation.

This file defines:
- Explanation attribute families and their possible values.
- Synthetic user archetypes (Minimalist, ContextHungry, VisualLearner, NormFollower).
- Normative priors over attributes for different response types.
- A helper function to simulate noisy user feedback about explanations.

All typing is Python 3.8 compatible (uses typing.Optional, Dict, List).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import random

EVENT_SALIENCE_PROB = 0.7  # keep existing behavior by default

# --- Reproducible seeding (Python's built-in hash() is randomized per process) ---
import hashlib

SEED_MOD = 2 ** 32

def stable_seed(*parts: object, mod: int = SEED_MOD) -> int:
    """
    Deterministic seed from arbitrary parts (strings, ints, floats, etc.).
    Stable across runs, machines, and Python invocations.
    """
    s = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(s).digest()
    # Use first 8 bytes for a large integer; mod to fit Random seed range
    return int.from_bytes(digest[:8], "big") % mod


# --- Feedback parameters used by both sampler and expected-utility computations ---
P_LIKE_MATCH = 0.85
P_LIKE_MISMATCH = 0.25

def expected_like_prob(theta_pref_one: float, chosen_value: int,
                       p_like_match: float = P_LIKE_MATCH,
                       p_like_mismatch: float = P_LIKE_MISMATCH) -> float:
    """
    Expected P(feedback=1) under the SAME generative process as sample_feedback():

      - latent preferred value v* ~ Bernoulli(theta_pref_one)
      - if chosen matches v* => like with p_like_match else p_like_mismatch
    """
    theta = float(theta_pref_one)
    x = int(chosen_value)

    if x == 1:
        return theta * p_like_match + (1.0 - theta) * p_like_mismatch
    else:
        # x == 0 matches when v* == 0
        return (1.0 - theta) * p_like_match + theta * p_like_mismatch


# ---------------------------------------------------------------------------
# Attribute families and values
# ---------------------------------------------------------------------------

# Names of the binary explanation attribute families considered in the model.
ATTRIBUTE_FAMILIES: List[str] = ["modality", "scope", "detail"]

# Human-readable labels for the 0/1 values of each attribute family.
# Internally, we represent choices as integers {0,1}, but these labels
# make it easier to interpret results or log them for debugging.
ATTRIBUTE_VALUES: Dict[str, List[str]] = {
    "modality": ["text", "text_visual"],  # 0=text, 1=text+visual
    "scope": ["local", "global"],         # 0=local, 1=global
    "detail": ["brief", "detailed"],      # 0=brief, 1=detailed
}

# ---------------------------------------------------------------------------
# Synthetic user archetypes
# ---------------------------------------------------------------------------


@dataclass
class UserType:
    """
    Synthetic user archetype.

    Attributes:
        name: A short identifier for the archetype.
        theta_true: Mapping from attribute family name to a real value in [0,1],
                    interpreted as P(prefer value=1 for this family).
    """
    name: str
    theta_true: Dict[str, float]


# Minimalist: likes brief, local, text-only explanations.
MINIMALIST = UserType(
    name="Minimalist",
    theta_true={
        "modality": 0.2,  # mostly prefers text
        "scope": 0.2,     # mostly prefers local
        "detail": 0.1,    # strongly prefers brief
    },
)

# ContextHungry: likes global, detailed explanations.
CONTEXT_HUNGRY = UserType(
    name="ContextHungry",
    theta_true={
        "modality": 0.6,  # somewhat prefers visual
        "scope": 0.8,     # strongly prefers global
        "detail": 0.9,    # strongly prefers detailed
    },
)

# VisualLearner: strongly cares about visual modality.
VISUAL_LEARNER = UserType(
    name="VisualLearner",
    theta_true={
        "modality": 0.9,  # very strong preference for visual
        "scope": 0.7,
        "detail": 0.6,
    },
)

# NormFollower: roughly aligned with the normative priors we define later.
NORM_FOLLOWER = UserType(
    name="NormFollower",
    theta_true={
        "modality": 0.7,
        "scope": 0.7,
        "detail": 0.7,
    },
)

# List of all archetypes.
ALL_ARCHETYPES: List[UserType] = [
    MINIMALIST,
    CONTEXT_HUNGRY,
    VISUAL_LEARNER,
    NORM_FOLLOWER,
]

# Convenience mapping from name to archetype, used by the CLI.
ARCHETYPE_BY_NAME: Dict[str, UserType] = {u.name: u for u in ALL_ARCHETYPES}

# ---------------------------------------------------------------------------
# Response types and normative priors
# ---------------------------------------------------------------------------

# Types of responses/explanations the robot might provide in an episode.
RESPONSE_TYPES: List[str] = ["status", "why", "what_next"]

# NORM_PHI[f][response_type] = P(value 1) based on normative expectations.
# Example: "why" explanations are often expected to be global + detailed.
NORM_PHI: Dict[str, Dict[str, float]] = {
    "modality": {
        "status": 0.3,
        "why": 0.7,
        "what_next": 0.5,
    },
    "scope": {
        "status": 0.3,
        "why": 0.8,
        "what_next": 0.6,
    },
    "detail": {
        "status": 0.3,
        "why": 0.8,
        "what_next": 0.6,
    },
}

# ---------------------------------------------------------------------------
# Feedback generation
# ---------------------------------------------------------------------------


def sample_feedback(
    user_type: UserType,
    chosen_attrs: Dict[str, int],
    p_like_match: float = 0.85,
    p_like_mismatch: float = 0.25,
    rng: Optional[random.Random] = None,
) -> Dict[str, int]:
    """
    Simulate binary feedback y_{t,f} for each attribute family.

    Intuition:
        - For each family f, sample a "preferred" value v* in {0,1}
          from theta_true[f].
        - If the chosen value matches v*, the user "likes" it (y=1) with
          probability p_like_match.
        - If it does not match, the user "likes" it with probability
          p_like_mismatch (lower).

    This approximates a noisy mapping from latent preferences to
    binarized feedback (e.g., thresholded Likert ratings).

    Args:
        user_type: The synthetic user archetype.
        chosen_attrs: Mapping family -> {0,1} of the explanation chosen.
        p_like_match: P(y=1 | value matches hidden preference).
        p_like_mismatch: P(y=1 | value does not match hidden preference).
        rng: Optional random number generator for reproducibility.

    Returns:
        feedback: Mapping family -> {0,1}, where 1 means "liked".
    """
    if rng is None:
        rng = random

    feedback: Dict[str, int] = {}

    for fam in ATTRIBUTE_FAMILIES:
        theta_true = user_type.theta_true[fam]  # P(prefer value 1)
        # Sample internal "preferred" value for this episode.
        prefer_one = rng.random() < theta_true
        preferred_value = 1 if prefer_one else 0

        chosen_value = chosen_attrs[fam]
        matched = (chosen_value == preferred_value)

        p_like = p_like_match if matched else p_like_mismatch
        feedback[fam] = 1 if rng.random() < p_like else 0

    return feedback
