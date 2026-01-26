# Preference-Aware Robot Explanations: Simulation Framework
**HRI 2026 Late-Breaking Report – Camera-Ready Code Release**

This repository contains the *final, camera-ready* simulation framework accompanying the HRI 2026 Late-Breaking Report:

> **Simulating Preference-Aware Robot Explanations: A Probabilistic Perspective on When and How to Explain**

The framework models *when* and *how* a robot should explain its behavior under heterogeneous user preferences, normative expectations, and interaction costs. It is fully self-contained, deterministic (seeded), and designed for reproducibility and clarity.

---

## Overview

The core idea is to treat explanation generation as a **sequential decision-making problem under uncertainty**:

- At each episode, the robot observes a situation (response type + salience).
- It decides **whether to explain** and **which explanation attributes** to use.
- Users provide noisy binary feedback reflecting latent preferences.
- A Bayesian Adaptive (BA) policy learns user preferences online and balances them against norms.

The framework evaluates this trade-off using **utility**, **preference alignment**, **explanation cost**, **curiosity penalties**, and **oracle-based regret**.

---

## Implemented Explanation Attributes

Each explanation is parameterized by three **binary attribute families**:

| Attribute | 0 | 1 |
|---------|---|---|
| Modality | Text | Text + Visual |
| Scope | Local | Global |
| Detail | Brief | Detailed |

---

## User Archetypes

- Minimalist  
- ContextHungry  
- VisualLearner  
- NormFollower  

Defined in `config.py`.

---

## Explanation Policies

NE, AE, NO, PO, BA, OR (oracle for regret).

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Experiments

See paper Section 4. Commands mirror the camera-ready results.

---

## Citation

```bibtex
@inproceedings{halilovic2026preference,
  title={Simulating Preference-Aware Robot Explanations: A Probabilistic Perspective on When and How to Explain},
  author={Halilovic, Amar and Krivic, Senka},
  booktitle={Companion of the 2026 ACM/IEEE International Conference on Human-Robot Interaction},
  year={2026}
}
```
