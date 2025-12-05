# Preference-Aware Robot Explanations: Simulation Framework  
**HRI 2026 Late-Breaking Report – Code Release**

This repository contains the simulation framework used in our HRI 2026 Late-Breaking Report on *preference-aware robot explanations*.  
The framework models explanation decisions as probabilistic choices, learns user preferences via Beta–Bernoulli posteriors, and evaluates multiple explanation policies across diverse user archetypes.

---

## Key Features

- **Probabilistic explanation model**  
  Explanation decisions (whether to explain + how to explain) are treated as actions.

- **Bayesian preference learning**  
  Per-user Beta–Bernoulli updates from binary feedback.

- **Norm–preference mixing (α)**  
  Balances explanatory norms with user preferences.

- **Policies implemented**  
  - NE — Never Explain  
  - AE — Always Explain  
  - NO — Norm-Only  
  - PO — Preference-Only  
  - BA — Bayesian Adaptive  
  - OR — Oracle (for regret)

- **User archetypes**  
  Minimalist, ContextHungry, VisualLearner, NormFollower.

- **Two experiments**  
  - Experiment 1: Single-user adaptation + α sweep  
  - Experiment 2: Population-level fairness

- **Publication-ready PDF figures**

---

## Repository Structure

pref_expl_sim/
├── simulate.py # Main CLI simulation tool
├── visualize_results.py # Figure generation
├── config.py # Configuration and metrics
├── policies.py # Policy implementations
├── users.py # User models + feedback
├── README.md
├── requirements.txt
├── results/ # Auto-created experiment outputs
│ ├── exp1/
│ └── exp2/
└── figs/ # Auto-created figures


---

## Installation

Requires **Python 3.8**.

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python simulate.py --help

python simulate.py \
    --experiment 1 \
    --user-type Minimalist \
    --episodes 50 \
    --alpha-grid 0.0 0.25 0.5 0.75 1.0 \
    --output-dir results/exp1

results/exp1/experiment1_runs.csv

python simulate.py \
    --experiment 2 \
    --episodes 50 \
    --output-dir results/exp2

python visualize_results.py --help

python visualize_results.py \
    --experiment 1 \
    --input results/exp1/experiment1_runs.csv \
    --output-dir figs \
    --user-type Minimalist

python visualize_results.py \
    --experiment 2 \
    --input-stats results/exp2/experiment2_user_type_stats.csv \
    --input-fairness results/exp2/experiment2_fairness.csv \
    --output-dir figs

