# Stag Hunt Cooperation Task - Computational Modeling

**Bayesian Theory of Mind for Social Coordination**  
Robert D. Hawkins, Stanford University  
November 2025

---

## Repository Structure

```
StagHunt/
├── models/              # Core model implementations
├── fitting/             # Parameter fitting scripts
├── analysis/            # Analysis and hypothesis testing
├── visualization/       # Video generation and plotting
├── tests/              # Pytest test suite (75 tests)
├── inputs/             # Raw trial data (CSV files)
├── enriched_trials/    # Trials with model-based regressors
│
├── stag_hunt.py        # Main entry point
├── CLAUDE.md           # Full modeling documentation
├── NEURAL_REGRESSORS_SUMMARY.md  # Neural analysis guide
└── README.md           # This file
```

---

## Quick Start

### 1. Run the Integrated Model (Recommended)

```bash
python models/hierarchical_model_with_cross_trial_learning.py
```

Generates `integrated_model_lr0.3.csv` with all neural regressors.

### 2. Generate Visualization  

```bash
python visualization/make_video_belief_focus.py
```

Creates `stag_hunt_belief_primary.mp4` showing belief dynamics + cross-trial learning.

### 3. Fit Parameters

```bash
python fitting/fit_model.py --model integrated
```

Optimizes: learning_rate, goal_temperature, execution_temperature, timing_tolerance

See all models: `python fitting/fit_model.py --list`

---

## Key Files

### Models (`models/`)

**Primary:**
- `hierarchical_model_with_cross_trial_learning.py` - **MAIN MODEL** ⭐
- `hierarchical_goal_model.py` - Hierarchical decision model
- `belief_model_decision.py` - Bayesian belief updating
- `model_comparison_framework.py` - Model comparison framework

**Baselines:**
- `distance_with_random_tiebreak.py` - Distance-based baseline

### Fitting (`fitting/`)

- `fit_model.py` - **Unified fitting interface** ⭐
  - Fit any model: `--model integrated|hierarchical|distance|distance_tiebreak`
  - List models: `--list`
  - Save results: `--output results.json`

### Analysis (`analysis/`)

- `analyze_belief_as_primary_regressor.py` - **Why belief is primary** ⭐
- `extract_neural_regressors.py` - Extract neural regressors
- `analyze_cross_trial_learning.py` - Cross-trial dynamics

### Visualization (`visualization/`)

- `make_video_belief_focus.py` - **Main video** ⭐
- `make_video_coordination_focus.py` - Emphasizes P_coord
- `make_video_with_regressors.py` - Shows all regressors

---

## Neural Regressors

### Primary: BELIEF

**Variable:** `p1_belief_p2_stag`, `p2_belief_p1_stag`

**Definition:** Continuous probability [0,1] that partner will cooperate

**Why primary:**
- Pure social prediction (unconfounded)
- Bayesian posterior updated moment-by-moment
- Strong discrimination: Cooperation (0.88) vs. Defection (0.38)

**Neural predictions:**
- TPJ/mPFC: Belief ramping
- vmPFC: Belief × value integration
- Striatum/ACC: Belief prediction errors

### Secondary Regressors

- `p1/p2_cross_trial_expectation` - Slow cross-trial learning
- `p1/p2_EU_stag` - Expected utility of cooperation
- `p1/p2_P_coord` - Coordination probability
- `p1/p2_P_choose_stag` - Choice probabilities

See `NEURAL_REGRESSORS_SUMMARY.md` for details.

---

## Model Architecture

### Integrated Hierarchical Model

**Three cognitive levels:**

1. **Cross-Trial Learning** (slow)
   - Updates: `Expectation_t = (1-α) × Expectation_{t-1} + α × FinalBelief_{t-1}`
   
2. **Within-Trial Beliefs** (fast)
   - Bayesian updates from partner movements
   - Produces primary regressor: `belief`
   
3. **Hierarchical Decisions**
   - Goal selection (strategic)
   - Plan execution (motor)

---

## Key Results

**Belief Dynamics - Cooperation Trial:**
- Start: 0.09, 0.16 (pessimistic)
- End: 0.99, 0.99 (mutual recognition!)
- Change: +0.90, +0.83

**Belief Dynamics - Defection Trials:**
- Mean: 0.38-0.52 (low/uncertain)
- Collapse to ~0.01 when partner defects

---

## Testing

The repository includes a comprehensive pytest test suite with 75 tests.

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test modules
```bash
pytest tests/test_belief_dynamics.py -v          # 11 tests for belief updating
pytest tests/test_model_comparison.py -v         # 8 tests comparing models
pytest tests/test_decision_models.py -v          # 22 tests for decision models
```

### Test coverage
- Belief dynamics and updating (11 tests)
- Model comparison (8 tests)
- Decision model components (22 tests)
- Pure intention modeling (13 tests)
- Diagnostic tests (21 tests)

See `tests/README.md` for detailed documentation.

---

## Documentation

- **README.md** (this file) - Quick reference
- **CLAUDE.md** - Full modeling history
- **NEURAL_REGRESSORS_SUMMARY.md** - Neural analysis guide
- **tests/README.md** - Test suite documentation

---

## Contact

Robert D. Hawkins
Social Interaction Lab, Stanford University
