# Stag Hunt Cooperation Task: Computational Modeling

**Robert D. Hawkins**
**Social Interaction Lab, Stanford University**

This repository contains computational models and analysis code for the Stag Hunt cooperation task with iEEG patients. The models implement Bayesian belief updating and utility-based decision-making to understand real-time coordination dynamics.

---

## Quick Start

```python
from stag_hunt import DecisionModel, BeliefModel

# Recommended: Coordinated decision model
decision_model = DecisionModel(
    model_type='coordinated',
    params={'temperature': 3.049, 'timing_tolerance': 0.865, 'action_noise': 10.0}
)

# Recommended: Decision-based belief inference
belief_model = BeliefModel(
    inference_type='decision',
    decision_model=decision_model
)

# Run belief updating on trial
trial_with_beliefs = belief_model.run_trial(trial_data)
```

---

## Repository Structure

### Core Models

| File | Description |
|------|-------------|
| **`stag_hunt.py`** | **Unified API** - Main entrypoint with clean interface for all model variants (recommended) |
| `belief_model_distance.py` | Distance-based belief inference using movement heuristics |
| `belief_model_decision.py` | Decision-based belief inference via inverse inference (recommended) |
| `decision_model_basic.py` | Basic utility model with free weight parameters |
| `decision_model_coordinated.py` | Coordinated model with explicit timing constraints (recommended) |

### Analysis Scripts

| File | Description |
|------|-------------|
| `compare_models_fair.py` | Fair comparison of WITH vs WITHOUT belief models (both independently fit) |
| `fit_coordinated_model.py` | Fit the coordinated decision model with explicit timing |
| `fit_parameters_fast.py` | Fast parameter fitting for basic decision model |
| `fit_parameters.py` | Original parameter fitting (slower) |

### Visualization

| File | Description |
|------|-------------|
| `make_video.py` | **Main visualization** - Creates video with trajectories and belief dynamics |
| `visualize_model_trajectories.py` | Simulates and visualizes model-generated trajectories vs actual |

### Development/Testing

| File | Description |
|------|-------------|
| `test_decision_model.py` | Initial testing of decision model with discrete likelihood |
| `test_action_space.py` | Tests action space granularity (8, 16, 32, 64 directions) |
| `test_continuous_likelihood.py` | Validates continuous likelihood approach |

### Documentation

| File | Description |
|------|-------------|
| `CLAUDE.md` | **Detailed technical documentation** - Full analysis notes, model specifications, results |
| `README.md` | This file - Quick reference and workflow guide |

---

## Recommended Workflow

### 1. Generate Beliefs and Visualization

The main output for neural data analysis is the belief timeseries. Generate it using:

```bash
python make_video.py
```

**What it does:**
- Loads all 12 trials from `inputs/`
- Runs decision-based Bayesian model with fitted parameters
- Generates beliefs about partner intentions at each timestep
- Creates visualization video showing trajectories + belief dynamics

**Output:**
- `stag_hunt_trajectories_with_beliefs.mp4` - Video visualization
- Belief timeseries embedded in trial data (for neural analysis)

**Model Configuration:**
- Uses `model_with_decision.py` for principled inverse inference
- Fitted parameters: temp=3.049, timing_tol=0.865, action_noise=10.0

### 2. Model Comparison and Validation

To verify that beliefs improve predictions:

```bash
python compare_models_fair.py
```

**What it does:**
- Fits decision model WITH beliefs (dynamic from Bayesian model)
- Fits decision model WITHOUT beliefs (beliefs fixed at 1.0)
- Compares log-likelihood, AIC, BIC
- Shows parameter differences

**Expected result:**
- WITH beliefs: LL ≈ -5952
- WITHOUT beliefs: LL ≈ -6033
- Improvement: ~80 log-likelihood points

### 3. Simulate Model Trajectories

To validate that the model produces reasonable behavior:

```bash
python visualize_model_trajectories.py
```

**What it does:**
- Uses fitted decision model to simulate complete trajectories
- Compares simulated paths to actual player movements
- Generates side-by-side visualizations

**Output:**
- `outputs/trajectory_trial{N}.png` - Actual vs simulated paths

---

## Key Results

### Model Comparison

| Model | Parameters | Log-Likelihood | Interpretation |
|-------|-----------|----------------|----------------|
| **Coordinated** (recommended) | 3: temp, timing_tol, noise | -5952.65 | Uses actual payoffs + explicit coordination |
| **Original with beliefs** | 4: w_stag, w_rabbit, temp, noise | -5952.22 | Free weights (w_ratio=1.76 unexplained) |
| **Without beliefs** | 4: w_stag, w_rabbit, temp, noise | -6032.85 | Shows beliefs matter (+80 LL) |

**Key Insight:** The coordinated model achieves identical fit to the flexible-weight model but with:
- **Fewer parameters** (3 vs 4)
- **Actual payoffs** instead of free weights
- **Explicit timing** mechanism explaining why cooperation is difficult

### Fitted Parameters (Coordinated Model)

```python
temperature = 3.049        # Moderately deterministic decisions
timing_tolerance = 0.865   # Players tolerate ~0.86 time units of asynchrony
action_noise = 10.0        # High motor variability (hit upper bound)
```

**Interpretation:**
- Tight timing tolerance (0.86) explains low cooperation rate
- Players make fairly decisive choices (temp~3)
- Substantial motor noise in joystick control

---

## Data Files

### Input Data

- `inputs/stag_hunt_coop_trial*.csv` - 12 trials from August 24, 2024
  - Position data for players, stag, rabbit (~20-40 Hz)
  - Changing reward values (`value` column)
  - Event outcomes (catch events)

### Output Data

- `outputs/trajectory_trial*.png` - Simulated vs actual trajectories
- `stag_hunt_trajectories_with_beliefs.mp4` - Main visualization video

---

## Model Details

### Bayesian Belief Updating

The recommended `model_with_decision.py` implements inverse inference:

```python
# Compute P(observed_movement | partner_intention) using decision model
likelihood_if_stag = decision_model.get_likelihood(
    observed_angle,
    belief_partner_going_for_stag=high
)

likelihood_if_rabbit = decision_model.get_likelihood(
    observed_angle,
    belief_partner_going_for_stag=low
)

# Bayes' rule
belief_new = (likelihood_if_stag * belief_old) / normalizer
```

**Advantages over distance-based model:**
- Principled likelihood computation
- Accounts for coordination constraints
- Uses fitted parameters from actual data
- More realistic uncertainty quantification

### Decision Model

The coordinated decision model (`decision_model_coordinated.py`) computes:

```python
# Utility of moving toward stag
u_stag = stag_value × coordination_prob × gain_toward_stag

# Coordination probability
coordination_prob = belief_partner_stag × timing_alignment

# Timing alignment (Gaussian)
timing_alignment = exp(-0.5 × (time_diff / tolerance)²)

# Action selection (softmax)
P(action) ∝ exp(temperature × utility(action))
```

**Key features:**
- Endogenizes coordination difficulty via timing
- No free weight parameters (uses actual payoffs)
- Naturally explains low cooperation rates

---

## Requirements

```python
numpy
pandas
scipy
matplotlib
```

For video generation:
```bash
ffmpeg  # Required by matplotlib.animation
```

---

## Citation

```
Hawkins, R. D. (2025). Computational modeling of real-time coordination
in the Stag Hunt task. Social Interaction Lab, Stanford University.
```

---

## Notes

- See `CLAUDE.md` for detailed technical documentation
- All models use 8-direction action space for computational efficiency
- Continuous likelihood evaluation prevents discretization artifacts
- Fitted parameters from `compare_models_fair.py` and `fit_coordinated_model.py`

---

## Future Extensions

**Immediate:**
- [ ] Cross-validation on held-out trials
- [ ] Individual parameter fits (Chinese vs US patients)
- [ ] Sensitivity analysis for key parameters

**Short-term:**
- [ ] Integration with iEEG neural data
- [ ] Condition-specific analysis (TI vs CRD, opponent framing)
- [ ] Hierarchical Bayesian parameter estimation

**Long-term:**
- [ ] Level-k strategic reasoning models
- [ ] Value-sensitive belief updating
- [ ] Neural correlates of beliefs and utilities
