# Stag Hunt Cooperation Task: Analysis & Computational Modeling

**Robert D. Hawkins, Stanford University | Updated: November 13, 2025**

---

## Experimental Design

### Task
Stag Hunt coordination game with iEEG patients. Players choose to cooperate (hunt stag together) or defect (hunt rabbit alone).

**Players:** Chinese iEEG patient (P1, red) vs US iEEG patient (P2, yellow)
**Design:** 2 reward versions × 4 opponent framings = 8 conditions, 12 trials/block

**Reward Versions:**
- TI: Rabbit value increases → temptation for defection increases
- CRD: Stag value decreases → cooperative reward decreases

**Opponent Framings:** Computer, Same-country, Different-country, Bonding

**Data:** 20-40 Hz sampling with positions, values, events, timing

---

## Key Behavioral Findings

### Cooperation Rate
- **8.3% cooperation** (1/12 trials: Trial 3 only)
- **91.7% defection** (P1: 5 defections, P2: 6 defections)
- Trial sequence: P2 defects → P1 defects → **COOPERATION** → 9 consecutive defections

### Spatial Structure
- **Initial positions:** P1 at (-198, 198), P2 at (198, -198)
- **Prey positions:** Stag at (198, 198), Rabbit at (-198, -198)
- Creates spatial asymmetry: each player starts closer to different prey

### Trial Dynamics
- **Average duration:** 6.5s (SD: 5.7s)
- **Cooperation trial:** 9.2s (longer, suggests coordination process)
- **Defection detection:** Beliefs drop to 0.01 within 5-7 timesteps

---

## Computational Models

### 1. Mathematical Framework: Bayesian Theory of Mind

**Belief Updating (Bayes' Rule):**
```
b_{t+1} = P(partner→stag | observation) = P(obs | stag) × b_t / Z
```

**Coordination Probability:**
```
P_coord = b_partner × T_align

T_align = exp(-0.5 × (Δt / τ)²)
where Δt = |arrival_time_player - arrival_time_partner|
```

**Decision Model (Coordinated):**
```
U(θ) = V_stag × P_coord × gain_stag(θ) + V_rabbit × gain_rabbit(θ)
P(action) = Softmax(β × U)
```

**Key Parameters:**
- β (temperature): Decision stochasticity (fitted: ~3.0)
- τ (timing_tolerance): Coordination constraint (fitted: 150.0)
- κ (action_noise): Motor noise (fitted: 1.0)

### 2. Planning-Based Belief Model (RECOMMENDED)

**Problem with decision-based inference:** Likelihood ratios ≈ 1.0 when far from targets → beliefs stuck at 0.5

**Solution:** Inverse planning with trajectory integration
- Forward model: Q(s, a | goal) = alignment × urgency
- Inverse model: P(goal | trajectory) via Bayes' rule
- **Key innovation:** Accumulate evidence over sliding window (15 timesteps)

**Results:**
- Cooperation trial: beliefs 0.5 → 0.99 over ~8 timesteps ✓
- Defection trials: beliefs 0.5 → 0.01 within 5-7 timesteps ✓

**Implementation:**
```python
from stag_hunt import BeliefModel

model = BeliefModel(
    inference_type='planning',
    rationality=2.0,
    integration_window=15,
    action_noise=1.0
)
```

### 3. "Imagined We" Framework (Gao et al. 2020)

**Extension:** Adds timing alignment to capture coordination feasibility

**Coordination Decomposition:**
```
P_coord = b_partner × T_align

Requires BOTH:
- Intentional alignment (partner wants to cooperate)
- Temporal alignment (can arrive simultaneously)
```

**Parameters:**
```python
model = BeliefModel(
    inference_type='imagined_we',  # Default
    rationality=2.0,
    integration_window=15,
    timing_tolerance=150.0,  # Critical: allows spatial variation
    action_noise=1.0
)
```

**Parameter Rationale:**
- `timing_tolerance=150.0`: Arena is ~800×800 pixels; 150px tolerance appropriate
- `integration_window=15`: Long enough to infer goals when 500+ pixels from targets
- Previous values (τ=50, window=5) were too restrictive

### 4. Bootstrapping Mechanism

**Problem:** Chicken-and-egg dilemma
- P1 won't commit unless P1 believes P2 will cooperate
- P2 won't commit unless P2 believes P1 will cooperate

**Solution:** Positive feedback loop
1. Small initial movements toward stag
2. Partner observes → belief updates slightly upward
3. Stronger commitment → clearer signals
4. Positive feedback amplifies to convergence

**Why cooperation is rare (8.3%):**
- Both must start moving toward stag (not guaranteed with prior=0.5)
- Sustained commitment required (any deviation breaks loop)
- Spatial alignment required throughout
- Beliefs crash faster than they rise

**Fixed points:**
- (0, 0): Mutual defection (stable)
- (1, 1): Mutual cooperation (stable)
- (0.5, 0.5): Uncertainty (unstable)

---

## Model Comparison

| Model | Inference | Integration | Beliefs Update? | Status |
|-------|-----------|-------------|-----------------|--------|
| Distance-based | Heuristic | Single timestep | Yes (clipped) | Baseline |
| Decision-based | Utility matching | Single timestep | No (stuck at 0.5) | ✗ Failed |
| **Planning-based** | **Inverse planning** | **Trajectory window** | **Yes (smooth)** | **✓ Recommended** |
| **Imagined We** | **Planning + timing** | **Trajectory window** | **Yes + P_coord** | **✓ Default** |

**Performance:**
- Decision model: +204 log-likelihood improvement with beliefs vs without
- Motor noise: κ=1.0 optimal (moderate variability)
- Action space: 16+ directions converge to same fit with continuous likelihood

---

## Visualizations

### 1. Enhanced Video (stag_hunt_trajectories_with_beliefs.mp4)
- 3-panel display: Arena + P1 beliefs + P2 beliefs + P_coord
- 88.4 seconds, 30 fps, 5.1M
- Shows bootstrapping in Trial 3: beliefs 0.5 → 0.99 over ~8 timesteps

### 2. Bootstrapping Analysis (bootstrapping_trial3_analysis.png)
- 4-panel figure: Trajectory + Beliefs + P_coord + Timeline
- Identifies critical decision points:
  - t=4: P1 commits (belief > 0.6)
  - t=49: P2 responds
  - t=7/51: Lock-in (belief > 0.9)
  - t=276: Successful coordination

**Generation:**
```bash
python make_video.py                # ~2-3 min
python visualize_bootstrapping.py   # ~2 sec
```

---

## Code Files

### Core Models
| File | Purpose | Status |
|------|---------|--------|
| `stag_hunt.py` | **Unified model interface** | ✓ Complete |
| `belief_model_planning.py` | **Planning-based model (recommended)** | ✓ Complete |
| `belief_model_imagined_we.py` | **Imagined We model (default)** | ✓ Complete |
| `belief_model_distance.py` | Distance heuristic baseline | ✓ Complete |
| `belief_model_decision.py` | Decision-based (deprecated) | ✗ Doesn't work |
| `decision_model_basic.py` | Basic utility model | ✓ Complete |
| `decision_model_coordinated.py` | Coordinated decision model | ✓ Complete |

### Analysis & Visualization
| File | Purpose | Status |
|------|---------|--------|
| `make_video.py` | **Video with belief overlays** | ✓ Complete |
| `visualize_bootstrapping.py` | Bootstrapping analysis figure | ✓ Complete |
| `fit_parameters.py` | Parameter estimation | ✓ Complete |
| `fit_coordinated_model.py` | Fit coordinated model | ✓ Complete |
| `compare_models_fair.py` | Model comparison | ✓ Complete |

### Documentation
- `PLANNING_MODEL_DESIGN.md`: Planning model framework
- `IMAGINED_WE_DESIGN.md`: Imagined We framework
- `IMAGINED_WE_VISUALIZATIONS.md`: Visualization concepts

---

## Neural Predictions

### 1. Belief Encoding
- Ramping activity as beliefs rise (cooperation trials)
- Sharp drops when partner defects
- Prediction errors for unexpected partner movements

### 2. Coordination Probability
- Motor planning activity reflects P_coord, not just beliefs
- Multiplicative coding: b × T_align
- Reduced activity when timing misaligned

### 3. Bootstrapping Dynamics
- Neural synchrony increases during belief convergence
- Phase locking in cooperation trials
- Desynchronization in defection trials

### 4. Prediction Errors
- Surprise signals for unexpected partner behavior
- Scaled by belief confidence
- Dopaminergic signatures

---

## Suggested Analyses (Full Dataset)

1. **Reward manipulation effects:** Compare TI vs CRD cooperation rates
2. **Opponent framing effects:** Computer vs Same-country vs Different-country vs Bonding
3. **Trajectory-based intention detection:** Commitment timing, critical decision points
4. **Learning effects:** Within/between block patterns, early cooperation → later cooperation?
5. **Neural correlates:** Link belief dynamics to iEEG signals

---

## Key Insights

1. **Beliefs predict behavior:** +204 log-likelihood improvement
2. **Planning > decision inference:** Trajectory integration solves far-from-target problem
3. **Cooperation requires dual alignment:** Intention (beliefs) + execution (timing)
4. **Bootstrapping is gradual:** ~8 timesteps, not instantaneous
5. **Positive feedback is fragile:** Any deviation breaks cooperation loop
6. **Tight timing constraint:** τ=150 pixels (vs arena ~800×800)

---

## Usage Examples

### Run Planning Model
```python
from stag_hunt import BeliefModel
import pandas as pd

# Load trial data
trial = pd.read_csv('inputs/trial_01.csv')

# Run planning model
model = BeliefModel(inference_type='planning')
results = model.run_trial(trial)

# Access beliefs
p1_beliefs = results['p1_belief_p2_stag']
p2_beliefs = results['p2_belief_p1_stag']
```

### Run Imagined We Model
```python
# Default: includes timing alignment
model = BeliefModel(
    inference_type='imagined_we',
    timing_tolerance=150.0
)
results = model.run_trial(trial)

# Access coordination probabilities
p1_coord = results['p1_coord_prob']
p2_coord = results['p2_coord_prob']
```

### Fit Parameters
```bash
python fit_parameters.py           # Basic model
python fit_coordinated_model.py    # Coordinated model
python compare_models_fair.py      # Model comparison (AIC/BIC)
```

---

## References

- Baker, Saxe, & Tenenbaum (2009): Bayesian Theory of Mind as inverse planning
- Gao et al. (2020): "Imagined We" framework for joint goal inference

---

## Technical Notes

- **Data quality:** 20-40 Hz, no missing data, typo fixed (`plater1_y` → `player1_y`)
- **Computation:** <1s for 12 trials, ~2000 timesteps on CPU
- **Numerical stability:** Log-space computations, bounded optimization
- **Missing metadata:** Reward version and opponent framing TBD

---

## Questions for Discussion

1. Which condition do these 12 trials represent?
2. Which opponent framing was used?
3. Was there feedback between trials?
4. Any relevant patient demographics/clinical characteristics?
5. How was deception handled in debriefing?
