# Stag Hunt Cooperation Task: Initial Behavioral Data Analysis

**Robert D. Hawkins**  
**Social Interaction Lab, Stanford University**  
**November 12, 2025**

---

## Overview

This document summarizes the initial analysis of behavioral data from a Stag Hunt cooperation task involving iEEG patients from China and the US. The dataset represents one block of 12 trials collected on August 24, 2024.

## Experimental Design

### Task Structure
The Stag Hunt is a classic coordination game where two players must decide whether to cooperate (hunt the stag together) or defect (hunt the rabbit alone). The game creates a tension between the potentially higher reward from cooperation and the safety of individual action.

**Players:**
- Player 1: Chinese iEEG patient (represented as red ball in the task)
- Player 2: US iEEG patient (represented as yellow ball in the task)

**Important Design Feature:** Unbeknownst to the patients, they are actually playing against the same iEEG patient in China across all blocks, regardless of what they are told about their opponent.

### Experimental Manipulations

**2 Reward Versions** (manipulations to decrease cooperation motive):
1. **Rabbit-value-increasing (TI version):** The amount of rewards for catching the rabbit is increasing during the trial â†’ **Temptation for defection Increases**
2. **Stag-value-decreasing (CRD version):** The amount of rewards for catching the stag is decreasing during the trial â†’ **Cooperative Reward Decreases**

**4 Behavior Sessions** (opponent framing):
1. **Computer:** Subjects believe that they are playing with a computer
2. **Same-country:** Subjects believe that they are playing with a same-country player
3. **Different-country:** Subjects play with a different-country player  
4. **Bonding:** Subjects play with a different-country player after a social bonding session where they have an online interaction and social bonding

**Design Structure:** 2 reward types Ã— 4 opponent types = 8 conditions per block, with 12 trials per block

**Note on Current Dataset:** The file naming convention suggests these 12 trials may be from the CRD (stag-value-decreasing) version based on the folder structure described in the data dictionary ("rabbit and stag as the folder name for each of the two versions"). However, the specific opponent framing condition and confirmation of the reward version for these trials is not yet indicated in the metadata. Heejung indicated this metadata will be added with future data collection.

### Data Structure

The behavioral data is recorded at approximately 20-40 Hz (timestamped continuously) and includes:

**Spatial coordinates:**
- `player1_x`, `player1_y`: Position of Player 1 (Chinese patient)
- `player2_x`, `player2_y`: Position of Player 2 (US patient)
- `stag_x`, `stag_y`: Position of the stag
- `rabbit_x`, `rabbit_y`: Position of the rabbit

**Game state:**
- `value`: The changing value of the dynamic prey (stag in CRD version, rabbit in TI version)
- `event`: Outcome codes (0=nothing, 1=P1 touched stag alone, 2=P2 touched stag alone, 3=P1 caught rabbit, 4=P2 caught rabbit, 5=Both caught stag)

**Timing:**
- `time_point`: Server-side timestamp when data was sent to clients
- `delay`: Network latency
- `now_time`: Client-side timestamp when data was received

---

## Key Findings

### Cooperation Rate
Out of 12 trials in this block, only **1 trial (8.3%) resulted in successful cooperation** where both players caught the stag together (Trial 3). The remaining 11 trials (91.7%) resulted in defection, with one player catching the rabbit.

### Defection Patterns
- **Player 1 (Chinese):** Defected in 5 trials (41.7%)
- **Player 2 (US):** Defected in 6 trials (50.0%)

The roughly equal distribution of defections between the two players suggests that neither player consistently exploited the other. Instead, both players showed a strong tendency to defect.

### Trial Duration Patterns
- **Average duration:** 6.5 seconds (SD: 5.7s)
- **Range:** 2.7 to 22.7 seconds
- **Notable:** The cooperation trial (Trial 3) lasted 9.2 seconds, longer than the average defection trial, suggesting players may have initially moved toward the stag before successfully coordinating.

### Temporal Patterns
Looking at the sequence of outcomes:
1. Trial 1: P2 defects
2. Trial 2: P1 defects
3. **Trial 3: COOPERATION** âœ“
4. Trials 4-12: Consistent defection pattern

The single cooperation event occurred early in the block (Trial 3), followed by 9 consecutive defection trials. This pattern raises interesting questions about whether:
- Players learned that cooperation was risky after the early attempt
- The reward structure or opponent framing shifted after Trial 3
- Initial coordination attempts failed in later trials

---

## Spatial Dynamics

### Initial Positions
Players consistently start at opposite corners:
- Player 1: (-198, 198)
- Player 2: (198, -198)
- Stag: (198, 198) â€” near Player 1's starting position
- Rabbit: (-198, -198) â€” near Player 2's starting position

This spatial arrangement creates an interesting asymmetry: each player starts closer to a different prey option, which may influence their movement decisions and coordination dynamics.

### Movement Patterns

**Cooperation Trial (Trial 3):**
The trajectory analysis shows both players initially moved toward the stag, with their distances to the stag decreasing over time. The successful coordination suggests mutual commitment signals were exchanged through their movement patterns.

**Defection Trials:**
In defection trials, we typically see one player moving toward the rabbit while the other may initially approach the stag but then diverts (or moves toward the stag but fails to coordinate timing).

---

## Analytical Considerations

### Current Limitations
1. **Missing Metadata:** We don't yet know which reward version (TI vs. CRD) or opponent framing condition these trials represent. This limits our ability to understand the experimental manipulations' effects.

2. **Single Block:** This represents only 12 trials from one block. We'll need data from multiple blocks across different conditions to assess the robustness of these patterns.

3. **Incomplete Event Detection:** The current data shows outcome events (who caught what) but doesn't include detailed information about:
   - When players first committed to approaching a particular prey
   - Near-misses where both players approached the stag but failed to arrive simultaneously
   - The exact timing and coordination requirements for successful stag hunting

### Suggested Analyses for Full Dataset

Once the complete dataset with metadata becomes available, key analyses should include:

1. **Effect of Reward Manipulation:**
   - Compare cooperation rates between TI (rabbit-increasing) and CRD (stag-decreasing) versions
   - Analyze how the changing value parameter affects decision-making dynamics
   - Test whether the timing of defection differs between conditions

2. **Effect of Opponent Framing:**
   - Compare cooperation rates across the four opponent types (Computer, Same-country, Different-country, Bonding)
   - Assess whether social bonding increases coordination success
   - Examine whether in-group bias affects cooperation

3. **Trajectory-Based Intention Detection:**
   - Develop measures of commitment to cooperation vs. defection based on movement patterns
   - Identify the critical decision points where players commit to a strategy
   - Analyze whether players respond to each other's movements in real-time

4. **Learning Effects:**
   - Track cooperation rates across trials within and between blocks
   - Test for order effects or learning patterns
   - Examine whether early cooperation success predicts later cooperation

5. **Neural Correlates (with iEEG data):**
   - Identify neural signatures associated with cooperation vs. defection decisions
   - Examine neural synchrony between players during coordination attempts
   - Analyze neural responses to opponent movements and apparent intentions

---

## Statistical Power Considerations

With a baseline cooperation rate of ~8% observed in this block, power analyses for the full study should consider:

- **Expected effect sizes:** If reward/opponent manipulations produce moderate effects (e.g., doubling or halving cooperation rates), we'll need substantial trial counts per condition
- **Individual differences:** Variance between patients may be high; mixed-effects models accounting for patient-level random effects will be important
- **Multiple comparisons:** With 2 reward types Ã— 4 opponent types = 8 conditions, corrections for multiple testing should be planned

---

## Methodological Strengths

This study design has several notable strengths:

1. **Ecological validity:** Real-time coordination with actual human partners (despite deception about opponent identity)
2. **Fine-grained behavioral data:** High-frequency position tracking enables detailed trajectory analysis
3. **Clinical population:** iEEG patients provide unique neural data alongside behavior
4. **Cross-cultural comparison:** Chinese and US patients may show different cooperation strategies
5. **Dynamic task structure:** Changing reward values create time pressure and strategic tension

---

## Next Steps

### Immediate
1. **Request metadata:** Obtain reward version and opponent framing information for these 12 trials
2. **Expand analysis:** Once metadata is available, segment analyses by condition
3. **Validate findings:** Check if this block is representative of the larger dataset

### Short-term
1. **Develop trajectory metrics:** Create quantitative measures of commitment timing and coordination attempts
2. **Build predictive models:** Test whether early movement patterns predict trial outcomes
3. **Examine failed coordination:** Analyze near-miss trials where both players approached the stag but failed to coordinate

### Long-term
1. **Integrate neural data:** Once iEEG preprocessing is complete, link neural activity to behavioral dynamics
2. **Cross-cultural analysis:** Compare cooperation patterns between Chinese and US patients
3. **Computational modeling:** Fit game-theoretic models (e.g., level-k reasoning, Bayesian theory of mind) to behavioral data

---

## Technical Notes

### Data Quality
- **Column naming:** Note that the CSV files contain a typo (`plater1_y` instead of `player1_y`) â€” this has been corrected in the analysis scripts
- **Sampling rate:** Variable but approximately 25-40 Hz
- **Missing data:** No obvious missing or corrupt data points in this block
- **Coordinate system:** Ranges approximately from -400 to +400 in both x and y dimensions

### Reproducibility
All analysis code is available in `analyze_stag_hunt.py`. Visualizations and summary statistics can be regenerated from raw CSV files.

---

## Questions for Discussion

1. **Reward version:** Which condition do these 12 trials represent? The relatively low cooperation rate suggests either a TI condition (high rabbit temptation) or that this was an early block before players learned to coordinate.

2. **Opponent framing:** Which opponent type were patients told they were playing against? The pattern of mutual defection might differ depending on whether they thought they were playing a computer vs. another human.

3. **Trial structure:** How were trials sequenced? Was there feedback between trials about outcomes and rewards?

4. **Patient characteristics:** Are there any relevant patient demographics or clinical characteristics (e.g., seizure focus location) that might relate to cooperation behavior?

5. **Deception debriefing:** How was the deception about opponent identity handled in the debriefing? This is important for understanding whether patients suspected the manipulation.

---

## Conclusion

This initial analysis reveals a strong defection bias in the Stag Hunt task, with only 8.3% of trials resulting in successful cooperation. The single cooperation event occurred early in the block, followed by persistent defection. The spatial structure of the task, combined with the temporal dynamics of changing rewards, creates a rich environment for studying real-time coordination and decision-making.

The next critical step is obtaining the metadata about experimental conditions, which will enable us to assess whether this low cooperation rate is specific to particular reward structures or opponent framings, or represents a general pattern in the task. The high-resolution behavioral data provides an excellent foundation for detailed trajectory analysis and eventual integration with neural recordings.

---

# Computational Modeling: Decision Model Development

**Updated: November 12, 2025**

---

## Overview

Following the initial behavioral analysis and Bayesian belief model, we developed a utility-based decision model to predict how players choose their joystick movements at each timestep. This model integrates:

1. **Beliefs** about partner intentions (from the Bayesian model)
2. **Values** of prey targets (changing stag/rabbit rewards)
3. **Spatial positions** (distances to targets)
4. **Action selection** via softmax decision rule

The goal is to test whether beliefs about partner cooperation actually influence moment-to-moment movement decisions.

---

## Model Architecture

### Core Idea

At each timestep, a player chooses a directional movement (joystick angle θ) by evaluating the **utility** of moving in different directions:

**Utility Function:**
```
U(θ) = w_stag × belief_partner × stag_value × gain_stag(θ) +
       w_rabbit × rabbit_value × gain_rabbit(θ)
```

Where:
- `belief_partner`: Probability that partner is going for stag (from Bayesian model)
- `stag_value`, `rabbit_value`: Current rewards for each prey
- `gain_stag(θ)`, `gain_rabbit(θ)`: How much closer does moving in direction θ bring you to each target?
- `w_stag`, `w_rabbit`: Weight parameters (relative importance)

### Key Insight: Beliefs Naturally Model Coordination

The `belief_partner` term elegantly captures the coordination requirement:
- **High belief** → Stag is valuable (partner will cooperate, we can catch it together)
- **Low belief** → Stag becomes less valuable (partner won't help, I'll fail alone)
- **Rabbit** → No belief term needed (guaranteed regardless of partner)

This directly implements the Stag Hunt dilemma in the utility function.

### Action Selection: Softmax Decision Rule

Given utilities for all possible directions, the player probabilistically chooses:

```
P(θ) = exp(β × U(θ)) / Σ_θ' exp(β × U(θ'))
```

Where `β` (temperature) controls how deterministic vs. random the choice is.

### Continuous Likelihood Evaluation

To evaluate how well the model predicts observed angles, we use a **mixture of von Mises distributions**:

```
P(θ_observed) = Σ_i P(action_i) × VonMises(θ_observed | μ=action_i, κ=action_noise)
```

This treats the discrete action space as defining a continuous directional distribution, allowing us to score any empirical angle without discretization artifacts.

---

## Implementation Details

### Files Created

1. **`decision_model.py`**: Core decision model class
   - `UtilityDecisionModel`: Main class implementing utility computation and action selection
   - `compute_utility()`: Evaluates utility of each possible action
   - `compute_action_probabilities()`: Softmax over actions
   - `get_continuous_log_likelihood()`: Evaluates fit to observed data
   - `evaluate_trial_continuous()`: Computes log-likelihood for full trial

2. **`test_decision_model.py`**: Initial testing with discrete likelihood
   - Showed beliefs improve predictions by ~444 log-likelihood points
   - Revealed discretization artifacts (finer grids penalized)

3. **`test_action_space.py`**: Action space granularity testing
   - Tested 8, 16, 32, 64, 128 directions
   - Confirmed discrete likelihood issues

4. **`test_continuous_likelihood.py`**: Continuous likelihood validation
   - Resolved discretization artifacts
   - Showed 16+ directions converge to same fit
   - Identified best motor noise parameter

5. **`fit_parameters.py`**: Maximum likelihood parameter estimation
   - Fits free parameters: `w_stag`, `w_rabbit`, `temperature`, `action_noise`
   - Uses scipy.optimize with L-BFGS-B
   - Tests multiple initializations to avoid local minima

### Design Decisions

**Action Space:**
- Use **16 discrete directions** for computational efficiency
- Continuous likelihood evaluation removes discretization bias
- Can scale to 32/64 directions if needed (performance is similar)

**Motor Noise:**
- Modeled as von Mises concentration parameter κ
- Higher κ = more precise execution (lower noise)
- Best fit: κ ≈ 1.0 (moderate motor variability)

**Speed Parameter:**
- Currently set to 1.0 (arbitrary units)
- Could be fit to data or derived from actual movement distances

---

## Model Validation Results

### Continuous Likelihood Performance

Using 16 directions and optimal motor noise (κ = 1.0):

**Model Comparison:**
- **Without beliefs** (belief fixed at 1.0): log-lik = -6,260
- **With beliefs** (dynamic from Bayesian model): log-lik = -6,056
- **Improvement: +204 log-likelihood points (~3.3%)**

This confirms that **beliefs significantly improve movement predictions**.

### Action Space Convergence

With continuous likelihood, discretization level doesn't matter:
- 8 directions: log-lik = -6,168
- 16 directions: log-lik = -6,167
- 32+ directions: log-lik = -6,167 (identical)

This validates the continuous likelihood approach.

### Motor Noise Parameter

Best fit with different κ values:
- κ = 0.5: -6,135 (too noisy)
- **κ = 1.0: -6,056** ← Best
- κ = 2.0: -6,167
- κ = 5.0: -6,710
- κ = 10.0: -7,003 (too precise)

The optimal κ ≈ 1.0 suggests moderate motor variability in joystick control.

---

## Parameter Fitting

### Free Parameters

The model has 4 free parameters to fit:

1. **w_stag**: Weight on stag utility (relative importance)
2. **w_rabbit**: Weight on rabbit utility
3. **temperature (β)**: Softmax inverse temperature (decision noise)
4. **action_noise (κ)**: Motor noise (execution variability)

### Fitting Procedure

**Method:** Maximum likelihood estimation via scipy.optimize
- **Algorithm:** L-BFGS-B (bounded optimization)
- **Bounds:** All parameters constrained to (0.01, 10.0)
- **Multiple initializations:** Test 4 different starting points to avoid local minima
- **Data:** All 12 trials, both players (24 trajectories total)

### Current Status

**Parameter fitting is currently running** (`fit_parameters.py`).

Initial tests with baseline parameters (all = 1.0) showed:
- Baseline log-likelihood: -6,056
- Room for improvement through parameter optimization

Expected fitted parameters will reveal:
- Relative importance of stag vs. rabbit (weight ratio)
- How deterministic vs. stochastic decisions are (temperature)
- Level of motor precision (action noise)

---

## Model Variants

We tested several model variants to assess which features improve predictions:

### Model 0: Baseline (No Beliefs)
```
U(θ) = w_stag × stag_value × gain_stag(θ) +
       w_rabbit × rabbit_value × gain_rabbit(θ)
```
- Ignores partner's behavior
- Log-lik: -6,260

### Model 1: With Beliefs (Current)
```
U(θ) = w_stag × belief_partner × stag_value × gain_stag(θ) +
       w_rabbit × rabbit_value × gain_rabbit(θ)
```
- Beliefs modulate stag value
- Log-lik: -6,056
- **+204 improvement** ✓

### Future Extensions

**Value-Sensitive Beliefs:**
- Incorporate changing reward values into belief updates
- Higher stag value → movements toward stag more diagnostic

**Strategic Sophistication (Level-k):**
- Level 0: Random movement
- Level 1: Move toward best prey (ignore partner)
- Level 2: Infer partner's goal and respond
- Level 3+: Recursive theory of mind

**Temporal Dynamics:**
- Weight recent movements more heavily
- Identify critical decision points
- Model commitment timing

**Full Trajectory Simulation:**
- Generate complete trial paths from model
- Validate qualitative behavior patterns
- Check if simulated agents successfully cooperate

---

## Code Structure

### `decision_model.py` API

**Main Class:**
```python
model = UtilityDecisionModel(
    n_directions=16,      # Action space granularity
    temperature=1.0,      # Softmax inverse temperature
    w_stag=1.0,          # Stag weight
    w_rabbit=1.0,        # Rabbit weight
    speed=1.0            # Movement speed
)
```

**Key Methods:**

1. **Compute utilities:**
```python
utilities = model.compute_utility(
    player_x, player_y,
    stag_x, stag_y, stag_value,
    rabbit_x, rabbit_y, rabbit_value,
    belief_partner_stag,
    action_angle
)
```

2. **Get action probabilities:**
```python
probs, utilities = model.compute_action_probabilities(...)
```

3. **Sample action:**
```python
action_angle, action_idx, probs = model.sample_action(...)
```

4. **Evaluate trial:**
```python
total_ll, mean_ll, results_df = model.evaluate_trial_continuous(
    trial_data,
    player='player1',
    belief_column='p1_belief_p2_stag',
    action_noise=1.0
)
```

### Integration with Belief Model

The decision model requires beliefs as input:

```python
# Step 1: Run belief model
belief_model = BayesianIntentionModel(prior_stag=0.5, concentration=1.5)
trial_with_beliefs = belief_model.run_trial(trial_data)

# Step 2: Evaluate decision model
decision_model = UtilityDecisionModel(n_directions=16, temperature=1.0)
log_lik, _, _ = decision_model.evaluate_trial_continuous(
    trial_with_beliefs,
    player='player1',
    belief_column='p1_belief_p2_stag',
    action_noise=1.0
)
```

---

## Next Steps

### Immediate (This Session)
1. ✓ Implement utility-based decision model
2. ✓ Implement continuous likelihood evaluation
3. ✓ Validate that beliefs improve predictions
4. ⏳ Fit free parameters to data (currently running)
5. ⏳ Document results in CLAUDE.md

### Short-term
1. **Simulate complete trajectories** from fitted model
   - Do simulated agents make reasonable decisions?
   - Can model reproduce cooperation patterns?
   - Visualize predicted vs. actual paths

2. **Parameter sensitivity analysis**
   - How do predictions change with different parameter values?
   - Which parameters matter most?
   - Are parameters identifiable from current data?

3. **Model comparison framework**
   - Implement AIC/BIC for model selection
   - Test extensions (value-sensitive beliefs, level-k, etc.)
   - Cross-validation for generalization testing

### Medium-term
1. **Extended model variants**
   - Implement value-sensitive beliefs
   - Implement level-k strategic reasoning
   - Test temporal attention weighting

2. **Individual differences**
   - Fit parameters separately for each player
   - Compare Chinese vs. US patient parameters
   - Test for clinical characteristic correlations

3. **Condition effects**
   - Once metadata available: fit by condition
   - Test reward type effects (TI vs. CRD)
   - Test opponent framing effects

### Long-term
1. **Neural integration**
   - Identify neural correlates of utilities
   - Test for ramping activity toward decision
   - Examine prediction error signals

2. **Hierarchical Bayesian estimation**
   - Group-level parameter distributions
   - Individual-level random effects
   - Cultural group comparisons

---

## Technical Notes

### Computational Efficiency

**Vectorization:**
- von Mises PDF computation is vectorized over all action angles
- ~16 seconds to test 9 model configurations on 12 trials
- Efficient enough for parameter fitting with scipy.optimize

**Scaling:**
- Current implementation handles 12 trials easily
- Should scale well to full dataset (hundreds of trials)
- Could parallelize across trials if needed

### Numerical Stability

**Log-space computations:**
- Use `logsumexp` for softmax normalization
- Add small constant (1e-10) to prevent log(0)
- Clip extreme values where appropriate

**Bounded optimization:**
- All parameters constrained to positive values
- Prevents optimizer from exploring invalid regions
- L-BFGS-B handles bounds efficiently

---

## Research Questions Addressed

### Can beliefs predict behavior?
**Yes** - Including beliefs improves log-likelihood by +204 points (~3.3%).

This suggests that players' moment-to-moment movements are influenced by their inferences about what their partner is doing.

### How noisy is joystick control?
**Moderately noisy** - Best fit: κ ≈ 1.0

This indicates players don't execute perfectly precise movements toward their intended target. There's substantial motor variability.

### Does discretization matter?
**Not with continuous likelihood** - 8 vs. 32 vs. 64 directions all converge to same fit.

The continuous likelihood approach successfully removes discretization artifacts. We can use coarser action spaces for efficiency without sacrificing accuracy.

---

## Interpretations & Implications

### Belief-Action Coupling

The significant improvement from including beliefs suggests a **tight coupling** between social inference and motor control:

- Players continuously update beliefs about partner's intentions
- These beliefs immediately influence movement decisions
- When belief in cooperation drops → pivot toward rabbit

This supports **real-time social cognition** rather than pre-planned strategies.

### Motor Noise Level

The moderate motor noise (κ ≈ 1.0) has implications:

- Joystick control isn't perfectly precise
- Some observed variability is execution noise, not decision noise
- Need to account for motor noise when interpreting behavior

### Coordination Difficulty

The belief-modulated utility naturally explains why cooperation is hard:

- **Chicken-and-egg problem:** Both players need high beliefs simultaneously
- **Fragility:** Any wobble toward rabbit immediately decreases partner's belief
- **Positive feedback:** Defection signals can cascade rapidly

The model suggests cooperation requires **sustained mutual commitment signals** throughout the trial.

---

## Code Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `model.py` | Bayesian belief model | ✓ Complete |
| `decision_model.py` | Utility-based action selection | ✓ Complete |
| `make_video.py` | Visualization with belief overlays | ✓ Complete |
| `test_decision_model.py` | Initial discrete likelihood tests | ✓ Complete |
| `test_action_space.py` | Action space granularity exploration | ✓ Complete |
| `test_continuous_likelihood.py` | Continuous likelihood validation | ✓ Complete |
| `fit_parameters.py` | Maximum likelihood parameter estimation | ⏳ Running |

---

## Reproducibility

All analyses can be reproduced from the trial CSV files in `inputs/`:

```bash
# Test decision model with beliefs
python test_decision_model.py

# Test continuous likelihood
python test_continuous_likelihood.py

# Fit parameters
python fit_parameters.py
```

Parameters and random seeds are documented in each script.

---
