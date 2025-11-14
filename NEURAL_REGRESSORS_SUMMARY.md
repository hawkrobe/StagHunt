# Neural Regressors for iEEG Analysis
**Stag Hunt Cooperation Task**
**Generated:** November 13, 2025

---

## Overview

This document summarizes the computational model and neural regressors developed for analyzing intracranial EEG data from the Stag Hunt cooperation task.

---

## Primary Neural Regressor: BELIEF ABOUT PARTNER

### **What it is:**
`p1_belief_p2_stag`, `p2_belief_p1_stag`

The probability that a player assigns to their partner choosing to cooperate (go for stag) at each moment in time.

### **Why it's primary:**
1. **Pure social prediction** - Unconfounded by spatial/motor factors
2. **Continuous Bayesian updates** - Updates moment-by-moment based on partner's movements
3. **Clear interpretation** - Range [0, 1]:
   - 0.0 = Confident partner will defect (rabbit)
   - 0.5 = Maximum uncertainty
   - 1.0 = Confident partner will cooperate (stag)
4. **Rich temporal dynamics** - Ramping/declining activity ideal for neural analysis

### **How it's computed:**

**Bayesian belief update:**
```
P(partner→stag | observations) =
    P(observations | partner→stag) × P(partner→stag) / Z
```

Where:
- **Likelihood**: Computed via inverse planning (what movement would I expect if partner goes for stag?)
- **Prior**: Initialized from cross-trial expectations (evolving across trials)
- **Posterior**: Updated continuously as partner moves

### **Evidence from data:**

**Cooperation trial (Trial 3):**
- Start: P1 belief = 0.09, P2 belief = 0.16 (pessimistic from past failures)
- Middle: P1 = 0.99, P2 = 0.99 (mutual recognition!)
- Change: Δ = +0.90, +0.83 (massive evidence accumulation)

**Defection trials:**
- Mean belief: 0.38-0.52 (low/uncertain)
- Beliefs collapse to ~0.01 when partner moves toward rabbit

---

## Secondary Neural Regressors

### **1. Cross-Trial Expectations**
`p1_cross_trial_expectation`, `p2_cross_trial_expectation`

**What:** Evolving priors about partner's tendency to cooperate, learned across trials

**How computed:**
```
Expectation_t = (1-α) × Expectation_{t-1} + α × FinalBelief_{t-1}
```
Where α (learning rate) is a fitted parameter

**Neural prediction:** Slow baseline shifts in belief regions (TPJ/mPFC) across trials

**Evidence:** Expectations decline from 0.50 → 0.09 across defection trials

---

### **2. Belief × Value (Expected Utility)**
`p1_EU_stag`, `p2_EU_stag`

**What:** Integration of belief with stag value for decision-making

**Formula:**
```
EU(stag) = Belief(partner→stag) × Value(stag) × TimingAlignment
```

**Neural prediction:** vmPFC parametric modulation by belief × value

**Use case:** Testing value integration in decision-making regions

---

### **3. Belief Prediction Errors**
(Computed at trial end)

**What:** Difference between expected and actual partner choice

**Formula:**
```
PE = Actual - Belief_final

Where Actual = 1 if partner cooperated, 0 if defected
```

**Neural prediction:** Striatal/ACC response at trial outcomes

**Evidence:** Large PEs (±0.99) when beliefs are strongly disconfirmed

---

## Model Architecture

### **Integrated Hierarchical Model with Cross-Trial Learning**

**Three cognitive levels:**

1. **Cross-Trial Learning** (slow timescale)
   - Tracks partner's cooperation tendency across trials
   - Learning rate: Fitted parameter (~0.2-0.5)
   - Updates expectations for next trial

2. **Within-Trial Belief Updates** (fast timescale)
   - Bayesian inference from partner's movements
   - Continuous updating each timestep
   - Produces primary regressor: `belief`

3. **Hierarchical Decision-Making**
   - Goal selection: Stag vs. Rabbit (strategic)
   - Plan execution: Movement toward goal (motor)
   - Separate temperatures for strategy vs. execution

### **Fitted Parameters:**
- `learning_rate`: Cross-trial expectation updating (currently fitting)
- `goal_temperature`: Strategic choice softness (currently fitting)
- `execution_temperature`: Motor precision (currently fitting)
- `timing_tolerance`: Coordination window (currently fitting)

---

## Neural Predictions

### **Region-Specific Hypotheses:**

#### **Theory of Mind Network (TPJ, mPFC, precuneus)**
- **Regressor:** Belief ramping
- **Prediction:** Activity tracks belief updates
- **Test:** Parametric modulation by belief time series
- **Expected:** Positive correlation in cooperation trials, negative in defection trials

#### **Value Regions (vmPFC)**
- **Regressor:** Belief × Value
- **Prediction:** Integration of social prediction with reward value
- **Test:** Parametric modulation by EU(stag)
- **Expected:** Higher when both belief and value are high

#### **Prediction Error Regions (Striatum, ACC)**
- **Regressor:** Belief PE at trial end
- **Prediction:** Response to unexpected partner choices
- **Test:** Event-related analysis at outcome
- **Expected:** Positive PE → positive signal, negative PE → negative signal

#### **Learning Regions (ACC, DLPFC)**
- **Regressor:** Cross-trial expectations
- **Prediction:** Baseline shifts across trials
- **Test:** Trial-by-trial adaptation
- **Expected:** Declining activity as expectations decrease

---

## Data Files

### **Primary Outputs:**

1. **`integrated_model_optimal.csv`** (once fitting completes)
   - All timesteps across all trials
   - Contains all regressors
   - Optimal fitted parameters

2. **`belief_as_primary_regressor.png`**
   - Static visualization of belief dynamics
   - Shows cooperation vs. defection differences

3. **`stag_hunt_belief_primary.mp4`** (generating now)
   - Video emphasizing belief as primary regressor
   - Large belief panel with interpretive zones
   - Real-time belief percentages

### **Regressor Columns in CSV:**

**Primary (use these first):**
- `p1_belief_p2_stag`, `p2_belief_p1_stag`

**Secondary:**
- `p1_cross_trial_expectation`, `p2_cross_trial_expectation`
- `p1_EU_stag`, `p2_EU_stag`
- `p1_EU_rabbit`, `p2_EU_rabbit`
- `p1_P_choose_stag`, `p2_P_choose_stag`

**Tertiary (optional):**
- `p1_P_coord`, `p2_P_coord` (belief × timing - confounded)
- `p1_timing_alignment`, `p2_timing_alignment`
- `p1_dist_to_stag`, `p2_dist_to_stag`
- `p1_choice_entropy`, `p2_choice_entropy`

---

## Analysis Recommendations

### **Step 1: Basic Correlation**
Correlate `belief` time series with neural activity in Theory of Mind regions
- **Expect:** Positive correlation in TPJ/mPFC
- **Control:** Use `dist_to_stag` as spatial control

### **Step 2: Cooperation vs. Defection**
Compare belief ramping in cooperation vs. defection trials
- **Cooperation:** Beliefs should ramp up (0.09 → 0.99)
- **Defection:** Beliefs should drop (0.50 → 0.01)

### **Step 3: Prediction Errors**
Event-related analysis at trial outcomes
- Compute: `PE = actual - belief_final`
- Test striatal/ACC responses to PE magnitude

### **Step 4: Cross-Trial Learning**
Test for baseline shifts across trials
- Regressor: Trial-by-trial expectation values
- Predict: Declining baseline in belief regions

### **Step 5: Value Integration**
Test belief × value interaction in vmPFC
- Regressor: `EU_stag = belief × value × timing`
- Predict: Parametric modulation

---

## Model Comparison Results

**Best behavioral fits:**
1. Distance + coin flip: LL = -5417.64 (2 params) - **Best simple model**
2. Distance-based: LL = -5430.04 (1 param)
3. Hierarchical (no learning): LL = -5699.88 (2 params)
4. Integrated (with learning): Fitting now (4 params)

**However:** For neural analysis, we prioritize **interpretability and cognitive plausibility** over pure behavioral fit.

The integrated model provides:
- ✓ Pure social prediction signal (belief)
- ✓ Cross-trial learning dynamics
- ✓ Hierarchical decision structure
- ✓ Testable neural predictions

---

## Key Findings

### **1. Beliefs Strongly Discriminate Cooperation**
- Cooperation: Mean belief = 0.88-0.95
- Defection: Mean belief = 0.38-0.52
- **Conclusion:** Belief is highly diagnostic of trial outcomes

### **2. Dramatic Belief Updates During Cooperation**
- Trial 3 shows +0.90 belief change for P1, +0.83 for P2
- **Conclusion:** Strong evidence accumulation when cooperation succeeds

### **3. Cross-Trial Learning Occurs**
- Expectations decline from 0.50 → 0.09 across defection trials
- **Conclusion:** Players adapt priors based on experience

### **4. Large Prediction Errors for Learning**
- PEs ≈ ±0.99 when beliefs are strongly violated
- **Conclusion:** Strong error signals for neural learning

---

## Next Steps

### **For Neural Analysis:**
1. Load `integrated_model_optimal.csv` once fitting completes
2. Align timesteps with neural recordings using `time_point` column
3. Extract `p1_belief_p2_stag`, `p2_belief_p1_stag` as primary regressors
4. Run GLM with belief as parametric modulator
5. Look for ramping activity in TPJ/mPFC during cooperation trials

### **For Model Development:**
1. Fit individual patient parameters (different learning rates, etc.)
2. Compare parameters across conditions (Computer vs. Human opponent, etc.)
3. Test value-sensitive belief updates (higher stag value → movements more diagnostic)
4. Implement recursive ToM (what partner thinks I think they'll do)

---

## References

**Computational Framework:**
- Baker, Saxe & Tenenbaum (2009): Hierarchical goal inference
- Yoshida et al. (2010): Bayesian inference in cooperation
- Hampton, Bossaerts & O'Doherty (2008): Neural prediction errors in social learning

**Model Files:**
- `hierarchical_model_with_cross_trial_learning.py`: Core integrated model
- `fit_integrated_model.py`: Parameter fitting
- `analyze_belief_as_primary_regressor.py`: Belief analysis
- `make_video_belief_focus.py`: Visualization

---

## Contact

Robert D. Hawkins
Social Interaction Lab, Stanford University

**Questions about:**
- Model implementation → See code files
- Neural predictions → See "Neural Predictions" section
- Regressor selection → Use `p1_belief_p2_stag` as primary
