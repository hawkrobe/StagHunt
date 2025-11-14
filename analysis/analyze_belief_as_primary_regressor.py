#!/usr/bin/env python3
"""
Analysis: Belief about partner as primary social coordination regressor.

Key insight: The belief P(partner goes for stag) is the PURE social prediction,
unconfounded by spatial/motor factors.

This script analyzes:
1. How beliefs evolve during cooperation vs. defection trials
2. Belief dynamics at critical decision points
3. Belief prediction errors at trial outcomes
4. Cross-trial belief evolution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print("BELIEF AS PRIMARY SOCIAL COORDINATION REGRESSOR")
print("="*70)
print()

# Load integrated model output (use lr=0.3 version)
data_file = 'integrated_model_lr0.3.csv'

if not Path(data_file).exists():
    print(f"Running integrated model to generate {data_file}...")
    import subprocess
    subprocess.run(['python', 'hierarchical_model_with_cross_trial_learning.py'])

df = pd.read_csv(data_file)

# Detect outcomes
trial_outcomes = []
for trial_num in sorted(df['trial_num'].unique()):
    trial_df = df[df['trial_num'] == trial_num]

    if trial_df['event'].max() == 5:
        outcome = 'cooperation'
    elif 3 in trial_df['event'].values:
        outcome = 'p1_defect'
    elif 4 in trial_df['event'].values:
        outcome = 'p2_defect'
    else:
        outcome = 'unknown'

    trial_outcomes.append({
        'trial': trial_num,
        'outcome': outcome,
        'n_frames': len(trial_df)
    })

outcomes_df = pd.DataFrame(trial_outcomes)

print("Trial outcomes:")
print(outcomes_df.to_string(index=False))
print()

# ============================================================================
# ANALYSIS 1: Belief trajectories by outcome type
# ============================================================================

print("="*70)
print("ANALYSIS 1: BELIEF DYNAMICS BY TRIAL OUTCOME")
print("="*70)
print()

coop_trials = outcomes_df[outcomes_df['outcome'] == 'cooperation']['trial'].values
defect_trials = outcomes_df[outcomes_df['outcome'].str.contains('defect')]['trial'].values

# Get belief statistics
for outcome_type, trial_list in [('Cooperation', coop_trials), ('Defection', defect_trials)]:
    if len(trial_list) == 0:
        continue

    belief_p1_all = []
    belief_p2_all = []

    for trial in trial_list:
        trial_df = df[df['trial_num'] == trial]
        belief_p1_all.extend(trial_df['p1_belief_p2_stag'].values)
        belief_p2_all.extend(trial_df['p2_belief_p1_stag'].values)

    print(f"{outcome_type} trials (n={len(trial_list)}):")
    print(f"  P1 belief: mean={np.mean(belief_p1_all):.3f}, "
          f"median={np.median(belief_p1_all):.3f}, "
          f"std={np.std(belief_p1_all):.3f}")
    print(f"  P2 belief: mean={np.mean(belief_p2_all):.3f}, "
          f"median={np.median(belief_p2_all):.3f}, "
          f"std={np.std(belief_p2_all):.3f}")
    print()

# ============================================================================
# ANALYSIS 2: Belief at critical timepoints
# ============================================================================

print("="*70)
print("ANALYSIS 2: BELIEF AT CRITICAL TIMEPOINTS")
print("="*70)
print()

for trial_num in sorted(df['trial_num'].unique()):
    trial_df = df[df['trial_num'] == trial_num].reset_index(drop=True)
    outcome = outcomes_df[outcomes_df['trial'] == trial_num]['outcome'].values[0]

    # Initial belief (cross-trial expectation)
    initial_p1 = trial_df['p1_cross_trial_expectation'].iloc[0]
    initial_p2 = trial_df['p2_cross_trial_expectation'].iloc[0]

    # Midpoint belief
    mid_idx = len(trial_df) // 2
    mid_p1 = trial_df['p1_belief_p2_stag'].iloc[mid_idx]
    mid_p2 = trial_df['p2_belief_p1_stag'].iloc[mid_idx]

    # Final belief
    final_p1 = trial_df['p1_belief_p2_stag'].iloc[-1]
    final_p2 = trial_df['p2_belief_p1_stag'].iloc[-1]

    # Belief change
    change_p1 = final_p1 - initial_p1
    change_p2 = final_p2 - initial_p2

    print(f"Trial {trial_num:2d} ({outcome:12s}): "
          f"Init: P1={initial_p1:.2f}, P2={initial_p2:.2f} | "
          f"Mid: P1={mid_p1:.2f}, P2={mid_p2:.2f} | "
          f"Final: P1={final_p1:.2f}, P2={final_p2:.2f} | "
          f"Δ: P1={change_p1:+.2f}, P2={change_p2:+.2f}")

print()

# ============================================================================
# ANALYSIS 3: Belief prediction errors
# ============================================================================

print("="*70)
print("ANALYSIS 3: BELIEF PREDICTION ERRORS AT OUTCOME")
print("="*70)
print()

print("Prediction error = actual_outcome - expected_outcome")
print("where expected = belief at trial end\n")

for trial_num in sorted(df['trial_num'].unique()):
    trial_df = df[df['trial_num'] == trial_num]
    outcome = outcomes_df[outcomes_df['trial'] == trial_num]['outcome'].values[0]

    final_p1 = trial_df['p1_belief_p2_stag'].iloc[-1]
    final_p2 = trial_df['p2_belief_p1_stag'].iloc[-1]

    # Compute prediction errors
    if outcome == 'cooperation':
        # Both cooperated (stag)
        actual_p1 = 1.0  # P2 actually chose stag
        actual_p2 = 1.0  # P1 actually chose stag
    elif outcome == 'p2_defect':
        # P2 defected (rabbit)
        actual_p1 = 0.0  # P2 chose rabbit
        actual_p2 = 1.0  # P1 went for stag (or tried)
    elif outcome == 'p1_defect':
        # P1 defected (rabbit)
        actual_p1 = 1.0  # P2 went for stag (or tried)
        actual_p2 = 0.0  # P1 chose rabbit
    else:
        continue

    pe_p1 = actual_p1 - final_p1
    pe_p2 = actual_p2 - final_p2

    print(f"Trial {trial_num:2d} ({outcome:12s}): "
          f"P1 PE={pe_p1:+.3f} (believed {final_p1:.2f}, was {actual_p1:.1f}), "
          f"P2 PE={pe_p2:+.3f} (believed {final_p2:.2f}, was {actual_p2:.1f})")

print()

# ============================================================================
# ANALYSIS 4: Key neural predictions
# ============================================================================

print("="*70)
print("NEURAL PREDICTIONS FOR BELIEF REGRESSOR")
print("="*70)
print()

print("1. BELIEF RAMPING (within-trial dynamics)")
print("   - Prediction: Neural activity in TPJ/mPFC tracks belief updates")
print("   - When partner moves toward stag → belief↑ → neural activity↑")
print("   - When partner moves toward rabbit → belief↓ → neural activity↓")
print("   - Test: Correlate belief time series with neural data")
print()

print("2. BELIEF PREDICTION ERRORS (at trial outcome)")
print("   - Prediction: Striatum/ACC responds to belief PEs")
print("   - Positive PE: Partner cooperated more than expected")
print("   - Negative PE: Partner defected despite high belief")
print("   - Test: Event-related response at trial end")
print()

print("3. CROSS-TRIAL LEARNING (slow adaptation)")
print("   - Prediction: Belief at trial START shifts based on history")
print("   - Repeated defections → lower initial belief")
print("   - Recent cooperation → higher initial belief")
print("   - Test: Trial-by-trial adaptation of baseline activity")
print()

print("4. BELIEF × VALUE INTEGRATION (decision-making)")
print("   - Prediction: vmPFC combines belief with stag value")
print("   - EU(stag) = belief × V_stag")
print("   - Higher when both belief and value are high")
print("   - Test: Parametric modulation by belief × value")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("="*70)
print("CREATING VISUALIZATIONS")
print("="*70)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Belief trajectories for cooperation vs defection
ax = axes[0, 0]
for trial in coop_trials:
    trial_df = df[df['trial_num'] == trial].reset_index(drop=True)
    times = np.arange(len(trial_df))
    ax.plot(times, trial_df['p1_belief_p2_stag'], 'r-', alpha=0.5, linewidth=2)
    ax.plot(times, trial_df['p2_belief_p1_stag'], 'orange', alpha=0.5, linewidth=2)

ax.set_title('Belief Trajectories: Cooperation Trials', fontsize=12, weight='bold')
ax.set_xlabel('Time (frames)')
ax.set_ylabel('Belief P(partner → stag)')
ax.set_ylim([0, 1])
ax.grid(alpha=0.3)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

# Plot 2: Defection trials
ax = axes[0, 1]
for trial in defect_trials[:5]:  # Show first 5
    trial_df = df[df['trial_num'] == trial].reset_index(drop=True)
    times = np.arange(len(trial_df))
    ax.plot(times, trial_df['p1_belief_p2_stag'], 'r-', alpha=0.3)
    ax.plot(times, trial_df['p2_belief_p1_stag'], 'orange', alpha=0.3)

ax.set_title('Belief Trajectories: Defection Trials', fontsize=12, weight='bold')
ax.set_xlabel('Time (frames)')
ax.set_ylabel('Belief P(partner → stag)')
ax.set_ylim([0, 1])
ax.grid(alpha=0.3)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

# Plot 3: Belief distribution by outcome
ax = axes[1, 0]

coop_beliefs = []
defect_beliefs = []

for trial in coop_trials:
    trial_df = df[df['trial_num'] == trial]
    coop_beliefs.extend(trial_df['p1_belief_p2_stag'].values)
    coop_beliefs.extend(trial_df['p2_belief_p1_stag'].values)

for trial in defect_trials:
    trial_df = df[df['trial_num'] == trial]
    defect_beliefs.extend(trial_df['p1_belief_p2_stag'].values)
    defect_beliefs.extend(trial_df['p2_belief_p1_stag'].values)

ax.hist(coop_beliefs, bins=20, alpha=0.7, color='green', label='Cooperation', density=True)
ax.hist(defect_beliefs, bins=20, alpha=0.7, color='red', label='Defection', density=True)
ax.set_xlabel('Belief P(partner → stag)')
ax.set_ylabel('Density')
ax.set_title('Belief Distributions by Outcome', fontsize=12, weight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Cross-trial expectation evolution
ax = axes[1, 1]
trial_nums = []
p1_expectations = []
p2_expectations = []

for trial in sorted(df['trial_num'].unique()):
    trial_df = df[df['trial_num'] == trial]
    trial_nums.append(trial)
    p1_expectations.append(trial_df['p1_cross_trial_expectation'].iloc[0])
    p2_expectations.append(trial_df['p2_cross_trial_expectation'].iloc[0])

ax.plot(trial_nums, p1_expectations, 'r-o', linewidth=2, markersize=6, label='P1 expectation')
ax.plot(trial_nums, p2_expectations, 'orange', marker='o', linewidth=2, markersize=6, label='P2 expectation')
ax.set_xlabel('Trial number')
ax.set_ylabel('Cross-trial expectation')
ax.set_title('Cross-Trial Learning Dynamics', fontsize=12, weight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, label='Neutral')

plt.tight_layout()
plt.savefig('belief_as_primary_regressor.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to belief_as_primary_regressor.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*70)
print("SUMMARY: WHY BELIEF IS THE PRIMARY SOCIAL REGRESSOR")
print("="*70)
print()

print("1. PURE SOCIAL PREDICTION")
print("   - Belief = P(partner goes for stag)")
print("   - Unconfounded by spatial/motor factors")
print("   - Direct readout of Theory of Mind computation")
print()

print("2. CONTINUOUS UPDATING")
print("   - Updates moment-by-moment based on partner's actions")
print("   - Bayesian posterior probability")
print("   - Rich temporal dynamics for neural analysis")
print()

print("3. CLEAR INTERPRETATION")
print("   - Range: [0, 1] probability")
print("   - 0 = confident partner will defect")
print("   - 1 = confident partner will cooperate")
print("   - 0.5 = maximum uncertainty")
print()

print("4. DISSOCIABLE FROM P_COORD")
print("   - P_coord = belief × timing_alignment")
print("   - Belief: social prediction (pure)")
print("   - Timing: spatial constraint (confounded)")
print("   - Can separately test social vs spatial signals")
print()

print("RECOMMENDED PRIMARY REGRESSOR: p1_belief_p2_stag, p2_belief_p1_stag")
print("RECOMMENDED SECONDARY: P_coord (belief × timing)")
print("RECOMMENDED TERTIARY: Cross-trial expectations (slow learning)")
