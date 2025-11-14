#!/usr/bin/env python3
"""
Analyze how beliefs evolve across trials.

Key question: Do participants build up expectations about their partner
across repeated trials, rather than starting fresh each time?
"""

import pandas as pd
import numpy as np
import glob
from models.belief_model_decision import BayesianIntentionModelWithDecision
import matplotlib.pyplot as plt


def analyze_cross_trial_beliefs():
    """Analyze belief trajectories across all trials in sequence."""

    print("="*70)
    print("CROSS-TRIAL BELIEF DYNAMICS")
    print("="*70)
    print()

    # Load trials in order
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*_2024_08_24_0848.csv'))

    # Initialize belief model
    belief_model = BayesianIntentionModelWithDecision(
        decision_model_params={
            'temperature': 5.0,
            'timing_tolerance': 150.0,
            'action_noise': 5.0,
            'n_directions': 8
        },
        prior_stag=0.5,
        belief_bounds=(0.01, 0.99)
    )

    # Track beliefs across trials
    trial_results = []

    print("Running belief model on each trial...")
    print()

    for trial_file in trial_files:
        trial_data = pd.read_csv(trial_file)
        if 'plater1_y' in trial_data.columns:
            trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})

        trial_num = int(trial_file.split('trial')[1].split('_')[0])

        # Run belief model
        trial_data = belief_model.run_trial(trial_data)

        # Get outcome
        outcome = "Unknown"
        if trial_data['event'].max() == 5:
            outcome = "COOPERATION"
        elif 3 in trial_data['event'].values:
            outcome = "P1 rabbit"
        elif 4 in trial_data['event'].values:
            outcome = "P2 rabbit"

        # Get final beliefs
        final_p1_belief = trial_data['p1_belief_p2_stag'].iloc[-1]
        final_p2_belief = trial_data['p2_belief_p1_stag'].iloc[-1]

        # Get initial beliefs (after first timestep, before any movement)
        initial_p1_belief = trial_data['p1_belief_p2_stag'].iloc[0]
        initial_p2_belief = trial_data['p2_belief_p1_stag'].iloc[0]

        trial_results.append({
            'trial': trial_num,
            'outcome': outcome,
            'duration_s': len(trial_data) * 0.04,  # ~25Hz sampling
            'p1_initial': initial_p1_belief,
            'p1_final': final_p1_belief,
            'p1_change': final_p1_belief - initial_p1_belief,
            'p2_initial': initial_p2_belief,
            'p2_final': final_p2_belief,
            'p2_change': final_p2_belief - initial_p2_belief,
        })

        print(f"Trial {trial_num:2d}: {outcome:15s} | "
              f"P1: {initial_p1_belief:.3f}→{final_p1_belief:.3f} ({final_p1_belief-initial_p1_belief:+.3f}) | "
              f"P2: {initial_p2_belief:.3f}→{final_p2_belief:.3f} ({final_p2_belief-initial_p2_belief:+.3f})")

    results_df = pd.DataFrame(trial_results)

    print("\n" + "="*70)
    print("CROSS-TRIAL LEARNING ANALYSIS")
    print("="*70)
    print()

    # Question 1: Should final beliefs predict next trial's initial beliefs?
    print("1. Do final beliefs from trial N predict trial N+1 outcomes?")
    print()

    for i in range(len(results_df) - 1):
        curr = results_df.iloc[i]
        next_trial = results_df.iloc[i + 1]

        print(f"   Trial {curr['trial']}: ended with P1={curr['p1_final']:.3f}, P2={curr['p2_final']:.3f}")
        print(f"   Trial {next_trial['trial']}: {next_trial['outcome']:15s}")

        # If they carried beliefs forward, would they predict the outcome?
        avg_final_belief = (curr['p1_final'] + curr['p2_final']) / 2
        if avg_final_belief > 0.5 and next_trial['outcome'] == "COOPERATION":
            print(f"      ✓ High final beliefs ({avg_final_belief:.3f}) → cooperation")
        elif avg_final_belief < 0.5 and next_trial['outcome'] != "COOPERATION":
            print(f"      ✓ Low final beliefs ({avg_final_belief:.3f}) → defection")
        else:
            print(f"      ✗ Mismatch: beliefs={avg_final_belief:.3f}, outcome={next_trial['outcome']}")
        print()

    # Question 2: Is there a trend in beliefs over time?
    print("\n2. Do beliefs systematically increase or decrease across trials?")
    print()

    # Compute trial-by-trial averages
    avg_beliefs = (results_df['p1_final'] + results_df['p2_final']) / 2

    # Linear trend
    trials = results_df['trial'].values
    slope, intercept = np.polyfit(trials, avg_beliefs, 1)

    print(f"   Linear trend: belief = {intercept:.3f} + {slope:.4f} × trial")
    if abs(slope) < 0.01:
        print(f"   → Beliefs remain stable across trials (flat trend)")
    elif slope > 0:
        print(f"   → Beliefs increase over trials (learning to cooperate)")
    else:
        print(f"   → Beliefs decrease over trials (learning to defect)")

    # Question 3: Impact of cooperation trial
    print("\n3. What happened after the cooperation trial?")
    print()

    coop_idx = results_df[results_df['outcome'] == 'COOPERATION'].index[0]
    coop_trial_num = results_df.iloc[coop_idx]['trial']

    print(f"   Cooperation occurred in trial {coop_trial_num}")
    print(f"   Final beliefs: P1={results_df.iloc[coop_idx]['p1_final']:.3f}, "
          f"P2={results_df.iloc[coop_idx]['p2_final']:.3f}")

    if coop_idx < len(results_df) - 1:
        print(f"\n   Subsequent trials:")
        for i in range(coop_idx + 1, min(coop_idx + 4, len(results_df))):
            row = results_df.iloc[i]
            print(f"      Trial {row['trial']}: {row['outcome']:15s} | "
                  f"Final beliefs: P1={row['p1_final']:.3f}, P2={row['p2_final']:.3f}")

    # Question 4: Model with carry-over priors
    print("\n" + "="*70)
    print("4. WHAT IF beliefs carried over between trials?")
    print("="*70)
    print()

    print("Current model: Each trial starts with prior = 0.5")
    print("Alternative: Each trial starts with prior = previous trial's final belief")
    print()

    # Simulate carry-over
    carried_beliefs = []
    p1_prior = 0.5
    p2_prior = 0.5

    print("Simulated carry-over priors:")
    print()

    for i, row in results_df.iterrows():
        print(f"   Trial {row['trial']:2d}: Would start with P1={p1_prior:.3f}, P2={p2_prior:.3f}")
        print(f"              Actual outcome: {row['outcome']}")

        # Update priors for next trial
        p1_prior = row['p1_final']
        p2_prior = row['p2_final']

        # Track for correlation analysis
        carried_beliefs.append((p1_prior, p2_prior))
        print()

    # Save results
    results_df.to_csv('cross_trial_beliefs.csv', index=False)
    print(f"\n✓ Saved results to cross_trial_beliefs.csv")

    return results_df


if __name__ == '__main__':
    results = analyze_cross_trial_beliefs()
