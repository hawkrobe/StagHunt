"""
Test continuous likelihood approach.

This should resolve the issue where finer action discretization was
penalized - now we evaluate likelihood in continuous space.
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from belief_model_distance import BayesianIntentionModel
from decision_model_basic import UtilityDecisionModel


def load_trial(filepath):
    """Load and clean a single trial CSV."""
    df = pd.read_csv(filepath)
    if 'plater1_y' in df.columns:
        df = df.rename(columns={'plater1_y': 'player1_y'})
    df = df.dropna()
    return df


def test_model_continuous(n_directions, action_noise, trial_files, belief_model):
    """Test model with continuous likelihood."""
    decision_model = UtilityDecisionModel(
        n_directions=n_directions,
        temperature=1.0,
        w_stag=1.0,
        w_rabbit=1.0
    )

    total_log_lik = 0

    for trial_file in trial_files:
        trial_data = load_trial(trial_file)
        trial_with_beliefs = belief_model.run_trial(trial_data)

        p1_log_lik, _, _ = decision_model.evaluate_trial_continuous(
            trial_with_beliefs,
            player='player1',
            belief_column='p1_belief_p2_stag',
            action_noise=action_noise
        )
        p2_log_lik, _, _ = decision_model.evaluate_trial_continuous(
            trial_with_beliefs,
            player='player2',
            belief_column='p2_belief_p1_stag',
            action_noise=action_noise
        )

        total_log_lik += p1_log_lik + p2_log_lik

    mean_log_lik = total_log_lik / len(trial_files)
    return total_log_lik, mean_log_lik


def main():
    # Load all trials
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*.csv'))
    print(f"Found {len(trial_files)} trials\n")

    # Initialize belief model
    belief_model = BayesianIntentionModel(
        prior_stag=0.5,
        concentration=1.5,
        belief_bounds=(0.01, 0.99)
    )

    print("="*70)
    print("CONTINUOUS LIKELIHOOD: Testing action space granularity")
    print("="*70)

    # Test different numbers of directions
    n_directions_list = [8, 16, 32, 64]
    action_noise = 2.0  # Motor noise parameter

    print(f"\nAction noise (κ) = {action_noise}")
    print(f"{'-'*70}\n")

    results = []
    for n_dirs in n_directions_list:
        total_ll, mean_ll = test_model_continuous(
            n_dirs, action_noise, trial_files, belief_model
        )
        results.append({
            'n_directions': n_dirs,
            'total_log_lik': total_ll,
            'mean_log_lik': mean_ll
        })
        print(f"Directions: {n_dirs:3d} | Total LL: {total_ll:9.2f} | "
              f"Mean LL: {mean_ll:8.2f}")

    print(f"\n{'='*70}")
    print("\nBest model:")
    best = max(results, key=lambda x: x['total_log_lik'])
    print(f"  Directions: {best['n_directions']}")
    print(f"  Total log-likelihood: {best['total_log_lik']:.2f}")
    print(f"  Mean log-likelihood: {best['mean_log_lik']:.2f}")

    # Now test different action noise values with best n_directions
    print(f"\n{'='*70}")
    print("Testing different action noise (motor precision) values")
    print(f"{'='*70}")

    best_n_dirs = best['n_directions']
    noise_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\nUsing {best_n_dirs} directions")
    print(f"{'-'*70}\n")

    noise_results = []
    for noise in noise_values:
        total_ll, mean_ll = test_model_continuous(
            best_n_dirs, noise, trial_files, belief_model
        )
        noise_results.append({
            'action_noise': noise,
            'total_log_lik': total_ll,
            'mean_log_lik': mean_ll
        })
        print(f"Action noise: {noise:5.1f} | Total LL: {total_ll:9.2f} | "
              f"Mean LL: {mean_ll:8.2f}")

    print(f"\n{'='*70}")
    print("\nBest action noise:")
    best_noise = max(noise_results, key=lambda x: x['total_log_lik'])
    print(f"  κ = {best_noise['action_noise']}")
    print(f"  Total log-likelihood: {best_noise['total_log_lik']:.2f}")
    print(f"  Mean log-likelihood: {best_noise['mean_log_lik']:.2f}")

    # Compare with and without beliefs
    print(f"\n{'='*70}")
    print("Comparing models with vs without beliefs")
    print(f"{'='*70}\n")

    # With beliefs (what we just tested)
    with_beliefs_ll = best_noise['total_log_lik']

    # Without beliefs (set belief = 1.0)
    decision_model = UtilityDecisionModel(
        n_directions=best_n_dirs,
        temperature=1.0,
        w_stag=1.0,
        w_rabbit=1.0
    )

    total_log_lik_no_belief = 0
    for trial_file in trial_files:
        trial_data = load_trial(trial_file)
        trial_with_beliefs = belief_model.run_trial(trial_data)

        # Override beliefs with constant 1.0
        trial_no_belief = trial_with_beliefs.copy()
        trial_no_belief['p1_belief_p2_stag'] = 1.0
        trial_no_belief['p2_belief_p1_stag'] = 1.0

        p1_log_lik, _, _ = decision_model.evaluate_trial_continuous(
            trial_no_belief,
            player='player1',
            belief_column='p1_belief_p2_stag',
            action_noise=best_noise['action_noise']
        )
        p2_log_lik, _, _ = decision_model.evaluate_trial_continuous(
            trial_no_belief,
            player='player2',
            belief_column='p2_belief_p1_stag',
            action_noise=best_noise['action_noise']
        )

        total_log_lik_no_belief += p1_log_lik + p2_log_lik

    print(f"Without beliefs: {total_log_lik_no_belief:9.2f}")
    print(f"With beliefs:    {with_beliefs_ll:9.2f}")
    print(f"Improvement:     {with_beliefs_ll - total_log_lik_no_belief:9.2f}")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
