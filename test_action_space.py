"""
Test how action space granularity affects model performance.

Compare models with 8, 16, 32, 64 discrete directions.
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


def test_model(n_directions, trial_files, belief_model):
    """Test model with specified number of directions."""
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

        p1_log_lik, _, _ = decision_model.evaluate_trial(
            trial_with_beliefs,
            player='player1',
            belief_column='p1_belief_p2_stag'
        )
        p2_log_lik, _, _ = decision_model.evaluate_trial(
            trial_with_beliefs,
            player='player2',
            belief_column='p2_belief_p1_stag'
        )

        total_log_lik += p1_log_lik + p2_log_lik

    mean_log_lik = total_log_lik / len(trial_files)
    return total_log_lik, mean_log_lik


def main():
    # Load all trials
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*.csv'))

    # Initialize belief model
    belief_model = BayesianIntentionModel(
        prior_stag=0.5,
        concentration=1.5,
        belief_bounds=(0.01, 0.99)
    )

    print("Testing action space granularity...")
    print(f"{'='*60}\n")

    # Test different numbers of directions
    n_directions_list = [8, 16, 32, 64, 128]

    results = []
    for n_dirs in n_directions_list:
        total_ll, mean_ll = test_model(n_dirs, trial_files, belief_model)
        results.append({
            'n_directions': n_dirs,
            'total_log_lik': total_ll,
            'mean_log_lik': mean_ll
        })
        print(f"Directions: {n_dirs:3d} | Total LL: {total_ll:9.2f} | Mean LL: {mean_ll:8.2f}")

    print(f"\n{'='*60}")
    print("\nBest model:")
    best = max(results, key=lambda x: x['total_log_lik'])
    print(f"  Directions: {best['n_directions']}")
    print(f"  Total log-likelihood: {best['total_log_lik']:.2f}")
    print(f"  Mean log-likelihood: {best['mean_log_lik']:.2f}")


if __name__ == '__main__':
    main()
