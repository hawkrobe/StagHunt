"""
Test the decision model on actual trial data.

This script:
1. Loads all trial data
2. Runs the belief model to infer beliefs over time
3. Evaluates decision model log-likelihood
4. Shows how well the model predicts actual behavior
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

    # Fix the typo in column name if present
    if 'plater1_y' in df.columns:
        df = df.rename(columns={'plater1_y': 'player1_y'})

    # Remove rows with NaN values (malformed last row in CSVs)
    df = df.dropna()

    return df


def main():
    # Load all trials
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*.csv'))
    print(f"Found {len(trial_files)} trials\n")

    # Initialize models
    belief_model = BayesianIntentionModel(
        prior_stag=0.5,
        concentration=1.5,
        belief_bounds=(0.01, 0.99)
    )

    # Test different model configurations
    model_configs = [
        {
            'name': 'Baseline (no beliefs)',
            'n_directions': 16,
            'temperature': 1.0,
            'w_stag': 1.0,
            'w_rabbit': 1.0,
            'use_beliefs': False
        },
        {
            'name': 'With beliefs',
            'n_directions': 16,
            'temperature': 1.0,
            'w_stag': 1.0,
            'w_rabbit': 1.0,
            'use_beliefs': True
        },
    ]

    # Test each model configuration
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"Model: {config['name']}")
        print(f"{'='*60}")

        decision_model = UtilityDecisionModel(
            n_directions=config['n_directions'],
            temperature=config['temperature'],
            w_stag=config['w_stag'],
            w_rabbit=config['w_rabbit']
        )

        total_log_lik = 0
        trial_results = []

        for trial_file in trial_files:
            trial_num = Path(trial_file).stem.split('trial')[1].split('_')[0]

            # Load trial
            trial_data = load_trial(trial_file)

            # Run belief model to get beliefs
            trial_with_beliefs = belief_model.run_trial(trial_data)

            # Evaluate decision model
            if config['use_beliefs']:
                # Use actual beliefs
                p1_log_lik, p1_mean, results_p1 = decision_model.evaluate_trial(
                    trial_with_beliefs,
                    player='player1',
                    belief_column='p1_belief_p2_stag'
                )
                p2_log_lik, p2_mean, results_p2 = decision_model.evaluate_trial(
                    trial_with_beliefs,
                    player='player2',
                    belief_column='p2_belief_p1_stag'
                )
            else:
                # Fixed belief = 1.0 (ignore beliefs)
                trial_no_belief = trial_with_beliefs.copy()
                trial_no_belief['p1_belief_p2_stag'] = 1.0
                trial_no_belief['p2_belief_p1_stag'] = 1.0

                p1_log_lik, p1_mean, results_p1 = decision_model.evaluate_trial(
                    trial_no_belief,
                    player='player1',
                    belief_column='p1_belief_p2_stag'
                )
                p2_log_lik, p2_mean, results_p2 = decision_model.evaluate_trial(
                    trial_no_belief,
                    player='player2',
                    belief_column='p2_belief_p1_stag'
                )

            trial_log_lik = p1_log_lik + p2_log_lik
            total_log_lik += trial_log_lik

            # Get outcome
            final_event = trial_data.iloc[-1]['event']
            outcome_map = {
                0: 'No catch',
                1: 'P1 caught stag alone',
                2: 'P2 caught stag alone',
                3: 'P1 caught rabbit',
                4: 'P2 caught rabbit',
                5: 'Both caught stag (cooperation!)'
            }
            outcome = outcome_map.get(final_event, f'Unknown ({final_event})')

            trial_results.append({
                'trial': trial_num,
                'outcome': outcome,
                'log_lik': trial_log_lik,
                'p1_mean_log_lik': p1_mean,
                'p2_mean_log_lik': p2_mean
            })

            print(f"Trial {trial_num:>2s}: {outcome:30s} | "
                  f"Log-lik: {trial_log_lik:8.2f} | "
                  f"P1: {p1_mean:6.3f}, P2: {p2_mean:6.3f}")

        print(f"\n{'-'*60}")
        print(f"Total log-likelihood: {total_log_lik:.2f}")
        print(f"Mean log-lik per trial: {total_log_lik/len(trial_files):.2f}")
        print(f"{'='*60}\n")

    print("\nDone!")


if __name__ == '__main__':
    main()
