#!/usr/bin/env python3
"""
Test different timing_tolerance values to fix the decision model.

Goal: Find a timing_tolerance that allows the decision model to properly
distinguish intentions and update beliefs.
"""

import pytest
import pandas as pd
import numpy as np
from models.belief_model_decision import BayesianIntentionModelWithDecision


@pytest.fixture
def trial6_data():
    """Load Trial 6 data."""
    trial_data = pd.read_csv('data/stag_hunt_coop_trial6_2024_08_24_0848.csv')
    if 'plater1_y' in trial_data.columns:
        trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})
    return trial_data


class TestTimingToleranceFix:
    """
    Test different timing_tolerance values to find one that works.
    """

    @pytest.mark.parametrize("timing_tolerance", [
        1.0,     # Original (too strict)
        10.0,    # 10× larger
        50.0,    # 50× larger
        100.0,   # 100× larger (recommended)
        150.0,   # 150× larger (recommended)
        200.0,   # 200× larger
    ])
    def test_timing_tolerance_values(self, trial6_data, timing_tolerance):
        """
        Test decision model with different timing_tolerance values.

        Success criteria:
        - Beliefs should vary (variance > 0.01)
        - Likelihoods should favor correct intentions
        - Beliefs should update on goal switches
        """
        # Create model with this timing tolerance
        model = BayesianIntentionModelWithDecision(
            decision_model_params={
                'temperature': 3.049,
                'timing_tolerance': timing_tolerance,
                'action_noise': 10.0,
                'n_directions': 8
            },
            prior_stag=0.5,
            belief_bounds=(0.01, 0.99)
        )

        # Run on Trial 6
        results = model.run_trial(trial6_data)

        # Check belief variance
        belief_variance = results['p1_belief_p2_stag'].var()
        belief_range = results['p1_belief_p2_stag'].max() - results['p1_belief_p2_stag'].min()

        # Check likelihood at t=120 (P2 moving toward rabbit)
        row_prev = trial6_data.iloc[119]
        row = trial6_data.iloc[120]

        dx = row['player2_x'] - row_prev['player2_x']
        dy = row['player2_y'] - row_prev['player2_y']
        observed_angle = np.arctan2(dy, dx)

        lik_stag = model.likelihood_movement_given_intention(
            observed_angle,
            player_x=row['player1_x'], player_y=row['player1_y'],
            partner_x=row['player2_x'], partner_y=row['player2_y'],
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=row['value'],
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=1.0,
            partner_believes_stag=0.5
        )

        lik_rabbit = model.likelihood_movement_given_intention(
            observed_angle,
            player_x=row['player1_x'], player_y=row['player1_y'],
            partner_x=row['player2_x'], partner_y=row['player2_y'],
            stag_x=row['rabbit_x'], stag_y=row['rabbit_y'], stag_value=1.0,
            rabbit_x=row['stag_x'], rabbit_y=row['stag_y'], rabbit_value=row['value'],
            partner_believes_stag=0.5
        )

        ratio = lik_rabbit / lik_stag if lik_stag > 0 else 0

        # Coordination probability at t=120
        dist_p1 = np.sqrt((row['stag_x'] - row['player1_x'])**2 +
                         (row['stag_y'] - row['player1_y'])**2)
        dist_p2 = np.sqrt((row['stag_x'] - row['player2_x'])**2 +
                         (row['stag_y'] - row['player2_y'])**2)
        timing_align = np.exp(-0.5 * ((dist_p1 - dist_p2) / timing_tolerance)**2)
        coord_prob = 0.5 * timing_align

        print(f"\nτ = {timing_tolerance:6.1f}:")
        print(f"  Belief variance:      {belief_variance:.6f}")
        print(f"  Belief range:         {belief_range:.3f}")
        print(f"  Likelihood ratio:     {ratio:.2f}:1 {'✓' if ratio > 1.5 else '✗'}")
        print(f"  Coord prob (t=120):   {coord_prob:.6f}")

        # Report success
        success = (belief_variance > 0.01 and ratio > 1.5)
        if success:
            print(f"  ✓ WORKS! Beliefs update properly.")
        else:
            print(f"  ✗ Doesn't work yet.")

    def test_optimal_timing_tolerance(self, trial6_data):
        """
        Find the optimal timing_tolerance based on our diagnostics.

        We want:
        1. Non-zero coordination probability for reasonable distances
        2. Beliefs that update substantially
        3. Correct likelihood ratios
        """
        print("\n" + "="*70)
        print("FINDING OPTIMAL TIMING TOLERANCE")
        print("="*70)

        # Test range of values
        tolerance_values = [1.0, 10.0, 50.0, 100.0, 150.0, 200.0, 300.0]
        results = []

        for tau in tolerance_values:
            model = BayesianIntentionModelWithDecision(
                decision_model_params={
                    'temperature': 3.049,
                    'timing_tolerance': tau,
                    'action_noise': 10.0,
                    'n_directions': 8
                },
                prior_stag=0.5,
                belief_bounds=(0.01, 0.99)
            )

            trial_results = model.run_trial(trial6_data)
            variance = trial_results['p1_belief_p2_stag'].var()
            belief_range = trial_results['p1_belief_p2_stag'].max() - trial_results['p1_belief_p2_stag'].min()

            results.append({
                'tau': tau,
                'variance': variance,
                'range': belief_range
            })

        print("\nResults:")
        print(f"{'τ':>8} {'Variance':>12} {'Range':>8} {'Status':>10}")
        print("-" * 45)

        best_tau = None
        best_variance = 0

        for r in results:
            status = "✓ Good" if r['variance'] > 0.01 else "✗ Too low"
            print(f"{r['tau']:>8.1f} {r['variance']:>12.6f} {r['range']:>8.3f} {status:>10}")

            if r['variance'] > best_variance:
                best_variance = r['variance']
                best_tau = r['tau']

        print("\n" + "="*70)
        print(f"RECOMMENDATION: Use timing_tolerance = {best_tau}")
        print(f"  Achieves variance = {best_variance:.6f}")
        print("="*70)

        # Verify the best one works
        assert best_variance > 0.01, \
            f"Best timing_tolerance ({best_tau}) should give variance > 0.01 (got {best_variance:.6f})"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
