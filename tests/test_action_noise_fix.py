#!/usr/bin/env python3
"""
Test different action_noise values to fix likelihood ratios.

The action_noise parameter (κ in von Mises distribution) controls how noisy
action execution is. If it's too high, all actions become equally likely.
"""

import pytest
import pandas as pd
import numpy as np
from models.belief_model_decision import BayesianIntentionModelWithDecision


@pytest.fixture
def trial6_data():
    """Load Trial 6 data."""
    trial_data = pd.read_csv('inputs/stag_hunt_coop_trial6_2024_08_24_0848.csv')
    if 'plater1_y' in trial_data.columns:
        trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})
    return trial_data


class TestActionNoiseFix:
    """
    Test if reducing action_noise fixes the likelihood ratios.

    Current: action_noise = 10.0
    This might be too HIGH, making all movements equally likely.

    Lower action_noise → sharper von Mises → more discriminative likelihoods
    """

    @pytest.mark.parametrize("action_noise,timing_tolerance", [
        (10.0, 150.0),   # Current
        (5.0, 150.0),    # Half the noise
        (2.0, 150.0),    # 1/5 the noise
        (1.0, 150.0),    # 1/10 the noise
        (0.5, 150.0),    # 1/20 the noise
    ])
    def test_action_noise_values(self, trial6_data, action_noise, timing_tolerance):
        """
        Test decision model with different action_noise values.

        At t=120, P2 moves West (~180°) toward rabbit.
        Likelihood ratio should favor rabbit > 1.5:1
        """
        model = BayesianIntentionModelWithDecision(
            decision_model_params={
                'temperature': 3.049,
                'timing_tolerance': timing_tolerance,
                'action_noise': action_noise,
                'n_directions': 8
            },
            prior_stag=0.5,
            belief_bounds=(0.01, 0.99)
        )

        # Check likelihood at t=120
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

        # Run full trial
        results = model.run_trial(trial6_data)
        variance = results['p1_belief_p2_stag'].var()

        print(f"\naction_noise = {action_noise:4.1f}, τ = {timing_tolerance:5.1f}:")
        print(f"  Likelihood ratio:     {ratio:.2f}:1 {'✓' if ratio > 1.5 else '✗'}")
        print(f"  Belief variance:      {variance:.6f}")

        # Also check what actions are predicted
        actions = model.decision_model.action_angles

        # For stag intention
        probs_stag, _ = model.decision_model.compute_action_probabilities(
            player_x=row['player2_x'], player_y=row['player2_y'],
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=row['value'],
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=1.0,
            belief_partner_stag=0.5,
            partner_x=row['player1_x'], partner_y=row['player1_y']
        )

        # For rabbit intention
        probs_rabbit, _ = model.decision_model.compute_action_probabilities(
            player_x=row['player2_x'], player_y=row['player2_y'],
            stag_x=row['rabbit_x'], stag_y=row['rabbit_y'], stag_value=1.0,
            rabbit_x=row['stag_x'], rabbit_y=row['stag_y'], rabbit_value=row['value'],
            belief_partner_stag=0.5,
            partner_x=row['player1_x'], partner_y=row['player1_y']
        )

        # Which direction does each intention favor?
        max_stag_idx = np.argmax(probs_stag)
        max_rabbit_idx = np.argmax(probs_rabbit)

        print(f"  Stag intention predicts:   {np.degrees(actions[max_stag_idx]):6.1f}° (prob={probs_stag[max_stag_idx]:.3f})")
        print(f"  Rabbit intention predicts: {np.degrees(actions[max_rabbit_idx]):6.1f}° (prob={probs_rabbit[max_rabbit_idx]:.3f})")
        print(f"  Observed movement:         {np.degrees(observed_angle):6.1f}°")

    def test_grid_search(self, trial6_data):
        """
        Grid search over action_noise and timing_tolerance.
        """
        print("\n" + "="*80)
        print("GRID SEARCH: action_noise × timing_tolerance")
        print("="*80)

        noise_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        tolerance_values = [50.0, 100.0, 150.0, 200.0, 300.0]

        best_config = None
        best_score = 0  # Want high variance AND correct ratio

        print(f"\n{'Noise':>6} {'τ':>6} {'Var':>8} {'Ratio':>8} {'Score':>8}")
        print("-" * 50)

        for noise in noise_values:
            for tau in tolerance_values:
                model = BayesianIntentionModelWithDecision(
                    decision_model_params={
                        'temperature': 3.049,
                        'timing_tolerance': tau,
                        'action_noise': noise,
                        'n_directions': 8
                    },
                    prior_stag=0.5,
                    belief_bounds=(0.01, 0.99)
                )

                # Check likelihood at t=120
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

                results = model.run_trial(trial6_data)
                variance = results['p1_belief_p2_stag'].var()

                # Score: want variance > 0.02 AND ratio > 1.5
                score = variance * (1.0 if ratio > 1.5 else 0.5)

                status = ""
                if variance > 0.02 and ratio > 1.5:
                    status = " ✓✓"
                elif variance > 0.02 or ratio > 1.5:
                    status = " ✓"

                print(f"{noise:>6.1f} {tau:>6.0f} {variance:>8.4f} {ratio:>8.2f} {score:>8.4f}{status}")

                if score > best_score:
                    best_score = score
                    best_config = (noise, tau, variance, ratio)

        print("\n" + "="*80)
        if best_config:
            noise, tau, var, ratio = best_config
            print(f"BEST CONFIGURATION:")
            print(f"  action_noise = {noise}")
            print(f"  timing_tolerance = {tau}")
            print(f"  Variance = {var:.4f}")
            print(f"  Likelihood ratio = {ratio:.2f}:1")
        print("="*80)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
