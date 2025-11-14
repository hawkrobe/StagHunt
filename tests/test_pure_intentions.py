#!/usr/bin/env python3
"""
Test that pure intention modeling works correctly.

Key insight: When modeling intentions for inverse inference, we should zero out
the alternative option symmetrically:
- Stag intention: set rabbit_value=0 → only considers stag
- Rabbit intention: set stag_value=0 → only considers rabbit

This ensures that even when P_coord ≈ 0, the stag intention predicts flat/indifferent
actions rather than actively preferring rabbit.
"""

import pytest
import pandas as pd
import numpy as np
from models.belief_model_decision import BayesianIntentionModelWithDecision


class TestPureIntentions:

    @pytest.fixture
    def trial6_data(self):
        """Load Trial 6 for testing."""
        df = pd.read_csv('inputs/stag_hunt_coop_trial6_2024_08_24_0848.csv')
        if 'plater1_y' in df.columns:
            df = df.rename(columns={'plater1_y': 'player1_y'})
        return df

    @pytest.fixture
    def model(self):
        """Fixed decision model."""
        return BayesianIntentionModelWithDecision(
            decision_model_params={
                'temperature': 3.049,
                'timing_tolerance': 150.0,
                'action_noise': 5.0,
                'n_directions': 8
            },
            prior_stag=0.5,
            belief_bounds=(0.01, 0.99)
        )

    def test_stag_intention_zeros_rabbit(self, trial6_data, model):
        """
        When modeling stag intention, rabbit_value should be 0.

        This ensures utility is U = V_stag × P_coord × gain_stag (no rabbit term).
        Even if P_coord ≈ 0, this gives flat probabilities, not preference for rabbit.
        """
        row = trial6_data.iloc[220]

        # Stag intention: only considers stag
        probs_stag, utils_stag = model.decision_model.compute_action_probabilities(
            player_x=row['player1_x'], player_y=row['player1_y'],
            partner_x=row['player2_x'], partner_y=row['player2_y'],
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=row['value'],
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=0.0,  # ZERO
            belief_partner_stag=0.472
        )

        # Get most likely action
        max_idx = np.argmax(probs_stag)
        best_angle = model.decision_model.action_angles[max_idx]

        # Angle toward stag from P1
        angle_to_stag = np.arctan2(row['stag_y'] - row['player1_y'],
                                    row['stag_x'] - row['player1_x'])

        # Best action should be toward stag (or at least not toward rabbit)
        angle_to_rabbit = np.arctan2(row['rabbit_y'] - row['player1_y'],
                                      row['rabbit_x'] - row['player1_x'])

        dist_to_stag = abs(best_angle - angle_to_stag)
        if dist_to_stag > np.pi:
            dist_to_stag = 2*np.pi - dist_to_stag

        dist_to_rabbit = abs(best_angle - angle_to_rabbit)
        if dist_to_rabbit > np.pi:
            dist_to_rabbit = 2*np.pi - dist_to_rabbit

        print(f"\nStag intention (rabbit_value=0):")
        print(f"  Best action: {np.degrees(best_angle):.1f}°")
        print(f"  Distance to stag: {np.degrees(dist_to_stag):.1f}°")
        print(f"  Distance to rabbit: {np.degrees(dist_to_rabbit):.1f}°")

        # Should prefer stag direction over rabbit
        assert dist_to_stag < dist_to_rabbit, \
            f"Stag intention should prefer stag direction (closer to stag than rabbit)"

    def test_rabbit_intention_zeros_stag(self, trial6_data, model):
        """
        When modeling rabbit intention, stag_value should be 0.

        This ensures utility is U = V_rabbit × gain_rabbit (no stag term).
        """
        row = trial6_data.iloc[220]

        # Rabbit intention: only considers rabbit
        probs_rabbit, utils_rabbit = model.decision_model.compute_action_probabilities(
            player_x=row['player1_x'], player_y=row['player1_y'],
            partner_x=row['player2_x'], partner_y=row['player2_y'],
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=0.0,  # ZERO
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=1.0,
            belief_partner_stag=0.0
        )

        # Get most likely action
        max_idx = np.argmax(probs_rabbit)
        best_angle = model.decision_model.action_angles[max_idx]

        # Angle toward rabbit from P1
        angle_to_rabbit = np.arctan2(row['rabbit_y'] - row['player1_y'],
                                      row['rabbit_x'] - row['player1_x'])

        # Angle toward stag from P1
        angle_to_stag = np.arctan2(row['stag_y'] - row['player1_y'],
                                    row['stag_x'] - row['player1_x'])

        dist_to_rabbit = abs(best_angle - angle_to_rabbit)
        if dist_to_rabbit > np.pi:
            dist_to_rabbit = 2*np.pi - dist_to_rabbit

        dist_to_stag = abs(best_angle - angle_to_stag)
        if dist_to_stag > np.pi:
            dist_to_stag = 2*np.pi - dist_to_stag

        print(f"\nRabbit intention (stag_value=0):")
        print(f"  Best action: {np.degrees(best_angle):.1f}°")
        print(f"  Distance to rabbit: {np.degrees(dist_to_rabbit):.1f}°")
        print(f"  Distance to stag: {np.degrees(dist_to_stag):.1f}°")

        # Should prefer rabbit direction
        assert dist_to_rabbit < dist_to_stag, \
            f"Rabbit intention should prefer rabbit direction"

    def test_likelihoods_discriminative_even_when_coordination_impossible(self, trial6_data, model):
        """
        Even when P_coord ≈ 0 (coordination impossible), likelihoods should be discriminative.

        At t=220 in Trial 6:
        - P1 is 977 units from stag, P2 is 522 units → timing alignment ≈ 0.01
        - P_coord = belief × timing ≈ 0.005 (essentially zero)
        - P1 moves toward rabbit (-135°)

        With OLD approach (didn't zero out rabbit for stag intention):
        - Both intentions predicted moving toward rabbit → likelihoods identical

        With NEW approach (symmetric pure intentions):
        - Stag intention: flat/indifferent probabilities
        - Rabbit intention: clear preference for rabbit
        - Likelihoods are discriminative!
        """
        row_prev = trial6_data.iloc[219]
        row = trial6_data.iloc[220]

        # P1's movement
        dx = row['player1_x'] - row_prev['player1_x']
        dy = row['player1_y'] - row_prev['player1_y']
        observed_angle = np.arctan2(dy, dx)

        print(f"\nAt t=220 (coordination impossible):")
        print(f"  P1→stag distance: 977 units")
        print(f"  P2→stag distance: 522 units")
        print(f"  Timing alignment: ~0.01")
        print(f"  P_coord ≈ 0.005")
        print(f"  P1 observed movement: {np.degrees(observed_angle):.1f}°")

        # Likelihood under stag intention
        lik_stag = model.likelihood_movement_given_intention(
            observed_angle,
            player_x=row['player2_x'], player_y=row['player2_y'],
            partner_x=row['player1_x'], partner_y=row['player1_y'],
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=row['value'],
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=0.0,
            partner_believes_stag=0.472
        )

        # Likelihood under rabbit intention
        lik_rabbit = model.likelihood_movement_given_intention(
            observed_angle,
            player_x=row['player2_x'], player_y=row['player2_y'],
            partner_x=row['player1_x'], partner_y=row['player1_y'],
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=0.0,
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=1.0,
            partner_believes_stag=0.0
        )

        ratio = lik_rabbit / lik_stag if lik_stag > 0 else 0

        print(f"  P(move | stag): {lik_stag:.6f}")
        print(f"  P(move | rabbit): {lik_rabbit:.6f}")
        print(f"  Likelihood ratio: {ratio:.2f}:1")

        # Likelihoods should be discriminative (ratio significantly > 1 or < 1)
        # Since P1 moves perfectly toward rabbit, ratio should favor rabbit
        assert ratio > 1.5, \
            f"Rabbit intention should be more likely (ratio {ratio:.2f} should be > 1.5)"

        print(f"  ✓ Likelihoods discriminative even with P_coord ≈ 0!")

    def test_belief_update_implementation_uses_pure_intentions(self, trial6_data, model):
        """
        Verify that the belief update code correctly uses pure intentions.

        This tests the actual implementation in belief_model_decision.py.
        """
        row_prev = trial6_data.iloc[219]
        row = trial6_data.iloc[220]

        # Run belief update for P2's belief about P1
        # player = P2 (observer), partner = P1 (being observed)
        belief_before = 0.472
        belief_after = model.update_belief(
            belief_before,
            player_x=row['player2_x'], player_y=row['player2_y'],  # P2 position (observer)
            partner_x_prev=row_prev['player1_x'], partner_y_prev=row_prev['player1_y'],  # P1 previous
            partner_x_curr=row['player1_x'], partner_y_curr=row['player1_y'],  # P1 current
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=row['value'],
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=1.0
        )

        print(f"\nBelief update at t=220:")
        print(f"  Before: {belief_before:.3f}")
        print(f"  After:  {belief_after:.3f}")
        print(f"  Change: {belief_after - belief_before:.3f}")

        # Belief should decrease (P1 moving toward rabbit)
        assert belief_after < belief_before, \
            f"Belief should decrease when P1 moves toward rabbit"

        # Should be a meaningful update (not stuck)
        assert abs(belief_after - belief_before) > 0.05, \
            f"Belief should update meaningfully (change > 0.05)"

        print(f"  ✓ Belief updates correctly!")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
