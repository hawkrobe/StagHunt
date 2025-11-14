#!/usr/bin/env python3
"""
Detailed debugging of decision model likelihood computation.

This test helps understand why the decision model produces backwards likelihoods.
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


@pytest.fixture
def decision_model():
    """Decision-based model."""
    return BayesianIntentionModelWithDecision(
        decision_model_params={
            'temperature': 3.049,
            'timing_tolerance': 0.865,
            'action_noise': 10.0,
            'n_directions': 8
        },
        prior_stag=0.5,
        belief_bounds=(0.01, 0.99)
    )


class TestDecisionModelLikelihood:
    """
    Debug the decision model's likelihood computation at t=120.

    At this timestep:
    - P2 is at (x, y)
    - P2 moves West (angle ≈ 180°) toward rabbit
    - Should give higher likelihood for "rabbit intention"
    """

    def test_movement_angle_computation(self, trial6_data):
        """Verify we're computing movement angle correctly."""
        row_prev = trial6_data.iloc[119]
        row = trial6_data.iloc[120]

        # Movement vector
        dx = row['player2_x'] - row_prev['player2_x']
        dy = row['player2_y'] - row_prev['player2_y']
        movement_angle = np.arctan2(dy, dx)

        print(f"\nMovement at t=120:")
        print(f"  P2 position: ({row['player2_x']:.1f}, {row['player2_y']:.1f})")
        print(f"  Movement: dx={dx:.1f}, dy={dy:.1f}")
        print(f"  Angle: {np.degrees(movement_angle):.1f}°")

        # Should be approximately West (-180° or +180°)
        movement_deg = np.degrees(movement_angle)
        is_westward = (abs(movement_deg - 180) < 30) or (abs(movement_deg + 180) < 30)
        assert is_westward, f"Movement should be westward, got {movement_deg:.1f}°"

    def test_target_angles(self, trial6_data):
        """Check angles to stag and rabbit."""
        row = trial6_data.iloc[120]

        # Angles from P2 to targets
        angle_to_stag = np.arctan2(row['stag_y'] - row['player2_y'],
                                    row['stag_x'] - row['player2_x'])
        angle_to_rabbit = np.arctan2(row['rabbit_y'] - row['player2_y'],
                                     row['rabbit_x'] - row['player2_x'])

        print(f"\nTarget angles from P2 at t=120:")
        print(f"  Stag at: ({row['stag_x']:.1f}, {row['stag_y']:.1f})")
        print(f"  Rabbit at: ({row['rabbit_x']:.1f}, {row['rabbit_y']:.1f})")
        print(f"  Angle to stag: {np.degrees(angle_to_stag):.1f}°")
        print(f"  Angle to rabbit: {np.degrees(angle_to_rabbit):.1f}°")

    def test_decision_model_action_probabilities(self, trial6_data, decision_model):
        """
        Check what actions the decision model predicts for stag vs rabbit intentions.

        This tests whether the decision model thinks:
        - "If P2 goes for stag, what movements are likely?"
        - "If P2 goes for rabbit, what movements are likely?"
        """
        row = trial6_data.iloc[120]

        # Get decision model's action probabilities for STAG intention
        # We need to call the internal decision model
        print(f"\nDecision model's predicted action probabilities:")
        print(f"\nIf P2 intends STAG (with belief about P1 = 0.5):")

        # Get actions and their angles
        actions = decision_model.decision_model.action_angles

        # Compute utilities for stag intention
        # P2's perspective: going for stag
        probs_stag, _ = decision_model.decision_model.compute_action_probabilities(
            player_x=row['player2_x'],
            player_y=row['player2_y'],
            stag_x=row['stag_x'],
            stag_y=row['stag_y'],
            stag_value=row['value'],
            rabbit_x=row['rabbit_x'],
            rabbit_y=row['rabbit_y'],
            rabbit_value=1.0,
            belief_partner_stag=0.5,  # P2's belief about P1
            partner_x=row['player1_x'],
            partner_y=row['player1_y']
        )

        # Find most likely action for stag
        max_idx_stag = np.argmax(probs_stag)
        print(f"  Most likely action: {np.degrees(actions[max_idx_stag]):.1f}° (prob={probs_stag[max_idx_stag]:.3f})")

        # Compute utilities for rabbit intention
        print(f"\nIf P2 intends RABBIT:")
        probs_rabbit, _ = decision_model.decision_model.compute_action_probabilities(
            player_x=row['player2_x'],
            player_y=row['player2_y'],
            stag_x=row['rabbit_x'],  # Swap stag/rabbit
            stag_y=row['rabbit_y'],
            stag_value=1.0,
            rabbit_x=row['stag_x'],
            rabbit_y=row['stag_y'],
            rabbit_value=row['value'],
            belief_partner_stag=0.5,  # Irrelevant for rabbit (no coordination)
            partner_x=row['player1_x'],
            partner_y=row['player1_y']
        )

        max_idx_rabbit = np.argmax(probs_rabbit)
        print(f"  Most likely action: {np.degrees(actions[max_idx_rabbit]):.1f}° (prob={probs_rabbit[max_idx_rabbit]:.3f})")

        # Check which intention predicts westward movement better
        # Westward is approximately 180° or -180°
        westward_angles = [a for a in actions if abs(np.degrees(a) - 180) < 45 or abs(np.degrees(a) + 180) < 45]

        if westward_angles:
            westward_prob_stag = sum(probs_stag[i] for i, a in enumerate(actions) if a in westward_angles)
            westward_prob_rabbit = sum(probs_rabbit[i] for i, a in enumerate(actions) if a in westward_angles)

            print(f"\nProbability of westward movement (~180°):")
            print(f"  If intending stag:   {westward_prob_stag:.3f}")
            print(f"  If intending rabbit: {westward_prob_rabbit:.3f}")

            if westward_prob_rabbit > westward_prob_stag:
                print("  ✓ Rabbit intention favors westward movement (correct!)")
            else:
                print("  ✗ Stag intention favors westward movement (WRONG!)")
                print("\n  This is the bug! The decision model thinks going for stag")
                print("  leads to westward movement, when it should be rabbit.")

    def test_coordination_probability(self, trial6_data, decision_model):
        """
        Check if coordination probability is affecting likelihoods.

        Maybe the issue is that:
        - Stag intention has LOW coordination prob → spreads out action probs
        - Spreading out could accidentally give higher prob to westward movement
        """
        row = trial6_data.iloc[120]

        # Manually compute coordination probability
        dist_p1_to_stag = np.sqrt((row['stag_x'] - row['player1_x'])**2 +
                                  (row['stag_y'] - row['player1_y'])**2)
        dist_p2_to_stag = np.sqrt((row['stag_x'] - row['player2_x'])**2 +
                                  (row['stag_y'] - row['player2_y'])**2)

        speed = 1.0
        t1 = dist_p1_to_stag / speed
        t2 = dist_p2_to_stag / speed

        timing_tolerance = 0.865
        timing_alignment = np.exp(-0.5 * ((t1 - t2) / timing_tolerance)**2)

        # With belief = 0.5
        coord_prob = 0.5 * timing_alignment

        print(f"\nCoordination calculation at t=120:")
        print(f"  P1 distance to stag: {dist_p1_to_stag:.1f}")
        print(f"  P2 distance to stag: {dist_p2_to_stag:.1f}")
        print(f"  Arrival time difference: {abs(t1-t2):.1f}")
        print(f"  Timing alignment: {timing_alignment:.3f}")
        print(f"  Coordination probability: {coord_prob:.3f}")

        if coord_prob < 0.1:
            print("\n  ⚠ Very low coordination probability!")
            print("  This might make stag utility very low,")
            print("  causing the model to predict random movement.")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
