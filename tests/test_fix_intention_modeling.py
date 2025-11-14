#!/usr/bin/env python3
"""
Test the correct way to model intentions for inverse inference.

The problem: The decision model uses:
  U = V_stag * P_coord * gain_stag + V_rabbit * gain_rabbit

For actual decision-making, this is correct (player considers both options).

But for intention inference, we need to model what a player would do
if they had a PURE intention:
  - Stag intention: Only consider stag
  - Rabbit intention: Only consider rabbit

Currently, we model rabbit intention by swapping stag/rabbit positions,
which gives backwards predictions when P_coord ≈ 0.
"""

import pytest
import pandas as pd
import numpy as np
from models.decision_model_coordinated import CoordinatedDecisionModel


@pytest.fixture
def trial6_data():
    """Load Trial 6 data."""
    trial_data = pd.read_csv('data/stag_hunt_coop_trial6_2024_08_24_0848.csv')
    if 'plater1_y' in trial_data.columns:
        trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})
    return trial_data


class TestIntentionModeling:
    """
    Show the correct way to model pure intentions.
    """

    def test_current_approach_is_wrong(self, trial6_data):
        """
        Current approach: Swap stag/rabbit positions for rabbit intention.

        This fails when P_coord ≈ 0 because:
          U_rabbit = V_stag * 0 * gain_rabbit + V_rabbit * gain_stag
               ≈ gain_stag  (predicts moving toward stag!)
        """
        model = CoordinatedDecisionModel(
            n_directions=8,
            temperature=3.049,
            timing_tolerance=150.0,
            speed=1.0
        )

        row = trial6_data.iloc[120]

        # Current approach for rabbit intention
        probs, utils = model.compute_action_probabilities(
            player_x=row['player2_x'], player_y=row['player2_y'],
            partner_x=row['player1_x'], partner_y=row['player1_y'],
            stag_x=row['rabbit_x'], stag_y=row['rabbit_y'], stag_value=1.0,  # SWAP
            rabbit_x=row['stag_x'], rabbit_y=row['stag_y'], rabbit_value=row['value'],  # SWAP
            belief_partner_stag=0.5
        )

        max_idx = np.argmax(probs)
        predicted_angle = model.action_angles[max_idx]

        # What angle points toward rabbit?
        angle_to_rabbit = np.arctan2(row['rabbit_y'] - row['player2_y'],
                                     row['rabbit_x'] - row['player2_x'])

        print(f"\nCurrent approach (swap positions):")
        print(f"  Predicted action: {np.degrees(predicted_angle):6.1f}°")
        print(f"  Angle to rabbit:  {np.degrees(angle_to_rabbit):6.1f}°")
        print(f"  Difference: {np.degrees(abs(predicted_angle - angle_to_rabbit)):6.1f}°")

        if abs(predicted_angle - angle_to_rabbit) > np.pi/4:
            print(f"  ✗ WRONG! Prediction doesn't point toward rabbit")
        else:
            print(f"  ✓ Correct")

    def test_correct_approach_set_other_value_to_zero(self, trial6_data):
        """
        Correct approach: Model pure rabbit intention by setting stag value to 0.

        This gives:
          U_rabbit = 0 * P_coord * gain_stag + V_rabbit * gain_rabbit
                   = V_rabbit * gain_rabbit  (correct!)
        """
        model = CoordinatedDecisionModel(
            n_directions=8,
            temperature=3.049,
            timing_tolerance=150.0,
            speed=1.0
        )

        row = trial6_data.iloc[120]

        # Correct approach: Set stag value to 0
        probs, utils = model.compute_action_probabilities(
            player_x=row['player2_x'], player_y=row['player2_y'],
            partner_x=row['player1_x'], partner_y=row['player1_y'],
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=0.0,  # Zero out stag!
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=1.0,
            belief_partner_stag=0.5  # Doesn't matter since stag_value=0
        )

        max_idx = np.argmax(probs)
        predicted_angle = model.action_angles[max_idx]

        # What angle points toward rabbit?
        angle_to_rabbit = np.arctan2(row['rabbit_y'] - row['player2_y'],
                                     row['rabbit_x'] - row['player2_x'])

        print(f"\nCorrect approach (set stag_value=0):")
        print(f"  Predicted action: {np.degrees(predicted_angle):6.1f}°")
        print(f"  Angle to rabbit:  {np.degrees(angle_to_rabbit):6.1f}°")
        print(f"  Difference: {np.degrees(abs(predicted_angle - angle_to_rabbit)):6.1f}°")

        if abs(predicted_angle - angle_to_rabbit) < np.pi/4:
            print(f"  ✓ CORRECT! Prediction points toward rabbit")
        else:
            print(f"  ✗ Still wrong")

    def test_stag_intention_modeling(self, trial6_data):
        """
        For stag intention, we can use the actual model (no modification needed).
        """
        model = CoordinatedDecisionModel(
            n_directions=8,
            temperature=3.049,
            timing_tolerance=150.0,
            speed=1.0
        )

        row = trial6_data.iloc[120]

        # Stag intention: use actual values
        probs, utils = model.compute_action_probabilities(
            player_x=row['player2_x'], player_y=row['player2_y'],
            partner_x=row['player1_x'], partner_y=row['player1_y'],
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=row['value'],
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=1.0,
            belief_partner_stag=0.5
        )

        max_idx = np.argmax(probs)
        predicted_angle = model.action_angles[max_idx]

        print(f"\nStag intention (actual model):")
        print(f"  Predicted action: {np.degrees(predicted_angle):6.1f}°")

        # Note: For stag intention with low P_coord, model might predict
        # moving toward rabbit (the fallback option). This is actually correct!

    def test_proposed_fix_summary(self):
        """
        Print summary of the fix.
        """
        print("\n" + "="*70)
        print("FIX FOR INVERSE PLANNING MODEL")
        print("="*70)

        print("\n🐛 PROBLEM:")
        print("  Current approach swaps stag/rabbit positions for rabbit intention.")
        print("  When P_coord ≈ 0, utility becomes:")
        print("    U = 0 * gain_rabbit + 1 * gain_stag")
        print("  Model predicts moving toward STAG (wrong!)")

        print("\n✅ SOLUTION:")
        print("  For rabbit intention: Set stag_value=0")
        print("    U = 0 * P_coord * gain_stag + V_rabbit * gain_rabbit")
        print("    U = V_rabbit * gain_rabbit  (correct!)")

        print("\n  For stag intention: Use actual model")
        print("    U = V_stag * P_coord * gain_stag + V_rabbit * gain_rabbit")
        print("    (Player considers both, prefers stag if P_coord high)")

        print("\n" + "="*70)
        print("IMPLEMENTATION:")
        print("="*70)

        print("\nModify belief_model_decision.py:")
        print("  likelihood_movement_given_intention() should:")
        print("    - For STAG intention: use actual stag/rabbit values")
        print("    - For RABBIT intention: set stag_value=0, rabbit_value=1")
        print("    - DO NOT swap positions!")

        print("\n" + "="*70)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
