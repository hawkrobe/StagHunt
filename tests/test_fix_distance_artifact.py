#!/usr/bin/env python3
"""
Test that distance-dependent gain is causing the backwards likelihoods.

The decision model uses:
  gain = alignment * (1.0 / (1.0 + dist / 100.0))

This penalizes movements toward distant targets, which can make
the model incorrectly predict actions.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def trial6_data():
    """Load Trial 6 data."""
    trial_data = pd.read_csv('inputs/stag_hunt_coop_trial6_2024_08_24_0848.csv')
    if 'plater1_y' in trial_data.columns:
        trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})
    return trial_data


class TestDistanceArtifact:
    """
    Demonstrate that distance-dependent gain causes problems.
    """

    def test_distance_penalty_at_t120(self, trial6_data):
        """
        At t=120, check how distance affects gains.

        P2 is moving West toward rabbit.
        - Rabbit is far (~338 pixels)
        - Stag is closer (~491 pixels from origin, but check actual distance)

        If gains are penalized by distance, movements toward distant
        rabbit get low utility even if aligned.
        """
        row = trial6_data.iloc[120]

        # P2's position
        p2_x = row['player2_x']
        p2_y = row['player2_y']

        # Target positions
        stag_x, stag_y = row['stag_x'], row['stag_y']
        rabbit_x, rabbit_y = row['rabbit_x'], row['rabbit_y']

        # Distances to targets
        dist_to_stag = np.sqrt((stag_x - p2_x)**2 + (stag_y - p2_y)**2)
        dist_to_rabbit = np.sqrt((rabbit_x - p2_x)**2 + (rabbit_y - p2_y)**2)

        # Observed movement
        row_prev = trial6_data.iloc[119]
        dx = row['player2_x'] - row_prev['player2_x']
        dy = row['player2_y'] - row_prev['player2_y']
        movement_angle = np.arctan2(dy, dx)

        # Angles to targets
        angle_to_stag = np.arctan2(stag_y - p2_y, stag_x - p2_x)
        angle_to_rabbit = np.arctan2(rabbit_y - p2_y, rabbit_x - p2_x)

        # Alignments (cosine similarity)
        align_stag = np.cos(movement_angle - angle_to_stag)
        align_rabbit = np.cos(movement_angle - angle_to_rabbit)

        # Distance penalties (current model)
        penalty_stag = 1.0 / (1.0 + dist_to_stag / 100.0)
        penalty_rabbit = 1.0 / (1.0 + dist_to_rabbit / 100.0)

        # Gains with penalty (current model)
        gain_stag_penalized = align_stag * penalty_stag
        gain_rabbit_penalized = align_rabbit * penalty_rabbit

        # Gains without penalty (proposed fix)
        gain_stag_clean = align_stag
        gain_rabbit_clean = align_rabbit

        print(f"\nAt t=120 (P2 moving West toward rabbit):")
        print(f"\nDistances:")
        print(f"  To stag:   {dist_to_stag:6.1f} px")
        print(f"  To rabbit: {dist_to_rabbit:6.1f} px")

        print(f"\nAlignments (no distance penalty):")
        print(f"  Stag:   {align_stag:+.3f}")
        print(f"  Rabbit: {align_rabbit:+.3f}")
        print(f"  → Rabbit is {abs(align_rabbit/align_stag):.2f}× more aligned ✓")

        print(f"\nDistance penalties:")
        print(f"  Stag:   {penalty_stag:.3f}")
        print(f"  Rabbit: {penalty_rabbit:.3f}")

        print(f"\nGains WITH distance penalty (current model):")
        print(f"  Stag:   {gain_stag_penalized:+.3f}")
        print(f"  Rabbit: {gain_rabbit_penalized:+.3f}")

        if abs(gain_stag_penalized) > abs(gain_rabbit_penalized):
            print(f"  ✗ WRONG! Stag has higher gain despite lower alignment")
            print(f"    This causes backwards likelihoods!")
        else:
            print(f"  ✓ Correct: Rabbit has higher gain")

        print(f"\nGains WITHOUT distance penalty (proposed fix):")
        print(f"  Stag:   {gain_stag_clean:+.3f}")
        print(f"  Rabbit: {gain_rabbit_clean:+.3f}")

        if abs(gain_rabbit_clean) > abs(gain_stag_clean):
            print(f"  ✓ Correct: Rabbit has higher gain")
        else:
            print(f"  ✗ Something else is wrong")

        # The problem: rabbit alignment is negative (moving away)
        # So even without distance penalty, it has negative gain
        # But stag ALSO has negative alignment!
        # The less negative one will be chosen

        print(f"\nDIAGNOSIS:")
        if align_stag < 0 and align_rabbit < 0:
            print(f"  Both alignments are NEGATIVE (moving away from both targets)")
            print(f"  Stag: {align_stag:.3f}, Rabbit: {align_rabbit:.3f}")
            print(f"  Model will choose the LESS negative option")

            if abs(align_rabbit) < abs(align_stag):
                print(f"  → Rabbit is less negative (closer to movement direction) ✓")
            else:
                print(f"  → Stag is less negative ✗")

    def test_action_predictions_with_distance_weighting(self, trial6_data):
        """
        Show that distance weighting causes wrong action predictions.
        """
        from models.decision_model_coordinated import CoordinatedDecisionModel

        row = trial6_data.iloc[120]

        model = CoordinatedDecisionModel(
            n_directions=8,
            temperature=3.049,
            timing_tolerance=150.0,
            speed=1.0
        )

        # Predict actions for each intention
        probs_stag, utils_stag = model.compute_action_probabilities(
            player_x=row['player2_x'], player_y=row['player2_y'],
            partner_x=row['player1_x'], partner_y=row['player1_y'],
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=row['value'],
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=1.0,
            belief_partner_stag=0.5
        )

        probs_rabbit, utils_rabbit = model.compute_action_probabilities(
            player_x=row['player2_x'], player_y=row['player2_y'],
            partner_x=row['player1_x'], partner_y=row['player1_y'],
            stag_x=row['rabbit_x'], stag_y=row['rabbit_y'], stag_value=1.0,  # Swap
            rabbit_x=row['stag_x'], rabbit_y=row['stag_y'], rabbit_value=row['value'],
            belief_partner_stag=0.5
        )

        actions = model.action_angles

        print(f"\nAction predictions WITH distance weighting:")
        print(f"\nIf intending STAG:")
        for i, (angle, prob, util) in enumerate(zip(actions, probs_stag, utils_stag)):
            print(f"  {np.degrees(angle):6.1f}°: prob={prob:.3f}, util={util:+.3f}")

        print(f"\nIf intending RABBIT:")
        for i, (angle, prob, util) in enumerate(zip(actions, probs_rabbit, utils_rabbit)):
            print(f"  {np.degrees(angle):6.1f}°: prob={prob:.3f}, util={util:+.3f}")

        max_stag = np.argmax(probs_stag)
        max_rabbit = np.argmax(probs_rabbit)

        print(f"\nMost likely actions:")
        print(f"  Stag:   {np.degrees(actions[max_stag]):6.1f}° (prob={probs_stag[max_stag]:.3f})")
        print(f"  Rabbit: {np.degrees(actions[max_rabbit]):6.1f}° (prob={probs_rabbit[max_rabbit]:.3f})")

        row_prev = trial6_data.iloc[119]
        dx = row['player2_x'] - row_prev['player2_x']
        dy = row['player2_y'] - row_prev['player2_y']
        obs_angle = np.arctan2(dy, dx)

        print(f"\nObserved: {np.degrees(obs_angle):6.1f}°")

        # Distance from observed to each prediction
        def angle_dist(a1, a2):
            diff = (a1 - a2) % (2*np.pi)
            if diff > np.pi:
                diff = 2*np.pi - diff
            return diff

        dist_to_stag_pred = np.degrees(angle_dist(obs_angle, actions[max_stag]))
        dist_to_rabbit_pred = np.degrees(angle_dist(obs_angle, actions[max_rabbit]))

        print(f"\nDistance from observed:")
        print(f"  To stag prediction:   {dist_to_stag_pred:.1f}°")
        print(f"  To rabbit prediction: {dist_to_rabbit_pred:.1f}°")

        if dist_to_stag_pred < dist_to_rabbit_pred:
            print(f"\n  ✗ Stag prediction is closer! This causes backwards likelihoods.")
        else:
            print(f"\n  ✓ Rabbit prediction is closer.")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
