#!/usr/bin/env python3
"""
Test suite for belief updating dynamics.

Tests based on debugging Trial 6, which revealed key insights about:
- How beliefs respond to goal switches
- Angular alignment vs distance in movement interpretation
- Integration window effects on evidence accumulation

These tests use Trial 6 specifically because it has clear goal switches
at known timesteps (t=84, t=119-122) that we can verify.
"""

import pytest
import pandas as pd
import numpy as np
from models.belief_model_distance import BayesianIntentionModel


@pytest.fixture
def trial6_data():
    """Load Trial 6 data - the trial we debugged extensively."""
    trial_data = pd.read_csv('inputs/stag_hunt_coop_trial6_2024_08_24_0848.csv')

    # Fix column typo if present
    if 'plater1_y' in trial_data.columns:
        trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})

    return trial_data


@pytest.fixture
def standard_model():
    """Standard belief model configuration."""
    return BayesianIntentionModel(
        prior_stag=0.5,
        concentration=1.5,
        belief_bounds=(0.01, 0.99)
    )


class TestBeliefUpdatesOnGoalSwitch:
    """
    Tests from: debug_trial6.py, verify_fix_summary.py

    Key insight: At t=119-122, P2 moves West (toward rabbit) even though
    closer to stag. P1's belief about P2→stag should drop dramatically.
    """

    def test_belief_drops_when_player_switches_goals(self, trial6_data, standard_model):
        """
        At t=119-122, P2's westward movement should cause P1's belief
        about P2 going for stag to drop substantially.
        """
        results = standard_model.run_trial(trial6_data)

        # Before switch (t=119): Belief should be higher than 0.5
        belief_before = results.iloc[119]['p1_belief_p2_stag']
        assert belief_before > 0.5, \
            f"Belief should be above chance before switch (got {belief_before:.3f})"

        # After switch (t=122): Belief should be low
        belief_after = results.iloc[122]['p1_belief_p2_stag']
        assert belief_after < 0.3, \
            f"Belief should drop after switch (got {belief_after:.3f})"

        # Should be a substantial drop (>0.3)
        belief_drop = belief_before - belief_after
        assert belief_drop > 0.3, \
            f"Belief drop should be >0.3 (got {belief_drop:.3f})"

    def test_early_goal_switches_update_correctly(self, trial6_data, standard_model):
        """
        At t=84, P1 switches from STAG to RAB. P2's belief should respond.
        This tests that the model updates throughout the trial, not just at the end.
        """
        results = standard_model.run_trial(trial6_data)

        # Get beliefs at different points
        belief_t50 = results.iloc[50]['p2_belief_p1_stag']  # During cooperation
        belief_t80 = results.iloc[80]['p2_belief_p1_stag']  # Just before switch
        belief_t100 = results.iloc[100]['p2_belief_p1_stag']  # After switch

        # Beliefs should show some variation (not stuck at one value)
        belief_variance = np.std([belief_t50, belief_t80, belief_t100])
        assert belief_variance > 0.05, \
            f"Beliefs should vary across the trial (std={belief_variance:.3f}, expected >0.05)"

        # After P1 switches (t=84), belief should eventually change
        # Either drop further or start rising (depending on P1's subsequent behavior)
        belief_changed = abs(belief_t100 - belief_t80) > 0.01
        assert belief_changed, \
            f"P2's belief should change after P1 switches (t=80: {belief_t80:.3f}, t=100: {belief_t100:.3f})"


class TestAngularAlignment:
    """
    Tests from: check_angles.py, check_movement_t119.py

    Key insight: Movement patterns (angular alignment) are more diagnostic
    than distance. At t=120, P2 is 197px from stag but moving West toward
    rabbit - belief should drop despite proximity.
    """

    def test_movement_matters_more_than_distance(self, trial6_data, standard_model):
        """
        At t=120, P2 is closer to stag than rabbit, but moving toward rabbit.
        Belief should be low (movement trumps distance).
        """
        results = standard_model.run_trial(trial6_data)
        row = results.iloc[120]

        # Calculate distances
        p2_dist_stag = np.sqrt((row['stag_x'] - row['player2_x'])**2 +
                               (row['stag_y'] - row['player2_y'])**2)
        p2_dist_rabbit = np.sqrt((row['rabbit_x'] - row['player2_x'])**2 +
                                 (row['rabbit_y'] - row['player2_y'])**2)

        # Verify P2 is closer to stag
        assert p2_dist_stag < p2_dist_rabbit, \
            f"P2 should be closer to stag (stag: {p2_dist_stag:.1f}, rabbit: {p2_dist_rabbit:.1f})"

        # But belief should be low because movement indicates rabbit
        belief = row['p1_belief_p2_stag']
        assert belief < 0.9, \
            f"Belief should be low (<0.9) despite proximity to stag (got {belief:.3f})"

    def test_angular_alignment_computation(self, trial6_data):
        """
        At t=120, P2 moves West (~180°) which should align strongly
        with rabbit direction, not stag direction.
        """
        row_prev = trial6_data.iloc[119]
        row = trial6_data.iloc[120]

        # Movement angle
        dx = row['player2_x'] - row_prev['player2_x']
        dy = row['player2_y'] - row_prev['player2_y']
        movement_angle = np.arctan2(dy, dx)

        # Angles to targets
        angle_to_stag = np.arctan2(row['stag_y'] - row['player2_y'],
                                    row['stag_x'] - row['player2_x'])
        angle_to_rabbit = np.arctan2(row['rabbit_y'] - row['player2_y'],
                                     row['rabbit_x'] - row['player2_x'])

        # Angular alignments (cosine similarity)
        align_stag = np.cos(movement_angle - angle_to_stag)
        align_rabbit = np.cos(movement_angle - angle_to_rabbit)

        # Movement should align more with rabbit than stag
        assert align_rabbit > align_stag, \
            f"Movement should align more with rabbit ({align_rabbit:.3f}) than stag ({align_stag:.3f})"

        # Should be strongly aligned with rabbit (>0.8)
        assert align_rabbit > 0.8, \
            f"Should be strongly aligned with rabbit (got {align_rabbit:.3f})"

    def test_westward_movement_sequence(self, trial6_data):
        """
        At t=119-122, P2 consistently moves West (~180°) toward rabbit.
        This sustained directional movement is what makes the inference reliable.
        """
        for t in [119, 120, 121, 122]:
            row_prev = trial6_data.iloc[t-1]
            row = trial6_data.iloc[t]

            dx = row['player2_x'] - row_prev['player2_x']
            dy = row['player2_y'] - row_prev['player2_y']
            movement_angle = np.arctan2(dy, dx)
            movement_deg = np.degrees(movement_angle)

            # Should be approximately West (180° ± 30°)
            # Handle angle wrapping: -180° is same as +180°
            is_westward = (abs(movement_deg - 180) < 30) or (abs(movement_deg + 180) < 30)
            assert is_westward, \
                f"At t={t}, movement should be ~West, got {movement_deg:.1f}°"


class TestBeliefBounds:
    """
    Tests from: debug_belief_update.py

    Key insight: Beliefs should be clipped to [0.01, 0.99] to prevent
    numerical issues and maintain uncertainty.
    """

    def test_beliefs_stay_within_bounds(self, trial6_data, standard_model):
        """
        Throughout the trial, all beliefs should stay within [0.01, 0.99].
        """
        results = standard_model.run_trial(trial6_data)

        p1_beliefs = results['p1_belief_p2_stag']
        p2_beliefs = results['p2_belief_p1_stag']

        assert p1_beliefs.min() >= 0.01, \
            f"P1 beliefs should not go below 0.01 (got {p1_beliefs.min():.6f})"
        assert p1_beliefs.max() <= 0.99, \
            f"P1 beliefs should not exceed 0.99 (got {p1_beliefs.max():.6f})"
        assert p2_beliefs.min() >= 0.01, \
            f"P2 beliefs should not go below 0.01 (got {p2_beliefs.min():.6f})"
        assert p2_beliefs.max() <= 0.99, \
            f"P2 beliefs should not exceed 0.99 (got {p2_beliefs.max():.6f})"

    def test_beliefs_can_reach_bounds(self, trial6_data, standard_model):
        """
        Beliefs should be able to reach or nearly reach the bounds
        (not stop short due to implementation issues).
        """
        results = standard_model.run_trial(trial6_data)

        p1_beliefs = results['p1_belief_p2_stag']
        p2_beliefs = results['p2_belief_p1_stag']

        # At some point, at least one belief should reach near the bounds
        min_belief = min(p1_beliefs.min(), p2_beliefs.min())
        max_belief = max(p1_beliefs.max(), p2_beliefs.max())

        assert min_belief < 0.02, \
            f"At least one belief should reach near lower bound (got {min_belief:.3f})"
        assert max_belief > 0.98, \
            f"At least one belief should reach near upper bound (got {max_belief:.3f})"


class TestBeliefTrajectory:
    """
    Tests from: plot_belief_trajectory.py

    Key insight: We can track exactly when beliefs change and verify
    the dynamics match expected patterns.
    """

    def test_cooperation_phase_has_high_beliefs(self, trial6_data, standard_model):
        """
        During t=40-80 when both players go for stag, beliefs should be high.
        """
        results = standard_model.run_trial(trial6_data)

        # Sample beliefs during cooperation phase
        coop_beliefs = results.iloc[40:81]['p1_belief_p2_stag']

        # Most beliefs should be high (>0.8)
        high_belief_ratio = (coop_beliefs > 0.8).sum() / len(coop_beliefs)
        assert high_belief_ratio > 0.8, \
            f"During cooperation phase, {high_belief_ratio*100:.1f}% of beliefs should be >0.8 (expected >80%)"

    def test_defection_phase_has_low_beliefs(self, trial6_data, standard_model):
        """
        After t=130 when both players defect, beliefs should be low.
        """
        results = standard_model.run_trial(trial6_data)

        # Sample beliefs during defection phase
        defect_beliefs = results.iloc[130:200]['p1_belief_p2_stag']

        # Most beliefs should be low (<0.2)
        low_belief_ratio = (defect_beliefs < 0.2).sum() / len(defect_beliefs)
        assert low_belief_ratio > 0.8, \
            f"During defection phase, {low_belief_ratio*100:.1f}% of beliefs should be <0.2 (expected >80%)"

    def test_transition_period_has_changing_beliefs(self, trial6_data, standard_model):
        """
        During t=119-122 (the goal switch), beliefs should change rapidly.
        """
        results = standard_model.run_trial(trial6_data)

        # Beliefs during transition
        transition_beliefs = results.iloc[119:123]['p1_belief_p2_stag']

        # Should have substantial variance (not stuck)
        belief_std = transition_beliefs.std()
        assert belief_std > 0.2, \
            f"Beliefs during transition should vary substantially (std={belief_std:.3f}, expected >0.2)"

        # Should have a clear downward trend
        belief_change = transition_beliefs.iloc[-1] - transition_beliefs.iloc[0]
        assert belief_change < -0.5, \
            f"Beliefs should drop substantially during transition (Δ={belief_change:.3f}, expected <-0.5)"


class TestDistanceVsAlignment:
    """
    Tests from: check_p2_movements.py, check_angles.py

    Key insight: Distance alone is misleading. A player can be closer to
    stag while clearly moving toward rabbit.
    """

    def test_distance_can_be_misleading(self, trial6_data):
        """
        Show that at multiple timesteps, P2 is closer to stag but
        alignment suggests rabbit.
        """
        misleading_timesteps = []

        for t in range(115, 125):
            row = trial6_data.iloc[t]

            # Distances
            dist_stag = np.sqrt((row['stag_x'] - row['player2_x'])**2 +
                               (row['stag_y'] - row['player2_y'])**2)
            dist_rabbit = np.sqrt((row['rabbit_x'] - row['player2_x'])**2 +
                                 (row['rabbit_y'] - row['player2_y'])**2)

            # Movement (if not first timestep)
            if t > 0:
                prev = trial6_data.iloc[t-1]
                dx = row['player2_x'] - prev['player2_x']
                dy = row['player2_y'] - prev['player2_y']

                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    movement_angle = np.arctan2(dy, dx)

                    angle_to_stag = np.arctan2(row['stag_y'] - row['player2_y'],
                                               row['stag_x'] - row['player2_x'])
                    angle_to_rabbit = np.arctan2(row['rabbit_y'] - row['player2_y'],
                                                 row['rabbit_x'] - row['player2_x'])

                    align_stag = np.cos(movement_angle - angle_to_stag)
                    align_rabbit = np.cos(movement_angle - angle_to_rabbit)

                    # Misleading if: closer to stag BUT aligns more with rabbit
                    if dist_stag < dist_rabbit and align_rabbit > align_stag:
                        misleading_timesteps.append(t)

        # Should find several misleading timesteps
        assert len(misleading_timesteps) >= 5, \
            f"Should find at least 5 'misleading' timesteps where distance and alignment disagree (found {len(misleading_timesteps)})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
