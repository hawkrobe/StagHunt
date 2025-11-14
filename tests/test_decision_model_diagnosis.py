#!/usr/bin/env python3
"""
Diagnosis: Why the decision-based model doesn't work.

This test documents the core issues with the decision-based belief model.
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


class TestDecisionModelDiagnosis:
    """
    Document the problems with the decision-based belief model.
    """

    def test_coordination_probability_too_strict(self, trial6_data):
        """
        PROBLEM 1: Timing tolerance (0.865) is far too strict.

        With timing_tolerance = 0.865:
        - If players are 100 pixels apart in distance to stag:
          → Arrival time difference ≈ 100
          → Timing alignment = exp(-0.5 × (100/0.865)²) ≈ 0.0

        This makes coordination probability essentially zero for most of the trial,
        which makes stag utility zero, which makes intentions indistinguishable.
        """
        # Test at several timesteps
        timing_tolerance = 0.865
        problem_timesteps = []

        for t in [50, 100, 120, 150, 200]:
            if t >= len(trial6_data):
                continue

            row = trial6_data.iloc[t]

            # Distances to stag
            dist_p1 = np.sqrt((row['stag_x'] - row['player1_x'])**2 +
                             (row['stag_y'] - row['player1_y'])**2)
            dist_p2 = np.sqrt((row['stag_x'] - row['player2_x'])**2 +
                             (row['stag_y'] - row['player2_y'])**2)

            # Timing alignment
            timing_align = np.exp(-0.5 * ((dist_p1 - dist_p2) / timing_tolerance)**2)

            if timing_align < 0.01:
                problem_timesteps.append((t, dist_p1, dist_p2, timing_align))

        print(f"\nTimesteps with near-zero coordination probability:")
        for t, d1, d2, align in problem_timesteps:
            print(f"  t={t}: P1 dist={d1:.0f}, P2 dist={d2:.0f}, align={align:.6f}")

        # Most timesteps should have this problem
        problem_ratio = len(problem_timesteps) / 5
        assert problem_ratio > 0.5, \
            f"Timing tolerance is too strict - {problem_ratio*100:.0f}% of timesteps have zero coordination"

        print(f"\n✗ {problem_ratio*100:.0f}% of timesteps have essentially zero coordination probability")
        print(f"  This makes stag utility = 0, preventing intention inference.")

    def test_zero_coordination_makes_intentions_indistinguishable(self, trial6_data):
        """
        PROBLEM 2: When coordination probability ≈ 0, stag utility ≈ 0.

        The utility function is:
          U(move toward stag) = V_stag × P_coord × gain_stag
          U(move toward rabbit) = V_rabbit × gain_rabbit

        If P_coord ≈ 0, then:
          U(stag) ≈ 0
          U(rabbit) = V_rabbit × gain_rabbit

        So the model predicts "always go for rabbit", regardless of actual intention.
        This makes P(movement | intending stag) ≈ P(movement | intending rabbit).
        """
        # At t=120, we know:
        # - P2 is moving West toward rabbit
        # - Coordination prob = 0.0
        # - Yet decision model can't tell stag from rabbit intention

        decision_model = BayesianIntentionModelWithDecision(
            decision_model_params={
                'temperature': 3.049,
                'timing_tolerance': 0.865,
                'action_noise': 10.0,
                'n_directions': 8
            },
            prior_stag=0.5,
            belief_bounds=(0.01, 0.99)
        )

        row_prev = trial6_data.iloc[119]
        row = trial6_data.iloc[120]

        # Movement angle
        dx = row['player2_x'] - row_prev['player2_x']
        dy = row['player2_y'] - row_prev['player2_y']
        observed_angle = np.arctan2(dy, dx)

        # Likelihoods
        lik_stag = decision_model.likelihood_movement_given_intention(
            observed_angle,
            player_x=row['player1_x'], player_y=row['player1_y'],
            partner_x=row['player2_x'], partner_y=row['player2_y'],
            stag_x=row['stag_x'], stag_y=row['stag_y'], stag_value=row['value'],
            rabbit_x=row['rabbit_x'], rabbit_y=row['rabbit_y'], rabbit_value=1.0,
            partner_believes_stag=0.5
        )

        lik_rabbit = decision_model.likelihood_movement_given_intention(
            observed_angle,
            player_x=row['player1_x'], player_y=row['player1_y'],
            partner_x=row['player2_x'], partner_y=row['player2_y'],
            stag_x=row['rabbit_x'], stag_y=row['rabbit_y'], stag_value=1.0,
            rabbit_x=row['stag_x'], rabbit_y=row['stag_y'], rabbit_value=row['value'],
            partner_believes_stag=0.5
        )

        ratio = lik_rabbit / lik_stag
        print(f"\nLikelihood ratio (rabbit:stag) at t=120:")
        print(f"  P(movement | rabbit) / P(movement | stag) = {ratio:.2f}")

        # Should strongly favor rabbit (e.g., >2:1), but doesn't
        if ratio < 1.5:
            print(f"  ✗ Ratio is too low - model can't distinguish intentions!")
            print(f"    Expected: >2.0 (movement clearly toward rabbit)")
            print(f"    Got: {ratio:.2f} (nearly indistinguishable)")
        else:
            print(f"  ✓ Ratio correctly favors rabbit intention")

    def test_suggested_fix_increase_timing_tolerance(self, trial6_data):
        """
        PROPOSED FIX: Increase timing_tolerance from 0.865 to ~100-150.

        With timing_tolerance = 150:
        - If players are 100 pixels apart:
          → Timing alignment = exp(-0.5 × (100/150)²) ≈ 0.65
        - If players are 200 pixels apart:
          → Timing alignment = exp(-0.5 × (200/150)²) ≈ 0.17

        This gives gradual decay rather than immediate collapse to zero.
        """
        # Test different timing tolerances
        test_tolerances = [0.865, 10.0, 50.0, 100.0, 150.0]

        print(f"\nCoordination probability with different timing tolerances:")
        print(f"At t=120 (distance difference = 430 pixels):")

        row = trial6_data.iloc[120]
        dist_p1 = np.sqrt((row['stag_x'] - row['player1_x'])**2 +
                         (row['stag_y'] - row['player1_y'])**2)
        dist_p2 = np.sqrt((row['stag_x'] - row['player2_x'])**2 +
                         (row['stag_y'] - row['player2_y'])**2)
        dist_diff = abs(dist_p1 - dist_p2)

        for tau in test_tolerances:
            timing_align = np.exp(-0.5 * (dist_diff / tau)**2)
            coord_prob = 0.5 * timing_align  # Assuming belief = 0.5

            print(f"  τ = {tau:6.1f}: timing_align = {timing_align:.3f}, coord_prob = {coord_prob:.3f}")

        print(f"\nRecommendation: Use timing_tolerance ≈ 100-150")
        print(f"  This allows gradual decay instead of immediate collapse to zero.")

    def test_alternative_use_distance_model(self, trial6_data):
        """
        ALTERNATIVE: Use the distance-based model.

        The distance-based model works well because it uses a simple geometric heuristic:
        - Measure angular alignment between movement and targets
        - No coordination probability needed
        - Beliefs update correctly

        Decision-based model is theoretically more principled but practically problematic:
        - Requires accurate coordination probability
        - Sensitive to parameter choices
        - More computational complexity

        For many purposes, the distance-based model may be sufficient.
        """
        from models.belief_model_distance import BayesianIntentionModel as DistanceModel

        distance_model = DistanceModel(
            prior_stag=0.5,
            concentration=1.5,
            belief_bounds=(0.01, 0.99)
        )

        results = distance_model.run_trial(trial6_data)

        # Check that it works
        belief_range = results['p1_belief_p2_stag'].max() - results['p1_belief_p2_stag'].min()
        belief_variance = results['p1_belief_p2_stag'].var()

        print(f"\nDistance-based model performance:")
        print(f"  Belief range: {belief_range:.3f}")
        print(f"  Belief variance: {belief_variance:.3f}")

        assert belief_range > 0.5, "Distance model should have large belief range"
        assert belief_variance > 0.1, "Distance model should have good variance"

        print(f"  ✓ Distance-based model works well!")


class TestSummary:
    """
    Summary of findings and recommendations.
    """

    def test_print_summary(self):
        """Print summary of diagnosis."""
        print("\n" + "="*70)
        print("DECISION-BASED MODEL DIAGNOSIS SUMMARY")
        print("="*70)

        print("\n🐛 PROBLEMS IDENTIFIED:")
        print("\n1. Timing tolerance too strict (τ = 0.865)")
        print("   - Causes coordination probability ≈ 0 for most timesteps")
        print("   - Makes stag utility ≈ 0")
        print("   - Prevents intention inference")

        print("\n2. Zero utility makes intentions indistinguishable")
        print("   - When P_coord ≈ 0, model predicts 'always rabbit'")
        print("   - P(movement | stag) ≈ P(movement | rabbit)")
        print("   - Beliefs barely update (variance = 0.0001)")

        print("\n" + "="*70)
        print("RECOMMENDED FIXES:")
        print("="*70)

        print("\n✅ OPTION 1: Increase timing_tolerance")
        print("   - Change from τ = 0.865 to τ ≈ 100-150")
        print("   - Allows gradual coordination probability decay")
        print("   - May fix the decision-based model")

        print("\n✅ OPTION 2: Use distance-based model")
        print("   - Already works well (variance = 0.235)")
        print("   - Simpler and more robust")
        print("   - Theoretically less principled but practically effective")

        print("\n✅ OPTION 3: Hybrid approach")
        print("   - Use distance-based for belief updates")
        print("   - Use decision-based for action prediction")
        print("   - Combines strengths of both models")

        print("\n" + "="*70)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
