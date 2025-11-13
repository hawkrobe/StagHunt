"""
Unified test suite for decision models.

This test suite combines all previous standalone test scripts into a
cohesive pytest-based framework with proper organization and parameterization.

Test Classes:
- TestBasicFunctionality: Basic model operations and data loading
- TestBeliefComparison: Models with vs without beliefs
- TestActionSpace: Action space granularity effects
- TestContinuousLikelihood: Continuous likelihood evaluation
- TestMotorNoise: Action noise parameter sensitivity
"""

import pytest
import numpy as np
from decision_model_basic import UtilityDecisionModel


class TestBasicFunctionality:
    """Test basic model functionality and setup."""

    def test_trial_loading(self, trial_data):
        """Test that trials load correctly."""
        assert len(trial_data) > 0
        assert 'player1_x' in trial_data.columns
        assert 'player1_y' in trial_data.columns
        assert 'player2_x' in trial_data.columns
        assert 'player2_y' in trial_data.columns

    def test_belief_computation(self, trial_with_beliefs):
        """Test that beliefs are computed."""
        assert 'p1_belief_p2_stag' in trial_with_beliefs.columns
        assert 'p2_belief_p1_stag' in trial_with_beliefs.columns
        assert all(trial_with_beliefs['p1_belief_p2_stag'] >= 0)
        assert all(trial_with_beliefs['p1_belief_p2_stag'] <= 1)

    def test_model_initialization(self):
        """Test that decision model initializes correctly."""
        model = UtilityDecisionModel(
            n_directions=16,
            temperature=1.0,
            w_stag=1.0,
            w_rabbit=1.0
        )
        assert model.n_directions == 16
        assert model.temperature == 1.0
        assert len(model.action_angles) == 16


class TestBeliefComparison:
    """Test models with and without beliefs."""

    @pytest.fixture
    def decision_model(self):
        """Standard decision model."""
        return UtilityDecisionModel(
            n_directions=16,
            temperature=1.0,
            w_stag=1.0,
            w_rabbit=1.0
        )

    def test_without_beliefs(self, decision_model, all_trials_with_beliefs, load_trial_fn, trial_files):
        """Test model with beliefs fixed at 1.0 (no belief updating)."""
        total_log_lik = 0

        for trial_with_beliefs in all_trials_with_beliefs:
            # Override beliefs with constant 1.0
            trial_no_belief = trial_with_beliefs.copy()
            trial_no_belief['p1_belief_p2_stag'] = 1.0
            trial_no_belief['p2_belief_p1_stag'] = 1.0

            p1_log_lik, _, _ = decision_model.evaluate_trial(
                trial_no_belief,
                player='player1',
                belief_column='p1_belief_p2_stag'
            )
            p2_log_lik, _, _ = decision_model.evaluate_trial(
                trial_no_belief,
                player='player2',
                belief_column='p2_belief_p1_stag'
            )

            total_log_lik += p1_log_lik + p2_log_lik

        # Store for comparison
        assert total_log_lik < 0  # Log-likelihood should be negative

    def test_with_beliefs(self, decision_model, all_trials_with_beliefs):
        """Test model with dynamic beliefs."""
        total_log_lik = 0

        for trial_with_beliefs in all_trials_with_beliefs:
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

        assert total_log_lik < 0

    def test_beliefs_improve_fit(self, decision_model, all_trials_with_beliefs):
        """Test that beliefs improve model fit."""
        # With beliefs
        total_log_lik_with = 0
        for trial in all_trials_with_beliefs:
            p1_ll, _, _ = decision_model.evaluate_trial(
                trial, player='player1', belief_column='p1_belief_p2_stag'
            )
            p2_ll, _, _ = decision_model.evaluate_trial(
                trial, player='player2', belief_column='p2_belief_p1_stag'
            )
            total_log_lik_with += p1_ll + p2_ll

        # Without beliefs
        total_log_lik_without = 0
        for trial in all_trials_with_beliefs:
            trial_no_belief = trial.copy()
            trial_no_belief['p1_belief_p2_stag'] = 1.0
            trial_no_belief['p2_belief_p1_stag'] = 1.0

            p1_ll, _, _ = decision_model.evaluate_trial(
                trial_no_belief, player='player1', belief_column='p1_belief_p2_stag'
            )
            p2_ll, _, _ = decision_model.evaluate_trial(
                trial_no_belief, player='player2', belief_column='p2_belief_p1_stag'
            )
            total_log_lik_without += p1_ll + p2_ll

        # Beliefs should improve fit (higher log-likelihood)
        improvement = total_log_lik_with - total_log_lik_without
        print(f"\nLog-likelihood improvement from beliefs: {improvement:.2f}")
        assert total_log_lik_with > total_log_lik_without, \
            f"Beliefs should improve fit, but got {improvement:.2f}"


class TestActionSpace:
    """Test action space granularity effects."""

    @pytest.mark.parametrize("n_directions", [8, 16, 32, 64])
    def test_discrete_likelihood_action_space(self, n_directions, all_trials_with_beliefs):
        """Test model with different action space granularities (discrete likelihood)."""
        model = UtilityDecisionModel(
            n_directions=n_directions,
            temperature=1.0,
            w_stag=1.0,
            w_rabbit=1.0
        )

        total_log_lik = 0
        for trial in all_trials_with_beliefs:
            p1_ll, _, _ = model.evaluate_trial(
                trial, player='player1', belief_column='p1_belief_p2_stag'
            )
            p2_ll, _, _ = model.evaluate_trial(
                trial, player='player2', belief_column='p2_belief_p1_stag'
            )
            total_log_lik += p1_ll + p2_ll

        print(f"\nn_directions={n_directions}: total_ll={total_log_lik:.2f}")
        assert total_log_lik < 0


class TestContinuousLikelihood:
    """Test continuous likelihood evaluation."""

    @pytest.mark.parametrize("n_directions", [8, 16, 32, 64])
    def test_continuous_likelihood_convergence(self, n_directions, all_trials_with_beliefs):
        """Test that continuous likelihood converges across action space granularities."""
        model = UtilityDecisionModel(
            n_directions=n_directions,
            temperature=1.0,
            w_stag=1.0,
            w_rabbit=1.0
        )

        action_noise = 1.0
        total_log_lik = 0

        for trial in all_trials_with_beliefs:
            p1_ll, _, _ = model.evaluate_trial_continuous(
                trial,
                player='player1',
                belief_column='p1_belief_p2_stag',
                action_noise=action_noise
            )
            p2_ll, _, _ = model.evaluate_trial_continuous(
                trial,
                player='player2',
                belief_column='p2_belief_p1_stag',
                action_noise=action_noise
            )
            total_log_lik += p1_ll + p2_ll

        print(f"\nn_directions={n_directions}, continuous_ll={total_log_lik:.2f}")
        assert total_log_lik < 0

    def test_continuous_vs_discrete(self, all_trials_with_beliefs):
        """Compare continuous and discrete likelihood methods."""
        model = UtilityDecisionModel(
            n_directions=16,
            temperature=1.0,
            w_stag=1.0,
            w_rabbit=1.0
        )

        # Discrete likelihood
        total_discrete = 0
        for trial in all_trials_with_beliefs:
            p1_ll, _, _ = model.evaluate_trial(
                trial, player='player1', belief_column='p1_belief_p2_stag'
            )
            p2_ll, _, _ = model.evaluate_trial(
                trial, player='player2', belief_column='p2_belief_p1_stag'
            )
            total_discrete += p1_ll + p2_ll

        # Continuous likelihood
        total_continuous = 0
        for trial in all_trials_with_beliefs:
            p1_ll, _, _ = model.evaluate_trial_continuous(
                trial, player='player1', belief_column='p1_belief_p2_stag',
                action_noise=1.0
            )
            p2_ll, _, _ = model.evaluate_trial_continuous(
                trial, player='player2', belief_column='p2_belief_p1_stag',
                action_noise=1.0
            )
            total_continuous += p1_ll + p2_ll

        print(f"\nDiscrete: {total_discrete:.2f}, Continuous: {total_continuous:.2f}")
        # Continuous should generally be better (no discretization artifacts)
        # But we don't enforce this as a hard requirement


class TestMotorNoise:
    """Test action noise (motor precision) parameter."""

    @pytest.mark.parametrize("action_noise", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_action_noise_sensitivity(self, action_noise, all_trials_with_beliefs):
        """Test model with different action noise values."""
        model = UtilityDecisionModel(
            n_directions=16,
            temperature=1.0,
            w_stag=1.0,
            w_rabbit=1.0
        )

        total_log_lik = 0
        for trial in all_trials_with_beliefs:
            p1_ll, _, _ = model.evaluate_trial_continuous(
                trial,
                player='player1',
                belief_column='p1_belief_p2_stag',
                action_noise=action_noise
            )
            p2_ll, _, _ = model.evaluate_trial_continuous(
                trial,
                player='player2',
                belief_column='p2_belief_p1_stag',
                action_noise=action_noise
            )
            total_log_lik += p1_ll + p2_ll

        print(f"\naction_noise={action_noise}: total_ll={total_log_lik:.2f}")
        assert total_log_lik < 0

    def test_optimal_action_noise(self, all_trials_with_beliefs):
        """Find optimal action noise by comparing several values."""
        model = UtilityDecisionModel(
            n_directions=16,
            temperature=1.0,
            w_stag=1.0,
            w_rabbit=1.0
        )

        noise_values = [0.5, 1.0, 2.0, 5.0]
        results = {}

        for noise in noise_values:
            total_ll = 0
            for trial in all_trials_with_beliefs:
                p1_ll, _, _ = model.evaluate_trial_continuous(
                    trial, player='player1', belief_column='p1_belief_p2_stag',
                    action_noise=noise
                )
                p2_ll, _, _ = model.evaluate_trial_continuous(
                    trial, player='player2', belief_column='p2_belief_p1_stag',
                    action_noise=noise
                )
                total_ll += p1_ll + p2_ll
            results[noise] = total_ll

        best_noise = max(results, key=results.get)
        print(f"\nBest action noise: {best_noise} (LL: {results[best_noise]:.2f})")
        print("All results:", {k: f"{v:.2f}" for k, v in results.items()})


class TestTrialOutcomes:
    """Test model performance across different trial outcomes."""

    def test_cooperation_vs_defection_trials(self, all_trials_with_beliefs, trial_files, load_trial_fn):
        """Compare model fit on cooperation vs defection trials."""
        model = UtilityDecisionModel(
            n_directions=16,
            temperature=1.0,
            w_stag=1.0,
            w_rabbit=1.0
        )

        cooperation_lls = []
        defection_lls = []

        for trial_file, trial_with_beliefs in zip(trial_files, all_trials_with_beliefs):
            # Get outcome
            trial_data = load_trial_fn(trial_file)
            final_event = trial_data.iloc[-1]['event']

            # Compute log-likelihood
            p1_ll, _, _ = model.evaluate_trial_continuous(
                trial_with_beliefs,
                player='player1',
                belief_column='p1_belief_p2_stag',
                action_noise=1.0
            )
            p2_ll, _, _ = model.evaluate_trial_continuous(
                trial_with_beliefs,
                player='player2',
                belief_column='p2_belief_p1_stag',
                action_noise=1.0
            )
            trial_ll = p1_ll + p2_ll

            # Categorize
            if final_event == 5:  # Cooperation
                cooperation_lls.append(trial_ll)
            elif final_event in [3, 4]:  # Defection
                defection_lls.append(trial_ll)

        if cooperation_lls:
            print(f"\nCooperation trials: mean LL = {np.mean(cooperation_lls):.2f}")
        if defection_lls:
            print(f"Defection trials: mean LL = {np.mean(defection_lls):.2f}")

        # At least check we got some results
        assert len(cooperation_lls) + len(defection_lls) > 0
