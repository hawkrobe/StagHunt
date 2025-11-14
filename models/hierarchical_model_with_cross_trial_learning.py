#!/usr/bin/env python3
"""
Integrated Hierarchical Model with Cross-Trial Learning

Combines:
1. Cross-trial learning: Expectations evolve based on previous trial outcomes
2. Within-trial beliefs: Moment-to-moment Bayesian updates
3. Hierarchical decisions: Goal selection → Plan execution

This is the "core" model that includes all cognitive components.
"""

import numpy as np
import pandas as pd
import glob
from typing import Tuple, Dict, List
from hierarchical_goal_model import HierarchicalGoalModel
from belief_model_decision import BayesianIntentionModelWithDecision


class CrossTrialLearningModule:
    """
    Tracks expectations across trials with Bayesian learning.

    After each trial, update priors based on observed behavior:
    - Cooperation → increase expectation
    - Defection → decrease expectation
    """

    def __init__(self,
                 initial_prior: float = 0.5,
                 learning_rate: float = 0.3,
                 belief_bounds: Tuple[float, float] = (0.05, 0.95)):
        self.initial_prior = initial_prior
        self.learning_rate = learning_rate
        self.belief_bounds = belief_bounds

        # Track evolving priors
        self.p1_expectation_p2 = initial_prior  # P1's expectation about P2
        self.p2_expectation_p1 = initial_prior  # P2's expectation about P1

        # History tracking
        self.expectation_history_p1 = [initial_prior]
        self.expectation_history_p2 = [initial_prior]
        self.trial_outcomes = []

    def get_priors(self) -> Tuple[float, float]:
        """Get current cross-trial expectations to initialize beliefs."""
        return self.p1_expectation_p2, self.p2_expectation_p1

    def update_from_trial(self,
                         p1_final_belief: float,
                         p2_final_belief: float,
                         trial_outcome: str):
        """
        Update expectations based on trial outcome.

        Args:
            p1_final_belief: P1's final belief about P2 at trial end
            p2_final_belief: P2's final belief about P1 at trial end
            trial_outcome: 'cooperation', 'p1_rabbit', 'p2_rabbit', or 'unknown'
        """
        # Exponential moving average: new = (1-α) * old + α * evidence
        self.p1_expectation_p2 = (
            (1 - self.learning_rate) * self.p1_expectation_p2 +
            self.learning_rate * p1_final_belief
        )

        self.p2_expectation_p1 = (
            (1 - self.learning_rate) * self.p2_expectation_p1 +
            self.learning_rate * p2_final_belief
        )

        # Clip to bounds
        self.p1_expectation_p2 = np.clip(self.p1_expectation_p2, *self.belief_bounds)
        self.p2_expectation_p1 = np.clip(self.p2_expectation_p1, *self.belief_bounds)

        # Record history
        self.expectation_history_p1.append(self.p1_expectation_p2)
        self.expectation_history_p2.append(self.p2_expectation_p1)
        self.trial_outcomes.append(trial_outcome)

    def get_history(self) -> pd.DataFrame:
        """Get full history of expectation evolution."""
        return pd.DataFrame({
            'trial': range(len(self.trial_outcomes)),
            'p1_expectation': self.expectation_history_p1[1:],
            'p2_expectation': self.expectation_history_p2[1:],
            'outcome': self.trial_outcomes
        })


class IntegratedHierarchicalModel:
    """
    Full cognitive model combining:
    - Cross-trial learning (evolving priors)
    - Within-trial belief updates (Bayesian inference)
    - Hierarchical decision-making (goal + plan)

    This is the "core" model that generates all cognitive variables.
    """

    def __init__(self,
                 # Cross-trial parameters
                 initial_prior: float = 0.5,
                 learning_rate: float = 0.3,
                 # Belief update parameters
                 within_trial_belief_bounds: Tuple[float, float] = (0.01, 0.99),
                 # Decision parameters
                 goal_temperature: float = 2.0,
                 execution_temperature: float = 12.0,
                 timing_tolerance: float = 150.0,
                 action_noise: float = 5.0,
                 n_directions: int = 8):

        # Cross-trial learning module
        self.cross_trial = CrossTrialLearningModule(
            initial_prior=initial_prior,
            learning_rate=learning_rate
        )

        # Within-trial belief model
        self.belief_model = BayesianIntentionModelWithDecision(
            decision_model_params={
                'temperature': goal_temperature,
                'timing_tolerance': timing_tolerance,
                'action_noise': action_noise,
                'n_directions': n_directions
            },
            prior_stag=initial_prior,  # Will be overridden per trial
            belief_bounds=within_trial_belief_bounds
        )

        # Hierarchical decision model
        self.decision_model = HierarchicalGoalModel(
            goal_temperature=goal_temperature,
            execution_temperature=execution_temperature,
            n_directions=n_directions
        )
        self.decision_model.timing_tolerance = timing_tolerance

    def process_trial(self, trial_data: pd.DataFrame, trial_num: int) -> pd.DataFrame:
        """
        Process single trial with full cognitive model.

        Returns enriched dataframe with:
        - Cross-trial expectations (priors)
        - Within-trial beliefs (moment-to-moment)
        - Decision variables (utilities, probabilities, coordination)
        """
        # Get cross-trial priors for this trial
        p1_prior, p2_prior = self.cross_trial.get_priors()

        # Override belief model priors for this trial
        self.belief_model.prior_belief_p1_stag = p1_prior
        self.belief_model.prior_belief_p2_stag = p2_prior

        # Run within-trial belief updates
        trial_data = self.belief_model.run_trial(trial_data)

        # Add cross-trial expectations as constant columns
        trial_data['p1_cross_trial_expectation'] = p1_prior
        trial_data['p2_cross_trial_expectation'] = p2_prior
        trial_data['trial_num'] = trial_num

        # Compute decision variables (coordination estimates, etc.)
        trial_data = self._compute_coordination_estimates(trial_data)

        # Detect trial outcome
        outcome = self._detect_outcome(trial_data)

        # Get final beliefs for learning update
        p1_final = trial_data['p1_belief_p2_stag'].iloc[-1]
        p2_final = trial_data['p2_belief_p1_stag'].iloc[-1]

        # Update cross-trial expectations for next trial
        self.cross_trial.update_from_trial(p1_final, p2_final, outcome)

        return trial_data

    def _compute_coordination_estimates(self, trial_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute coordination-related variables (key markers for cooperation).

        Focus on:
        - P_coord: Belief × timing alignment
        - Timing alignment: Can they arrive together?
        - Expected utilities: EU_stag vs EU_rabbit
        - Choice probabilities: P(choose stag)
        """
        p1_coords = []
        p2_coords = []

        for idx, row in trial_data.iterrows():
            # === P1's coordination estimate ===
            p1_state = {
                'player_x': row['player1_x'],
                'player_y': row['player1_y'],
                'partner_x': row['player2_x'],
                'partner_y': row['player2_y'],
                'stag_x': row['stag_x'],
                'stag_y': row['stag_y'],
                'stag_value': row['value'],
                'rabbit_x': row['rabbit_x'],
                'rabbit_y': row['rabbit_y'],
                'rabbit_value': 1.0
            }

            p1_belief = row['p1_belief_p2_stag']

            # Distances
            dist_p1_stag = np.sqrt((p1_state['stag_x'] - p1_state['player_x'])**2 +
                                  (p1_state['stag_y'] - p1_state['player_y'])**2)
            dist_p2_stag = np.sqrt((p1_state['stag_x'] - p1_state['partner_x'])**2 +
                                  (p1_state['stag_y'] - p1_state['partner_y'])**2)

            # Timing alignment
            time_diff = abs(dist_p1_stag - dist_p2_stag) / self.decision_model.speed
            timing_align_p1 = np.exp(-0.5 * (time_diff / self.decision_model.timing_tolerance)**2)

            # Coordination probability (KEY MARKER)
            P_coord_p1 = p1_belief * timing_align_p1

            # Expected utilities
            EU_stag_p1 = self.decision_model._compute_goal_utility_stag(p1_state, p1_belief)
            EU_rabbit_p1 = self.decision_model._compute_goal_utility_rabbit(p1_state)

            # Choice probabilities
            goal_utils = np.array([EU_stag_p1, EU_rabbit_p1])
            exp_utils = np.exp(self.decision_model.goal_temp * goal_utils)
            goal_probs = exp_utils / exp_utils.sum()
            P_choose_stag_p1 = goal_probs[0]

            p1_coords.append({
                'p1_P_coord': P_coord_p1,
                'p1_timing_alignment': timing_align_p1,
                'p1_EU_stag': EU_stag_p1,
                'p1_EU_rabbit': EU_rabbit_p1,
                'p1_P_choose_stag': P_choose_stag_p1,
                'p1_dist_to_stag': dist_p1_stag,
            })

            # === P2's coordination estimate ===
            p2_state = {
                'player_x': row['player2_x'],
                'player_y': row['player2_y'],
                'partner_x': row['player1_x'],
                'partner_y': row['player1_y'],
                'stag_x': row['stag_x'],
                'stag_y': row['stag_y'],
                'stag_value': row['value'],
                'rabbit_x': row['rabbit_x'],
                'rabbit_y': row['rabbit_y'],
                'rabbit_value': 1.0
            }

            p2_belief = row['p2_belief_p1_stag']

            dist_p2_stag_2 = np.sqrt((p2_state['stag_x'] - p2_state['player_x'])**2 +
                                    (p2_state['stag_y'] - p2_state['player_y'])**2)
            dist_p1_stag_2 = np.sqrt((p2_state['stag_x'] - p2_state['partner_x'])**2 +
                                    (p2_state['stag_y'] - p2_state['partner_y'])**2)

            time_diff = abs(dist_p2_stag_2 - dist_p1_stag_2) / self.decision_model.speed
            timing_align_p2 = np.exp(-0.5 * (time_diff / self.decision_model.timing_tolerance)**2)

            P_coord_p2 = p2_belief * timing_align_p2

            EU_stag_p2 = self.decision_model._compute_goal_utility_stag(p2_state, p2_belief)
            EU_rabbit_p2 = self.decision_model._compute_goal_utility_rabbit(p2_state)

            goal_utils = np.array([EU_stag_p2, EU_rabbit_p2])
            exp_utils = np.exp(self.decision_model.goal_temp * goal_utils)
            goal_probs = exp_utils / exp_utils.sum()
            P_choose_stag_p2 = goal_probs[0]

            p2_coords.append({
                'p2_P_coord': P_coord_p2,
                'p2_timing_alignment': timing_align_p2,
                'p2_EU_stag': EU_stag_p2,
                'p2_EU_rabbit': EU_rabbit_p2,
                'p2_P_choose_stag': P_choose_stag_p2,
                'p2_dist_to_stag': dist_p2_stag_2,
            })

        # Add to dataframe
        p1_df = pd.DataFrame(p1_coords)
        p2_df = pd.DataFrame(p2_coords)

        for col in p1_df.columns:
            trial_data[col] = p1_df[col].values
        for col in p2_df.columns:
            trial_data[col] = p2_df[col].values

        return trial_data

    def _detect_outcome(self, trial_data: pd.DataFrame) -> str:
        """Detect trial outcome from event codes."""
        if trial_data['event'].max() == 5:
            return 'cooperation'
        elif 3 in trial_data['event'].values:
            return 'p1_rabbit'
        elif 4 in trial_data['event'].values:
            return 'p2_rabbit'
        else:
            return 'unknown'

    def process_all_trials(self, trial_files: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process all trials in chronological order.

        Returns:
            all_data: Combined dataframe with all timesteps and regressors
            expectation_history: Cross-trial expectation evolution
        """
        enriched_trials = []

        print("="*70)
        print("INTEGRATED MODEL: PROCESSING TRIALS WITH CROSS-TRIAL LEARNING")
        print("="*70)
        print()
        print(f"Parameters:")
        print(f"  Learning rate: {self.cross_trial.learning_rate}")
        print(f"  Goal temperature: {self.decision_model.goal_temp}")
        print(f"  Execution temperature: {self.decision_model.exec_temp}")
        print(f"  Timing tolerance: {self.decision_model.timing_tolerance}")
        print()

        for trial_file in trial_files:
            trial_data = pd.read_csv(trial_file)
            if 'plater1_y' in trial_data.columns:
                trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})

            trial_num = int(trial_file.split('trial')[1].split('_')[0])

            # Get priors before processing
            p1_prior, p2_prior = self.cross_trial.get_priors()

            # Process trial
            trial_data = self.process_trial(trial_data, trial_num)

            # Get outcome and final beliefs
            outcome = self._detect_outcome(trial_data)
            p1_final = trial_data['p1_belief_p2_stag'].iloc[-1]
            p2_final = trial_data['p2_belief_p1_stag'].iloc[-1]

            print(f"Trial {trial_num:2d}: {outcome:15s} | "
                  f"Priors: P1={p1_prior:.3f}, P2={p2_prior:.3f} | "
                  f"Final: P1={p1_final:.3f}, P2={p2_final:.3f}")

            enriched_trials.append(trial_data)

        # Combine all trials
        all_data = pd.concat(enriched_trials, ignore_index=True)

        # Get expectation history
        expectation_history = self.cross_trial.get_history()

        print()
        print("="*70)
        print("CROSS-TRIAL LEARNING SUMMARY")
        print("="*70)
        print()
        print("Expectation evolution:")
        print(expectation_history.to_string(index=False))
        print()

        # Compute cooperation-related statistics
        p_coord_mean = all_data['p1_P_coord'].mean()
        p_coord_std = all_data['p1_P_coord'].std()
        timing_mean = all_data['p1_timing_alignment'].mean()

        print(f"Coordination estimates:")
        print(f"  Mean P_coord: {p_coord_mean:.3f} (SD: {p_coord_std:.3f})")
        print(f"  Mean timing alignment: {timing_mean:.3f}")
        print()

        return all_data, expectation_history


def test_integrated_model():
    """Test the integrated model on all trials."""

    # Load trials
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*_2024_08_24_0848.csv'))

    # Test different learning rates
    for lr in [0.2, 0.3, 0.5]:
        print(f"\n{'='*70}")
        print(f"TESTING LEARNING RATE = {lr}")
        print('='*70)

        model = IntegratedHierarchicalModel(
            initial_prior=0.5,
            learning_rate=lr,
            goal_temperature=2.408,      # From fitting
            execution_temperature=14.992, # From fitting
            timing_tolerance=150.0
        )

        all_data, expectation_history = model.process_all_trials(trial_files)

        # Save outputs
        output_file = f'integrated_model_lr{lr:.1f}.csv'
        all_data.to_csv(output_file, index=False)

        expectation_file = f'cross_trial_expectations_lr{lr:.1f}.csv'
        expectation_history.to_csv(expectation_file, index=False)

        print(f"✓ Saved enriched data to {output_file}")
        print(f"✓ Saved expectation history to {expectation_file}")


if __name__ == '__main__':
    test_integrated_model()
