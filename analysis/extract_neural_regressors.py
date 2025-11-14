#!/usr/bin/env python3
"""
Extract model-based regressors for neural analysis.

Key innovation: Model cross-trial learning
- Priors evolve based on previous trial outcomes
- Repeated defections → lower expectations for cooperation
- Creates richer set of belief-related regressors

Output: Enhanced dataframes with columns for:
- Cross-trial expectations (evolving priors)
- Within-trial beliefs (moment-to-moment updates)
- Expected values (EU_stag, EU_rabbit)
- Choice probabilities (P_choose_stag)
- Uncertainty measures (entropy, conflict)
- Coordination estimates (P_coord, timing_alignment)
"""

import pandas as pd
import numpy as np
import glob
from typing import List, Tuple
from models.belief_model_decision import BayesianIntentionModelWithDecision
from models.hierarchical_goal_model import HierarchicalGoalModel


class CrossTrialLearningModel:
    """
    Model that tracks expectations across trials.

    After each trial, update prior belief based on outcome:
    - Cooperation → increase prior
    - Defection → decrease prior
    - Learning rate controls how much each trial impacts next prior
    """

    def __init__(self,
                 initial_prior: float = 0.5,
                 learning_rate: float = 0.3,
                 belief_bounds: Tuple[float, float] = (0.05, 0.95)):
        self.initial_prior = initial_prior
        self.learning_rate = learning_rate
        self.belief_bounds = belief_bounds

        # Track evolving priors for each player about each player
        self.p1_expectation_p2 = initial_prior  # P1's expectation about P2
        self.p2_expectation_p1 = initial_prior  # P2's expectation about P1

    def update_expectations(self, trial_outcome: str,
                          p1_final_belief: float,
                          p2_final_belief: float):
        """
        Update cross-trial expectations based on trial outcome.

        Args:
            trial_outcome: 'cooperation', 'p1_rabbit', 'p2_rabbit'
            p1_final_belief: P1's final belief about P2 at trial end
            p2_final_belief: P2's final belief about P1 at trial end
        """
        # Learning rule: weighted average of current prior and observed evidence
        # Evidence = final belief from within-trial updates

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

    def get_expectations(self) -> Tuple[float, float]:
        """Get current cross-trial expectations."""
        return self.p1_expectation_p2, self.p2_expectation_p1


def compute_rich_regressors(trial_data: pd.DataFrame,
                            cross_trial_prior_p1: float,
                            cross_trial_prior_p2: float,
                            hierarchical_model: HierarchicalGoalModel) -> pd.DataFrame:
    """
    Compute rich set of regressors for neural analysis.

    Returns dataframe with additional columns:
    - Cross-trial expectations (priors from previous trials)
    - Expected values (EU_stag, EU_rabbit)
    - Choice probabilities
    - Uncertainty measures
    - Coordination estimates
    """

    # Add cross-trial priors as constant columns for this trial
    trial_data['p1_cross_trial_expectation'] = cross_trial_prior_p1
    trial_data['p2_cross_trial_expectation'] = cross_trial_prior_p2

    # Compute trial-by-trial regressors from hierarchical model
    p1_regressors = []
    p2_regressors = []

    for idx, row in trial_data.iterrows():
        # P1's regressors
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

        # Compute goal utilities
        EU_stag_p1 = hierarchical_model._compute_goal_utility_stag(p1_state, p1_belief)
        EU_rabbit_p1 = hierarchical_model._compute_goal_utility_rabbit(p1_state)

        # Compute choice probabilities (softmax over goals)
        goal_utils = np.array([EU_stag_p1, EU_rabbit_p1])
        exp_utils = np.exp(hierarchical_model.goal_temp * goal_utils)
        goal_probs = exp_utils / exp_utils.sum()

        P_choose_stag_p1 = goal_probs[0]
        P_choose_rabbit_p1 = goal_probs[1]

        # Compute coordination estimate
        dist_p1 = np.sqrt((p1_state['stag_x'] - p1_state['player_x'])**2 +
                         (p1_state['stag_y'] - p1_state['player_y'])**2)
        dist_p2 = np.sqrt((p1_state['stag_x'] - p1_state['partner_x'])**2 +
                         (p1_state['stag_y'] - p1_state['partner_y'])**2)

        time_diff = abs(dist_p1 - dist_p2) / hierarchical_model.speed
        timing_alignment_p1 = np.exp(-0.5 * (time_diff / hierarchical_model.timing_tolerance)**2)
        P_coord_p1 = p1_belief * timing_alignment_p1

        # Uncertainty measures
        choice_entropy_p1 = -np.sum(goal_probs * np.log(goal_probs + 1e-10))
        goal_conflict_p1 = 1 - abs(P_choose_stag_p1 - P_choose_rabbit_p1)  # Max when 50/50

        p1_regressors.append({
            'p1_EU_stag': EU_stag_p1,
            'p1_EU_rabbit': EU_rabbit_p1,
            'p1_P_choose_stag': P_choose_stag_p1,
            'p1_P_choose_rabbit': P_choose_rabbit_p1,
            'p1_choice_entropy': choice_entropy_p1,
            'p1_goal_conflict': goal_conflict_p1,
            'p1_P_coord': P_coord_p1,
            'p1_timing_alignment': timing_alignment_p1,
            'p1_dist_to_stag': dist_p1,
            'p1_dist_to_rabbit': np.sqrt((p1_state['rabbit_x'] - p1_state['player_x'])**2 +
                                         (p1_state['rabbit_y'] - p1_state['player_y'])**2),
        })

        # P2's regressors (same computations)
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

        EU_stag_p2 = hierarchical_model._compute_goal_utility_stag(p2_state, p2_belief)
        EU_rabbit_p2 = hierarchical_model._compute_goal_utility_rabbit(p2_state)

        goal_utils = np.array([EU_stag_p2, EU_rabbit_p2])
        exp_utils = np.exp(hierarchical_model.goal_temp * goal_utils)
        goal_probs = exp_utils / exp_utils.sum()

        P_choose_stag_p2 = goal_probs[0]
        P_choose_rabbit_p2 = goal_probs[1]

        dist_p2_stag = np.sqrt((p2_state['stag_x'] - p2_state['player_x'])**2 +
                               (p2_state['stag_y'] - p2_state['player_y'])**2)
        dist_p1_stag = np.sqrt((p2_state['stag_x'] - p2_state['partner_x'])**2 +
                               (p2_state['stag_y'] - p2_state['partner_y'])**2)

        time_diff = abs(dist_p2_stag - dist_p1_stag) / hierarchical_model.speed
        timing_alignment_p2 = np.exp(-0.5 * (time_diff / hierarchical_model.timing_tolerance)**2)
        P_coord_p2 = p2_belief * timing_alignment_p2

        choice_entropy_p2 = -np.sum(goal_probs * np.log(goal_probs + 1e-10))
        goal_conflict_p2 = 1 - abs(P_choose_stag_p2 - P_choose_rabbit_p2)

        p2_regressors.append({
            'p2_EU_stag': EU_stag_p2,
            'p2_EU_rabbit': EU_rabbit_p2,
            'p2_P_choose_stag': P_choose_stag_p2,
            'p2_P_choose_rabbit': P_choose_rabbit_p2,
            'p2_choice_entropy': choice_entropy_p2,
            'p2_goal_conflict': goal_conflict_p2,
            'p2_P_coord': P_coord_p2,
            'p2_timing_alignment': timing_alignment_p2,
            'p2_dist_to_stag': dist_p2_stag,
            'p2_dist_to_rabbit': np.sqrt((p2_state['rabbit_x'] - p2_state['player_x'])**2 +
                                         (p2_state['rabbit_y'] - p2_state['player_y'])**2),
        })

    # Add regressors to dataframe
    p1_df = pd.DataFrame(p1_regressors)
    p2_df = pd.DataFrame(p2_regressors)

    for col in p1_df.columns:
        trial_data[col] = p1_df[col].values
    for col in p2_df.columns:
        trial_data[col] = p2_df[col].values

    return trial_data


def generate_neural_regressors(learning_rate: float = 0.3):
    """
    Generate full set of neural regressors across all trials.

    Includes cross-trial learning dynamics.
    """

    print("="*70)
    print("GENERATING NEURAL REGRESSORS WITH CROSS-TRIAL LEARNING")
    print("="*70)
    print()
    print(f"Learning rate: {learning_rate}")
    print()

    # Initialize models
    belief_model = BayesianIntentionModelWithDecision(
        decision_model_params={
            'temperature': 5.0,
            'timing_tolerance': 150.0,
            'action_noise': 5.0,
            'n_directions': 8
        },
        prior_stag=0.5,
        belief_bounds=(0.01, 0.99)
    )

    hierarchical_model = HierarchicalGoalModel(
        goal_temperature=2.0,
        execution_temperature=12.0
    )

    cross_trial_model = CrossTrialLearningModel(
        initial_prior=0.5,
        learning_rate=learning_rate
    )

    # Load trials in chronological order
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*_2024_08_24_0848.csv'))

    enriched_trials = []

    print("Processing trials in chronological order:")
    print()

    for trial_file in trial_files:
        trial_data = pd.read_csv(trial_file)
        if 'plater1_y' in trial_data.columns:
            trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})

        trial_num = int(trial_file.split('trial')[1].split('_')[0])

        # Get cross-trial expectations for this trial (priors from previous trials)
        p1_prior, p2_prior = cross_trial_model.get_expectations()

        # Run belief model with current cross-trial priors
        # NOTE: For now we still use fixed prior=0.5 within trial
        # Could extend to use cross_trial priors as initial beliefs
        trial_data = belief_model.run_trial(trial_data)

        # Compute rich regressors
        trial_data = compute_rich_regressors(
            trial_data, p1_prior, p2_prior, hierarchical_model
        )

        # Get outcome
        outcome = "unknown"
        if trial_data['event'].max() == 5:
            outcome = "cooperation"
        elif 3 in trial_data['event'].values:
            outcome = "p1_rabbit"
        elif 4 in trial_data['event'].values:
            outcome = "p2_rabbit"

        # Get final beliefs
        p1_final = trial_data['p1_belief_p2_stag'].iloc[-1]
        p2_final = trial_data['p2_belief_p1_stag'].iloc[-1]

        print(f"Trial {trial_num:2d}: {outcome:15s} | "
              f"Priors: P1={p1_prior:.3f}, P2={p2_prior:.3f} | "
              f"Final: P1={p1_final:.3f}, P2={p2_final:.3f}")

        # Update cross-trial expectations for next trial
        cross_trial_model.update_expectations(outcome, p1_final, p2_final)

        # Add trial number
        trial_data['trial_num'] = trial_num

        enriched_trials.append(trial_data)

    print()
    print("="*70)
    print("REGRESSOR SUMMARY")
    print("="*70)
    print()

    # Concatenate all trials
    all_data = pd.concat(enriched_trials, ignore_index=True)

    # List all regressors
    regressor_cols = [col for col in all_data.columns if any(x in col for x in
                     ['belief', 'expectation', 'EU_', 'P_choose', 'entropy', 'conflict',
                      'P_coord', 'timing', 'dist_to'])]

    print("Generated regressors:")
    for col in sorted(regressor_cols):
        mean_val = all_data[col].mean()
        std_val = all_data[col].std()
        min_val = all_data[col].min()
        max_val = all_data[col].max()
        print(f"  {col:35s}: mean={mean_val:6.3f}, std={std_val:6.3f}, "
              f"range=[{min_val:6.3f}, {max_val:6.3f}]")

    print()
    print(f"Total timesteps: {len(all_data)}")
    print(f"Total regressors: {len(regressor_cols)}")

    # Save
    output_file = 'neural_regressors_with_cross_trial_learning.csv'
    all_data.to_csv(output_file, index=False)

    print()
    print(f"✓ Saved enriched data to {output_file}")

    # Also save individual trial files with regressors
    import os
    os.makedirs('enriched_trials', exist_ok=True)

    for trial_data in enriched_trials:
        trial_num = trial_data['trial_num'].iloc[0]
        trial_output = f'enriched_trials/trial_{trial_num:02d}_with_regressors.csv'
        trial_data.to_csv(trial_output, index=False)

    print(f"✓ Saved individual trial files to enriched_trials/")

    return all_data, regressor_cols


if __name__ == '__main__':
    # Test different learning rates
    for lr in [0.2, 0.3, 0.5]:
        print(f"\n{'='*70}")
        print(f"TESTING LEARNING RATE = {lr}")
        print('='*70)
        data, regressors = generate_neural_regressors(learning_rate=lr)
