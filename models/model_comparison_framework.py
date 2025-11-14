#!/usr/bin/env python3
"""
Framework for comparing different computational models against human behavior.

Models to compare:
1. Distance-based (baseline): Just move toward closer target
2. Individual planning with P_coord: Current model
3. Joint planning (IW): Tao Gao's "imagined we" approach

Evaluation metrics:
- Log-likelihood of human trajectories
- Action prediction accuracy
- Belief trajectory correlation
"""

import numpy as np
import pandas as pd
from scipy.stats import vonmises
from typing import Tuple, Dict, List
import glob
from abc import ABC, abstractmethod


class BeliefModel(ABC):
    """Abstract base class for belief/action models."""

    @abstractmethod
    def predict_action_distribution(self, state: Dict, belief: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return action angles and probabilities.

        Returns:
            angles: Array of possible action angles
            probs: Probability for each angle
        """
        pass

    @abstractmethod
    def update_belief(self, belief: float, observed_action: float, state: Dict) -> float:
        """Update belief given observed action."""
        pass

    def evaluate_trajectory(self, trial_data: pd.DataFrame, player: str) -> float:
        """
        Compute log-likelihood of observed human trajectory.

        Returns:
            total_log_likelihood: Sum of log P(action_t | state_t, belief_t)
        """
        belief_col = f'p{player[-1]}_belief_p{"1" if player[-1]=="2" else "2"}_stag'

        total_ll = 0.0
        n_timesteps = len(trial_data)

        for t in range(1, n_timesteps):
            # Get state
            state = self._extract_state(trial_data.iloc[t], player)
            belief = trial_data.iloc[t-1][belief_col] if t > 0 else 0.5

            # Observed action
            dx = trial_data.iloc[t][f'{player}_x'] - trial_data.iloc[t-1][f'{player}_x']
            dy = trial_data.iloc[t][f'{player}_y'] - trial_data.iloc[t-1][f'{player}_y']

            if abs(dx) < 1 and abs(dy) < 1:
                continue

            observed_angle = np.arctan2(dy, dx)

            # Get model predictions
            angles, probs = self.predict_action_distribution(state, belief)

            # Compute likelihood (von Mises mixture)
            likelihood = 0.0
            kappa = 1.0  # concentration parameter
            for angle, prob in zip(angles, probs):
                likelihood += prob * vonmises.pdf(observed_angle, kappa, loc=angle)

            if likelihood > 0:
                total_ll += np.log(likelihood)

        return total_ll

    def _extract_state(self, row: pd.Series, player: str) -> Dict:
        """Extract state information for planning."""
        partner = 'player1' if player == 'player2' else 'player2'

        return {
            'player_x': row[f'{player}_x'],
            'player_y': row[f'{player}_y'],
            'partner_x': row[f'{partner}_x'],
            'partner_y': row[f'{partner}_y'],
            'stag_x': row['stag_x'],
            'stag_y': row['stag_y'],
            'stag_value': row['value'],
            'rabbit_x': row['rabbit_x'],
            'rabbit_y': row['rabbit_y'],
            'rabbit_value': 1.0
        }


class DistanceBasedModel(BeliefModel):
    """Baseline: just move toward closer target."""

    def __init__(self, n_directions: int = 8):
        self.action_angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
        self.temperature = 3.0

    def predict_action_distribution(self, state: Dict, belief: float) -> Tuple[np.ndarray, np.ndarray]:
        """Move toward closer target."""
        # Distance to targets
        dist_stag = np.sqrt((state['stag_x'] - state['player_x'])**2 +
                           (state['stag_y'] - state['player_y'])**2)
        dist_rabbit = np.sqrt((state['rabbit_x'] - state['player_x'])**2 +
                             (state['rabbit_y'] - state['player_y'])**2)

        # Pick closer target
        if dist_stag < dist_rabbit:
            target_x, target_y = state['stag_x'], state['stag_y']
        else:
            target_x, target_y = state['rabbit_x'], state['rabbit_y']

        # Angle to target
        angle_to_target = np.arctan2(target_y - state['player_y'],
                                     target_x - state['player_x'])

        # Softmax over actions based on alignment
        utilities = np.zeros(len(self.action_angles))
        for i, angle in enumerate(self.action_angles):
            alignment = np.cos(angle - angle_to_target)
            utilities[i] = alignment

        # Softmax
        exp_utils = np.exp(self.temperature * utilities)
        probs = exp_utils / exp_utils.sum()

        return self.action_angles, probs

    def update_belief(self, belief: float, observed_action: float, state: Dict) -> float:
        """Distance-based model doesn't update beliefs."""
        return belief


class IndividualPlanningModel(BeliefModel):
    """Current model: individual planning with P_coord."""

    def __init__(self, temperature: float = 5.0, timing_tolerance: float = 150.0,
                 n_directions: int = 8):
        self.temperature = temperature
        self.timing_tolerance = timing_tolerance
        self.action_angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
        self.speed = 1.0

    def predict_action_distribution(self, state: Dict, belief: float) -> Tuple[np.ndarray, np.ndarray]:
        """Individual planning with coordination belief."""
        # Compute P_coord
        dist_player = np.sqrt((state['stag_x'] - state['player_x'])**2 +
                            (state['stag_y'] - state['player_y'])**2)
        dist_partner = np.sqrt((state['stag_x'] - state['partner_x'])**2 +
                              (state['stag_y'] - state['partner_y'])**2)

        time_diff = abs(dist_player - dist_partner) / self.speed
        timing_alignment = np.exp(-0.5 * (time_diff / self.timing_tolerance)**2)
        P_coord = belief * timing_alignment

        # Compute utilities for each action
        utilities = np.zeros(len(self.action_angles))

        for i, angle in enumerate(self.action_angles):
            # Gains toward each target
            angle_to_stag = np.arctan2(state['stag_y'] - state['player_y'],
                                      state['stag_x'] - state['player_x'])
            angle_to_rabbit = np.arctan2(state['rabbit_y'] - state['player_y'],
                                        state['rabbit_x'] - state['player_x'])

            gain_stag = np.cos(angle - angle_to_stag) / (1 + dist_player / 100)
            gain_rabbit = np.cos(angle - angle_to_rabbit)

            # Individual utility with coordination
            utilities[i] = state['stag_value'] * P_coord * gain_stag + state['rabbit_value'] * gain_rabbit

        # Softmax
        exp_utils = np.exp(self.temperature * utilities)
        probs = exp_utils / exp_utils.sum()

        return self.action_angles, probs

    def update_belief(self, belief: float, observed_action: float, state: Dict) -> float:
        """Update using pure intention model (symmetric zeroing)."""
        # This would call the full Bayesian update
        # For now, simplified placeholder
        return belief


class JointPlanningModel(BeliefModel):
    """Tao Gao's IW model: joint planning for 'we' agent."""

    def __init__(self, temperature: float = 5.0, n_directions: int = 8):
        self.temperature = temperature
        self.action_angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

    def predict_action_distribution(self, state: Dict, belief: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Joint planning: solve for optimal coordinated actions.

        For "we" intends stag:
          U_we(a_player, a_partner) = P(catch stag | both actions) - costs

        For "we" intends rabbit:
          Each agent independently pursues rabbit
        """
        # Compute joint utilities for stag intention
        stag_utilities = self._compute_joint_stag_utilities(state)

        # Compute individual utilities for rabbit intention
        rabbit_utilities = self._compute_rabbit_utilities(state)

        # Mix based on belief
        utilities = belief * stag_utilities + (1 - belief) * rabbit_utilities

        # Softmax
        exp_utils = np.exp(self.temperature * utilities)
        probs = exp_utils / exp_utils.sum()

        return self.action_angles, probs

    def _compute_joint_stag_utilities(self, state: Dict) -> np.ndarray:
        """
        Compute utility of each player action under joint stag planning.

        Key idea: For each player action, compute best partner action,
        then evaluate joint success.
        """
        utilities = np.zeros(len(self.action_angles))

        for i, player_angle in enumerate(self.action_angles):
            # For this player action, what's the best partner action?
            best_joint_utility = -np.inf

            for partner_angle in self.action_angles:
                # Evaluate joint action
                joint_utility = self._evaluate_joint_action_stag(
                    state, player_angle, partner_angle
                )
                best_joint_utility = max(best_joint_utility, joint_utility)

            utilities[i] = best_joint_utility

        return utilities

    def _evaluate_joint_action_stag(self, state: Dict,
                                   player_angle: float, partner_angle: float) -> float:
        """
        Evaluate how well joint action coordinates toward stag.

        Good coordination means:
        - Both move toward stag
        - They arrive from complementary angles (surround)
        - Timing is synchronized
        """
        # How aligned is each action with stag?
        angle_to_stag_player = np.arctan2(state['stag_y'] - state['player_y'],
                                         state['stag_x'] - state['player_x'])
        angle_to_stag_partner = np.arctan2(state['stag_y'] - state['partner_y'],
                                           state['stag_x'] - state['partner_x'])

        player_alignment = np.cos(player_angle - angle_to_stag_player)
        partner_alignment = np.cos(partner_angle - angle_to_stag_partner)

        # Joint progress toward stag
        joint_progress = (player_alignment + partner_alignment) / 2

        # Bonus for approaching from different angles (encirclement)
        angle_diff = abs(player_angle - partner_angle)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Shortest angular distance
        encirclement_bonus = np.abs(np.pi - angle_diff) / np.pi  # Max when 180° apart

        return state['stag_value'] * (joint_progress + 0.5 * encirclement_bonus)

    def _compute_rabbit_utilities(self, state: Dict) -> np.ndarray:
        """Individual rabbit pursuit (no coordination needed)."""
        utilities = np.zeros(len(self.action_angles))

        angle_to_rabbit = np.arctan2(state['rabbit_y'] - state['player_y'],
                                     state['rabbit_x'] - state['player_x'])

        for i, angle in enumerate(self.action_angles):
            alignment = np.cos(angle - angle_to_rabbit)
            utilities[i] = state['rabbit_value'] * alignment

        return utilities

    def update_belief(self, belief: float, observed_action: float, state: Dict) -> float:
        """Update using joint planning likelihoods."""
        # Would implement full Bayesian update with joint planning
        return belief


def compare_models_on_human_data():
    """Compare all models on human trajectory data."""

    print("="*70)
    print("MODEL COMPARISON ON HUMAN DATA")
    print("="*70)

    # Import belief model
    from belief_model_decision import BayesianIntentionModelWithDecision

    # Load human data
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*_2024_08_24_0848.csv'))

    # Initialize belief model to generate beliefs
    print("\nRunning belief model to generate belief trajectories...")
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

    # Initialize models
    models = {
        'Distance-based': DistanceBasedModel(),
        'Individual Planning': IndividualPlanningModel(temperature=5.0),
        'Joint Planning (IW)': JointPlanningModel(temperature=5.0)
    }

    # Evaluate each model
    results = {name: [] for name in models.keys()}

    for trial_file in trial_files:
        trial_data = pd.read_csv(trial_file)
        if 'plater1_y' in trial_data.columns:
            trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})

        # Run belief model to get beliefs
        try:
            trial_data = belief_model.run_trial(trial_data)
        except Exception as e:
            print(f"ERROR running belief model on {trial_file}: {e}")
            continue

        trial_num = trial_file.split('trial')[1].split('_')[0]

        for model_name, model in models.items():
            try:
                # Evaluate both players
                ll_p1 = model.evaluate_trajectory(trial_data, 'player1')
                ll_p2 = model.evaluate_trajectory(trial_data, 'player2')
                total_ll = ll_p1 + ll_p2

                results[model_name].append(total_ll)
                print(f"Trial {trial_num}, {model_name}: LL = {total_ll:.2f}")
            except Exception as e:
                print(f"Trial {trial_num}, {model_name}: ERROR - {e}")
                results[model_name].append(np.nan)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for model_name in models.keys():
        valid_lls = [ll for ll in results[model_name] if not np.isnan(ll)]
        if valid_lls:
            mean_ll = np.mean(valid_lls)
            total_ll = np.sum(valid_lls)
            print(f"\n{model_name}:")
            print(f"  Mean LL per trial: {mean_ll:.2f}")
            print(f"  Total LL: {total_ll:.2f}")
            print(f"  N trials: {len(valid_lls)}")

    return results


if __name__ == '__main__':
    results = compare_models_on_human_data()
