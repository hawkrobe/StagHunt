#!/usr/bin/env python3
"""
Hierarchical Goal + Plan model (Baker, Saxe & Tenenbaum framework)

Two-level hierarchy:
1. GOAL SELECTION: Choose target (stag vs rabbit) based on:
   - Expected value
   - Coordination beliefs (P_coord)
   - Risk preferences

2. PLAN EXECUTION: Move toward chosen goal with near-optimal paths
   - High temperature (deterministic execution)
   - Direct movement toward goal

This separates strategic reasoning (what to pursue) from motor control
(how to get there).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy.stats import vonmises
from model_comparison_framework import BeliefModel


class HierarchicalGoalModel(BeliefModel):
    """
    Hierarchical model with goal selection and plan execution.

    Level 1: Goal selection
      - Compute expected utility of each target
      - Sample goal from softmax(utilities)
      - Low temperature (strategic uncertainty)

    Level 2: Plan execution
      - Plan direct path to chosen goal
      - Execute with high precision
      - High temperature (nearly deterministic)
    """

    def __init__(self,
                 goal_temperature: float = 2.0,      # Goal selection softness
                 execution_temperature: float = 10.0, # Movement precision
                 n_directions: int = 8):
        self.goal_temp = goal_temperature
        self.exec_temp = execution_temperature
        self.action_angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
        self.speed = 1.0
        self.timing_tolerance = 150.0

    def predict_action_distribution(self, state: Dict, belief: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hierarchical prediction:
        1. Select goal (stag or rabbit) based on expected utility
        2. Plan movement toward selected goal
        """

        # ===== LEVEL 1: GOAL SELECTION =====

        # Compute expected utility of each goal
        U_stag = self._compute_goal_utility_stag(state, belief)
        U_rabbit = self._compute_goal_utility_rabbit(state)

        # Softmax over goals (strategic choice)
        goal_utils = np.array([U_stag, U_rabbit])
        exp_goal_utils = np.exp(self.goal_temp * goal_utils)
        goal_probs = exp_goal_utils / exp_goal_utils.sum()

        P_choose_stag = goal_probs[0]
        P_choose_rabbit = goal_probs[1]

        # ===== LEVEL 2: PLAN EXECUTION =====

        # Compute action distribution for each goal
        actions_if_stag = self._plan_toward_target(
            state['player_x'], state['player_y'],
            state['stag_x'], state['stag_y']
        )

        actions_if_rabbit = self._plan_toward_target(
            state['player_x'], state['player_y'],
            state['rabbit_x'], state['rabbit_y']
        )

        # Mixture of action distributions weighted by goal probabilities
        action_probs = P_choose_stag * actions_if_stag + P_choose_rabbit * actions_if_rabbit

        return self.action_angles, action_probs

    def _compute_goal_utility_stag(self, state: Dict, belief: float) -> float:
        """
        Expected utility of pursuing stag.

        EU(stag) = V_stag × P(success) - Cost

        P(success) = P_coord × P(timing works out)
        """
        # Coordination probability
        dist_player = np.sqrt((state['stag_x'] - state['player_x'])**2 +
                             (state['stag_y'] - state['player_y'])**2)
        dist_partner = np.sqrt((state['stag_x'] - state['partner_x'])**2 +
                              (state['stag_y'] - state['partner_y'])**2)

        time_diff = abs(dist_player - dist_partner) / self.speed
        timing_alignment = np.exp(-0.5 * (time_diff / self.timing_tolerance)**2)

        P_coord = belief * timing_alignment

        # Expected value
        EU_stag = state['stag_value'] * P_coord

        return EU_stag

    def _compute_goal_utility_rabbit(self, state: Dict) -> float:
        """
        Expected utility of pursuing rabbit.

        EU(rabbit) = V_rabbit × 1.0 (guaranteed, no coordination needed)
        """
        return state['rabbit_value'] * 1.0  # Guaranteed payoff

    def _plan_toward_target(self, player_x: float, player_y: float,
                           target_x: float, target_y: float) -> np.ndarray:
        """
        Plan nearly-optimal movement toward target.

        Uses high temperature to create nearly deterministic path
        toward target (models precise motor control).
        """
        # Angle to target
        angle_to_target = np.arctan2(target_y - player_y, target_x - player_x)

        # Utility of each action = alignment with target direction
        utilities = np.zeros(len(self.action_angles))
        for i, angle in enumerate(self.action_angles):
            alignment = np.cos(angle - angle_to_target)
            utilities[i] = alignment

        # Softmax with high temperature (nearly deterministic)
        exp_utils = np.exp(self.exec_temp * utilities)
        probs = exp_utils / exp_utils.sum()

        return probs

    def update_belief(self, belief: float, observed_action: float, state: Dict) -> float:
        """Belief update (placeholder for now)."""
        return belief


def test_hierarchical_model():
    """Test the hierarchical model on human data."""
    import glob
    from belief_model_decision import BayesianIntentionModelWithDecision

    print("="*70)
    print("TESTING HIERARCHICAL GOAL + PLAN MODEL")
    print("="*70)
    print()

    # Initialize belief model to generate beliefs
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

    # Load trials
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*_2024_08_24_0848.csv'))

    # Test different parameter combinations
    configs = [
        {'goal_temp': 1.0, 'exec_temp': 10.0, 'label': 'Low goal temp, High exec temp'},
        {'goal_temp': 2.0, 'exec_temp': 10.0, 'label': 'Med goal temp, High exec temp'},
        {'goal_temp': 2.0, 'exec_temp': 15.0, 'label': 'Med goal temp, Very high exec temp'},
        {'goal_temp': 3.0, 'exec_temp': 12.0, 'label': 'High goal temp, High exec temp'},
    ]

    print("Testing different parameter combinations:")
    print()

    for config in configs:
        model = HierarchicalGoalModel(
            goal_temperature=config['goal_temp'],
            execution_temperature=config['exec_temp']
        )

        total_ll = 0.0
        n_trials = 0

        for trial_file in trial_files:
            trial_data = pd.read_csv(trial_file)
            if 'plater1_y' in trial_data.columns:
                trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})

            try:
                # Add beliefs
                trial_data = belief_model.run_trial(trial_data)

                # Evaluate model
                ll = model.evaluate_trajectory(trial_data, 'player1')
                ll += model.evaluate_trajectory(trial_data, 'player2')
                total_ll += ll
                n_trials += 1
            except Exception as e:
                continue

        mean_ll = total_ll / n_trials if n_trials > 0 else 0

        print(f"{config['label']:50s}: LL = {total_ll:8.2f} (mean: {mean_ll:7.2f})")

    print()
    print("="*70)
    print("COMPARISON TO BASELINE MODELS")
    print("="*70)
    print()
    print("Distance-based:         LL = -5430.04 (mean: -452.50)")
    print("Individual Planning:    LL = -5988.32 (mean: -499.03)")
    print("Joint Planning (IW):    LL = -6023.36 (mean: -501.95)")


if __name__ == '__main__':
    test_hierarchical_model()
