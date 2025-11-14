#!/usr/bin/env python3
"""
Distance-based model with random tie-breaking.

Instead of always defaulting to rabbit when equidistant,
flip a coin (or use soft choice) when distances are close.

This is more principled than arbitrary tie-breaking.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy.stats import vonmises
from model_comparison_framework import BeliefModel
import glob
from belief_model_decision import BayesianIntentionModelWithDecision


class DistanceWithRandomTiebreak(BeliefModel):
    """
    Distance-based with probabilistic tie-breaking.

    When targets are equidistant (or close), choose randomly
    instead of always defaulting to rabbit.
    """

    def __init__(self,
                 n_directions: int = 8,
                 temperature: float = 12.0,
                 tiebreak_threshold: float = 50.0,  # Distance difference threshold
                 tiebreak_temp: float = 0.5):        # Softness of tie-breaking
        self.action_angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
        self.temperature = temperature
        self.tiebreak_threshold = tiebreak_threshold
        self.tiebreak_temp = tiebreak_temp

    def predict_action_distribution(self, state: Dict, belief: float) -> Tuple[np.ndarray, np.ndarray]:
        """Move toward closer target, with random tie-breaking."""

        # Distances to targets
        dist_stag = np.sqrt((state['stag_x'] - state['player_x'])**2 +
                           (state['stag_y'] - state['player_y'])**2)
        dist_rabbit = np.sqrt((state['rabbit_x'] - state['player_x'])**2 +
                             (state['rabbit_y'] - state['player_y'])**2)

        dist_diff = abs(dist_stag - dist_rabbit)

        # If distances are close (within threshold), probabilistic choice
        if dist_diff < self.tiebreak_threshold:
            # Softmax over targets based on small distance difference
            # Negative distance = better (closer)
            stag_util = -dist_stag
            rabbit_util = -dist_rabbit

            exp_utils = np.exp(self.tiebreak_temp * np.array([stag_util, rabbit_util]))
            target_probs = exp_utils / exp_utils.sum()

            P_stag = target_probs[0]
            P_rabbit = target_probs[1]

        else:
            # Clear winner based on distance
            if dist_stag < dist_rabbit:
                P_stag = 1.0
                P_rabbit = 0.0
            else:
                P_stag = 0.0
                P_rabbit = 1.0

        # Compute action distributions for each target
        actions_stag = self._move_toward(
            state['player_x'], state['player_y'],
            state['stag_x'], state['stag_y']
        )

        actions_rabbit = self._move_toward(
            state['player_x'], state['player_y'],
            state['rabbit_x'], state['rabbit_y']
        )

        # Mixture
        action_probs = P_stag * actions_stag + P_rabbit * actions_rabbit

        return self.action_angles, action_probs

    def _move_toward(self, player_x: float, player_y: float,
                    target_x: float, target_y: float) -> np.ndarray:
        """Plan movement toward target."""
        angle_to_target = np.arctan2(target_y - player_y, target_x - player_x)

        utilities = np.zeros(len(self.action_angles))
        for i, angle in enumerate(self.action_angles):
            alignment = np.cos(angle - angle_to_target)
            utilities[i] = alignment

        exp_utils = np.exp(self.temperature * utilities)
        probs = exp_utils / exp_utils.sum()

        return probs

    def update_belief(self, belief: float, observed_action: float, state: Dict) -> float:
        """No belief updates."""
        return belief


def test_random_tiebreak():
    """Test distance model with random tie-breaking."""

    print("="*70)
    print("DISTANCE-BASED WITH RANDOM TIE-BREAKING")
    print("="*70)
    print()

    # Load trials with beliefs
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

    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*_2024_08_24_0848.csv'))

    # Test different threshold and temperature settings
    configs = [
        {'threshold': 10.0, 'tiebreak_temp': 0.1, 'label': 'Tight threshold (10), very soft'},
        {'threshold': 50.0, 'tiebreak_temp': 0.5, 'label': 'Medium threshold (50), soft'},
        {'threshold': 100.0, 'tiebreak_temp': 1.0, 'label': 'Wide threshold (100), medium'},
        {'threshold': 100.0, 'tiebreak_temp': 0.0, 'label': 'Wide threshold (100), coin flip'},
    ]

    print("Testing different configurations:")
    print()

    for config in configs:
        model = DistanceWithRandomTiebreak(
            temperature=12.855,  # Use optimal execution temperature
            tiebreak_threshold=config['threshold'],
            tiebreak_temp=config['tiebreak_temp']
        )

        total_ll = 0.0
        n_trials = 0

        for trial_file in trial_files:
            trial_data = pd.read_csv(trial_file)
            if 'plater1_y' in trial_data.columns:
                trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})

            try:
                trial_data = belief_model.run_trial(trial_data)
                ll = model.evaluate_trajectory(trial_data, 'player1')
                ll += model.evaluate_trajectory(trial_data, 'player2')
                total_ll += ll
                n_trials += 1
            except:
                continue

        mean_ll = total_ll / n_trials if n_trials > 0 else 0

        print(f"{config['label']:45s}: LL = {total_ll:8.2f} (mean: {mean_ll:7.2f})")

    print()
    print("="*70)
    print("COMPARISON")
    print("="*70)
    print()
    print("Distance-based (rabbit default): LL = -5430.04")
    print("Hierarchical (med params):        LL = -5702.08")


if __name__ == '__main__':
    test_random_tiebreak()
