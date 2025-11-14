"""
Decision model with explicit coordination probability.

This version endogenizes the difficulty of cooperation by modeling:
1. Timing synchronization between players
2. Explicit coordination probability
3. Actual payoff values (not free weights)
"""

import numpy as np
import pandas as pd
from scipy.stats import vonmises


class CoordinatedDecisionModel:
    """
    Utility-based decision model with explicit coordination modeling.

    Key differences from base model:
    - Stag value is multiplied by coordination probability
    - Coordination probability = belief × timing_alignment
    - Timing alignment based on estimated arrival times
    - Weights are actual reward values (not free parameters)
    """

    def __init__(self, n_directions=8, temperature=1.0,
                 timing_tolerance=1.0, speed=1.0):
        """
        Parameters:
        -----------
        n_directions : int
            Number of discrete action directions
        temperature : float
            Softmax inverse temperature (higher = more deterministic)
        timing_tolerance : float
            How tolerant to timing differences (seconds)
            Higher = more tolerant of asynchrony
        speed : float
            Movement speed (pixels per timestep)
        """
        self.n_directions = n_directions
        self.temperature = temperature
        self.timing_tolerance = timing_tolerance
        self.speed = speed

        # Discrete action angles (evenly spaced)
        self.action_angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

    def compute_distance_gain(self, player_x, player_y, target_x, target_y,
                              action_angle, speed):
        """
        Compute how much closer action brings player to target.

        Returns gain in range [-1, 1]:
        - Positive: moving toward target
        - Negative: moving away from target
        """
        # Current distance to target
        dx = target_x - player_x
        dy = target_y - player_y
        current_dist = np.sqrt(dx**2 + dy**2) + 1e-10

        # Direction to target
        angle_to_target = np.arctan2(dy, dx)

        # Cosine similarity (how aligned is action with target direction?)
        alignment = np.cos(action_angle - angle_to_target)

        # Normalize by current distance (closer = higher urgency)
        gain = alignment * (1.0 / (1.0 + current_dist / 100.0))

        return gain

    def compute_timing_alignment(self, player_x, player_y,
                                 partner_x, partner_y,
                                 stag_x, stag_y):
        """
        Compute alignment of arrival times at stag.

        Returns value in [0, 1]:
        - 1.0: Both arrive at same time
        - 0.0: Very different arrival times
        """
        # Distance to stag for both players
        player_dist = np.sqrt((stag_x - player_x)**2 + (stag_y - player_y)**2)
        partner_dist = np.sqrt((stag_x - partner_x)**2 + (stag_y - partner_y)**2)

        # Estimated time to reach stag (assuming constant speed)
        player_time = player_dist / (self.speed + 1e-10)
        partner_time = partner_dist / (self.speed + 1e-10)

        # Time difference
        time_diff = abs(player_time - partner_time)

        # Convert to alignment score using Gaussian
        # tolerance controls width of Gaussian
        alignment = np.exp(-0.5 * (time_diff / self.timing_tolerance)**2)

        return alignment

    def compute_coordination_probability(self, belief_partner_stag,
                                        timing_alignment):
        """
        Compute probability of successful coordination.

        Combines:
        - Belief that partner intends to go for stag
        - Timing alignment (can we arrive together?)

        Returns probability in [0, 1]
        """
        # Simple multiplicative model:
        # Need BOTH intention AND good timing
        p_coord = belief_partner_stag * timing_alignment

        return p_coord

    def compute_utility(self, player_x, player_y,
                       partner_x, partner_y,
                       stag_x, stag_y, stag_value,
                       rabbit_x, rabbit_y, rabbit_value,
                       belief_partner_stag,
                       action_angle):
        """
        Compute utility of taking action in given direction.

        Parameters:
        -----------
        player_x, player_y : float
            Current player position
        partner_x, partner_y : float
            Current partner position
        stag_x, stag_y, stag_value : float
            Stag position and current reward value
        rabbit_x, rabbit_y, rabbit_value : float
            Rabbit position and reward value (usually fixed at 1.0)
        belief_partner_stag : float
            Belief that partner is going for stag
        action_angle : float or array
            Action direction(s) to evaluate

        Returns:
        --------
        utility : float or array
            Expected utility of action
        """
        # Gains from moving toward each prey
        gain_stag = self.compute_distance_gain(
            player_x, player_y, stag_x, stag_y, action_angle, self.speed
        )
        gain_rabbit = self.compute_distance_gain(
            player_x, player_y, rabbit_x, rabbit_y, action_angle, self.speed
        )

        # Timing alignment for coordination
        timing_alignment = self.compute_timing_alignment(
            player_x, player_y, partner_x, partner_y, stag_x, stag_y
        )

        # Coordination probability
        p_coord = self.compute_coordination_probability(
            belief_partner_stag, timing_alignment
        )

        # Utilities with actual reward values
        u_stag = stag_value * p_coord * gain_stag
        u_rabbit = rabbit_value * gain_rabbit

        total_utility = u_stag + u_rabbit

        return total_utility

    def compute_action_probabilities(self, player_x, player_y,
                                    partner_x, partner_y,
                                    stag_x, stag_y, stag_value,
                                    rabbit_x, rabbit_y, rabbit_value,
                                    belief_partner_stag):
        """
        Compute probability distribution over actions via softmax.

        Returns:
        --------
        probs : array (n_directions,)
            Probability of each action
        utilities : array (n_directions,)
            Utility of each action
        """
        # Compute utilities for all actions
        utilities = self.compute_utility(
            player_x, player_y,
            partner_x, partner_y,
            stag_x, stag_y, stag_value,
            rabbit_x, rabbit_y, rabbit_value,
            belief_partner_stag,
            self.action_angles
        )

        # Softmax to get probabilities
        exp_utilities = np.exp(self.temperature * utilities)
        probs = exp_utilities / np.sum(exp_utilities)

        return probs, utilities

    def sample_action(self, player_x, player_y,
                     partner_x, partner_y,
                     stag_x, stag_y, stag_value,
                     rabbit_x, rabbit_y, rabbit_value,
                     belief_partner_stag):
        """
        Sample an action from the model.

        Returns:
        --------
        action_angle : float
            Sampled action direction
        action_idx : int
            Index of sampled action
        probs : array
            Probability distribution used
        """
        probs, utilities = self.compute_action_probabilities(
            player_x, player_y,
            partner_x, partner_y,
            stag_x, stag_y, stag_value,
            rabbit_x, rabbit_y, rabbit_value,
            belief_partner_stag
        )

        action_idx = np.random.choice(self.n_directions, p=probs)
        action_angle = self.action_angles[action_idx]

        return action_angle, action_idx, probs

    def get_continuous_log_likelihood(self, observed_angle,
                                     player_x, player_y,
                                     partner_x, partner_y,
                                     stag_x, stag_y, stag_value,
                                     rabbit_x, rabbit_y, rabbit_value,
                                     belief_partner_stag,
                                     action_noise):
        """
        Compute log-likelihood of observed angle under model.

        Uses continuous von Mises mixture model:
        - Each discrete action defines a von Mises component
        - Observed angle is evaluated under mixture

        Parameters:
        -----------
        observed_angle : float
            Actual observed angle (radians)
        action_noise : float
            Motor noise concentration parameter (kappa)
            Higher = more precise execution

        Returns:
        --------
        log_prob : float
            Log-probability of observed angle
        """
        # Get action probabilities from utility-based softmax
        probs, utilities = self.compute_action_probabilities(
            player_x, player_y,
            partner_x, partner_y,
            stag_x, stag_y, stag_value,
            rabbit_x, rabbit_y, rabbit_value,
            belief_partner_stag
        )

        # Compute von Mises likelihood for each action
        # vonmises.pdf is VECTORIZED over loc
        vm_probs = vonmises.pdf(observed_angle,
                               kappa=action_noise,
                               loc=self.action_angles)

        # Mixture likelihood
        total_prob = np.sum(probs * vm_probs)

        # Log likelihood (add small constant for numerical stability)
        log_prob = np.log(total_prob + 1e-10)

        return log_prob

    def evaluate_trial_continuous(self, trial_data, player='player1',
                                 belief_column='p1_belief_p2_stag',
                                 action_noise=1.0):
        """
        Evaluate model log-likelihood on a trial.

        Parameters:
        -----------
        trial_data : DataFrame
            Trial data with beliefs already computed
        player : str
            Which player to evaluate ('player1' or 'player2')
        belief_column : str
            Column name for relevant beliefs
        action_noise : float
            Motor noise parameter

        Returns:
        --------
        total_ll : float
            Total log-likelihood for trial
        mean_ll : float
            Mean log-likelihood per timestep
        results_df : DataFrame
            Per-timestep results
        """
        results = []

        # Get partner name
        partner = 'player2' if player == 'player1' else 'player1'

        for i in range(1, len(trial_data)):
            # Current positions
            px = trial_data.iloc[i][f'{player}_x']
            py = trial_data.iloc[i][f'{player}_y']
            partner_x = trial_data.iloc[i][f'{partner}_x']
            partner_y = trial_data.iloc[i][f'{partner}_y']

            # Previous positions (for computing angle)
            prev_px = trial_data.iloc[i-1][f'{player}_x']
            prev_py = trial_data.iloc[i-1][f'{player}_y']

            # Observed movement angle
            dx = px - prev_px
            dy = py - prev_py
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue  # Skip if no movement

            observed_angle = np.arctan2(dy, dx)

            # Game state
            stag_x = trial_data.iloc[i-1]['stag_x']
            stag_y = trial_data.iloc[i-1]['stag_y']
            rabbit_x = trial_data.iloc[i-1]['rabbit_x']
            rabbit_y = trial_data.iloc[i-1]['rabbit_y']
            stag_value = trial_data.iloc[i-1]['value']
            rabbit_value = 1.0

            # Belief
            belief = trial_data.iloc[i-1][belief_column]

            # Compute log-likelihood
            ll = self.get_continuous_log_likelihood(
                observed_angle,
                prev_px, prev_py,
                partner_x, partner_y,
                stag_x, stag_y, stag_value,
                rabbit_x, rabbit_y, rabbit_value,
                belief,
                action_noise
            )

            results.append({
                'timestep': i,
                'log_likelihood': ll,
                'observed_angle': observed_angle,
                'belief': belief
            })

        results_df = pd.DataFrame(results)
        total_ll = results_df['log_likelihood'].sum()
        mean_ll = results_df['log_likelihood'].mean()

        return total_ll, mean_ll, results_df


if __name__ == '__main__':
    import pandas as pd

    # Quick test
    model = CoordinatedDecisionModel(
        n_directions=8,
        temperature=1.0,
        timing_tolerance=1.0,
        speed=1.0
    )

    # Test coordination probability
    print("Testing coordination probability computation:")
    print("-" * 50)

    # Case 1: Both close, high belief
    timing1 = model.compute_timing_alignment(
        player_x=0, player_y=0,
        partner_x=0, partner_y=0,
        stag_x=100, stag_y=100
    )
    p_coord1 = model.compute_coordination_probability(
        belief_partner_stag=0.9,
        timing_alignment=timing1
    )
    print(f"Case 1 (both equidistant, high belief):")
    print(f"  Timing alignment: {timing1:.3f}")
    print(f"  P(coord): {p_coord1:.3f}\n")

    # Case 2: One far, one close
    timing2 = model.compute_timing_alignment(
        player_x=0, player_y=0,
        partner_x=150, partner_y=150,
        stag_x=100, stag_y=100
    )
    p_coord2 = model.compute_coordination_probability(
        belief_partner_stag=0.9,
        timing_alignment=timing2
    )
    print(f"Case 2 (different distances, high belief):")
    print(f"  Timing alignment: {timing2:.3f}")
    print(f"  P(coord): {p_coord2:.3f}\n")

    # Case 3: Both close, low belief
    timing3 = model.compute_timing_alignment(
        player_x=0, player_y=0,
        partner_x=0, partner_y=0,
        stag_x=100, stag_y=100
    )
    p_coord3 = model.compute_coordination_probability(
        belief_partner_stag=0.1,
        timing_alignment=timing3
    )
    print(f"Case 3 (both equidistant, low belief):")
    print(f"  Timing alignment: {timing3:.3f}")
    print(f"  P(coord): {p_coord3:.3f}\n")

    print("Model test complete!")
