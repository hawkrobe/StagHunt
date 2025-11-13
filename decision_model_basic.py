"""
Utility-Based Decision Model for Stag Hunt

Models how players choose movement directions based on:
- Expected values of prey (stag and rabbit)
- Beliefs about partner's intentions
- Current distances to prey

Uses softmax choice rule over discrete action space (directions).
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp


class UtilityDecisionModel:
    """
    Models player's joystick direction choices using utility maximization.

    At each timestep, the player evaluates the utility of moving in different
    directions based on how much closer each direction brings them to valuable
    prey targets.
    """

    def __init__(self, n_directions=16, temperature=1.0,
                 w_stag=1.0, w_rabbit=1.0, speed=1.0):
        """
        Parameters:
        -----------
        n_directions : int
            Number of discrete directions in action space (e.g., 8, 16, 32)
            Evenly spaced around the circle [0, 2π)
        temperature : float
            Inverse temperature β for softmax. Higher = more deterministic.
            β → ∞: always choose highest utility action
            β → 0: uniform random choice
        w_stag : float
            Weight/importance of stag (relative attraction)
        w_rabbit : float
            Weight/importance of rabbit (relative attraction)
        speed : float
            Distance moved per timestep (for computing distance gains)
        """
        self.n_directions = n_directions
        self.temperature = temperature
        self.w_stag = w_stag
        self.w_rabbit = w_rabbit
        self.speed = speed

        # Pre-compute action space: evenly spaced angles
        self.action_angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

    def compute_distance(self, x1, y1, x2, y2):
        """Euclidean distance between two points."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def compute_distance_after_action(self, player_x, player_y,
                                      target_x, target_y, action_angle):
        """
        Compute distance to target after moving in direction action_angle.

        Projects player's position forward by speed in the action direction,
        then computes new distance to target.
        """
        # New position after action
        new_x = player_x + self.speed * np.cos(action_angle)
        new_y = player_y + self.speed * np.sin(action_angle)

        # Distance from new position to target
        new_distance = self.compute_distance(new_x, new_y, target_x, target_y)

        return new_distance

    def compute_distance_gain(self, player_x, player_y,
                             target_x, target_y, action_angle):
        """
        Distance gain: how much closer does this action bring you to target?

        Positive values = getting closer
        Negative values = getting farther
        """
        current_distance = self.compute_distance(player_x, player_y,
                                                 target_x, target_y)
        new_distance = self.compute_distance_after_action(
            player_x, player_y, target_x, target_y, action_angle
        )

        return current_distance - new_distance

    def compute_utility(self, player_x, player_y,
                       stag_x, stag_y, stag_value,
                       rabbit_x, rabbit_y, rabbit_value,
                       belief_partner_stag, action_angle):
        """
        Compute utility of taking action (moving in direction action_angle).

        U(action) = w_stag × belief × stag_value × gain_stag +
                    w_rabbit × rabbit_value × gain_rabbit

        The belief term models coordination requirement: stag is only valuable
        if you believe your partner will also go for it.
        """
        # Distance gains for each target
        gain_stag = self.compute_distance_gain(
            player_x, player_y, stag_x, stag_y, action_angle
        )
        gain_rabbit = self.compute_distance_gain(
            player_x, player_y, rabbit_x, rabbit_y, action_angle
        )

        # Utility components
        u_stag = self.w_stag * belief_partner_stag * stag_value * gain_stag
        u_rabbit = self.w_rabbit * rabbit_value * gain_rabbit

        total_utility = u_stag + u_rabbit

        return total_utility

    def compute_action_probabilities(self, player_x, player_y,
                                     stag_x, stag_y, stag_value,
                                     rabbit_x, rabbit_y, rabbit_value,
                                     belief_partner_stag):
        """
        Compute probability distribution over actions using softmax.

        P(action) ∝ exp(β × U(action))

        Returns:
        --------
        probs : array of shape (n_directions,)
            Probability of each action
        utilities : array of shape (n_directions,)
            Utility of each action
        """
        # Compute utility for each possible action
        utilities = np.array([
            self.compute_utility(
                player_x, player_y,
                stag_x, stag_y, stag_value,
                rabbit_x, rabbit_y, rabbit_value,
                belief_partner_stag, angle
            )
            for angle in self.action_angles
        ])

        # Softmax: P(a) = exp(β × U(a)) / Σ exp(β × U(a'))
        log_probs = self.temperature * utilities
        log_probs = log_probs - logsumexp(log_probs)  # Normalize (log space)
        probs = np.exp(log_probs)

        return probs, utilities

    def sample_action(self, player_x, player_y,
                     stag_x, stag_y, stag_value,
                     rabbit_x, rabbit_y, rabbit_value,
                     belief_partner_stag):
        """
        Sample an action from the probability distribution.

        Returns:
        --------
        action_angle : float
            Selected direction angle
        action_idx : int
            Index of selected action
        probs : array
            Full probability distribution
        """
        probs, utilities = self.compute_action_probabilities(
            player_x, player_y,
            stag_x, stag_y, stag_value,
            rabbit_x, rabbit_y, rabbit_value,
            belief_partner_stag
        )

        # Sample from distribution
        action_idx = np.random.choice(self.n_directions, p=probs)
        action_angle = self.action_angles[action_idx]

        return action_angle, action_idx, probs

    def get_continuous_log_likelihood(self, observed_angle,
                                       player_x, player_y,
                                       stag_x, stag_y, stag_value,
                                       rabbit_x, rabbit_y, rabbit_value,
                                       belief_partner_stag,
                                       action_noise=1.0):
        """
        Compute continuous log-likelihood of an observed angle.

        Uses a mixture of von Mises distributions, one centered on each discrete
        action, weighted by the action probabilities from softmax.

        P(θ_obs) = Σ_i P(action_i) × VonMises(θ_obs; μ=action_i, κ=action_noise)

        This treats the discrete actions as a representation of a continuous
        directional distribution.

        Parameters:
        -----------
        observed_angle : float
            The actual angle the player moved (radians)
        action_noise : float
            Concentration parameter κ for von Mises (motor noise).
            Higher = more precise execution (lower noise)
            Lower = more variable execution (higher noise)
            Typical range: 1-10

        Returns:
        --------
        log_prob : float
            Log probability of the observed angle under the mixture model
        """
        from scipy.stats import vonmises

        # Get action probabilities from utility-based softmax
        probs, utilities = self.compute_action_probabilities(
            player_x, player_y,
            stag_x, stag_y, stag_value,
            rabbit_x, rabbit_y, rabbit_value,
            belief_partner_stag
        )

        # Compute mixture likelihood (VECTORIZED for speed)
        # P(θ) = Σ_i P(action_i) × VonMises(θ | μ_i, κ)

        # Compute all von Mises PDFs at once
        vm_probs = vonmises.pdf(observed_angle,
                               kappa=action_noise,
                               loc=self.action_angles)

        # Weighted sum
        total_prob = np.sum(probs * vm_probs)

        # Log probability (add small constant for numerical stability)
        log_prob = np.log(total_prob + 1e-10)

        return log_prob

    def get_log_likelihood_of_action(self, observed_angle,
                                     player_x, player_y,
                                     stag_x, stag_y, stag_value,
                                     rabbit_x, rabbit_y, rabbit_value,
                                     belief_partner_stag):
        """
        Compute log-likelihood of an observed action under the model.

        Finds the closest action in our discrete action space to the observed
        continuous angle, then returns its log probability.

        Parameters:
        -----------
        observed_angle : float
            The actual angle the player moved (radians)
        ... (other parameters as above)

        Returns:
        --------
        log_prob : float
            Log probability of the observed action
        closest_action_idx : int
            Index of closest discrete action
        """
        probs, utilities = self.compute_action_probabilities(
            player_x, player_y,
            stag_x, stag_y, stag_value,
            rabbit_x, rabbit_y, rabbit_value,
            belief_partner_stag
        )

        # Find closest action to observed angle
        # Handle circular distance (angles wrap around at 2π)
        angular_distances = np.abs(self.action_angles - observed_angle)
        angular_distances = np.minimum(angular_distances,
                                      2*np.pi - angular_distances)
        closest_action_idx = np.argmin(angular_distances)

        # Log probability of closest action
        log_prob = np.log(probs[closest_action_idx] + 1e-10)  # Add small constant for numerical stability

        return log_prob, closest_action_idx

    def evaluate_trial(self, trial_data, player='player1',
                      belief_column='p2_belief_p1_stag'):
        """
        Evaluate model's log-likelihood on a trial.

        Parameters:
        -----------
        trial_data : DataFrame
            Trial data with positions, values, and beliefs
        player : str
            Which player to evaluate ('player1' or 'player2')
        belief_column : str
            Column containing this player's belief about their partner

        Returns:
        --------
        total_log_lik : float
            Sum of log-likelihoods across all timesteps
        mean_log_lik : float
            Mean log-likelihood per timestep
        results_df : DataFrame
            Trial data with added columns for model predictions
        """
        results = trial_data.copy()
        n_timesteps = len(trial_data)

        # Arrays to store results
        log_likelihoods = np.full(n_timesteps, np.nan)
        predicted_actions = np.full(n_timesteps, np.nan)

        # Skip first timestep (no previous position for movement angle)
        for t in range(1, n_timesteps):
            # Get current state
            if player == 'player1':
                player_x = trial_data.iloc[t-1]['player1_x']
                player_y = trial_data.iloc[t-1]['player1_y']
                # Movement angle from t-1 to t
                dx = trial_data.iloc[t]['player1_x'] - player_x
                dy = trial_data.iloc[t]['player1_y'] - player_y
            else:
                player_x = trial_data.iloc[t-1]['player2_x']
                player_y = trial_data.iloc[t-1]['player2_y']
                dx = trial_data.iloc[t]['player2_x'] - player_x
                dy = trial_data.iloc[t]['player2_y'] - player_y

            # Skip if no movement
            if dx == 0 and dy == 0:
                continue

            observed_angle = np.arctan2(dy, dx)

            stag_x = trial_data.iloc[t-1]['stag_x']
            stag_y = trial_data.iloc[t-1]['stag_y']
            rabbit_x = trial_data.iloc[t-1]['rabbit_x']
            rabbit_y = trial_data.iloc[t-1]['rabbit_y']
            stag_value = trial_data.iloc[t-1]['value']
            rabbit_value = 1.0  # Assume rabbit has constant value

            belief = trial_data.iloc[t-1][belief_column]

            # Compute log-likelihood
            log_lik, action_idx = self.get_log_likelihood_of_action(
                observed_angle,
                player_x, player_y,
                stag_x, stag_y, stag_value,
                rabbit_x, rabbit_y, rabbit_value,
                belief
            )

            log_likelihoods[t] = log_lik
            predicted_actions[t] = self.action_angles[action_idx]

        # Add to results
        results[f'{player}_log_lik'] = log_likelihoods
        results[f'{player}_predicted_angle'] = predicted_actions

        # Compute summary statistics (excluding NaN values)
        valid_log_liks = log_likelihoods[~np.isnan(log_likelihoods)]
        total_log_lik = np.sum(valid_log_liks)
        mean_log_lik = np.mean(valid_log_liks) if len(valid_log_liks) > 0 else np.nan

        return total_log_lik, mean_log_lik, results

    def evaluate_trial_continuous(self, trial_data, player='player1',
                                  belief_column='p2_belief_p1_stag',
                                  action_noise=1.0):
        """
        Evaluate model's log-likelihood on a trial using CONTINUOUS likelihood.

        This is more principled than discretizing observed angles - it treats
        the discrete action space as defining a continuous mixture distribution.

        Parameters:
        -----------
        trial_data : DataFrame
            Trial data with positions, values, and beliefs
        player : str
            Which player to evaluate ('player1' or 'player2')
        belief_column : str
            Column containing this player's belief about their partner
        action_noise : float
            Motor noise parameter (von Mises concentration)

        Returns:
        --------
        total_log_lik : float
            Sum of log-likelihoods across all timesteps
        mean_log_lik : float
            Mean log-likelihood per timestep
        results_df : DataFrame
            Trial data with added columns for model predictions
        """
        results = trial_data.copy()
        n_timesteps = len(trial_data)

        # Arrays to store results
        log_likelihoods = np.full(n_timesteps, np.nan)

        # Skip first timestep (no previous position for movement angle)
        for t in range(1, n_timesteps):
            # Get current state
            if player == 'player1':
                player_x = trial_data.iloc[t-1]['player1_x']
                player_y = trial_data.iloc[t-1]['player1_y']
                # Movement angle from t-1 to t
                dx = trial_data.iloc[t]['player1_x'] - player_x
                dy = trial_data.iloc[t]['player1_y'] - player_y
            else:
                player_x = trial_data.iloc[t-1]['player2_x']
                player_y = trial_data.iloc[t-1]['player2_y']
                dx = trial_data.iloc[t]['player2_x'] - player_x
                dy = trial_data.iloc[t]['player2_y'] - player_y

            # Skip if no movement
            if dx == 0 and dy == 0:
                continue

            observed_angle = np.arctan2(dy, dx)

            stag_x = trial_data.iloc[t-1]['stag_x']
            stag_y = trial_data.iloc[t-1]['stag_y']
            rabbit_x = trial_data.iloc[t-1]['rabbit_x']
            rabbit_y = trial_data.iloc[t-1]['rabbit_y']
            stag_value = trial_data.iloc[t-1]['value']
            rabbit_value = 1.0  # Assume rabbit has constant value

            belief = trial_data.iloc[t-1][belief_column]

            # Compute continuous log-likelihood
            log_lik = self.get_continuous_log_likelihood(
                observed_angle,
                player_x, player_y,
                stag_x, stag_y, stag_value,
                rabbit_x, rabbit_y, rabbit_value,
                belief,
                action_noise=action_noise
            )

            log_likelihoods[t] = log_lik

        # Add to results
        results[f'{player}_log_lik_continuous'] = log_likelihoods

        # Compute summary statistics (excluding NaN values)
        valid_log_liks = log_likelihoods[~np.isnan(log_likelihoods)]
        total_log_lik = np.sum(valid_log_liks)
        mean_log_lik = np.mean(valid_log_liks) if len(valid_log_liks) > 0 else np.nan

        return total_log_lik, mean_log_lik, results
