"""
Simplified Bayesian Intention Model (removes concentration/noise redundancy)

This version uses only concentration + belief bounds, removing the redundant
noise_level parameter for clarity.
"""

import numpy as np
import pandas as pd
from scipy.stats import vonmises


class BayesianIntentionModel:
    """
    Models a player's belief about their partner's goal (stag vs. rabbit)
    using Bayesian updating based on observed movement directions.
    
    All uncertainty is controlled via the concentration parameter.
    """
    
    def __init__(self, prior_stag=0.5, concentration=1.5, 
                 belief_bounds=(0.01, 0.99), forgetting_rate=0.0):
        """
        Parameters:
        -----------
        prior_stag : float
            Initial probability that partner is going for stag (0 to 1)
        concentration : float
            Controls how diagnostic movement direction is for inferring goals.
            Higher = more diagnostic (sharper updates)
            Lower = less diagnostic (gradual updates, more uncertain)
            Recommended range: 1.0-2.0 for noisy real data
        belief_bounds : tuple (min, max)
            Bounds to prevent beliefs from hitting 0 or 1 exactly.
            Default (0.01, 0.99) maintains minimum uncertainty.
        forgetting_rate : float
            Rate of decay toward prior (0 = no forgetting, higher = faster decay).
            Implements: belief_t = (1-λ) × belief_updated + λ × prior
        """
        self.prior_stag = prior_stag
        self.concentration = concentration
        self.belief_bounds = belief_bounds
        self.forgetting_rate = forgetting_rate
        
    def compute_movement_vector(self, x1, y1, x2, y2):
        """Compute movement vector and angle from position 1 to position 2."""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return 0, 0, None
            
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        return dx, dy, angle
    
    def compute_direction_to_target(self, player_x, player_y, target_x, target_y):
        """Compute angle from player's position to target."""
        dx = target_x - player_x
        dy = target_y - player_y
        
        if dx == 0 and dy == 0:
            return None
            
        return np.arctan2(dy, dx)
    
    def likelihood_movement_given_goal(self, movement_angle, target_angle):
        """
        Likelihood of observed movement angle given partner's goal.
        
        Uses von Mises (circular normal) distribution centered on the
        direction toward the target.
        """
        if movement_angle is None or target_angle is None:
            return 0.5
        
        # von Mises PDF
        likelihood = vonmises.pdf(movement_angle, 
                                  kappa=self.concentration, 
                                  loc=target_angle)
        
        # Normalize by maximum possible likelihood
        max_likelihood = vonmises.pdf(0, kappa=self.concentration, loc=0)
        normalized = likelihood / max_likelihood
        
        # Clip to prevent extreme likelihoods
        # Even "perfect" movements aren't 100% diagnostic
        # Even "wrong" movements aren't 0% probable
        return np.clip(normalized, 0.1, 0.9)
    
    def update_belief(self, belief_stag, partner_x_prev, partner_y_prev, 
                      partner_x_curr, partner_y_curr,
                      stag_x, stag_y, rabbit_x, rabbit_y):
        """Update belief about partner's goal using Bayes' rule."""
        
        # Compute observed movement
        _, _, movement_angle = self.compute_movement_vector(
            partner_x_prev, partner_y_prev,
            partner_x_curr, partner_y_curr
        )
        
        if movement_angle is None:
            return belief_stag, None
        
        # Compute expected directions to each target
        angle_to_stag = self.compute_direction_to_target(
            partner_x_prev, partner_y_prev, stag_x, stag_y
        )
        angle_to_rabbit = self.compute_direction_to_target(
            partner_x_prev, partner_y_prev, rabbit_x, rabbit_y
        )
        
        # Compute likelihoods
        likelihood_if_stag = self.likelihood_movement_given_goal(
            movement_angle, angle_to_stag
        )
        likelihood_if_rabbit = self.likelihood_movement_given_goal(
            movement_angle, angle_to_rabbit
        )
        
        # Bayes' rule
        numerator = likelihood_if_stag * belief_stag
        denominator = (likelihood_if_stag * belief_stag + 
                      likelihood_if_rabbit * (1 - belief_stag))
        
        if denominator == 0:
            updated_belief = belief_stag
        else:
            updated_belief = numerator / denominator
        
        # Apply forgetting (optional)
        if self.forgetting_rate > 0:
            updated_belief = ((1 - self.forgetting_rate) * updated_belief + 
                            self.forgetting_rate * self.prior_stag)
        
        # Clip beliefs to maintain minimum uncertainty
        updated_belief = np.clip(updated_belief, 
                                self.belief_bounds[0], 
                                self.belief_bounds[1])
        
        return updated_belief, movement_angle
    
    def run_trial(self, trial_data):
        """Run the model on a single trial's data."""
        results = trial_data.copy()
        n_timesteps = len(trial_data)
        
        # Initialize belief arrays
        p1_beliefs = np.zeros(n_timesteps)
        p2_beliefs = np.zeros(n_timesteps)
        p1_angles = np.full(n_timesteps, np.nan)
        p2_angles = np.full(n_timesteps, np.nan)
        
        # Set initial beliefs (priors)
        p1_beliefs[0] = self.prior_stag
        p2_beliefs[0] = self.prior_stag
        
        # Update beliefs at each timestep
        for t in range(1, n_timesteps):
            # Get positions
            p1_x_prev = trial_data.iloc[t-1]['player1_x']
            p1_y_prev = trial_data.iloc[t-1]['player1_y']
            p1_x_curr = trial_data.iloc[t]['player1_x']
            p1_y_curr = trial_data.iloc[t]['player1_y']
            
            p2_x_prev = trial_data.iloc[t-1]['player2_x']
            p2_y_prev = trial_data.iloc[t-1]['player2_y']
            p2_x_curr = trial_data.iloc[t]['player2_x']
            p2_y_curr = trial_data.iloc[t]['player2_y']
            
            stag_x = trial_data.iloc[t]['stag_x']
            stag_y = trial_data.iloc[t]['stag_y']
            rabbit_x = trial_data.iloc[t]['rabbit_x']
            rabbit_y = trial_data.iloc[t]['rabbit_y']
            
            # Player 1 updates belief about Player 2
            p1_beliefs[t], p2_angles[t] = self.update_belief(
                p1_beliefs[t-1],
                p2_x_prev, p2_y_prev, p2_x_curr, p2_y_curr,
                stag_x, stag_y, rabbit_x, rabbit_y
            )
            
            # Player 2 updates belief about Player 1
            p2_beliefs[t], p1_angles[t] = self.update_belief(
                p2_beliefs[t-1],
                p1_x_prev, p1_y_prev, p1_x_curr, p1_y_curr,
                stag_x, stag_y, rabbit_x, rabbit_y
            )
        
        # Add results to dataframe
        results['p1_belief_p2_stag'] = p1_beliefs
        results['p2_belief_p1_stag'] = p2_beliefs
        results['p1_movement_angle'] = p1_angles
        results['p2_movement_angle'] = p2_angles
        
        return results
