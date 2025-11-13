"""
Bayesian Intention Model with Decision Model Inverse Inference

This version uses the fitted decision model to compute likelihoods of observed
movements, providing principled inverse inference from actions to intentions.
"""

import numpy as np
import pandas as pd
from decision_model_coordinated import CoordinatedDecisionModel


class BayesianIntentionModelWithDecision:
    """
    Models a player's belief about their partner's goal using inverse inference
    through a fitted decision model.

    Key improvement over simple distance-based model:
    - Uses actual decision model to compute P(movement | intention)
    - Accounts for coordination probability and timing
    - More principled likelihood computation
    """

    def __init__(self, decision_model_params, prior_stag=0.5,
                 belief_bounds=(0.01, 0.99)):
        """
        Parameters:
        -----------
        decision_model_params : dict
            Fitted parameters for the decision model:
            {
                'temperature': float,
                'timing_tolerance': float,
                'action_noise': float,
                'n_directions': int (default 8)
            }
        prior_stag : float
            Initial probability that partner is going for stag
        belief_bounds : tuple
            Min/max bounds on beliefs to maintain uncertainty
        """
        self.prior_stag = prior_stag
        self.belief_bounds = belief_bounds

        # Initialize decision model with fitted parameters
        self.decision_model = CoordinatedDecisionModel(
            n_directions=decision_model_params.get('n_directions', 8),
            temperature=decision_model_params['temperature'],
            timing_tolerance=decision_model_params['timing_tolerance'],
            speed=1.0
        )
        self.action_noise = decision_model_params['action_noise']

    def compute_movement_angle(self, x_prev, y_prev, x_curr, y_curr):
        """Compute angle of observed movement."""
        dx = x_curr - x_prev
        dy = y_curr - y_prev

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None

        return np.arctan2(dy, dx)

    def likelihood_movement_given_intention(self, observed_angle,
                                           player_x, player_y,
                                           partner_x, partner_y,
                                           stag_x, stag_y, stag_value,
                                           rabbit_x, rabbit_y, rabbit_value,
                                           partner_believes_stag):
        """
        Compute P(observed_movement | partner_intention) using decision model.

        Parameters:
        -----------
        observed_angle : float
            Observed movement angle (radians)
        player_x, player_y : float
            Observer's position (to compute coordination probability)
        partner_x, partner_y : float
            Partner's position
        stag_x, stag_y, stag_value : float
            Stag position and value
        rabbit_x, rabbit_y, rabbit_value : float
            Rabbit position and value
        partner_believes_stag : float
            What the PARTNER believes about YOU going for stag
            (used for their coordination calculation)

        Returns:
        --------
        likelihood : float
            Probability of observed movement under this intention hypothesis
        """
        if observed_angle is None:
            return 0.5

        # Use decision model to get likelihood
        # Note: we're computing from the PARTNER's perspective
        # So "player" and "partner" are swapped from their viewpoint
        log_likelihood = self.decision_model.get_continuous_log_likelihood(
            observed_angle,
            partner_x, partner_y,      # Partner's position
            player_x, player_y,         # Player's position (partner's "partner")
            stag_x, stag_y, stag_value,
            rabbit_x, rabbit_y, rabbit_value,
            partner_believes_stag,      # Partner's belief about player
            self.action_noise
        )

        # Convert log-likelihood to likelihood
        likelihood = np.exp(log_likelihood)

        # Clip to reasonable range
        return np.clip(likelihood, 1e-10, 1.0)

    def update_belief(self, belief_partner_stag,
                     player_x, player_y,
                     partner_x_prev, partner_y_prev,
                     partner_x_curr, partner_y_curr,
                     stag_x, stag_y, stag_value,
                     rabbit_x, rabbit_y, rabbit_value):
        """
        Update belief about partner's intention using Bayes' rule.

        Key insight: We need to reason about what the PARTNER believes about US
        when computing their likelihoods. This requires recursive reasoning.

        Simplified approach: Assume partner has symmetric beliefs
        (i.e., partner_believes[we go stag] ≈ we_believe[partner goes stag])
        """
        # Compute observed movement
        observed_angle = self.compute_movement_angle(
            partner_x_prev, partner_y_prev,
            partner_x_curr, partner_y_curr
        )

        if observed_angle is None:
            return belief_partner_stag

        # For partner's coordination calculation, we need to estimate what they
        # believe about us. Simplified assumption: mirror our beliefs about them.
        partner_believes_player_stag = belief_partner_stag

        # Hypothesis 1: Partner is going for stag
        # Under this hypothesis, partner's belief about player going for stag
        # affects their coordination probability
        likelihood_if_stag = self.likelihood_movement_given_intention(
            observed_angle,
            player_x, player_y,
            partner_x_prev, partner_y_prev,
            stag_x, stag_y, stag_value,
            rabbit_x, rabbit_y, rabbit_value,
            partner_believes_stag=partner_believes_player_stag
        )

        # Hypothesis 2: Partner is going for rabbit
        # Under this hypothesis, coordination doesn't matter
        # We can model this as partner having low belief about player going for stag
        likelihood_if_rabbit = self.likelihood_movement_given_intention(
            observed_angle,
            player_x, player_y,
            partner_x_prev, partner_y_prev,
            stag_x, stag_y, stag_value,
            rabbit_x, rabbit_y, rabbit_value,
            partner_believes_stag=0.01  # Partner thinks player going for rabbit
        )

        # Bayes' rule (in log space for numerical stability)
        prior = belief_partner_stag

        # Add small constant to avoid log(0)
        likelihood_if_stag = max(likelihood_if_stag, 1e-10)
        likelihood_if_rabbit = max(likelihood_if_rabbit, 1e-10)
        prior = max(min(prior, 0.99), 0.01)

        # Compute in log space
        log_numerator = np.log(likelihood_if_stag) + np.log(prior)
        log_denom_term1 = np.log(likelihood_if_stag) + np.log(prior)
        log_denom_term2 = np.log(likelihood_if_rabbit) + np.log(1 - prior)

        # LogSumExp trick for denominator
        max_log = max(log_denom_term1, log_denom_term2)
        log_denominator = max_log + np.log(
            np.exp(log_denom_term1 - max_log) +
            np.exp(log_denom_term2 - max_log)
        )

        # Final belief
        log_belief = log_numerator - log_denominator
        updated_belief = np.exp(log_belief)

        # Clip to bounds
        updated_belief = np.clip(updated_belief,
                                self.belief_bounds[0],
                                self.belief_bounds[1])

        return updated_belief

    def run_trial(self, trial_data):
        """Run belief updating on a full trial."""
        results = trial_data.copy()
        n_timesteps = len(trial_data)

        # Initialize belief arrays
        p1_beliefs = np.zeros(n_timesteps)
        p2_beliefs = np.zeros(n_timesteps)

        # Set initial beliefs (prior)
        p1_beliefs[0] = self.prior_stag
        p2_beliefs[0] = self.prior_stag

        # Update beliefs at each timestep
        for t in range(1, n_timesteps):
            # Get current game state
            stag_x = trial_data.iloc[t-1]['stag_x']
            stag_y = trial_data.iloc[t-1]['stag_y']
            rabbit_x = trial_data.iloc[t-1]['rabbit_x']
            rabbit_y = trial_data.iloc[t-1]['rabbit_y']
            stag_value = trial_data.iloc[t-1]['value']
            rabbit_value = 1.0

            # Skip if any values are NaN
            if (pd.isna(stag_x) or pd.isna(stag_y) or pd.isna(rabbit_x) or
                pd.isna(rabbit_y) or pd.isna(stag_value)):
                p1_beliefs[t] = p1_beliefs[t-1]
                p2_beliefs[t] = p2_beliefs[t-1]
                continue

            # Player 1's belief about Player 2
            try:
                p1_beliefs[t] = self.update_belief(
                    p1_beliefs[t-1],
                    player_x=trial_data.iloc[t]['player1_x'],
                    player_y=trial_data.iloc[t]['player1_y'],
                    partner_x_prev=trial_data.iloc[t-1]['player2_x'],
                    partner_y_prev=trial_data.iloc[t-1]['player2_y'],
                    partner_x_curr=trial_data.iloc[t]['player2_x'],
                    partner_y_curr=trial_data.iloc[t]['player2_y'],
                    stag_x=stag_x, stag_y=stag_y, stag_value=stag_value,
                    rabbit_x=rabbit_x, rabbit_y=rabbit_y, rabbit_value=rabbit_value
                )
                # Check for NaN
                if np.isnan(p1_beliefs[t]):
                    p1_beliefs[t] = p1_beliefs[t-1]
            except:
                p1_beliefs[t] = p1_beliefs[t-1]

            # Player 2's belief about Player 1
            try:
                p2_beliefs[t] = self.update_belief(
                    p2_beliefs[t-1],
                    player_x=trial_data.iloc[t]['player2_x'],
                    player_y=trial_data.iloc[t]['player2_y'],
                    partner_x_prev=trial_data.iloc[t-1]['player1_x'],
                    partner_y_prev=trial_data.iloc[t-1]['player1_y'],
                    partner_x_curr=trial_data.iloc[t]['player1_x'],
                    partner_y_curr=trial_data.iloc[t]['player1_y'],
                    stag_x=stag_x, stag_y=stag_y, stag_value=stag_value,
                    rabbit_x=rabbit_x, rabbit_y=rabbit_y, rabbit_value=rabbit_value
                )
                # Check for NaN
                if np.isnan(p2_beliefs[t]):
                    p2_beliefs[t] = p2_beliefs[t-1]
            except:
                p2_beliefs[t] = p2_beliefs[t-1]

        # Add beliefs to dataframe
        results['p1_belief_p2_stag'] = p1_beliefs
        results['p2_belief_p1_stag'] = p2_beliefs

        # Also compute coordination probabilities
        results['p1_coord_prob'] = self._compute_coordination_probability(
            results, player='player1'
        )
        results['p2_coord_prob'] = self._compute_coordination_probability(
            results, player='player2'
        )

        return results

    def _compute_coordination_probability(self, trial_data, player='player1'):
        """Compute coordination probability at each timestep."""
        coord_probs = np.zeros(len(trial_data))

        if player == 'player1':
            partner = 'player2'
            belief_col = 'p1_belief_p2_stag'
        else:
            partner = 'player1'
            belief_col = 'p2_belief_p1_stag'

        for t in range(len(trial_data)):
            player_x = trial_data.iloc[t][f'{player}_x']
            player_y = trial_data.iloc[t][f'{player}_y']
            partner_x = trial_data.iloc[t][f'{partner}_x']
            partner_y = trial_data.iloc[t][f'{partner}_y']
            stag_x = trial_data.iloc[t]['stag_x']
            stag_y = trial_data.iloc[t]['stag_y']
            belief = trial_data.iloc[t][belief_col]

            # Compute timing alignment
            timing_alignment = self.decision_model.compute_timing_alignment(
                player_x, player_y,
                partner_x, partner_y,
                stag_x, stag_y
            )

            # Coordination probability = belief × timing
            # Add floor to prevent extreme values
            coord_probs[t] = max(
                self.decision_model.compute_coordination_probability(
                    belief, timing_alignment
                ),
                1e-6  # Minimum coordination probability
            )

        return coord_probs


if __name__ == '__main__':
    """Test the model on sample data."""
    import glob

    print("Testing decision-based belief model...")

    # Load a trial
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*.csv'))
    trial_data = pd.read_csv(trial_files[0])

    # Fix column name typo if present
    if 'plater1_y' in trial_data.columns:
        trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})

    # Initialize model with fitted parameters
    fitted_params = {
        'temperature': 3.049,
        'timing_tolerance': 0.865,
        'action_noise': 10.0,
        'n_directions': 8
    }

    model = BayesianIntentionModelWithDecision(
        decision_model_params=fitted_params,
        prior_stag=0.5,
        belief_bounds=(0.01, 0.99)
    )

    # Run model
    print("\nRunning model on trial 1...")
    results = model.run_trial(trial_data)

    print(f"\nBelief trajectory (first 10 timesteps):")
    print(results[['p1_belief_p2_stag', 'p2_belief_p1_stag',
                   'p1_coord_prob', 'p2_coord_prob']].head(10))

    print(f"\nFinal beliefs:")
    print(f"  P1 believes P2 going for stag: {results.iloc[-1]['p1_belief_p2_stag']:.3f}")
    print(f"  P2 believes P1 going for stag: {results.iloc[-1]['p2_belief_p1_stag']:.3f}")

    print("\nModel test complete!")
