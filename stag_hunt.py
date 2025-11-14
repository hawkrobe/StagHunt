"""
Unified Models for Stag Hunt Cooperation Task

This module provides clean entrypoints for all model variants.

Quick Start:
------------
```python
from models import DecisionModel, BeliefModel

# Recommended: Coordinated decision model
decision_model = DecisionModel(
    model_type='coordinated',
    params={'temperature': 3.049, 'timing_tolerance': 0.865, 'action_noise': 10.0}
)

# Recommended: Decision-based belief inference
belief_model = BeliefModel(
    inference_type='decision',
    decision_model=decision_model
)

# Run belief updating on trial
trial_with_beliefs = belief_model.run_trial(trial_data)
```

Model Types:
------------
DecisionModel:
  - 'coordinated': Explicit timing constraints (recommended)
  - 'basic': Free weight parameters

BeliefModel:
  - 'decision': Inverse inference through decision model (recommended)
  - 'distance': Distance-based heuristics (faster, less principled)
"""

import numpy as np
import pandas as pd
from models.decision_model_basic import UtilityDecisionModel
from models.decision_model_coordinated import CoordinatedDecisionModel
from models.belief_model_distance import BayesianIntentionModel
from models.belief_model_decision import BayesianIntentionModelWithDecision


class DecisionModel:
    """
    Unified interface for decision models.

    Parameters:
    -----------
    model_type : str
        'coordinated' - Explicit timing constraints (recommended)
        'basic' - Free weight parameters
    params : dict
        Model-specific parameters

    For 'coordinated':
        - temperature: float
        - timing_tolerance: float
        - action_noise: float
        - n_directions: int (default 8)

    For 'basic':
        - temperature: float
        - w_stag: float
        - w_rabbit: float
        - action_noise: float
        - n_directions: int (default 8)
    """

    def __init__(self, model_type='coordinated', params=None):
        self.model_type = model_type

        if params is None:
            # Use fitted defaults
            if model_type == 'coordinated':
                params = {
                    'temperature': 3.049,
                    'timing_tolerance': 0.865,
                    'action_noise': 10.0,
                    'n_directions': 8
                }
            elif model_type == 'basic':
                params = {
                    'temperature': 9.39,
                    'w_stag': 4.98,
                    'w_rabbit': 8.76,
                    'action_noise': 0.73,
                    'n_directions': 8
                }

        self.params = params

        # Initialize underlying model
        if model_type == 'coordinated':
            self.model = CoordinatedDecisionModel(
                n_directions=params.get('n_directions', 8),
                temperature=params['temperature'],
                timing_tolerance=params['timing_tolerance'],
                speed=1.0
            )
        elif model_type == 'basic':
            self.model = UtilityDecisionModel(
                n_directions=params.get('n_directions', 8),
                temperature=params['temperature'],
                w_stag=params['w_stag'],
                w_rabbit=params['w_rabbit'],
                speed=1.0
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def __getattr__(self, name):
        """Delegate method calls to underlying model."""
        return getattr(self.model, name)

    def __repr__(self):
        return f"DecisionModel(type='{self.model_type}', params={self.params})"


class BeliefModel:
    """
    Unified interface for belief inference models.

    Parameters:
    -----------
    inference_type : str
        'decision' - Use decision model for inverse inference (recommended)
        'distance' - Use distance-based heuristics (faster)
    decision_model : DecisionModel or dict, optional
        For 'decision' type: either DecisionModel instance or params dict
    prior_stag : float
        Initial belief that partner goes for stag (default 0.5)
    belief_bounds : tuple
        (min, max) bounds on beliefs (default (0.01, 0.99))
    concentration : float
        Only for 'distance' type: likelihood sharpness (default 1.5)
    """

    def __init__(self, inference_type='decision', decision_model=None,
                 prior_stag=0.5, belief_bounds=(0.01, 0.99), concentration=1.5):
        self.inference_type = inference_type

        if inference_type == 'decision':
            # Decision-based inference
            if decision_model is None:
                # Use default coordinated model
                decision_model = DecisionModel(model_type='coordinated')
            elif isinstance(decision_model, dict):
                # Create from params
                decision_model = DecisionModel(**decision_model)

            # Extract params for the belief model
            if hasattr(decision_model, 'params'):
                decision_params = decision_model.params
            else:
                decision_params = {
                    'temperature': decision_model.temperature,
                    'timing_tolerance': decision_model.timing_tolerance,
                    'action_noise': getattr(decision_model, 'action_noise', 1.0),
                    'n_directions': decision_model.n_directions
                }

            self.model = BayesianIntentionModelWithDecision(
                decision_model_params=decision_params,
                prior_stag=prior_stag,
                belief_bounds=belief_bounds
            )

        elif inference_type == 'distance':
            # Distance-based inference
            self.model = BayesianIntentionModel(
                prior_stag=prior_stag,
                concentration=concentration,
                belief_bounds=belief_bounds,
                forgetting_rate=0.0
            )
        else:
            raise ValueError(f"Unknown inference_type: {inference_type}")

    def run_trial(self, trial_data):
        """Run belief inference on a trial."""
        return self.model.run_trial(trial_data)

    def __repr__(self):
        return f"BeliefModel(type='{self.inference_type}')"


# Convenience function for recommended configuration
def get_recommended_models():
    """
    Get recommended model configuration.

    Returns:
    --------
    decision_model : DecisionModel
        Coordinated model with fitted parameters
    belief_model : BeliefModel
        Decision-based inference using the decision model
    """
    decision_model = DecisionModel(model_type='coordinated')
    belief_model = BeliefModel(
        inference_type='decision',
        decision_model=decision_model
    )
    return decision_model, belief_model


if __name__ == '__main__':
    """Test the unified interface."""
    print("Testing unified model interface...\n")

    # Test 1: Recommended configuration
    print("="*70)
    print("Test 1: Recommended configuration")
    print("="*70)
    decision_model, belief_model = get_recommended_models()
    print(f"Decision model: {decision_model}")
    print(f"Belief model: {belief_model}")

    # Test 2: Custom configuration
    print("\n" + "="*70)
    print("Test 2: Custom configuration")
    print("="*70)

    # Basic decision model
    decision_basic = DecisionModel(
        model_type='basic',
        params={'temperature': 5.0, 'w_stag': 1.0, 'w_rabbit': 2.0, 'action_noise': 1.0}
    )
    print(f"Basic decision model: {decision_basic}")

    # Distance-based belief
    belief_distance = BeliefModel(
        inference_type='distance',
        concentration=2.0
    )
    print(f"Distance belief model: {belief_distance}")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
