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
import json
import os
from models.decision_model_basic import UtilityDecisionModel
from models.decision_model_coordinated import CoordinatedDecisionModel
from models.belief_model_distance import BayesianIntentionModel
from models.belief_model_decision import BayesianIntentionModelWithDecision


def load_fitted_params(config_path='fitted_params.json'):
    """
    Load fitted parameters from config file.

    Returns:
    --------
    dict : Model parameters, or None if file doesn't exist
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


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
            # Load from fitted_params.json (required)
            fitted_params = load_fitted_params()

            if fitted_params is None or model_type not in fitted_params:
                raise ValueError(
                    f"No fitted parameters found for model '{model_type}'.\n"
                    f"Please run: python stag_hunt.py --fit {model_type}"
                )

            # Use fitted parameters from config
            params = {k: v for k, v in fitted_params[model_type].items()
                     if k not in ['description', 'fit_info']}

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
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Stag Hunt unified model interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit a model and save as defaults
  python stag_hunt.py --fit integrated

  # Test the unified interface
  python stag_hunt.py
        """
    )
    parser.add_argument('--fit', choices=['integrated', 'hierarchical', 'distance', 'distance_tiebreak'],
                       help='Fit model and save parameters as defaults')
    parser.add_argument('--method', default='L-BFGS-B',
                       help='Optimization method for fitting (default: L-BFGS-B)')

    args = parser.parse_args()

    # Handle --fit flag
    if args.fit:
        print(f"Fitting model '{args.fit}'...\n")
        from models.fit import fit_model, save_as_defaults

        # Run fitting
        results = fit_model(args.fit, method=args.method, verbose=True)

        # Save as defaults
        save_as_defaults(results, verbose=True)

        print(f"\n✓ Model '{args.fit}' fitted and saved as defaults")
        print(f"  You can now use: DecisionModel(model_type='{args.fit}')")
        sys.exit(0)

    # Otherwise show usage example
    print("Stag Hunt Unified Model Interface")
    print("="*70)
    print("\nUsage examples:")
    print("\n1. Use fitted defaults:")
    print("   from stag_hunt import DecisionModel, BeliefModel")
    print("   decision_model = DecisionModel(model_type='coordinated')")
    print("   belief_model = BeliefModel(inference_type='decision', decision_model=decision_model)")
    print("\n2. Use custom parameters:")
    print("   decision_model = DecisionModel(model_type='basic',")
    print("                                  params={'temperature': 5.0, 'w_stag': 1.0, ...})")
    print("\n3. Fit new defaults:")
    print("   python stag_hunt.py --fit integrated")
    print("\nFor tests, run: pytest tests/")
    print("="*70)
