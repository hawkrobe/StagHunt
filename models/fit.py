#!/usr/bin/env python3
"""
Fit Stag Hunt models with a simple command-line interface.

Usage:
    python fitting/fit_model.py integrated
    python fitting/fit_model.py hierarchical --method nelder-mead
    python fitting/fit_model.py distance --output results.json
"""

import numpy as np
import pandas as pd
import glob
import argparse
import json
from scipy.optimize import minimize
from typing import Dict, List, Tuple


# ==============================================================================
# Model Configurations
# ==============================================================================

MODELS = {
    'integrated': {
        'description': 'Integrated hierarchical model with cross-trial learning',
        'params': ['learning_rate', 'goal_temperature', 'execution_temperature', 'timing_tolerance'],
        'bounds': [(0.01, 0.99), (0.1, 10.0), (0.1, 20.0), (10.0, 300.0)],
        'initial': [[0.2, 2.0, 12.0, 150.0],
                   [0.3, 2.4, 15.0, 150.0],
                   [0.5, 2.0, 12.0, 150.0]],
        'requires_beliefs': False
    },

    'hierarchical': {
        'description': 'Hierarchical goal + plan model',
        'params': ['goal_temperature', 'execution_temperature'],
        'bounds': [(0.1, 10.0), (0.1, 20.0)],
        'initial': [[2.0, 12.0],
                   [3.0, 15.0]],
        'requires_beliefs': True
    },

    'distance': {
        'description': 'Distance-based planning',
        'params': ['temperature'],
        'bounds': [(0.1, 10.0)],
        'initial': [[1.0], [2.0], [3.0]],
        'requires_beliefs': False
    },

    'distance_tiebreak': {
        'description': 'Distance with probabilistic tie-breaking',
        'params': ['temperature', 'tiebreak_prob'],
        'bounds': [(0.1, 10.0), (0.0, 1.0)],
        'initial': [[1.0, 0.5], [2.0, 0.5]],
        'requires_beliefs': False
    },
}


# ==============================================================================
# Data Loading
# ==============================================================================

def load_trials(with_beliefs=False):
    """Load all trials, optionally with precomputed beliefs."""
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*_2024_08_24_0848.csv'))
    trials = []

    for filepath in trial_files:
        df = pd.read_csv(filepath)
        if 'plater1_y' in df.columns:
            df = df.rename(columns={'plater1_y': 'player1_y'})
        df = df.dropna()
        trials.append(df)

    if with_beliefs:
        from models.belief_model_decision import BayesianIntentionModelWithDecision

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

        trials_with_beliefs = []
        for trial in trials:
            trial_with_beliefs = belief_model.run_trial(trial.copy())
            trials_with_beliefs.append(trial_with_beliefs)

        return trials_with_beliefs

    return trials


# ==============================================================================
# Model Objective Functions
# ==============================================================================

def get_objective_function(model_name, trials):
    """Get the objective function for a model."""

    if model_name == 'integrated':
        from models.hierarchical_model_with_cross_trial_learning import IntegratedHierarchicalModel

        def objective(params):
            learning_rate, goal_temp, exec_temp, timing_tolerance = params

            if (learning_rate <= 0 or learning_rate >= 1 or
                goal_temp <= 0 or exec_temp <= 0 or timing_tolerance <= 0):
                return 1e10

            model = IntegratedHierarchicalModel(
                initial_prior=0.5,
                learning_rate=learning_rate,
                goal_temperature=goal_temp,
                execution_temperature=exec_temp,
                timing_tolerance=timing_tolerance,
                action_noise=5.0,
                n_directions=8
            )

            total_ll = 0.0
            for trial_idx, trial_data in enumerate(trials):
                try:
                    trial_data = model.process_trial(trial_data.copy(), trial_idx + 1)
                    for player in ['player1', 'player2']:
                        ll = model.decision_model.evaluate_trajectory(trial_data, player)
                        total_ll += ll
                except:
                    continue

            return -total_ll

    elif model_name == 'hierarchical':
        from models.hierarchical_goal_model import HierarchicalGoalModel

        def objective(params):
            goal_temp, exec_temp = params

            if goal_temp <= 0 or exec_temp <= 0:
                return 1e10

            model = HierarchicalGoalModel(
                goal_temperature=goal_temp,
                execution_temperature=exec_temp
            )

            total_ll = 0.0
            for trial in trials:
                for player in ['player1', 'player2']:
                    try:
                        ll = model.evaluate_trajectory(trial, player)
                        total_ll += ll
                    except:
                        continue

            return -total_ll

    elif model_name == 'distance':
        from models.model_comparison_framework import DistanceBasedModel

        def objective(params):
            temperature = params[0]

            if temperature <= 0:
                return 1e10

            model = DistanceBasedModel(temperature=temperature)

            total_ll = 0.0
            for trial in trials:
                for player in ['player1', 'player2']:
                    try:
                        ll = model.evaluate_trajectory(trial, player)
                        total_ll += ll
                    except:
                        continue

            return -total_ll

    elif model_name == 'distance_tiebreak':
        from models.distance_with_random_tiebreak import DistanceModelWithTiebreak

        def objective(params):
            temperature, tiebreak_prob = params

            if temperature <= 0 or tiebreak_prob < 0 or tiebreak_prob > 1:
                return 1e10

            model = DistanceModelWithTiebreak(
                temperature=temperature,
                tiebreak_prob=tiebreak_prob
            )

            total_ll = 0.0
            for trial in trials:
                for player in ['player1', 'player2']:
                    try:
                        ll = model.evaluate_trajectory(trial, player)
                        total_ll += ll
                    except:
                        continue

            return -total_ll

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return objective


# ==============================================================================
# Fitting
# ==============================================================================

def fit_model(model_name, method='L-BFGS-B', verbose=True):
    """Fit a model to data."""

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    config = MODELS[model_name]

    if verbose:
        print("=" * 70)
        print(f"FITTING MODEL: {model_name.upper()}")
        print("=" * 70)
        print(f"Description: {config['description']}")
        print(f"Parameters: {', '.join(config['params'])}")
        print()

    # Load data
    trials = load_trials(with_beliefs=config['requires_beliefs'])

    if verbose:
        print(f"Loaded {len(trials)} trials")

    # Get objective function
    objective = get_objective_function(model_name, trials)

    # Try multiple initializations
    best_result = None
    best_ll = -np.inf

    if verbose:
        print(f"Trying {len(config['initial'])} initializations...")

    for i, init_params in enumerate(config['initial']):
        if verbose:
            param_str = ', '.join(f'{name}={val:.3f}'
                                 for name, val in zip(config['params'], init_params))
            print(f"\n  Init {i+1}: {param_str}")

        result = minimize(
            objective,
            x0=init_params,
            method=method,
            bounds=config['bounds'],
            options={'maxiter': 1000}
        )

        ll = -result.fun

        if verbose:
            print(f"    → LL = {ll:.2f}")

        if ll > best_ll:
            best_ll = ll
            best_result = result

    # Package results
    optimal_params = {name: val for name, val in
                     zip(config['params'], best_result.x)}

    # Compute AIC/BIC
    n_params = len(config['params'])
    n_data = sum(len(trial) for trial in trials) * 2  # Both players
    aic = 2 * n_params - 2 * best_ll
    bic = n_params * np.log(n_data) - 2 * best_ll

    results = {
        'model': model_name,
        'parameters': optimal_params,
        'log_likelihood': best_ll,
        'aic': aic,
        'bic': bic,
        'n_parameters': n_params,
        'n_data_points': n_data,
        'success': best_result.success
    }

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print("\nOptimal parameters:")
        for name, value in optimal_params.items():
            print(f"  {name:25s} = {value:.6f}")

        print(f"\nModel fit:")
        print(f"  Log-likelihood:  {best_ll:.2f}")
        print(f"  AIC:             {aic:.2f}")
        print(f"  BIC:             {bic:.2f}")
        print(f"  Success:         {best_result.success}")

    return results


def save_as_defaults(results, verbose=True):
    """
    Save fitted parameters as defaults in fitted_params.json.

    Parameters:
    -----------
    results : dict
        Results dictionary from fit_model()
    verbose : bool
        Print confirmation message
    """
    import json
    from datetime import datetime

    config_path = 'fitted_params.json'

    # Load current config (or create empty if doesn't exist)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}

    model_name = results['model']

    # Prepare updated entry
    model_config = results['parameters'].copy()
    model_config['description'] = MODELS[model_name]['description']
    model_config['fit_info'] = {
        'method': 'optimize',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'log_likelihood': results['log_likelihood'],
        'aic': results['aic'],
        'bic': results['bic']
    }

    # Update config
    config[model_name] = model_config

    # Save
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"\n✓ Saved fitted parameters to {config_path}")
        print(f"  Model '{model_name}' defaults updated")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fit Stag Hunt models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit integrated model
  python fitting/fit_model.py --model integrated

  # Fit with different optimization method
  python fitting/fit_model.py --model hierarchical --method nelder-mead

  # Save results to file
  python fitting/fit_model.py --model distance --output results.json

  # List available models
  python fitting/fit_model.py --list
        """
    )

    parser.add_argument('--model', choices=list(MODELS.keys()),
                       help='Model to fit')
    parser.add_argument('--list', action='store_true',
                       help='List available models')
    parser.add_argument('--method', default='L-BFGS-B',
                       help='Optimization method (default: L-BFGS-B)')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--save-defaults', action='store_true',
                       help='Save fitted parameters as defaults in fitted_params.json')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')

    args = parser.parse_args()

    # List models
    if args.list:
        print("Available models:")
        for name, config in MODELS.items():
            print(f"\n  {name}:")
            print(f"    {config['description']}")
            print(f"    Parameters: {', '.join(config['params'])}")
        return

    if not args.model:
        parser.print_help()
        return

    # Fit model
    results = fit_model(args.model, method=args.method, verbose=not args.quiet)

    # Save as defaults
    if args.save_defaults:
        save_as_defaults(results, verbose=not args.quiet)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == '__main__':
    main()
