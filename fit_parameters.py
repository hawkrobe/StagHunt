"""
Fit decision model parameters to behavioral data.

Free parameters:
- w_stag: Weight on stag utility
- w_rabbit: Weight on rabbit utility
- temperature: Softmax inverse temperature (decision noise)
- action_noise: Motor noise (von Mises concentration)

Features:
- Cached belief computation (huge speedup)
- Progress tracking during optimization
- Configurable thoroughness (directions, iterations, initializations)
- Comparison with baseline and no-belief models

Usage:
    python fit_parameters.py                    # Fast mode (default, for development)
    python fit_parameters.py --production       # Production mode (for final results)
    python fit_parameters.py --n-directions 16  # Custom settings
"""

import numpy as np
import pandas as pd
import glob
import argparse
from scipy.optimize import minimize
from belief_model_distance import BayesianIntentionModel
from decision_model_basic import UtilityDecisionModel


def load_trial(filepath):
    """Load and clean a single trial CSV."""
    df = pd.read_csv(filepath)
    if 'plater1_y' in df.columns:
        df = df.rename(columns={'plater1_y': 'player1_y'})
    df = df.dropna()
    return df


# Global cache for beliefs (computed once, reused many times)
BELIEF_CACHE = {}


def get_beliefs_for_trials(trial_data_list, belief_model):
    """Get beliefs for all trials (cached)."""
    cache_key = id(trial_data_list[0])  # Simple cache key

    if cache_key not in BELIEF_CACHE:
        print("Computing beliefs for all trials (this happens once)...")
        beliefs = []
        for i, trial_data in enumerate(trial_data_list):
            trial_with_beliefs = belief_model.run_trial(trial_data)
            beliefs.append(trial_with_beliefs)
            print(f"  Trial {i+1}/{len(trial_data_list)} beliefs computed")
        BELIEF_CACHE[cache_key] = beliefs
        print("Beliefs cached!\n")

    return BELIEF_CACHE[cache_key]


# Global counter for function evaluations
EVAL_COUNTER = {'count': 0, 'best_ll': -np.inf}


def negative_log_likelihood(params, trials_with_beliefs, n_directions=8):
    """
    Compute negative log-likelihood for optimization.

    Parameters:
    - params[0]: w_stag
    - params[1]: w_rabbit
    - params[2]: temperature
    - params[3]: action_noise
    """
    EVAL_COUNTER['count'] += 1

    w_stag, w_rabbit, temperature, action_noise = params

    # Safety bounds
    if w_stag <= 0 or w_rabbit <= 0 or temperature <= 0 or action_noise <= 0:
        return 1e10

    # Create decision model with these parameters
    decision_model = UtilityDecisionModel(
        n_directions=n_directions,
        temperature=temperature,
        w_stag=w_stag,
        w_rabbit=w_rabbit
    )

    # Compute total log-likelihood across all trials
    total_log_lik = 0

    for trial_with_beliefs in trials_with_beliefs:
        # Evaluate both players
        p1_ll, _, _ = decision_model.evaluate_trial_continuous(
            trial_with_beliefs,
            player='player1',
            belief_column='p1_belief_p2_stag',
            action_noise=action_noise
        )
        p2_ll, _, _ = decision_model.evaluate_trial_continuous(
            trial_with_beliefs,
            player='player2',
            belief_column='p2_belief_p1_stag',
            action_noise=action_noise
        )

        total_log_lik += p1_ll + p2_ll

    nll = -total_log_lik

    # Track best and print progress every few evaluations
    if total_log_lik > EVAL_COUNTER['best_ll']:
        EVAL_COUNTER['best_ll'] = total_log_lik
        print(f"Eval {EVAL_COUNTER['count']:3d}: LL={total_log_lik:8.2f} | "
              f"w_stag={w_stag:.3f}, w_rabbit={w_rabbit:.3f}, "
              f"temp={temperature:.3f}, noise={action_noise:.3f}")
    elif EVAL_COUNTER['count'] % 5 == 0:
        print(f"Eval {EVAL_COUNTER['count']:3d}: LL={total_log_lik:8.2f}")

    return nll


def fit_parameters(trials_with_beliefs, initial_params=None,
                   n_directions=8, maxiter=20):
    """
    Fit parameters using scipy.optimize.

    Parameters
    ----------
    trials_with_beliefs : list of DataFrames
        Trial data with beliefs already computed
    initial_params : list, optional
        Starting values for [w_stag, w_rabbit, temperature, action_noise]
    n_directions : int, optional
        Number of discrete directions in action space
    maxiter : int, optional
        Maximum iterations for optimizer
    """
    if initial_params is None:
        initial_params = [1.0, 1.0, 1.0, 1.0]

    print(f"\nStarting optimization from: w_stag={initial_params[0]:.3f}, "
          f"w_rabbit={initial_params[1]:.3f}, "
          f"temp={initial_params[2]:.3f}, "
          f"noise={initial_params[3]:.3f}")

    # Reset counter
    EVAL_COUNTER['count'] = 0
    EVAL_COUNTER['best_ll'] = -np.inf

    # Bounds: all parameters must be positive
    bounds = [
        (0.01, 10.0),   # w_stag
        (0.01, 10.0),   # w_rabbit
        (0.01, 10.0),   # temperature
        (0.1, 10.0)     # action_noise
    ]

    # Optimize
    result = minimize(
        negative_log_likelihood,
        x0=initial_params,
        args=(trials_with_beliefs, n_directions),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': False, 'maxiter': maxiter}
    )

    print(f"Optimization finished after {EVAL_COUNTER['count']} evaluations\n")

    return result


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Fit decision model parameters to behavioral data'
    )
    parser.add_argument('--production', action='store_true',
                        help='Use production mode (more directions, iterations, initializations for final results)')
    parser.add_argument('--n-directions', type=int, default=None,
                        help='Number of discrete directions (default: 8, production: 16)')
    parser.add_argument('--maxiter', type=int, default=None,
                        help='Maximum optimizer iterations (default: 20, production: 100)')
    parser.add_argument('--n-inits', type=int, default=None,
                        help='Number of random initializations (default: 2, production: 4)')

    args = parser.parse_args()

    # Set defaults based on mode
    if args.production:
        n_directions = args.n_directions or 16
        maxiter = args.maxiter or 100
        n_inits = args.n_inits or 4
        mode_name = "PRODUCTION"
    else:
        n_directions = args.n_directions or 8
        maxiter = args.maxiter or 20
        n_inits = args.n_inits or 2
        mode_name = "FAST (DEVELOPMENT)"

    print(f"{'='*70}")
    print(f"{mode_name} PARAMETER FITTING")
    print(f"Settings: {n_directions} directions, {maxiter} max iterations, {n_inits} initializations")
    print(f"{'='*70}\n")

    # Load all trials
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*.csv'))
    print(f"Loading {len(trial_files)} trials...")
    trial_data_list = [load_trial(f) for f in trial_files]
    print("Trials loaded!\n")

    # Initialize belief model
    belief_model = BayesianIntentionModel(
        prior_stag=0.5,
        concentration=1.5,
        belief_bounds=(0.01, 0.99)
    )

    # Pre-compute beliefs (cached for all optimizations)
    trials_with_beliefs = get_beliefs_for_trials(trial_data_list, belief_model)

    # Compute baseline
    print(f"{'-'*70}")
    print("Baseline model (all params = 1.0):")
    print(f"{'-'*70}")
    baseline_nll = negative_log_likelihood(
        [1.0, 1.0, 1.0, 1.0], trials_with_beliefs, n_directions=n_directions
    )
    print(f"\nBaseline log-likelihood: {-baseline_nll:.2f}\n")

    # Define initializations based on n_inits
    all_initializations = [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 0.5, 2.0, 2.0],
        [0.5, 2.0, 0.5, 0.5],
        [3.0, 3.0, 3.0, 3.0],
    ]
    initializations = all_initializations[:n_inits]

    print(f"{'='*70}")
    print("FITTING WITH DIFFERENT INITIALIZATIONS")
    print(f"{'='*70}\n")

    results = []
    for i, init in enumerate(initializations):
        print(f"{'-'*70}")
        print(f"Initialization {i+1}/{len(initializations)}")
        print(f"{'-'*70}")

        res = fit_parameters(
            trials_with_beliefs,
            initial_params=init,
            n_directions=n_directions,
            maxiter=maxiter
        )
        results.append(res)

        print(f"\nFinal results:")
        print(f"  w_stag:       {res.x[0]:.4f}")
        print(f"  w_rabbit:     {res.x[1]:.4f}")
        print(f"  temperature:  {res.x[2]:.4f}")
        print(f"  action_noise: {res.x[3]:.4f}")
        print(f"  Log-likelihood: {-res.fun:.2f}")
        print(f"  Success: {res.success}")
        print()

    # Find best
    best_result = min(results, key=lambda r: r.fun)

    print(f"{'='*70}")
    print("BEST FIT ACROSS ALL INITIALIZATIONS")
    print(f"{'='*70}")
    print(f"w_stag:       {best_result.x[0]:.4f}")
    print(f"w_rabbit:     {best_result.x[1]:.4f}")
    print(f"temperature:  {best_result.x[2]:.4f}")
    print(f"action_noise: {best_result.x[3]:.4f}")
    print(f"\nLog-likelihood: {-best_result.fun:.2f}")
    print(f"Improvement over baseline: {baseline_nll - best_result.fun:.2f}")
    print(f"{'='*70}\n")

    # Compare with and without beliefs
    print(f"{'='*70}")
    print("COMPARING WITH AND WITHOUT BELIEFS")
    print(f"{'='*70}\n")

    # Create no-belief trials
    trials_no_beliefs = []
    for trial in trials_with_beliefs:
        trial_nb = trial.copy()
        trial_nb['p1_belief_p2_stag'] = 1.0
        trial_nb['p2_belief_p1_stag'] = 1.0
        trials_no_beliefs.append(trial_nb)

    # Evaluate best params without beliefs
    print("Evaluating best parameters without beliefs...")
    no_belief_nll = negative_log_likelihood(
        best_result.x, trials_no_beliefs, n_directions=n_directions
    )

    print(f"\nWith beliefs:    {-best_result.fun:.2f}")
    print(f"Without beliefs: {-no_belief_nll:.2f}")
    print(f"Belief contribution: {no_belief_nll - best_result.fun:.2f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
