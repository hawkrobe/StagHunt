"""
Fit the coordinated decision model with explicit timing.

This model has fewer free parameters because it uses actual payoffs:
- Free parameters: temperature, timing_tolerance, action_noise (3 params)
- Fixed: stag/rabbit values come from data

Compare to original model with 4 free parameters:
- Free parameters: w_stag, w_rabbit, temperature, action_noise
"""

import numpy as np
import pandas as pd
import glob
from scipy.optimize import minimize
from belief_model_distance import BayesianIntentionModel
from decision_model_coordinated import CoordinatedDecisionModel


def load_trial(filepath):
    """Load and clean a single trial CSV."""
    df = pd.read_csv(filepath)
    if 'plater1_y' in df.columns:
        df = df.rename(columns={'plater1_y': 'player1_y'})
    df = df.dropna()
    return df


def compute_beliefs_for_all_trials(trial_files):
    """Precompute beliefs for all trials (caching)."""
    print("Computing beliefs for all trials...")
    belief_model = BayesianIntentionModel(
        prior_stag=0.5,
        concentration=1.5,
        belief_bounds=(0.01, 0.99)
    )

    trials_with_beliefs = []
    for filepath in trial_files:
        trial_data = load_trial(filepath)
        trial_with_beliefs = belief_model.run_trial(trial_data)
        trials_with_beliefs.append(trial_with_beliefs)

    print(f"Beliefs cached for {len(trials_with_beliefs)} trials!\n")
    return trials_with_beliefs


def evaluate_model_on_trials(trials, temperature, timing_tolerance,
                            action_noise, n_directions=8):
    """
    Evaluate coordinated model on all trials.

    Returns total log-likelihood across all trials and both players.
    """
    model = CoordinatedDecisionModel(
        n_directions=n_directions,
        temperature=temperature,
        timing_tolerance=timing_tolerance,
        speed=1.0  # Fixed
    )

    total_ll = 0.0
    n_observations = 0

    for trial_data in trials:
        # Player 1
        ll_p1, _, _ = model.evaluate_trial_continuous(
            trial_data,
            player='player1',
            belief_column='p1_belief_p2_stag',
            action_noise=action_noise
        )
        total_ll += ll_p1

        # Player 2
        ll_p2, _, _ = model.evaluate_trial_continuous(
            trial_data,
            player='player2',
            belief_column='p2_belief_p1_stag',
            action_noise=action_noise
        )
        total_ll += ll_p2

        n_observations += (len(trial_data) - 1) * 2  # Both players

    return total_ll, n_observations


def objective_function(params, trials, n_directions):
    """
    Objective function for optimization (negative log-likelihood).

    Parameters to fit:
    - params[0]: temperature
    - params[1]: timing_tolerance
    - params[2]: action_noise
    """
    temperature, timing_tolerance, action_noise = params

    # Compute negative log-likelihood
    total_ll, n_obs = evaluate_model_on_trials(
        trials, temperature, timing_tolerance, action_noise, n_directions
    )

    neg_ll = -total_ll

    return neg_ll


def fit_model(trials, initial_params, n_directions=8, maxiter=50):
    """
    Fit coordinated model using maximum likelihood.

    Parameters:
    -----------
    trials : list of DataFrames
        Trials with beliefs already computed
    initial_params : list
        [temperature, timing_tolerance, action_noise]
    n_directions : int
        Number of discrete action directions
    maxiter : int
        Maximum optimization iterations

    Returns:
    --------
    result : OptimizeResult
        Optimization result with fitted parameters
    """
    # Bounds: all parameters must be positive
    bounds = [
        (0.01, 20.0),   # temperature
        (0.01, 10.0),   # timing_tolerance
        (0.01, 10.0),   # action_noise
    ]

    # Global counter for printing progress
    eval_counter = {'count': 0}

    def callback(xk):
        eval_counter['count'] += 1
        temperature, timing_tolerance, action_noise = xk
        ll = -objective_function(xk, trials, n_directions)

        if eval_counter['count'] % 1 == 0:  # Print every eval
            print(f"  Eval {eval_counter['count']:3d}: LL={ll:.2f} | "
                  f"temp={temperature:.3f}, timing_tol={timing_tolerance:.3f}, "
                  f"noise={action_noise:.3f}")

    print(f"Init: temp={initial_params[0]}, timing_tol={initial_params[1]}, "
          f"noise={initial_params[2]}")

    # Initial evaluation
    ll_init = -objective_function(initial_params, trials, n_directions)
    print(f"  Eval   1: LL={ll_init:.2f} | temp={initial_params[0]:.3f}, "
          f"timing_tol={initial_params[1]:.3f}, noise={initial_params[2]:.3f}")

    result = minimize(
        objective_function,
        x0=initial_params,
        args=(trials, n_directions),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'disp': False},
        callback=callback
    )

    print(f"  Optimization finished after {eval_counter['count']} evaluations\n")

    return result


def main():
    # Load trials
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*.csv'))
    print(f"Found {len(trial_files)} trials\n")

    # Compute beliefs (cached)
    trials_with_beliefs = compute_beliefs_for_all_trials(trial_files)

    print("="*70)
    print("FITTING COORDINATED DECISION MODEL")
    print("="*70)
    print("\nModel structure:")
    print("  - Stag/rabbit values = actual payoffs from data")
    print("  - Coordination prob = belief × timing_alignment")
    print("  - Free parameters: temperature, timing_tolerance, action_noise")
    print(f"\n{'='*70}\n")

    # Use 8 directions for speed
    n_directions = 8

    # Try multiple initializations
    initializations = [
        [1.0, 1.0, 1.0],     # Baseline
        [5.0, 2.0, 1.0],     # Higher temp, moderate tolerance
        [10.0, 0.5, 0.5],    # Very deterministic, tight timing
    ]

    results = []

    for i, init_params in enumerate(initializations):
        print(f"{'-'*70}")
        print(f"Initialization {i+1}/{len(initializations)}")
        print(f"{'-'*70}\n")

        result = fit_model(
            trials_with_beliefs,
            initial_params=init_params,
            n_directions=n_directions,
            maxiter=50
        )

        results.append(result)

    # Find best result
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    best_result = min(results, key=lambda r: r.fun)
    best_ll = -best_result.fun
    best_temperature, best_timing_tol, best_noise = best_result.x

    print(f"Best fit:")
    print(f"  temperature:       {best_temperature:.4f}")
    print(f"  timing_tolerance:  {best_timing_tol:.4f}")
    print(f"  action_noise:      {best_noise:.4f}")
    print(f"  Log-likelihood:    {best_ll:.2f}")
    print(f"\n{'-'*70}")

    # Compute AIC/BIC
    _, n_obs = evaluate_model_on_trials(
        trials_with_beliefs,
        best_temperature,
        best_timing_tol,
        best_noise,
        n_directions
    )

    k = 3  # Number of free parameters
    aic = 2 * k - 2 * best_ll
    bic = k * np.log(n_obs) - 2 * best_ll

    print(f"\nModel selection criteria:")
    print(f"  AIC: {aic:.2f}")
    print(f"  BIC: {bic:.2f}")
    print(f"  Observations: {n_obs}")

    print(f"\n{'='*70}")
    print("\nNext step: Compare to original model with free weights")
    print("  Original: 4 params (w_stag, w_rabbit, temp, noise)")
    print("  Coordinated: 3 params (temp, timing_tol, noise)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
