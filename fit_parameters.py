"""
Fit decision model parameters to behavioral data.

Free parameters:
- w_stag: Weight on stag utility
- w_rabbit: Weight on rabbit utility
- temperature: Softmax inverse temperature (decision noise)
- action_noise: Motor noise (von Mises concentration)

Uses scipy.optimize to maximize log-likelihood.
"""

import numpy as np
import pandas as pd
import glob
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


def negative_log_likelihood(params, trial_data_list, belief_model, n_directions=16):
    """
    Compute negative log-likelihood for optimization.

    Parameters to fit:
    - params[0]: w_stag (constrained > 0)
    - params[1]: w_rabbit (constrained > 0)
    - params[2]: temperature (constrained > 0)
    - params[3]: action_noise (constrained > 0)
    """
    w_stag, w_rabbit, temperature, action_noise = params

    # Safety bounds
    if w_stag <= 0 or w_rabbit <= 0 or temperature <= 0 or action_noise <= 0:
        return 1e10  # Return large penalty

    # Create decision model with these parameters
    decision_model = UtilityDecisionModel(
        n_directions=n_directions,
        temperature=temperature,
        w_stag=w_stag,
        w_rabbit=w_rabbit
    )

    # Compute total log-likelihood across all trials
    total_log_lik = 0

    for trial_data in trial_data_list:
        # Run belief model
        trial_with_beliefs = belief_model.run_trial(trial_data)

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

    # Return NEGATIVE log-likelihood (for minimization)
    return -total_log_lik


def fit_parameters(trial_data_list, belief_model,
                   initial_params=None, n_directions=16):
    """
    Fit parameters using scipy.optimize.

    Returns:
    --------
    result : OptimizeResult
        Optimization result with fitted parameters
    """
    # Initial parameter values
    if initial_params is None:
        initial_params = [
            1.0,  # w_stag
            1.0,  # w_rabbit
            1.0,  # temperature
            1.0   # action_noise
        ]

    print("Starting parameter optimization...")
    print(f"Initial params: w_stag={initial_params[0]:.3f}, "
          f"w_rabbit={initial_params[1]:.3f}, "
          f"temp={initial_params[2]:.3f}, "
          f"noise={initial_params[3]:.3f}")

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
        args=(trial_data_list, belief_model, n_directions),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': 100}
    )

    return result


def main():
    # Load all trials
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*.csv'))
    print(f"Loading {len(trial_files)} trials...")

    trial_data_list = [load_trial(f) for f in trial_files]

    # Initialize belief model
    belief_model = BayesianIntentionModel(
        prior_stag=0.5,
        concentration=1.5,
        belief_bounds=(0.01, 0.99)
    )

    print(f"\n{'='*70}")
    print("FITTING DECISION MODEL PARAMETERS")
    print(f"{'='*70}\n")

    # Fit parameters
    result = fit_parameters(trial_data_list, belief_model, n_directions=16)

    # Extract fitted parameters
    w_stag_fit, w_rabbit_fit, temp_fit, noise_fit = result.x

    print(f"\n{'='*70}")
    print("FITTED PARAMETERS")
    print(f"{'='*70}")
    print(f"w_stag:       {w_stag_fit:.4f}")
    print(f"w_rabbit:     {w_rabbit_fit:.4f}")
    print(f"temperature:  {temp_fit:.4f}")
    print(f"action_noise: {noise_fit:.4f}")
    print(f"\nNegative log-likelihood: {result.fun:.2f}")
    print(f"Log-likelihood:          {-result.fun:.2f}")
    print(f"Success: {result.success}")
    print(f"{'='*70}\n")

    # Compare with baseline (equal weights, temp=1, noise=1)
    print("Comparing with baseline model (all params = 1.0)...")
    baseline_params = [1.0, 1.0, 1.0, 1.0]
    baseline_nll = negative_log_likelihood(
        baseline_params, trial_data_list, belief_model, n_directions=16
    )

    print(f"\nBaseline log-likelihood:    {-baseline_nll:.2f}")
    print(f"Fitted log-likelihood:      {-result.fun:.2f}")
    print(f"Improvement:                {baseline_nll - result.fun:.2f}")
    print(f"{'='*70}\n")

    # Test different initializations to check for local minima
    print("Testing different initializations...")
    print(f"{'-'*70}")

    initializations = [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 0.5, 2.0, 2.0],
        [0.5, 2.0, 0.5, 0.5],
        [3.0, 3.0, 3.0, 3.0],
    ]

    results = []
    for i, init in enumerate(initializations):
        print(f"\nInit {i+1}: w_stag={init[0]}, w_rabbit={init[1]}, "
              f"temp={init[2]}, noise={init[3]}")
        res = fit_parameters(trial_data_list, belief_model,
                           initial_params=init, n_directions=16)
        results.append(res)
        print(f"  Final LL: {-res.fun:.2f}")
        print(f"  Params: w_stag={res.x[0]:.3f}, w_rabbit={res.x[1]:.3f}, "
              f"temp={res.x[2]:.3f}, noise={res.x[3]:.3f}")

    # Find best
    best_result = min(results, key=lambda r: r.fun)

    print(f"\n{'='*70}")
    print("BEST FIT ACROSS ALL INITIALIZATIONS")
    print(f"{'='*70}")
    print(f"w_stag:       {best_result.x[0]:.4f}")
    print(f"w_rabbit:     {best_result.x[1]:.4f}")
    print(f"temperature:  {best_result.x[2]:.4f}")
    print(f"action_noise: {best_result.x[3]:.4f}")
    print(f"\nLog-likelihood: {-best_result.fun:.2f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
