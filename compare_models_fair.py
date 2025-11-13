"""
Fair model comparison: WITH vs WITHOUT beliefs.

Both models are fit independently to find their optimal parameters,
then we compare their best log-likelihoods.
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


# Global cache for beliefs
BELIEF_CACHE = {}


def get_beliefs_for_trials(trial_data_list, belief_model):
    """Get beliefs for all trials (cached)."""
    cache_key = id(trial_data_list[0])
    if cache_key not in BELIEF_CACHE:
        print("Computing beliefs for all trials...")
        beliefs = []
        for i, trial_data in enumerate(trial_data_list):
            trial_with_beliefs = belief_model.run_trial(trial_data)
            beliefs.append(trial_with_beliefs)
        BELIEF_CACHE[cache_key] = beliefs
        print("Beliefs cached!\n")
    return BELIEF_CACHE[cache_key]


# Global counter for function evaluations
EVAL_COUNTER = {'count': 0, 'best_ll': -np.inf}


def negative_log_likelihood(params, trials_data, n_directions=8):
    """Compute negative log-likelihood."""
    EVAL_COUNTER['count'] += 1

    w_stag, w_rabbit, temperature, action_noise = params

    if w_stag <= 0 or w_rabbit <= 0 or temperature <= 0 or action_noise <= 0:
        return 1e10

    decision_model = UtilityDecisionModel(
        n_directions=n_directions,
        temperature=temperature,
        w_stag=w_stag,
        w_rabbit=w_rabbit
    )

    total_log_lik = 0
    for trial_data in trials_data:
        p1_ll, _, _ = decision_model.evaluate_trial_continuous(
            trial_data,
            player='player1',
            belief_column='p1_belief_p2_stag',
            action_noise=action_noise
        )
        p2_ll, _, _ = decision_model.evaluate_trial_continuous(
            trial_data,
            player='player2',
            belief_column='p2_belief_p1_stag',
            action_noise=action_noise
        )
        total_log_lik += p1_ll + p2_ll

    nll = -total_log_lik

    if total_log_lik > EVAL_COUNTER['best_ll']:
        EVAL_COUNTER['best_ll'] = total_log_lik
        print(f"  Eval {EVAL_COUNTER['count']:3d}: LL={total_log_lik:8.2f} | "
              f"w_stag={w_stag:.3f}, w_rabbit={w_rabbit:.3f}, "
              f"temp={temperature:.3f}, noise={action_noise:.3f}")
    elif EVAL_COUNTER['count'] % 10 == 0:
        print(f"  Eval {EVAL_COUNTER['count']:3d}: LL={total_log_lik:8.2f}")

    return nll


def fit_model(trials_data, initial_params=None, n_directions=8):
    """Fit parameters using scipy.optimize."""
    if initial_params is None:
        initial_params = [1.0, 1.0, 1.0, 1.0]

    EVAL_COUNTER['count'] = 0
    EVAL_COUNTER['best_ll'] = -np.inf

    bounds = [
        (0.01, 10.0),   # w_stag
        (0.01, 10.0),   # w_rabbit
        (0.01, 10.0),   # temperature
        (0.1, 10.0)     # action_noise
    ]

    result = minimize(
        negative_log_likelihood,
        x0=initial_params,
        args=(trials_data, n_directions),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': False, 'maxiter': 20}
    )

    print(f"  Optimization finished after {EVAL_COUNTER['count']} evaluations\n")
    return result


def main():
    print(f"{'='*70}")
    print("FAIR MODEL COMPARISON: WITH vs WITHOUT BELIEFS")
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

    # Pre-compute beliefs
    trials_with_beliefs = get_beliefs_for_trials(trial_data_list, belief_model)

    # =========================================================================
    # MODEL 1: WITH BELIEFS (fit independently)
    # =========================================================================
    print(f"{'='*70}")
    print("MODEL 1: WITH BELIEFS")
    print(f"{'='*70}\n")

    print("Fitting model WITH dynamic beliefs...")
    print(f"{'-'*70}\n")

    # Try two initializations
    results_with = []
    for init_params in [[1.0, 1.0, 1.0, 1.0], [2.0, 0.5, 2.0, 2.0]]:
        print(f"Init: w_stag={init_params[0]}, w_rabbit={init_params[1]}, "
              f"temp={init_params[2]}, noise={init_params[3]}")
        res = fit_model(trials_with_beliefs, initial_params=init_params, n_directions=8)
        results_with.append(res)

    best_with = min(results_with, key=lambda r: r.fun)

    print(f"{'-'*70}")
    print("BEST FIT WITH BELIEFS:")
    print(f"  w_stag:       {best_with.x[0]:.4f}")
    print(f"  w_rabbit:     {best_with.x[1]:.4f}")
    print(f"  temperature:  {best_with.x[2]:.4f}")
    print(f"  action_noise: {best_with.x[3]:.4f}")
    print(f"  Log-likelihood: {-best_with.fun:.2f}")
    print(f"{'-'*70}\n")

    # =========================================================================
    # MODEL 2: WITHOUT BELIEFS (fit independently)
    # =========================================================================
    print(f"{'='*70}")
    print("MODEL 2: WITHOUT BELIEFS")
    print(f"{'='*70}\n")

    # Create no-belief trials (beliefs fixed at 1.0)
    trials_no_beliefs = []
    for trial in trials_with_beliefs:
        trial_nb = trial.copy()
        trial_nb['p1_belief_p2_stag'] = 1.0
        trial_nb['p2_belief_p1_stag'] = 1.0
        trials_no_beliefs.append(trial_nb)

    print("Fitting model WITHOUT beliefs (beliefs fixed at 1.0)...")
    print(f"{'-'*70}\n")

    # Try two initializations
    results_without = []
    for init_params in [[1.0, 1.0, 1.0, 1.0], [2.0, 0.5, 2.0, 2.0]]:
        print(f"Init: w_stag={init_params[0]}, w_rabbit={init_params[1]}, "
              f"temp={init_params[2]}, noise={init_params[3]}")
        res = fit_model(trials_no_beliefs, initial_params=init_params, n_directions=8)
        results_without.append(res)

    best_without = min(results_without, key=lambda r: r.fun)

    print(f"{'-'*70}")
    print("BEST FIT WITHOUT BELIEFS:")
    print(f"  w_stag:       {best_without.x[0]:.4f}")
    print(f"  w_rabbit:     {best_without.x[1]:.4f}")
    print(f"  temperature:  {best_without.x[2]:.4f}")
    print(f"  action_noise: {best_without.x[3]:.4f}")
    print(f"  Log-likelihood: {-best_without.fun:.2f}")
    print(f"{'-'*70}\n")

    # =========================================================================
    # FAIR COMPARISON
    # =========================================================================
    print(f"{'='*70}")
    print("FAIR MODEL COMPARISON")
    print(f"{'='*70}\n")

    ll_with = -best_with.fun
    ll_without = -best_without.fun
    improvement = ll_with - ll_without

    print(f"Model WITH beliefs:     {ll_with:9.2f}")
    print(f"Model WITHOUT beliefs:  {ll_without:9.2f}")
    print(f"{'-'*70}")
    print(f"Improvement:            {improvement:+9.2f}")
    print()

    # Number of parameters (both models have 4 params)
    n_params = 4
    n_datapoints = sum([len(trial) for trial in trials_with_beliefs])

    # AIC comparison
    aic_with = -2 * ll_with + 2 * n_params
    aic_without = -2 * ll_without + 2 * n_params

    print(f"AIC WITH beliefs:       {aic_with:9.2f}")
    print(f"AIC WITHOUT beliefs:    {aic_without:9.2f}")
    print(f"Δ AIC:                  {aic_without - aic_with:+9.2f} (negative = WITH wins)")
    print()

    # BIC comparison
    bic_with = -2 * ll_with + n_params * np.log(n_datapoints)
    bic_without = -2 * ll_without + n_params * np.log(n_datapoints)

    print(f"BIC WITH beliefs:       {bic_with:9.2f}")
    print(f"BIC WITHOUT beliefs:    {bic_without:9.2f}")
    print(f"Δ BIC:                  {bic_without - bic_with:+9.2f} (negative = WITH wins)")
    print()

    print(f"{'='*70}\n")

    # Parameter comparison
    print(f"{'='*70}")
    print("PARAMETER COMPARISON")
    print(f"{'='*70}\n")

    print(f"{'Parameter':<15} {'WITH beliefs':>15} {'WITHOUT beliefs':>15} {'Difference':>15}")
    print(f"{'-'*70}")
    print(f"{'w_stag':<15} {best_with.x[0]:15.4f} {best_without.x[0]:15.4f} {best_with.x[0] - best_without.x[0]:+15.4f}")
    print(f"{'w_rabbit':<15} {best_with.x[1]:15.4f} {best_without.x[1]:15.4f} {best_with.x[1] - best_without.x[1]:+15.4f}")
    print(f"{'temperature':<15} {best_with.x[2]:15.4f} {best_without.x[2]:15.4f} {best_with.x[2] - best_without.x[2]:+15.4f}")
    print(f"{'action_noise':<15} {best_with.x[3]:15.4f} {best_without.x[3]:15.4f} {best_with.x[3] - best_without.x[3]:+15.4f}")
    print(f"{'-'*70}")
    print(f"{'w_ratio':<15} {best_with.x[1]/best_with.x[0]:15.4f} {best_without.x[1]/best_without.x[0]:15.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
