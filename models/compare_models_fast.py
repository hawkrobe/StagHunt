#!/usr/bin/env python3
"""
Fast vectorized model comparison for Stag Hunt.

All computations are vectorized with NumPy - no Python loops over timesteps.
Uses cosine similarity instead of von Mises for speed.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import load_trial, find_trial_files, get_trial_info


# =============================================================================
# Vectorized Model Functions
# =============================================================================

def compute_movement_angles(trial: pd.DataFrame, player: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute movement angles and mask for valid movements."""
    dx = np.diff(trial[f'{player}_x'].values)
    dy = np.diff(trial[f'{player}_y'].values)

    # Valid movements (not stationary)
    valid = (np.abs(dx) > 0.5) | (np.abs(dy) > 0.5)
    angles = np.arctan2(dy, dx)

    return angles, valid


def angle_to_target(player_x: np.ndarray, player_y: np.ndarray,
                    target_x: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    """Compute angles from player positions to target positions."""
    dx = target_x - player_x
    dy = target_y - player_y
    return np.arctan2(dy, dx)


def cosine_log_likelihood(observed: np.ndarray, predicted: np.ndarray,
                          kappa: float = 2.0) -> float:
    """
    Fast log-likelihood using cosine similarity.

    LL = sum(kappa * cos(observed - predicted))

    This is equivalent to von Mises up to a constant.
    """
    angle_diff = observed - predicted
    return np.sum(kappa * np.cos(angle_diff))


# =============================================================================
# Model Implementations (Vectorized)
# =============================================================================

def null_model_ll(trial: pd.DataFrame, player: str) -> float:
    """Null model: uniform random. LL = n * log(1/2π)"""
    angles, valid = compute_movement_angles(trial, player)
    n_valid = valid.sum()
    return n_valid * np.log(1 / (2 * np.pi))


def distance_model_ll(trial: pd.DataFrame, player: str, kappa: float = 2.0) -> float:
    """Distance model: move toward closer target."""
    angles, valid = compute_movement_angles(trial, player)
    if not valid.any():
        return 0.0

    # Get positions (using t for current position, t+1 for targets)
    px = trial[f'{player}_x'].values[:-1][valid]
    py = trial[f'{player}_y'].values[:-1][valid]

    stag_x = trial['stag_x'].values[1:][valid]
    stag_y = trial['stag_y'].values[1:][valid]
    rabbit_x = trial['rabbit_x'].values[1:][valid]
    rabbit_y = trial['rabbit_y'].values[1:][valid]

    # Distance to each target
    dist_stag = np.sqrt((stag_x - px)**2 + (stag_y - py)**2)
    dist_rabbit = np.sqrt((rabbit_x - px)**2 + (rabbit_y - py)**2)

    # Choose closer target
    go_stag = dist_stag < dist_rabbit

    # Predicted angle (toward closer target)
    pred_angle = np.where(
        go_stag,
        angle_to_target(px, py, stag_x, stag_y),
        angle_to_target(px, py, rabbit_x, rabbit_y)
    )

    return cosine_log_likelihood(angles[valid], pred_angle, kappa)


def belief_model_ll(trial: pd.DataFrame, player: str, kappa: float = 2.0) -> float:
    """Belief model: weight targets by belief."""
    angles, valid = compute_movement_angles(trial, player)
    if not valid.any():
        return 0.0

    # Get positions
    px = trial[f'{player}_x'].values[:-1][valid]
    py = trial[f'{player}_y'].values[:-1][valid]

    stag_x = trial['stag_x'].values[1:][valid]
    stag_y = trial['stag_y'].values[1:][valid]
    rabbit_x = trial['rabbit_x'].values[1:][valid]
    rabbit_y = trial['rabbit_y'].values[1:][valid]

    # Get beliefs
    partner = '2' if player[-1] == '1' else '1'
    belief_col = f'p{player[-1]}_belief_p{partner}_stag'
    if belief_col in trial.columns:
        beliefs = trial[belief_col].values[:-1][valid]
    else:
        beliefs = np.full(valid.sum(), 0.5)

    # Angles to targets
    angle_stag = angle_to_target(px, py, stag_x, stag_y)
    angle_rabbit = angle_to_target(px, py, rabbit_x, rabbit_y)

    # Predicted angle: weighted circular mean
    # Use complex exponentials for circular averaging
    z_stag = np.exp(1j * angle_stag)
    z_rabbit = np.exp(1j * angle_rabbit)
    z_mean = beliefs * z_stag + (1 - beliefs) * z_rabbit
    pred_angle = np.angle(z_mean)

    return cosine_log_likelihood(angles[valid], pred_angle, kappa)


def coordination_model_ll(trial: pd.DataFrame, player: str,
                          kappa: float = 2.0, timing_tol: float = 150.0) -> float:
    """Coordination model: P_coord = belief × timing_alignment."""
    angles, valid = compute_movement_angles(trial, player)
    if not valid.any():
        return 0.0

    partner_name = 'player2' if player == 'player1' else 'player1'

    # Get positions
    px = trial[f'{player}_x'].values[:-1][valid]
    py = trial[f'{player}_y'].values[:-1][valid]
    partner_x = trial[f'{partner_name}_x'].values[:-1][valid]
    partner_y = trial[f'{partner_name}_y'].values[:-1][valid]

    stag_x = trial['stag_x'].values[1:][valid]
    stag_y = trial['stag_y'].values[1:][valid]
    rabbit_x = trial['rabbit_x'].values[1:][valid]
    rabbit_y = trial['rabbit_y'].values[1:][valid]

    # Get beliefs
    partner = '2' if player[-1] == '1' else '1'
    belief_col = f'p{player[-1]}_belief_p{partner}_stag'
    if belief_col in trial.columns:
        beliefs = trial[belief_col].values[:-1][valid]
    else:
        beliefs = np.full(valid.sum(), 0.5)

    # Compute P_coord = belief × timing_alignment
    dist_player = np.sqrt((stag_x - px)**2 + (stag_y - py)**2)
    dist_partner = np.sqrt((stag_x - partner_x)**2 + (stag_y - partner_y)**2)
    time_diff = np.abs(dist_player - dist_partner)
    timing_align = np.exp(-0.5 * (time_diff / timing_tol)**2)
    P_coord = beliefs * timing_align

    # Angles to targets
    angle_stag = angle_to_target(px, py, stag_x, stag_y)
    angle_rabbit = angle_to_target(px, py, rabbit_x, rabbit_y)

    # Weighted circular mean
    z_stag = np.exp(1j * angle_stag)
    z_rabbit = np.exp(1j * angle_rabbit)
    z_mean = P_coord * z_stag + (1 - P_coord) * z_rabbit
    pred_angle = np.angle(z_mean)

    return cosine_log_likelihood(angles[valid], pred_angle, kappa)


def hierarchical_model_ll(trial: pd.DataFrame, player: str,
                          kappa: float = 2.0, goal_temp: float = 2.0,
                          timing_tol: float = 150.0) -> float:
    """Hierarchical: softmax goal selection + execution."""
    angles, valid = compute_movement_angles(trial, player)
    if not valid.any():
        return 0.0

    partner_name = 'player2' if player == 'player1' else 'player1'

    # Get positions
    px = trial[f'{player}_x'].values[:-1][valid]
    py = trial[f'{player}_y'].values[:-1][valid]
    partner_x = trial[f'{partner_name}_x'].values[:-1][valid]
    partner_y = trial[f'{partner_name}_y'].values[:-1][valid]

    stag_x = trial['stag_x'].values[1:][valid]
    stag_y = trial['stag_y'].values[1:][valid]
    rabbit_x = trial['rabbit_x'].values[1:][valid]
    rabbit_y = trial['rabbit_y'].values[1:][valid]

    # Get beliefs
    partner = '2' if player[-1] == '1' else '1'
    belief_col = f'p{player[-1]}_belief_p{partner}_stag'
    if belief_col in trial.columns:
        beliefs = trial[belief_col].values[:-1][valid]
    else:
        beliefs = np.full(valid.sum(), 0.5)

    # Compute P_coord for utility
    dist_player = np.sqrt((stag_x - px)**2 + (stag_y - py)**2)
    dist_partner = np.sqrt((stag_x - partner_x)**2 + (stag_y - partner_y)**2)
    time_diff = np.abs(dist_player - dist_partner)
    timing_align = np.exp(-0.5 * (time_diff / timing_tol)**2)
    P_coord = beliefs * timing_align

    # Goal utilities
    U_stag = P_coord  # Expected value of stag
    U_rabbit = np.ones_like(P_coord)  # Guaranteed rabbit

    # Softmax goal selection
    exp_stag = np.exp(goal_temp * U_stag)
    exp_rabbit = np.exp(goal_temp * U_rabbit)
    P_choose_stag = exp_stag / (exp_stag + exp_rabbit)

    # Angles to targets
    angle_stag = angle_to_target(px, py, stag_x, stag_y)
    angle_rabbit = angle_to_target(px, py, rabbit_x, rabbit_y)

    # Mixture prediction
    z_stag = np.exp(1j * angle_stag)
    z_rabbit = np.exp(1j * angle_rabbit)
    z_mean = P_choose_stag * z_stag + (1 - P_choose_stag) * z_rabbit
    pred_angle = np.angle(z_mean)

    return cosine_log_likelihood(angles[valid], pred_angle, kappa)


# =============================================================================
# Add beliefs to trials
# =============================================================================

def add_beliefs_fast(trial: pd.DataFrame) -> pd.DataFrame:
    """Add belief columns using vectorized distance-based heuristic."""
    from belief_model_distance import BayesianIntentionModel
    model = BayesianIntentionModel(prior_stag=0.5, concentration=1.5)
    return model.run_trial(trial)


# =============================================================================
# Model Comparison
# =============================================================================

MODELS = {
    'Null': {'func': null_model_ll, 'n_params': 0},
    'Distance': {'func': distance_model_ll, 'n_params': 1},
    'Belief': {'func': belief_model_ll, 'n_params': 1},
    'Coordination': {'func': coordination_model_ll, 'n_params': 2},
    'Hierarchical': {'func': hierarchical_model_ll, 'n_params': 3},
}


def evaluate_all_models(trials: List[pd.DataFrame],
                        add_beliefs: bool = True) -> pd.DataFrame:
    """Evaluate all models on trials."""

    # Add beliefs if needed
    if add_beliefs:
        print("Adding beliefs to trials...", end=" ", flush=True)
        trials_with_beliefs = []
        for trial in trials:
            if 'p1_belief_p2_stag' not in trial.columns:
                try:
                    trial = add_beliefs_fast(trial)
                except:
                    continue
            trials_with_beliefs.append(trial)
        trials = trials_with_beliefs
        print(f"done ({len(trials)} trials)")

    results = []
    n_datapoints = sum(len(t) * 2 for t in trials)

    for name, config in MODELS.items():
        print(f"  {name}...", end=" ", flush=True)

        total_ll = 0.0
        for trial in trials:
            total_ll += config['func'](trial, 'player1')
            total_ll += config['func'](trial, 'player2')

        k = config['n_params']
        aic = 2 * k - 2 * total_ll
        bic = k * np.log(n_datapoints) - 2 * total_ll

        results.append({
            'model': name,
            'n_params': k,
            'log_likelihood': total_ll,
            'aic': aic,
            'bic': bic
        })
        print(f"LL = {total_ll:.1f}")

    return pd.DataFrame(results)


def load_all_trials(subjects: Optional[List[str]] = None) -> List[pd.DataFrame]:
    """Load all main task trials."""
    files = find_trial_files(task_type='main')
    trials = []

    for f in files:
        info = get_trial_info(f)
        subject = info.get('subject')

        if not subject:
            continue
        if subjects and subject not in subjects:
            continue

        try:
            trial = load_trial(f)
            trials.append(trial)
        except:
            continue

    return trials


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fast model comparison')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects')
    parser.add_argument('--output', help='Save results to CSV')

    args = parser.parse_args()

    print("=" * 60)
    print("FAST MODEL COMPARISON")
    print("=" * 60)

    print("\nLoading trials...")
    trials = load_all_trials(args.subjects)
    print(f"Loaded {len(trials)} trials")

    print("\nEvaluating models:")
    results = evaluate_all_models(trials)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS (sorted by AIC)")
    print("=" * 60)

    results_sorted = results.sort_values('aic')
    best_aic = results_sorted['aic'].iloc[0]

    print(f"\n{'Model':<20} {'Params':>6} {'Log-Lik':>12} {'ΔAIC':>10}")
    print("-" * 50)

    for _, row in results_sorted.iterrows():
        delta_aic = row['aic'] - best_aic
        print(f"{row['model']:<20} {row['n_params']:>6} {row['log_likelihood']:>12.1f} {delta_aic:>10.1f}")

    print("\n" + "=" * 60)
    print(f"Best model: {results_sorted.iloc[0]['model']}")
    print("=" * 60)

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
