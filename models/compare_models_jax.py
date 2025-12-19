#!/usr/bin/env python3
"""
JAX-accelerated model comparison for Stag Hunt.

Key insight: batch ALL observations into single arrays, then vmap over them.
No Python loops over trials or timesteps.
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import vonmises
from jax.scipy.special import logsumexp
from jax import jit, vmap
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import load_trial, find_trial_files, get_trial_info

# Use CPU for consistency
jax.config.update('jax_platform_name', 'cpu')

# 8 discrete action directions
ACTION_ANGLES = jnp.array([i * 2 * jnp.pi / 8 for i in range(8)])


# =============================================================================
# Core likelihood functions (operate on single observations)
# =============================================================================

def single_obs_ll_distance(obs_angle, px, py, stag_x, stag_y, rabbit_x, rabbit_y,
                           temperature, kappa):
    """Log-likelihood for single observation under distance model."""
    # Distance to each target
    dist_stag = jnp.sqrt((stag_x - px)**2 + (stag_y - py)**2)
    dist_rabbit = jnp.sqrt((rabbit_x - px)**2 + (rabbit_y - py)**2)

    # Angle to each target
    angle_stag = jnp.arctan2(stag_y - py, stag_x - px)
    angle_rabbit = jnp.arctan2(rabbit_y - py, rabbit_x - px)

    # Choose closer target
    go_stag = dist_stag < dist_rabbit
    target_angle = jnp.where(go_stag, angle_stag, angle_rabbit)

    # Utility for each of 8 actions (cosine alignment with target)
    action_utils = jnp.cos(ACTION_ANGLES - target_angle)

    # Softmax to get action probabilities
    action_probs = jax.nn.softmax(action_utils * temperature)

    # Log-likelihood: mixture of von Mises
    log_components = vonmises.logpdf(obs_angle - ACTION_ANGLES, kappa) + jnp.log(action_probs)
    return logsumexp(log_components)


def single_obs_ll_belief(obs_angle, px, py, stag_x, stag_y, rabbit_x, rabbit_y,
                         belief, temperature, kappa):
    """Log-likelihood for single observation under belief model."""
    # Angle to each target
    angle_stag = jnp.arctan2(stag_y - py, stag_x - px)
    angle_rabbit = jnp.arctan2(rabbit_y - py, rabbit_x - px)

    # Utility weighted by belief
    util_stag = jnp.cos(ACTION_ANGLES - angle_stag)
    util_rabbit = jnp.cos(ACTION_ANGLES - angle_rabbit)
    action_utils = belief * util_stag + (1 - belief) * util_rabbit

    # Softmax
    action_probs = jax.nn.softmax(action_utils * temperature)

    # Log-likelihood
    log_components = vonmises.logpdf(obs_angle - ACTION_ANGLES, kappa) + jnp.log(action_probs)
    return logsumexp(log_components)


def single_obs_ll_coordination(obs_angle, px, py, partner_x, partner_y,
                               stag_x, stag_y, rabbit_x, rabbit_y,
                               belief, temperature, kappa, timing_tol):
    """Log-likelihood for single observation under coordination model."""
    # Compute P_coord = belief × timing_alignment
    dist_player = jnp.sqrt((stag_x - px)**2 + (stag_y - py)**2)
    dist_partner = jnp.sqrt((stag_x - partner_x)**2 + (stag_y - partner_y)**2)
    time_diff = jnp.abs(dist_player - dist_partner)
    timing_align = jnp.exp(-0.5 * (time_diff / timing_tol)**2)
    P_coord = belief * timing_align

    # Angles to targets
    angle_stag = jnp.arctan2(stag_y - py, stag_x - px)
    angle_rabbit = jnp.arctan2(rabbit_y - py, rabbit_x - px)

    # Utilities weighted by P_coord
    util_stag = jnp.cos(ACTION_ANGLES - angle_stag)
    util_rabbit = jnp.cos(ACTION_ANGLES - angle_rabbit)
    action_utils = P_coord * util_stag + (1 - P_coord) * util_rabbit

    # Softmax
    action_probs = jax.nn.softmax(action_utils * temperature)

    # Log-likelihood
    log_components = vonmises.logpdf(obs_angle - ACTION_ANGLES, kappa) + jnp.log(action_probs)
    return logsumexp(log_components)


def single_obs_ll_imagined_we(obs_angle, px, py, partner_x, partner_y,
                               stag_x, stag_y, rabbit_x, rabbit_y,
                               joint_goal_belief, temperature, kappa):
    """
    Log-likelihood for single observation under Imagined We model (Tang et al.).

    Key difference from Coordination model:
    - Uses joint goal belief P(stag is OUR joint goal) instead of P(partner wants stag)
    - The joint goal directly determines action selection (no timing modulation)

    In the IW framework, agents imagine a "We" that has committed to a joint goal.
    Given this joint goal, each agent moves toward that target.
    """
    # Angles to targets
    angle_stag = jnp.arctan2(stag_y - py, stag_x - px)
    angle_rabbit = jnp.arctan2(rabbit_y - py, rabbit_x - px)

    # Under IW: action is determined by joint goal
    # If joint goal = stag, go to stag; if joint goal = rabbit, go to rabbit
    util_stag = jnp.cos(ACTION_ANGLES - angle_stag)
    util_rabbit = jnp.cos(ACTION_ANGLES - angle_rabbit)
    action_utils = joint_goal_belief * util_stag + (1 - joint_goal_belief) * util_rabbit

    # Softmax action selection
    action_probs = jax.nn.softmax(action_utils * temperature)

    # Log-likelihood
    log_components = vonmises.logpdf(obs_angle - ACTION_ANGLES, kappa) + jnp.log(action_probs)
    return logsumexp(log_components)


# =============================================================================
# Vectorized versions using vmap
# =============================================================================

# vmap over all observations at once
_vmap_distance = jit(vmap(single_obs_ll_distance,
                          in_axes=(0, 0, 0, 0, 0, 0, 0, None, None)))

_vmap_belief = jit(vmap(single_obs_ll_belief,
                        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None)))

_vmap_coordination = jit(vmap(single_obs_ll_coordination,
                              in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None)))

_vmap_imagined_we = jit(vmap(single_obs_ll_imagined_we,
                              in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None)))


@jit
def null_model_ll(n_obs):
    """Null model: uniform over circle."""
    return n_obs * jnp.log(1.0 / (2 * jnp.pi))


def distance_model_ll(data, temperature=3.0, kappa=2.0):
    """Distance model on batched data."""
    lls = _vmap_distance(
        data['obs'], data['px'], data['py'],
        data['stag_x'], data['stag_y'],
        data['rabbit_x'], data['rabbit_y'],
        temperature, kappa
    )
    return jnp.sum(lls)


def belief_model_ll(data, temperature=3.0, kappa=2.0):
    """Belief model on batched data."""
    lls = _vmap_belief(
        data['obs'], data['px'], data['py'],
        data['stag_x'], data['stag_y'],
        data['rabbit_x'], data['rabbit_y'],
        data['beliefs'],
        temperature, kappa
    )
    return jnp.sum(lls)


def coordination_model_ll(data, temperature=3.0, kappa=2.0, timing_tol=500.0):
    """Coordination model on batched data."""
    lls = _vmap_coordination(
        data['obs'], data['px'], data['py'],
        data['partner_x'], data['partner_y'],
        data['stag_x'], data['stag_y'],
        data['rabbit_x'], data['rabbit_y'],
        data['beliefs'],
        temperature, kappa, timing_tol
    )
    return jnp.sum(lls)


def imagined_we_model_ll(data, temperature=3.0, kappa=2.0):
    """
    Imagined We model on batched data.

    Uses joint goal belief P(stag is OUR joint goal) instead of
    individual partner intention belief.
    """
    lls = _vmap_imagined_we(
        data['obs'], data['px'], data['py'],
        data['partner_x'], data['partner_y'],
        data['stag_x'], data['stag_y'],
        data['rabbit_x'], data['rabbit_y'],
        data['iw_beliefs'],  # Joint goal beliefs from IW model
        temperature, kappa
    )
    return jnp.sum(lls)


# =============================================================================
# Data extraction - batch ALL trials into single arrays
# =============================================================================

def extract_all_data(trials: List[pd.DataFrame]) -> Dict[str, jnp.ndarray]:
    """Extract and concatenate data from all trials into batched arrays."""
    all_obs = []
    all_px, all_py = [], []
    all_partner_x, all_partner_y = [], []
    all_stag_x, all_stag_y = [], []
    all_rabbit_x, all_rabbit_y = [], []
    all_beliefs = []
    all_iw_beliefs = []

    for trial in trials:
        for player in ['player1', 'player2']:
            partner = 'player2' if player == 'player1' else 'player1'

            # Movement angles
            dx = np.diff(trial[f'{player}_x'].values)
            dy = np.diff(trial[f'{player}_y'].values)
            valid = (np.abs(dx) > 0.5) | (np.abs(dy) > 0.5)

            if not valid.any():
                continue

            idx = np.where(valid)[0]

            all_obs.append(np.arctan2(dy[valid], dx[valid]))
            all_px.append(trial[f'{player}_x'].values[idx])
            all_py.append(trial[f'{player}_y'].values[idx])
            all_partner_x.append(trial[f'{partner}_x'].values[idx])
            all_partner_y.append(trial[f'{partner}_y'].values[idx])
            all_stag_x.append(trial['stag_x'].values[idx + 1])
            all_stag_y.append(trial['stag_y'].values[idx + 1])
            all_rabbit_x.append(trial['rabbit_x'].values[idx + 1])
            all_rabbit_y.append(trial['rabbit_y'].values[idx + 1])

            # Standard beliefs (partner intention)
            p_num = player[-1]
            partner_num = '2' if p_num == '1' else '1'
            belief_col = f'p{p_num}_belief_p{partner_num}_stag'
            if belief_col in trial.columns:
                all_beliefs.append(trial[belief_col].values[idx])
            else:
                all_beliefs.append(np.full(len(idx), 0.5))

            # IW beliefs (joint goal)
            if 'joint_goal_stag' in trial.columns:
                all_iw_beliefs.append(trial['joint_goal_stag'].values[idx])
            else:
                all_iw_beliefs.append(np.full(len(idx), 0.5))

    # Concatenate and convert to JAX arrays
    return {
        'obs': jnp.array(np.concatenate(all_obs)),
        'px': jnp.array(np.concatenate(all_px)),
        'py': jnp.array(np.concatenate(all_py)),
        'partner_x': jnp.array(np.concatenate(all_partner_x)),
        'partner_y': jnp.array(np.concatenate(all_partner_y)),
        'stag_x': jnp.array(np.concatenate(all_stag_x)),
        'stag_y': jnp.array(np.concatenate(all_stag_y)),
        'rabbit_x': jnp.array(np.concatenate(all_rabbit_x)),
        'rabbit_y': jnp.array(np.concatenate(all_rabbit_y)),
        'beliefs': jnp.array(np.concatenate(all_beliefs)),
        'iw_beliefs': jnp.array(np.concatenate(all_iw_beliefs)),
    }


def add_beliefs_to_trials(trials: List[pd.DataFrame], prior=0.5, concentration=1.5) -> List[pd.DataFrame]:
    """Add belief columns to trials using fast JAX version (both standard and IW)."""
    from belief_model_jax import add_beliefs_batch_fast
    from belief_model_iw import add_iw_beliefs_batch

    # Add standard beliefs (partner intention)
    trials = add_beliefs_batch_fast(trials, prior=prior, concentration=concentration)

    # Add IW beliefs (joint goal)
    trials = add_iw_beliefs_batch(trials, prior=prior, concentration=concentration)

    return trials


def load_trials(subjects: Optional[List[str]] = None) -> List[pd.DataFrame]:
    """Load trials from raw data."""
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
            trials.append(load_trial(f))
        except:
            continue

    return trials


# =============================================================================
# Parameter fitting with JAX native optimizer
# =============================================================================

from jax.scipy.optimize import minimize as jax_minimize


def fit_model(model_name: str, data: Dict[str, jnp.ndarray]) -> Dict:
    """Fit model parameters using JAX's native BFGS optimizer."""

    if model_name == 'Null':
        n_obs = len(data['obs'])
        return {'params': {}, 'll': float(null_model_ll(n_obs)), 'n_params': 0}

    elif model_name == 'Distance':
        def neg_ll(params):
            # Use softplus to keep params positive
            temp = jax.nn.softplus(params[0])
            kappa = jax.nn.softplus(params[1])
            return -distance_model_ll(data, temperature=temp, kappa=kappa)

        # Initialize in unconstrained space (inverse softplus of [3.0, 2.0])
        x0 = jnp.array([jnp.log(jnp.exp(3.0) - 1), jnp.log(jnp.exp(2.0) - 1)])
        result = jax_minimize(neg_ll, x0=x0, method='BFGS')

        temp = float(jax.nn.softplus(result.x[0]))
        kappa = float(jax.nn.softplus(result.x[1]))
        return {
            'params': {'temperature': temp, 'kappa': kappa},
            'll': -float(result.fun),
            'n_params': 2
        }

    elif model_name == 'Belief':
        def neg_ll(params):
            temp = jax.nn.softplus(params[0])
            kappa = jax.nn.softplus(params[1])
            return -belief_model_ll(data, temperature=temp, kappa=kappa)

        x0 = jnp.array([jnp.log(jnp.exp(3.0) - 1), jnp.log(jnp.exp(2.0) - 1)])
        result = jax_minimize(neg_ll, x0=x0, method='BFGS')

        temp = float(jax.nn.softplus(result.x[0]))
        kappa = float(jax.nn.softplus(result.x[1]))
        return {
            'params': {'temperature': temp, 'kappa': kappa},
            'll': -float(result.fun),
            'n_params': 2
        }

    elif model_name == 'Coordination':
        def neg_ll(params):
            temp = jax.nn.softplus(params[0])
            kappa = jax.nn.softplus(params[1])
            tol = jnp.exp(params[2])  # log-scale for timing_tol
            return -coordination_model_ll(data, temperature=temp, kappa=kappa, timing_tol=tol)

        x0 = jnp.array([
            jnp.log(jnp.exp(3.0) - 1),  # softplus^-1(3)
            jnp.log(jnp.exp(2.0) - 1),  # softplus^-1(2)
            jnp.log(500.0)               # log(500)
        ])
        result = jax_minimize(neg_ll, x0=x0, method='BFGS')

        temp = float(jax.nn.softplus(result.x[0]))
        kappa = float(jax.nn.softplus(result.x[1]))
        tol = float(jnp.exp(result.x[2]))
        return {
            'params': {'temperature': temp, 'kappa': kappa, 'timing_tol': tol},
            'll': -float(result.fun),
            'n_params': 3
        }

    elif model_name == 'ImagineWe':
        def neg_ll(params):
            temp = jax.nn.softplus(params[0])
            kappa = jax.nn.softplus(params[1])
            return -imagined_we_model_ll(data, temperature=temp, kappa=kappa)

        x0 = jnp.array([jnp.log(jnp.exp(3.0) - 1), jnp.log(jnp.exp(2.0) - 1)])
        result = jax_minimize(neg_ll, x0=x0, method='BFGS')

        temp = float(jax.nn.softplus(result.x[0]))
        kappa = float(jax.nn.softplus(result.x[1]))
        return {
            'params': {'temperature': temp, 'kappa': kappa},
            'll': -float(result.fun),
            'n_params': 2
        }

    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Model comparison
# =============================================================================

MODELS = ['Null', 'Distance', 'Belief', 'Coordination', 'ImagineWe']


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+')
    parser.add_argument('--output')
    parser.add_argument('--fit', action='store_true', help='Fit model parameters')

    args = parser.parse_args()

    print("=" * 60)
    print("JAX MODEL COMPARISON (vmap vectorized)")
    print("=" * 60)

    # Load data
    print("\nLoading trials...")
    t0 = time.time()
    trials = load_trials(args.subjects)
    print(f"Loaded {len(trials)} trials ({time.time() - t0:.1f}s)")

    # Add beliefs using fast JAX version
    print("Adding beliefs (JAX batched)...", end=" ", flush=True)
    t0 = time.time()
    trials = add_beliefs_to_trials(trials)
    print(f"done ({time.time() - t0:.1f}s)")

    # Extract all data into batched arrays
    print("Extracting batched data...", end=" ", flush=True)
    t0 = time.time()
    data = extract_all_data(trials)
    n_obs = len(data['obs'])
    print(f"done ({n_obs:,} observations, {time.time() - t0:.1f}s)")

    # Warmup JIT
    print("\nWarming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = distance_model_ll(data)
    _ = belief_model_ll(data)
    _ = coordination_model_ll(data)
    _ = imagined_we_model_ll(data)
    print(f"done ({time.time() - t0:.1f}s)")

    # Fit or evaluate models
    results = []

    if args.fit:
        print("\nFitting models:")
        for name in MODELS:
            print(f"  {name}...", end=" ", flush=True)
            t0 = time.time()
            fit_result = fit_model(name, data)
            elapsed = time.time() - t0

            k = fit_result['n_params']
            ll = fit_result['ll']
            aic = 2 * k - 2 * ll
            bic = k * np.log(n_obs) - 2 * ll

            results.append({
                'model': name,
                'n_params': k,
                'log_likelihood': ll,
                'aic': aic,
                'bic': bic,
                'time': elapsed,
                'params': fit_result['params']
            })
            print(f"LL = {ll:.1f} ({elapsed:.1f}s) params={fit_result['params']}")
    else:
        print("\nEvaluating models (default params):")
        for name in MODELS:
            print(f"  {name}...", end=" ", flush=True)
            t0 = time.time()

            if name == 'Null':
                ll = float(null_model_ll(n_obs))
                k = 0
            elif name == 'Distance':
                ll = float(distance_model_ll(data))
                k = 2
            elif name == 'Belief':
                ll = float(belief_model_ll(data))
                k = 2
            elif name == 'Coordination':
                ll = float(coordination_model_ll(data))
                k = 3
            elif name == 'ImagineWe':
                ll = float(imagined_we_model_ll(data))
                k = 2

            elapsed = time.time() - t0
            aic = 2 * k - 2 * ll
            bic = k * np.log(n_obs) - 2 * ll

            results.append({
                'model': name,
                'n_params': k,
                'log_likelihood': ll,
                'aic': aic,
                'bic': bic,
                'time': elapsed
            })
            print(f"LL = {ll:.1f} ({elapsed:.3f}s)")

    # Print results
    df = pd.DataFrame(results).sort_values('aic')
    best_aic = df['aic'].iloc[0]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Model':<15} {'Params':>6} {'Log-Lik':>12} {'ΔAIC':>10} {'Time':>8}")
    print("-" * 55)

    for _, row in df.iterrows():
        delta = row['aic'] - best_aic
        print(f"{row['model']:<15} {row['n_params']:>6} {row['log_likelihood']:>12.1f} "
              f"{delta:>10.1f} {row['time']:>7.3f}s")

    print(f"\nBest model: {df.iloc[0]['model']}")

    if args.output:
        df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
