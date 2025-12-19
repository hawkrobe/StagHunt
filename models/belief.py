#!/usr/bin/env python3
"""
Belief models for Stag Hunt cooperation task.

Two models implemented:
1. Standard: Each player infers partner's individual intention
2. Imagined We (IW): Both players infer a shared joint goal

Both use JAX for fast vectorized computation with vmap across trials.

References:
- Tang et al. (2025). Proc. R. Soc. B - "Imagined We" model
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.scipy.stats import vonmises
import numpy as np
import pandas as pd
from typing import List

jax.config.update('jax_platform_name', 'cpu')


# =============================================================================
# Standard Belief Model (per-player intention inference)
# =============================================================================

@jit
def _standard_update_step(carry, inputs):
    """Single step of standard belief update."""
    belief_p1, belief_p2, prior, concentration, bounds_min, bounds_max = carry

    p1_x_prev, p1_y_prev, p1_x_curr, p1_y_curr = inputs[:4]
    p2_x_prev, p2_y_prev, p2_x_curr, p2_y_curr = inputs[4:8]
    stag_x, stag_y, rabbit_x, rabbit_y = inputs[8:12]

    max_ll = vonmises.pdf(0.0, concentration)

    # P1 observes P2's movement
    dx2, dy2 = p2_x_curr - p2_x_prev, p2_y_curr - p2_y_prev
    moved2 = (jnp.abs(dx2) > 0.5) | (jnp.abs(dy2) > 0.5)
    angle2 = jnp.arctan2(dy2, dx2)
    angle_to_stag_2 = jnp.arctan2(stag_y - p2_y_prev, stag_x - p2_x_prev)
    angle_to_rabbit_2 = jnp.arctan2(rabbit_y - p2_y_prev, rabbit_x - p2_x_prev)

    ll_stag_2 = jnp.clip(vonmises.pdf(angle2 - angle_to_stag_2, concentration) / max_ll, 0.1, 0.9)
    ll_rabbit_2 = jnp.clip(vonmises.pdf(angle2 - angle_to_rabbit_2, concentration) / max_ll, 0.1, 0.9)

    num1 = ll_stag_2 * belief_p1
    denom1 = num1 + ll_rabbit_2 * (1 - belief_p1)
    new_belief_p1 = jnp.where(moved2 & (denom1 > 0), num1 / denom1, belief_p1)
    new_belief_p1 = jnp.clip(new_belief_p1, bounds_min, bounds_max)

    # P2 observes P1's movement
    dx1, dy1 = p1_x_curr - p1_x_prev, p1_y_curr - p1_y_prev
    moved1 = (jnp.abs(dx1) > 0.5) | (jnp.abs(dy1) > 0.5)
    angle1 = jnp.arctan2(dy1, dx1)
    angle_to_stag_1 = jnp.arctan2(stag_y - p1_y_prev, stag_x - p1_x_prev)
    angle_to_rabbit_1 = jnp.arctan2(rabbit_y - p1_y_prev, rabbit_x - p1_x_prev)

    ll_stag_1 = jnp.clip(vonmises.pdf(angle1 - angle_to_stag_1, concentration) / max_ll, 0.1, 0.9)
    ll_rabbit_1 = jnp.clip(vonmises.pdf(angle1 - angle_to_rabbit_1, concentration) / max_ll, 0.1, 0.9)

    num2 = ll_stag_1 * belief_p2
    denom2 = num2 + ll_rabbit_1 * (1 - belief_p2)
    new_belief_p2 = jnp.where(moved1 & (denom2 > 0), num2 / denom2, belief_p2)
    new_belief_p2 = jnp.clip(new_belief_p2, bounds_min, bounds_max)

    return (new_belief_p1, new_belief_p2, prior, concentration, bounds_min, bounds_max), (new_belief_p1, new_belief_p2)


# =============================================================================
# Imagined We Model (joint goal inference)
# =============================================================================

@jit
def _iw_update_step(carry, inputs):
    """Single step of Imagined We belief update."""
    joint_belief, prior, concentration, bounds_min, bounds_max = carry

    p1_x_prev, p1_y_prev, p1_x_curr, p1_y_curr = inputs[:4]
    p2_x_prev, p2_y_prev, p2_x_curr, p2_y_curr = inputs[4:8]
    stag_x, stag_y, rabbit_x, rabbit_y = inputs[8:12]

    max_ll = vonmises.pdf(0.0, concentration)

    # Player 1's movement
    dx1, dy1 = p1_x_curr - p1_x_prev, p1_y_curr - p1_y_prev
    moved1 = (jnp.abs(dx1) > 0.5) | (jnp.abs(dy1) > 0.5)
    angle1 = jnp.arctan2(dy1, dx1)
    ll_p1_stag = jnp.clip(vonmises.pdf(angle1 - jnp.arctan2(stag_y - p1_y_prev, stag_x - p1_x_prev), concentration) / max_ll, 0.1, 0.9)
    ll_p1_rabbit = jnp.clip(vonmises.pdf(angle1 - jnp.arctan2(rabbit_y - p1_y_prev, rabbit_x - p1_x_prev), concentration) / max_ll, 0.1, 0.9)

    # Player 2's movement
    dx2, dy2 = p2_x_curr - p2_x_prev, p2_y_curr - p2_y_prev
    moved2 = (jnp.abs(dx2) > 0.5) | (jnp.abs(dy2) > 0.5)
    angle2 = jnp.arctan2(dy2, dx2)
    ll_p2_stag = jnp.clip(vonmises.pdf(angle2 - jnp.arctan2(stag_y - p2_y_prev, stag_x - p2_x_prev), concentration) / max_ll, 0.1, 0.9)
    ll_p2_rabbit = jnp.clip(vonmises.pdf(angle2 - jnp.arctan2(rabbit_y - p2_y_prev, rabbit_x - p2_x_prev), concentration) / max_ll, 0.1, 0.9)

    # Joint likelihood: both players' movements
    ll_joint_stag = ll_p1_stag * ll_p2_stag
    ll_joint_rabbit = ll_p1_rabbit * ll_p2_rabbit

    # Bayes update
    num = ll_joint_stag * joint_belief
    denom = num + ll_joint_rabbit * (1 - joint_belief)
    either_moved = moved1 | moved2
    new_belief = jnp.where(either_moved & (denom > 0), num / denom, joint_belief)
    new_belief = jnp.clip(new_belief, bounds_min, bounds_max)

    return (new_belief, prior, concentration, bounds_min, bounds_max), new_belief


# =============================================================================
# Batched computation (vmap across trials)
# =============================================================================

def _run_standard_padded(inputs, prior, concentration, bounds_min, bounds_max, valid_mask):
    init = (prior, prior, prior, concentration, bounds_min, bounds_max)
    _, (b1, b2) = lax.scan(_standard_update_step, init, inputs)
    return jnp.concatenate([jnp.array([prior]), b1]), jnp.concatenate([jnp.array([prior]), b2])


def _run_iw_padded(inputs, prior, concentration, bounds_min, bounds_max, valid_mask):
    init = (prior, prior, concentration, bounds_min, bounds_max)
    _, beliefs = lax.scan(_iw_update_step, init, inputs)
    return jnp.concatenate([jnp.array([prior]), beliefs])


_vmap_standard = jit(vmap(_run_standard_padded, in_axes=(0, None, None, None, None, 0)))
_vmap_iw = jit(vmap(_run_iw_padded, in_axes=(0, None, None, None, None, 0)))


def _prepare_batch(trials: List[pd.DataFrame]):
    """Prepare batched inputs from list of trial DataFrames."""
    lengths = [len(t) for t in trials]
    max_len = max(lengths)
    n_trials = len(trials)

    batch = np.zeros((n_trials, max_len - 1, 12))
    masks = np.zeros((n_trials, max_len))

    for i, trial in enumerate(trials):
        n = len(trial)
        masks[i, :n] = 1.0
        batch[i, :n-1, 0] = trial['player1_x'].values[:-1]
        batch[i, :n-1, 1] = trial['player1_y'].values[:-1]
        batch[i, :n-1, 2] = trial['player1_x'].values[1:]
        batch[i, :n-1, 3] = trial['player1_y'].values[1:]
        batch[i, :n-1, 4] = trial['player2_x'].values[:-1]
        batch[i, :n-1, 5] = trial['player2_y'].values[:-1]
        batch[i, :n-1, 6] = trial['player2_x'].values[1:]
        batch[i, :n-1, 7] = trial['player2_y'].values[1:]
        batch[i, :n-1, 8] = trial['stag_x'].values[1:]
        batch[i, :n-1, 9] = trial['stag_y'].values[1:]
        batch[i, :n-1, 10] = trial['rabbit_x'].values[1:]
        batch[i, :n-1, 11] = trial['rabbit_y'].values[1:]

    return jnp.array(batch), jnp.array(masks), lengths


# =============================================================================
# Public API
# =============================================================================

def add_standard_beliefs(trials: List[pd.DataFrame], prior=0.5, concentration=1.5,
                         bounds=(0.01, 0.99)) -> List[pd.DataFrame]:
    """Add per-player belief columns to trials (batched)."""
    batch, masks, lengths = _prepare_batch(trials)
    b1, b2 = _vmap_standard(batch, prior, concentration, bounds[0], bounds[1], masks)

    results = []
    for i, trial in enumerate(trials):
        result = trial.copy()
        result['p1_belief_p2_stag'] = np.array(b1[i, :lengths[i]])
        result['p2_belief_p1_stag'] = np.array(b2[i, :lengths[i]])
        results.append(result)
    return results


def add_iw_beliefs(trials: List[pd.DataFrame], prior=0.5, concentration=1.5,
                   bounds=(0.01, 0.99)) -> List[pd.DataFrame]:
    """Add joint goal belief column to trials (batched)."""
    batch, masks, lengths = _prepare_batch(trials)
    beliefs = _vmap_iw(batch, prior, concentration, bounds[0], bounds[1], masks)

    results = []
    for i, trial in enumerate(trials):
        result = trial.copy()
        result['joint_goal_stag'] = np.array(beliefs[i, :lengths[i]])
        results.append(result)
    return results


# Aliases for backwards compatibility
add_beliefs_batch_fast = add_standard_beliefs
add_iw_beliefs_batch = add_iw_beliefs


if __name__ == '__main__':
    import sys
    import time
    sys.path.insert(0, '..')
    from data_loader import load_trial, find_trial_files

    files = find_trial_files(task_type='main')
    trials = [load_trial(f) for f in files]
    print(f"Loaded {len(trials)} trials")

    # Warmup
    _ = add_iw_beliefs(trials[:10])
    _ = add_standard_beliefs(trials[:10])

    t0 = time.time()
    results = add_iw_beliefs(trials)
    print(f"IW: {time.time() - t0:.2f}s")

    t0 = time.time()
    results = add_standard_beliefs(trials)
    print(f"Standard: {time.time() - t0:.2f}s")
