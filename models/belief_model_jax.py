#!/usr/bin/env python3
"""
Fast JAX-based belief computation.

Uses jax.lax.scan for sequential updates and vmap to parallelize across trials.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.scipy.stats import vonmises
import numpy as np
import pandas as pd
from typing import List

jax.config.update('jax_platform_name', 'cpu')


@jit
def belief_update_step(carry, inputs):
    """Single step of belief update (for use with scan)."""
    belief_p1, belief_p2, prior, concentration, bounds_min, bounds_max = carry

    # Unpack inputs
    p1_x_prev, p1_y_prev, p1_x_curr, p1_y_curr = inputs[:4]
    p2_x_prev, p2_y_prev, p2_x_curr, p2_y_curr = inputs[4:8]
    stag_x, stag_y, rabbit_x, rabbit_y = inputs[8:12]

    # --- P1 observes P2's movement ---
    dx2, dy2 = p2_x_curr - p2_x_prev, p2_y_curr - p2_y_prev
    moved2 = (jnp.abs(dx2) > 0.5) | (jnp.abs(dy2) > 0.5)
    movement_angle_2 = jnp.arctan2(dy2, dx2)

    angle_to_stag_2 = jnp.arctan2(stag_y - p2_y_prev, stag_x - p2_x_prev)
    angle_to_rabbit_2 = jnp.arctan2(rabbit_y - p2_y_prev, rabbit_x - p2_x_prev)

    # von Mises likelihoods
    max_ll = vonmises.pdf(0.0, concentration)
    ll_stag_2 = vonmises.pdf(movement_angle_2 - angle_to_stag_2, concentration) / max_ll
    ll_rabbit_2 = vonmises.pdf(movement_angle_2 - angle_to_rabbit_2, concentration) / max_ll

    # Clip likelihoods
    ll_stag_2 = jnp.clip(ll_stag_2, 0.1, 0.9)
    ll_rabbit_2 = jnp.clip(ll_rabbit_2, 0.1, 0.9)

    # Bayes update for P1's belief about P2
    num1 = ll_stag_2 * belief_p1
    denom1 = num1 + ll_rabbit_2 * (1 - belief_p1)
    new_belief_p1 = jnp.where(moved2 & (denom1 > 0), num1 / denom1, belief_p1)
    new_belief_p1 = jnp.clip(new_belief_p1, bounds_min, bounds_max)

    # --- P2 observes P1's movement ---
    dx1, dy1 = p1_x_curr - p1_x_prev, p1_y_curr - p1_y_prev
    moved1 = (jnp.abs(dx1) > 0.5) | (jnp.abs(dy1) > 0.5)
    movement_angle_1 = jnp.arctan2(dy1, dx1)

    angle_to_stag_1 = jnp.arctan2(stag_y - p1_y_prev, stag_x - p1_x_prev)
    angle_to_rabbit_1 = jnp.arctan2(rabbit_y - p1_y_prev, rabbit_x - p1_x_prev)

    ll_stag_1 = vonmises.pdf(movement_angle_1 - angle_to_stag_1, concentration) / max_ll
    ll_rabbit_1 = vonmises.pdf(movement_angle_1 - angle_to_rabbit_1, concentration) / max_ll

    ll_stag_1 = jnp.clip(ll_stag_1, 0.1, 0.9)
    ll_rabbit_1 = jnp.clip(ll_rabbit_1, 0.1, 0.9)

    # Bayes update for P2's belief about P1
    num2 = ll_stag_1 * belief_p2
    denom2 = num2 + ll_rabbit_1 * (1 - belief_p2)
    new_belief_p2 = jnp.where(moved1 & (denom2 > 0), num2 / denom2, belief_p2)
    new_belief_p2 = jnp.clip(new_belief_p2, bounds_min, bounds_max)

    new_carry = (new_belief_p1, new_belief_p2, prior, concentration, bounds_min, bounds_max)
    outputs = (new_belief_p1, new_belief_p2)

    return new_carry, outputs


def run_belief_trial_jax(p1_x, p1_y, p2_x, p2_y, stag_x, stag_y, rabbit_x, rabbit_y,
                         prior=0.5, concentration=1.5, bounds=(0.01, 0.99)):
    """Run belief model on a single trial using JAX scan."""
    n = len(p1_x)

    # Stack inputs for scan
    inputs = jnp.stack([
        p1_x[:-1], p1_y[:-1], p1_x[1:], p1_y[1:],
        p2_x[:-1], p2_y[:-1], p2_x[1:], p2_y[1:],
        stag_x[1:], stag_y[1:], rabbit_x[1:], rabbit_y[1:]
    ], axis=1)

    # Initial carry
    init_carry = (prior, prior, prior, concentration, bounds[0], bounds[1])

    # Run scan
    _, (beliefs_p1, beliefs_p2) = lax.scan(belief_update_step, init_carry, inputs)

    # Prepend initial beliefs
    beliefs_p1 = jnp.concatenate([jnp.array([prior]), beliefs_p1])
    beliefs_p2 = jnp.concatenate([jnp.array([prior]), beliefs_p2])

    return beliefs_p1, beliefs_p2


def add_beliefs_jax(trial: pd.DataFrame, prior=0.5, concentration=1.5) -> pd.DataFrame:
    """Add belief columns to a trial using JAX."""
    result = trial.copy()

    p1_x = jnp.array(trial['player1_x'].values)
    p1_y = jnp.array(trial['player1_y'].values)
    p2_x = jnp.array(trial['player2_x'].values)
    p2_y = jnp.array(trial['player2_y'].values)
    stag_x = jnp.array(trial['stag_x'].values)
    stag_y = jnp.array(trial['stag_y'].values)
    rabbit_x = jnp.array(trial['rabbit_x'].values)
    rabbit_y = jnp.array(trial['rabbit_y'].values)

    beliefs_p1, beliefs_p2 = run_belief_trial_jax(
        p1_x, p1_y, p2_x, p2_y, stag_x, stag_y, rabbit_x, rabbit_y,
        prior=prior, concentration=concentration
    )

    result['p1_belief_p2_stag'] = np.array(beliefs_p1)
    result['p2_belief_p1_stag'] = np.array(beliefs_p2)

    return result


def add_beliefs_batch(trials: List[pd.DataFrame], prior=0.5, concentration=1.5) -> List[pd.DataFrame]:
    """Add beliefs to all trials."""
    return [add_beliefs_jax(t, prior, concentration) for t in trials]


# =============================================================================
# Batched version - vmap across trials
# =============================================================================

def _run_belief_trial_padded(inputs, prior, concentration, bounds_min, bounds_max, valid_mask):
    """Run belief on a padded trial, respecting valid_mask."""
    init_carry = (prior, prior, prior, concentration, bounds_min, bounds_max)
    _, (beliefs_p1, beliefs_p2) = lax.scan(belief_update_step, init_carry, inputs)
    beliefs_p1 = jnp.concatenate([jnp.array([prior]), beliefs_p1])
    beliefs_p2 = jnp.concatenate([jnp.array([prior]), beliefs_p2])
    return beliefs_p1, beliefs_p2


# vmap over batch dimension
_vmap_belief_trials = jit(vmap(
    _run_belief_trial_padded,
    in_axes=(0, None, None, None, None, 0)
))


def add_beliefs_batch_fast(trials: List[pd.DataFrame], prior=0.5, concentration=1.5,
                           bounds=(0.01, 0.99)) -> List[pd.DataFrame]:
    """Add beliefs to all trials using batched vmap."""
    # Find max length and pad
    lengths = [len(t) for t in trials]
    max_len = max(lengths)

    # Prepare batched input arrays
    n_trials = len(trials)
    batch_inputs = np.zeros((n_trials, max_len - 1, 12))
    valid_masks = np.zeros((n_trials, max_len))

    for i, trial in enumerate(trials):
        n = len(trial)
        valid_masks[i, :n] = 1.0

        p1_x = trial['player1_x'].values
        p1_y = trial['player1_y'].values
        p2_x = trial['player2_x'].values
        p2_y = trial['player2_y'].values
        stag_x = trial['stag_x'].values
        stag_y = trial['stag_y'].values
        rabbit_x = trial['rabbit_x'].values
        rabbit_y = trial['rabbit_y'].values

        batch_inputs[i, :n-1, 0] = p1_x[:-1]
        batch_inputs[i, :n-1, 1] = p1_y[:-1]
        batch_inputs[i, :n-1, 2] = p1_x[1:]
        batch_inputs[i, :n-1, 3] = p1_y[1:]
        batch_inputs[i, :n-1, 4] = p2_x[:-1]
        batch_inputs[i, :n-1, 5] = p2_y[:-1]
        batch_inputs[i, :n-1, 6] = p2_x[1:]
        batch_inputs[i, :n-1, 7] = p2_y[1:]
        batch_inputs[i, :n-1, 8] = stag_x[1:]
        batch_inputs[i, :n-1, 9] = stag_y[1:]
        batch_inputs[i, :n-1, 10] = rabbit_x[1:]
        batch_inputs[i, :n-1, 11] = rabbit_y[1:]

    # Run batched
    batch_inputs_jax = jnp.array(batch_inputs)
    valid_masks_jax = jnp.array(valid_masks)

    beliefs_p1_batch, beliefs_p2_batch = _vmap_belief_trials(
        batch_inputs_jax, prior, concentration, bounds[0], bounds[1], valid_masks_jax
    )

    # Unpack results
    results = []
    for i, trial in enumerate(trials):
        n = lengths[i]
        result = trial.copy()
        result['p1_belief_p2_stag'] = np.array(beliefs_p1_batch[i, :n])
        result['p2_belief_p1_stag'] = np.array(beliefs_p2_batch[i, :n])
        results.append(result)

    return results


if __name__ == '__main__':
    import sys
    import time
    sys.path.insert(0, '..')
    from data_loader import load_trial, find_trial_files

    print("Testing JAX belief model...")

    # Load trials
    files = find_trial_files(task_type='main')
    trials = [load_trial(f) for f in files]
    print(f"Loaded {len(trials)} trials")

    # Warmup with small batch
    _ = add_beliefs_batch_fast(trials[:10])

    # Time batched version
    t0 = time.time()
    results = add_beliefs_batch_fast(trials)
    elapsed = time.time() - t0

    print(f"Batched: {len(trials)} trials in {elapsed:.2f}s ({elapsed/len(trials)*1000:.1f}ms/trial)")
