#!/usr/bin/env python3
"""
Imagined We (IW) belief model from Tang et al. (2025).

Key insight: Each agent infers a JOINT goal for "We" rather than
inferring the partner's individual intention.

The joint goal is updated using Bayesian inference based on BOTH
players' movements, not just the partner's.

References:
- Tang et al. (2025). Proc. R. Soc. B
- Tang et al. (2020). "Imagined We" CogSci proceedings
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
def iw_belief_update_step(carry, inputs):
    """
    Single step of Imagined We belief update (for use with scan).

    Key difference from standard belief model:
    - Updates based on BOTH players' movements (joint action)
    - Both agents infer the same joint goal (up to observation noise)
    """
    joint_goal_belief, prior, concentration, bounds_min, bounds_max = carry

    # Unpack inputs for both players
    p1_x_prev, p1_y_prev, p1_x_curr, p1_y_curr = inputs[:4]
    p2_x_prev, p2_y_prev, p2_x_curr, p2_y_curr = inputs[4:8]
    stag_x, stag_y, rabbit_x, rabbit_y = inputs[8:12]

    # --- Player 1's movement ---
    dx1, dy1 = p1_x_curr - p1_x_prev, p1_y_curr - p1_y_prev
    moved1 = (jnp.abs(dx1) > 0.5) | (jnp.abs(dy1) > 0.5)
    movement_angle_1 = jnp.arctan2(dy1, dx1)

    angle_to_stag_1 = jnp.arctan2(stag_y - p1_y_prev, stag_x - p1_x_prev)
    angle_to_rabbit_1 = jnp.arctan2(rabbit_y - p1_y_prev, rabbit_x - p1_x_prev)

    # --- Player 2's movement ---
    dx2, dy2 = p2_x_curr - p2_x_prev, p2_y_curr - p2_y_prev
    moved2 = (jnp.abs(dx2) > 0.5) | (jnp.abs(dy2) > 0.5)
    movement_angle_2 = jnp.arctan2(dy2, dx2)

    angle_to_stag_2 = jnp.arctan2(stag_y - p2_y_prev, stag_x - p2_x_prev)
    angle_to_rabbit_2 = jnp.arctan2(rabbit_y - p2_y_prev, rabbit_x - p2_x_prev)

    # von Mises likelihoods (normalize by max)
    max_ll = vonmises.pdf(0.0, concentration)

    # P1's movement consistency with each joint goal
    ll_p1_stag = vonmises.pdf(movement_angle_1 - angle_to_stag_1, concentration) / max_ll
    ll_p1_rabbit = vonmises.pdf(movement_angle_1 - angle_to_rabbit_1, concentration) / max_ll

    # P2's movement consistency with each joint goal
    ll_p2_stag = vonmises.pdf(movement_angle_2 - angle_to_stag_2, concentration) / max_ll
    ll_p2_rabbit = vonmises.pdf(movement_angle_2 - angle_to_rabbit_2, concentration) / max_ll

    # Clip individual likelihoods
    ll_p1_stag = jnp.clip(ll_p1_stag, 0.1, 0.9)
    ll_p1_rabbit = jnp.clip(ll_p1_rabbit, 0.1, 0.9)
    ll_p2_stag = jnp.clip(ll_p2_stag, 0.1, 0.9)
    ll_p2_rabbit = jnp.clip(ll_p2_rabbit, 0.1, 0.9)

    # Joint action likelihood: P(both actions | joint goal)
    # If joint goal = stag, both should move toward stag
    # If joint goal = rabbit, both should move toward rabbit
    ll_joint_stag = ll_p1_stag * ll_p2_stag
    ll_joint_rabbit = ll_p1_rabbit * ll_p2_rabbit

    # Bayes update for joint goal belief
    # P(JG=stag | both actions) ∝ P(both actions | JG=stag) × P(JG=stag)
    num = ll_joint_stag * joint_goal_belief
    denom = num + ll_joint_rabbit * (1 - joint_goal_belief)

    # Only update if at least one player moved
    either_moved = moved1 | moved2
    new_belief = jnp.where(either_moved & (denom > 0), num / denom, joint_goal_belief)
    new_belief = jnp.clip(new_belief, bounds_min, bounds_max)

    new_carry = (new_belief, prior, concentration, bounds_min, bounds_max)
    return new_carry, new_belief


def run_iw_belief_trial(p1_x, p1_y, p2_x, p2_y, stag_x, stag_y, rabbit_x, rabbit_y,
                        prior=0.5, concentration=1.5, bounds=(0.01, 0.99)):
    """Run IW belief model on a single trial using JAX scan."""
    n = len(p1_x)

    # Stack inputs for scan
    inputs = jnp.stack([
        p1_x[:-1], p1_y[:-1], p1_x[1:], p1_y[1:],
        p2_x[:-1], p2_y[:-1], p2_x[1:], p2_y[1:],
        stag_x[1:], stag_y[1:], rabbit_x[1:], rabbit_y[1:]
    ], axis=1)

    # Initial carry
    init_carry = (prior, prior, concentration, bounds[0], bounds[1])

    # Run scan
    _, joint_goal_beliefs = lax.scan(iw_belief_update_step, init_carry, inputs)

    # Prepend initial belief
    joint_goal_beliefs = jnp.concatenate([jnp.array([prior]), joint_goal_beliefs])

    return joint_goal_beliefs


def add_iw_beliefs(trial: pd.DataFrame, prior=0.5, concentration=1.5) -> pd.DataFrame:
    """Add IW joint goal belief column to a trial."""
    result = trial.copy()

    p1_x = jnp.array(trial['player1_x'].values)
    p1_y = jnp.array(trial['player1_y'].values)
    p2_x = jnp.array(trial['player2_x'].values)
    p2_y = jnp.array(trial['player2_y'].values)
    stag_x = jnp.array(trial['stag_x'].values)
    stag_y = jnp.array(trial['stag_y'].values)
    rabbit_x = jnp.array(trial['rabbit_x'].values)
    rabbit_y = jnp.array(trial['rabbit_y'].values)

    joint_goal_beliefs = run_iw_belief_trial(
        p1_x, p1_y, p2_x, p2_y, stag_x, stag_y, rabbit_x, rabbit_y,
        prior=prior, concentration=concentration
    )

    # Both players share the same joint goal belief in IW
    result['joint_goal_stag'] = np.array(joint_goal_beliefs)

    return result


# =============================================================================
# Batched version - vmap across trials
# =============================================================================

def _run_iw_trial_padded(inputs, prior, concentration, bounds_min, bounds_max, valid_mask):
    """Run IW belief on a padded trial."""
    init_carry = (prior, prior, concentration, bounds_min, bounds_max)
    _, beliefs = lax.scan(iw_belief_update_step, init_carry, inputs)
    beliefs = jnp.concatenate([jnp.array([prior]), beliefs])
    return beliefs


# vmap over batch dimension
_vmap_iw_trials = jit(vmap(
    _run_iw_trial_padded,
    in_axes=(0, None, None, None, None, 0)
))


def add_iw_beliefs_batch(trials: List[pd.DataFrame], prior=0.5, concentration=1.5,
                         bounds=(0.01, 0.99)) -> List[pd.DataFrame]:
    """Add IW beliefs to all trials using batched vmap."""
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

    beliefs_batch = _vmap_iw_trials(
        batch_inputs_jax, prior, concentration, bounds[0], bounds[1], valid_masks_jax
    )

    # Unpack results
    results = []
    for i, trial in enumerate(trials):
        n = lengths[i]
        result = trial.copy()
        result['joint_goal_stag'] = np.array(beliefs_batch[i, :n])
        results.append(result)

    return results


if __name__ == '__main__':
    import sys
    import time
    sys.path.insert(0, '..')
    from data_loader import load_trial, find_trial_files

    print("Testing IW belief model...")

    # Load trials
    files = find_trial_files(task_type='main')
    trials = [load_trial(f) for f in files]
    print(f"Loaded {len(trials)} trials")

    # Warmup with small batch
    _ = add_iw_beliefs_batch(trials[:10])

    # Time batched version
    t0 = time.time()
    results = add_iw_beliefs_batch(trials)
    elapsed = time.time() - t0

    print(f"Batched: {len(trials)} trials in {elapsed:.2f}s ({elapsed/len(trials)*1000:.1f}ms/trial)")

    # Show sample result
    sample = results[0]
    print(f"\nSample trial beliefs (first 10 rows):")
    print(sample[['joint_goal_stag']].head(10))
