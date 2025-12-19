#!/usr/bin/env python3
"""
Unified Model Comparison Framework for Stag Hunt

Compares multiple computational models on trajectory prediction:
1. Null (random baseline)
2. Distance-only (move toward closer target)
3. Belief-based (Bayesian belief updates)
4. Hierarchical (goal selection + plan execution)
5. Cross-trial learning (evolving priors)

Outputs:
- Log-likelihood per model
- AIC/BIC for model selection
- Per-subject breakdown
- Cross-validation results

Usage:
    python models/compare_models.py
    python models/compare_models.py --subjects 120 258
    python models/compare_models.py --cv  # Leave-one-subject-out CV
"""

import numpy as np
import pandas as pd
from scipy.stats import vonmises
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_trial, find_trial_files, get_trial_info, get_outcome


# =============================================================================
# Abstract Base Class
# =============================================================================

class TrajectoryModel(ABC):
    """Base class for trajectory prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for display."""
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of free parameters."""
        pass

    @abstractmethod
    def predict_action_probs(self, state: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict action distribution given current state.

        Returns:
            angles: Array of possible movement angles
            probs: Probability of each angle
        """
        pass

    def compute_log_likelihood(self, trial_data: pd.DataFrame, player: str,
                               subsample: int = 1) -> float:
        """
        Compute log-likelihood of observed trajectory.

        Args:
            trial_data: Trial dataframe
            player: 'player1' or 'player2'
            subsample: Only evaluate every Nth timestep (1 = all, 10 = every 10th)
        """
        partner = 'player2' if player == 'player1' else 'player1'

        # Compute all movements at once
        dx = trial_data[f'{player}_x'].diff().values[1:]
        dy = trial_data[f'{player}_y'].diff().values[1:]

        # Find steps with actual movement
        movement_mask = (np.abs(dx) >= 0.5) | (np.abs(dy) >= 0.5)
        if not movement_mask.any():
            return 0.0

        # Get indices of moving timesteps
        moving_indices = np.where(movement_mask)[0]

        # Subsample if requested
        if subsample > 1:
            moving_indices = moving_indices[::subsample]

        if len(moving_indices) == 0:
            return 0.0

        # Observed angles
        observed_angles = np.arctan2(dy[moving_indices], dx[moving_indices])

        # Get state arrays
        prev_indices = moving_indices
        player_x = trial_data[f'{player}_x'].values[prev_indices]
        player_y = trial_data[f'{player}_y'].values[prev_indices]
        partner_x = trial_data[f'{partner}_x'].values[prev_indices]
        partner_y = trial_data[f'{partner}_y'].values[prev_indices]
        stag_x = trial_data['stag_x'].values[prev_indices + 1]
        stag_y = trial_data['stag_y'].values[prev_indices + 1]
        rabbit_x = trial_data['rabbit_x'].values[prev_indices + 1]
        rabbit_y = trial_data['rabbit_y'].values[prev_indices + 1]

        # Beliefs
        belief_col = f'p{player[-1]}' + '_belief_p' + f'{"2" if player[-1]=="1" else "1"}' + '_stag'
        if belief_col in trial_data.columns:
            beliefs = trial_data[belief_col].values[prev_indices]
        else:
            beliefs = np.full(len(prev_indices), 0.5)

        # Precompute angles array once
        angles = self.predict_action_probs({'player_x': 0, 'player_y': 0,
                                            'partner_x': 0, 'partner_y': 0,
                                            'stag_x': 1, 'stag_y': 0,
                                            'rabbit_x': -1, 'rabbit_y': 0,
                                            'belief': 0.5})[0]

        # Compute log-likelihood
        total_ll = 0.0
        kappa = 2.0

        for i in range(len(observed_angles)):
            state = {
                'player_x': player_x[i], 'player_y': player_y[i],
                'partner_x': partner_x[i], 'partner_y': partner_y[i],
                'stag_x': stag_x[i], 'stag_y': stag_y[i],
                'rabbit_x': rabbit_x[i], 'rabbit_y': rabbit_y[i],
                'belief': beliefs[i]
            }

            _, probs = self.predict_action_probs(state)

            # Von Mises mixture likelihood (vectorized)
            likelihood = np.sum(probs * vonmises.pdf(observed_angles[i], kappa, loc=angles))
            total_ll += np.log(max(likelihood, 1e-10))

        # Scale up if subsampled
        if subsample > 1:
            total_ll *= subsample

        return total_ll


# =============================================================================
# Model Implementations
# =============================================================================

class NullModel(TrajectoryModel):
    """Baseline: uniform random movement."""

    def __init__(self, n_directions: int = 8):
        self.angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

    @property
    def name(self) -> str:
        return "Null (Random)"

    @property
    def n_params(self) -> int:
        return 0

    def predict_action_probs(self, state: Dict) -> Tuple[np.ndarray, np.ndarray]:
        probs = np.ones(len(self.angles)) / len(self.angles)
        return self.angles, probs


class DistanceModel(TrajectoryModel):
    """Move toward closer target (no belief reasoning)."""

    def __init__(self, temperature: float = 3.0, n_directions: int = 8):
        self.temperature = temperature
        self.angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

    @property
    def name(self) -> str:
        return "Distance-Only"

    @property
    def n_params(self) -> int:
        return 1  # temperature

    def predict_action_probs(self, state: Dict) -> Tuple[np.ndarray, np.ndarray]:
        # Distance to each target
        dist_stag = np.sqrt((state['stag_x'] - state['player_x'])**2 +
                           (state['stag_y'] - state['player_y'])**2)
        dist_rabbit = np.sqrt((state['rabbit_x'] - state['player_x'])**2 +
                             (state['rabbit_y'] - state['player_y'])**2)

        # Choose closer target
        if dist_stag < dist_rabbit:
            target_x, target_y = state['stag_x'], state['stag_y']
        else:
            target_x, target_y = state['rabbit_x'], state['rabbit_y']

        # Angle to target
        angle_to_target = np.arctan2(target_y - state['player_y'],
                                     target_x - state['player_x'])

        # Softmax over alignment
        utilities = np.cos(self.angles - angle_to_target)
        exp_utils = np.exp(self.temperature * utilities)
        probs = exp_utils / exp_utils.sum()

        return self.angles, probs


class BeliefModel(TrajectoryModel):
    """Use beliefs about partner to decide between targets."""

    def __init__(self, temperature: float = 3.0, n_directions: int = 8):
        self.temperature = temperature
        self.angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

    @property
    def name(self) -> str:
        return "Belief-Based"

    @property
    def n_params(self) -> int:
        return 1  # temperature

    def predict_action_probs(self, state: Dict) -> Tuple[np.ndarray, np.ndarray]:
        belief = state.get('belief', 0.5)

        # Angles to targets
        angle_stag = np.arctan2(state['stag_y'] - state['player_y'],
                                state['stag_x'] - state['player_x'])
        angle_rabbit = np.arctan2(state['rabbit_y'] - state['player_y'],
                                  state['rabbit_x'] - state['player_x'])

        # Utility = weighted combination based on belief
        utilities = np.zeros(len(self.angles))
        for i, angle in enumerate(self.angles):
            util_stag = np.cos(angle - angle_stag)
            util_rabbit = np.cos(angle - angle_rabbit)
            utilities[i] = belief * util_stag + (1 - belief) * util_rabbit

        exp_utils = np.exp(self.temperature * utilities)
        probs = exp_utils / exp_utils.sum()

        return self.angles, probs


class CoordinationModel(TrajectoryModel):
    """Belief-based with coordination probability (P_coord = belief × timing)."""

    def __init__(self, temperature: float = 3.0, timing_tolerance: float = 150.0,
                 n_directions: int = 8):
        self.temperature = temperature
        self.timing_tolerance = timing_tolerance
        self.angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

    @property
    def name(self) -> str:
        return "Coordination (P_coord)"

    @property
    def n_params(self) -> int:
        return 2  # temperature, timing_tolerance

    def predict_action_probs(self, state: Dict) -> Tuple[np.ndarray, np.ndarray]:
        belief = state.get('belief', 0.5)

        # Compute P_coord = belief × timing_alignment
        dist_player = np.sqrt((state['stag_x'] - state['player_x'])**2 +
                             (state['stag_y'] - state['player_y'])**2)
        dist_partner = np.sqrt((state['stag_x'] - state['partner_x'])**2 +
                              (state['stag_y'] - state['partner_y'])**2)

        time_diff = abs(dist_player - dist_partner)
        timing_align = np.exp(-0.5 * (time_diff / self.timing_tolerance)**2)
        P_coord = belief * timing_align

        # Angles to targets
        angle_stag = np.arctan2(state['stag_y'] - state['player_y'],
                                state['stag_x'] - state['player_x'])
        angle_rabbit = np.arctan2(state['rabbit_y'] - state['player_y'],
                                  state['rabbit_x'] - state['player_x'])

        # Expected utility under coordination
        utilities = np.zeros(len(self.angles))
        for i, angle in enumerate(self.angles):
            util_stag = np.cos(angle - angle_stag)
            util_rabbit = np.cos(angle - angle_rabbit)
            # Stag only worth pursuing if coordination likely
            utilities[i] = P_coord * util_stag + (1 - P_coord) * util_rabbit

        exp_utils = np.exp(self.temperature * utilities)
        probs = exp_utils / exp_utils.sum()

        return self.angles, probs


class HierarchicalModel(TrajectoryModel):
    """Two-level: goal selection (soft) + plan execution (hard)."""

    def __init__(self, goal_temp: float = 2.0, exec_temp: float = 10.0,
                 timing_tolerance: float = 150.0, n_directions: int = 8):
        self.goal_temp = goal_temp
        self.exec_temp = exec_temp
        self.timing_tolerance = timing_tolerance
        self.angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

    @property
    def name(self) -> str:
        return "Hierarchical (Goal+Plan)"

    @property
    def n_params(self) -> int:
        return 3  # goal_temp, exec_temp, timing_tolerance

    def predict_action_probs(self, state: Dict) -> Tuple[np.ndarray, np.ndarray]:
        belief = state.get('belief', 0.5)

        # Level 1: Goal selection
        # P_coord for stag utility
        dist_player = np.sqrt((state['stag_x'] - state['player_x'])**2 +
                             (state['stag_y'] - state['player_y'])**2)
        dist_partner = np.sqrt((state['stag_x'] - state['partner_x'])**2 +
                              (state['stag_y'] - state['partner_y'])**2)
        time_diff = abs(dist_player - dist_partner)
        timing_align = np.exp(-0.5 * (time_diff / self.timing_tolerance)**2)
        P_coord = belief * timing_align

        U_stag = P_coord  # Expected value of stag
        U_rabbit = 1.0    # Guaranteed rabbit value

        # Softmax goal selection
        goal_utils = np.array([U_stag, U_rabbit])
        exp_goal = np.exp(self.goal_temp * goal_utils)
        P_stag = exp_goal[0] / exp_goal.sum()
        P_rabbit = 1 - P_stag

        # Level 2: Plan execution
        angle_stag = np.arctan2(state['stag_y'] - state['player_y'],
                                state['stag_x'] - state['player_x'])
        angle_rabbit = np.arctan2(state['rabbit_y'] - state['player_y'],
                                  state['rabbit_x'] - state['player_x'])

        # Action distribution if going for stag
        utils_stag = np.cos(self.angles - angle_stag)
        exp_stag = np.exp(self.exec_temp * utils_stag)
        probs_stag = exp_stag / exp_stag.sum()

        # Action distribution if going for rabbit
        utils_rabbit = np.cos(self.angles - angle_rabbit)
        exp_rabbit = np.exp(self.exec_temp * utils_rabbit)
        probs_rabbit = exp_rabbit / exp_rabbit.sum()

        # Mixture
        probs = P_stag * probs_stag + P_rabbit * probs_rabbit

        return self.angles, probs


# =============================================================================
# Model Comparison
# =============================================================================

def add_beliefs_to_trial(trial_data: pd.DataFrame) -> pd.DataFrame:
    """Add belief columns using distance-based model."""
    from belief_model_distance import BayesianIntentionModel

    model = BayesianIntentionModel(
        prior_stag=0.5,
        concentration=1.5,
        belief_bounds=(0.01, 0.99)
    )
    return model.run_trial(trial_data)


def evaluate_model_on_trials(model: TrajectoryModel,
                             trials: List[pd.DataFrame],
                             add_beliefs: bool = True,
                             subsample: int = 1) -> Dict:
    """Evaluate a model on a list of trials."""
    total_ll = 0.0
    n_trials = 0
    n_datapoints = 0
    trial_lls = []

    for trial in trials:
        # Add beliefs if needed
        if add_beliefs and 'p1_belief_p2_stag' not in trial.columns:
            try:
                trial = add_beliefs_to_trial(trial)
            except:
                continue

        try:
            ll_p1 = model.compute_log_likelihood(trial, 'player1', subsample=subsample)
            ll_p2 = model.compute_log_likelihood(trial, 'player2', subsample=subsample)
            trial_ll = ll_p1 + ll_p2

            total_ll += trial_ll
            trial_lls.append(trial_ll)
            n_trials += 1
            n_datapoints += len(trial) * 2
        except Exception as e:
            continue

    # Compute AIC/BIC
    k = model.n_params
    aic = 2 * k - 2 * total_ll
    bic = k * np.log(n_datapoints) - 2 * total_ll if n_datapoints > 0 else np.inf

    return {
        'model': model.name,
        'n_params': k,
        'log_likelihood': total_ll,
        'mean_ll_per_trial': total_ll / n_trials if n_trials > 0 else 0,
        'n_trials': n_trials,
        'n_datapoints': n_datapoints,
        'aic': aic,
        'bic': bic,
        'trial_lls': trial_lls
    }


def load_trials_by_subject(subjects: Optional[List[str]] = None) -> Dict[str, List[pd.DataFrame]]:
    """Load trials organized by subject."""
    trials_by_subject = defaultdict(list)

    files = find_trial_files(task_type='main')

    for f in files:
        info = get_trial_info(f)
        subject = info.get('subject')

        # Skip if no subject or filtered out
        if not subject:
            continue
        if subjects and subject not in subjects:
            continue

        try:
            trial = load_trial(f)
            trials_by_subject[subject].append(trial)
        except:
            continue

    return dict(trials_by_subject)


def run_comparison(subjects: Optional[List[str]] = None,
                   verbose: bool = True,
                   subsample: int = 1) -> pd.DataFrame:
    """Run full model comparison."""

    # Initialize models
    models = [
        NullModel(),
        DistanceModel(temperature=3.0),
        BeliefModel(temperature=3.0),
        CoordinationModel(temperature=3.0, timing_tolerance=150.0),
        HierarchicalModel(goal_temp=2.0, exec_temp=10.0, timing_tolerance=150.0),
    ]

    # Load data
    if verbose:
        print("=" * 70)
        print("MODEL COMPARISON ON STAG HUNT DATA")
        print("=" * 70)
        if subsample > 1:
            print(f"(Fast mode: evaluating every {subsample}th timestep)")
        print("\nLoading trials...")

    trials_by_subject = load_trials_by_subject(subjects)
    all_trials = [t for trials in trials_by_subject.values() for t in trials]

    if verbose:
        print(f"Loaded {len(all_trials)} trials from {len(trials_by_subject)} subjects")
        print(f"Subjects: {sorted(trials_by_subject.keys())}")

    # Evaluate each model
    results = []

    if verbose:
        print("\n" + "-" * 70)
        print("Evaluating models...")
        print("-" * 70)

    for model in models:
        if verbose:
            print(f"\n  {model.name}...", end=" ", flush=True)

        result = evaluate_model_on_trials(model, all_trials, subsample=subsample)
        results.append(result)

        if verbose:
            print(f"LL = {result['log_likelihood']:.1f}")

    # Create results dataframe
    df = pd.DataFrame(results)

    # Add delta columns (relative to null)
    null_ll = df[df['model'] == 'Null (Random)']['log_likelihood'].values[0]
    df['delta_ll'] = df['log_likelihood'] - null_ll

    null_aic = df[df['model'] == 'Null (Random)']['aic'].values[0]
    df['delta_aic'] = df['aic'] - null_aic

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print("\n{:<25} {:>8} {:>12} {:>10} {:>10}".format(
            "Model", "Params", "Log-Lik", "ΔAIC", "ΔBIC"))
        print("-" * 70)

        # Sort by AIC
        df_sorted = df.sort_values('aic')
        best_aic = df_sorted['aic'].iloc[0]
        best_bic = df_sorted['bic'].iloc[0]

        for _, row in df_sorted.iterrows():
            delta_aic = row['aic'] - best_aic
            delta_bic = row['bic'] - best_bic
            print("{:<25} {:>8} {:>12.1f} {:>10.1f} {:>10.1f}".format(
                row['model'], row['n_params'], row['log_likelihood'],
                delta_aic, delta_bic))

        print("\n" + "=" * 70)
        winner = df_sorted.iloc[0]['model']
        print(f"Best model by AIC: {winner}")
        print("=" * 70)

    return df


def run_cross_validation(verbose: bool = True) -> pd.DataFrame:
    """Leave-one-subject-out cross-validation."""

    models = [
        NullModel(),
        DistanceModel(temperature=3.0),
        BeliefModel(temperature=3.0),
        CoordinationModel(temperature=3.0, timing_tolerance=150.0),
        HierarchicalModel(goal_temp=2.0, exec_temp=10.0, timing_tolerance=150.0),
    ]

    if verbose:
        print("=" * 70)
        print("LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION")
        print("=" * 70)

    # Load all data
    trials_by_subject = load_trials_by_subject()
    subjects = sorted(trials_by_subject.keys())

    if verbose:
        print(f"\nSubjects: {subjects}")
        print(f"Total trials: {sum(len(t) for t in trials_by_subject.values())}")

    # Track results per fold
    cv_results = {model.name: [] for model in models}

    for held_out in subjects:
        if verbose:
            print(f"\n  Fold: hold out {held_out}...", end=" ", flush=True)

        # Test trials = held out subject
        test_trials = trials_by_subject[held_out]

        for model in models:
            result = evaluate_model_on_trials(model, test_trials)
            cv_results[model.name].append(result['log_likelihood'])

        if verbose:
            print("done")

    # Aggregate
    summary = []
    for model in models:
        lls = cv_results[model.name]
        summary.append({
            'model': model.name,
            'n_params': model.n_params,
            'mean_cv_ll': np.mean(lls),
            'std_cv_ll': np.std(lls),
            'total_cv_ll': np.sum(lls)
        })

    df = pd.DataFrame(summary)

    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 70)

        print("\n{:<25} {:>12} {:>12} {:>12}".format(
            "Model", "Mean CV LL", "Std", "Total CV LL"))
        print("-" * 70)

        df_sorted = df.sort_values('total_cv_ll', ascending=False)
        for _, row in df_sorted.iterrows():
            print("{:<25} {:>12.1f} {:>12.1f} {:>12.1f}".format(
                row['model'], row['mean_cv_ll'], row['std_cv_ll'], row['total_cv_ll']))

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare trajectory models')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to analyze')
    parser.add_argument('--cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--fast', action='store_true', help='Fast mode (subsample timesteps)')
    parser.add_argument('--subsample', type=int, default=1, help='Evaluate every Nth timestep')
    parser.add_argument('--output', help='Save results to CSV')

    args = parser.parse_args()

    subsample = 10 if args.fast else args.subsample

    if args.cv:
        results = run_cross_validation()
    else:
        results = run_comparison(subjects=args.subjects, subsample=subsample)

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
