"""
Shared pytest fixtures for decision model tests.

This module provides common setup code for all tests:
- Trial data loading
- Model initialization
- Belief computation
"""

import pytest
import pandas as pd
import glob
from pathlib import Path
from models.belief_model_distance import BayesianIntentionModel


@pytest.fixture
def trial_files():
    """Get list of all trial CSV files."""
    files = sorted(glob.glob('inputs/stag_hunt_coop_trial*.csv'))
    if not files:
        pytest.skip("No trial files found in inputs/ directory")
    return files


@pytest.fixture
def single_trial_file(trial_files):
    """Get a single trial file for quick tests."""
    return trial_files[0]


def load_trial(filepath):
    """Load and clean a single trial CSV."""
    df = pd.read_csv(filepath)

    # Fix the typo in column name if present
    if 'plater1_y' in df.columns:
        df = df.rename(columns={'plater1_y': 'player1_y'})

    # Remove rows with NaN values (malformed last row in CSVs)
    df = df.dropna()

    return df


@pytest.fixture
def load_trial_fn():
    """Return the trial loading function."""
    return load_trial


@pytest.fixture
def belief_model():
    """Initialize standard belief model."""
    return BayesianIntentionModel(
        prior_stag=0.5,
        concentration=1.5,
        belief_bounds=(0.01, 0.99)
    )


@pytest.fixture
def trial_data(single_trial_file, load_trial_fn):
    """Load a single trial's data."""
    return load_trial_fn(single_trial_file)


@pytest.fixture
def trial_with_beliefs(trial_data, belief_model):
    """Load trial data with beliefs computed."""
    return belief_model.run_trial(trial_data)


@pytest.fixture
def all_trials_data(trial_files, load_trial_fn):
    """Load all trials data."""
    return [load_trial_fn(f) for f in trial_files]


@pytest.fixture
def all_trials_with_beliefs(all_trials_data, belief_model):
    """Load all trials with beliefs computed."""
    return [belief_model.run_trial(trial) for trial in all_trials_data]
