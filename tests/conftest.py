"""
Shared pytest fixtures for decision model tests.

This module provides common setup code for all tests:
- Trial data loading (using unified data_loader)
- Model initialization
- Belief computation
"""

import pytest
import pandas as pd
import glob
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_trial, find_trial_files, load_all_trials, get_trial_info, RAW_DATA_DIR
from models.belief_model_distance import BayesianIntentionModel


@pytest.fixture
def trial_files():
    """Get list of all main task trial files."""
    files = find_trial_files(task_type='main')
    if files:
        return sorted(files)

    pytest.skip(f"No trial files found in {RAW_DATA_DIR}")


@pytest.fixture
def single_trial_file(trial_files):
    """Get a single trial file for quick tests."""
    return trial_files[0]


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


# New fixtures for accessing trials by condition

@pytest.fixture
def trials_by_opponent():
    """Get trials organized by opponent type."""
    result = {}
    for opponent in ['computer', 'same', 'diff', 'ieeg']:
        files = find_trial_files(opponent=opponent, task_type='main')
        if files:
            result[opponent] = [load_trial(f) for f in files[:12]]  # Limit to 12 per condition
    return result


@pytest.fixture
def trials_by_subject():
    """Get trials organized by subject."""
    result = {}
    for sub in ['120', '231', '233', '236', '237', '244', '255', '258']:
        files = find_trial_files(subject=sub, task_type='main')
        if files:
            result[sub] = [load_trial(f) for f in files[:12]]  # Limit to 12 per subject
    return result


@pytest.fixture
def cooperation_trials():
    """Get trials that ended in cooperation."""
    from data_loader import get_outcome
    trials = load_all_trials(task_type='main')
    return [t for t in trials if get_outcome(t)['outcome'] == 'cooperation']


@pytest.fixture
def defection_trials():
    """Get trials that ended in mutual defection."""
    from data_loader import get_outcome
    trials = load_all_trials(task_type='main')
    return [t for t in trials if get_outcome(t)['outcome'] == 'mutual_defection']
