"""
Stag Hunt Cooperation Task - Unified API

Quick Start:
------------
```python
from stag_hunt import load_trial, load_all_trials, add_beliefs

# Load data
trial = load_trial('data/raw/sub-120/.../trial.tsv')
trials = load_all_trials(subject='120', opponent='ieeg')

# Add beliefs using Imagined We model (best fit)
trials_with_beliefs = add_beliefs(trials)

# Or run full model comparison
from models.compare_models import main
# python models/compare_models.py --fit
```

Models:
-------
- Imagined We (IW): Joint goal inference (best model, ΔAIC=136k over alternatives)
- Standard: Per-player intention inference
- See models/compare_models.py for full comparison

Data Loading:
-------------
- load_trial(filepath): Load a single trial
- load_all_trials(...): Load multiple trials with filters
- find_trial_files(...): Find trial files
- get_outcome(trial): Get cooperation/defection outcome
"""

# Re-export data loading utilities
from data_loader import (
    load_trial,
    load_all_trials,
    find_trial_files,
    get_trial_info,
    get_outcome,
    summarize_data,
    save_derivative,
    save_all_derivatives,
    RAW_DATA_DIR,
    DERIVATIVES_DIR
)

# Re-export belief models
from models.belief_model_iw import add_iw_beliefs_batch as add_beliefs
from models.belief_model_jax import add_beliefs_batch_fast as add_standard_beliefs


__all__ = [
    # Data loading
    'load_trial',
    'load_all_trials',
    'find_trial_files',
    'get_trial_info',
    'get_outcome',
    'summarize_data',
    'save_derivative',
    'save_all_derivatives',
    'RAW_DATA_DIR',
    'DERIVATIVES_DIR',
    # Belief models
    'add_beliefs',
    'add_standard_beliefs',
]


if __name__ == '__main__':
    print("Stag Hunt Unified API")
    print("=" * 60)
    print("\nUsage:")
    print("  from stag_hunt import load_trial, add_beliefs")
    print("  trials = load_all_trials(subject='120')")
    print("  trials = add_beliefs(trials)  # IW model")
    print("\nModel comparison:")
    print("  python models/compare_models.py --fit")
    print("=" * 60)
