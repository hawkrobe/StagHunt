# Stag Hunt Cooperation Task - Computational Modeling

Bayesian belief inference models for analyzing cooperation in a Stag Hunt coordination game with iEEG patients.

---

## Data Setup

**Data is not included in this repository.** To use this codebase:

1. Obtain the data files from the research team
2. Unzip the data into the `data/raw/` directory:

```bash
mkdir -p data/raw
# Unzip the data archive into data/raw/
unzip Archive.zip -d data/raw/
```

The expected structure after setup:
```
data/
  raw/
    sub-120/
      ses-01/
        ieeg/
          sub-120_task-cooperation_run-01_opponent-...tsv
          ...
    sub-231/
    sub-233/
    ...
```

3. Generate derivative files with belief regressors:
```bash
python generate_derivatives.py
```

This creates `data/derivatives/` with processed trial files.

---

## Repository Structure

```
StagHunt/
├── models/                  # Core model implementations
│   ├── belief_model_distance.py    # Distance-based belief inference
│   ├── belief_model_decision.py    # Decision-based inference
│   ├── decision_model_*.py         # Decision models
│   └── fit.py                      # Parameter fitting
├── visualization/           # Visualization tools
│   ├── make_video.py               # Animated trajectory videos
│   ├── make_gallery.py             # Static gallery figures
│   └── make_gallery_v2.py          # Improved gallery (recommended)
├── tests/                   # Pytest test suite
├── data/                    # Data directory (not in repo)
│   ├── raw/                        # Original trial files
│   └── derivatives/                # Processed files with beliefs
│
├── stag_hunt.py             # Main API (models + data loading)
├── data_loader.py           # Unified data loader
├── generate_derivatives.py  # Batch processing script
├── fitted_params.json       # Fitted model parameters
├── CLAUDE.md                # Full documentation
└── README.md                # This file
```

---

## Quick Start

```python
from stag_hunt import load_trial, load_all_trials, get_outcome
from models.belief_model_distance import BayesianIntentionModel

# Load a single trial
trial = load_trial('data/raw/sub-120/ses-01/ieeg/sub-120_task-...tsv')

# Load all trials with filters
trials = load_all_trials(opponent='ieeg', reward='stagdecrease')

# Run belief model
model = BayesianIntentionModel(prior_stag=0.5, concentration=1.5)
trial_with_beliefs = model.run_trial(trial)

# Check outcome
outcome = get_outcome(trial_with_beliefs)
print(outcome['outcome'])  # 'cooperation' or 'mutual_defection' etc.
```

---

## Visualizations

```bash
# Summary statistics figure
python visualization/make_gallery.py --summary

# Comparison figures (recommended)
python visualization/make_gallery_v2.py

# Animated video (slow - use filters)
python visualization/make_video.py --subject 120 --opponent ieeg --max-trials 12
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Key Findings (703 trials, 8 subjects)

- **42.8% cooperation** overall
- **Strong opponent effect:** computer 19% → same 54% → diff 49% → ieeg 58%
- **No reward effect:** 43% vs 44%
- **Belief dynamics predict outcomes:** cooperation trials show rising beliefs, defection shows crashing beliefs

See `CLAUDE.md` for full documentation.
