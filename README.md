# Stag Hunt Cooperation Task - Computational Modeling

---

## Repository Structure

```
StagHunt/
├── models/              # Core model implementations & fitting
├── analysis/            # Analysis and hypothesis testing
├── visualization/       # Video generation and plotting
├── tests/              # Pytest test suite (75 tests)
├── data/               # Data files
│   ├── trial_*.csv              # Raw trial data
│   └── enriched_trials/         # Trials with model-based regressors
│
├── stag_hunt.py        # Main entry point & fitting interface
├── fitted_params.json  # Fitted model parameters (updated by --fit)
├── CLAUDE.md           # Full modeling documentation
└── README.md           # This file
```

---

## Quick Start

```bash
# Fit model and save as defaults
python stag_hunt.py --fit integrated
```

This will:
1. Optimize parameters (learning_rate, goal_temperature, execution_temperature, timing_tolerance)
2. Save fitted values to `fitted_params.json`
3. Make them available as defaults for all model instantiations

---

## Testing

```bash
pytest tests/ -v
```
