# Model Fitting

Unified command-line interface for fitting all Stag Hunt computational models.

## Quick Start

```bash
# List available models
python models/fit.py --list

# Fit a model
python models/fit.py --model integrated

# Save results
python models/fit.py --model hierarchical --output results.json

# Use different optimization method
python models/fit.py --model distance --method nelder-mead
```

## Available Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| `integrated` | learning_rate, goal_temperature, execution_temperature, timing_tolerance | Main model with cross-trial learning |
| `hierarchical` | goal_temperature, execution_temperature | Hierarchical decisions without cross-trial learning |
| `distance` | temperature | Simple distance-based baseline |
| `distance_tiebreak` | temperature, tiebreak_prob | Distance model with tie-breaking |

## Usage

### Basic Fitting

```bash
python models/fit.py --model MODEL_NAME
```

This will:
1. Load all trial data
2. Try multiple parameter initializations
3. Report best fit with AIC/BIC
4. Show optimal parameters

### Options

```
--model MODEL        Model to fit (required)
--method METHOD      Optimization method (default: L-BFGS-B)
--output FILE        Save results to JSON file
--quiet              Suppress output
--list               List available models
```

### Optimization Methods

- `L-BFGS-B` (default) - Bounded quasi-Newton
- `Nelder-Mead` - Simplex method
- `Powell` - Powell's method
- `TNC` - Truncated Newton
- `SLSQP` - Sequential Least Squares

## Examples

### Fit integrated model
```bash
python models/fit.py --model integrated
```

Output:
```
======================================================================
FITTING MODEL: INTEGRATED
======================================================================
Description: Integrated hierarchical model with cross-trial learning
Parameters: learning_rate, goal_temperature, execution_temperature, timing_tolerance

Loaded 12 trials
Trying 3 initializations...

  Init 1: learning_rate=0.200, goal_temperature=2.000, execution_temperature=12.000, timing_tolerance=150.000
    → LL = -5430.12

  Init 2: learning_rate=0.300, goal_temperature=2.400, execution_temperature=15.000, timing_tolerance=150.000
    → LL = -5425.83

  Init 3: learning_rate=0.500, goal_temperature=2.000, execution_temperature=12.000, timing_tolerance=150.000
    → LL = -5432.45

======================================================================
RESULTS
======================================================================

Optimal parameters:
  learning_rate             = 0.298765
  goal_temperature          = 2.387654
  execution_temperature     = 14.876543
  timing_tolerance          = 149.123456

Model fit:
  Log-likelihood:  -5425.83
  AIC:             10859.66
  BIC:             10889.23
  Success:         True
```

### Save results to file
```bash
python models/fit.py --model distance --output distance_fit.json
```

Creates `distance_fit.json`:
```json
{
  "model": "distance",
  "parameters": {
    "temperature": 2.134567
  },
  "log_likelihood": -5430.04,
  "aic": 10862.08,
  "bic": 10869.47,
  "n_parameters": 1,
  "n_data_points": 3820,
  "success": true
}
```

### Try different optimization method
```bash
python models/fit.py --model hierarchical --method nelder-mead
```

## Model Details

### integrated
**Integrated hierarchical model with cross-trial learning**

Combines:
- Cross-trial expectation learning (learning_rate)
- Within-trial Bayesian belief updating
- Hierarchical goal + plan decisions

**Parameters:**
- `learning_rate` (0.01-0.99): How much to update expectations after each trial
- `goal_temperature` (0.1-10.0): Strategic goal selection softness
- `execution_temperature` (0.1-20.0): Motor execution precision
- `timing_tolerance` (10.0-300.0): Coordination timing window

**Initial guesses:** 3 combinations tested

**Usage:**
```bash
python models/fit.py --model integrated
```

### hierarchical
**Hierarchical goal + plan model**

Two-level decision model without cross-trial learning. Requires precomputed beliefs.

**Parameters:**
- `goal_temperature` (0.1-10.0): Strategic goal selection
- `execution_temperature` (0.1-20.0): Motor execution precision

**Initial guesses:** 2 combinations tested

**Usage:**
```bash
python models/fit.py --model hierarchical
```

### distance
**Distance-based planning**

Simple baseline: move toward nearest target.

**Parameters:**
- `temperature` (0.1-10.0): Action selection softness

**Initial guesses:** 3 values tested

**Usage:**
```bash
python models/fit.py --model distance
```

### distance_tiebreak
**Distance with probabilistic tie-breaking**

Distance model with explicit tie-breaking when targets are equidistant.

**Parameters:**
- `temperature` (0.1-10.0): Action selection softness
- `tiebreak_prob` (0.0-1.0): P(choose stag | equidistant)

**Initial guesses:** 2 combinations tested

**Usage:**
```bash
python models/fit.py --model distance_tiebreak
```

## Output Format

### Terminal Output
- Model description
- Number of trials loaded
- Fitting progress for each initialization
- Best parameters found
- Log-likelihood, AIC, BIC
- Optimization success status

### JSON Output (--output FILE)
```json
{
  "model": "model_name",
  "parameters": {
    "param1": value,
    "param2": value
  },
  "log_likelihood": -1234.56,
  "aic": 2478.12,
  "bic": 2501.34,
  "n_parameters": 4,
  "n_data_points": 3820,
  "success": true
}
```

## Model Comparison

To compare models, fit each one and compare AIC/BIC:

```bash
# Fit each model
python models/fit.py --model distance --output distance.json
python models/fit.py --model hierarchical --output hierarchical.json
python models/fit.py --model integrated --output integrated.json

# Compare AICs (lower is better)
```

**Rule of thumb:**
- ΔAIC < 2: Models essentially equivalent
- ΔAIC 2-7: Moderate evidence for better model
- ΔAIC > 10: Strong evidence for better model

## Adding New Models

To add a new model, edit `models/fit.py`:

1. Add model configuration to `MODELS` dict:
```python
MODELS['my_model'] = {
    'description': 'My custom model',
    'params': ['param1', 'param2'],
    'bounds': [(0.0, 1.0), (0.1, 10.0)],
    'initial': [[0.5, 2.0], [0.3, 3.0]],
    'requires_beliefs': False
}
```

2. Add objective function in `get_objective_function()`:
```python
elif model_name == 'my_model':
    from models.my_model import MyModel

    def objective(params):
        param1, param2 = params
        # ... compute likelihood
        return -total_ll
```

3. Test:
```bash
python models/fit.py --model my_model
```

## Troubleshooting

### ImportError

Ensure you're running from the repository root:
```bash
cd /path/to/StagHunt
python models/fit.py --model integrated
```

### Optimization fails

Try a different method:
```bash
python models/fit.py --model MODEL --method nelder-mead
```

### Very slow

Some models (especially `integrated`) can take 5-10 minutes. Use `--quiet` for cleaner output.

## Performance Tips

1. **Start with simple models** - Try `distance` first to verify data loads
2. **Use L-BFGS-B for speed** - Generally fastest for bounded optimization
3. **Save results** - Use `--output` to avoid re-running
4. **Multiple initializations** - Script automatically tries several starting points

## Citation

If you use these fitting tools, please cite:

```bibtex
@software{stagHunt2025,
  title={Stag Hunt Cooperation Task: Bayesian Theory of Mind},
  author={Hawkins, Robert D.},
  year={2025},
  institution={Stanford University}
}
```
