# Stag Hunt Test Suite

This directory contains comprehensive tests for the Stag Hunt decision and belief models.

## Overview

The test suite includes:
- **`test_decision_models.py`** (22 tests): Decision model components and continuous likelihood
- **`test_belief_dynamics.py`** (11 tests): Belief updating dynamics from Trial 6 debugging
- **`test_model_comparison.py`** (8 tests): Compare distance-based vs decision-based belief models
- **`test_decision_model_debug.py`** (4 tests): Detailed debugging of decision model likelihoods
- **`test_decision_model_diagnosis.py`** (5 tests): Root cause analysis of decision model issues

**Total: 50 tests, ~1 minute runtime**

### Previous Test Files (now consolidated)
- Standalone debug scripts (debug_trial6.py, check_angles.py, etc.) → `test_belief_dynamics.py`
- test_decision_model.py, test_action_space.py, test_continuous_likelihood.py → `test_decision_models.py`

## Installation

Ensure you have pytest installed:

```bash
pip install pytest
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test class
```bash
pytest tests/test_decision_models.py::TestBeliefComparison
pytest tests/test_decision_models.py::TestActionSpace
pytest tests/test_decision_models.py::TestMotorNoise
```

### Run specific test
```bash
pytest tests/test_decision_models.py::TestBeliefComparison::test_beliefs_improve_fit
```

### Run tests matching a pattern
```bash
pytest -k "belief"          # All tests with "belief" in name
pytest -k "action_space"    # All action space tests
pytest -k "continuous"      # All continuous likelihood tests
```

### Run parametrized tests with specific values
```bash
pytest tests/test_decision_models.py::TestActionSpace -v
pytest tests/test_decision_models.py::TestMotorNoise::test_action_noise_sensitivity[1.0]
```

### Show print statements during tests
```bash
pytest -s
```

### Run tests and see detailed output
```bash
pytest -v -s
```

## Test Organization

### test_belief_dynamics.py (11 tests)

Tests belief updating dynamics using Trial 6 data, consolidating insights from debug scripts.

1. **TestBeliefUpdatesOnGoalSwitch** - Beliefs respond to goal changes
   - When players switch goals, beliefs should update accordingly
   - Tests at t=84 and t=119-122 (known switch points in Trial 6)

2. **TestAngularAlignment** - Movement patterns vs distance
   - Angular alignment (direction) more diagnostic than proximity
   - Tests cosine similarity between movement and target directions
   - Validates sustained directional movement (t=119-122)

3. **TestBeliefBounds** - Numerical stability
   - Beliefs stay within [0.01, 0.99]
   - Can reach near-boundary values (not artificially constrained)

4. **TestBeliefTrajectory** - Belief dynamics over time
   - High beliefs during cooperation phase (t=40-80)
   - Low beliefs during defection phase (t=130-200)
   - Rapid changes during transitions (t=119-122)

5. **TestDistanceVsAlignment** - Distance can be misleading
   - Multiple timesteps where closer to stag but moving toward rabbit

### test_model_comparison.py (8 tests)

Compares distance-based and decision-based belief models to identify differences.

1. **TestModelComparison** - Basic comparison tests
   - Both models complete trials without errors
   - Both respect belief bounds [0.01, 0.99]
   - Both update beliefs (deviate from prior)
   - Both show belief drops on goal switches

2. **TestModelDifferences** - Document expected differences
   - Distance model produces more extreme beliefs (avg deviation: 0.48)
   - Decision model produces moderate beliefs (avg deviation: 0.09)
   - Distance model has high variance (0.235), decision model low (0.0001)

3. **TestDecisionModelSpecific** - Decision model diagnostics
   - Verifies parameters are set correctly
   - Tests likelihood computation

**Key Finding:** Decision model beliefs barely update (variance = 0.0001) ⚠️

### test_decision_model_debug.py (4 tests)

Detailed debugging of decision model's likelihood computation at critical timesteps.

1. **test_movement_angle_computation** - Verify movement angle calculation
2. **test_target_angles** - Check angles to stag and rabbit
3. **test_decision_model_action_probabilities** - What actions does model predict?
   - If intending stag → predicts 225° (Southwest)
   - If intending rabbit → predicts 270° (South)
   - Westward movement (~180°) more likely under stag than rabbit ⚠️
4. **test_coordination_probability** - Coordination calculation
   - At t=120: coordination probability = 0.0 ⚠️

**Key Finding:** Likelihoods are backwards - model thinks westward movement favors stag!

### test_decision_model_diagnosis.py (5 tests)

Root cause analysis documenting why the decision-based model doesn't work.

1. **test_coordination_probability_too_strict**
   - Timing tolerance (0.865) far too strict
   - 100% of timesteps have coordination ≈ 0
   - Makes stag utility = 0

2. **test_zero_coordination_makes_intentions_indistinguishable**
   - When P_coord = 0, stag utility = 0
   - Model predicts "always rabbit" regardless of intention
   - Likelihood ratio: 0.44:1 (should be >2:1)

3. **test_suggested_fix_increase_timing_tolerance**
   - Recommends τ ≈ 100-150 (instead of 0.865)
   - Allows gradual coordination decay
   - With τ = 150: coord_prob = 0.008 (still low but non-zero)

4. **test_alternative_use_distance_model**
   - Distance model works well (variance = 0.235)
   - Simpler and more robust

5. **test_print_summary**
   - Comprehensive summary of findings
   - Three recommended fixes:
     1. Increase timing_tolerance to ~100-150
     2. Use distance-based model
     3. Hybrid approach

**Diagnosis:** Decision model fails because timing_tolerance is too strict, causing zero coordination probability and zero stag utility throughout most trials.

### test_decision_models.py (22 tests)

Tests decision model components and continuous likelihood evaluation.

1. **TestBasicFunctionality**
   - Test data loading and basic model operations
   - Validate trial structure and belief computation
   - Verify model initialization

2. **TestBeliefComparison**
   - Compare models with vs without beliefs
   - Verify that beliefs improve model fit
   - Key validation of the belief-based approach

3. **TestActionSpace**
   - Test action space granularity (8, 16, 32, 64 directions)
   - Verify discrete likelihood computation
   - Assess computational efficiency vs accuracy trade-offs

4. **TestContinuousLikelihood**
   - Test continuous likelihood evaluation
   - Verify convergence across action space granularities
   - Compare continuous vs discrete likelihood methods

5. **TestMotorNoise**
   - Test action noise (motor precision) parameter
   - Find optimal noise parameter
   - Sensitivity analysis

6. **TestTrialOutcomes**
   - Compare model fit across trial types
   - Analyze cooperation vs defection trials
   - Validate outcome-specific patterns

## Fixtures (conftest.py)

Shared fixtures available to all tests:

- `trial_files` - List of all trial CSV files
- `single_trial_file` - Single trial for quick tests
- `load_trial_fn` - Function to load trial data
- `belief_model` - Initialized belief model
- `trial_data` - Single trial data loaded
- `trial_with_beliefs` - Single trial with beliefs computed
- `all_trials_data` - All trials loaded
- `all_trials_with_beliefs` - All trials with beliefs

## Key Tests to Run

### Quick validation (fast)
```bash
pytest tests/test_decision_models.py::TestBasicFunctionality -v
```

### Core hypothesis test (beliefs improve fit)
```bash
pytest tests/test_decision_models.py::TestBeliefComparison::test_beliefs_improve_fit -v -s
```

### Full model comparison
```bash
pytest tests/test_decision_models.py::TestBeliefComparison -v -s
```

### Action space analysis
```bash
pytest tests/test_decision_models.py::TestActionSpace -v -s
```

### Motor noise optimization
```bash
pytest tests/test_decision_models.py::TestMotorNoise::test_optimal_action_noise -v -s
```

## Expected Results

Based on previous analyses:

- **Beliefs should improve fit** by ~200-440 log-likelihood points
- **Continuous likelihood** should converge across different action space granularities (8-64 directions)
- **Optimal motor noise** should be around κ = 1.0-2.0
- **Model should work** on both cooperation and defection trials

## Troubleshooting

### No trial files found
Ensure you're running tests from the repository root directory:
```bash
cd /path/to/StagHunt
pytest
```

### Import errors
Ensure all model files are in the repository root or on Python path:
- `belief_model_distance.py`
- `decision_model_basic.py`

### Tests taking too long
Run specific test classes or use faster parametrization:
```bash
pytest tests/test_decision_models.py::TestBasicFunctionality  # Fast
```

## Development

### Adding new tests

1. Add test methods to existing test classes or create new class
2. Use fixtures from `conftest.py` for common setup
3. Use `@pytest.mark.parametrize` for parameter sweeps
4. Follow naming convention: `test_*`

Example:
```python
class TestNewFeature:
    """Test new model feature."""

    def test_something(self, trial_with_beliefs):
        """Test description."""
        # Test code here
        assert True
```

### Adding new fixtures

Add to `tests/conftest.py`:
```python
@pytest.fixture
def my_fixture():
    """Fixture description."""
    return some_value
```

## Performance

Approximate test run times (on M1 Mac):
- Basic functionality: <1 second
- Belief comparison: ~5-10 seconds
- Action space tests: ~10-20 seconds (parametrized)
- Motor noise tests: ~10-20 seconds (parametrized)
- Full suite: ~30-60 seconds

## CI/CD Integration

To integrate with continuous integration:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest -v
```
