# Decision Model Test Suite

This directory contains a unified test suite for the Stag Hunt decision models.

## Overview

The test suite consolidates all previous standalone test scripts into a cohesive pytest-based framework with proper organization, fixtures, and parameterization.

### Previous Test Files (now archived)
- `test_decision_model.py` → Unified into test suite
- `test_action_space.py` → Unified into test suite
- `test_continuous_likelihood.py` → Unified into test suite

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

### Test Classes

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
