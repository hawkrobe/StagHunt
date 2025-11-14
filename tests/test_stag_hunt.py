"""
Tests for the unified stag_hunt.py interface.

This test suite covers the unified DecisionModel and BeliefModel interfaces,
including parameter loading from fitted_params.json.
"""

import pytest
import json
import os
from stag_hunt import (
    DecisionModel,
    BeliefModel,
    get_recommended_models,
    load_fitted_params
)


class TestLoadFittedParams:
    """Test parameter loading from fitted_params.json."""

    def test_load_fitted_params_success(self):
        """Test successful loading of fitted parameters."""
        params = load_fitted_params()
        assert params is not None
        assert isinstance(params, dict)

    def test_fitted_params_has_required_models(self):
        """Test that fitted_params.json contains expected models."""
        params = load_fitted_params()
        assert 'coordinated' in params
        assert 'basic' in params
        assert 'integrated' in params

    def test_fitted_params_structure(self):
        """Test that fitted params have the expected structure."""
        params = load_fitted_params()

        # Check coordinated model structure
        assert 'temperature' in params['coordinated']
        assert 'timing_tolerance' in params['coordinated']
        assert 'action_noise' in params['coordinated']
        assert 'n_directions' in params['coordinated']
        assert 'description' in params['coordinated']
        assert 'fit_info' in params['coordinated']


class TestDecisionModel:
    """Test the unified DecisionModel interface."""

    def test_coordinated_with_fitted_params(self):
        """Test coordinated model loads from fitted_params.json."""
        model = DecisionModel(model_type='coordinated')

        assert model.model_type == 'coordinated'
        assert model.params is not None
        assert 'temperature' in model.params
        assert 'timing_tolerance' in model.params
        assert 'action_noise' in model.params
        assert model.params['temperature'] > 0

    def test_coordinated_with_custom_params(self):
        """Test coordinated model with explicit custom parameters."""
        custom_params = {
            'temperature': 5.0,
            'timing_tolerance': 100.0,
            'action_noise': 2.0,
            'n_directions': 16
        }
        model = DecisionModel(model_type='coordinated', params=custom_params)

        assert model.model_type == 'coordinated'
        assert model.params == custom_params
        assert model.params['temperature'] == 5.0
        assert model.params['timing_tolerance'] == 100.0

    def test_basic_with_fitted_params(self):
        """Test basic model loads from fitted_params.json."""
        model = DecisionModel(model_type='basic')

        assert model.model_type == 'basic'
        assert model.params is not None
        assert 'temperature' in model.params
        assert 'w_stag' in model.params
        assert 'w_rabbit' in model.params
        assert 'action_noise' in model.params

    def test_basic_with_custom_params(self):
        """Test basic model with explicit custom parameters."""
        custom_params = {
            'temperature': 5.0,
            'w_stag': 1.0,
            'w_rabbit': 2.0,
            'action_noise': 1.0,
            'n_directions': 8
        }
        model = DecisionModel(model_type='basic', params=custom_params)

        assert model.model_type == 'basic'
        assert model.params == custom_params
        assert model.params['w_stag'] == 1.0
        assert model.params['w_rabbit'] == 2.0

    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError, match="No fitted parameters found"):
            DecisionModel(model_type='invalid')

    def test_repr(self):
        """Test string representation of DecisionModel."""
        model = DecisionModel(
            model_type='coordinated',
            params={'temperature': 3.0, 'timing_tolerance': 100.0,
                   'action_noise': 5.0, 'n_directions': 8}
        )
        repr_str = repr(model)
        assert 'DecisionModel' in repr_str
        assert 'coordinated' in repr_str

    def test_model_has_underlying_methods(self):
        """Test that DecisionModel delegates to underlying model."""
        model = DecisionModel(
            model_type='coordinated',
            params={'temperature': 3.0, 'timing_tolerance': 100.0,
                   'action_noise': 5.0, 'n_directions': 8}
        )
        # Should have methods from underlying model
        assert hasattr(model, 'compute_action_probabilities')
        assert hasattr(model, 'n_directions')


class TestBeliefModel:
    """Test the unified BeliefModel interface."""

    def test_decision_inference_with_default_model(self):
        """Test decision-based inference with default decision model."""
        belief_model = BeliefModel(inference_type='decision')

        assert belief_model.inference_type == 'decision'
        assert belief_model.model is not None

    def test_decision_inference_with_custom_decision_model(self):
        """Test decision-based inference with custom decision model."""
        decision_model = DecisionModel(
            model_type='coordinated',
            params={'temperature': 5.0, 'timing_tolerance': 100.0,
                   'action_noise': 2.0, 'n_directions': 8}
        )
        belief_model = BeliefModel(
            inference_type='decision',
            decision_model=decision_model
        )

        assert belief_model.inference_type == 'decision'
        assert belief_model.model is not None

    def test_decision_inference_with_params_dict(self):
        """Test decision-based inference with params dict."""
        belief_model = BeliefModel(
            inference_type='decision',
            decision_model={
                'model_type': 'coordinated',
                'params': {'temperature': 5.0, 'timing_tolerance': 100.0,
                          'action_noise': 2.0, 'n_directions': 8}
            }
        )

        assert belief_model.inference_type == 'decision'
        assert belief_model.model is not None

    def test_distance_inference(self):
        """Test distance-based inference."""
        belief_model = BeliefModel(
            inference_type='distance',
            concentration=2.0
        )

        assert belief_model.inference_type == 'distance'
        assert belief_model.model is not None

    def test_custom_prior_and_bounds(self):
        """Test custom prior and belief bounds."""
        belief_model = BeliefModel(
            inference_type='distance',
            prior_stag=0.7,
            belief_bounds=(0.05, 0.95)
        )

        assert belief_model.model is not None

    def test_invalid_inference_type(self):
        """Test that invalid inference type raises error."""
        with pytest.raises(ValueError, match="Unknown inference_type"):
            BeliefModel(inference_type='invalid')

    def test_repr(self):
        """Test string representation of BeliefModel."""
        model = BeliefModel(inference_type='distance')
        repr_str = repr(model)
        assert 'BeliefModel' in repr_str
        assert 'distance' in repr_str

    def test_has_run_trial_method(self):
        """Test that BeliefModel has run_trial method."""
        model = BeliefModel(inference_type='distance')
        assert hasattr(model, 'run_trial')
        assert callable(model.run_trial)


class TestGetRecommendedModels:
    """Test the get_recommended_models() convenience function."""

    def test_returns_both_models(self):
        """Test that function returns both decision and belief models."""
        decision_model, belief_model = get_recommended_models()

        assert decision_model is not None
        assert belief_model is not None
        assert isinstance(decision_model, DecisionModel)
        assert isinstance(belief_model, BeliefModel)

    def test_decision_model_is_coordinated(self):
        """Test that decision model uses coordinated type."""
        decision_model, _ = get_recommended_models()
        assert decision_model.model_type == 'coordinated'

    def test_belief_model_is_decision_based(self):
        """Test that belief model uses decision-based inference."""
        _, belief_model = get_recommended_models()
        assert belief_model.inference_type == 'decision'


class TestErrorHandling:
    """Test error handling when fitted params are missing."""

    def test_missing_model_in_fitted_params(self, tmp_path, monkeypatch):
        """Test error when model not in fitted_params.json."""
        # Create a fitted_params.json without 'coordinated'
        config_file = tmp_path / "fitted_params.json"
        config_file.write_text(json.dumps({'basic': {'temperature': 1.0}}))

        # Monkey patch to use our test config
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="No fitted parameters found"):
            DecisionModel(model_type='coordinated')

    def test_missing_fitted_params_file(self, tmp_path, monkeypatch):
        """Test error when fitted_params.json doesn't exist."""
        # Change to directory without fitted_params.json
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="No fitted parameters found"):
            DecisionModel(model_type='coordinated')


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self):
        """Test complete workflow: create models, they should work together."""
        # Get recommended models
        decision_model, belief_model = get_recommended_models()

        # Verify they're properly configured
        assert decision_model.model_type == 'coordinated'
        assert belief_model.inference_type == 'decision'

        # Verify decision model has expected attributes
        assert hasattr(decision_model, 'temperature')
        assert hasattr(decision_model, 'timing_tolerance')

        # Verify belief model has run_trial method
        assert hasattr(belief_model, 'run_trial')

    def test_custom_workflow(self):
        """Test workflow with custom parameters."""
        # Create custom decision model
        decision_model = DecisionModel(
            model_type='basic',
            params={
                'temperature': 5.0,
                'w_stag': 1.0,
                'w_rabbit': 2.0,
                'action_noise': 1.0,
                'n_directions': 8
            }
        )

        # Create distance-based belief model
        belief_model = BeliefModel(
            inference_type='distance',
            concentration=2.0
        )

        # Both should be properly initialized
        assert decision_model.params['temperature'] == 5.0
        assert belief_model.inference_type == 'distance'
