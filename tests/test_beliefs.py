"""Tests for belief models."""

import pytest
import numpy as np
from models.belief import add_standard_beliefs, add_iw_beliefs


class TestStandardBeliefs:
    """Test standard (per-player) belief model."""

    def test_adds_belief_columns(self, trial_data):
        """Test that standard beliefs add expected columns."""
        trials = add_standard_beliefs([trial_data])
        assert len(trials) == 1
        result = trials[0]
        assert 'p1_belief_p2_stag' in result.columns
        assert 'p2_belief_p1_stag' in result.columns

    def test_beliefs_in_range(self, trial_data):
        """Test that beliefs are bounded between 0 and 1."""
        result = add_standard_beliefs([trial_data])[0]
        assert all(result['p1_belief_p2_stag'] >= 0)
        assert all(result['p1_belief_p2_stag'] <= 1)
        assert all(result['p2_belief_p1_stag'] >= 0)
        assert all(result['p2_belief_p1_stag'] <= 1)

    def test_batch_processing(self, all_trials_data):
        """Test batch processing of multiple trials."""
        results = add_standard_beliefs(all_trials_data[:10])
        assert len(results) == 10
        for result in results:
            assert 'p1_belief_p2_stag' in result.columns


class TestIWBeliefs:
    """Test Imagined We (joint goal) belief model."""

    def test_adds_belief_column(self, trial_data):
        """Test that IW beliefs add expected column."""
        result = add_iw_beliefs([trial_data])[0]
        assert 'joint_goal_stag' in result.columns

    def test_beliefs_in_range(self, trial_data):
        """Test that IW beliefs are bounded."""
        result = add_iw_beliefs([trial_data])[0]
        assert all(result['joint_goal_stag'] >= 0)
        assert all(result['joint_goal_stag'] <= 1)

    def test_batch_processing(self, all_trials_data):
        """Test batch processing works."""
        results = add_iw_beliefs(all_trials_data[:10])
        assert len(results) == 10
        for result in results:
            assert 'joint_goal_stag' in result.columns


class TestBeliefDifferences:
    """Test differences between standard and IW beliefs."""

    def test_iw_uses_both_movements(self, trial_data):
        """IW should update on both players' movements, standard only on partner."""
        std_result = add_standard_beliefs([trial_data])[0]
        iw_result = add_iw_beliefs([trial_data])[0]

        # Both should have beliefs, but they track different things
        assert 'p1_belief_p2_stag' in std_result.columns
        assert 'joint_goal_stag' in iw_result.columns

        # IW belief should generally differ from average of standard beliefs
        std_avg = (std_result['p1_belief_p2_stag'] + std_result['p2_belief_p1_stag']) / 2
        iw_belief = iw_result['joint_goal_stag']

        # They won't be identical (IW updates on both movements)
        assert not np.allclose(std_avg.values, iw_belief.values, rtol=0.1)
