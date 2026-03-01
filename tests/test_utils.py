import numpy as np
import pandas as pd

from src.utils import deep_merge, find_optimal_threshold, load_test_data, read_config, read_configs, save_test_data


class TestUtils:

    def test_deep_merge_flat_key_override(self, sample_base_config):
        override_config = {'A': 2}
        config = deep_merge(sample_base_config, override_config)
        assert config.get('A') == 2

    def test_deep_merge_adding_new_keys(self, sample_base_config):
        override_config = {'B': 1}
        config = deep_merge(sample_base_config, override_config)
        assert config.get('A') == 1
        assert config.get('B') == 1
        
    def test_deep_merge_nested_merge(self):
        base_config = {'A': {'B': 1, 'C': 2}, 'D': 3}
        override_config = {'A': {'B': 2}}
        config = deep_merge(base_config, override_config)
        assert config == {'A': {'B': 2, 'C': 2}, 'D': 3}

    def test_deep_merge_empty(self, sample_base_config):
        override_config = {}
        config = deep_merge(sample_base_config, override_config)
        assert config.get('A') == 1

    def test_deep_merge_base_not_mutated(self, sample_base_config):
        override_config = {'B': 1}
        deep_merge(sample_base_config, override_config)
        assert sample_base_config == {'A': 1}
    
    def test_find_optimal_threshold_perfect_separation(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([.1, .3, .6, .8, .9])
        threshold = find_optimal_threshold(y_true, y_pred)
        assert threshold == .6 # at .6 recall=1.0, precision=1.0 and f1=1.0

    def test_find_optimal_threshold_boundaries(self):
        y_true = np.array([1, 0, 1, 1, 0, 1])
        y_pred = np.array([.9, .1, .3, .6, .8, .9])
        threshold = find_optimal_threshold(y_true, y_pred)
        assert 0 <= threshold <= 1

    def test_find_optimal_threshold_imbalanced(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0 , 1])
        y_pred = np.array([.1, .3, .4 , .05, .3, .44, .14, .88, .04, .06, .6])
        threshold = find_optimal_threshold(y_true, y_pred)
        assert threshold == .6

    def test_read_config_happy_path(self, tmp_path):
        yaml_path = tmp_path / 'mock.yaml'
        yaml_path.write_text("A:\n  B: 1\n  C: 2\nD: 3")

        result = read_config(yaml_path)
        assert result == {'A': {'B': 1, 'C': 2}, 'D': 3}

    def test_read_configs_happy_path(self, tmp_path):
        base_yaml_path = tmp_path / 'base.yaml'
        base_yaml_path.write_text("A:\n  B: 1\n  C: 2\nD: 3")
        override_yaml_path = tmp_path / 'override.yaml'
        override_yaml_path.write_text("A:\n  B: 2")

        result = read_configs(base_yaml_path, override_yaml_path)
        assert result == {'A': {'B': 2, 'C': 2}, 'D': 3}

    def test_save_and_load_data(self, tmp_path):
        X = pd.DataFrame({'A': [1, 2, 3]})
        y = pd.Series([0, 1, 1])

        config = {
            'data_paths': {
                'X_test': tmp_path / 'X_test.joblib',
                'y_test': tmp_path / 'y_test.joblib'
            }
        }

        save_test_data(config, X, y)
        X_read, y_read = load_test_data(config)

        pd.testing.assert_frame_equal(X, X_read)
        pd.testing.assert_series_equal(y, y_read)
