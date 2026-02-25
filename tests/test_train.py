import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from xgboost import XGBClassifier

from src.train import build_model, build_preprocessor


class TestTrain:

    def test_build_preprocessor_numerical_data(self, sample_column_config, sample_data):
        preprocessor = build_preprocessor(sample_column_config)
        result = preprocessor.fit_transform(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert result.isna().sum().sum() == 0
        assert np.isclose(np.mean(result['num_a']), 0, atol=1e-8)
        assert np.isclose(np.mean(result['num_b']), 0, atol=1e-8)
        assert np.isclose(np.mean(result['num_c']), 0, atol=1e-8)

    def test_build_preprocessor_binary_data(self, sample_column_config, sample_data):
        preprocessor = build_preprocessor(sample_column_config)
        result = preprocessor.fit_transform(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result['binary_a'].isin([0, 1]).all()

    def test_build_preprocessor_categorical_data(self, sample_column_config, sample_data):
        preprocessor = build_preprocessor(sample_column_config)
        result = preprocessor.fit_transform(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert 'multiclass_a_x' not in result.columns
        assert 'multiclass_a_z' in result.columns
        assert 'multiclass_b_u' not in result.columns
        assert 'multiclass_b_v' in result.columns
        assert 'multiclass_b_w' in result.columns

    def test_build_model_logistic_regression(self, sample_logistic_regression_config):
        model = build_model('logistic_regression', sample_logistic_regression_config)
        assert isinstance(model, LogisticRegression)
        params = model.get_params()
        assert params['random_state'] == sample_logistic_regression_config.get('random_state')

    def test_build_model_logistic_regression_cv(self, sample_logistic_regression_cv_config):
        model = build_model('logistic_regression_cv', sample_logistic_regression_cv_config)
        assert isinstance(model, LogisticRegressionCV)
        
        params = model.get_params()
        expected_hyperparams = sample_logistic_regression_cv_config.get('hyperparams')
        assert params['Cs'] == expected_hyperparams.get('Cs')
        assert params['cv'] == expected_hyperparams.get('cv')
        assert params['random_state'] == sample_logistic_regression_cv_config.get('random_state')
        assert params['class_weight'] == expected_hyperparams.get('class_weight')
        assert params['scoring'] == expected_hyperparams.get('scoring')

    def test_build_model_xgboost(self, sample_xgboost_config):
        model = build_model('xgboost', sample_xgboost_config)
        assert isinstance(model, XGBClassifier)

        params = model.get_params()
        expected_hyperparams = sample_xgboost_config.get('hyperparams')
        assert params['n_estimators'] == expected_hyperparams.get('n_estimators')
        assert params['max_depth'] == expected_hyperparams.get('max_depth')
        assert params['learning_rate'] == expected_hyperparams.get('learning_rate')
        assert params['subsample'] == expected_hyperparams.get('subsample')
        assert params['colsample_bytree'] == expected_hyperparams.get('colsample_bytree')
        assert params['scale_pos_weight'] == expected_hyperparams.get('scale_pos_weight')
        assert params['objective'] == expected_hyperparams.get('objective')
        assert params['eval_metric'] == expected_hyperparams.get('eval_metric')
        assert params['random_state'] == sample_xgboost_config.get('random_state')

    def test_build_model_unknown_model(self):
        with pytest.raises(ValueError, match=r"(?i)(?=.*Unknown)(?=.*LSTM)"):
            build_model('LSTM', {'hyperparams': {}})

