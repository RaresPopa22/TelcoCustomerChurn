import pandas as pd
import pytest

@pytest.fixture
def sample_base_config():
    return {'A': 1}

@pytest.fixture
def sample_column_config():
    return {
            'numeric': ['num_a', 'num_b', 'num_c'],
            'binary': ['binary_a'],
            'multiclass': ['multiclass_a', 'multiclass_b']
        }

@pytest.fixture
def sample_data():
    return pd.DataFrame({
            'num_a': [1, None, 3],
            'num_b': [4, 5, 6],
            'num_c': [7, 8, 9],
            'binary_a': ['No', 'No', 'Yes'],
            'multiclass_a': ['x', 'z', 'z'],
            'multiclass_b': ['u', 'v', 'w']
        })

@pytest.fixture
def sample_logistic_regression_config():
    return {
        'random_state': 1,
        'hyperparams': {
            'cv': 5,
            'scoring': 'average_precision',
            'n_jobs': 1
        }
    }

@pytest.fixture
def sample_logistic_regression_cv_config():
    return {
        'random_state': 1,
        'hyperparams': {
            'Cs': 10,
            'cv': 5,
            'class_weight': None,
            'scoring': 'average_precision'
        }
    }

@pytest.fixture
def sample_xgboost_config():
    return {
        'random_state': 1,
        'hyperparams': {
            'n_estimators': 500,
            'n_estimators_final': 5000,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 3,
            'scoring': 'average_precision',
            'objective': 'binary:logistic',
            'early_stopping_rounds': 50,
            'eval_metric': 'aucpr',
            'cv': 5,
            'n_jobs': -1
        }
    }