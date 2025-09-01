import numpy as np
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from xgboost import XGBClassifier

from src.data_processing import preprocess_data
from src.utils import parse_args_and_get_config, save_model, save_test_data

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


def train(config):
    model_name = config['model_name']
    X_train, X_test, y_train, y_test = preprocess_data(config)

    if model_name == 'logistic_regression_cv':
        hyperparams = config['hyperparams']
        model = LogisticRegressionCV(
            Cs=hyperparams['Cs'],
            cv=hyperparams['cv'],
            random_state=1,
            class_weight=hyperparams['class_weight'])
        model.fit(X_train, y_train)
    elif model_name == 'logistic_regression':
        hyperparams = config['hyperparams']
        model = LogisticRegression(
            max_iter=hyperparams['max_iter'], class_weight=hyperparams['class_weight'], random_state=1)
        param_grid_config = config['param_grid']
        C_config = param_grid_config['C']
        param_grid = {
            'penalty': param_grid_config['penalty'],
            'C': np.logspace(C_config['start'], C_config['stop'], C_config['num']),
            'solver': param_grid_config['solver']
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=hyperparams['cv'],
            scoring=hyperparams['scoring'],
            n_jobs=hyperparams['n_jobs']
        )
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
    elif model_name == 'xgboost':
        hyperparams = config['hyperparams']
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_train, y_train, test_size=hyperparams['cv_size'], random_state=1, stratify=y_train
        )

        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        model = XGBClassifier(
            objective=hyperparams['objective'],
            eval_metric=hyperparams['eval_metric'],
            scale_pos_weight=scale_pos_weight,
            n_estimators=hyperparams['n_estimators'],
            random_state=1,
            early_stopping_rounds=hyperparams['early_stopping_rounds']
        )

        if config['tune']:
            param_distributions = {
                'max_depth': randint(
                    hyperparams['tune']['max_depth']['min'],
                    hyperparams['tune']['max_depth']['max']
                ),
                'learning_rate': uniform(
                    hyperparams['tune']['learning_rate']['min'],
                    hyperparams['tune']['learning_rate']['max'] - hyperparams['tune']['learning_rate']['min'],
                )
            }

            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_distributions,
                n_iter=hyperparams['n_iter'],
                cv=hyperparams['cv'],
                scoring=hyperparams['scoring'],
                n_jobs=hyperparams['n_jobs'],
                verbose=hyperparams['verbose'],
                random_state=1,
            )
            random_search.fit(X_train, y_train, eval_set=[(X_eval, y_eval)])
            model = random_search.best_estimator_
        else:
            model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)

    save_model(config, model)
    save_test_data(config, X_test, y_test)


if __name__ == '__main__':
    config = parse_args_and_get_config('train')
    train(config)
