import json
import logging

from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

from src.data_processing import preprocess_data
from src.utils import find_optimal_threshold, parse_args_and_get_config, save_pipeline, save_test_data, suppress_arithmetic_noise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_eval_metrics(y_eval, probas, threshold):
    y_pred = (probas >= threshold).astype(int)
    logger.info(f'ROC AUC: {roc_auc_score(y_eval, probas):.4f}')
    logger.info(f'Recall (Churn): {recall_score(y_eval, y_pred, pos_label=1):.4f}')
    logger.info(f'Precision (Churn): {precision_score(y_eval, y_pred, pos_label=1):.4f}')
    logger.info(f'F1-Score (Churn): {f1_score(y_eval, y_pred, pos_label=1):.4f}')
    logger.info(f'PRAUC: {average_precision_score(y_eval, probas):.4f}')


def build_preprocessor(columns_config):
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    binary_pipeline = Pipeline(steps=[
        ('encoder', OrdinalEncoder())
    ])

    multiclass_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, columns_config['numeric']),
            ('binary', binary_pipeline, columns_config['binary']),
            ('cat', multiclass_pipeline, columns_config['multiclass'])
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    preprocessor.set_output(transform='pandas')

    return preprocessor

def build_model(model_name, config):
    hyperparam_config = config['hyperparams']
    if model_name == 'logistic_regression':
        return LogisticRegression(
            random_state=config.get('random_state', 1)
            )
    elif model_name == 'logistic_regression_cv':
        return LogisticRegressionCV(
            Cs=hyperparam_config['Cs'],
            cv=hyperparam_config['cv'],
            random_state=config.get('random_state', 1),
            class_weight=hyperparam_config['class_weight'],
            scoring=hyperparam_config['scoring']
        )
    elif model_name == 'xgboost':
        return XGBClassifier(
            n_estimators=hyperparam_config['n_estimators'],
            max_depth=hyperparam_config['max_depth'],
            learning_rate=hyperparam_config['learning_rate'],
            subsample=hyperparam_config['subsample'],
            colsample_bytree=hyperparam_config['colsample_bytree'],
            scale_pos_weight=hyperparam_config['scale_pos_weight'],
            objective=hyperparam_config['objective'],
            eval_metric=hyperparam_config['eval_metric'],
            random_state=config.get('random_state', 1)
        )
    else:
        raise ValueError(f'Unknown model requested: {model_name}')


def fit_pipeline(pipeline, model_name, config, X_train, y_train, X_eval, y_eval):
    hyperparam_config = config['hyperparams']
    if model_name == 'logistic_regression':
        param_grid_config = config['param_grid']
        param_grid = {
            'model__C': param_grid_config['C'],
            'model__penalty': param_grid_config['penalty'],
            'model__solver': param_grid_config['solver'],
            'model__max_iter': param_grid_config['max_iter'],
            'model__class_weight': param_grid_config['class_weight']
        }

        grid = GridSearchCV(
            pipeline,
            param_grid,
            scoring=hyperparam_config['scoring'],
            cv=hyperparam_config['cv'],
            n_jobs=hyperparam_config['n_jobs'],
            return_train_score=True,
        )

        
        grid.fit(X_train, y_train)

        logger.info(f'Best params: {grid.best_params_}')

        return grid.best_estimator_
    elif model_name == 'logistic_regression_cv':
        return pipeline.fit(X_train, y_train)
    elif model_name == 'xgboost':
        param_grid_config = config['param_grid']
        param_grid = {
            'model__max_depth': param_grid_config['max_depth'],
            'model__learning_rate': param_grid_config['learning_rate'],
            'model__subsample': param_grid_config['subsample'],
            'model__colsample_bytree': param_grid_config['colsample_bytree'],
            'model__scale_pos_weight': param_grid_config['scale_pos_weight'],
        }

        grid = GridSearchCV(
            pipeline,
            param_grid,
            scoring=hyperparam_config['scoring'],
            cv=hyperparam_config['cv'],
            n_jobs=hyperparam_config['n_jobs'],
            return_train_score=True
        )

        grid.fit(X_train, y_train, model__verbose=False)
        logger.info(f'Best params: {grid.best_params_}')

        pipeline.set_params(**grid.best_params_)
        pipeline['model'].set_params(
            n_estimators=hyperparam_config['n_estimators_final'],
            early_stopping_rounds=hyperparam_config['early_stopping_rounds']
        )

        pipeline['preprocessor'].fit(X_train)
        X_eval = pipeline['preprocessor'].transform(X_eval)
        pipeline.fit(X_train, y_train, model__eval_set=[(X_eval, y_eval)], model__verbose=False)

        return pipeline
    else:
        raise ValueError(f'Unknown model requested: {model_name}')


def train(config):
    model_name = config['model_name']
    random_state = config.get('random_state', 1)

    X_train, X_test, y_train, y_test = preprocess_data(config)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train, y_train,
        test_size=config['train_test_split']['eval_size'],
        random_state=random_state,
        stratify=y_train,
    )

    preprocessor = build_preprocessor(config['columns'])
    model = build_model(model_name, config)
    

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    logger.info(f'Training {model_name}...')
    with suppress_arithmetic_noise():
        pipeline = fit_pipeline(pipeline, model_name, config, X_train, y_train, X_eval, y_eval)

    with suppress_arithmetic_noise():
        probas = pipeline.predict_proba(X_eval)[:, 1]

    threshold = find_optimal_threshold(y_eval, probas)
    logger.info(f'Optimal threshold: {threshold:.4f}')

    log_eval_metrics(y_eval, probas, threshold)

    threshold_path = config['model_output_paths']['threshold']
    with open(threshold_path, 'w') as f:
        json.dump({'threshold': float(threshold)}, f)

    save_pipeline(config, pipeline)
    save_test_data(config, X_test, y_test)
    logger.info('Pipeline and test data saved.')


if __name__ == '__main__':
    config = parse_args_and_get_config('train')
    train(config)
