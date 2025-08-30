from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier

from src.data_processing import preprocess_data
from src.utils import parse_args_and_get_config, save_model, save_test_data

from sklearn.model_selection import train_test_split


def train(config):
    model_name = config['model_name']
    X_train, X_test, y_train, y_test = preprocess_data(config)

    if model_name == 'logistic_regression_cv':
        model = LogisticRegressionCV(Cs=10, cv=5, random_state=1, class_weight='balanced')
        model.fit(X_train, y_train)
    elif model_name == 'xgboost':
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_train, y_train, test_size=0.2, random_state=1, stratify=y_train
        )

        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=scale_pos_weight,
            n_estimators=1000,
            random_state=1,
            early_stopping_rounds=10
        )
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)

    save_model(config, model)
    save_test_data(config, X_test, y_test)


if __name__ == '__main__':
    config = parse_args_and_get_config('train')
    train(config)
