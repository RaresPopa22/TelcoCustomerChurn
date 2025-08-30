import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.utils import load_dataset


def preprocess_data(config):
    columns_config = config['columns']
    data = load_dataset(config)

    data = data.drop(columns_config['id'], axis=1)

    total_charges_col = columns_config['total_charges']
    data[total_charges_col] = pd.to_numeric(data[total_charges_col], errors='coerce')

    X = data.drop('Churn', axis=1)
    y = data['Churn'].map({'Yes': 1, 'No': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

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

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    print(f'After processing: X_train.shape={X_train.shape}, X_test.shape={X_test.shape}')

    return X_train, X_test, y_train, y_test
