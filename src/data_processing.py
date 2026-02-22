import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_dataset

def preprocess_data(config, split_for_eval=False):
    columns_config = config['columns']
    data = load_dataset(config)

    data = data.drop(columns_config['id'], axis=1)

    total_charges_col = columns_config['total_charges']
    data[total_charges_col] = pd.to_numeric(data[total_charges_col], errors='coerce')

    X = data.drop('Churn', axis=1)
    y = data['Churn'].map({'Yes': 1, 'No': 0})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['train_test_split']['test_size'], random_state=1, stratify=y)

    
    return X_train, X_test, y_train, y_test
