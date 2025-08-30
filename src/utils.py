import argparse
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt


def load_dataset(config):
    return pd.read_csv(config['data_paths']['raw_data'])


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def read_configs(base_path, specific_path):
    base_config = read_config(base_path)
    specific_config = read_config(specific_path)

    return {**base_config, **specific_config}


def parse_args_and_get_config(stage):
    base_config = "../config/base_config.yaml"
    parser = argparse.ArgumentParser()

    if stage == 'train':
        parser.add_argument('--config', required=True,  help='Path to configuration file')
        args = parser.parse_args()
        return read_configs(base_config, args.config)
    elif stage == 'evaluate':
        parser.add_argument('--models', nargs="+", required=True, help='Paths to model files')
        parser.add_argument('--x-test', required=True, help='Paths to X_test_scaled.csv')
        parser.add_argument('--y-test', required=True, help='Paths to y_test.csv')
        args = parser.parse_args()
        return read_config(base_config), args.models
    else:
        raise ValueError("Unknown stage. Only 'train' and 'evaluate' supported so far")


def save_model(config, model):
    model_config = config['model_output_paths']
    os.makedirs(model_config['dir'], exist_ok=True)
    joblib.dump(model, model_config['model'])


def save_test_data(config, X_test, y_test):
    data_config = config['data_paths']
    np.save(data_config['X_test_npy'], X_test)
    np.save(data_config['y_test_npy'], y_test)


def load_test_data(config):
    data_config = config['data_paths']
    X_test = np.load(data_config['X_test_npy'])
    y_test = np.load(data_config['y_test_npy'])

    return X_test, y_test.squeeze()


def load_model(path):
    return joblib.load(path)


def plot_precision_recall_curve(recalls, precisions, labels, auprcs):
    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(len(recalls)):
        ax.plot(recalls[i], precisions[i], label=f'{labels[i]} (AUPRC = {auprcs[i]:.3f})')

    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precission', fontsize=14)
    ax.set_title('Precision-Recall Curve for Customer Churn', fontsize=16)
    ax.legend(loc='lower left')
    plt.show()
