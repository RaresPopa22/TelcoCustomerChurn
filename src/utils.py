import argparse
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(config):
    return pd.read_csv(config['data_paths']['raw_data'])


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def deep_merge(base, override):
    result = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def read_configs(base_path, specific_path):
    base_config = read_config(base_path)
    specific_config = read_config(specific_path)

    return deep_merge(base_config, specific_config)


def parse_args_and_get_config(stage):
    base_config = Path(__file__).parent.parent / 'config' / 'base_config.yaml'
    parser = argparse.ArgumentParser()

    if stage == 'train':
        parser.add_argument('--config', required=True, help='Path to configuration file')
        args = parser.parse_args()
        return read_configs(base_config, args.config)
    elif stage == 'evaluate':
        parser.add_argument('--configs', nargs="+", required=True, help='Paths to model config files')
        args = parser.parse_args()
        base = read_config(base_config)
        return [deep_merge(base, read_config(c)) for c in args.configs]
    else:
        raise ValueError("Unknown stage. Only 'train' and 'evaluate' supported so far")


def save_pipeline(config, model):
    model_config = config['model_output_paths']
    os.makedirs(model_config['dir'], exist_ok=True)
    joblib.dump(model, model_config['model'])


def save_test_data(config, X_test, y_test):
    data_config = config['data_paths']
    X_test.to_csv(data_config['X_test'], index=False)
    y_test.to_csv(data_config['y_test'], index=False)


def load_test_data(config):
    data_config = config['data_paths']
    X_test = pd.read_csv(data_config['X_test'])
    y_test = pd.read_csv(data_config['y_test'])

    return X_test, y_test.squeeze()


def load_pipeline(path):
    return joblib.load(path)


def find_optimal_threshold(y_true, y_pred_proba):
    precision, recall, threshold = precision_recall_curve(y_true, y_pred_proba)
    f1_score = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_score)
    return threshold[best_idx]


@contextmanager
def suppress_arithmetic_noise():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='overflow|invalid|divide by zero', category=RuntimeWarning)
        yield


def plot_precision_recall_curve(recalls, precisions, labels, auprcs):
    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(len(recalls)):
        ax.plot(recalls[i], precisions[i], label=f'{labels[i]} (AUPRC = {auprcs[i]:.3f})')

    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve for Customer Churn', fontsize=16)
    ax.legend(loc='lower left')
    plt.show()


def plot_calibration_curve(probs_true, probs_pred, labels):
    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(len(probs_true)):
        ax.plot(probs_pred[i], probs_true[i], marker='o', label=f'{labels[i]}')

    ax.plot([0,1], [0,1], transform=ax.transAxes, color="black", linestyle="--")
    ax.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax.set_ylabel('Fraction of Positives', fontsize=14)
    ax.set_title('Calibration Curve for Customer Churn', fontsize=16)
    ax.legend(loc='lower right')
    plt.show()

