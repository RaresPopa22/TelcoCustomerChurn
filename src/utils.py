import argparse
import os

import joblib
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve


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
    base_config = "config/base_config.yaml"
    parser = argparse.ArgumentParser()

    if stage == 'train':
        parser.add_argument('--config', required=True, help='Path to configuration file')
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
    X_test.to_csv(data_config['X_test'], index=False)
    y_test.to_csv(data_config['y_test'], index=False)


def load_test_data(config):
    data_config = config['data_paths']
    X_test = pd.read_csv(data_config['X_test'])
    y_test = pd.read_csv(data_config['y_test'])

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


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()


def plot_roc(y_test, y_pred_probas, labels):
    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(len(y_pred_probas)):
        fpr, tpr, _ = roc_curve(y_test, y_pred_probas[i])
        auc = roc_auc_score(y_test, y_pred_probas[i])
        ax.plot(fpr, tpr, label=f'{labels[i]} (AUC = {auc:.3f})')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc=4)
    plt.show()
