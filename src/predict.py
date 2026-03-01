import argparse

import joblib
import pandas as pd

from src.utils import read_config

def predict_from_dataframe(model, data: pd.DataFrame):
    data = data.copy()
    y_pred_proba = model.predict_proba(data)[:, 1]
    return y_pred_proba


def predict(config, input_path):
    model = joblib.load(config['model_output_paths']['model'])
    data = pd.read_csv(input_path)

    return predict_from_dataframe(model, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run churn prediction on new data')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--input', required=True, help='Path to input')
    parser.add_argument('--output', required=True, help='Path to save prediction in CSV format')
    args = parser.parse_args()


    predictions = predict(read_config(args.config), args.input)
    results = pd.DataFrame({
        'predictions': predictions
    })
    results.to_csv(args.output, index=False)