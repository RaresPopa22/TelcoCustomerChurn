import pandas as pd
import yaml


def load_dataset(config):
    return pd.read_csv(config['data_paths']['raw_data'])


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config