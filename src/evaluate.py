import json
import logging
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, roc_auc_score, \
    f1_score, recall_score, precision_score

from sklearn.calibration import calibration_curve

from src.utils import parse_args_and_get_config, load_test_data, load_pipeline, plot_precision_recall_curve, plot_calibration_curve, suppress_arithmetic_noise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(configs):
    X_test, y_true = load_test_data(configs[0])

    results, precisions, recalls, labels, auprcs = [], [], [], [], []
    cal_probs_true, cal_probs_pred = [], []

    for config in configs:
        model_path = config['model_output_paths']['model']
        model = load_pipeline(model_path)

        with suppress_arithmetic_noise():
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        threshold_path = config['model_output_paths']['threshold']
        with open(threshold_path, 'r') as f:
            threshold = json.load(f).get('threshold')
             
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)

        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        cal_probs_true.append(prob_true)
        cal_probs_pred.append(prob_pred)

        report = classification_report(y_true, y_pred, target_names=['Non-Churn', 'Churn'])
        logger.info(f"Classification report for {config['model_name']}")
        logger.info(report)

        recalls.append(recall)
        precisions.append(precision)
        labels.append(config['model_name'])
        auprcs.append(auprc)

        results.append({
            'Model': config['model_name'],
            'AUPRC': auprc,
            'ROC AUC': roc_auc_score(y_true, y_pred_proba),
            'Recall (Churn)': recall_score(y_true, y_pred, pos_label=1),
            'Precision (Churn)': precision_score(y_true, y_pred, pos_label=1),
            'F1-Score (Churn)': f1_score(y_true, y_pred, pos_label=1)
        })

    results_df = pd.DataFrame(results).set_index('Model')
    pd.set_option('display.max_columns', None)
    logger.info(f"\n{results_df.round(4).to_string()}")

    plot_precision_recall_curve(recalls, precisions, labels, auprcs)
    plot_calibration_curve(cal_probs_true, cal_probs_pred, labels)



if __name__ == '__main__':
    configs = parse_args_and_get_config('evaluate')
    evaluate(configs)
