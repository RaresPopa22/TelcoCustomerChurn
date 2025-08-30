from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, roc_auc_score, \
    f1_score

from src.utils import parse_args_and_get_config, load_test_data, load_model, plot_precision_recall_curve


def evaluate(config, model_paths):
    X_test, y_test = load_test_data(config)

    results, precisions, recalls, labels, auprcs = [], [], [], [], []

    for path in model_paths:
        model_path = Path(path)
        model = load_model(model_path)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)

        report = classification_report(y_test, y_pred, target_names=['Non-Churn', 'Churn'])
        print(f'Classification report for {model_path.stem}')
        print(report)

        recalls.append(recall)
        precisions.append(precision)
        labels.append(model_path.stem)
        auprcs.append(auprc)

        results.append({
            'Model': model_path.stem,
            'AUPRC': auprc,
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            'Recall (Churn)': recall,
            'Precision (Churn)': precision,
            'F1-Score (Churn)': f1_score(y_test, y_pred, pos_label=1)
        })

    results_df = pd.DataFrame(results).set_index('Model')
    print(results_df.round(4))

    plot_precision_recall_curve(recalls, precisions, labels, auprcs)


if __name__ == '__main__':
    config, model_paths = parse_args_and_get_config('evaluate')
    evaluate(config, model_paths)
