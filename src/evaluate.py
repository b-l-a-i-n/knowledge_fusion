import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def default_report(true_labels, pred_labels):
    result = dict()
    result["precision"] = precision_score(true_labels, pred_labels)
    result["recall"] = recall_score(true_labels, pred_labels)
    result["f1"] = f1_score(true_labels, pred_labels)
    result["accuracy"] = accuracy_score(true_labels, pred_labels)
    return result


def evaluate_prediction(preds, df, report=False):
    new_df = {"sample_id": [], "prediction": [], "type": []}

    for k, (_, row) in enumerate(preds.iterrows()):
        for _, id in row["match_dict"].items():
            new_df["sample_id"].append(id)
            new_df["prediction"].append(id in row["prediction_sample_id"])
            new_df["type"].append(types[k])

    result = pd.DataFrame(data=new_df)
    result = result.sort_values(by=["sample_id"])
    result.prediction = result.prediction.astype(np.int32)
    df = df.sort_values(by=["sample_id"])

    if report:
        true_labels = df["correct"].astype(np.int32).values
        pred_labels = result["prediction"].astype(np.int32).values
        metrics = default_report(true_labels, pred_labels)
        return result, metrics
    
    return result
