import pandas as pd
import numpy as np

from uncertainty_metrics import *


def ensemble_predictions(models_preds, metric_func, comp_func):
    """
    Aggregate predictions from multiple models based on the given metric.

    :param models_preds: List of dictionaries containing 'prediction' and 'probs_extracted' for each model.
    :param metric_func: Metric function to apply to probabilities (e.g., maxprob, margin, entropy).
    :param comp_func: Function to compare the aggregated scores for each choice (e.g., np.argmax to select the highest score).
    :return: The selected answer letter based on the ensemble.
    """
    metric_values = [metric_func(model['probs_extracted']) for model in models_preds]
    best_index = comp_func(metric_values)
    return models_preds[best_index]['prediction']


def process_csv_files(file_paths, metric_func, comp_func):
    """
    Process multiple CSV files and perform ensemble predictions using pandas grouping. 
    CSV columns:
        question: (str) The text of the question being answered.
        choices: (str) Comma-separated list of answer choices.
        logprobs: (dict-like) A dictionary of log probabilities corresponding to each token in 'vocab'.
        prediction: (str) The model's predicted answer.
    
    :param file_paths: List of file paths to CSV files containing model predictions.
    :param metric_func: Metric function to apply to probabilities (e.g., maxprob, margin, entropy).
    :param comp_func: Function to compare the aggregated scores for each choice (e.g., np.argmax).
    :return: DataFrame with questions and ensemble predictions.
    """
    def extract_probs(row):
        choices = row['choices']
        full_logprobs = row['logprobs']
        logprobs = np.array([full_logprobs.get(str(c), -np.inf) for c in choices])
        probs = np.exp(logprobs - np.max(logprobs))
        probs /= probs.sum()
        return probs
    
    def apply_ensemble(group):
        models_preds = group[['probs_extracted', 'prediction']].to_dict('records')
        return ensemble_predictions(models_preds, metric_func, comp_func)
    
    df_list = [pd.read_csv(file_path) for file_path in file_paths]
    all_predictions_df = pd.concat(df_list, ignore_index=True)
    all_predictions_df['choices'] = all_predictions_df['choices'].apply(lambda x: x.split(','))
    all_predictions_df['logprobs'] = all_predictions_df['logprobs'].apply(eval)
    all_predictions_df['probs_extracted'] = all_predictions_df.apply(extract_probs, axis=1)
    
    ensemble_results_df = all_predictions_df.groupby('question').apply(apply_ensemble).reset_index()
    ensemble_results_df.columns = ['question', 'prediction']
    return ensemble_results_df


if __name__ == "__main__":
    import glob
    file_paths = glob.glob('path_to_csv_files/*.csv')
    ensemble_df = process_csv_files(file_paths, maxprob, np.argmax)
    ensemble_df.to_csv('path_to_ensemble_predictions.csv', index=False)
    print(ensemble_df)
