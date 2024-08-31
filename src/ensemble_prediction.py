import glob
import pandas as pd
import numpy as np

from uncertainty_metrics import *


def ensemble_predictions(models_preds, metric_func, comp_func):
    """
    Aggregate predictions from multiple models based on the given metric.

    :param models_preds: List of dictionaries containing 'logprobs' and 'prediction' for each model.
    :param metric_func: Metric function to apply to probabilities (e.g., maxprob, margin, entropy).
    :param comp_func: Function to compare the aggregated scores for each choice (e.g., np.argmax to select the highest score).
    :return: The selected answer letter based on the ensemble.
    """
    choice_tokens = models_preds[0]['choices']
    num_choices = len(choice_tokens)
    aggregated_scores = np.zeros(num_choices)
    
    for model in models_preds:
        logprobs = model['logprobs_extracted']
        probs = np.exp(logprobs - np.max(logprobs))
        probs /= probs.sum()
        aggregated_scores += np.array([metric_func(probs)])
    
    best_choice_index = comp_func(aggregated_scores)
    return choice_tokens[best_choice_index]


def process_csv_files(file_paths, metric_func, comp_func):
    """
    Process multiple CSV files and perform ensemble predictions using pandas grouping. 
    CSV columns:
        question: (str) The text of the question being answered.
        choices: (str) Comma-separated list of answer choices.
        logprobs: (dict-like) A dictionary of log probabilities corresponding to each token in 'vocab'.
        prediction: (Optional) (str) The model's predicted answer.
    
    :param file_paths: List of file paths to CSV files containing model predictions.
    :param metric_func: Metric function to apply to probabilities (e.g., maxprob, margin, entropy).
    :param comp_func: Function to compare the aggregated scores for each choice (e.g., np.argmax).
    :return: DataFrame with questions and ensemble predictions.
    """
    def extract_logprobs(row):
        choices = row['choices']
        full_logprobs = row['logprobs']
        return np.array([full_logprobs.get(str(c), -np.inf) for c in choices])
    
    df_list = [pd.read_csv(file_path) for file_path in file_paths]
    all_predictions_df = pd.concat(df_list, ignore_index=True)
    all_predictions_df['choices'] = all_predictions_df['choices'].apply(lambda x: x.split(','))
    all_predictions_df['logprobs'] = all_predictions_df['logprobs'].apply(eval)
    all_predictions_df['logprobs_extracted'] = all_predictions_df.apply(extract_logprobs, axis=1)
    
    def apply_ensemble(group):
        models_preds = group[['choices', 'logprobs_extracted', 'prediction']].to_dict('records')
        return ensemble_predictions(models_preds, metric_func, comp_func)

    ensemble_results_df = all_predictions_df.groupby('question').apply(apply_ensemble).reset_index()
    ensemble_results_df.columns = ['question', 'prediction']
    return ensemble_results_df


if __name__ == "__main__":
    file_paths = glob.glob('path_to_csv_files/*.csv')
    metric_func = maxprob
    comp_func = np.argmax

    ensemble_df = process_csv_files(file_paths, metric_func, comp_func)
    ensemble_df.to_csv('/home/admin/graphs/test/ensemble_predictions.csv', index=False)
    print(ensemble_df)
