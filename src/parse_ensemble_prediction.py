import re
import warnings
from itertools import permutations

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from scipy.special import softmax

from .uncertainty_metrics import maxprob
from .evaluate import evaluate_prediction


def get_answers(text):
    if re.fullmatch(r"\d+", text):
        matches = re.findall(r"\d+", text)
    elif re.fullmatch(r"[\d+, ]+", text):
        matches = re.findall(r"\d+", text)
    else:
        matches = re.findall(r"\d+", text)[-1:]
    return [int(m) for m in matches] if matches else []


def process_ensemble(preds_list, metric, comparator):
    combined_preds = preds_list[0].copy()
    combined_preds["top_tokens"] = []

    for i in range(len(combined_preds)):
        top_tokens = [preds[][i] for preds in preds_list]
        metric_values = [preds[metric][i] for preds in preds_list]
        
        best_metric_index = comparator(metric_values)
        best_token = top_tokens[best_metric_index]

        combined_preds.at[i, "top_tokens"] = top_tokens
        combined_preds.at[i, "prediction"] = best_token

    combined_preds["prediction_sample_id"] = [
        [x["match_dict"].get(j, None) for j in x["prediction"] if j in x["match_dict"]] 
        for _, x in combined_preds.iterrows()
    ]

    return combined_preds


def get_ordered_logprobs(row):
    tokens = list(row["match_dict"].keys())
    result = []
    for prob in row["prob"]:
        logprob_dict = prob
        min_logprob = min(logprob_dict.values())
        logprobs = [logprob_dict.get(str(token), min_logprob) for token in tokens]
        result.append(logprobs)
    return resultd


def init_preds(preds, func):
    preds["raw_prediction"] = preds["prediction"]
    preds["prob"] = preds["prob"].apply(eval)
    preds["prob"] = [
        r["prob"][::3] if re.fullmatch(r"\d+", r["raw_prediction"]) \
        or re.fullmatch(r"[\d+, ]+", r["raw_prediction"]) else r["prob"][:1] 
        for _, r in preds.iterrows()
    ]
    preds["match_dict"] = preds["match_dict"].apply(eval)
    preds["prediction"] = preds["prediction"].apply(get_answers)
    
    ordered_logprobs = [get_ordered_logprobs(row) for _, row in preds.iterrows()]
    preds["softmax_probs"] = [[list(softmax(i)) for i in x] for x in ordered_logprobs]
    preds[func.__name__] = preds["softmax_probs"].apply(lambda x: sum([func(i) for i in x]) / len(x))
    preds["top_tokens"] = preds["raw_prediction"].apply(get_answers)
    return preds
    

if __name__ == "__main__":
    metric_func = maxprob
    
    preds_paths = [
        "output/llama_70B_train_sample_ds_ws_probs.tsv", 
        "output/llama_70B_train_sample_ds_probs.tsv",
        "output/llama_70B_train_sample_ds_tg_probs.tsv"
    ]
    preds_list = [init_preds(pd.read_csv(p, sep="\t"), func=metric_func) for p in preds_paths]
    combined_preds = process_ensemble(preds_list, metric=metric_func.__name__, comparator=np.argmax)
    
    main_df = pd.read_csv("TextGraphs17-shared-task/data/tsv/train_sample.tsv", sep="\t")
    res = evaluate_prediction(combined_preds, main_df, True)
    print(res)
