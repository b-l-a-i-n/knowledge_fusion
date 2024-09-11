
import os
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from datasets import load_dataset
from transformers.utils import logging
logging.set_verbosity_error() 

from prompts import combine_two_contexts, make_no_context_prompt


def convert_epsilon_to_text(d: dict):
    entity = f"Wikidata entity: {d['entity']['label']} ({d['entity']['description']})"
    text = entity
    if len(d["neighbors"]) > 0:    
        text += ", Entity properties: "
        text += ";\n".join([f"{n['relation_label']}, {n['neighbor_label']} ({n['neighbor_description']})" for n in d['neighbors']])
    text += "."
    return text


def generate_content(prompt):
    url = "http://10.11.1.8:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "/archive/beliakin/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393/",
        "messages": [{
        "role": "user",
        "content": prompt}],
        "max_tokens": 100,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 256,
    }
    
    response = requests.post(url, headers=headers, json=data)
    text = response.json()["choices"][0]["message"]["content"]
    return text, response.json()["choices"][0]["logprobs"]["content"]


def predict(
    ds, zero_path, 
    size_wikidata=None, 
    size_google=250,
    size_ddg=250,
    size_wikipedia=250,
    number_of_context=5, 
    **kwargs
):
    zero_df = pd.read_csv(zero_path)
    ddg_df = pd.read_csv(ddg_path) if "ddg_path" in kwargs else None
    google_df = pd.read_csv(google_path) if "google_path" in kwargs else None
    wikidata_df = pd.read_csv(wikidata_path) if "wikidata_path" in kwargs else None
    wikipedia_df = pd.read_csv(wikipedia_path) if "wikipedia_path" in kwargs else None
    
    stats, correct_labels = [], []
    for idx, row in tqdm(zero_df.iterrows(), total=zero_df.shape[0]):
        question = row['question']
        answers, labels = shuffle(
            ds[idx]['mc1_targets']['choices'], ds[idx]['mc1_targets']['labels'],
            random_state=0
        )
        correct_labels.append(labels)
        
        contexts = {key: [] for key in ["wikipedia", "wikidata", "ddg", "google"]}
        for i in range (1, number_of_context + 1):
            
            if wikidata_df is not None and wikidata_df[f'context_{i}'].iloc[idx]:
                if type(wikidata_df[f'context_{i}'].iloc[idx]) is str:
                    text = convert_epsilon_to_text(eval(wikidata_df[f'context_{i}'].iloc[idx]))
                    contexts["wikidata"].append(text[:size_wikidata])
            
            if wikipedia_df is not None and wikipedia_df[f'context_{i}'].iloc[idx]:
                ctx = eval(wikipedia_df[f'context_{i}'].iloc[idx])
                if len(ctx) > 0:
                    contexts["wikipedia"].append(ctx['context'][:size_wikipedia])
            
            if ddg_df is not None and ddg_df[f'context_{i}'].iloc[idx]:
                contexts["ddg"].append(ddg_df[f'context_{i}'].iloc[idx][:size_ddg])
            
            if google_df is not None and google_df[f'context_{i}'].iloc[idx]:
                contexts["google"].append(google_df[f'context_{i}'].iloc[idx][:size_google])
                        
        if any(contexts.values()):
            joint_context = [
                "\n".join(["- " + s.replace("\n", " ") + "..." for s in c]) 
                for c in contexts.values() if c
            ]
            text = combine_two_contexts(joint_context, question, answers)
        else:
            text = make_no_context_prompt(question, answers)
        
        stat = cache_prediction(text)
        for s in stat:
            s['idx'] = idx
            s['correct'] = chr(ord('A') + correct_labels[idx].index(1))
        stats += stat
    
    return stats


def calculate_entropy(logprobs):
    entropies = []
    for s_lp in logprobs:
        entropies.append([])
        for lp in s_lp:
            mask = ~np.isinf(lp)
            entropies[-1].append(-np.sum(np.array(lp[mask]) * np.exp(lp[mask])))
    return entropies


def calculate_msp(token_logprobs):
    return -np.sum([logprobs[0] for logprobs in token_logprobs])


def calculate_perplexity(token_logprobs):
    sum_logprobs = np.sum([logprobs[0] for logprobs in token_logprobs])
    return -(sum_logprobs / len(token_logprobs))
        
        
def cache_prediction(text):
    generated_texts, token_logprobs = generate_content(text)
    logprobs = [np.array([t['logprob'] for t in tl['top_logprobs']]) for tl in token_logprobs]
    msp = calculate_msp(logprobs)
    perplexity = calculate_perplexity(logprobs)
    mean_token_entropy = np.mean([calculate_entropy(lp) for lp in logprobs])
    stat = {
        'msp': msp,
        'perplexity': perplexity,
        'entropy': mean_token_entropy,
        'text': generated_texts,
        'logprobs': token_logprobs,
    }
    return stat


if __name__ == "__main__":    
    data_path = Path(os.getenv("DATA_PATH", "../../data"))
    dataset_name = Path("truthfulqa_multichoice")
    
    zero_path = data_path / dataset_name / "questions.csv"
    ddg_path = data_path / dataset_name / "duckduckgo.csv"
    google_path = data_path / dataset_name / "google.csv"
    wikipedia_path = data_path / dataset_name / "wikipedia.csv"
    wikidata_path = data_path / dataset_name / "wikidata.csv"
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
    
    stats = predict(
        ds, zero_path=zero_path, 
        ddg_path=ddg_path,
        # google_path=google_path,
        # wikipedia_path=wikipedia_path,
        # wikidata_path=wikidata_path
    )
    df = pd.DataFrame(stats)
    df.to_csv(data_path / f"test_answers/{dataset_name}_ddg_llama_70b.csv", index=False)
