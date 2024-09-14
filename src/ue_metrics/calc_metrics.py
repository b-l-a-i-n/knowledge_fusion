import numpy as np
import pandas as pd
from transformers.utils import logging
from transformers import AutoModelForCausalLM
logging.set_verbosity_error() 
import os
os.environ['CUDA_VISIBLE_DEVICES']="3"
import torch
from tqdm import tqdm
from lm_polygraph import WhiteboxModel
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.stat_calculators.embeddings import get_embeddings_from_output
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel, Model
from lm_polygraph.utils.processor import Processor
from transformers import AutoTokenizer
from sklearn.utils import shuffle
from datasets import load_dataset

#model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
model_name = 'meta-llama/Meta-Llama-3.1-8B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = WhiteboxModel.from_pretrained(
    model_name, device_map='auto',
)

from criteria import StopWordCriteria
stop_words = ["\n", "QUESTION"]
stopping_criteria = StopWordCriteria(tokenizer=tokenizer, prompts=[], stop_words=stop_words)
from greedy_probs import GreedyProbsCalculator
from prompts import combine_two_contexts, make_no_context_prompt, make_prompt

name = 'medicine'
ddg_path = f'../fusion/knowledge_fusion/data/retrieve_to_models/{name}_mmlu/duckduckgo.csv'
google_path = f'../fusion/knowledge_fusion/data/retrieve_to_models/{name}_mmlu/google.csv'
wikipedia_path = f'../fusion/knowledge_fusion/data/retrieve_to_models/{name}_mmlu/wikipedia.csv'
general_path = f'../fusion/knowledge_fusion/data/retrieve_to_models/{name}_mmlu.csv'


retrieved_truthful_qa_mc_1 = pd.read_csv(ddg_path)
retrieved_truthful_qa_mc_2 = pd.read_csv(google_path)
retrieved_truthful_qa_mc_3 = pd.read_csv(wikipedia_path)
ds = pd.read_csv(general_path, index_col=[0])


def predict(
    ds, 
    size_wikidata=None, 
    size_google=500,
    size_ddg=500,
    size_wikipedia=500,
    number_of_context=5, 
    **kwargs
):
    ddg_df = pd.read_csv(ddg_path) if "ddg_path" in kwargs else None
    google_df = pd.read_csv(google_path) if "google_path" in kwargs else None
    wikidata_df = pd.read_csv(wikidata_path) if "wikidata_path" in kwargs else None
    wikipedia_df = pd.read_csv(wikipedia_path) if "wikipedia_path" in kwargs else None
    
    stats, correct_labels, ue_metrics = [], [], []
    for idx, row in tqdm(retrieved_truthful_qa_mc_1.iterrows(), total=retrieved_truthful_qa_mc_1.shape[0]):
        question = row['question']
        # answers, labels = shuffle(ds[idx]['mc1_targets']['choices'], ds[idx]['mc1_targets']['labels'], random_state=0)
        answers = eval(ds.iloc[idx].choices)
        correct_labels.append(ds.iloc[idx].answer)
        
        contexts = {key: [] for key in ["wikipedia", "wikidata", "ddg", "google"]}
        for i in range (1, number_of_context + 1):
            
            if wikidata_df is not None and type(wikidata_df[f'context_{i}'].iloc[idx]) is str:
                if type(wikidata_df[f'context_{i}'].iloc[idx]) is str:
                    text = convert_epsilon_to_text(eval(wikidata_df[f'context_{i}'].iloc[idx]))
                    contexts["wikidata"].append(text[:size_wikidata])
            
            if wikipedia_df is not None and type(wikipedia_df[f'context_{i}'].iloc[idx]) is str:
                ctx = eval(wikipedia_df[f'context_{i}'].iloc[idx])
                if len(ctx) == 0:
                    continue
                elif ctx['is_summary'] is True:
                    contexts["wikipedia"].append(ctx['context'][:size_wikipedia])
                else:
                    contexts["wikipedia"].append(ctx['context'][:size_wikipedia])
            
            if ddg_df is not None and type(ddg_df[f'context_{i}'].iloc[idx]) is str:
                contexts["ddg"].append(ddg_df[f'context_{i}'].iloc[idx][:size_ddg])
            
            if google_df is not None and type(google_df[f'context_{i}'].iloc[idx]) is str:
                contexts["google"].append(google_df[f'context_{i}'].iloc[idx][:size_google])
                        
        if len([c for c in contexts.values() if c]):
            texts = [
                combine_two_contexts(
                    [
                        "\n".join(["- " + s.replace("\n", " ") + "..." for s in c]) 
                        for c in contexts.values() if c
                    ], 
                    question,
                    answers
                )
            ]
        else:
            texts = [
                make_no_context_prompt(
                    question,
                    answers
                )
            ]
        
        stat = {}
        flg = False
        for calculator in [
            GreedyProbsCalculator()
        ]:
            try:
                stat.update(calculator(stat, texts, model))    
            except IndexError as err:
                print(texts)
                flg = True
                break
        if flg:
            stats.append({})
            ue_metrics.append({})
            continue
        log_likelihoods = stat['greedy_log_likelihoods']
        msp = np.array([-np.sum(log_likelihood) for log_likelihood in log_likelihoods])
        perplexity = np.array([-np.mean(ll) for ll in log_likelihoods])
        mean_token_entropy = np.array([np.mean(e) for e in stat['entropy']])
        stats.append(stat)
        ue_metrics.append({'msp': msp, 'perplexity': perplexity, 'entropy': mean_token_entropy})
        
    return stats, correct_labels, ue_metrics

def save(stats, correct_labels, ue_metrics, name, mmlu_topic):
    df = []
    for idx, s in enumerate(stats):
        # print(chr(ord('A') + correct_labels[idx].index(1)))
        answers = ''
        for v in s['greedy_texts']:
            answers += v.strip()[0]
        #print(answers)
        df.append({'idx': idx, 'correct': chr(ord('A') + correct_labels[idx]), 'answers': answers,
                   'msp': ue_metrics[idx]['msp'][0], 'perplexity': ue_metrics[idx]['perplexity'][0], 'entropy': ue_metrics[idx]['entropy'][0]})
    d = pd.DataFrame(df)
    d.to_csv(f'../fusion/knowledge_fusion/data/model_answers/{mmlu_topic}_mmlu/{mmlu_topic}_mmlu_{name}_llama.csv')


stats, correct_labels, ue_metrics = stats, correct_labels, ue_metrics = predict(
        ds,
        ddg_path=ddg_path,
        # google_path=google_path,
        # wikipedia_path=wikipedia_path,
        # wikidata_path=wikidata_path
        mmlu_topic='medicine'
        )
save(stats, correct_labels, ue_metrics, 'ddg', mmlu_topic='medicine')

stats, correct_labels, ue_metrics = stats, correct_labels, ue_metrics = predict(
        ds,
        # ddg_path=ddg_path,
        google_path=google_path,
        # wikipedia_path=wikipedia_path,
        # wikidata_path=wikidata_path
        mmlu_topic='medicine'
)
save(stats, correct_labels, ue_metrics, 'google', mmlu_topic='medicine')

stats, correct_labels, ue_metrics = stats, correct_labels, ue_metrics = predict(
        ds,
        # ddg_path=ddg_path,
        # google_path=google_path,
        wikipedia_path=wikipedia_path,
        # wikidata_path=wikidata_path
        mmlu_topic='medicine'
)
save(stats, correct_labels, ue_metrics, 'wikipedia', mmlu_topic='medicine')

stats, correct_labels, ue_metrics = stats, correct_labels, ue_metrics = predict(
        ds,
        ddg_path=ddg_path,
        # google_path=google_path,
        wikipedia_path=wikipedia_path,
        # wikidata_path=wikidata_path
        mmlu_topic='medicine'
)
save(stats, correct_labels, ue_metrics, 'ddg_wikipedia', mmlu_topic='medicine')

stats, correct_labels, ue_metrics = stats, correct_labels, ue_metrics = predict(
        ds,
        ddg_path=ddg_path,
        google_path=google_path,
        # wikipedia_path=wikipedia_path,
        # wikidata_path=wikidata_path
        mmlu_topic='medicine'
)
save(stats, correct_labels, ue_metrics, 'ddg_google', mmlu_topic='medicine')

stats, correct_labels, ue_metrics = stats, correct_labels, ue_metrics = predict(
        ds,
        # ddg_path=ddg_path,
        google_path=google_path,
        wikipedia_path=wikipedia_path,
        # wikidata_path=wikidata_path
        mmlu_topic='medicine'
)
save(stats, correct_labels, ue_metrics, 'google_wikipedia',mmlu_topic='medicine')

stats, correct_labels, ue_metrics = stats, correct_labels, ue_metrics = predict(
        ds,
        ddg_path=ddg_path,
        google_path=google_path,
        wikipedia_path=wikipedia_path,
        # wikidata_path=wikidata_path
        mmlu_topic='medicine'
)
save(stats, correct_labels, ue_metrics, 'ddg_google_wikipedia', mmlu_topic='medicine')

