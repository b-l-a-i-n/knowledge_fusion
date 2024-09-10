import numpy as np
import pandas as pd
from transformers.utils import logging
from transformers import AutoModelForCausalLM
logging.set_verbosity_error() 
import os
import random
os.environ['CUDA_VISIBLE_DEVICES']="2,3"
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.utils import shuffle
from datasets import load_dataset

#model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
#from criteria import StopWordCriteria
#stop_words = ["\n", "QUESTION"]
#stopping_criteria = StopWordCriteria(tokenizer=tokenizer, prompts=[], stop_words=stop_words)
#from greedy_probs import GreedyProbsCalculator
from prompts import combine_two_contexts, make_no_context_prompt, make_prompt

def convert_epsilon_to_text(d: dict):
    entity = f'{d["entity"]["label"]} is a {d["entity"]["description"]}\n'
    text = entity
    for n in d['neighbors']:
        label = f"{n['neighbor_label']} is a {n['neighbor_description']}\n"
        text += label
    return text

def generate_content(text):
    url = "http://10.11.1.8:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
            "model": "/archive/beliakin/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393/",
            "messages": [{
            "role": "user",
            "content": text}],
            "max_tokens": 100,
            "temperature": 0,
             "logprobs": True,
            "top_logprobs": 1,
            }
    
    response = requests.post(url, headers=headers, json=data)
    text = response.json()["choices"][0]["message"]["content"]
    return text, response.json()["choices"][0]["logprobs"]["top_logprobs"]

def calc_entropy(logprobs):
    entropies = []
    for s_lp in logprobs:
        entropies.append([])
        for lp in s_lp:
            mask = ~np.isinf(lp)
            entropies[-1].append(-np.sum(np.array(lp[mask]) * np.exp(lp[mask])))
    return entropies


ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
retrieved_ddg_path = '../../data/retrieve_to_models/truthfulqa_multichoice/duckduckgo.csv'
retrieved_google_path = '../../data/retrieve_to_models/truthfulqa_multichoice/google.csv'
retrieved_wikipedia_path = '../../data/retrieve_to_models/truthfulqa_multichoice/wikipedia.csv'
retrieved_wikidata_path = '../../data/retrieve_to_models/truthfulqa_multichoice/wikidata.csv'
retrieved_truthful_qa_mc = pd.read_csv(retrieved_ddg_path)
retrieved_truthful_qa_mc_add = pd.read_csv(retrieved_google_path)


stats = []
ue_metrics = []
correct_labels = []
for idx, row in tqdm(retrieved_truthful_qa_mc.iterrows()):
    question = row['question']
    contexts = []
    answers, labels = shuffle(ds[idx]['mc1_targets']['choices'], ds[idx]['mc1_targets']['labels'],random_state=0)
    correct_labels.append(labels)
    number_of_context = len(row.index.values) - 2
    number_of_second_context = len(retrieved_truthful_qa_mc_add.iloc[idx].index.values) - 2 
    for i in range (1, number_of_context + 1):
        # wikidata
        # if type(row[f'context_{i}']) is not str:
            # break
        # text = convert_epsilon_to_text(eval(row[f'context_{i}']))
        # contexts.append(text)
        
        # wikipedia
        # ctx = eval(row[f'context_{i}'])
        # if len(ctx) == 0:
        #     continue
        # elif ctx['is_summary'] is True:
        #     contexts.append(ctx['context'])
        # else:
        #     contexts.append(ctx['context'][:500])

        # ddg and google
        # contexts.append(row[f'context_{i}'])

        # several contexts
        if type(row[f'context_{i}']) is not str:
            break
        contexts.append(row[f'context_{i}'])
        google_index = random.randint(1, number_of_second_context)
        while type(retrieved_truthful_qa_mc_add.iloc[idx][f'context_{google_index}']) is not str:
            google_index = random.randint(1, number_of_second_context)
        contexts.append(retrieved_truthful_qa_mc_add.iloc[idx][f'context_{google_index}'])
    
    # for separate contexts
    # texts = [make_prompt(c, question, answers) for c in contexts]

    # for united contexts
    texts = [combine_two_contexts(contexts, question, answers)]
    
    # for empty context
    # texts.append(make_no_context_prompt(question, answers))
    cur_msp = []
    current_perplexity = []
    current_entropy = []
    for text in texts:
        generated_texts, log_likelihoods = generate_content(text)
        msp = np.array([-np.sum(log_likelihood) for log_likelihood in log_likelihoods])
        cur_msp.append(msp)
        perplexity = np.array([-np.mean(ll) for ll in log_likelihoods])
        current_perplexity.append(perplexity)
        mean_token_entropy = np.array([np.mean(e) for e in calc_entropy(log_likelihoods)])
        current_entropy.append(mean_token_entropy)
    stats.append(stat)
    ue_metrics.append({'msp': msp, 'perplexity': perplexity, 'entropy': mean_token_entropy})

df_src = []
for idx, s in enumerate(stats):
    answers = ''
    for v in s['greedy_texts']:
        # mistral
        answers += v[0]
        # llama
        answer += v.strip()
    df_src.append({'idx': idx, 'correct': chr(ord('A') + correct_labels[idx].index(1)), 'answers': answers,
               'msp': ue_metrics[idx]['msp'], 'perplexity': ue_metrics[idx]['perplexity'], 'entropy': ue_metrics[idx]['entropy']})

df = pd.DataFrame(df_src)
df.to_csv('../../data/model_answers/truthful_qa_mc/truthful_qa_multichoice_ddg_google_llama_70b.csv')
