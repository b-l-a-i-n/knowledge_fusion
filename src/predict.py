import os
import json
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

tqdm.pandas()


MODEL = os.getenv("MODEL", "")
PROMPT = """
{additional_context}
Question: '{question}'
Options:
"""


def get_context(df, include_graph=True, include_description=True, use_web_search=False, top_k_web=-1):
    additional_context = ""
    if use_web_search:
        search_results = df["web_search"].iloc[0]
        if search_results is not np.nan:
            search_results = "\n".join(search_results.split("\n")[:top_k_web])
            additional_context += (
                f"\nBelow are the facts that might be relevant to answer the question:\n{search_results}"
                "\nIf there is no relevant fact, rely on your knowledge or choose a more likely option."
            )
        
    decode_dict = {}
    items = zip(df["answerEntity"], df["answerEntityId"], df["linearized_graph"], df["description"], df["sample_id"])
    message = PROMPT.format(question=df["question"].iloc[0], additional_context=additional_context)
    for idx, (answer, wiki_id, graph, description, sample_id) in enumerate(items):
        decode_dict[idx] = sample_id
        data = {"answer": answer, "WikiDataID": wiki_id}
        if include_graph:
            data["WikiDataGraph"] = graph
        if include_description:
            data["description"] = description
        message += str(idx) + ". " + json.dumps(data) + "\n"
    
    return message, decode_dict


def linearize_graph(graph):
    graph = eval(graph)
    nodes = {node["id"]: f"{node["label"]} ({node["type"]}, {node["name_"]})" for node in graph["nodes"]}
    edges = []
    for link in graph["links"]:
        source_node = nodes[link["source"]]
        target_node = nodes[link["target"]]
        label = link["label"]
        edges.append(f"{source_node} - {label} -> {target_node}")
    graph_result = '"' + '; '.join(edges) + '"'
    instruct = (
        "Transform this wiki graph into text, "
        "write only the new string that contains the text representation of the graph:\n"
    )
    text_result = generate_content(instruct + graph_result)
    result = text_result.replace("\n", " ").replace("  ", " ")
    return result


def generate_content(text):
    url = "http://10.11.1.11:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a helpful assistant.
You must follow the rules before answering:
- A question and its answer options will be provided.
- The question has only one correct option.
- The correct answer is always given.
- Write only the number of the correct option.
- If you do not know the answer, write only the number of the most likely one."""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "max_tokens": 256,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 256,
    }
    
    response = requests.post(url, headers=headers, json=data)
    text = response.json()["choices"][0]["message"]["content"]
    return text, response.json()["choices"][0]["logprobs"]["top_logprobs"]


def predict(row):
    message, decode_dict = get_context(row)
    attempts = 5
    res = False
    while not res:
      try:
        text, probs = generate_content(message)
        res = True
      except Exception as e:
        print(e)
        time.sleep(20 - attempts)
        attempts -= 1

      if attempts == 0:
        text = "Failed"
        probs = []
        res = True
    return text, probs, decode_dict


def test(df):
    all_rows = []
    for question, row in tqdm(df.groupby("question")):
        text, probs, match = predict(row)
        all_rows.append(
            {
                "questionEntityId": row["questionEntityId"].values[0],
                "prediction": text,
                "prob": probs, 
                "match_dict": match
            }
        )
    predictions = pd.DataFrame(data=all_rows)
    return predictions


if __name__ == "__main__":
    test_file_path = "TextGraphs17-shared-task/data/tsv/test.tsv"
    train_file_path = "TextGraphs17-shared-task/data/tsv/train.tsv"

    # Load DataFrames
    data_frames = {
        "train": pd.read_csv(train_file_path, sep="\t"),
        "test": pd.read_csv(test_file_path) #, sep="\t"),
    }

    # Load web search results
    id2ws = {}
    ws_df = pd.read_csv("data/ddgo_search_results_updated.csv", index_col=0)
    for idx, row in ws_df.iterrows():
        id2ws[row["question"]] = row["web_search_response"]

    # Load descriptions
    id2d = {}
    df_descriptions = pd.read_csv("data/wikidata_descriptions.csv", index_col=0)
    for idx, row in df_descriptions.iterrows():
        id2d[row["answerEntityId"]] = row["description"]

    # Apply transformations to both train and test DataFrames
    for key in data_frames:
        df = data_frames[key]
        df["web_search"] = df["question"].apply(lambda x: id2ws[x] if x not in ["NF", "ND"] else "")
        df["description"] = df["answerEntityId"].apply(lambda x: id2d[x])
        # TODO: replace
        df["linearized_graph"] = df["graph"].progress_apply(linearize_graph)

    # Make predictions
    for key in data_frames:
        df = data_frames[key]
        pred = test(df)
        pred.to_csv(f"output/llama_70B_{key}_ds_rawgr_probs.tsv", sep="\t", index=False)
