import time
import pandas as pd
from tqdm import tqdm
from duckduckgo_search import DDGS


def search_api_response(question, max_results=20):
    results = DDGS().text(question, max_results=max_results)
    return results


def get_web_search_results(question, start_time=5):
    res = False
    while not res:
      try:
        web_search_response = search_api_response(question)
        res = True
      except Exception as e:
        print(e)
        start_time += 1
        time.sleep(start_time)
          
    return web_search_response


if __name__ == "__main__":
    web_search_results = []
    save_path = "data/ddgo_search_results_rest.csv"
    df = pd.read_csv("TextGraphs17-shared-task/data/tsv/test.tsv")
    
    for question in tqdm(df["question"].values):
        web_search_context = get_web_search_results(question)
        web_search_results.append({"question": question, "web_search": web_search_context})
    df_web_search = pd.DataFrame(web_search_results)
    
    web_search_text_responses = []
    for ws in df_web_search["web_search"].values:
        text = ""
        for idx, r in enumerate(ws):
            body = r["body"]
            text += f"{idx + 1}. {body}\n"
        web_search_text_responses.append(text)
    df_web_search["web_search_response"] = web_search_text_responses
    df_web_search.to_csv(save_path)
