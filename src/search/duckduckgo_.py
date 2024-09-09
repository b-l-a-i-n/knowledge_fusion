import pandas as pd
from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def search_api_response(question, max_results=5):
    """Fetch search API response using DuckDuckGo."""
    try:
        results = DDGS().text(question, max_results=max_results)
        return results
    except Exception as e:
        raise RuntimeError(f"Error fetching results for '{question}': {e}")


def get_web_search_results(question, start_time=5, retry_attempts=5):
    """Try fetching web search results with retries on failure."""
    for attempt in range(retry_attempts):
        try:
            return search_api_response(question)
        except Exception as e:
            print(f"Error on attempt {attempt + 1} for question '{question}': {e}")
            import time
            time.sleep(start_time)
            start_time += 1
    raise Exception(f"Failed to fetch results after {retry_attempts} attempts for question: {question}")


def fetch_all_search_results(questions):
    """Fetch search results for all questions using multithreading."""
    web_search_results = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_web_search_results, question): question for question in questions}

        for future in tqdm(as_completed(futures), total=len(questions)):
            question = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Skipping question '{question}' due to repeated errors: {e}")
                result = []
            web_search_results.append({"question": question, "web_search": result})
    
    return pd.DataFrame(web_search_results)


def process_search_results(df_web_search):
    """Extract and process search results into a cleaner format."""
    web_search_text_responses = []

    for ws in df_web_search["web_search"].values:
        context = {idx + 1: r.get("body", "") for idx, r in enumerate(ws)} if ws else {}
        web_search_text_responses.append(context)

    df_web_search["context"] = web_search_text_responses
    return df_web_search


if __name__ == "__main__":
    datasets = ["truthfulqa_gen", "truthfulqa_multichoice", "mmlu"]

    for dataset in datasets:
        input_path = f"./data/{dataset}_questions.csv"
        save_path = f"./data/{dataset}_duckduckgo.csv"

        df = pd.read_csv(input_path)
        web_search_results_df = fetch_all_search_results(df["question"].values)
        processed_df = process_search_results(web_search_results_df)
        processed_df.to_csv(save_path, index=False)
