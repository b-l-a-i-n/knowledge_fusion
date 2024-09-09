import pandas as pd
import wikipedia
from tqdm import tqdm
import concurrent.futures

wikipedia.set_lang("en")


def fetch_wikipedia_summary(question, num_results=10):
    """
    Fetch Wikipedia summaries for a given question.

    Args:
    question (str): The question to search for.
    num_results (int, optional): The number of results to fetch. Defaults to 10.

    Returns:
    list: A list of dictionaries containing the title, summary, and page content.
    """
    try:
        titles = wikipedia.search(question, results=num_results)
        context = []
        for title in titles:
            try:
                result = wikipedia.page(title)
                context.append({"title": title, "summary": result.summary, "page": result.content})
            except Exception as e:
                print(f"Error fetching page for title '{title}': {e}")
        return context
    except Exception as e:
        print(f"Error searching for question '{question}': {e}")
        return []


def fetch_all_summaries(questions):
    """
    Fetch summaries for a list of questions using multithreading.

    Args:
    questions (list): A list of questions to fetch summaries for.

    Returns:
    pd.DataFrame: A DataFrame containing the questions, raw context, and context.
    """
    summaries = []

    with tqdm(total=len(questions)) as pbar:
        def process_question(question):
            context = fetch_wikipedia_summary(question)
            if not context:
                print(f"No context found for question '{question}'")
                context = []
            pbar.update(1)
            return {
                "question": question,
                "raw_context": context,
                "context": {i: c.get("summary", c["page"]) for i, c in enumerate(context)}
            }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_question, question) for question in questions]
            for future in concurrent.futures.as_completed(futures):
                summaries.append(future.result())

    return pd.DataFrame(summaries)


if __name__ == "__main__":
    datasets = ["truthfulqa_gen", "truthfulqa_multichoice", "mmlu"]
    
    for dataset in datasets:
        input_path = f"/home/admin/graphs/knowledge_fusion/data/{dataset}_questions.csv"
        save_path = f"/home/admin/graphs/knowledge_fusion/data/{dataset}_wikipedia.csv"

        df = pd.read_csv(input_path)
        wikipedia_results_df = fetch_all_summaries(df["question"].values)
        wikipedia_results_df.to_csv(save_path, index=False)
