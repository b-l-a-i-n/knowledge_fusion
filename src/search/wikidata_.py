import requests
import pandas as pd
import wikipedia
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON


wikipedia.set_lang("en")


def get_one_hop_neighbors(entity_id, limit=20):
    """
    Get one-hop neighbors of a Wikidata entity along with the relations.
    
    :param entity_id: The Wikidata entity ID (e.g., Q42 for Douglas Adams).
    :param limit: The maximum number of results to return.
    :return: A list of dictionaries with neighbor IDs, labels, and the relation.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?neighbor ?neighborLabel ?p ?pLabel WHERE {{
        {{
            wd:{entity_id} ?p ?neighbor .
        }}
        UNION
        {{
            ?neighbor ?p wd:{entity_id} .
        }}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT {limit}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    
    neighbors_with_relations = []
    for result in results["results"]["bindings"]:
        neighbor_id = result["neighbor"]["value"].split("/")[-1]
        neighbor_label = result.get("neighborLabel", {}).get("value", "")
        relation_id = result["p"]["value"].split("/")[-1]
        relation_label = result.get("pLabel", {}).get("value", "No label available")

        neighbors_with_relations.append({
            'neighbor_id': neighbor_id,
            'neighbor_label': neighbor_label,
            'relation_id': relation_id,
            'relation_label': relation_label
        })

    return neighbors_with_relations


def search_wikidata_entities(search_term, language='en', limit=5):
    """
    Search for entities on Wikidata based on a label or alias.
    
    :param search_term: The term to search for (label or alias).
    :param language: The language to search in (default is English).
    :param limit: The maximum number of results to return.
    :return: A list of search results with entity IDs and labels.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': language,
        'search': search_term,
        'limit': limit
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error: Received status code {response.status_code}")
    
    data = response.json()

    results = []
    for item in data.get('search', []):
        entity_id = item.get('id')
        label = item.get('label')
        description = item.get('description', 'No description available')
        results.append({
            'id': entity_id,
            'label': label,
            'description': description
        })

    return results


def fetch_neighborhood(question, num_results=5):
    """
    Fetch Wikipedia entities and their one-hop neighbors.
    
    :param question: The search term.
    :param num_results: The maximum number of Wikipedia results.
    :return: A dictionary of entity neighborhoods.
    """
    titles = wikipedia.search(question, results=num_results)
    context = {}
    for title in titles:
        entities = search_wikidata_entities(title)
        for entity in entities:
            neighbors = get_one_hop_neighbors(entity['id'])
            context[entity['id']] = [entity] + neighbors
    return context


def fetch_all_neighborhoods(questions):
    """
    Fetch neighborhoods for all questions using multithreading.
    
    :param questions: A list of questions.
    :return: A DataFrame containing questions and their corresponding context.
    """
    summaries = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_neighborhood, question): question for question in questions}
        
        for future in tqdm(as_completed(futures), total=len(questions)):
            question = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Error for question '{question}': {e}")
                result = {}
            summaries.append({"question": question, "context": result})
    
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    datasets = ["truthfulqa_gen", "truthfulqa_multichoice", "mmlu"]
    
    for dataset in datasets:
        input_path = f"./data/{dataset}_questions.csv"
        save_path = f"./data/{dataset}_wikidata.csv"

        df = pd.read_csv(input_path)
        wikipedia_results_df = fetch_all_neighborhoods(df["question"].values)
        wikipedia_results_df.to_csv(save_path, index=False)
