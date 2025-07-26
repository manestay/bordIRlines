"""
This script provides functionality for counting the number of paragraphs (documents)
in each language of the Wikipedia dataset for each entity, and saves results to a 
json file.

You can run the script with this command from the main folder:
python3 information_retrieval/query_lang_count.py --retrieval_over rel_langs
"""
import json
import sys
import argparse
import os
from typing import List

from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

sys.path.append('.')
from lib import get_territory_langs_map, load_borderlines_hf, get_territory_names

from ir_lib import get_relevant_langs, load_paragraphs, get_languages_from_data

output_file_path = 'information_retrieval/query_lang_counts.json'

# Load the Chroma DB from cache, or generate a new one if from_cache is false
def get_lang_counts(
        entity_name: str, 
        languages: List[str], 
        embedding: str, 
        from_cache: bool = True, 
        first_n: int = 20, 
        isLLM: bool = False
):
    lang_counts = dict()

    if from_cache:
        for language in languages:
            persist_dir = f"./chroma_db/{entity_name.replace(' ','_').replace('/','_')}/{language}/{embedding.model}"
            print(f'loading db from {persist_dir}')
            db = Chroma(persist_directory=persist_dir, 
                        embedding_function=embedding,
                        collection_metadata={"hnsw:space": "cosine"})
            
            # Fetch the number of documents (paragraphs) in the db
            db_data = db._collection.get(include=['documents'])
            paragraph_count = len(db_data['documents'])
            lang_counts[language] = paragraph_count
    else:
        print(f'creating vector db locally (may be slow)...')
        for language in languages:
            # Load paragraphs directly if not in cache
            paragraphs = load_paragraphs(entity_name, language, first_n=first_n, isLLM = isLLM)
            if not paragraphs:
                print(f"No paragraphs found for entity {entity_name}.")
                lang_counts[language] = 0
                continue
            lang_counts[language] = len(paragraphs)

    return lang_counts


def run_lang_counting(entity_name, language, query, embedding, from_cache=True, first_n=20, k=10, isLLM=False, relevant_langs=None):
    """
    Function to run an IR system
    """
    languages = relevant_langs or [language]

    print(f'Query: {query}')
    print(f'Entity Name (docs relevant to): {entity_name}')
    print(f'Lang (docs in this): {", ".join(languages)}')

    print("\nGetting Chroma database and logging document counts")
    lang_counts = get_lang_counts(entity_name, languages, embedding, from_cache=from_cache, first_n=first_n, isLLM=isLLM)
    return lang_counts


def save_results(
        lang_counts, 
        query_id, 
):
    # Check if the output file already exists and load existing data
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as file:
            data = json.load(file)
    else:
        data = {}
    
    # Update the dictionary with the new query_id and lang_counts
    data[query_id] = lang_counts
    
    # Save the updated data back to the JSON file
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Saved language counts for query ID {query_id} to {output_file_path}")


def main():
    # 0 args: loop over all territories, over claimant langs for each
    # 1 arg: 1 territory, loop over claimant langs
    # 2 args: 1 territory, 1 lang
    # 3 args: 1 territory, 1 lang, user query
    parser = argparse.ArgumentParser(description='Information Retrieval System')
    parser.add_argument('entity_name', nargs='?', type=str, help='Name of the entity')
    parser.add_argument('language', nargs='?', type=str, help='Language of the entity')
    parser.add_argument('query', nargs='?', default=None, help='Query to search for (optional)')
    parser.add_argument('--embedding', '-e', type=str, default='text-embedding-3-large', help='Type of embedding model to use')
    parser.add_argument('--first_n', '-n', type=int, default=20, help='Number of paragraphs to load from database (if from cache is false)')
    parser.add_argument('--from_cache', '-c', action='store_true', default=True, help='Whether to load Chroma db from cache')
    parser.add_argument('--store_results', '-s', action='store_true', default=True, help='Whether to store paragraphs in txt file')
    parser.add_argument('--isLLM', action='store_true', default=False, help='Flag to indicate if LLM based documents are used.')
    parser.add_argument('--retrieval_over', type=str, default='qlang', 
                        choices=['qlang', 'qlang_en', 'en', 'rel_langs'], 
                        help=('Specify the languages to retrieve over (will be overriden if language is already specified). Options: '
                            'qlang (query language), '
                            'qlang_en (query language and English), '
                            'en (English), '
                            'rel_langs (relevant territory languages and English).'))
    args = parser.parse_args()

    entity_name = args.entity_name
    language = args.language
    query = args.query
    first_n = args.first_n
    from_cache = args.from_cache
    embedding = OpenAIEmbeddings(model=args.embedding)
    store_results = args.store_results
    isLLM = args.isLLM
    retrieval_over = args.retrieval_over if not language else None
    
    # Clear output file if it exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # load queries and entities from the BorderLines Dataset
    if not query or not language or not entity_name:
        territories, countries, queries = load_borderlines_hf()
        countries_info = {x["Country"]: x for x in countries}
        territory_langs_map = get_territory_langs_map(territories, countries_info)

    # Get list of all languages under data/raw/wikipedia
    all_languages = get_languages_from_data()
    
    # Languages to loop over
    languages = [language] if language else all_languages
    
    num_queries = 0
    seen_query_ids = set()
    
    # Run IR on each language and entity needed
    for curr_language in languages:

        # Get territory names with a query for this language
        territory_names_with_query = []
        if not query or not language or not entity_name:
            if not curr_language in queries:
                continue
            query_ds = queries[curr_language]
            territory_names_with_query = get_territory_names(query_ds) # all territories with queries in this language
        
        territory_names = [entity_name] if entity_name else territory_names_with_query # the territories to loop over

        # Loop over territories from queries dataset
        for curr_entity_name in territory_names:
             # Load a query if not provided
            if not query:
                if curr_entity_name not in territory_names_with_query: # if the entity doesn't have a matching query
                    print(f'[INFO]: {entity_name} not found in query for {curr_language}')
                    continue # skip entity if no queries found in this language
                idx = territory_names_with_query.index(curr_entity_name)
                curr_query = query_ds[idx]['Query_Native']
                curr_query_id = query_ds[idx]['QueryID']
                seen_query_ids.add(curr_query_id)
            else:
                curr_query = query
                
            num_queries += 1

            relevant_langs = get_relevant_langs(
                retrieval_over, curr_language, curr_entity_name, territory_langs_map
            ) if territory_langs_map else None

            lang_counts = run_lang_counting(curr_entity_name, curr_language, curr_query, embedding, from_cache, first_n, k=50, isLLM=isLLM, relevant_langs=relevant_langs)
            
            if (store_results):
                save_results(lang_counts, curr_query_id)
        
        # Loop over territories from territories dataset if language is english, and no entity name provided
        if curr_language == "en" and not entity_name:
            for territory_item in territories:
                if territory_item['QueryID'] not in seen_query_ids:
                    curr_entity_name = territory_item['Territory']
                    curr_query = territory_item['Query']
                    curr_query_id = territory_item['QueryID']
                    num_queries += 1

                    relevant_langs = get_relevant_langs(
                        retrieval_over, curr_language, curr_entity_name, territory_langs_map
                    ) if territory_langs_map else None
                    
                    lang_counts = run_lang_counting(curr_entity_name, curr_language, curr_query, embedding, from_cache, first_n, k=50, isLLM=isLLM, relevant_langs=relevant_langs)
                    
                    if (store_results):
                        save_results(lang_counts, curr_query_id)
                
    # Check that the number of queries is 720
    print("Num queries", num_queries)


if __name__ == "__main__":
    main()
