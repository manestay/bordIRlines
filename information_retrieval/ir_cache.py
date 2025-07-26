import os
import sys
import argparse
import shutil

from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

sys.path.append('.')
from lib import load_borderlines_hf, get_territory_names

from ir_lib import load_paragraphs, get_languages_from_data


def get_chroma_path(entity_name, language, embedding):
    return f"./chroma_db/{entity_name.replace(' ','_').replace('/','_')}/{language}/{embedding.model}"


def run_caching_on_entity(entity_name, language, embedding, first_n):
    '''Caches Chroma embeddings of an entity and language.'''

    print(f'Entity Name (docs relevant to): {entity_name}')
    print(f'Lang (docs in this): {language}')
    print(f'First n paragraphs (to cache): {first_n}')

    # Load paragraphs from linked documents
    paragraphs = load_paragraphs(entity_name, language, first_n=first_n)
    docs = [Document(page_content=paragraph, metadata={"source":"local"}) for paragraph in paragraphs]

    if paragraphs:
        path = get_chroma_path(entity_name, language, embedding)

        # Don't cache if the path exists to avoid duplicate caching
        if os.path.exists(path):
            if os.path.exists(os.path.join(path, 'chroma.sqlite3')) and len(os.listdir(path)) >= 2:
                print("Dataset Exists")
                db_check = Chroma(persist_directory=path, embedding_function=embedding)
                if len(db_check) != len(paragraphs):
                    print(len(db_check), len(paragraphs))
                    print("Missing Paragraphs")
                    # The folder with missing paragraphs should be removed manually, and the script rerun
                    exit()
            return False
        
        # Cache the documents
        try:
            db = Chroma.from_documents(
                docs, 
                embedding, 
                persist_directory=path, 
                collection_metadata={"hnsw:space": "cosine"}
            )
            db.persist()
        except Exception as e:
            print("Exception:", e)
            return False
            
        print("Paragraphs:\n")
        for i, para in enumerate(paragraphs, 1):
            print(f"{i}: {para}\n")

    else:
        print(f"No paragraphs found for entity {entity_name}.")
    
    return True


def main():
    # 0 args: loop over all territories, over claimant langs for each
    # 1 arg: 1 territory, loop over claimant langs
    # 2 args: 1 territory, 1 lang
    # 3 args: 1 territory, 1 lang, user query
    parser = argparse.ArgumentParser(description='Information Retrieval System Caching')
    parser.add_argument('entity_name', nargs='?', type=str, help='Name of the entity')
    parser.add_argument('language', nargs='?', type=str, help='Language of the entity')
    parser.add_argument('--embedding', '-e', type=str, default='text-embedding-3-large', help='Type of embedding model to use')
    parser.add_argument('--first_n', '-n', type=int, default=-1, help='Number of paragraphs to load')
    args = parser.parse_args()

    entity_name = args.entity_name
    language = args.language
    first_n = args.first_n
    embedding = OpenAIEmbeddings(model=args.embedding)
    
    if not language or not entity_name: # load queries and entities from the BorderLines Dataset
        territories, countries, queries = load_borderlines_hf() # load queries and entities from the BorderLines Dataset
    
    # Get list of all languages under data/raw/wikipedia
    all_languages = get_languages_from_data()
    
    # Languages to loop over
    languages = [language] if language else all_languages

    # Check caching 5 times to recover from non-deterministic Chroma caching failures
    for i in range(5):
        # Run IR on each language and entity needed
        for curr_language in languages:

            # Get territory names for this language
            if not language or not entity_name:
                if not curr_language in queries:
                    continue
                query_ds = queries[curr_language]
                territory_names_with_query = get_territory_names(query_ds) # all territories with queries in this language
            territory_names = [entity_name] if entity_name else territory_names_with_query # the territories to loop over
            
            # If english, cache across every territory in the territories datasets
            if not entity_name and curr_language == "en":
                territory_names = territories['Territory']

            for curr_entity_name in territory_names:
                # Cache the top first_n paragraphs for this entity and language
                if not run_caching_on_entity(curr_entity_name, curr_language, embedding, first_n):
                    # If caching returns false due to an error and the cache is missing folders, remove the cache
                    path = get_chroma_path(curr_entity_name, curr_language, embedding)
                    if not os.path.exists(os.path.join(path, 'chroma.sqlite3')) or len(os.listdir(path)) < 2:
                        print("Removing", path)
                        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
