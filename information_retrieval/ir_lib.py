import json
import os
from functools import lru_cache
from typing import Sequence

from tqdm import tqdm

from lib import check_languages, get_territory_names

DEFAULT_DATA_DIR = "data/raw/wikipedia"
LLM_DATA_DIR = "data/raw/LLM"
ZH_LANGS = {"zhs", "zht"}


@lru_cache(maxsize=128)
def load_json_file(file_path, mode="r", encoding="utf-8"):
    with open(file_path, mode, encoding=encoding) as f:
        data = json.load(f)
    return data


def postprocess_paragraphs(paragraphs, paragraphs_ids, strip, dedup):
    if not strip and not dedup:
        return paragraphs, paragraphs_ids
    seen_text = {}
    paragraphs_new = []
    paragraphs_ids_new = []
    for para, para_id in zip(paragraphs, paragraphs_ids):
        if strip:
            para = para.strip()
        if dedup:
            if para in seen_text:
                # print(f'skipping duplicate paragraph with id "{para_id}", text: "{para}"')
                continue
            seen_text[para] = para_id
        if not para:
            # print(f'skipping empty paragraph with id "{para_id}"')
            continue
        paragraphs_new.append(para)
        paragraphs_ids_new.append(para_id)

    return paragraphs_new, paragraphs_ids_new


def load_paragraphs(
    entity_name: str,
    languages: str | Sequence[str],
    data_dir: str = DEFAULT_DATA_DIR,
    return_para_ids: bool = False,
    first_n: int = -1,
    isLLM: bool = False,
    strip: bool = True,
    dedup: bool = True,
    handle_zh: bool = False,
) -> list[str] | tuple[list[str], list[str]]:
    """Function to load paragraphs from linked documents for a given entity name and languages.

    Args:
        entity_name: Name of the entity.
        languages: Language(s) to use.
        data_dir: Directory containing the data files.
        return_para_ids: Whether to return paragraph IDs.
        first_n: Number of paragraphs to return for each document. If -1, return all paragraphs.
        isLLM: Whether the data is from LLM.
        strip: Whether to strip leading and trailing whitespaces from paragraphs.
        dedup: Whether to remove duplicate paragraphs.
        handle_zh: Whether to handle chinese IDs, `zhs` and `zht`, differently.
    Returns:
        paragraphs_all_langs: List of paragraphs for all languages.
        paragraphs_ids_all_langs: (optionally) List of paragraph IDs for all languages.
    """

    def load_paragraphs_lang(language: str):
        """Helper function to load paragraphs for a given entity name and one language."""
        metadata_file = os.path.join(data_dir, "metadata", f"entity2ids.{language}.json")

        # Check if the metadata file exists
        if not os.path.exists(metadata_file):
            print(f"Metadata file {metadata_file} not found.")
            return [], []

        metadata = load_json_file(metadata_file)
        ids = metadata.get(entity_name)

        paragraphs_all = []
        paragraphs_ids_all = []

        if not ids:
            print(f"Entity {entity_name} not found in metadata.")
            return [], []
        for doc_id in ids["ids"]:
            paragraphs = []
            paragraphs_ids = []
            num_added = 0
            if isLLM:
                doc_file = os.path.join(data_dir, language, f"data_{doc_id}.json")
            else:
                doc_file = os.path.join(data_dir, language, f"Q{doc_id}.json")
            if not os.path.exists(doc_file):
                print(f"Document file {doc_file} not found.")
                continue

            doc_data = load_json_file(doc_file)
            for sec_num, section in enumerate(doc_data["article"]):
                paragraphs.append(section)
                para_id = f"{doc_id}_p{sec_num}"
                if handle_zh and language in ZH_LANGS:
                    para_id = f"{language}.{para_id}"
                paragraphs_ids.append(para_id)
                num_added += 1
            if first_n != -1 and num_added >= first_n:
                paragraphs = paragraphs[:first_n]
                paragraphs_ids = paragraphs_ids[:first_n]
            paragraphs_all.extend(paragraphs)
            paragraphs_ids_all.extend(paragraphs_ids)

        paragraphs_all, paragraphs_ids_all = postprocess_paragraphs(
            paragraphs_all, paragraphs_ids_all, strip, dedup
        )
        return paragraphs_all, paragraphs_ids_all

    if isinstance(languages, str):
        languages = [languages]

    empty_ret = [] if not return_para_ids else ([], [])

    # Check if the data folder exists
    if isLLM:
        data_dir = LLM_DATA_DIR
    if not os.path.exists(data_dir):
        print(f"Data folder {data_dir} not found.")
        return empty_ret

    paragraphs_all_langs = []
    paragraphs_ids_all_langs = []
    for language in languages:
        paragraphs_all, paragraphs_ids_all = load_paragraphs_lang(language)
        paragraphs_all_langs.extend(paragraphs_all)
        paragraphs_ids_all_langs.extend(paragraphs_ids_all)

    if return_para_ids:
        return paragraphs_all_langs, paragraphs_ids_all_langs

    return paragraphs_all_langs


def get_relevant_langs(
    retrieval_over: str, lang: str, entity_name: str, territory_langs_map: dict[str, set[str]]
) -> set[str]:
    relevant_langs = []
    if retrieval_over == "qlang":
        relevant_langs = [lang]
    elif retrieval_over == "qlang_en":
        relevant_langs = [lang, "en"]
    elif retrieval_over == "en":
        relevant_langs = ["en"]
    elif retrieval_over == "rel_langs":
        relevant_langs = territory_langs_map[entity_name]
        relevant_langs.add("en")
    return set(relevant_langs)


def get_queries_ir(query_ds, territories_ds, lang, entity_name=""):
    if query_ds is None:  # handle control, all territories
        return {row["QueryID"]: row["Query"] for row in territories_ds}

    territory_names = get_territory_names(query_ds)
    qids = query_ds["QueryID"]
    if entity_name:
        if entity_name not in territory_names:
            return {}
        idx = territory_names.index(entity_name)
        query = query_ds[idx]["Query_Native"]
        queries_list, qids = [query], [query_ds["QueryID"][idx]]
    else:
        queries_list = query_ds["Query_Native"]

    queries_ir = {qid: query for qid, query in zip(qids, queries_list)}
    return queries_ir


def get_queries_ir_all(queries_ds, territories_ds, languages):
    queries_all = {}
    languages = check_languages(languages, queries_ds)
    for lang in tqdm(languages):
        query_ds_lang = queries_ds.get(lang)

        queries_ir = get_queries_ir(query_ds_lang, territories_ds, lang)
        queries_all.update(queries_ir)

    return queries_all


def get_languages_from_data():
    folders = []
    # Iterate over all items in the directory
    for item in os.listdir(DEFAULT_DATA_DIR):
        # Check if the item is a directory
        if os.path.isdir(os.path.join(DEFAULT_DATA_DIR, item)) and item != "metadata":
            # If it's a directory, append its name to the list
            folders.append(item)
    return folders
