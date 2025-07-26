import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Optional

import datasets
import tiktoken
import torch
from data_lib import LETTERS_MC, LLM_MAP, ModelDetails
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

sys.path.append("..")

from lib import ALL_LANGS, join_into_qid

TokenizerClass = PreTrainedTokenizer | tiktoken.Encoding
QueryDict = dict[str, dict[str, str | list]]
COUNTRIES_INFO = None
RAG_TEMPLATE = None
TOKENIZER = None


def get_chat_template():
    # used for commandr
    global RAG_TEMPLATE
    if RAG_TEMPLATE is None:
        with open("templates/rag_custom.jinja", "r") as f:
            RAG_TEMPLATE = f.read().strip()
    return RAG_TEMPLATE


def get_default_tokenizer():
    # used for OpenAI, default to the commandr tokenizer to get the chat template
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(LLM_MAP["commandr"])
    return TOKENIZER


def get_model_details(model_name: str) -> ModelDetails:
    model_id = LLM_MAP.get(model_name)
    if "gpt" in model_name:
        tokenizer = tiktoken.encoding_for_model(model_name)
        return ModelDetails(model_name, tokenizer, None, None)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.eval()
    model_details = ModelDetails(model_name, tokenizer, model, model.device)
    return model_details


def load_dataset(
    lang: str, split: str, num_paragraphs: int, relevance_filter: str
) -> datasets.Dataset:
    try:
        relevance_filter = relevance_filter or "all"
        ds = datasets.load_dataset(
            "borderlines/bordirlines",
            lang,
            split=split,
            n_hits=num_paragraphs,
            trust_remote_code=True,
            relevance_filter=relevance_filter,
            cache_dir=f"temp_{relevance_filter}",  # TODO: remove this once we fix relevance filter
        )
        return ds
    except ValueError as e:
        print(f"Warning for loading {lang} {split}: {e}")
        return None


def load_with_progress(func, args_list, num_threads=8, desc="Processing"):
    print(f"Running {desc} with {num_threads} threads")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(func, args): i for i, args in enumerate(args_list)}
        results = [None] * len(futures)  # Preallocate list to hold results in order
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            index = futures[future]
            results[index] = future.result()
    return results


def load_bordirlines(
    retrieval_over: str, num_paragraphs: int, embedder: str, relevance_filter: Optional[str] = None
) -> dict[str, dict]:
    def entry():
        return {"query": None, "nhits": 0, "hits": []}

    def set_or_assert_equal(d, key, value):
        if d.get(key) is not None:
            assert d[key] == value
        else:
            d[key] = value

    data = defaultdict(entry)

    if retrieval_over == "no_ir":
        return data

    langs = ["control"] + ALL_LANGS
    if "en" in langs:
        langs.remove("en")

    ds_list = []
    split = f"{embedder}.{retrieval_over}"
    print(f"Loading from HF, {split} split for {num_paragraphs} docs")
    func = partial(
        load_dataset,
        split=split,
        num_paragraphs=num_paragraphs,
        relevance_filter=relevance_filter,
    )
    ds_list = load_with_progress(func, langs, desc="Loading datasets")
    ds_list = [ds for ds in ds_list if ds is not None]
    ds = datasets.concatenate_datasets(ds_list)

    # organize rows into dict, where key is the query name, and hits field of value stores the aggregated rows

    for row in ds:
        qid = join_into_qid(row["territory"], row["query_lang"])
        row_d = {
            "language": row["doc_lang"],
            "pid": row["doc_id"],
            "rank": row["rank"],
            "score": row["score"],
            "text": row["doc_text"],
        }
        set_or_assert_equal(data[qid], "query", row["query"])
        data[qid]["hits"].append(row_d)
        data[qid]["nhits"] += 1

    for qid, row in data.items():
        assert row["nhits"] <= num_paragraphs
    return data


def load_json_files(directory: str, retrieval_type: str, num_paragraphs: int) -> dict:
    """
    Loads JSON files from the specified directory.

    Parameters:
    - directory (str): Path to the directory containing JSON files.
    - retrieval_type (str): Type of retrieval ('qlang', 'qlangen', 'rellang', 'en').
    - num_paragraphs (int): Number of paragraphs to retrieve (10 or 50).

    Returns:
    - data (dict): Dictionary containing data from the loaded JSON files.
    """
    print("WARNING: load_json_files() is deprecated, use load_bordirlines() instead")
    # TODO: update the ZIP with the proper names, and remove this
    if retrieval_type == "rel_langs":
        retrieval_type = "rellang"
    elif retrieval_type == "qlang_en":
        retrieval_type = "qlangen"
    elif retrieval_type == "no_ir":
        return {}

    # Constructing the filenames based on retrieval type and number of paragraphs
    json_files = [
        f"openai_results_{retrieval_type}_{num_paragraphs}.json",
        f"openai_results_{retrieval_type}_{num_paragraphs}.json",
    ]

    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")


def langcode2lang():
    global COUNTRIES_INFO
    if COUNTRIES_INFO is None:
        COUNTRIES_INFO = datasets.load_dataset("manestay/borderlines", "countries")["train"]

    lc2l = {}
    for entry in COUNTRIES_INFO:
        lc2l[entry["Lang_Code"]] = entry["Lang_Name"]
    lc2l["zhs"] = "Chinese"
    lc2l["zht"] = "Traditional Chinese"
    return lc2l


def add_lang_phrase(prompt, lang):
    if "|lang|" not in prompt:
        return prompt

    lang2l = langcode2lang()
    if lang == "en" or lang == "txt":
        replace_str = ""
    elif lang not in lang2l:
        print(f"WARNING: could not find code for {lang}")
        replace_str = ""
    else:
        replace_str = f" in {lang2l[lang]}"

    prompt = prompt.replace("|lang|", replace_str)
    return prompt


def match_one_letter(text: str) -> tuple[int, str]:
    # if text starts with a letter, return it
    if len(text) == 1:
        text += ")"
    for i, letter_mc in enumerate(LETTERS_MC):
        if text.startswith(letter_mc):
            return i, letter_mc

    # check if text contains only 1 letter
    letter_exists = [letter in text for letter in LETTERS_MC]
    if sum(letter_exists) == 1:
        letter_ind = letter_exists.index(True)
        return letter_ind, LETTERS_MC[letter_ind]
    else:
        return None, None


def get_user_prompt(
    documents: list,
    query_entry: dict[str, str | list],
    citation: str,
    use_docs: bool,
    tokenizer: TokenizerClass,
) -> str:
    if citation:
        assert use_docs, "Citation should only be used with documents"
        return get_user_prompt_citation(documents, query_entry, tokenizer)
    else:
        return get_user_prompt_direct(documents, query_entry, use_docs)


def replace_multiple(s, to_replace, replacement, n):
    for old in to_replace:
        s = s.replace(old, replacement, n)
    return s


def get_user_prompt_citation(
    documents: list, query_entry: dict[str, str | list], tokenizer: TokenizerClass
) -> str:
    multiple_choice = get_multiple_choice(query_entry["claimants_native"])
    options = "Options:\n" + "\n".join(multiple_choice)
    query = f"Query: {query_entry['query']}"
    chat_template = get_chat_template()

    tok_orig = tokenizer
    if isinstance(tok_orig, tiktoken.Encoding):
        tokenizer = get_default_tokenizer()

    conversation = [{"role": "user", "content": query}]

    documents_ = [{"text": doc} for doc in documents]
    tokenizer.chat_template = chat_template
    grounded_generation_prompt = tokenizer.apply_chat_template(
        conversation,
        documents=documents_,
        citation_mode="accurate",  # or "fast"
        tokenize=False,
        options=options,
        add_generation_prompt=True,
    )

    if isinstance(tok_orig, tiktoken.Encoding):
        # remove special tokens for use with OpenAI
        ggp = re.sub(r"<\|.*?\|>", "", grounded_generation_prompt)
        ggp = replace_multiple(
            ggp,
            ["<results>", "</results>", "<BOS_TOKEN>"],
            "\n",
            1,
        ).strip()
        grounded_generation_prompt = ggp
        # NOTE: the system prompt and user prompt are concatenated, we will split later
    return grounded_generation_prompt


def get_user_prompt_direct(
    documents: list, query_entry: dict[str, str | list], use_docs: bool
) -> str:
    query = query_entry["query"]

    prompt = ""
    # first load documents
    if use_docs:
        # Append each document to the prompt
        for i, doc in enumerate(documents):
            prompt += f"Document: {i}\nText: {doc}\n\n"
        source_str = "the provided documents and your prior knowledge"
    else:
        source_str = "your prior knowledge"
    prompt += f"Instruction: Considering {source_str}, answer the query. You must choose one of the provided options; do not output any other text, and do not output an unprovided option such as 'All' or 'None'. Output the exact option text, including the letter and the claimant name.\n\n"

    # Compose the prompt
    prompt += f"Query: {query}\n\n"
    multiple_choice = get_multiple_choice(query_entry["claimants_native"])

    prompt += "Options:\n" + "\n".join(multiple_choice) + "\n\nAnswer: "

    return prompt


def get_multiple_choice(items: list[str]) -> list[str]:
    return [f"{let} {item}" for let, item in zip(LETTERS_MC, items)]


def get_other_lang(query_lang: str, claimant_langs: list[str]) -> str:
    # Given some lang, returns the other language.
    # For 2 claimant languages, this is just the other language.
    # For >2, we choose the next lang in a circular manner.

    if query_lang not in claimant_langs:
        return None
    if len(claimant_langs) == 1:
        return query_lang

    index = claimant_langs.index(query_lang)

    index_next = (index + 1) % len(claimant_langs)
    other_lang = claimant_langs[index_next]
    assert other_lang != query_lang
    return other_lang


def clean_query_dict(query_dict: QueryDict) -> QueryDict:
    all_langs = set(ALL_LANGS)
    for _, query_entry in query_dict.items():
        claimant_langs = []
        seen = set()
        for lang in query_entry["claimant_langs"]:
            if lang in all_langs and lang not in seen:
                claimant_langs.append(lang)
                seen.add(lang)
        query_entry["claimant_langs"] = claimant_langs
    return query_dict
