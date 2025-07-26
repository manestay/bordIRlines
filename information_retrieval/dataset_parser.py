"""
Parses a results JSON file from an IR system, and saves a directory to be used with a hf datasets loading script.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.append("..")
from lib import EN_TERRS, split_qid

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_json", "-i", type=str, required=True, help="Path to the results JSON file."
)
parser.add_argument(
    "--retrieval_over", "-r", type=str, required=True, help="The mode of retrieval over docs."
)
parser.add_argument("--source", "-s", type=str, default="openai", help="The name of the IR system.")
parser.add_argument(
    "--overwrite",
    action=argparse.BooleanOptionalAction,
    help="Overwrite existing files.",
)
parser.add_argument(
    "--load_queries",
    action=argparse.BooleanOptionalAction,
    help="Load queries, instead of writing a new one.",
)


def sort_doc_id(entry):
    # sort by multi-index (article_id, paragraph_id)
    key = entry[0].split(".")[-1].split("_p")[0]
    return [int(x) for x in key]


def get_lang_doc_map(ir_data):
    lang_doc_map = defaultdict(dict)
    # Definitions:
    # an article consists of multiple paragraphs
    # a paragraph is a doc (doc to be consistent with prior work)

    for key, value in ir_data.items():
        terr, lang_code = key.rsplit("_", 1)

        for hit in value["hits"]:
            lang_hit = hit["language"]
            doc_id_text_map = lang_doc_map[lang_hit]

            doc_id = hit["pid"]
            if doc_id not in doc_id_text_map:
                doc_id_text_map[doc_id] = hit["text"]
    return lang_doc_map


def to_row_str(row, delimiter="\t"):
    return delimiter.join(str(x) for x in row) + "\n"


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.input_json, "r") as file:
        ir_data = json.load(file)

    root_folder = "data"
    source = args.source
    retrieval_mode = args.retrieval_over

    langs = set()
    query_id_text_map = {}

    for i, (key, value) in enumerate(ir_data.items(), 1):
        # make folders for each language
        territory, lang_code = split_qid(key)

        mode_folder = os.path.join(root_folder, lang_code, source, retrieval_mode)
        os.makedirs(mode_folder, exist_ok=True)

        # create a mapping of query_id to query_text
        query_id = f"Q{i}"
        query_text = value["query"]
        if not args.load_queries:
            query_id_text_map[query_id] = (query_text, territory, lang_code)

        langs.add(lang_code)

    mode_folder = os.path.join(root_folder, "control", source, retrieval_mode)
    os.makedirs(mode_folder, exist_ok=True)
    langs.add("control")

    query_mapping_path = os.path.join(root_folder, "queries.tsv")
    if args.load_queries:
        query_mapping_path = os.path.join(root_folder, "queries.tsv")
        with open(query_mapping_path, "r") as query_mapping_file:
            query_mapping_file.readline()
            for line in query_mapping_file:
                query_id, query_text = line.strip().split("\t")
                query_id_text_map[query_id] = (query_text, territory, lang_code)
        print(f"Loaded {len(query_id_text_map)} queries from {query_mapping_path}.")
    else:
        with open(query_mapping_path, "w", newline="") as query_mapping_file:
            query_mapping_file.write("query_id\tquery_text\tterritory\tlanguage\n")

            for query_id, (query_text, territory, lang_code) in query_id_text_map.items():
                query_mapping_file.write(f"{query_id}\t{query_text}\t{territory}\t{lang_code}\n")

        print(
            f"Created {query_mapping_path} with {len(query_id_text_map)} unique sequential query IDs and texts."
        )

    query_text_id_map = {v: k for k, v in query_id_text_map.items()}
    lang_doc_map = get_lang_doc_map(ir_data)

    # save the docs for each language
    all_docs_d = defaultdict(dict)
    doc_json_path = os.path.join(root_folder, "all_docs.json")
    if os.path.exists(doc_json_path) and not args.overwrite:
        with open(doc_json_path, "r") as json_file:
            existing_doc_d = json.load(json_file)
        all_docs_d.update(existing_doc_d)

    for lang_code, doc_d in lang_doc_map.items():
        doc_d = dict(sorted(doc_d.items(), key=sort_doc_id))
        if all_docs_d[lang_code]:
            loaded = True
            num_docs = len(all_docs_d[lang_code])
        else:
            loaded = False
        all_docs_d[lang_code].update(doc_d)
        num_docs_new = len(all_docs_d[lang_code])

        if loaded:
            diff = max(num_docs_new - num_docs, 0)
            print(f"Loaded {num_docs} docs, added {diff} docs for {lang_code}.")
        else:
            print(f"Wrote {num_docs_new} docs for {lang_code}.")

    all_docs_d["control"] = {}
    with open(doc_json_path, "w") as json_file:
        json.dump(all_docs_d, json_file, ensure_ascii=False, indent=2)

    # save the query hits for each language
    lang_file_map = {
        lang_code: open(
            os.path.join(
                root_folder, lang_code, source, retrieval_mode, f"{lang_code}_query_hits.tsv"
            ),
            "w",
        )
        for lang_code in langs
    }

    for lang_file in lang_file_map.values():
        lang_file.write("query_id\tterritory\trank\tscore\tdoc_id\tdoc_lang\n")

    for key, value in ir_data.items():
        territory, lang_code = key.rsplit("_", 1)
        if lang_code == "en":
            lang_code = "control"

        query_hits_file = lang_file_map[lang_code]

        query_id = query_text_id_map.get(value["query"])

        for hit in value["hits"]:
            doc_id = hit["pid"]
            doc_lang = hit["language"]
            row = [query_id, territory, hit["rank"], hit["score"], doc_id, doc_lang]
            row_str = to_row_str(row)
            query_hits_file.write(row_str)

            if key in EN_TERRS:
                en_file = lang_file_map["en"]
                en_file.write(row_str)

    for lang_file in lang_file_map.values():
        lang_file.close()

    print("Query hits TSV files created for each language with query and doc details.")
