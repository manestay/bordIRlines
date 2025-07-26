"""
This module processes queries and stores information in a JSON file.

The information stored includes:
- query_id: The unique identifier for each query.
- territories: The territories corresponding to each query.
- languages: The languages of each territory.

The main functionality of this module is to parse the queries and extract the necessary information to be stored in a structured JSON format.
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from information_retrieval.ir_lib import get_relevant_langs
from lib import (
    get_territory_claimants_and_langs_map,
    get_territory_langs_map,
    get_territory_names,
    load_borderlines_hf,
)


def parse_queries(output_file: str) -> None:
    """
    Parses the queries and stores the necessary information in a JSON file.

    Parameters:
    - query_file (str): Path to the file containing the queries.
    - output_file (str): Path to the output JSON file.
    """
    territories, countries, queries = load_borderlines_hf()
    countries_info = {x["Country"]: x for x in countries}
    territory_langs_map = get_territory_langs_map(territories, countries_info)
    territory_claimants_map = get_territory_claimants_and_langs_map(territories, countries_info)
    languages = set(queries.keys())

    query_dict = {}
    seen_query_ids = set()

    for row in territories:  # process the control dataset
        curr_query_id = row["QueryID"]
        curr_entity_name = row["Territory"]
        curr_language = "en"
        territory_entry = territory_claimants_map[curr_entity_name]
        relevant_langs = get_relevant_langs(
            "rel_langs", curr_language, curr_entity_name, territory_langs_map
        )
        territory_d = {
            "query": row["Query"],
            "query_lang": curr_language,
            "territory": curr_entity_name,
            "claimants": row["Claimants"],
            "claimants_native": row["Claimants"],
            "claimant_langs": list(territory_entry["claimant_langs"]),
            "relevant_langs": list(relevant_langs),
        }
        query_dict[curr_query_id] = territory_d

    for curr_language in languages:  # process the language datasets
        query_ds = queries[curr_language]
        territory_names = get_territory_names(query_ds)

        for curr_entity_name in territory_names:
            territory_entry = territory_claimants_map[curr_entity_name]
            idx = territory_names.index(curr_entity_name)
            curr_query_id = query_ds[idx]["QueryID"]
            seen_query_ids.add(curr_query_id)

            relevant_langs = get_relevant_langs(
                "rel_langs", curr_language, curr_entity_name, territory_langs_map
            )

            territory_d = {
                "query": query_ds[idx]["Query_Native"],
                "query_lang": curr_language,
                "territory": curr_entity_name,
                "claimants": territory_entry["claimants"],
                "claimants_native": query_ds[idx]["Claimants_Native"],
                "claimant_langs": list(territory_entry["claimant_langs"]),
                "relevant_langs": list(relevant_langs),
            }
            if curr_language == "en":
                # en queries are already processed in control, double check
                assert query_dict[curr_query_id] == territory_d
                continue
            query_dict[curr_query_id] = territory_d

    print("Num queries", len(query_dict))  # should be 720

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(query_dict, f, ensure_ascii=False, indent=4)

    print(f"Queries information saved to {output_file}")


def main():
    output_file = "generation/gen_results/queries.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    parse_queries(output_file)


if __name__ == "__main__":
    main()
