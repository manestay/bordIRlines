"""Generates a response table.

Aggregates responses from the rows for the 720 queries into rows for the 251 territories. Used
for calculating the CS metrics from BorderLines.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from data_lib import LETTERS

sys.path.append("..")
from lib import join_into_qid, load_borderlines_hf, split_qid

LETTER2NUM = {x: i for i, x in enumerate(LETTERS)}
PAD = " " * 20

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", "-i", type=Path, required=True)
parser.add_argument("--out_path", "-o", type=Path)
parser.add_argument("--response_dir", "-rd", type=Path)
parser.add_argument("--dataset_dir", "-dd", type=Path, help="path to dataset saved locally")
parser.add_argument("--quiet", "-q", action="store_true")
parser.add_argument("--qid_path", "-qid", type=Path)

if __name__ == "__main__":
    args = parser.parse_args()

    if not args.out_path:
        args.out_path = args.input_path.parent / f"{args.input_path.stem}.csv"

    # load territories df (which we will annotate with the aggregated responses)
    territories, countries, queries = load_borderlines_hf(args.dataset_dir)
    country2code = {entry["Country"]: entry["Lang_Code"] for entry in countries}

    territories = territories.map(
        lambda row: {
            "Claimant_Codes": [country2code[c] for c in row["Claimants"]],
            "Controller_Code": country2code[row["Controller"]]
            if row["Controller"] in country2code
            else "",
        }
    )
    territories = territories.to_pandas().set_index("Territory")
    claimants_map = territories["Claimants"].to_dict()
    control_lang_map = territories["Controller_Code"].to_dict()

    response_df = pd.read_json(args.input_path, orient="index")
    response_df["Territory"], response_df["Query_Lang"] = zip(*response_df.index.map(split_qid))

    # if qid_path was passed, filter out the responses that are not in the qid_path
    if args.qid_path:
        len_qids = len(response_df)
        with open(args.qid_path) as f:
            qids = json.load(f)
        response_df = response_df[response_df.index.isin(qids)]
        print(f"Using {len(response_df)}/{len_qids} responses")

    # get the en response for each territory
    response_df["Response"] = response_df.apply(
        lambda row: claimants_map[row["Territory"]][row["claimant_ind"]]
        if row["claimant_ind"] != -1
        else "None",
        axis=1,
    )

    # annotate territory df with the en responses
    territories["Response_en"] = territories.index.map(
        lambda x: response_df["Response"].get(join_into_qid(x, "en"))
    )
    territories["Responses_d"] = [{} for _ in range(len(territories))]

    # Vectorized operations for language picking logic
    response_df["Pick"] = response_df.apply(
        lambda row: claimants_map[row["Territory"]][0]
        if row["Response"] == "None"
        else row["Response"],
        axis=1,
    )

    response_df["Control_Lang"] = response_df["Territory"].map(control_lang_map)

    def process_row(row):
        pick = row["Pick"]
        control_lang = row["Control_Lang"]
        query_lang = row["Query_Lang"]

        if not control_lang:
            return "unk", country2code[pick] == query_lang
        elif control_lang != query_lang:
            return "non_control", country2code[pick] == control_lang
        else:
            return "control", country2code[pick] == control_lang

    response_df[["type", "result"]] = response_df.apply(process_row, axis=1, result_type="expand")

    picked_unk = response_df[response_df["type"] == "unk"]["result"].tolist()
    picked_non_control = response_df[response_df["type"] == "non_control"]["result"].tolist()
    picked_control = response_df[response_df["type"] == "control"]["result"].tolist()

    # annotate territories with the responses in all languages
    for row in response_df.itertuples():
        territory = row.Territory
        query_lang = row.Query_Lang
        territories.loc[territory, "Responses_d"][query_lang] = row.Pick

    territories["Response_Controller"] = territories.apply(
        lambda row: row["Responses_d"].get(row["Controller_Code"]), axis=1
    )
    territories["Unique_Claimants"] = territories["Responses_d"].map(lambda x: set(x.values()))

    territories_save = territories.copy()
    territories_save["Claimants"] = territories_save["Claimants"].str.join(";")
    territories_save["Claimant_Codes"] = territories_save["Claimant_Codes"].str.join(";")
    territories_save.drop(["Query"], axis=1, inplace=True)

    args.out_path.parent.mkdir(exist_ok=True, parents=True)

    territories_save.to_csv(args.out_path, index=False)
    print(f"saved to {args.out_path}")

    if not args.quiet:
        ds = territories[territories.Unique_Claimants.apply(len) != 1]

        print("\nOverall")
        print("-" * 10)
        territories_kno = territories[territories["Controller"] != "Unknown"]
        print("known:", (territories_kno["Controller"] == territories_kno["Response_en"]).mean())

        groups = territories.groupby("Region")
        for region, territoriesg in groups:
            print(region)
            print("-" * 10)
            territoriesg_kno = territoriesg[territoriesg["Controller"] != "Unknown"]
            print(
                "known:", (territoriesg_kno["Controller"] == territoriesg_kno["Response_en"]).mean()
            )
