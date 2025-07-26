import argparse
import json
import sys
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(".")
from lib import ALL_LANGS, join_into_qid, split_qid

parser = argparse.ArgumentParser()
parser.add_argument("--load1", action="store_true")
parser.add_argument("--load2", action="store_true")

OUT_PATH = "information_retrieval/data/"
DOCID2QID_IRMODE_PATH = OUT_PATH + "doc_id2qid_ir_mode.csv"
QID2PASSAGE_PATH = OUT_PATH + "qid2doc_id.csv"
QID2QSTR_PATH = OUT_PATH + "qid2qstr.json"


def as_json_str(obj):
    if isinstance(obj, set):
        return json.dumps(list(obj))
    return json.dumps(obj)


def sort_doc_id(doc_id):
    # sort by multi-index (article_id, paragraph_id)
    key = doc_id.split(".")[-1].split("_p")
    return [int(x) for x in key]


def get_query_mode_df(query_df: pd.DataFrame, mode: str):
    query_df_mode = query_df.copy()
    query_df_mode = query_df_mode.drop(columns=["doc_ids"])
    doc_ids_mode_col = []

    for row in query_df.itertuples():
        doc_ids = row.doc_ids
        # sample {'27421_p107': ['openai.qlang_en', 'openai.rel_langs', 'openai.qlang', 'openai.en'], '13885196_p0': ['openai.rel_langs', 'm3.qlang', 'm3.qlang_en', 'm3.rel_langs', 'openai.qlang', 'openai.qlang_en', 'openai.en', 'm3.en']}
        doc_ids_mode = defaultdict(list)
        # add all doc_ids that endswith mode
        for doc_id, ir_modes in doc_ids.items():
            for ir_mode in ir_modes:
                if ir_mode.endswith("." + mode):
                    doc_ids_mode[doc_id].append(ir_mode)

        doc_ids_mode_col.append(doc_ids_mode)

    query_df_mode["doc_ids"] = doc_ids_mode_col
    return query_df_mode


if __name__ == "__main__":
    args = parser.parse_args()
    ALL_LANGS.remove("en")
    ALL_LANGS.append("control")

    query_id2str = {}

    if not args.load1:
        passage_ds = {}
        it = tqdm(ALL_LANGS)
        for lang in it:
            it.set_description(lang)
            ds_lang = load_dataset("borderlines/bordirlines", lang)
            for ir_mode, ds_split in ds_lang.items():
                for row in ds_split:
                    query_id = row["query_id"]

                    lang_q = lang if lang != "control" else "en"
                    query_str = join_into_qid(row["territory"], lang_q)
                    if query_id not in query_id2str:
                        query_id2str[query_id] = query_str
                    else:
                        # make sure the mapping is consistent
                        assert query_id2str[query_id] == query_str

                    qid_ir_mode = (query_id, ir_mode)
                    if row["doc_id"] not in passage_ds:
                        passage_ds[row["doc_id"]] = {
                            "doc_id": row["doc_id"],
                            "doc_lang": row["doc_lang"],
                            "qid_ir_mode": {qid_ir_mode},
                            # "doc_text": row["doc_text"],
                        }
                    else:
                        assert qid_ir_mode not in passage_ds[row["doc_id"]]["qid_ir_mode"]
                        passage_ds[row["doc_id"]]["qid_ir_mode"].add(qid_ir_mode)

        # save query_id to query_str mapping
        with open(QID2QSTR_PATH, "w") as f:
            json.dump(query_id2str, f, indent=2)

        ir_df = pd.DataFrame.from_dict(passage_ds, orient="index")
        qid_ir_mode = ir_df["qid_ir_mode"].copy()
        ir_df["qid_ir_mode"] = ir_df["qid_ir_mode"].apply(as_json_str)

        # sort by the int part of the query_id
        ir_df["doc_id_idx"] = ir_df["doc_id"].apply(sort_doc_id)
        ir_df = ir_df.sort_values("doc_id_idx")
        ir_df = ir_df.drop(columns=["doc_id_idx"])

        ir_df.to_csv(DOCID2QID_IRMODE_PATH, index=False)
        print(f"Saved {len(ir_df)} passages to {DOCID2QID_IRMODE_PATH}")

        ir_df["qid_ir_mode"] = qid_ir_mode  # restore original format
    else:
        with open(QID2QSTR_PATH, "r") as f:
            query_id2str = json.load(f)
        ir_df = pd.read_csv(DOCID2QID_IRMODE_PATH)
        ir_df["doc_id_idx"] = ir_df["doc_id"].apply(sort_doc_id)
        ir_df["qid_ir_mode"] = ir_df["qid_ir_mode"].apply(json.loads)
        print(f"Loaded {len(ir_df)} passages from {DOCID2QID_IRMODE_PATH}")

    if not args.load2:
        query_ds = {}
        for row in ir_df.itertuples():
            for qid, ir_mode in row.qid_ir_mode:
                terr, qlang = split_qid(query_id2str[qid])
                if qid not in query_ds:
                    query_ds[qid] = {
                        "query_id": qid,
                        "territory": terr,
                        "query_lang": qlang,
                    }
                    doc_id_dict = defaultdict(list)
                    doc_id_dict[row.doc_id] = [ir_mode]
                    query_ds[qid]["doc_ids"] = doc_id_dict
                else:
                    query_ds[qid]["doc_ids"][row.doc_id].append(ir_mode)

        query_df = pd.DataFrame.from_dict(query_ds, orient="index")
        doc_ids = query_df["doc_ids"].copy()
        query_df["doc_ids"] = query_df["doc_ids"].apply(as_json_str)

        # sort by the int part of the query_id
        query_df["query_id_int"] = query_df["query_id"].apply(lambda x: int(x[1:]))
        query_df = query_df.sort_values("query_id_int")
        query_df = query_df.drop(columns=["query_id_int"])

        query_df.to_csv(QID2PASSAGE_PATH, index=False)
        query_df["doc_ids"] = doc_ids  # restore original format
        # print avg number of passages per query
        print(f"Saved {len(query_df)} queries to {QID2PASSAGE_PATH}")
    else:
        query_df = pd.read_csv(QID2PASSAGE_PATH)
        query_df["doc_ids"] = query_df["doc_ids"].apply(json.loads)
        print(f"Loaded {len(query_df)} queries from {QID2PASSAGE_PATH}")

    mean_passages = query_df.doc_ids.apply(len).mean()
    print(f"Average # passages per query: {mean_passages:.3f}")

    for mode in ["qlang", "en"]:
        mode_path = QID2PASSAGE_PATH.replace(".csv", f"_irmode-{mode}.csv")

        query_mode_df = get_query_mode_df(query_df, mode)
        doc_ids = query_mode_df["doc_ids"].copy()
        query_mode_df["doc_ids"] = query_mode_df["doc_ids"].apply(as_json_str)

        query_mode_df.to_csv(mode_path, index=False)
        query_mode_df["doc_ids"] = doc_ids  # restore original format
        print(f"Saved {len(query_mode_df)} queries to {mode_path}")

        mean_passages = query_mode_df.doc_ids.apply(len).mean()
        print(f"Average # passages per query ({mode}): {mean_passages:.3f}")
