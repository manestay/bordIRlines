"""
Adapted from https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB/MKQA
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint

import datasets
import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
from transformers import HfArgumentParser
from utils.normalize_text import normalize

sys.path.append(".")
sys.path.append("..")

from ir_lib import get_queries_ir_all, get_relevant_langs, load_paragraphs
from m3_lib import EvalArgs, ModelArgs

from lib import get_territory_langs_map, load_borderlines_hf, split_qid


@dataclass
class ModelArgsExt(ModelArgs):
    fp16: bool = field(default=True, metadata={"help": "Use fp16 in inference?"})


@dataclass
class EvalArgsExt(EvalArgs):
    max_passage_length: int = field(default=512, metadata={"help": "Max passage length."})
    max_query_length: int = field(default=512, metadata={"help": "Max query length."})
    docs_dir: Path = field(default=Path("../data/raw/wikipedia"), metadata={"help": "dir for docs"})


def get_model(model_args: ModelArgs):
    model = BGEM3FlagModel(
        model_args.encoder,
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.fp16,
    )
    return model


def parse_corpus(corpus: list[dict]):
    corpus_list = []
    for data in corpus:
        _id = str(data["_id"])
        content = data["text"].lower()
        content = normalize(content)
        corpus_list.append({"id": _id, "content": content})

    corpus = datasets.Dataset.from_list(corpus_list)
    return corpus


def generate_indexes(
    model: BGEM3FlagModel,
    corpus: datasets.Dataset,
    max_passage_length: int = 512,
    batch_size: int = 256,
):
    model_outputs = model.encode(
        corpus["content"],
        batch_size=batch_size,
        max_length=max_passage_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vectors = model_outputs["dense_vecs"]
    sparse_vectors = model_outputs["lexical_weights"]

    dim = dense_vectors.shape[-1]

    faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    dense_vectors = dense_vectors.astype(np.float32)
    faiss_index.train(dense_vectors)
    faiss_index.add(dense_vectors)

    # process sparse
    docids = list(corpus["id"])
    encoded_corpus_list = []
    for docid, vector in zip(docids, sparse_vectors):
        for key, value in vector.items():
            vector[key] = int(np.ceil(value * 100))

        encoded_corpus_list.append({"id": docid, "contents": "", "vector": vector})

    return faiss_index, encoded_corpus_list, docids


def encode_and_save_queries(
    queries_save_path: str,
    model: BGEM3FlagModel,
    queries_d: dict[str],
    max_query_length: int = 512,
    batch_size: int = 256,
):
    qids = list(queries_d.keys())
    vectors = model.encode(
        list(queries_d.values()),
        batch_size=batch_size,
        max_length=max_query_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )["lexical_weights"]

    encoded_queries_list = []
    for qid, vector in zip(qids, vectors):
        for key, value in vector.items():
            vector[key] = int(np.ceil(value * 100))

        topic_str = []
        for token in vector:
            topic_str += [str(token)] * vector[token]
        if len(topic_str) == 0:
            topic_str = "0"
        else:
            topic_str = " ".join(topic_str)
        encoded_queries_list.append(f"{str(qid)}\t{topic_str}")

    with open(queries_save_path, "w", encoding="utf-8") as f:
        for line in tqdm(encoded_queries_list, desc="Saving encoded queries"):
            f.write(line + "\n")


def save_result(index: faiss.Index, docid: list, index_save_dir: str):
    docid_save_path = index_save_dir / "docid"
    index_save_path = index_save_dir / "index"
    with open(docid_save_path, "w", encoding="utf-8") as f:
        for _id in docid:
            f.write(str(_id) + "\n")
    faiss.write_index(index, str(index_save_path))


def main():
    parser = HfArgumentParser([ModelArgsExt, EvalArgsExt])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgsExt
    eval_args: EvalArgsExt

    model = get_model(model_args=model_args)

    encoder = model_args.encoder
    if os.path.basename(encoder).startswith("checkpoint-"):
        encoder = os.path.dirname(encoder) + "_" + os.path.basename(encoder)

    print("==================================================")
    print("Start generating sparse+dense embeddings with model:")
    print(model_args.encoder)

    index_save_dir = eval_args.index_save_dir / os.path.basename(encoder)

    # get all queries for all langs
    territories_ds, countries_ds, queries_ds = load_borderlines_hf()
    countries_info = {x["Country"]: x for x in countries_ds}
    queries_all = get_queries_ir_all(queries_ds, territories_ds, eval_args.languages)
    territory_langs_map = get_territory_langs_map(territories_ds, countries_info)

    # for each query, save its corpus's dense+sparse embeddings
    for qid, query in tqdm(queries_all.items()):
        entity_name, lang = split_qid(qid)
        print(f"Query `{qid}`: {query}")

        relevant_langs = get_relevant_langs(
            eval_args.retrieval_over, lang, entity_name, territory_langs_map
        )
        print(f"  Retrieving over langs: {relevant_langs}")

        index_save_dir_qid = index_save_dir / qid
        index_save_dir_qid.mkdir(exist_ok=True, parents=True)
        sparse_corpus_path = index_save_dir_qid / "corpus_embd.tsv"

        if (
            (index_save_dir_qid / "index").exists()
            and sparse_corpus_path.exists()
            and not eval_args.overwrite
        ):
            print(f"  Embeddings for {index_save_dir_qid} already exist, skipping...")
            continue

        # process corpus
        paragraphs, paragraphs_ids = load_paragraphs(
            entity_name, relevant_langs, data_dir=eval_args.docs_dir, return_para_ids=True
        )
        corpus = [{"_id": pid, "text": para} for pid, para in zip(paragraphs_ids, paragraphs)]

        corpus = parse_corpus(corpus)

        dense_index, sparse_index, docid = generate_indexes(
            model=model,
            corpus=corpus,
            max_passage_length=eval_args.max_passage_length,
            batch_size=eval_args.batch_size,
        )

        print(f"Saving {len(corpus)} dense embeddings to: {index_save_dir_qid}")
        save_result(dense_index, docid, index_save_dir_qid)

        with open(sparse_corpus_path, "w", encoding="utf-8") as f:
            for line in tqdm(sparse_index, desc="Saving sparse embeddings"):
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("Generate sparse query embeddings")
    queries_save_path = index_save_dir / "sparse_queries.tsv"
    if queries_save_path.exists():
        print("Already exists, skipping")
    else:
        print(f"Saving {len(queries_all)} sparse query embeddings to: {queries_save_path}")
        encode_and_save_queries(
            queries_save_path=queries_save_path,
            model=model,
            queries_d=queries_all,
            max_query_length=eval_args.max_query_length,
            batch_size=eval_args.batch_size,
        )

    print("==================================================")
    print("Finish generating embeddings with following model:")
    pprint(model_args.encoder)


if __name__ == "__main__":
    main()
