import copy
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import pandas as pd
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
    reranker: str = field(default="BAAI/bge-m3", metadata={"help": "Name or path of reranker"})


@dataclass
class EvalArgsExt(EvalArgs):
    max_length: int = field(default=512, metadata={"help": "Max text length."})
    batch_size: int = field(default=24, metadata={"help": "Inference batch size."})
    top_k: int = field(
        default=50, metadata={"help": "Use reranker to rerank top-k retrieval results"}
    )
    search_result_save_dir: Path = field(
        default=Path("./output_results"),
        metadata={
            "help": "Dir to saving search results. Search results path is `result_save_dir/{encoder}/{lang}.txt`"
        },
    )
    rerank_result_save_dir: Path = field(
        default=Path("./rerank_results"),
        metadata={
            "help": "Dir to saving reranked results. Reranked results will be saved to `rerank_result_save_dir/{encoder}-{reranker}/{lang}.txt`"
        },
    )
    num_shards: int = field(default=1, metadata={"help": "num of shards"})
    shard_id: int = field(default=0, metadata={"help": "id of shard, start from 0"})
    cuda_id: int = field(default=0, metadata={"help": "CUDA ID to use. -1 means only use CPU."})
    dense_weight: float = field(
        default=1, metadata={"help": "The weight of dense score when hybriding all scores"}
    )
    sparse_weight: float = field(
        default=0.3, metadata={"help": "The weight of sparse score when hybriding all scores"}
    )
    colbert_weight: float = field(
        default=1, metadata={"help": "The weight of colbert score when hybriding all scores"}
    )
    docs_dir: Path = field(default=Path("../data/raw/wikipedia"), metadata={"help": "dir for docs"})


def get_reranker(model_args: ModelArgs, device: str = None):
    reranker = BGEM3FlagModel(
        model_name_or_path=model_args.reranker,
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        device=device,
    )
    return reranker


def get_search_result_dict(search_result_path: Path, top_k: int = 100):
    search_result_dict = {}
    if not search_result_path.exists():
        print(f"{search_result_path} does not exist, skipping")
        return search_result_dict
    flag = True
    for _, row in pd.read_csv(search_result_path, sep=" ", header=None).iterrows():
        qid = str(row.iloc[0])
        docid = row.iloc[2]
        rank = int(row.iloc[3])
        if qid not in search_result_dict:
            search_result_dict[qid] = []
            flag = False
        if rank > top_k:
            flag = True
        if flag:
            continue
        else:
            search_result_dict[qid].append(docid)
    return search_result_dict


def get_queries_dict(queries_path: str):
    queries_dict = {}
    dataset = datasets.load_dataset("json", data_files=queries_path)["train"]
    for data in dataset:
        qid = str(data["id"])
        query = data["question"]
        queries_dict[qid] = query
    return queries_dict


def save_rerank_results(
    queries_dict: dict,
    corpus_dict: dict,
    reranker: BGEM3FlagModel,
    search_result_dict: dict,
    rerank_result_save_path: dict,
    batch_size: int = 256,
    max_length: int = 512,
    dense_weight: float = 1,
    sparse_weight: float = 1,
    colbert_weight: float = 1,
):
    qid_list = []
    sentence_pairs = []

    for qid, docids in search_result_dict.items():
        qid_list.append(qid)
        query = queries_dict[qid]
        for docid in docids:
            passage = corpus_dict[docid]
            sentence_pairs.append((query, passage))

    print(batch_size)
    scores_dict = reranker.compute_score(
        sentence_pairs,
        batch_size=batch_size,
        max_query_length=max_length,
        max_passage_length=max_length,
        weights_for_different_modes=[dense_weight, sparse_weight, colbert_weight],
    )

    for sub_dir, _rerank_result_save_path in rerank_result_save_path.items():
        os.makedirs(os.path.dirname(_rerank_result_save_path), exist_ok=True)

        scores = scores_dict[sub_dir]
        with open(_rerank_result_save_path, "w", encoding="utf-8") as f:
            i = 0
            for qid in qid_list:
                docids = search_result_dict[qid]
                docids_scores = []
                for j in range(len(docids)):
                    docids_scores.append((docids[j], scores[i + j]))
                i += len(docids)

                docids_scores.sort(key=lambda x: x[1], reverse=True)
                for rank, docid_score in enumerate(docids_scores):
                    docid, score = docid_score
                    line = f"{qid} Q0 {docid} {rank+1} {score:.6f} Faiss"
                    f.write(line + "\n")


def get_shard(search_result_dict: dict, num_shards: int, shard_id: int):
    if num_shards <= 1:
        return search_result_dict
    keys_list = sorted(list(search_result_dict.keys()))

    shard_len = len(keys_list) // num_shards
    if shard_id == num_shards - 1:
        shard_keys_list = keys_list[shard_id * shard_len :]
    else:
        shard_keys_list = keys_list[shard_id * shard_len : (shard_id + 1) * shard_len]
    shard_search_result_dict = {k: search_result_dict[k] for k in shard_keys_list}
    return shard_search_result_dict


def rerank_results(eval_args: EvalArgs, model_args: ModelArgs, reranker: BGEM3FlagModel):
    eval_args = copy.deepcopy(eval_args)
    model_args = copy.deepcopy(model_args)

    num_shards = eval_args.num_shards
    shard_id = eval_args.shard_id
    if shard_id >= num_shards:
        raise ValueError(f"shard_id >= num_shards ({shard_id} >= {num_shards})")

    if os.path.basename(model_args.encoder).startswith("checkpoint-"):
        model_args.encoder = (
            os.path.dirname(model_args.encoder) + "_" + os.path.basename(model_args.encoder)
        )

    if os.path.basename(model_args.reranker).startswith("checkpoint-"):
        model_args.reranker = (
            os.path.dirname(model_args.reranker) + "_" + os.path.basename(model_args.reranker)
        )

    # get all queries for all langs
    territories_ds, countries_ds, queries_ds = load_borderlines_hf()
    countries_info = {x["Country"]: x for x in countries_ds}
    queries_all = get_queries_ir_all(queries_ds, territories_ds, eval_args.languages)
    territory_langs_map = get_territory_langs_map(territories_ds, countries_info)

    encoder = model_args.encoder
    if os.path.basename(encoder).startswith("checkpoint-"):
        encoder = os.path.dirname(encoder) + "_" + os.path.basename(encoder)

    search_result_save_dir = eval_args.search_result_save_dir / os.path.basename(encoder)
    search_result_path_dense = search_result_save_dir / "dense_results.txt"

    print("==================================================")
    print("Start reranking results for each query")

    # map every paragraph ID to paragraph text
    corpus_dict = {}
    for qid, query in tqdm(queries_all.items(), desc="mapping IDs to paragraphs"):
        entity_name, lang = split_qid(qid)

        relevant_langs = get_relevant_langs(
            eval_args.retrieval_over, lang, entity_name, territory_langs_map
        )

        # process corpus
        paragraphs, paragraphs_ids = load_paragraphs(
            entity_name, relevant_langs, data_dir=eval_args.docs_dir, return_para_ids=True
        )
        para_dict = {pid: normalize(para.lower()) for pid, para in zip(paragraphs_ids, paragraphs)}
        corpus_dict.update(para_dict)

    search_result_dict = get_search_result_dict(search_result_path_dense, top_k=eval_args.top_k)

    search_result_dict = get_shard(search_result_dict, num_shards=num_shards, shard_id=shard_id)

    print(f"shard {shard_id+1}/{num_shards}, len={len(search_result_dict)}")

    if num_shards > 1:
        suffix = f"_{shard_id+1}_{num_shards}"
    else:
        suffix = ""

    rerank_result_save_path = {}
    for sub_dir in ["colbert", "sparse", "dense", "colbert+sparse+dense"]:
        _rerank_result_save_path = (
            eval_args.rerank_result_save_dir
            / sub_dir
            / f"{os.path.basename(model_args.encoder)}-{os.path.basename(model_args.reranker)}"
            / f"search_results{suffix}.txt"
        )
        rerank_result_save_path[sub_dir] = _rerank_result_save_path

    save_rerank_results(
        queries_dict=queries_all,
        corpus_dict=corpus_dict,
        reranker=reranker,
        search_result_dict=search_result_dict,
        rerank_result_save_path=rerank_result_save_path,
        batch_size=eval_args.batch_size,
        max_length=eval_args.max_length,
        colbert_weight=eval_args.colbert_weight,
        sparse_weight=eval_args.sparse_weight,
        dense_weight=eval_args.dense_weight,
    )


def main():
    parser = HfArgumentParser([EvalArgsExt, ModelArgsExt])
    eval_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: EvalArgsExt
    model_args: ModelArgsExt

    cuda_id = eval_args.cuda_id
    device = "cpu" if cuda_id == -1 else f"cuda:{cuda_id}"

    reranker = get_reranker(model_args=model_args, device=device)

    rerank_results(eval_args, model_args, reranker=reranker)

    print("==================================================")
    print("Finish generating reranked results with following model and reranker:")
    print(model_args.encoder)
    print(model_args.reranker)
    print(f"Retrieval mode: {eval_args.retrieval_over}")


if __name__ == "__main__":
    main()
