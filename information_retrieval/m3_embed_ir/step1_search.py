import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint

import torch
from pyserini.output_writer import OutputFormat, get_output_writer
from pyserini.search.faiss import AutoQueryEncoder, FaissSearcher
from tqdm import tqdm
from transformers import HfArgumentParser, is_torch_npu_available

sys.path.append(".")
sys.path.append("..")
from ir_lib import get_queries_ir_all
from m3_lib import EvalArgs, ModelArgs

from lib import load_borderlines_hf, split_qid


@dataclass
class ModelArgsExt(ModelArgs):
    add_instruction: bool = field(default=False, metadata={"help": "Add query-side instruction?"})


@dataclass
class EvalArgsExt(EvalArgs):
    result_save_dir: Path = field(
        default="./search_results",
        metadata={
            "help": "Dir to saving search results. Search results will be saved to `result_save_dir/{encoder_name}/{lang}.txt`"
        },
    )
    threads: int = field(default=16, metadata={"help": "Maximum threads to use during search"})
    hits: int = field(default=100, metadata={"help": "Number of hits"})


def get_query_encoder(model_args: ModelArgs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif is_torch_npu_available():
        device = torch.device("npu")
    else:
        device = torch.device("cpu")
    model = AutoQueryEncoder(
        encoder_dir=model_args.encoder,
        device=device,
        pooling=model_args.pooling_method,
        l2_norm=model_args.normalize_embeddings,
    )
    return model


def save_result(search_results, result_save_path: str, qids: list, max_hits: int):
    output_writer = get_output_writer(
        result_save_path,
        OutputFormat(OutputFormat.TREC.value),
        "w",
        max_hits=max_hits,
        tag="Faiss",
        topics=qids,
        use_max_passage=False,
        max_passage_delimiter="#",
        max_passage_hits=1000,
    )
    with output_writer:
        for topic, hits in search_results:
            # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
            # Remove the query from the results.
            hits = [hit for hit in hits if hit.docid != topic]

            output_writer.write(topic, hits)


def run_dense_search(queries_all, index_save_dir, query_encoder, eval_args):
    search_results = []
    # for each query, search over dense of associated corpus
    for qid, query in tqdm(queries_all.items()):
        entity_name, lang = split_qid(qid)
        print(f"Query `{qid}`: {query}")

        index_save_dir_qid = index_save_dir / qid

        # process dense
        searcher = FaissSearcher(index_dir=index_save_dir_qid, query_encoder=query_encoder)

        # batch size always == 1, as each query has its own database
        queries, qids = [query], [qid]

        for start_idx in tqdm(
            range(0, len(queries), eval_args.batch_size),
            desc="Searching",
            disable=len(queries) <= eval_args.batch_size,
        ):
            batch_queries = queries[start_idx : start_idx + eval_args.batch_size]
            batch_qids = qids[start_idx : start_idx + eval_args.batch_size]
            batch_search_results = searcher.batch_search(
                queries=batch_queries, q_ids=batch_qids, k=eval_args.hits, threads=eval_args.threads
            )
            search_results.extend([(_id, batch_search_results[_id]) for _id in batch_qids])

    return search_results


def generate_sparse_index(
    corpus_embd_dir: str, index_save_dir: str, threads: int = 12, language: str = "en"
):
    # NOTE: --pretokenized means that language isn't used!
    language = language[:2]
    cmd = f'python -m pyserini.index.lucene \
--language {language} \
--collection JsonVectorCollection \
--input "{corpus_embd_dir}" \
--index "{index_save_dir}" \
--generator DefaultLuceneDocumentGenerator \
--threads {threads} \
--impact --pretokenized --optimize'

    index_save_dir.mkdir(parents=True, exist_ok=True)
    with (index_save_dir / "generate_index.log").open("w") as f:
        try:
            subprocess.run(cmd, shell=True, stdout=f, check=True)
        except subprocess.CalledProcessError as e:
            print(e)


def search_sparse_results(
    index_save_dir: str, query_row: str, batch_size: int = 32, threads: int = 12, hits: int = 1000
):
    with (index_save_dir / "search_index.log").open("w") as f, tempfile.NamedTemporaryFile(
        "w+", suffix=".tsv"
    ) as fquery, tempfile.NamedTemporaryFile("w+") as fout:
        fquery.write(query_row)
        fquery.flush()
        cmd = f'python -m pyserini.search.lucene \
--index "{index_save_dir}" \
--topics {fquery.name} \
--output {fout.name} \
--output-format trec \
--batch {batch_size} \
--threads {threads} \
--hits {hits} \
--impact'
        try:
            subprocess.run(cmd, shell=True, stdout=f, check=True)
        except subprocess.CalledProcessError as e:
            print(e)

        fout.seek(0)
        results = fout.read().strip().split("\n")
        return results


def run_sparse_search(queries_all, index_save_dir, queries_save_path, eval_args):
    with open(queries_save_path) as f:
        lines = f.readlines()
        qid2line = {}
        for line in lines:
            qid2line[line.split("\t", 1)[0]] = line

    # for each query, search over dense of associated corpus
    results_all = []
    for qid, query in tqdm(queries_all.items()):
        _, lang = split_qid(qid)
        print(f"Query `{qid}`: {query}")

        corpus_embd_dir_qid = index_save_dir / qid
        index_save_dir_qid = corpus_embd_dir_qid / "sparse_index"

        if (index_save_dir_qid / "segments_1").exists() and not eval_args.overwrite:
            print("Index for sparse results already exists. Skip...")
            pass
        else:
            generate_sparse_index(
                corpus_embd_dir=corpus_embd_dir_qid,
                index_save_dir=index_save_dir_qid,
                threads=eval_args.threads,
                language=lang,
            )

        results = search_sparse_results(
            index_save_dir=index_save_dir_qid,
            query_row=qid2line[qid],
            batch_size=eval_args.batch_size,
            threads=eval_args.threads,
            hits=eval_args.hits,
        )
        results_all.extend(results)
    return results_all


def main():
    parser = HfArgumentParser([ModelArgsExt, EvalArgsExt])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgsExt
    eval_args: EvalArgsExt

    query_encoder = get_query_encoder(model_args=model_args)

    encoder = model_args.encoder
    if os.path.basename(encoder).startswith("checkpoint-"):
        encoder = os.path.dirname(encoder) + "_" + os.path.basename(encoder)

    index_save_dir = eval_args.index_save_dir / os.path.basename(encoder)
    if not index_save_dir.exists():
        raise FileNotFoundError(f"{index_save_dir} not found")

    # get all queries for all langs
    territories_ds, countries_ds, queries_ds = load_borderlines_hf()
    queries_all = get_queries_ir_all(queries_ds, territories_ds, eval_args.languages)

    print("==================================================")
    print("Start generating dense search results with model:", encoder)

    dense_save_path = eval_args.result_save_dir / os.path.basename(encoder) / "dense_results.txt"
    dense_save_path.parent.mkdir(exist_ok=True, parents=True)

    if os.path.exists(dense_save_path) and not eval_args.overwrite:
        print("Search for dense results already exists. Skip...")
    else:
        dense_results = run_dense_search(queries_all, index_save_dir, query_encoder, eval_args)
        qids = list(queries_all.keys())

        save_result(
            search_results=dense_results,
            result_save_path=dense_save_path,
            qids=qids,
            max_hits=eval_args.hits,
        )

    print("==================================================")
    print("Start generating sparse search results with model:", encoder)

    sparse_save_path = eval_args.result_save_dir / os.path.basename(encoder) / "sparse_results.txt"
    sparse_save_path.parent.mkdir(exist_ok=True, parents=True)

    if os.path.exists(sparse_save_path) and not eval_args.overwrite:
        print("Search for sparse results already exists. Skip...")
    else:
        queries_save_path = index_save_dir / "sparse_queries.tsv"
        sparse_results = run_sparse_search(
            queries_all, index_save_dir, queries_save_path, eval_args
        )

        with sparse_save_path.open("w") as f:
            for line in sparse_results:
                f.write(line + "\n")

    print("==================================================")
    print("Finish generating search results with following model:")
    pprint(model_args.encoder)
    print(f"dense: {dense_save_path}")
    print(f"sparse: {sparse_save_path}")


if __name__ == "__main__":
    main()
