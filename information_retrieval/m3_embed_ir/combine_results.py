from collections import defaultdict
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from pathlib import Path

from m3_lib import EvalArgs


@dataclass
class EvalArgsExt(EvalArgs):
    top_k: int = field(
        default=50,
        metadata={'help': 'Use reranker to rerank top-k retrieval results'}
    )
    search_result_save_paths: list[Path] = field(
        default=('m3_embed_ir/rerank_results/dense/bge-m3-bge-m3/search_results.txt ',),
        metadata={'help': 'List of paths for saved search results.`',
                  "nargs": "*"}
    )
    weights: list[float] = field(
        default=(),
        metadata={'help': 'Hybrid weight of scores',
                  "nargs": "*"}
    )
    hybrid_result_save_path: Path = field(
        default=Path('m3_embed_ir/search_results/hybrid_results.txt'),
        metadata={'help': 'Path to save hybrid search results.'}
    )


def get_search_result_dict(search_result_path: str, top_k: int = 1000):
    search_result_dict = {}
    flag = True

    for _, row in pd.read_csv(search_result_path, sep=' ', header=None).iterrows():
        qid = str(row.iloc[0])
        docid = row.iloc[2]
        rank = int(row.iloc[3])
        score = float(row.iloc[4])
        if qid not in search_result_dict:
            search_result_dict[qid] = []
            flag = False
        if rank > top_k:
            flag = True
        if flag:
            continue
        else:
            search_result_dict[qid].append((docid, score))
    return search_result_dict


def save_hybrid_results(result_dicts: dict[str, dict[str, list]], weights_dict: dict[str, float], hybrid_result_save_path: str, top_k: int = 1000):
    if not os.path.exists(os.path.dirname(hybrid_result_save_path)):
        os.makedirs(os.path.dirname(hybrid_result_save_path))

    # use dict with None values as an ordered set
    qids = {}
    for search_result_dict in result_dicts.values():
        qids.update({k: None for k in search_result_dict.keys()})
    qid_list = list(qids.keys())

    hybrid_results_list = []
    for qid in tqdm(qid_list, desc="Hybriding dense, sparse and colbert scores"):
        results = defaultdict(float)
        for path, result_dict in result_dicts.items():
            for docid, score in result_dict.get(qid):
                results[docid] += score * weights_dict[path]

        hybrid_results = [(docid, score) for docid, score in results.items()]
        hybrid_results.sort(key=lambda x: x[1], reverse=True)

        hybrid_results_list.append(hybrid_results[:top_k])

    with open(hybrid_result_save_path, 'w', encoding='utf-8') as f:
        for qid, hybrid_results in tqdm(zip(qid_list, hybrid_results_list), desc="Saving hybrid search results"):
            for rank, docid_score in enumerate(hybrid_results):
                docid, score = docid_score
                line = f"{qid} Q0 {docid} {rank+1} {score:.6f} hybrid"
                f.write(line + '\n')


def main():
    parser = HfArgumentParser([EvalArgsExt])
    eval_args = parser.parse_args_into_dataclasses()[0]
    eval_args: EvalArgs

    print("**************************************************")
    print(f"Start hybrid search results ...")

    hybrid_result_save_path = eval_args.hybrid_result_save_path

    result_dicts = {}
    for search_result_save_path in eval_args.search_result_save_paths:
        result_d = get_search_result_dict(search_result_save_path, top_k=eval_args.top_k)
        result_dicts[str(search_result_save_path)] = result_d

    paths_str = [str(x) for x in eval_args.search_result_save_paths]
    if len(eval_args.weights) <= 1:
        weight = 1 if not len(eval_args.weights) else eval_args.weights[0]
        weights_d = {x: weight for x in paths_str}
    elif len(eval_args.weights) == len(paths_str):
        weights_d = dict(zip(paths_str, eval_args.weights))
    else:
        print('invalid weights passed in!')
        os._exit(-1)

    print('weights_d:')
    for k, v in weights_d.items():
        print(f'{k} = {v}')

    save_hybrid_results(
        result_dicts=result_dicts,
        weights_dict=weights_d,
        hybrid_result_save_path=hybrid_result_save_path,
        top_k=eval_args.top_k,
    )

    print("==================================================")
    print(f'Finish generating reranked results, saved to: {hybrid_result_save_path}')


if __name__ == "__main__":
    main()
