import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")
from ir_lib import ZH_LANGS, get_queries_ir, get_relevant_langs, load_paragraphs

from lib import check_languages, get_territory_langs_map, load_borderlines_hf, split_qid

parser = argparse.ArgumentParser()
parser.add_argument("search_file", type=Path)
parser.add_argument("--entity_name", "-e")
parser.add_argument("--languages", nargs="*", default=["all"])
parser.add_argument("--docs_dir", "-d", type=Path, default=Path("../data/raw/wikipedia"))
parser.add_argument("--num_hits", "-n", type=int, default=-1)
parser.add_argument("--out_path", "-o", type=Path)
parser.add_argument("--retrieval_over", "-r", type=str, default="rel_langs")


def parse_search_lines(lines, qids):
    qids = set(qids)
    search_d = defaultdict(list)
    for line in tqdm(lines, desc="parse search lines"):
        arr = line.split(" ")
        qid = arr[0]
        if qid not in qids:
            continue
        search_d[qid].append((arr[2], arr[3], arr[4]))
    return search_d


def handle_zh_pid(pid, pid_to_lang, seen_pids, curr_lang):
    # heuristic function to disambiguate a pid from Chinese, as they are shared
    # TODO: disambiguate between zht and zhs pids upstream, so this is no longer needed
    pid_zhs = f"zhs.{pid}"
    pid_zht = f"zht.{pid}"
    zhs_exists = pid_zhs in pid_to_lang
    zht_exists = pid_zht in pid_to_lang

    if zht_exists and zhs_exists:
        if curr_lang == "zht":
            curr_pid = pid_zht
            other_pid = pid_zhs
        else:
            curr_pid = pid_zhs
            other_pid = pid_zht
        pid = curr_pid if curr_pid not in seen_pids else other_pid
    elif zhs_exists:
        pid = pid_zhs
    elif zht_exists:
        pid = pid_zht
    else:
        print("case 2")
        import pdb

        pdb.set_trace()
    return pid


if __name__ == "__main__":
    args = parser.parse_args()

    queries_all = {}

    territories_ds, countries_ds, queries_ds = load_borderlines_hf()

    languages = check_languages(args.languages, queries_ds)

    countries_info = {x["Country"]: x for x in countries_ds}
    territory_langs_map = get_territory_langs_map(territories_ds, countries_info)

    for lang in languages:
        query_d_lang = queries_ds.get(lang)

        queries_ir = get_queries_ir(query_d_lang, territories_ds, lang, args.entity_name)

        queries_all.update(queries_ir)

    with args.search_file.open() as f:
        lines = f.readlines()
        search_d = parse_search_lines(lines, queries_all.keys())

    if not args.out_path:
        for qid, hits in search_d.items():
            query = queries_all[qid]
            print(f"Query `{qid}`: {query}")
            entity_name, lang = split_qid(qid)

            paragraph_d = dict()

            relevant_langs = get_relevant_langs(
                args.retrieval_over, lang, entity_name, territory_langs_map
            )

            for language in relevant_langs:
                paragraphs, paragraphs_ids = load_paragraphs(
                    entity_name,
                    language,
                    data_dir=args.docs_dir,
                    return_para_ids=True,
                    dedup=False,
                    handle_zh=True,
                )
                paragraph_d.update(zip(paragraphs_ids, paragraphs))

            if args.num_hits != -1:
                hits = hits[: args.num_hits]

                for i, hit in enumerate(hits, 1):
                    print(f"Hit {i} (weight: {hit[2]}):")
                    print(paragraph_d[hit[0]])
                    print()
                input()
    else:
        out_d = {}

        for qid, hits in tqdm(search_d.items(), desc="processing hits"):
            query = queries_all[qid]
            entity_name, lang = split_qid(qid)

            paragraph_d = dict()
            pid_to_lang = dict()

            relevant_langs = get_relevant_langs(
                args.retrieval_over, lang, entity_name, territory_langs_map
            )

            for rel_lang in relevant_langs:
                paragraphs, paragraphs_ids = load_paragraphs(
                    entity_name,
                    rel_lang,
                    data_dir=args.docs_dir,
                    return_para_ids=True,
                    dedup=False,
                    handle_zh=True,
                )
                paragraph_d.update(zip(paragraphs_ids, paragraphs))
                pid_to_lang.update(zip(paragraphs_ids, [rel_lang] * len(paragraphs_ids)))

            hits_l = []
            if args.num_hits != -1:
                hits = hits[: args.num_hits]

            seen_pids = set()

            for i, hit in enumerate(hits, 1):
                score = float(hit[2])
                if score == 0:
                    continue
                pid = hit[0]
                if pid not in paragraph_d:
                    pid = handle_zh_pid(pid, pid_to_lang, seen_pids, lang)

                hits_d = {
                    "pid": pid,
                    "language": pid_to_lang[pid],
                    "rank": int(hit[1]),
                    "score": score,
                    "text": paragraph_d[pid],
                }
                hits_l.append(hits_d)
                seen_pids.add(pid)
            out_d[qid] = {"query": query, "nhits": len(hits_l), "hits": hits_l}

        with args.out_path.open("w") as f:
            json.dump(out_d, f, indent=2, ensure_ascii=False)
