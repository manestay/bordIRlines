import argparse
import sys
from pathlib import Path

from transformers import AutoConfig

from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from ir_lib import load_paragraphs, get_queries_ir

from lib import load_borderlines_hf, check_languages, split_qid

parser = argparse.ArgumentParser()
parser.add_argument('--entity_name', '-e')
parser.add_argument('--out_name', '-o', type=Path, default='data/ir_beir')
parser.add_argument('--languages', nargs='*', default=['all'])
parser.add_argument('--query', help='specific query to use (for testing)')
parser.add_argument('--first_n', '-n', type=int, default=100,
                    help='Number of paragraphs to load')

ENTITY = 'Falkland Islands'
LANG = 'en'
all_dataset_names = None

def get_mdpr_model_name(lang):
    MODEL_BASENAME = 'castorini/mdpr-tied-pft-msmarco'
    ft_name = f'{MODEL_BASENAME}-ft-miracl-{lang}'
    try:
        model_name = AutoConfig.from_pretrained(ft_name, use_auth_token=False)
        print(ft_name)
    except OSError:
        model_name = MODEL_BASENAME
        print(f'`{lang}` finetune of {MODEL_BASENAME} not found, using base')
    return model_name

if __name__ == "__main__":
    args = parser.parse_args()

    queries_all = {}
    if args.query:
        query = args.query
        queries_all = {'one': query}
    else:
        territories_ds, countries_ds, queries_ds = load_borderlines_hf()

        languages = check_languages(args.languages, queries_ds)

        for lang in languages:
            query_d_lang = queries_ds.get(lang)

            queries_ir = get_queries_ir(query_d_lang, territories_ds, lang, args.entity_name)
            queries_all.update(queries_ir)

    for qid, query in queries_all.items():
        print(f'Query `{qid}`: {query}')
        entity_name, lang = split_qid(qid)
        print(f'Entity Name (docs relevant to): {entity_name}')
        print(f'Lang (docs in this): {lang}')

        paragraphs, paragraphs_ids = load_paragraphs(entity_name, lang, return_para_ids=True)
        para_d = {pid: {'text': p} for pid, p in zip(paragraphs_ids, paragraphs)}
        if not para_d:
            print(f'no contexts, skipping')
            continue
        one_query_d = {qid: query}

        ## dense retrieval
        model = DRES(models.SentenceBERT('paraphrase-multilingual-mpnet-base-v2'), batch_size=128)
        retriever = EvaluateRetrieval(model, score_function="cos_sim")
        results = retriever.retrieve(para_d, one_query_d)
        import pdb; pdb.set_trace()
