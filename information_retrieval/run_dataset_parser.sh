#!/bin/zsh
for mode in qlang qlang_en en rel_langs; do
# for mode in qlang ; do
    echo "Processing ${mode} retrieval mode"

    python dataset_parser.py -i openai_embed/openai_results_${mode}_50.json -r ${mode}

    python dataset_parser.py -i m3_embed_ir/rerank_results/search_results_over_$mode.json -r ${mode} -s m3 --load_queries
done
