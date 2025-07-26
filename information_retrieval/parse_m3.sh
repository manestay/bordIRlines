for mode in en qlang qlang_en rel_langs; do
    echo "Parsing search results for $mode"
    results_name=m3_embed_ir/rerank_results/over_$mode/colbert+sparse+dense/bge-m3-bge-m3/search_results.txt
    python parse_search_results.py $results_name -r $mode -o m3_embed_ir/rerank_results/search_results_over_$mode.json -n 50
done
