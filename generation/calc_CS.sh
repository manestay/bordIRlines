#!/bin/bash
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Define the possible choices for each argument
num_paragraph=10
role="vanilla"


SETTING=${1:-direct}

# if setting is "direct"
if [ $SETTING == "direct" ]; then
    retrieval_overs=("no_ir" "qlang" "qlang_en" "rel_langs" "en" "ablate_swap")
    llms=("gpt-4o" "gpt-4o-mini" "llama1b" "llama3b" "llama8b" "commandr" "commandr7b")
    mkdir -p gen_results/metrics
    for llm in "${llms[@]}"; do
        for retrieval_over in "${retrieval_overs[@]}"; do
            CONFIG="${retrieval_over}_${role}_${llm}_${num_paragraph}"
            STEM=${SCRIPT_DIR}/gen_results/generation_${CONFIG}
            METRICS=${SCRIPT_DIR}/gen_results/metrics/scores_${CONFIG}.json
            JSON_PATH=${STEM}.json
            CSV_PATH=${STEM}.csv
            python ${SCRIPT_DIR}/gen_response_table.py -i $JSON_PATH -q
            python ${SCRIPT_DIR}/../borderlines/calculate_CS.py $CSV_PATH -o ${METRICS} -q
        done
    done

    python ${SCRIPT_DIR}/gen_all_results_table.py -i gen_results/metrics/scores_*.json \
        -o gen_results/metrics/scores_for_all.csv --print_latex

elif [ $SETTING == "citation" ]; then
    retrieval_overs=("qlang" "qlang_en" "rel_langs" "en" "ablate_swap")
    llms_cite=("gpt-4o-mini" "commandr" "commandr7b")
    mkdir -p gen_results_cite/metrics
    for llm in "${llms_cite[@]}"; do
        for retrieval_over in "${retrieval_overs[@]}"; do
            CONFIG="${retrieval_over}_${role}_${llm}_${num_paragraph}"
            STEM=${SCRIPT_DIR}/gen_results_rel/gencitation_${CONFIG}
            METRICS=${SCRIPT_DIR}/gen_results_rel/metrics/scores_${CONFIG}.json
            JSON_PATH=${STEM}.json
            CSV_PATH=${STEM}.csv
            python ${SCRIPT_DIR}/gen_response_table.py -i $JSON_PATH -q
            python ${SCRIPT_DIR}/../borderlines/calculate_CS.py $CSV_PATH -o ${METRICS} -q
        done
    done

    python ${SCRIPT_DIR}/gen_all_results_table.py -i gen_results_cite/metrics/scores_*.json \
        -o gen_results_cite/metrics/scores_for_all.csv --print_latex

elif [ $SETTING == "relevance" ]; then
    retrieval_overs=("qlang" "qlang_en" "rel_langs" "en")
    llms_rel=("gpt-4o")

    METRICS_DIR=gen_results_rel/metrics/
    for folder in "all" "rel" "nrel" "no_ir" "metrics"; do
        mkdir -p gen_results_rel/${folder}/
    done

    JSON_PATHS=()

    for llm in "${llms_rel[@]}"; do
        for retrieval_over in "${retrieval_overs[@]}"; do
            CONFIG="${retrieval_over}_${role}_${llm}_${num_paragraph}"
            JSON_PATH_REL=gen_results_rel/generation_${CONFIG}.json
            JSON_PATH_NREL=gen_results_nonrel/generation_${CONFIG}.json
            JSON_PATHS+=($JSON_PATH_ALL $JSON_PATH_REL $JSON_PATH_NREL)

        done
    done
    QID_PATH=gen_results_rel/overlapping.json
    python find_overlapping_queries.py ${JSON_PATHS[@]} -o $QID_PATH

    for llm in "${llms_rel[@]}"; do
        for retrieval_over in "${retrieval_overs[@]}"; do
            CONFIG="${retrieval_over}_${role}_${llm}_${num_paragraph}"
            CONFIG_NOIR="no_ir_${role}_${llm}_${num_paragraph}"
            echo "Processing ${CONFIG}"

            JSON_PATH_ALL=gen_results/generation_${CONFIG}.json
            JSON_PATH_REL=gen_results_rel/generation_${CONFIG}.json
            JSON_PATH_NREL=gen_results_nonrel/generation_${CONFIG}.json
            JSON_PATH_NOIR=gen_results/generation_${CONFIG_NOIR}.json

            CSV_PATH_ALL=gen_results_rel/all/generation_${CONFIG}.csv
            CSV_PATH_REL=gen_results_rel/rel/generation_${CONFIG}.csv
            CSV_PATH_NREL=gen_results_rel/nrel/generation_${CONFIG}.csv
            CSV_PATH_NOIR=gen_results_rel/no_ir/generation_${CONFIG_NOIR}.csv
            exit

            # python ${SCRIPT_DIR}/gen_response_table.py -q -i $JSON_PATH_ALL -o $CSV_PATH_ALL --qid_path $QID_PATH
            # python ${SCRIPT_DIR}/gen_response_table.py -q -i $JSON_PATH_REL -o $CSV_PATH_REL --qid_path $QID_PATH
            # python ${SCRIPT_DIR}/gen_response_table.py -q -i $JSON_PATH_NREL -o $CSV_PATH_NREL --qid_path $QID_PATH
            python ${SCRIPT_DIR}/gen_response_table.py -q -i $JSON_PATH_NOIR -o $CSV_PATH_NOIR --qid_path $QID_PATH

            METRICS_REL=${METRICS_DIR}/scores-rel_${CONFIG}.json
            METRICS_ALL=${METRICS_DIR}scores-all_${CONFIG}.json
            METRICS_NREL=${METRICS_DIR}/scores-nonrel_${CONFIG}.json
            METRICS_NOIR=${METRICS_DIR}/scores_${CONFIG_NOIR}.json

            # python ${SCRIPT_DIR}/../borderlines/calculate_CS.py $CSV_PATH_ALL -o $METRICS_ALL -q
            # python ${SCRIPT_DIR}/../borderlines/calculate_CS.py $CSV_PATH_REL -o $METRICS_REL -q
            # python ${SCRIPT_DIR}/../borderlines/calculate_CS.py $CSV_PATH_NREL -o $METRICS_NREL -q
            python ${SCRIPT_DIR}/../borderlines/calculate_CS.py $CSV_PATH_NOIR -o $METRICS_NOIR -q
        done
    done

    python ${SCRIPT_DIR}/gen_all_results_table.py -i gen_results_rel/metrics/scores*.json \
        -o gen_results_rel/metrics/scores_for_all.csv --print_latex --with_role
fi
