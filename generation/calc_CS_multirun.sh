#!/bin/bash
# set -x

# Define the possible choices for each argument
num_paragraph=10
role="vanilla"
RANDOM_SEEDS=("" 2557 2558 2559 2560 2561 2562 2563 2564 2565)

SETTING=${1:-direct}

retrieval_overs=("no_ir" "qlang" "rel_langs" "qlang_en" "en" "ablate_swap")
llms=("llama8b" "commandr" "gpt-4o" "gpt-4o-mini")

mkdir -p gen_results/multirun/metrics

for llm in "${llms[@]}"; do
    for retrieval_over in "${retrieval_overs[@]}"; do
        CONFIG="${retrieval_over}_${role}_${llm}_${num_paragraph}"
        JSON_STEM=./gen_results/generation_${CONFIG}
        CSV_STEM=./gen_results/multirun/generation_${CONFIG}

        METRICS=./gen_results/multirun/metrics/scores_${CONFIG}.json
        if [ -f "${METRICS}" ]; then
            echo "Metrics file ${METRICS} already exists. Skipping..."
            continue
        else
            echo "Calculating metrics for ${CONFIG}..."
        fi

        for SEED in "${RANDOM_SEEDS[@]}"; do
            if [ -n "$SEED" ]; then
                EXT="_seed${SEED}"
            else
                EXT=""
            fi

            CSV_FILE="${CSV_STEM}${EXT}.csv"
            if [ -f "${CSV_FILE}" ]; then
                echo "CSV file ${CSV_FILE} already exists. Skipping..."
                continue
            else
                echo "Generating CSV file ${CSV_FILE}..."
            fi

            python ./gen_response_table.py -i ${JSON_STEM}${EXT}.json -o ${CSV_STEM}${EXT}.csv -q
        done
        python ./../borderlines/calculate_CS.py ${CSV_STEM}*.csv -o ${METRICS} -q
    done
done

TABLE_PATH=gen_results/multirun/metrics/scores_for_all.csv
echo "All metrics calculated, generating table at ${TABLE_PATH}..."
python ./gen_all_results_table_multirun.py -i gen_results/multirun/metrics/scores_*.json \
    -o ${TABLE_PATH} --print_latex
