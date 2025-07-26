#!/bin/bash

# NOTE: it's recommended to use slurm_gen.sh instead

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


# Define the possible choices for each argument
retrieval_overs=("no_ir" "qlang" "qlang_en" "rel_langs" "en")
num_paragraphs=(10)
llms=("gpt-4" "llama1b" "llama3b" "llama8b" "commandr")

# Loop through each combination of arguments
for retrieval_over in "${retrieval_overs[@]}"; do
    for num_paragraph in "${num_paragraphs[@]}"; do
        for llm in "${llms[@]}"; do
            python $SCRIPT_DIR/generation.py --retrieval_over "$retrieval_over" --num_paragraphs "$num_paragraph" --llm "$llm" --overwrite
        done
    done
done
