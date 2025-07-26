#!/bin/zsh
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=hybridIR
#SBATCH --nodelist=nlpgpu07
#SBATCH --gpus=1
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.log
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3

for i in 5 100 200 300 400 500; do
    (sleep ${i}m && nvidia-smi) &
done

# ensure num_shards is the the same as --array
num_shards=4

for mode in en qlang qlang_en rel_langs; do
    echo "Running over ${mode}..."

    srun python m3_embed_ir/step2_rerank.py --languages all \
    --search_result_save_dir m3_embed_ir/search_results/over_${mode} \
    --rerank_result_save_dir m3_embed_ir/rerank_results/over_${mode} \
    --retrieval_over ${mode} \
    --top_k 50 --num_shards $num_shards \
    --cuda_id 0 --shard_id $SLURM_ARRAY_TASK_ID
done

if [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    # wait for all shards to finish
    sleep 10m
    # concatenate the sharded search results
    for folder in m3_embed_ir/rerank_results/over_${mode}/*/*/; do
        cat $folder/search_results_*_${num_shards}.txt > $folder/search_results.txt
        num_lines=$(wc -l $folder/search_results.txt | awk '{print $1}')
        echo "Saved ${num_lines} to  ${folder}/search_results.txt"
    done
fi
