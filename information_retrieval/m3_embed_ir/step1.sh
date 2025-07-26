#!/bin/zsh
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=searchIR
#SBATCH --nodelist=nlpgpu08
#SBATCH --gpus=1
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.log
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=16

for i in 5 100 200 300 400 500; do
    (sleep ${i}m && nvidia-smi) &
done

for mode in en qlang qlang_en rel_langs; do
    echo "Running over ${mode}..."

    srun python m3_embed_ir/step1_search.py --languages all \
        --index_save_dir m3_embed_ir/bIRl-index_${mode} \
        --result_save_dir m3_embed_ir/search_results/over_${mode} \
        --threads 16 --batch_size 256 --hits 50
done
