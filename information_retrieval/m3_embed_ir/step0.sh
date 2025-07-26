#!/bin/zsh
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=embedIR
#SBATCH --nodelist=nlpgpu03
#SBATCH --gpus=1
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.log
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4

for i in 5 60 120 180 240 300; do
    (sleep ${i}m && nvidia-smi) &
done

for mode in en qlang qlang_en rel_langs; do
    echo "Running over ${mode}..."

    srun python m3_embed_ir/step0_gen_embed.py \
        --index_save_dir m3_embed_ir/bIRl-index_${mode} \
        --max_passage_length 512 --batch_size 512 --fp16 --retrieval_over ${mode}
done
