Run in this order
```
# 1. run inference for direct answer
sbatch slurm_gen.sh [llm_name]

# 2. run inference for citation answer
sbatch slurm_gen_cite.sh [llm_name]

# 3. calculate the CS metrics
./calc_cs.sh
```
