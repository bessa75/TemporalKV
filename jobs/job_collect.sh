#!/bin/bash
#SBATCH --job-name=activations
#SBATCH --output=reports/slurm-%j.out
#SBATCH --error=reports/slurm-%j.out
#comm SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=test
#SBATCH --time=1:00:00

source ~/.bashrc

# Check working directory and environment : to be modified when running our script
echo "Running from: $(pwd)"
echo "Hostname: $(hostname)"
echo "Job started at: $(date)"

# Change to target directory : to be modified when running our script
cd "/data/mgiles/shil6478/KVQuant/gradients" || { echo "Failed to cd to directory"; exit 1; }

#conda activate "/data/mgiles/shil6478/envs/grad2-copy" || { echo "Failed to activate conda env"; exit 1; }
conda activate "/data/mgiles/shil6478/envs/grad3" || { echo "Failed to activate conda env"; exit 1; }


CUDA_VISIBLE_DEVICES=0 python run-fisher2.py --model_name_or_path $1  --output_dir . --dataset $2 --seqlen $3 --maxseqlen $3 --num_examples $4
