#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=reports/slurm-%j.out
#SBATCH --error=reports/slurm-%j.out
#SBATCH --gres=gpu:a100:1
#comm SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=test
#SBATCH --time=23:59:00

source ~/.bashrc

# Checking working directory and environment
echo "Running from: $(pwd)"
echo "Hostname: $(hostname)"
echo "Job started at: $(date)"

# Move to directory : to be modified when running our script
cd "/data/mgiles/shil6478/KVQuant/gradients" || { echo "Failed to cd to directory"; exit 1; }

# Activate conda environment : to be modified when running our script
conda activate "/data/mgiles/shil6478/envs/grad3" || { echo "Failed to activate conda env"; exit 1; }

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
#export LINE_PROFILE=1
export CUDA_LAUNCH_BLOCKING=1

echo "Running Python script..."
#stdbuf -o0 -e0 accelerate launch test_model2.py --quant_type=$1 --fisher=$2 --size_centroids=$3 --start_layer=$4 --end_layer=$5 --model_name_or_path=$6 --num_examples=$7 --dataset=$8 2>&1 | tee python_output.log
stdbuf -o0 -e0 python -u test_model2.py --quant_type=$1 --fisher=$2 --size_centroids=$3 --all_layers=$4 --model_name_or_path=$5 --num_examples=$6 --outlier=$7 --dataset=$8 2>&1 | tee python_output.log
echo "Python finished with exit code: ${PIPESTATUS[0]}"

echo "Job finished at: $(date)"
