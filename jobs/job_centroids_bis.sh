#!/bin/bash
#SBATCH --job-name=centroids
#SBATCH --output=reports/slurm-%j.out
#SBATCH --error=reports/slurm-%j.out
#commSBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=medium
#SBATCH --time=20:00:00

source ~/.bashrc

echo "Running from: $(pwd)"
echo "Hostname: $(hostname)"
echo "Job started at: $(date)"

# Change to target directory : to modify in order to reproduce results on another device
cd "/data/mgiles/shil6478/KVQuant/gradients" || { echo "Failed to cd to directory"; exit 1; }

# Activate conda environment : to modify in order to reproduce results on another device
conda activate "/data/mgiles/shil6478/envs/grad2-copy" || { echo "Failed to activate conda env"; exit 1; }

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

echo "Running Python script..."
stdbuf -o0 -e0 python3 -u write_centroids_bis.py --norm=$1 --start_layer=$2 --end_layer=$3 --mean_only=$4 --fisher=$5 --size_centroids=$6 --model_name=$7 --nb_examples=$8 2>&1 | tee python_output.log
echo "Python finished with exit code: ${PIPESTATUS[0]}"

echo "Job finished at: $(date)"
