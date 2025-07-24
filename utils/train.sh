#!/bin/bash
#SBATCH --job-name=egnn-test-gpu
#SBATCH --partition=alpha
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=30:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

source /user/ljones/.gnn_juno/bin/activate
export NETWORKX_NO_BACKENDS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=2

cd /user/ljones/GNNs-for-JUNO

{
    echo "=== Job started at $(date) ==="
    echo "=== Running on host: $(hostname) ==="
    echo "=== CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES ==="
    python utils/train_equivar.py --config utils/config.yaml
    echo "=== Job ended at $(date) ==="
} &> utils/job_logs/train_equivar_GPU.txt