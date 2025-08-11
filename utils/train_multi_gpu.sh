#!/bin/bash
#SBATCH --job-name=egnn-multi-gpu
#SBATCH --partition=alpha
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

source /user/ljones/.gnn_juno/bin/activate
export NETWORKX_NO_BACKENDS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=1,5

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_P2P_DISABLE=1

cd /user/ljones/GNNs-for-JUNO

{
    echo "=== Job started at $(date) ==="
    echo "=== Running on host: $(hostname) ==="
    echo "=== CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES ==="
    torchrun --nproc_per_node=2 utils/train_multi_gpu_regress.py --config utils/config.yaml
    echo "=== Job ended at $(date) ==="
} &> utils/job_logs/train_multi_gpu_energy.txt

# nvidia-smi | grep -E '[0-9]+[ ]+C' | awk '{print $5}' | xargs -I{} ps -o user= -p {} <- Check users
# ps -p 3840646 -o lstart= <- when their job started