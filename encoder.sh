#!/bin/bash
source /junofs/users/ljones/.lem_juno/bin/activate
export NETWORKX_NO_BACKENDS=1

cd /junofs/users/ljones/new_lem

epochs=${1:-10}
batch=${2:-1}
limit=${3:-100}

{
    echo "=== Job started at $(date)"
    python encoder.py --input /junofs/users/ljones/py_reader/FC/nu_e/pmt_data_0.h5 --output latent --epochs $epochs --batch_size $batch #--limit $limit
    echo "=== Job ended at $(date)"
} &> log.txt