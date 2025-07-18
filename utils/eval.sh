#!/bin/bash
source /junofs/users/ljones/.lem_juno/bin/activate
export NETWORKX_NO_BACKENDS=1

cd /junofs/users/ljones/new_lem

{
    echo "=== Job started at $(date)"
    python utils/eval_regress.py --config utils/config.yaml 
    echo "=== Job ended at $(date)"
} &> utils/job_logs/eval_regress_small.txt