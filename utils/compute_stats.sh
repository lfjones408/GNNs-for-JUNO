#!/bin/bash
source /junofs/users/ljones/.lem_juno/bin/activate
export NETWORKX_NO_BACKENDS=1

cd /junofs/users/ljones/new_lem

{
    echo "=== Job started at $(date)"
    python utils/compute_stats.py
    echo "=== Job ended at $(date)"
} &> log_compute_stats_job.txt