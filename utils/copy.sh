#!/bin/bash
set -euo pipefail

echo "Starting transfer!"

SSH_OPTS="-T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

rsync -avzP -e "ssh $SSH_OPTS" \
  ljones@lxlogin.ihep.ac.cn:/junofs/users/ljones/py_reader/all_features/FC/ \
  /hepstore/ljones/atm_nu/J24.1.2/FC/

echo "Transfer done!"