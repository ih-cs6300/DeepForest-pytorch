#!/bin/bash

#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8
export TUNE_MAX_PENDING_TRIALS_PG=10

echo "starting ray head node"
# Launch the head node
ray start --head --node-ip-address=$1 --port=6379 --redis-password=$2
sleep infinity
