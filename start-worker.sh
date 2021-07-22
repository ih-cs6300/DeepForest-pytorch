#!/bin/bash

#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8

export TUNE_MAX_PENDING_TRIALS_PG=10

echo "starting ray worker node"
ray start --address $1 --redis-password=$2
sleep infinity
