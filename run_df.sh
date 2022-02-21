#!/bin/bash

for (( idx=1; idx<=$1; idx++ ))
do
   echo "Bash loop $idx"
   rm ./pred_result2/*
   seed=`awk "FNR == $idx" ./seeds.txt`
   python3 test1.py --seed $seed
   rm core.*
   echo ""
done
