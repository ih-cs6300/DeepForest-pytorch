#!/bin/bash

for (( idx=1; idx<=$1; idx++ ))
do
   echo "Bash loop $idx"
   rm ./pred_result2/*
   python3 test1.py
   rm core.*
   echo ""
done
