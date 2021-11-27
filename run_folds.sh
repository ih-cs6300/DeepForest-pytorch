#!/bin/bash
# implements 5x2cv 
# file names are fold_k0.csv and fold_k1.csv where k in [0, 4]

for idx in `seq 0 4`
do 
   rm ./pred_result2/*
   python3 test2.py --train fold_${idx}0.csv --test fold_${idx}1.csv
   python3 test2.py --train fold_${idx}1.csv --test fold_${idx}0.csv
   echo ""
done
