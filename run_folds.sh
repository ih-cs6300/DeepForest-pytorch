#!/bin/bash
# example arg1 = "./training3/niwo_folds"
folds=`ls $1/*.csv`

for entry in $folds 
do
   find $1 -type f | sort | grep -v $entry | xargs awk -F, '((NR==1) &&(FNR ==1)){print > "./training3/temp-train.csv"}; (FNR > 1){print >> "./training3/temp-train.csv"}'
   cp $entry ./training3/$(basename $entry)
   wc -l ./training3/NIWO-train.csv
   wc -l ./training3/temp-train.csv
   wc -l ./training3/$(basename $entry)

   rm ./pred_result2/*
   rm ./checkpoints/*
   python3 test1.py
   rm ./training3/fold_?.csv
   rm ./core.*
done
