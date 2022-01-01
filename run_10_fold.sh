#!/bin/bash
# example arg1 = "./training3/niwo_folds"
folds=`ls $1/fold_?.csv`

folds_dir=$1   # directory with foleds ex: "./training3/niwo_folds"
train_dir=$2   # working directory     ex: "training3"
site_name=$3   # site name             ex: "sjer"

# remove anything that may have been leftover from previous run
rm $1/temp-train.csv
rm $train_dir/temp-train.csv
rm $train_dir/fold_?.csv

for entry in $folds 
do
   echo "$entry"
   find $1 -type f | sort | grep -v $entry | xargs awk -F, '((NR==1) &&(FNR ==1)){print > "'$1'/temp-train.csv"}; (FNR > 1){print >> "'$1'/temp-train.csv"}'
   cat ${1}/fold_?.csv | wc -l
   tail -n +2 ${1}/temp-train.csv | wc -l
   tail -n +2 ${1}/$(basename $entry) | wc -l
   echo ""

   rm ./pred_result2/*
   cp ${folds_dir}/temp-train.csv ${train_dir}/temp-train.csv
   cp $entry $train_dir/
   python3 test2.py --site ${site_name} --train_dir ${train_dir} --test_dir ${train_dir} --train_ann temp-train.csv --test_ann $(basename $entry)
   rm ./checkpoints/*

   rm $1/temp-train.csv
   rm $train_dir/temp-train.csv
   rm $train_dir/$(basename $entry)
   #rm ./core.*
   echo ""
   echo ""
done
