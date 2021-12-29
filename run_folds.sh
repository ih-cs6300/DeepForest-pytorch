#!/bin/bash
# implements 5x2cv 
# file names are fold_k0.csv and fold_k1.csv where k in [0, 4]

folds_dir="training4/niwo_folds"
train_dir="training4"
for idx in `seq 0 4`
do 
   echo ""
   rm ./pred_result2/*
   echo "fold_${idx}:"
   cp ${folds_dir}/fold_${idx}?.csv ${train_dir}/
   python3 test2.py --site niwo --train_dir ${train_dir} --test_dir ${train_dir} --train_ann fold_${idx}0.csv --test_ann fold_${idx}1.csv
   python3 test2.py --site niwo --train_dir ${train_dir} --test_dir ${train_dir} --train_ann fold_${idx}1.csv --test_ann fold_${idx}0.csv
   rm ${train_dir}/fold_${idx}?.csv 
   rm core.*
   echo ""
   echo ""
done
