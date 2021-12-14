#!/bin/bash
# implements 5x2cv 
# file names are fold_k0.csv and fold_k1.csv where k in [0, 4]

folds_dir="../training3_bak/niwo_folds"
for idx in `seq 0 4`
do 
   echo ""
   rm ./pred_result2/*
   echo "fold_${idx}:"
   cp ${folds_dir}/fold_${idx}?.csv training3/
   python3 test2.py --site niwo --train_dir training3 --test_dir training3 --train_ann fold_${idx}0.csv --test_ann fold_${idx}1.csv
   python3 test2.py --site niwo --train_dir training3 --test_dir training3 --train_ann fold_${idx}1.csv --test_ann fold_${idx}0.csv
   rm training3/fold_${idx}?.csv 
   rm core.*
   echo ""
   echo ""
done

