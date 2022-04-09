#!/bin/bash
# example: "./run_df.sh 10 seeds.txt 7 training3 evaluation3 NIWO-train.csv NIWO-val.csv NIWO-test.csv niwo df_baseline_niwo.csv True 0.000125 3 0.87992 0.32658 0.5 0.95 0.5"

num_iter=$1
seed_file=$2     # file with seeds in it  ex: "./seeds.txt"
epochs=$3        # num epochs to train    ex: "5"
tr_dir=$4        # training directory     ex: "./training4
te_dir=$5        # testing directory      ex: "./evaluation4"
tr_ann=$6        # training annotations   ex: "niwo-train.csv"
va_ann=$7        # validation annotations ex: "niwo-val.csv
te_ann=$8        # testing annonations    ex: "niwo-test.csv"
site_name=$9     # site name              ex: "niwo"
log_file=${10}   # name of logging file   ex: "df_baseline_niwo.csv"
use_chm=${11}    # use chm                ex: "True"
norm_const=${12} # normalizatin const     ex: 6
pi_start=${13}   # epoch to start rule    ex: 3
a=${14}          # h-ca allometry const   ex: 0.1025
b=${15}          # h-ca allometry const   ex: 0.0628
k_sig=${16}      # sigma coeff            ex: 0.5
pi_0=${17}       # 1st pi_param term      ex: 0.95
pi_1=${18}       # 2nd pi_param term      ex: 0.5 

for (( idx=1; idx<=$num_iter; idx++ ))
do
   echo "Bash loop $idx"
   rm ./pred_result2/*
   seed=`awk "FNR == $idx" ${seed_file}`
   python3 -m pdb -c continue test1.py --seed ${seed} --site ${site_name} --train_dir ${tr_dir} --test_dir ${te_dir} --train_ann ${tr_ann} --val_ann ${va_ann} --test_ann ${te_ann} --epochs ${epochs} --log ${log_file} --chm ${use_chm} --C ${norm_const} --pi_start ${pi_start} --a ${a} --b ${b} --k_sig ${k_sig} --pi_0 ${pi_0} --pi_1 ${pi_1}
   rm core.*
   rm ./checkpoints/*
   echo ""
done

