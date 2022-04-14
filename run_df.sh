#!/bin/bash
# example: "./run_df.sh 10 seeds.txt 7 training3 evaluation3 NIWO-train.csv NIWO-val.csv NIWO-test.csv niwo df_baseline_niwo.csv False 0.000125 3 400 0.5 0.95 0.5"

num_iter=$1
seed_file=$2     # file with seeds in it  ex: "./seeds.txt"
epochs=$3        # num epochs to train    ex: "5"
tr_dir=$4        # training directory     ex: "./training3
te_dir=$5        # testing directory      ex: "./evaluation3"
tr_ann=$6        # training annotations   ex: "niwo-train.csv"
va_ann=$7        # validation annotations ex: "niwo-val.csv
te_ann=$8        # testing annonations    ex: "niwo-test.csv"
site_name=$9     # site name              ex: "niwo"
log_file=${10}   # name of logging file   ex: "df_baseline_niwo.csv"
use_chm=${11}    # use chm                ex: "True"
norm_const=${12} # normalization const    ex: 6
pi_start=${13}   # epoch to start rule    ex: 3
opt_w=${14}      # optim width            ex: 46
opt_h=${15}      # optim height           ex: 46
wllim=${16}      # width left limit       ex: 6.5
wrlim=${17}      # width right limit      ex: 93.2
hllim=${18}      # height left limit      ex: 6.5
hrlim=${19}      # height right limit     ex: 93.2
pi_0=${20}       # 1st pi_param term      ex: 0.95
pi_1=${21}       # 2nd pi_param term      ex: 0.5 

for (( idx=1; idx<=$num_iter; idx++ ))
do
   echo "Bash loop $idx"
   rm ./pred_result2/*
   seed=`awk "FNR == $idx" ${seed_file}`
   python3 test1.py --seed ${seed} --site ${site_name} --train_dir ${tr_dir} --test_dir ${te_dir} --train_ann ${tr_ann} --val_ann ${va_ann} --test_ann ${te_ann} --epochs ${epochs} --log ${log_file} --chm ${use_chm} --C ${norm_const} --pi_start ${pi_start} --opt_w ${opt_w} --opt_h ${opt_h} --wllim ${wllim} --wrlim ${wrlim} --hllim ${hllim} --hrlim ${hrlim} --pi_0 ${pi_0} --pi_1 ${pi_1}
   rm core.*
   rm ./checkpoints/*
   echo ""
done

