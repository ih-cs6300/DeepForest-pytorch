#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --mem=32gb
#SBATCH --job-name=r1_002    # Job name
#SBATCH --output=r1_%j.out   # Standard output and error log

# Load modules or your own conda environment here
module load git
module load conda && conda activate deepforest1
sleep 10

# baseline-standard
#./run_df.sh 101 seeds.txt 7 training3 evaluation3 NIWO-train.csv NIWO-val.csv NIWO-test.csv niwo df_rule1.csv False 0.000125 5 400 0.5 0.95 0.5
#./run_df.sh 101 seeds.txt 5 training3 evaluation3 TEAK-train.csv TEAK-val.csv TEAK-test.csv teak df_rule1_teak.csv False 0.01 4 2304 1e-6 0.95 0.5
#./run_df.sh 300 seeds.txt 5 training3 evaluation3 SJER-train.csv SJER-val.csv SJER-test.csv sjer df_rule1_sjer.csv False 0.000125 3 5184 0.5 0.80 0.5
#./run_df.sh 300 seeds.txt 5 training3 evaluation3 MLBS-train.csv MLBS-val.csv MLBS-test.csv mlbs df_rule1_mlbs.csv False 0.000125 1 2304 0.5 0.88 0.4

# baseline-cross site
#./run_df.sh 101 seeds.txt 7 training3 evaluation3 NIWO-train.csv TEAK-val.csv TEAK-test.csv niwo_teak df_niwo_teak_baseline.csv False
#./run_df.sh 301 seeds.txt 5 training3 evaluation3 TEAK-train.csv NIWO-val.csv NIWO-test.csv teak_niwo df_teak_niwo_baseline.csv False
#./run_df.sh 301 seeds.txt 5 training3 evaluation3 SJER-train.csv SJER-val.csv TEAK-test.csv sjer_teak df_sjer_teak_baseline.csv False
#./run_df.sh 301 seeds.txt 5 training3 evaluation3 TEAK-train.csv TEAK-val.csv SJER-test.csv teak_sjer df_teak_sjer_baseline.csv False

# baseline-chm
#./run_df.sh 101 seeds.txt 7 training4 evaluation4 NIWO-train.csv NIWO-val.csv NIWO-test.csv niwo df_niwo_chm_baseline.csv True
#./run_df.sh 101 seeds.txt 5 training4 evaluation4 TEAK-train.csv TEAK-val.csv TEAK-test.csv teak df_teak_chm_baseline.csv True
#./run_df.sh 300 seeds.txt 5 training4 evaluation4 SJER-train.csv SJER-val.csv SJER-test.csv sjer df_sjer_chm_baseline.csv True
#./run_df.sh 300 seeds.txt 5 training4 evaluation4 MLBS-train.csv MLBS-val.csv MLBS-test.csv mlbs df_mlbs_chm_baseline.csv True


# small datasets
#./run_df.sh 10 seeds.txt 7 training3 evaluation3 NIWO-train-1per.csv NIWO-val.csv NIWO-test.csv niwo df_niwo_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 7 training3 evaluation3 NIWO-train-2per.csv NIWO-val.csv NIWO-test.csv niwo df_niwo_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 7 training3 evaluation3 NIWO-train-5per.csv NIWO-val.csv NIWO-test.csv niwo df_niwo_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 7 training3 evaluation3 NIWO-train-10per.csv NIWO-val.csv NIWO-test.csv niwo df_niwo_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 7 training3 evaluation3 NIWO-train-20per.csv NIWO-val.csv NIWO-test.csv niwo df_niwo_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 7 training3 evaluation3 NIWO-train-30per.csv NIWO-val.csv NIWO-test.csv niwo df_niwo_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 7 training3 evaluation3 NIWO-train-50per.csv NIWO-val.csv NIWO-test.csv niwo df_niwo_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 7 training3 evaluation3 NIWO-train-75per.csv NIWO-val.csv NIWO-test.csv niwo df_niwo_small_pw_baseline.csv False

#./run_df.sh 10 seeds.txt 5 training3 evaluation3 TEAK-train-1per.csv TEAK-val.csv TEAK-test.csv teak df_teak_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 5 training3 evaluation3 TEAK-train-2per.csv TEAK-val.csv TEAK-test.csv teak df_teak_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 5 training3 evaluation3 TEAK-train-5per.csv TEAK-val.csv TEAK-test.csv teak df_teak_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 5 training3 evaluation3 TEAK-train-10per.csv TEAK-val.csv TEAK-test.csv teak df_teak_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 5 training3 evaluation3 TEAK-train-20per.csv TEAK-val.csv TEAK-test.csv teak df_teak_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 5 training3 evaluation3 TEAK-train-30per.csv TEAK-val.csv TEAK-test.csv teak df_teak_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 5 training3 evaluation3 TEAK-train-50per.csv TEAK-val.csv TEAK-test.csv teak df_teak_small_pw_baseline.csv False
#./run_df.sh 10 seeds.txt 5 training3 evaluation3 TEAK-train-75per.csv TEAK-val.csv TEAK-test.csv teak df_teak_small_pw_baseline.csv False
