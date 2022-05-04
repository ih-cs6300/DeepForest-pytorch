# this script creates the datasets needed NIWO

import pandas as pd
import numpy as np
from os.path import join
from glob import glob
from sklearn.model_selection import train_test_split

train_dir = "./training3"
eval_dir = "./evaluation3"

train_csv = "TEAK-train.csv"
val_csv = "TEAK-val.csv"
test_csv = "TEAK-test.csv"

#import pdb; pdb.set_trace()
# read in all TEAK csv files
train_df_list = []
train_list = glob(join(train_dir, "2018_TEAK_*.csv"))
for f in train_list:
   train_df_list.append(pd.read_csv(f))


train_df_all  = pd.concat(train_df_list)

train_df, val_df = train_test_split(train_df_all, test_size=0.1667, random_state=42)

eval_list = glob(join(eval_dir, "TEAK_*.csv"))

eval_df_list = []
for f in eval_list:
   eval_df_list.append(pd.read_csv(f))

test_df  = pd.concat(eval_df_list)

print("\n\n\n")
print("Train dataset: {} instances".format(train_df.shape[0]))
print("Val dataset:   {} instances".format(val_df.shape[0]))
print("Test dataset:  {} instances".format(test_df.shape[0])) 

print("\n\n\n")
print("Saving files")
print("   {}".format(join(train_dir, train_csv)))
train_df.to_csv(join(train_dir, train_csv), header=True, index=False)

print("   {}".format(join(train_dir, val_csv)))
val_df.to_csv(join(train_dir, val_csv), header=True, index=False)

print("   {}".format(join(eval_dir, test_csv)))
test_df.to_csv(join(eval_dir, test_csv), header=True, index=False)