# this script creates the datasets needed NIWO

import pandas as pd
import numpy as np
from os.path import join
from glob import glob
from sklearn.model_selection import train_test_split

train_dir = "./training3"
eval_dir = "./evaluation3"

train_csv = "NIWO-train.csv"
val_csv = "NIWO-val.csv"
test_csv = "NIWO-test.csv"

train_df_all = pd.read_csv(join(train_dir, "2018_NIWO_2_450000_4426000_image_crop.csv"))

train_df, val_df = train_test_split(train_df_all, test_size=0.1667, random_state=42)

eval_list = glob(join(eval_dir, "NIWO_*.csv"))

df_list = []
for f in eval_list:
   df_list.append(pd.read_csv(f))

test_df  = pd.concat(df_list)

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
