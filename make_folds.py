# make k-folds for cross validation

import pandas as pd
import numpy as np
from os.path import join
from os.path import isfile
from os.path import isdir
from os import mkdir
from os import remove
from math import floor
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--wdir', type=str, required=True, help='working directory')
parser.add_argument('--sdir', type=str, required=True, help='storage directory')
parser.add_argument('--fname', type=str, required=True, help='name of file to split')
parser.add_argument('--k', type=int, required=True, help='number of folds')
args = parser.parse_args()

# make sure directory and file exist
assert (isdir("./" + args.wdir)), "Directory {} doesn't exist".format(args.wdir)
assert (isfile(join(args.wdir, args.fname))), "Directory {} doesn't exist".format(join(args.wdir, args.fname))

print("\n\nReading {}...".format(args.fname))
# read csv
df_all = pd.read_csv(join(args.wdir, args.fname))
print("  {} instances...".format(len(df_all)))

# get images in file
# all annotations for an image must be in the same fold
img_groups = df_all.groupby("image_path")
group_names = df_all['image_path'].unique()
print("{} unique image files...".format(len(group_names)))


# make sure the argument for the number of folds is valid
assert ((args.k > 0) and (args.k <= len(group_names))), "k {} out of range".format(args.k)


if not (isdir(join(args.wdir, args.sdir))):
   try:
      mkdir(join(args.wdir, args.sdir))
      print("Creating directory {}...".format(join(args.wdir, args.sdir)))
   except OSError as e:
      if e.errno != errno.EEXIST:
         raise
else:
   print("Remove all files from {}? (yes/no)".format(join(args.wdir, args.sdir)))
   ans = input()
   if (ans == "yes"):
      if (len(glob(join(args.wdir, args.sdir, "*"))) > 0): 
         remove(join(args.wdir, args.sdir, "*"))

# shuffle image names and split them into folds
np.random.shuffle(group_names)
fold_imgs = np.array_split(group_names, args.k)

# write folds to file
total_len = 0
for idx in range(args.k):
    fold_name = join(args.wdir, args.sdir, "fold_" + str(idx) + ".csv")
    print("Saving {}...".format(fold_name))
    temp_df = df_all[df_all['image_path'].isin(fold_imgs[idx])]
    print("  {} instances...".format(len(temp_df)))
    total_len += len(temp_df)
    temp_df.to_csv(fold_name, index=False, header=True)

print("{} lines written".format(total_len))
print("Done!")
