# splits files that have either dimension greater than 500 pixels into a 400x400 px PNG files
# new files stored in "./train_data_folder2/"
# new annotation file is created called train_all.csv

import numpy as np
import pandas as pd
from PIL import Image
from os import remove, mkdir
from os.path import isfile, join, basename
from shutil import copyfile
import glob
from deepforest.preprocess import split_raster
from copy import deepcopy
from sklearn.model_selection import train_test_split

# directory with original image files
f_path = "./annotations"

# new directory to store processed image files
new_dir = "./train_data_folder2"

# make new directory if it doesn't exist
try:
    mkdir(new_dir)
except OSError as error:
    print(error)

# file containing the original annotations
annotations_file = "hand_annotations.csv"

# load annoations file into dataframe
annotations_df = pd.read_csv(join(f_path, annotations_file))

# turn annotations file into dataframe
train_df = pd.read_csv(join(f_path, annotations_file))
temp_df = deepcopy(train_df)

# get list of .TIFs  in f_path
image_files = glob.glob(join(f_path, "*.tif"))

# loop over TIFs only
for f in image_files:
    print("Image: {}".format(basename(f)))
    img = Image.open(f)

    # split files larger than 500 pixels in either dimension
    # the image file is saved as a PNG
    # the annotations file is saved as image_file_name.csv
    if (img.size[0] > 500 or img.size[1] > 500):

        # if a file is sent to split_raster but it has no annotations in train_df it causes an error
        # check if in train_df before sending to split_raster
        if (basename(f) in train_df['image_path'].unique()):
            print("spliting...")
            split_raster(f, join(f_path, annotations_file), new_dir, 400, 0.05)

            # remove entries of the oversized file from the annotations dataframe
            temp_df = temp_df[train_df.image_path != basename(f)]
    else:
        # copy files that are not too big to the new directory unchanged
        # make sure that file is listed in annotations before copying
        if (basename(f) in train_df['image_path'].unique()):
            print("copying...")
            copyfile(f, join(new_dir, basename(f)))
    print("\n")

# save the modified train_df (really temp_dir) to the new directory
temp_df.to_csv(join(new_dir, "train.csv"), index=False)

# get all the newly created csv files
print("consolidating csv's...")
csv_list = glob.glob(join(new_dir, "*.csv"))
df_list = [pd.read_csv(csv_file) for csv_file in csv_list]

# concat all dfs created from csv list
new_ann = pd.concat(df_list)


# create train_set, val_set, and test_set
print("Creating train, test, val...")
train_df2, test_df2 = train_test_split(new_ann, test_size=0.30, random_state=42)

# val = 10%, test = 20%
val_df3, test_df3 = train_test_split(test_df2, test_size=0.66667, random_state=42)

print("Writing to disk...")
new_ann.to_csv(join(new_dir, "all.csv"), index=False)

train_df2.to_csv(join(new_dir, "train.csv"), index=False)

val_df3.to_csv(join(new_dir, "val.csv"), index=False)

test_df3.to_csv(join(new_dir, "test.csv"), index=False)
