# removes files from hand_annotations.csv that aren't 400x400 pixels
# saves new csv file as train.csv

import numpy as np
import pandas as pd
from PIL import Image
from os import listdir
from os.path import isfile, join
import glob

# path to training files
f_path = "./annotations"

# file containing the annotations
annotations_file = "hand_annotations.csv"

# turn annotations file into dataframe
train_df = pd.read_csv(join(f_path, annotations_file))

# get list of .TIFs  in f_path
image_files = glob.glob(join(f_path, "*.tif"))

# loop over TIFs keep only those that are 400x400
for f in image_files:
    img = Image.open(f)
    if (img.size != (400, 400)):
        f_name = f.split("/")[-1]
        train_df = train_df[train_df.image_path != f_name]

# write dataframe to drive
train_df.to_csv(join(f_path, "train.csv"), index=False)
