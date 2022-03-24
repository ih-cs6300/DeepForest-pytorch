import pandas as pd
import numpy as np
from os.path import join

working_dir = "training3"
train_file = "TEAK-train.csv"
basename = "TEAK"

perc_list = [1, 2, 5, 10, 20, 30, 50, 75]
temp_df = pd.DataFrame({"image_path":[], "xmin":[], "ymin":[], "xmax":[], "ymax":[], "label":[]})
idx = 0
np.random.seed(42)

df_all = pd.read_csv(join(working_dir, train_file))

groups = df_all.groupby("image_path")
images = df_all.image_path.unique()
np.random.shuffle(images)

for img_name in images:
   group = groups.get_group(img_name)
   temp_df = pd.concat([temp_df, group])
   if ( (len(temp_df) / len(df_all)) * 100 ) > perc_list[idx]:
      fname = basename + "-train-{}per.csv".format(perc_list[idx])
      print("Saving file {} ...:".format(fname))
      print("  length = {}".format(len(temp_df)))
      print("  perc. tot. = {}".format(len(temp_df)/len(df_all)))
      print("")
      temp_df.to_csv(join(working_dir, fname), header=True, index=False)
      idx += 1
      if idx >= len(perc_list):
         break
