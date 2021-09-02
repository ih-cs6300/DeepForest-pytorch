# this script creates regional training and testing sets

import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join
from tqdm import tqdm
import numpy as np

def make_reg_sets(full_df, site, is_test_set):
   """
   full_df: description: df with multiple neon sites
   site: type: string description: the NEON abbreviation for the site
   is_test_set: type: boolean description: if the df is a test set

   if the dataframe is a testset then the number of instances is limited to 1200
   """

   region_df = full_df[full_df['image_path'].apply(lambda x: site in x)]

   if (is_test_set) and (full_df.shape[0] > 1200):
      region_df = region_df.sample(1200, random_state = 42)
   return region_df


def make_reg_val_sets(reg_train_df):
   """
   creates a validation set from the training dataframe.
   """

   new_train_df, val_df = train_test_split(reg_train_df, test_size=0.10, random_state=42)
   return new_train_df, val_df

def store_df(path, name, df):
   """
   stores the dataframe under the given name and path
   """

   df.to_csv(join(path, name), header=True, index=False)

def get_avg_bbsize(df):
   """
   gets the average size bbox for each region
   """

   y_dim = np.mean(df['ymax'] - df['ymin'])
   x_dim = np.mean(df['xmax'] - df['xmin'])

   return (x_dim, y_dim)

   
if __name__ == "__main__":

   # path to data
   path = "train_data_folder2"

   # regions of interest
   # sites are ordered by representation in dataset
   # not all sites listed 
   regions = ['NIWO', 'TEAK', 'SJER', 'MLBS', 'OSBS']
   train_df_list = []

   # for debugging
   #import pdb; pdb.set_trace()

   # load the regionally combined training and testing sets

   print("Loading combined datasets...")
   train_df = pd.read_csv(join(path, "train.csv"))
   test_df = pd.read_csv(join(path, "test.csv"))

   # create regional sets
   print("Splitting into regions...")

   for region in regions:
      print("   {}...".format(region))
      print("      Making training set...")
      reg_train = make_reg_sets(train_df, region, False)

      print("      Making validation set...")
      reg_train, reg_val = make_reg_val_sets(reg_train)

      train_df_list.append((region, reg_train))

      print("      Making testing set...")
      reg_test = make_reg_sets(test_df, region, True)

      print("   Storing...")
      
      new_train_name = region + "-train.csv"
      print("      {}...".format(new_train_name))
      store_df(path, new_train_name, reg_train)

      new_val_name = region + "-val.csv"
      print("      {}...".format(new_val_name))
      store_df(path, new_val_name, reg_val)

      new_test_name = region + "-test.csv"
      print("      {}...".format(new_test_name))
      store_df(path, new_test_name, reg_test)
      print()

   print()
   print("Getting average bounding box sizes...")   
   avg_bbx_list = []
   for region, df in train_df_list:
      tmp = get_avg_bbsize(df)
      avg_bbx_list.append(tmp)
      print("   {}: {}".format(region, tmp))

   avg_bbx_df = pd.DataFrame({"region": regions, "avg_bbx": avg_bbx_list})
   store_df(path, "avg_reg_bbx_size.csv", avg_bbx_df)

   print()
   print("Done!")
