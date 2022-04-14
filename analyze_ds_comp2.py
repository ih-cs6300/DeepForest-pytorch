# this version records the raster name and anotated coordinates of trees that are in competition

import pandas as pd
import numpy as np
from os.path import join
from itertools import combinations

path = "evaluation3"

df_list = ["NIWO-test.csv", "TEAK-test.csv", "SJER-test.csv", "MLBS-test.csv"]
res_df = pd.DataFrame({"image_path":[], "xmin":[], "ymin":[], "xmax":[], "ymax":[], "label":[], "area":[], "compet":[]})

def has_competition(df, res_df):
        """
        Description: finds trees with touching or intersecting crowns
        df - df with columns xmin, xmax, ymin, ymax, area
        trees_competing - list of competing trees; 0 => not competing, 1 => competing
        """
 
        trees_competing = [0] * len(df)

        # dist = L2 norm of difference
        # produces an array representing the radius of each bounding circle that contains the bounding box
        
        df_mat = df.loc[:, ["xmin", "ymin", "xmax", "ymax"]].to_numpy()        
        bb_rads = np.linalg.norm(df_mat[:, :2] - df_mat[:, 2:], ord=2, axis=1) / 2.

        # calculate the centroid of each bounding box
        bb_centroid = (df_mat[:, :2] + df_mat[:, 2:]) / 2.

        # calculate distance between each set of bounding box centers
        pair_count = 0
        for pair in combinations(range(0, df.shape[0]), 2):
            dist = np.linalg.norm(bb_centroid[pair[0]] - bb_centroid[pair[1]], ord = 2)

            #if dist between two BB centroids is less than or equal to the sum of the radii of their crown bounding cirlces, then consider trees to be touching
            if dist <= (bb_rads[pair[0]] + bb_rads[pair[1]]):
                trees_competing[pair[0]] = 1
                trees_competing[pair[1]] = 1

        #res = np.where(np.array(trees_competing) == 1.)[0].tolist()
        res = trees_competing
        #import pdb; pdb.set_trace()
        df['compet'] = res
        res_df = pd.concat([res_df, df], ignore_index=True)
        return res, res_df


res_dict = {x.split("-")[0] : [] for x in df_list}
for site in df_list:
   print("Analyzing {}...".format(site.split("-")[0]))
   num_comp_trees = 0
   num_noncomp_trees = 0

   num_comp_trees_gt_mean = 0
   num_comp_trees_lt_mean = 0

   num_noncomp_trees_gt_mean = 0
   num_noncomp_trees_lt_mean = 0

   df = pd.read_csv(join(path, site))
   df['area'] = (df.xmax - df.xmin) * (df.ymax - df.ymin)

   mean_area = np.mean(df.area)

   groups = df.groupby("image_path")
   for gname, group in groups:

      # figure out which trees are competing
      comp_trees, res_df = has_competition(group, res_df)
   
      num_comp_trees += sum(comp_trees)
      num_noncomp_trees += (len(group) - sum(comp_trees))

      trees_in_comp_df = group[np.array(comp_trees) == 1]
      trees_not_in_comp_df = group[np.array(comp_trees) == 0]

      num_comp_trees_gt_mean += len(trees_in_comp_df[trees_in_comp_df.area > mean_area])
      num_comp_trees_lt_mean += len(trees_in_comp_df[trees_in_comp_df.area <= mean_area])
 
      num_noncomp_trees_gt_mean += len(trees_not_in_comp_df[trees_not_in_comp_df.area > mean_area])
      num_noncomp_trees_lt_mean += len(trees_not_in_comp_df[trees_not_in_comp_df.area <= mean_area])

   
   res_list = [num_comp_trees_lt_mean, num_comp_trees_gt_mean, num_comp_trees, num_noncomp_trees_lt_mean, num_noncomp_trees_gt_mean, num_noncomp_trees, num_comp_trees+num_noncomp_trees]
   res_dict[site.split("-")[0]] = res_list   

print()
print()
print("Tree count")
final_df = pd.DataFrame(res_dict, index=["cmp ≤ mean", "cmp > mean", "cmp", "ncmp ≤ mean", "ncmp > mean", "ncmp", "total trees"])
#final_df.to_csv("comp_analysis.csv", index=True, header=True)
print(final_df)
print()
print()

print("Relative to all trees at site")
print(final_df.divide(final_df.iloc[-1, :], axis=1))
print()
print()

print("Relative to competing and non-competing trees")
final_df.iloc[:3, :] = final_df.iloc[:3, :].divide(final_df.iloc[2, :], axis=1)
final_df.iloc[3:6] = final_df.iloc[3:6].divide(final_df.iloc[5, :], axis=1)

print(final_df)


res_df[res_df.compet == 1].to_csv("compet.csv", index=False, header=True)
